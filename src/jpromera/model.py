"""Top-level Promera model: input embedding, recycled trunk, distogram,
diffusion sampling and confidence/contact heads.

MSA subsampling (PyTorch ``subsample_msa_per_recycle``) is a data-prep concern
and is NOT done inside this model: the caller provides ``feats`` with whatever
MSA depth should be used. This keeps the trunk a pure function of its inputs
(important for JIT and for gradient-based use in mosaic).
"""


from __future__ import annotations
import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float


from .backend import Embedding, LayerNorm, Linear, from_torch, register_from_torch
from .confidence import ConfidenceModule, ContactModule
from .diffusion import AtomDiffusion
from .layers import RelativePositionEncoder
from .trunk import (
    DistogramModule,
    InputEmbedder,
    MSAModule,
    PairformerModule,
    TrunkOutput,
)
from .sampler import edm_sample, edm_schedule


class InitialEmbedding(eqx.Module):
    s_inputs: Float[Array, "b n d"]
    s_init: Float[Array, "b n ts"]
    z_init: Float[Array, "b n n tz"]
    relpos: Float[Array, "b n n tz"]


class TrunkState(eqx.Module):
    s: Float[Array, "b n ts"]
    z: Float[Array, "b n n tz"]


@register_from_torch("promera.model.model.PromeraModel")
class JPromera(eqx.Module):
    input_embedder: InputEmbedder
    s_init: Linear
    z_init_1: Linear
    z_init_2: Linear
    rel_pos: RelativePositionEncoder
    token_bonds: Linear
    disto_embed: Embedding | None
    s_norm: LayerNorm
    z_norm: LayerNorm
    s_recycle: Linear
    z_recycle: Linear
    msa_module: MSAModule
    pairformer_module: PairformerModule
    distogram_module: DistogramModule
    structure_module: AtomDiffusion
    sm_confidence_module: ConfidenceModule
    contact_module: ContactModule

    @staticmethod
    def from_torch(m: _model.PromeraModel):
        return JPromera(
            input_embedder=from_torch(m.input_embedder),
            s_init=from_torch(m.s_init),
            z_init_1=from_torch(m.z_init_1),
            z_init_2=from_torch(m.z_init_2),
            rel_pos=from_torch(m.rel_pos),
            token_bonds=from_torch(m.token_bonds),
            disto_embed=from_torch(m.disto_embed) if hasattr(m, "disto_embed") else None,
            s_norm=from_torch(m.s_norm),
            z_norm=from_torch(m.z_norm),
            s_recycle=from_torch(m.s_recycle),
            z_recycle=from_torch(m.z_recycle),
            msa_module=from_torch(m.msa_module),
            pairformer_module=from_torch(m.pairformer_module),
            distogram_module=from_torch(m.distogram_module),
            structure_module=from_torch(m.structure_module),
            sm_confidence_module=from_torch(m.sm_confidence_module),
            contact_module=from_torch(m.contact_module),
        )

    # -- embedding --
    def embed_inputs(self, feats) -> InitialEmbedding:
        s_inputs = self.input_embedder(feats)
        s_init = self.s_init(s_inputs)
        z_init = self.z_init_1(s_inputs)[:, :, None] + self.z_init_2(s_inputs)[:, None, :]
        relpos = self.rel_pos(feats)
        z_init = z_init + relpos
        token_bonds = feats.token_bonds[..., None].astype(jnp.float32)
        z_init = z_init + self.token_bonds(token_bonds)
        # distogram conditioning (binder/target template; used by the Design task)
        if (
            self.disto_embed is not None
            and feats.distogram_emb is not None
            and feats.distogram_mask is not None
        ):
            z_init = z_init + jnp.where(
                feats.distogram_mask[..., None],
                self.disto_embed(feats.distogram_emb),
                0.0,
            )
        return InitialEmbedding(s_inputs=s_inputs, s_init=s_init, z_init=z_init,
                                relpos=relpos)

    # -- one trunk recycling iteration --
    def trunk_iteration(self, state: TrunkState, emb: InitialEmbedding, feats):
        mask = feats.token_pad_mask.astype(jnp.float32)
        pair_mask = mask[:, :, None] * mask[:, None, :]
        s = emb.s_init + self.s_recycle(self.s_norm(state.s))
        z = emb.z_init + self.z_recycle(self.z_norm(state.z))
        _, z_msa = self.msa_module(z, emb.s_inputs, feats)
        z = z + z_msa
        s, z = self.pairformer_module(s, z, mask=mask, pair_mask=pair_mask)
        return TrunkState(s=s, z=z)

    def trunk(self, feats, recycling_steps: int):
        emb = self.embed_inputs(feats)
        state = TrunkState(s=jnp.zeros_like(emb.s_init), z=jnp.zeros_like(emb.z_init))

        # Run the first (recycling_steps - 1) iterations with NO gradient (each
        # output detached, so no backward graph / activations are retained), then
        # one final differentiated iteration. This isolates the gradient to the
        # last recycle (matching PyTorch's set_grad_enabled(i == last)) and keeps
        # backprop memory bounded — forward values are unchanged.
        def no_grad_iter(i, st):
            return jax.lax.stop_gradient(self.trunk_iteration(st, emb, feats))

        state = jax.lax.fori_loop(0, recycling_steps - 1, no_grad_iter, state)
        state = jax.lax.stop_gradient(state)
        state = self.trunk_iteration(state, emb, feats)  # final, differentiated

        pdistogram = self.distogram_module(state.z)
        return emb, state, pdistogram

    @eqx.filter_jit
    def fold(self, feats, recycling_steps: int) -> TrunkOutput:
        """Run the trunk + distogram. Returns the conditioning for sampling."""
        emb, state, pdistogram = self.trunk(feats, recycling_steps)
        return TrunkOutput(
            s=state.s, z=state.z, s_inputs=emb.s_inputs, s_init=emb.s_init,
            z_init=emb.z_init, relpos=emb.relpos, pdistogram=pdistogram,
        )

    def sample(self, feats, out: TrunkOutput, *, num_steps, diffusion_cfg, key):
        """Run the EDM-churn diffusion sampler, generating randoms from ``key``."""
        atom_mask = feats.atom_pad_mask.astype(jnp.float32)
        B, M = atom_mask.shape
        sigma_t, sigma_s, gamma = edm_schedule(diffusion_cfg, num_steps)

        sigma0 = float((diffusion_cfg["sigma_max"] * diffusion_cfg["sigma_data"]))
        k_init, k_aug, k_rot, k_churn = jax.random.split(key, 4)
        init = sigma0 * jax.random.normal(k_init, (B, M, 3))

        # per-step random draws
        rot = _random_rotations(num_steps * B, k_rot).reshape(num_steps, B, 3, 3)
        tr = jax.random.normal(k_aug, (num_steps, B, 1, 3))
        churn = jax.random.normal(k_churn, (num_steps, B, M, 3))

        conditioning = dict(
            s_inputs=out.s_inputs, s_trunk=out.s, z_trunk=out.z,
            relative_position_encoding=out.relpos, feats=feats, multiplicity=1,
        )
        final, traj, noisy = edm_sample(
            self.structure_module, conditioning, atom_mask, init,
            jnp.asarray(sigma_t), jnp.asarray(sigma_s), jnp.asarray(gamma),
            churn, rot, tr,
            step_scale=diffusion_cfg["step_scale"],
            noise_scale=diffusion_cfg["noise_scale"],
        )
        return final, traj, noisy


# --- random rotations (matches torch3d quaternion sampling used by promera) ---
def _random_rotations(n, key):
    o = jax.random.normal(key, (n, 4))
    s = (o * o).sum(1)
    sign = jnp.where((jnp.sqrt(s) < 0) != (o[:, 0] < 0), -jnp.sqrt(s), jnp.sqrt(s))
    o = o / sign[:, None]
    r, i, j, k = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
    two_s = 2.0 / (o * o).sum(-1)
    out = jnp.stack([
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ], axis=-1)
    return out.reshape(n, 3, 3)
