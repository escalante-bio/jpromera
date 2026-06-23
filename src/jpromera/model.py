"""Top-level Promera model: input embedding, recycled trunk, distogram,
diffusion sampling and confidence/contact heads.

MSA subsampling (PyTorch ``subsample_msa_per_recycle``) is done inside the
trunk: each recycle draws a fresh random ``subsample`` rows from the full MSA
the caller passed in. This is JIT-clean because ``subsample`` is a static int,
so the gathered/padded output depth is fixed at trace time; the per-recycle
randomness comes from ``fold_in(key, i)`` on the loop index. The trunk
therefore requires a PRNG ``key`` — pass a fixed one (e.g. ``key(0)``) for
deterministic/parity runs.
"""


from __future__ import annotations
import dataclasses
import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from .config import MSAS_PER_TRUNK_ITER


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
    # Enables trunk dropout (MSA + Pairformer), gated like PyTorch's ``training``
    # flag. Static so it never affects JIT randomness when off. Default False ->
    # behaviour is bit-identical to the deterministic eval path used for parity.
    dropout: bool = eqx.field(static=True, default=False)

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
    def trunk_iteration(self, state: TrunkState, emb: InitialEmbedding, feats,
                        key=None):
        mask = feats.token_pad_mask.astype(jnp.float32)
        pair_mask = mask[:, :, None] * mask[:, None, :]
        s = emb.s_init + self.s_recycle(self.s_norm(state.s))
        z = emb.z_init + self.z_recycle(self.z_norm(state.z))
        # When dropout is off the modules run their deterministic (all-ones mask)
        # path and ignore the key entirely — identical to the pre-key behaviour.
        det = not self.dropout
        k_msa, k_pf = (None, None) if det else jax.random.split(key, 2)
        _, z_msa = self.msa_module(z, emb.s_inputs, feats,
                                   deterministic=det, key=k_msa)
        z = z + z_msa
        s, z = self.pairformer_module(s, z, mask=mask, pair_mask=pair_mask,
                                      deterministic=det, key=k_pf)
        return TrunkState(s=s, z=z)

    def trunk(self, feats, recycling_steps: int, key, *, subsample=MSAS_PER_TRUNK_ITER):
        emb = self.embed_inputs(feats)
        state = TrunkState(s=jnp.zeros_like(emb.s_init), z=jnp.zeros_like(emb.z_init))

        # Per-recycle MSA feats: a fresh random subset of ``subsample`` rows.
        # ``i`` is the (possibly traced) recycle index, so ``fold_in`` gives each
        # recycle an independent draw (PyTorch subsample_msa_per_recycle).
        def feats_for(i):
            return _subsample_msa(feats, jax.random.fold_in(key, i), subsample)

        # Dropout draws from a separate per-recycle key (only consumed when
        # ``self.dropout`` is on), so the MSA-subsample randomness above is byte
        # for byte unchanged whether or not dropout is enabled.
        def dropout_key(i):
            return jax.random.fold_in(jax.random.fold_in(key, 0xD0), i)

        # Run the first (recycling_steps - 1) iterations with NO gradient (each
        # output detached, so no backward graph / activations are retained), then
        # one final differentiated iteration. This isolates the gradient to the
        # last recycle (matching PyTorch's set_grad_enabled(i == last)) and keeps
        # backprop memory bounded — forward values are unchanged.
        def no_grad_iter(i, st):
            return jax.lax.stop_gradient(
                self.trunk_iteration(st, emb, feats_for(i), dropout_key(i))
            )

        state = jax.lax.fori_loop(0, recycling_steps - 1, no_grad_iter, state)
        state = jax.lax.stop_gradient(state)
        # final, differentiated iteration (recycle index recycling_steps - 1)
        last = recycling_steps - 1
        state = self.trunk_iteration(state, emb, feats_for(last), dropout_key(last))

        pdistogram = self.distogram_module(state.z)
        return emb, state, pdistogram

    @eqx.filter_jit
    def fold(self, feats, recycling_steps: int, key,
             *, subsample=MSAS_PER_TRUNK_ITER) -> TrunkOutput:
        """Run the trunk + distogram. Returns the conditioning for sampling.

        ``key`` seeds the per-recycle MSA subsampling (``subsample`` rows each
        recycle); pass a fixed key for deterministic runs.
        """
        emb, state, pdistogram = self.trunk(feats, recycling_steps, key,
                                            subsample=subsample)
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

        # Hoist the coordinate-independent conditioning (pair bias, atom-encoder
        # prefix, per-layer pair biases) out of the per-step scan — computed once.
        cache = self.structure_module.score_model.precompute_conditioning(
            s_trunk=out.s, z_trunk=out.z,
            relative_position_encoding=out.relpos, feats=feats,
        )
        conditioning = dict(
            s_inputs=out.s_inputs, s_trunk=out.s, z_trunk=out.z,
            relative_position_encoding=out.relpos, feats=feats, multiplicity=1,
            cache=cache,
        )
        final, traj, noisy = edm_sample(
            self.structure_module, conditioning, atom_mask, init,
            jnp.asarray(sigma_t), jnp.asarray(sigma_s), jnp.asarray(gamma),
            churn, rot, tr,
            step_scale=diffusion_cfg["step_scale"],
            noise_scale=diffusion_cfg["noise_scale"],
        )
        return final, traj, noisy


# --- MSA subsampling (PyTorch PromeraModel.subsample_msa) ---
_MSA_FIELDS = ("msa", "msa_mask", "msa_paired", "has_deletion", "deletion_value")


def _subsample_msa(feats, key, subsample: int):
    """Draw ``subsample`` MSA rows without replacement and pad back to that
    depth, matching PyTorch's ``subsample_msa`` (np.random.choice + F.pad).

    ``subsample`` is a static int and the source depth ``S`` is a static array
    dim, so ``k`` and the pad amount are known at trace time — JIT-clean. The
    gather index is traced (it depends on ``key``)."""
    S = feats.msa.shape[1]              # static
    k = min(S, subsample)               # static
    idx = jax.random.choice(key, S, (k,), replace=False)

    def take(x):
        x = x[:, idx]                   # gather rows (handles int or one-hot MSA)
        if k < subsample:
            pad = [(0, 0)] * x.ndim
            pad[1] = (0, subsample - k)
            x = jnp.pad(x, pad)
        return x

    return dataclasses.replace(feats, **{f: take(getattr(feats, f)) for f in _MSA_FIELDS})


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
