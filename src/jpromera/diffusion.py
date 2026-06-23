"""Diffusion score model + EDM preconditioning for Promera.

The score model recomputes the pairwise conditioning and per-layer pair biases
on every call (the cache-free path), which is numerically identical to the
PyTorch model_cache path used during sampling.
"""


from __future__ import annotations
import einops
import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array


from .backend import (
    AbstractFromTorch,
    LayerNorm,
    Linear,
    Sequential,
    from_torch,
    register_from_torch,
)
from .layers import FourierEmbedding
from .trunk import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    DiffusionTransformer,
)


class DiffusionCache(eqx.Module):
    """Coordinate-independent conditioning computed once per diffusion rollout.

    Mirrors the upstream ``model_cache``: everything derived from the trunk
    outputs and ref features (not from the noised coordinates ``r`` or noise
    level ``sigma``) is hoisted out of the sampling loop. ``None`` fields cover
    the depth-0 / non-structure-prediction cases.
    """

    z: Array
    q0: Array
    c: Array
    p: Array | None
    to_keys: object
    enc_atom_biases: Array | None
    tok_biases: Array | None
    dec_atom_biases: Array | None


@register_from_torch("promera.model.encoders.SingleConditioning")
class SingleConditioning(AbstractFromTorch):
    eps: float
    sigma_data: float
    norm_single: LayerNorm
    single_embed: Linear
    fourier_embed: FourierEmbedding
    norm_fourier: LayerNorm
    fourier_to_single: Linear
    transitions: list

    def __call__(self, times, s_trunk, s_inputs):
        s = jnp.concatenate([s_trunk, s_inputs], axis=-1)
        s = self.single_embed(self.norm_single(s))
        fourier_embed = self.fourier_embed(times)
        normed_fourier = self.norm_fourier(fourier_embed)
        fourier_to_single = self.fourier_to_single(normed_fourier)
        s = einops.rearrange(fourier_to_single, "b d -> b 1 d") + s
        for transition in self.transitions:
            s = transition(s) + s
        return s, normed_fourier


@register_from_torch("promera.model.encoders.PairwiseConditioning")
class PairwiseConditioning(AbstractFromTorch):
    dim_pairwise_init_proj: Sequential
    transitions: list

    def __call__(self, z_trunk, token_rel_pos_feats):
        z = jnp.concatenate([z_trunk, token_rel_pos_feats], axis=-1)
        z = self.dim_pairwise_init_proj(z)
        for transition in self.transitions:
            z = transition(z) + z
        return z


@register_from_torch("promera.model.diffusion.DiffusionModule")
class DiffusionModule(AbstractFromTorch):
    atoms_per_window_queries: int
    atoms_per_window_keys: int
    sigma_data: float
    single_conditioner: SingleConditioning
    pairwise_conditioner: PairwiseConditioning
    atom_attention_encoder: AtomAttentionEncoder
    s_to_a_linear: Sequential
    token_transformer: DiffusionTransformer
    a_norm: LayerNorm
    atom_attention_decoder: AtomAttentionDecoder

    def precompute_conditioning(
        self, s_trunk, z_trunk, relative_position_encoding, feats
    ) -> DiffusionCache:
        """Build the coordinate-independent ``DiffusionCache`` once per rollout.

        Runs the pairwise conditioner, the atom-encoder prefix, and every
        token-/atom-transformer layer's pair bias — all loop-invariant across
        the diffusion steps (``jax.lax.scan`` won't hoist them otherwise).
        """
        z = self.pairwise_conditioner(
            z_trunk=z_trunk, token_rel_pos_feats=relative_position_encoding
        )
        q0, c, p, to_keys, enc_atom_biases = self.atom_attention_encoder.precompute(
            feats=feats, s_trunk=s_trunk, z=z
        )
        tok_biases = self.token_transformer.precompute_biases(z)
        dec_atom_biases = self.atom_attention_decoder.precompute_biases(p)
        return DiffusionCache(
            z=z, q0=q0, c=c, p=p, to_keys=to_keys,
            enc_atom_biases=enc_atom_biases, tok_biases=tok_biases,
            dec_atom_biases=dec_atom_biases,
        )

    def __call__(
        self,
        s_inputs,
        s_trunk,
        z_trunk,
        r_noisy,
        times,
        relative_position_encoding,
        feats,
        multiplicity=1,
        model_cache=None,
        cache: DiffusionCache | None = None,
    ):
        assert multiplicity == 1, "JAX score model handles one sample at a time"
        s, normed_fourier = self.single_conditioner(
            times=times, s_trunk=s_trunk, s_inputs=s_inputs
        )

        if cache is None:
            # Cache-free path: recompute everything (numerically identical).
            z = self.pairwise_conditioner(
                z_trunk=z_trunk, token_rel_pos_feats=relative_position_encoding
            )
            a, q_skip, c_skip, p_skip, to_keys = self.atom_attention_encoder(
                feats=feats, s_trunk=s_trunk, z=z, r=r_noisy,
                multiplicity=multiplicity,
            )
            tok_biases = None
            dec_atom_biases = None
        else:
            z = cache.z
            a, q_skip, c_skip, p_skip, to_keys = (
                self.atom_attention_encoder.apply_coords(
                    cache.q0, cache.c, cache.p, cache.to_keys,
                    cache.enc_atom_biases, r_noisy, feats, multiplicity,
                )
            )
            tok_biases = cache.tok_biases
            dec_atom_biases = cache.dec_atom_biases

        a = a + self.s_to_a_linear(s)

        mask = feats.token_pad_mask
        a = self.token_transformer(
            a, s=s, z=z, mask=mask.astype(jnp.float32), multiplicity=multiplicity,
            biases=tok_biases,
        )
        a = self.a_norm(a)

        r_update = self.atom_attention_decoder(
            a=a, q=q_skip, c=c_skip, p=p_skip, feats=feats,
            multiplicity=multiplicity, to_keys=to_keys, atom_biases=dec_atom_biases,
        )
        return {"r_update": r_update, "token_a": a}


@register_from_torch("promera.model.diffusion.AtomDiffusion")
class AtomDiffusion(eqx.Module):
    score_model: DiffusionModule
    sigma_data: float
    has_alt_update: bool

    @staticmethod
    def from_torch(m):
        return AtomDiffusion(
            score_model=from_torch(m.score_model),
            sigma_data=float(m.cfg.diffusion.sigma_data),
            has_alt_update=bool(m.cfg.score_model_args.has_alt_update),
        )

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / jnp.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / jnp.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return jnp.log(jnp.clip(sigma / self.sigma_data, min=1e-20)) * 0.25

    def preconditioned_network_forward(
        self, noised_atom_coords, sigma, network_condition_kwargs
    ):
        batch = noised_atom_coords.shape[0]
        if jnp.ndim(sigma) == 0:
            sigma = jnp.full((batch,), sigma)
        padded_sigma = einops.rearrange(sigma, "b -> b 1 1")
        net_out = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )
        r_update = net_out["r_update"]
        denoised = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * r_update
        )
        return {"denoised_atom_coords": denoised, "token_a": net_out["token_a"]}
