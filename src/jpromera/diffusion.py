"""Diffusion score model + EDM preconditioning for Promera.

The score model recomputes the pairwise conditioning and per-layer pair biases
on every call (the cache-free path), which is numerically identical to the
PyTorch model_cache path used during sampling.
"""


from __future__ import annotations
import einops
import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float


from .backend import (
    AbstractFromTorch,
    LayerNorm,
    Linear,
    Sequential,
    from_torch,
    register_from_torch,
)
from .layers import FourierEmbedding, Transition
from .trunk import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    DiffusionTransformer,
)


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
    ):
        assert multiplicity == 1, "JAX score model handles one sample at a time"
        s, normed_fourier = self.single_conditioner(
            times=times, s_trunk=s_trunk, s_inputs=s_inputs
        )
        z = self.pairwise_conditioner(
            z_trunk=z_trunk, token_rel_pos_feats=relative_position_encoding
        )

        a, q_skip, c_skip, p_skip, to_keys = self.atom_attention_encoder(
            feats=feats, s_trunk=s_trunk, z=z, r=r_noisy, multiplicity=multiplicity
        )
        a = a + self.s_to_a_linear(s)

        mask = feats.token_pad_mask
        a = self.token_transformer(
            a, s=s, z=z, mask=mask.astype(jnp.float32), multiplicity=multiplicity
        )
        a = self.a_norm(a)

        r_update = self.atom_attention_decoder(
            a=a, q=q_skip, c=c_skip, p=p_skip, feats=feats,
            multiplicity=multiplicity, to_keys=to_keys,
        )
        return {"r_update": r_update, "token_a": a}


@register_from_torch("promera.model.diffusion.AtomDiffusion")
class AtomDiffusion(eqx.Module):
    score_model: DiffusionModule
    sigma_data: float
    has_alt_update: bool

    @staticmethod
    def from_torch(m: _diffusion.AtomDiffusion):
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
