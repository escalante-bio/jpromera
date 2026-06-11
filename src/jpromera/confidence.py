"""Confidence + contact head translations for Promera."""


from __future__ import annotations

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int


from .backend import (
    AbstractFromTorch,
    Embedding,
    LayerNorm,
    Linear,
    from_torch,
    register_from_torch,
)
from .feats import Feats
from .trunk import PairformerModule, TrunkOutput


class Confidence(eqx.Module):
    """Output of the confidence heads: per-bin logits and their expected values.

    pde/pae are expected distances (Å); plddt is the expected lDDT in [0, 1].
    """

    pde_logits: Float[Array, "B N N nb"]
    plddt_logits: Float[Array, "B N nb"]
    pae_logits: Float[Array, "B N N nb"]
    pde: Float[Array, "B N N"]
    plddt: Float[Array, "B N"]
    pae: Float[Array, "B N N"]


class Contact(eqx.Module):
    """Output of the contact head: per-pair inter-chain contact logits and the
    predicted token-token Cα distances they were computed from."""

    contact_logits: Float[Array, "B N N"]
    pred_dist: Float[Array, "B N N"]


def compute_aggregated_metric(
    logits: Float[Array, "*batch nb"], end: float = 1.0
) -> Float[Array, "*batch"]:
    """Expected value of a binned distribution: softmax(logits) · bin centers."""
    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    bounds = jnp.arange(0.5 * bin_width, end, bin_width)
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("... b, b -> ...", probs, bounds)


def _cdist(
    a: Float[Array, "B N 3"], b: Float[Array, "B M 3"]
) -> Float[Array, "B N M"]:
    r = a[:, :, None, :] - b[:, None, :, :]
    return jnp.sqrt(jnp.clip(jnp.sum(r * r, axis=-1), min=0.0))


def _gather_rep_atoms(
    x_pred: Float[Array, "B M 3"], token_to_rep_atom: Int[Array, "B N"]
) -> Float[Array, "B N 3"]:
    """Select each token's representative-atom coord from the atom-level coords."""
    idx = jnp.broadcast_to(token_to_rep_atom[..., None], token_to_rep_atom.shape + (3,))
    return jnp.take_along_axis(x_pred, idx, axis=1)


@register_from_torch("promera.model.confidence.ConfidenceHeads")
class ConfidenceHeads(AbstractFromTorch):
    to_pde_logits: Linear
    to_plddt_logits: Linear
    to_pae_logits: Linear
    _protenix: bool
    pae_ln: LayerNorm | None
    pde_ln: LayerNorm | None
    plddt_ln: LayerNorm | None

    def __call__(
        self,
        s: Float[Array, "B N Cs"],
        z: Float[Array, "B N N Cz"],
        feats: Feats | None = None,
        multiplicity: int = 1,
    ) -> Confidence:
        if self._protenix:
            plddt_logits = self.to_plddt_logits(self.plddt_ln(s))
            pde_logits = self.to_pde_logits(self.pde_ln(z + jnp.swapaxes(z, 1, 2)))
            pae_logits = self.to_pae_logits(self.pae_ln(z))
        else:
            plddt_logits = self.to_plddt_logits(s)
            pde_logits = self.to_pde_logits(z + jnp.swapaxes(z, 1, 2))
            pae_logits = self.to_pae_logits(z)
        return Confidence(
            pde_logits=pde_logits,
            plddt_logits=plddt_logits,
            pae_logits=pae_logits,
            pde=compute_aggregated_metric(pde_logits, end=32),
            plddt=compute_aggregated_metric(plddt_logits),
            pae=compute_aggregated_metric(pae_logits, end=32),
        )


@register_from_torch("promera.model.confidence.ConfidenceModule")
class ConfidenceModule(AbstractFromTorch):
    inp_only: bool
    distogram: bool
    boundaries: Float[Array, "nb"] | None
    dist_bin_pairwise_embed: Embedding | None
    s_to_z: Linear
    s_to_z_transpose: Linear
    s_norm: LayerNorm
    pairformer_stack: PairformerModule
    confidence_heads: ConfidenceHeads

    @classmethod
    def from_torch(cls, m) -> "ConfidenceModule":
        import jpromera.backend as B
        return cls(
            inp_only=bool(m.inp_only),
            distogram=hasattr(m, "boundaries"),
            boundaries=B.from_torch(m.boundaries) if hasattr(m, "boundaries") else None,
            dist_bin_pairwise_embed=B.from_torch(m.dist_bin_pairwise_embed)
            if hasattr(m, "dist_bin_pairwise_embed") else None,
            s_to_z=B.from_torch(m.s_to_z),
            s_to_z_transpose=B.from_torch(m.s_to_z_transpose),
            s_norm=B.from_torch(m.s_norm),
            pairformer_stack=B.from_torch(m.pairformer_stack),
            confidence_heads=B.from_torch(m.confidence_heads),
        )

    def __call__(
        self,
        feats: Feats,
        trunk: TrunkOutput,
        x_pred: Float[Array, "B M 3"] | None = None,
        multiplicity: int = 1,
    ) -> Confidence:
        s_inputs = trunk.s_inputs
        if self.inp_only:
            s, z = trunk.s_init, trunk.z_init
        else:
            s, z = trunk.s, trunk.z
        s = self.s_norm(s)

        z = (
            z
            + self.s_to_z(s_inputs)[:, :, None, :]
            + self.s_to_z_transpose(s_inputs)[:, None, :, :]
        )

        if x_pred is not None and self.distogram:
            token_to_rep_atom = feats.token_to_rep_atom
            x_pred_repr = _gather_rep_atoms(x_pred, token_to_rep_atom)
            d = _cdist(x_pred_repr, x_pred_repr)
            distogram = (d[..., None] > self.boundaries).sum(axis=-1).astype(jnp.int32)
            z = z + self.dist_bin_pairwise_embed(distogram)

        mask = feats.token_pad_mask
        pair_mask = mask[:, :, None] * mask[:, None, :]
        s, z = self.pairformer_stack(s, z, mask=mask, pair_mask=pair_mask)
        return self.confidence_heads(s=s, z=z, feats=feats, multiplicity=multiplicity)


@register_from_torch("promera.model.confidence.ContactModule")
class ContactModule(AbstractFromTorch):
    boundaries: Float[Array, "nb"]
    dist_bin_pairwise_embed: Embedding
    s_to_z: Linear
    s_to_z_transpose: Linear
    s_norm: LayerNorm
    pairformer_stack: PairformerModule
    to_contact_logits: Linear

    def __call__(
        self,
        feats: Feats,
        trunk: TrunkOutput,
        x_pred: Float[Array, "B M 3"],
        multiplicity: int = 1,
    ) -> Contact:
        s_inputs = trunk.s_inputs
        s = self.s_norm(trunk.s)
        z = trunk.z
        z = (
            z
            + self.s_to_z(s_inputs)[:, :, None, :]
            + self.s_to_z_transpose(s_inputs)[:, None, :, :]
        )

        token_to_rep_atom = feats.token_to_rep_atom
        x_pred_repr = _gather_rep_atoms(x_pred, token_to_rep_atom)
        pred_dist = _cdist(x_pred_repr, x_pred_repr)
        distogram = (pred_dist[..., None] > self.boundaries).sum(axis=-1).astype(jnp.int32)
        z = z + self.dist_bin_pairwise_embed(distogram)

        mask = feats.token_pad_mask
        pair_mask = mask[:, :, None] * mask[:, None, :]
        s, z = self.pairformer_stack(s, z, mask=mask, pair_mask=pair_mask)

        contact_logits = self.to_contact_logits(z + jnp.swapaxes(z, 1, 2))[..., 0]
        return Contact(contact_logits=contact_logits, pred_dist=pred_dist)
