"""Typed model-input container — replaces the feats dict.

Holds only the features the inference path actually reads, as a plain pytree
(so it JITs cleanly and `feats.restype` replaces `feats["restype"]`). String
labels (`asym_id_`, `residue_name`, ...) and training-only targets are dropped.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


class Feats(eqx.Module):
    """Dims:  B batch · N tokens · A atoms (multiple of 32) · S MSA rows."""

    # --- token / residue level (B, N) ---
    restype: Int[Array, "B N"]
    profile: Float[Array, "B N 34"]                 # 34 = num token types
    deletion_mean: Float[Array, "B N"]
    is_std: Bool[Array, "B N"]                       # standard-residue mask (mask_std_feats)
    is_epitope: Int[Array, "B N"]                    # epitope/paratope hotspots (0 = none)
    token_pad_mask: Int[Array, "B N"]
    token_to_rep_atom: Int[Array, "B N"]             # token -> representative atom index
    token_bonds: Int[Array, "B N N"]
    residue_index: Int[Array, "B N"]                 # ┐
    token_index: Int[Array, "B N"]                   # │ relative-position encoding
    asym_id: Int[Array, "B N"]                       # │ (chain / entity / symmetry ids)
    entity_id: Int[Array, "B N"]                     # │
    sym_id: Int[Array, "B N"]                        # ┘
    frames_mask: Bool[Array, "B N"]                  # tokens with a valid backbone frame
                                                     # (not model-read; for pTM/ipTM scoring)

    # --- atom level (B, A) ---
    ref_pos: Float[Array, "B A 3"]                   # reference conformer coords
    ref_element: Int[Array, "B A"]
    ref_charge: Int[Array, "B A"]
    ref_hydrogens: Int[Array, "B A"]
    ref_atom_name_chars: Int[Array, "B A 4"]
    ref_space_uid: Int[Array, "B A"]                 # conformer/residue grouping
    atom_to_token: Int[Array, "B A"]
    atom_pad_mask: Int[Array, "B A"]
    atom_is_std: Bool[Array, "B A"]

    # --- MSA (B, S, N) ---
    msa: Int[Array, "B S N"]
    msa_mask: Bool[Array, "B S N"]
    msa_paired: Bool[Array, "B S N"]
    has_deletion: Bool[Array, "B S N"]
    deletion_value: Float[Array, "B S N"]

    # --- optional distance conditioning (design); None when unused ---
    distogram_emb: Int[Array, "B N N"] | None = None
    distogram_mask: Bool[Array, "B N N"] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Feats":
        """Build from a featurizer dict, keeping only the fields we use."""
        import numpy as np
        names = {f.name for f in dataclasses.fields(cls)}

        def arr(v):
            v = np.asarray(v)
            return jnp.asarray(v.astype(np.float32) if v.dtype == np.float64 else v)

        return cls(**{k: arr(v) for k, v in d.items() if k in names and v is not None})

    def to_dict(self) -> dict:
        """Present fields as a plain {name: numpy array} dict (e.g. to feed the
        PyTorch reference model). Torch-free."""
        import numpy as np
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if v is not None:
                out[f.name] = np.asarray(v)
        return out
