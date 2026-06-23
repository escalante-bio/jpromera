"""Torch-free featurization: a schema dict -> model-ready feats.

Uses ``tinyprot`` (which is torch-free — pure numpy/rdkit/lmdb) plus numpy
reimplementations of promera's ``finalize_feats`` / ``collate``, so the full
sequence -> structure -> feats pipeline needs no PyTorch. ``tinyprot`` is
imported lazily (only when ``featurize`` is called), so ``import jpromera``
stays light.
"""

import numpy as np

_ATOM_KEYS = {
    "ref_pos", "ref_element", "ref_charge", "ref_space_uid", "ref_hydrogens",
    "atom_to_token", "ref_atom_name_chars", "atom_pad_mask", "atom_coords",
    "atom_resolved_mask", "atom_is_protein", "atom_is_rna", "atom_is_dna",
    "atom_is_ligand", "atom_is_std", "atom_supervise", "alt_coords",
    "alt_coords_mask",
}
_MSA_KEYS = {"msa", "msa_mask", "msa_chars", "msa_paired", "deletion_value",
             "has_deletion"}
_TOKEN_PAIR_KEYS = {"token_bonds", "token_contacts", "token_pair_supervise",
                    "distogram_supervise", "contact_supervise"}
_FLOATS = (np.float64, np.float32, np.float16)
_INTS = (np.int64, np.int32, np.int16, np.int8, np.uint8)


def _finalize(feats, name="x", seed_idx=0):
    """Numpy reimplementation of promera.inference.utils.finalize_feats."""
    feats["atom_pad_mask"] = np.ones_like(feats["ref_element"])
    feats["token_pad_mask"] = np.ones_like(feats["restype"])
    feats["is_epitope"] = np.zeros_like(feats["restype"])
    M = len(feats["restype"])
    bond_mat = np.zeros((M, M), dtype=int)
    bond_mat[tuple(feats["token_bonds"].T)] = 1
    bond_mat[tuple(feats["token_bonds"].T[::-1])] = 1
    feats["token_bonds"] = bond_mat
    feats["name"] = name
    feats["seed_idx"] = seed_idx
    a2t = feats["atom_to_token"]
    feats["atom_is_protein"] = feats["is_protein"][a2t]
    feats["atom_is_rna"] = feats["is_rna"][a2t]
    feats["atom_is_dna"] = feats["is_dna"][a2t]
    feats["atom_is_ligand"] = feats["is_ligand"][a2t]
    feats["atom_is_std"] = feats["is_std"][a2t]
    return feats


def _pad(v, dim, multiple_of=1):
    n = v.shape[dim]
    if n % multiple_of:
        n += multiple_of - n % multiple_of
    if n == v.shape[dim]:
        return v
    shape = list(v.shape)
    shape[dim] = n - v.shape[dim]
    return np.concatenate([v, np.zeros(shape, dtype=v.dtype)], dim)


def _collate_one(feats):
    """Numpy reimplementation of promera.data.utils.collate for one example
    (atoms padded to a multiple of 32; a leading batch axis added)."""
    out = {}
    for key, v in feats.items():
        if not isinstance(v, np.ndarray) or v.dtype not in _FLOATS + _INTS + (np.bool_,):
            out[key] = [v]
            continue
        if key in _ATOM_KEYS:
            v = _pad(v, 0, multiple_of=32)
        v = v[None]  # batch axis
        if v.dtype in _FLOATS:
            v = v.astype(np.float32)
        elif v.dtype in _INTS:
            v = v.astype(np.int64)
        out[key] = v
    return out


def build_msas(chains, cache=None, host_url="https://api.colabfold.com", verbose=True):
    """Build + cache a real MSA for any polymer chain that doesn't have one.

    ``tinyprot``'s featurizer only *looks up* precomputed MSAs (keyed by exact
    sequence) and silently falls back to a depth-1 dummy on a miss — so a novel
    or cropped sequence quietly loses its alignment. This builds the missing
    ones the same way the cache was originally populated: a ColabFold MMseqs2
    search (``https://api.colabfold.com``), written to the same cache key.

    Hits an external public server. Returns the query sequences newly built.
    """
    from tinyprot.msa import msa_exists_in_dir, _default_msa_dir, _get_seq_for_msa
    from tinyprot.mmseqs2 import download_from_server

    cache = cache or _default_msa_dir
    seqs = [c.seq if hasattr(c, "seq") else _get_seq_for_msa(c)
            for c in chains.values()
            if str(getattr(c, "type", "")).startswith("polymer")]
    missing = sorted({s for s in seqs if not msa_exists_in_dir(cache, s)})
    if missing:
        if verbose:
            print(f"[jpromera] building {len(missing)} MSA(s) via {host_url} "
                  f"(lengths {', '.join(str(len(s)) for s in missing)})")
        download_from_server(missing, cache, host_url=host_url)
    return missing


def ensure_tinyprot_data(verbose=True):
    """Make sure tinyprot's CCD + taxonomy LMDBs are present, downloading any
    that are missing — the programmatic equivalent of ``python -m tinyprot.init
    --download`` (prebuilt DBs from HuggingFace).

    Featurization needs the CCD (``Structure.from_schema`` reference conformers)
    and the taxonomy DB (cross-chain MSA pairing); without them tinyprot raises
    on the first ``featurize``. Only the missing database(s) are fetched. Returns
    the list of database names downloaded.
    """
    import os
    from tinyprot.init import (_ccd_path, _tax_path, _download_file,
                               _HF_CCD_URL, _HF_TAX_URL)

    targets = [(_ccd_path, _HF_CCD_URL, "CCD"), (_tax_path, _HF_TAX_URL, "taxonomy")]
    missing = [(p, u, d) for (p, u, d) in targets
               if not os.path.exists(os.path.join(p, "data.mdb"))]
    for path, url, label in missing:
        if verbose:
            print(f"[jpromera] tinyprot {label} DB not found at {path}; "
                  f"downloading prebuilt LMDB from HuggingFace ...")
        _download_file(url, os.path.join(path, "data.mdb"), f"Downloading {label}")
    return [d for _, _, d in missing]


def featurize(schema, name="x", seed_idx=0, build_msa=True, init_data=True):
    """Featurize a tinyprot schema dict -> (Feats, struct).

    Returns a typed ``Feats`` (jnp arrays, the inference feature set) and the
    tinyprot Structure (for writing the predicted coords back out). The full
    MSA is returned at its natural depth — the trunk subsamples it per recycle
    (see ``JPromera.trunk``), so there's no need to crop it here.

    ``build_msa=True`` (default) constructs + caches a real MSA for any chain
    missing one (ColabFold server) rather than silently using a depth-1 dummy;
    set ``build_msa=False`` to keep the old lookup-or-dummy behaviour offline.
    ``init_data=True`` (default) downloads tinyprot's CCD + taxonomy databases
    if absent (``python -m tinyprot.init --download``); set False to skip.
    """
    from tinyprot.structure import Structure
    from tinyprot.feature import AF3Featurizer
    from tinyprot.msa import load_msa_from_dir, construct_paired_msa

    from .feats import Feats

    if init_data:
        ensure_tinyprot_data()
    struct = Structure.from_schema(schema)
    if build_msa:
        build_msas(struct.chains)
    msas = load_msa_from_dir(seq=struct.chains)
    pairing = construct_paired_msa(msas)
    feats = AF3Featurizer(struct, msas, pairing).featurize(compute_frames=True)
    feats = _finalize(feats, name, seed_idx)
    batch = _collate_one(feats)
    return Feats.from_dict(batch), struct


# --- output: predicted coords -> structure (torch-free) -----------------------
def copy_coords(struct, coords):
    """Place predicted atom coords (``[M, 3]``) into a deep copy of the tinyprot
    Structure ``struct`` (numpy reimplementation of promera's
    ``_copy_sample_to_struct``). Atom order matches the featurizer's."""
    import copy as _copy

    struct = _copy.deepcopy(struct)
    coords = np.asarray(coords)
    if coords.ndim == 3:  # drop a leading batch/sample axis
        coords = coords[0]
    i = 0
    for chain in struct.chains.values():
        for j in range(len(chain.aname)):
            for k in range(len(chain.aname[j])):
                if chain.aname[j][k] != "":
                    chain.coords[j, k] = coords[i]
                    chain.mask[j, k] = True
                    i += 1
    return struct


def save_structure(struct, coords, path, metadata=True):
    """Write predicted coords to an mmCIF (torch-free; via tinyprot)."""
    s = copy_coords(struct, coords)
    s.to_mmcif(str(path), metadata=metadata)
    return s
