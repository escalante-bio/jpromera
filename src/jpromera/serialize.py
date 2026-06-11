"""Torch-free serialization of a converted JPromera model.

``save_model`` writes the array data plus a structural skeleton (with
``jax.ShapeDtypeStruct`` placeholders) so the model can be loaded for inference
without PyTorch / Promera installed.
"""

import pickle

import equinox as eqx
import jax


def save_model(model, path: str) -> None:
    eqx.tree_serialise_leaves(f"{path}.eqx", model)
    skeleton = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype) if eqx.is_array(x) else x,
        model,
        is_leaf=eqx.is_array,
    )
    with open(f"{path}.skeleton.pkl", "wb") as f:
        pickle.dump(skeleton, f)


def load_model(path: str):
    with open(f"{path}.skeleton.pkl", "rb") as f:
        skeleton = pickle.load(f)
    return eqx.tree_deserialise_leaves(f"{path}.eqx", skeleton)


def load_pretrained(repo_id: str = "escalante-bio/jpromera",
                    name: str = "promera_2606", revision: str | None = None):
    """Download converted JAX/Equinox weights from HuggingFace and load them.

    Fully torch-free: no PyTorch, no Promera source, no checkpoint conversion —
    just the published ``.eqx`` + skeleton. This is the fast path for inference;
    ``jpromera.load_model()`` (the ``[convert]`` extra) is only needed to convert
    a PyTorch checkpoint yourself.
    """
    from huggingface_hub import hf_hub_download

    # both files download into the same snapshot dir; load_model wants the stem
    stem = hf_hub_download(repo_id=repo_id, filename=f"{name}.eqx",
                           revision=revision)[: -len(".eqx")]
    hf_hub_download(repo_id=repo_id, filename=f"{name}.skeleton.pkl", revision=revision)
    return load_model(stem)
