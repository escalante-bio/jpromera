"""PyTorch conversion layer.

Importing this module wires every PyTorch / Promera type into the ``from_torch``
singledispatch dispatcher. The core package (``jpromera``) only *declares* the
mapping (by dotted-path string, in ``backend.REGISTRY``); this module resolves
those paths against the actually-installed torch/promera and registers the base
types, activations and a couple of special cases.

    import jpromera.convert            # enables conversion
    jp = jpromera.convert.load_model() # or jpromera.load_model()
"""

import importlib

import einops.layers.torch  # noqa: F401  (so "einops.layers.torch.Rearrange" resolves)
import numpy as np
import torch

import jpromera  # noqa: F401  (ensure core classes are imported -> REGISTRY populated)
from ..backend import (
    GELU,
    LayerNorm,
    ReLU,
    REGISTRY,
    Sigmoid,
    SiLU,
    SwiGLU,
    from_torch,
)
from .loading import default_weights, load_model, load_torch_model  # noqa: F401


# --- base python / torch container types --------------------------------------
from_torch.register(str, lambda x: x)
from_torch.register(torch.Tensor, lambda x: np.array(x.detach().cpu()))
from_torch.register(np.ndarray, lambda x: x)
from_torch.register(int, lambda x: x)
from_torch.register(float, lambda x: x)
from_torch.register(bool, lambda x: x)
from_torch.register(type(None), lambda x: x)
from_torch.register(tuple, lambda x: tuple(map(from_torch, x)))
from_torch.register(list, lambda x: [from_torch(v) for v in x])
from_torch.register(dict, lambda x: {k: from_torch(v) for k, v in x.items()})
from_torch.register(torch.nn.ModuleList, lambda x: [from_torch(m) for m in x])
try:
    from omegaconf.listconfig import ListConfig

    from_torch.register(ListConfig, lambda x: [from_torch(m) for m in x])
except Exception:  # pragma: no cover
    pass

# --- activations --------------------------------------------------------------
from_torch.register(torch.nn.ReLU, lambda _: ReLU())
from_torch.register(torch.nn.GELU, lambda _: GELU())
from_torch.register(torch.nn.Sigmoid, lambda _: Sigmoid())
from_torch.register(torch.nn.SiLU, lambda _: SiLU())

# --- promera special cases ----------------------------------------------------
import promera.model.mha as _mha
import promera.model.utils as _utils
import promera.model.layers.triangular_attention.primitives as _triprim

from_torch.register(_utils.SwiGLU, lambda _: SwiGLU())
from_torch.register(_mha.SwiGLU, lambda _: SwiGLU())


def _triprim_layer_norm(m):
    assert len(m.c_in) == 1
    return LayerNorm(weight=from_torch(m.weight), bias=from_torch(m.bias), eps=m.eps)


from_torch.register(_triprim.LayerNorm, _triprim_layer_norm)


# --- resolve the string-path registry declared by the core classes ------------
def _resolve(path: str):
    module_path, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), name)


for _path, _cls in REGISTRY:
    from_torch.register(_resolve(_path), _cls.from_torch)
