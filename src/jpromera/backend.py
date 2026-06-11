"""Core conversion framework (JAX-only).

Defines the ``from_torch`` singledispatch *dispatcher*, the ``AbstractFromTorch``
base class, ``register_from_torch``, and the vanilla leaf modules (Linear,
LayerNorm, Embedding, Sequential, Identity, Rearrange, activations).

This module imports NO torch / promera — the eqx classes are pure JAX. Their
``from_torch`` staticmethods reference torch types only in (deferred)
annotations, and the actual ``from_torch.register(...)`` calls that wire torch
types to these converters live in ``jpromera.convert`` (which does import
torch). So ``import jpromera`` is torch-free; ``import jpromera.convert`` enables
weight conversion.
"""

from __future__ import annotations

from dataclasses import fields
from functools import singledispatch

import einops
import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int


@singledispatch
def from_torch(x):
    raise NotImplementedError(
        f"from_torch not implemented for {type(x)}: {x}. "
        "Did you forget to `import jpromera.convert`?"
    )


# Records (dotted-path-of-torch-type, eqx-class) declared via the decorator.
# ``jpromera.convert`` resolves the paths and wires them into ``from_torch`` — so
# core never imports torch / promera, only declares the mapping by name.
REGISTRY: list[tuple[str, type]] = []


def register_from_torch(torch_type_path: str):
    """Decorator: declare ``cls.from_torch`` converts the torch type at the dotted
    path ``torch_type_path`` (e.g. ``"torch.nn.Linear"``). Resolved lazily by
    ``jpromera.convert`` so the class module stays torch-free."""

    def decorator(cls):
        REGISTRY.append((torch_type_path, cls))
        return cls

    return decorator


class AbstractFromTorch(eqx.Module):
    """Default ``from_torch`` for equinox modules.

    Matches the fields of the equinox module to the children, parameters and
    buffers of the torch module by name. Missing fields are allowed only if the
    corresponding equinox field type admits ``None``.
    """

    @classmethod
    def from_torch(cls, model):
        field_to_type = {field.name: field.type for field in fields(cls)}
        kwargs = {
            child: from_torch(child_module)
            for child, child_module in model.named_children()
        } | {
            parameter_name: from_torch(parameter)
            for parameter_name, parameter in model.named_parameters(recurse=False)
        } | {
            buffer_name: from_torch(buffer)
            for buffer_name, buffer in model.named_buffers(recurse=False)
            if buffer_name in field_to_type
        }

        for field_name, field_type in field_to_type.items():
            if field_name in kwargs:
                continue
            if not hasattr(model, field_name):
                if _is_optional(field_type):
                    kwargs[field_name] = None
                else:
                    raise ValueError(
                        f"Field {field_name} for {cls} is not optional but is "
                        f"missing from torch model {type(model)}"
                    )
            else:
                kwargs[field_name] = from_torch(getattr(model, field_name))

        torch_not_equinox = kwargs.keys() - field_to_type.keys()
        if torch_not_equinox:
            raise ValueError(
                f"Properties in torch model not found in equinox module {cls}: "
                f"{torch_not_equinox}"
            )

        return cls(**kwargs)


def _is_optional(field_type) -> bool:
    """Whether ``None`` is a valid value for an annotation (handles str types)."""
    if field_type is None or field_type is type(None):
        return True
    if isinstance(field_type, str):
        return "None" in field_type
    try:
        return isinstance(None, field_type)
    except TypeError:
        # typing constructs like `X | None`
        return "None" in str(field_type) or "NoneType" in str(field_type)


# --- activations (eqx.Module wrappers; bare fns can't be pytree leaves) -------
class ReLU(eqx.Module):
    def __call__(self, x):
        return jax.nn.relu(x)


class GELU(eqx.Module):
    def __call__(self, x):
        return jax.nn.gelu(x)


class Sigmoid(eqx.Module):
    def __call__(self, x):
        return jax.nn.sigmoid(x)


class SiLU(eqx.Module):
    def __call__(self, x):
        return jax.nn.silu(x)


class SwiGLU(eqx.Module):
    def __call__(self, x):
        a, gates = jnp.split(x, 2, axis=-1)
        return jax.nn.silu(gates) * a


# --- vanilla leaf modules -----------------------------------------------------
@register_from_torch("einops.layers.torch.Rearrange")
class Rearrange(eqx.Module):
    pattern: str
    axes_lengths: dict

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return einops.rearrange(x, self.pattern, **self.axes_lengths)

    @staticmethod
    def from_torch(r):
        return Rearrange(pattern=r.pattern, axes_lengths=r.axes_lengths)


@register_from_torch("torch.nn.Linear")
class Linear(eqx.Module):
    """Linear layer matching pytorch semantics (weight is (Out, In))."""

    weight: Float[Array, "Out In"]
    bias: Float[Array, "Out"] | None

    def __call__(self, x: Float[Array, "... In"]) -> Float[Array, "... Out"]:
        o = einops.einsum(x, self.weight, "... In, Out In -> ... Out")
        if self.bias is not None:
            o = o + jnp.broadcast_to(self.bias, x.shape[:-1] + (self.bias.shape[-1],))
        return o

    @staticmethod
    def from_torch(l):
        return Linear(weight=from_torch(l.weight), bias=from_torch(l.bias))


@register_from_torch("torch.nn.Identity")
class Identity(eqx.Module):
    def __call__(self, x):
        return x

    @staticmethod
    def from_torch(_):
        return Identity()


@register_from_torch("torch.nn.LayerNorm")
class LayerNorm(eqx.Module):
    """LayerNorm matching pytorch semantics."""

    weight: Float[Array, "Out"] | None
    bias: Float[Array, "Out"] | None
    eps: float

    def __call__(self, x: Float[Array, "... Out"]) -> Float[Array, "... Out"]:
        # normalize over the last axis (pytorch LayerNorm, biased variance)
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x

    @staticmethod
    def from_torch(l):
        return LayerNorm(weight=from_torch(l.weight), bias=from_torch(l.bias), eps=l.eps)


@register_from_torch("torch.nn.Sequential")
class Sequential(eqx.Module):
    # pytorch stores submodules in an OrderedDict keyed by string indices.
    _modules: dict

    def __call__(self, x):
        for idx in range(len(self._modules)):
            x = self._modules[str(idx)](x)
        return x

    @staticmethod
    def from_torch(module):
        return Sequential(_modules=from_torch(module._modules))


@register_from_torch("torch.nn.Embedding")
class Embedding(eqx.Module):
    weight: Float[Array, "V D"]

    def __call__(self, tokens: Int[Array, "..."]) -> Float[Array, "... D"]:
        return self.weight[tokens]

    @staticmethod
    def from_torch(m):
        return Embedding(weight=from_torch(m.weight))
