"""Leaf and small-composite layer translations for Promera.

Ported from the BoltzGen translation (joltzgen), adapted to Promera's module
structure and call signatures (which carry extra inference-only arguments such
as ``chunk_size`` that are no-ops in JAX).
"""


from __future__ import annotations
import einops
import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float


from .backend import (
    AbstractFromTorch,
    LayerNorm,
    Linear,
    Sequential,
    register_from_torch,
)


# --- Transition --------------------------------------------------------------
@register_from_torch("promera.model.layers.transition.Transition")
class Transition(AbstractFromTorch):
    norm: LayerNorm
    fc1: Linear
    fc2: Linear
    fc3: Linear
    silu: object
    hidden: int

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        v = self.norm(x)
        return self.fc3(jax.nn.silu(self.fc1(v)) * self.fc2(v))


# --- PairWeightedAveraging ---------------------------------------------------
@register_from_torch("promera.model.layers.pair_averaging.PairWeightedAveraging")
class PairWeightedAveraging(AbstractFromTorch):
    c_m: int
    c_z: int
    c_h: int
    num_heads: int
    inf: float
    norm_m: LayerNorm
    norm_z: LayerNorm
    proj_m: Linear
    proj_g: Linear
    proj_z: Linear
    proj_o: Linear

    def __call__(
        self,
        m: Float[Array, "B S N D"],
        z: Float[Array, "B N N D"],
        mask: Bool[Array, "B N N"],
    ) -> Float[Array, "B S N D"]:
        m = self.norm_m(m)
        z = self.norm_z(z)

        v = self.proj_m(m)
        v = v.reshape(*v.shape[:3], self.num_heads, self.c_h)
        v = jnp.transpose(v, (0, 3, 1, 2, 4))  # (b h s i d)

        b = self.proj_z(z)
        b = jnp.transpose(b, (0, 3, 1, 2))  # (b h i j)
        b = b + (1 - mask[:, None]) * -self.inf
        w = jax.nn.softmax(b, axis=-1)

        g = jax.nn.sigmoid(self.proj_g(m))

        o = jnp.einsum("bhij,bhsjd->bhsid", w, v)
        o = jnp.transpose(o, (0, 2, 3, 1, 4))
        o = o.reshape(*o.shape[:3], self.num_heads * self.c_h)
        return self.proj_o(g * o)


# --- OuterProductMean --------------------------------------------------------
@register_from_torch("promera.model.layers.outer_product_mean.OuterProductMean")
class OuterProductMean(AbstractFromTorch):
    c_hidden: int
    norm: LayerNorm
    proj_a: Linear
    proj_b: Linear
    proj_o: Linear

    def __call__(
        self, m: Float[Array, "B S N D"], mask: Bool[Array, "B S N"]
    ) -> Float[Array, "B N N c_out"]:
        mask = mask[..., None].astype(m.dtype)
        m = self.norm(m)
        a = self.proj_a(m) * mask
        b = self.proj_b(m) * mask

        mask = mask[:, :, None, :] * mask[:, :, :, None]
        num_mask = mask.sum(1).clip(min=1)
        z = jnp.einsum("bsic,bsjd->bijcd", a, b)
        z = z.reshape(*z.shape[:3], -1)
        z = z / num_mask
        return self.proj_o(z)


# --- Triangle multiplicative update (cueq kernel -> plain JAX) ----------------
@register_from_torch("promera.model.layers.triangular_mult.TriangleMultiplicationOutgoing")
class TriangleMultiplicationOutgoing(AbstractFromTorch):
    norm_in: LayerNorm
    p_in: Linear
    g_in: Linear
    norm_out: LayerNorm
    p_out: Linear
    g_out: Linear

    def __call__(
        self, x: Float[Array, "B N N D"], mask: Bool[Array, "B N N"]
    ) -> Float[Array, "B N N D"]:
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * jax.nn.sigmoid(self.g_in(x))
        x = x * mask[..., None]
        a, b = jnp.split(x, 2, axis=-1)
        x = jnp.einsum("bikd,bjkd->bijd", a, b)
        return self.p_out(self.norm_out(x)) * jax.nn.sigmoid(self.g_out(x_in))


@register_from_torch("promera.model.layers.triangular_mult.TriangleMultiplicationIncoming")
class TriangleMultiplicationIncoming(AbstractFromTorch):
    norm_in: LayerNorm
    p_in: Linear
    g_in: Linear
    norm_out: LayerNorm
    p_out: Linear
    g_out: Linear

    def __call__(
        self, x: Float[Array, "B N N D"], mask: Bool[Array, "B N N"]
    ) -> Float[Array, "B N N D"]:
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * jax.nn.sigmoid(self.g_in(x))
        x = x * mask[..., None]
        a, b = jnp.split(x, 2, axis=-1)
        x = jnp.einsum("bkid,bkjd->bijd", a, b)
        return self.p_out(self.norm_out(x)) * jax.nn.sigmoid(self.g_out(x_in))


# --- triangular_attention.primitives.Attention -------------------------------
@register_from_torch("promera.model.layers.triangular_attention.primitives.Attention")
class Attention(AbstractFromTorch):
    c_q: int
    c_k: int
    c_v: int
    c_hidden: int
    no_heads: int
    gating: bool
    linear_q: Linear
    linear_k: Linear
    linear_v: Linear
    linear_o: Linear
    linear_g: Linear | None
    sigmoid: object

    def __call__(
        self,
        q_x: Float[Array, "... Q C_q"],
        kv_x: Float[Array, "... K C_k"],
        biases: list | None,
    ) -> Float[Array, "... Q C_v"]:
        H, C = self.no_heads, self.c_hidden
        q = einops.rearrange(self.linear_q(q_x), "... Q (H C) -> ... Q H C", H=H)
        k = einops.rearrange(self.linear_k(kv_x), "... K (H C) -> ... K H C", H=H)
        v = einops.rearrange(self.linear_v(kv_x), "... V (H C) -> ... V H C", H=H)

        bias = biases[0]
        for b in biases[1:]:
            bias = bias + b  # additive, broadcasts to (..., H, Q, K)

        # dot_product_attention wants 4D (B, seq, H, C); flatten leading batch.
        batch, Q, K = q.shape[:-3], q.shape[-3], k.shape[-3]
        bias = jnp.broadcast_to(bias, batch + (H, Q, K))
        o = jax.nn.dot_product_attention(
            q.reshape(-1, Q, H, C), k.reshape(-1, K, H, C),
            v.reshape(-1, K, H, C), bias=bias.reshape(-1, H, Q, K),
            scale=C**-0.5,
        ).reshape(batch + (Q, H, C))

        if self.linear_g is not None:
            g = jax.nn.sigmoid(self.linear_g(q_x))
            o = o * einops.rearrange(g, "... (H C) -> ... H C", H=H)
        o = einops.rearrange(o, "... Q H C -> ... Q (H C)")
        return self.linear_o(o)


# --- TriangleAttention (starting / ending) -----------------------------------
@register_from_torch("promera.model.layers.triangular_attention.attention.TriangleAttention")
class TriangleAttention(AbstractFromTorch):
    c_in: int
    c_hidden: int
    no_heads: int
    starting: bool
    inf: float
    layer_norm: LayerNorm
    linear: Linear
    mha: Attention

    def __call__(
        self,
        x: Float[Array, "... I J C_in"],
        mask: Bool[Array, "... I J"] | None = None,
    ) -> Float[Array, "... I J C_in"]:
        if mask is None:
            mask = jnp.ones(x.shape[:-1], dtype=x.dtype)
        if not self.starting:
            x = einops.rearrange(x, "... I J C -> ... J I C")
            mask = einops.rearrange(mask, "... I J -> ... J I")

        x = self.layer_norm(x)
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        triangle_bias = einops.rearrange(self.linear(x), "... J I H -> ... 1 H J I")
        x = self.mha(q_x=x, kv_x=x, biases=[mask_bias, triangle_bias])

        if not self.starting:
            x = einops.rearrange(x, "... J I C -> ... I J C")
        return x


# TriangleAttentionEndingNode is a subclass with starting=False; register it too
register_from_torch("promera.model.layers.triangular_attention.attention.TriangleAttentionEndingNode")(TriangleAttention)


# --- AttentionPairBias -------------------------------------------------------
@register_from_torch("promera.model.layers.attention.AttentionPairBias")
class AttentionPairBias(AbstractFromTorch):
    c_s: int
    num_heads: int
    head_dim: int
    inf: float
    initial_norm: bool
    norm_s: LayerNorm | None
    proj_q: Linear
    proj_k: Linear
    proj_v: Linear
    proj_g: Linear
    proj_z: Sequential
    proj_o: Linear

    def __call__(
        self,
        s: Float[Array, "B S D"],
        z: Float[Array, "B N N P"],
        mask: Bool[Array, "B N N"],
        multiplicity: int = 1,
        to_keys=None,
        model_cache=None,
        bias=None,
    ):
        B = s.shape[0]
        if self.initial_norm:
            s = self.norm_s(s)

        if to_keys is not None:
            k_in = to_keys(s)
            mask = to_keys(mask[..., None])[..., 0]
        else:
            k_in = s

        q = self.proj_q(s).reshape(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).reshape(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).reshape(B, -1, self.num_heads, self.head_dim)

        # ``bias`` (= proj_z(z), shape (B, num_heads, N, N)) is constant across a
        # diffusion rollout, so the sampler precomputes it once; recompute only
        # on the cache-free path.
        if bias is None:
            bias = self.proj_z(z)
        g = jax.nn.sigmoid(self.proj_g(s))

        o = jax.nn.dot_product_attention(
            query=q, key=k, value=v, bias=bias,
            mask=mask.astype(bool)[:, None, None, :],
            scale=self.head_dim**-0.5,
        )  # (B, seq, num_heads, head_dim)
        o = o.reshape(B, -1, self.c_s)
        return self.proj_o(g * o)


# --- AdaLN -------------------------------------------------------------------
@register_from_torch("promera.model.transformers.AdaLN")
class AdaLN(AbstractFromTorch):
    a_norm: LayerNorm
    s_norm: LayerNorm
    s_scale: Linear
    s_bias: Linear

    def __call__(self, a, s):
        a = self.a_norm(a)
        s = self.s_norm(s)
        return jax.nn.sigmoid(self.s_scale(s)) * a + self.s_bias(s)


# --- ConditionedTransitionBlock ---------------------------------------------
@register_from_torch("promera.model.transformers.ConditionedTransitionBlock")
class ConditionedTransitionBlock(AbstractFromTorch):
    adaln: AdaLN
    swish_gate: Sequential
    a_to_b: Linear
    b_to_a: Linear
    output_projection: Sequential

    def __call__(self, a, s):
        a = self.adaln(a, s)
        b = self.swish_gate(a) * self.a_to_b(a)
        return self.output_projection(s) * self.b_to_a(b)


# --- FourierEmbedding --------------------------------------------------------
@register_from_torch("promera.model.encoders.FourierEmbedding")
class FourierEmbedding(AbstractFromTorch):
    proj: Linear

    def __call__(self, times):
        times = einops.rearrange(times, "b -> b 1")
        return jnp.cos(2 * jnp.pi * self.proj(times))


# --- RelativePositionEncoder -------------------------------------------------
@register_from_torch("promera.model.encoders.RelativePositionEncoder")
class RelativePositionEncoder(AbstractFromTorch):
    r_max: int
    s_max: int
    linear_layer: Linear

    def __call__(self, feats: dict):
        asym_id = feats.asym_id
        residue_index = feats.residue_index
        entity_id = feats.entity_id
        token_index = feats.token_index
        sym_id = feats.sym_id

        b_same_chain = jnp.equal(asym_id[:, :, None], asym_id[:, None, :])
        b_same_residue = jnp.equal(residue_index[:, :, None], residue_index[:, None, :])
        b_same_entity = jnp.equal(entity_id[:, :, None], entity_id[:, None, :])

        d_residue = jnp.clip(
            residue_index[:, :, None] - residue_index[:, None, :] + self.r_max,
            0, 2 * self.r_max,
        )
        d_residue = jnp.where(
            b_same_chain, d_residue, jnp.zeros_like(d_residue) + 2 * self.r_max + 1
        )
        a_rel_pos = jax.nn.one_hot(d_residue, 2 * self.r_max + 2)

        d_token = jnp.clip(
            token_index[:, :, None] - token_index[:, None, :] + self.r_max,
            0, 2 * self.r_max,
        )
        d_token = jnp.where(
            b_same_chain & b_same_residue,
            d_token, jnp.zeros_like(d_token) + 2 * self.r_max + 1,
        )
        a_rel_token = jax.nn.one_hot(d_token, 2 * self.r_max + 2)

        d_chain = jnp.clip(
            sym_id[:, :, None] - sym_id[:, None, :] + self.s_max, 0, 2 * self.s_max
        )
        d_chain = jnp.where(
            b_same_chain, jnp.zeros_like(d_chain) + 2 * self.s_max + 1, d_chain
        )
        a_rel_chain = jax.nn.one_hot(d_chain, 2 * self.s_max + 2)

        return self.linear_layer(
            jnp.concatenate(
                [a_rel_pos, a_rel_token, b_same_entity[..., None].astype(jnp.float32),
                 a_rel_chain],
                axis=-1,
            )
        )


# --- dropout (inference: deterministic, mask = 1) ----------------------------
def get_dropout_mask(dropout, z, training, columnwise=False, *, key=None):
    """Inference dropout mask. With training=False the mask is all ones."""
    if not training:
        shape = (
            (z.shape[0], 1, z.shape[2], 1)
            if columnwise
            else (z.shape[0], z.shape[1], 1, 1)
        )
        return jnp.ones(shape, dtype=z.dtype)
    rate = dropout
    shape = (
        (z.shape[0], 1, z.shape[2], 1) if columnwise else (z.shape[0], z.shape[1], 1, 1)
    )
    d = jax.random.bernoulli(key, 1 - rate, shape).astype(z.dtype)
    return d / (1 - rate)
