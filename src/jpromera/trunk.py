"""Trunk + atom-encoder/transformer translations for Promera.

Includes the windowed atom attention encoder/decoder, the diffusion
transformer (used both as the atom transformer and the token transformer), the
MSA and Pairformer stacks (scanned), the input embedder and the distogram head.
"""


from __future__ import annotations
import einops
import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jax import tree
from jaxtyping import Array, Bool, Float



from .backend import (
    AbstractFromTorch,
    Embedding,
    Identity,
    LayerNorm,
    Linear,
    Sequential,
    from_torch,
    register_from_torch,
)
from .config import NTOKS as _ntoks
from .layers import (
    AdaLN,
    AttentionPairBias,
    ConditionedTransitionBlock,
    OuterProductMean,
    PairWeightedAveraging,
    Transition,
    TriangleAttention,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
    get_dropout_mask,
)


class TrunkOutput(eqx.Module):
    """Typed result of the trunk (``JPromera.fold``); also the conditioning
    consumed by the diffusion sampler and the confidence/contact heads."""

    s: Float[Array, "b n ts"]
    z: Float[Array, "b n n tz"]
    s_inputs: Float[Array, "b n d"]
    s_init: Float[Array, "b n ts"]
    z_init: Float[Array, "b n n tz"]
    relpos: Float[Array, "b n n tz"]
    pdistogram: Float[Array, "b n n nbins"]


# --- atom windowing helpers --------------------------------------------------
def get_indexing_matrix(K, W, H):
    """Numpy port of promera.model.encoders.get_indexing_matrix (trace-time)."""
    assert W % 2 == 0
    assert H % (W // 2) == 0
    h = H // (W // 2)
    assert h % 2 == 0
    arange = np.arange(2 * K)
    index = np.clip((arange[None, :] - arange[:, None]) + h // 2, 0, h + 1)
    index = index.reshape(K, 2, 2 * K)[:, 0, :]  # (K, 2K)
    onehot = np.eye(h + 2, dtype=np.float32)[index][..., 1:-1]  # (K, 2K, h)
    onehot = onehot.transpose(1, 0, 2)  # (2K, K, h)
    return onehot.reshape(2 * K, h * K)  # (2K, h*K)


class SingleToKeys(eqx.Module):
    """Gather window keys: (B, N, D) -> (B, K, H, D)."""

    indexing_matrix: Float[Array, "J K"]
    W: int
    H: int

    def __call__(self, single: Float[Array, "B N D"]) -> Float[Array, "B K H D"]:
        B, N, D = single.shape
        K = N // self.W
        single = single.reshape(B, 2 * K, self.W // 2, D)
        r = jnp.einsum("b j i d, j k -> b k i d", single, self.indexing_matrix)
        return r.reshape(B, K, self.H, D)


class ReshapedSTK(eqx.Module):
    """to_keys wrapper used inside the windowed atom transformer."""

    to_keys: SingleToKeys
    B: int
    NW: int
    W: int
    H: int

    def __call__(self, x):
        return self.to_keys(x.reshape(self.B, self.NW * self.W, -1)).reshape(
            self.B * self.NW, self.H, -1
        )


# --- DiffusionTransformer ----------------------------------------------------
@register_from_torch("promera.model.transformers.DiffusionTransformerLayer")
class DiffusionTransformerLayer(AbstractFromTorch):
    adaln: AdaLN
    pair_bias_attn: AttentionPairBias
    output_projection_linear: Linear
    output_projection: Sequential
    transition: ConditionedTransitionBlock

    def __call__(self, a, s, z, mask=None, to_keys=None, multiplicity=1, bias=None):
        b = self.adaln(a, s)
        b = self.pair_bias_attn(
            s=b, z=z, mask=mask, multiplicity=multiplicity, to_keys=to_keys, bias=bias
        )
        b = self.output_projection(s) * b
        a = a + b
        a = a + self.transition(a, s)
        return a


@register_from_torch("promera.model.transformers.DiffusionTransformer")
class DiffusionTransformer(eqx.Module):
    stacked_parameters: DiffusionTransformerLayer | None
    static: DiffusionTransformerLayer | None
    depth: int

    @staticmethod
    def from_torch(m: _transformers.DiffusionTransformer):
        layers = [from_torch(layer) for layer in m.layers]
        if len(layers) == 0:
            return DiffusionTransformer(None, None, 0)
        _, static = eqx.partition(layers[0], eqx.is_inexact_array)
        stacked = tree.map(
            lambda *v: jnp.stack(v, 0),
            *[eqx.filter(layer, eqx.is_inexact_array) for layer in layers],
        )
        return DiffusionTransformer(stacked, static, len(layers))

    def precompute_biases(self, z):
        """Per-layer pair bias ``proj_z(z)`` for every stacked layer.

        ``z`` is constant across a diffusion rollout, so the sampler computes
        these once. Returns a ``[depth, ...]`` array (None if depth == 0).
        """
        if self.depth == 0:
            return None

        def one(params):
            layer = eqx.combine(self.static, params)
            return layer.pair_bias_attn.proj_z(z)

        return jax.vmap(one)(self.stacked_parameters)

    def __call__(self, a, s, z, mask=None, to_keys=None, multiplicity=1, biases=None):
        if self.depth == 0:
            return a

        @jax.checkpoint
        def body_fn(a, layer_inputs):
            params, bias = layer_inputs
            layer = eqx.combine(self.static, params)
            return layer(a, s, z, mask=mask, to_keys=to_keys,
                         multiplicity=multiplicity, bias=bias), None

        return jax.lax.scan(body_fn, a, (self.stacked_parameters, biases))[0]


@register_from_torch("promera.model.transformers.AtomTransformer")
class AtomTransformer(AbstractFromTorch):
    attn_window_queries: int
    attn_window_keys: int
    diffusion_transformer: DiffusionTransformer

    def _window_p(self, p):
        """Reshape the atom-pair rep ``p`` into the windowed form the inner
        transformer consumes (mirrors the ``p`` reshape in ``__call__``)."""
        W, H = self.attn_window_queries, self.attn_window_keys
        if W is None:
            return p
        return p.reshape(p.shape[0] * p.shape[1], W, H, -1)

    def precompute_biases(self, p):
        """Per-layer pair bias for the windowed atom-pair rep ``p`` (constant
        across a diffusion rollout)."""
        return self.diffusion_transformer.precompute_biases(self._window_p(p))

    def __call__(self, q, c, p, to_keys=None, mask=None, multiplicity=1, biases=None):
        W = self.attn_window_queries
        H = self.attn_window_keys
        if W is not None:
            B, N, D = q.shape
            NW = N // W
            q = q.reshape(B * NW, W, -1)
            c = c.reshape(B * NW, W, -1)
            if mask is not None:
                mask = mask.reshape(B * NW, W)
            p = p.reshape(p.shape[0] * NW, W, H, -1)
            to_keys_new = ReshapedSTK(to_keys=to_keys, B=B, NW=NW, W=W, H=H)
        else:
            to_keys_new = None

        q = self.diffusion_transformer(
            a=q, s=c, z=p, mask=mask.astype(jnp.float32),
            to_keys=to_keys_new, multiplicity=multiplicity, biases=biases,
        )
        if W is not None:
            q = q.reshape(B, NW * W, D)
        return q


# --- AtomAttentionEncoder ----------------------------------------------------
@register_from_torch("promera.model.encoders.AtomAttentionEncoder")
class AtomAttentionEncoder(AbstractFromTorch):
    embed_atom_features: Linear
    embed_atompair_ref_pos: Linear | None
    embed_atompair_ref_dist: Linear | None
    embed_atompair_mask: Linear | None
    atoms_per_window_queries: int
    atoms_per_window_keys: int
    structure_prediction: bool
    atom_encoder_depth: int
    s_to_c_trans: Sequential | None
    z_to_p_trans: Sequential | None
    r_to_q_trans: Linear | None
    c_to_p_trans_k: Sequential | None
    c_to_p_trans_q: Sequential | None
    p_mlp: Sequential | None
    atom_encoder: AtomTransformer
    atom_to_token_trans: Sequential

    def precompute(self, feats, s_trunk=None, z=None):
        """Coordinate-independent prefix of the encoder.

        Everything here depends only on ref features + trunk outputs, not on the
        diffusion coordinates ``r``, so the sampler runs it once per rollout.
        Returns ``(q0, c, p, to_keys, atom_biases)`` where ``q0`` is the atom
        embedding before the per-step ``r`` projection and ``atom_biases`` are
        the inner atom-transformer's per-layer pair biases.
        """
        B, N, _ = feats.ref_pos.shape
        atom_mask = feats.atom_pad_mask.astype(bool)
        # `restype.shape[1]` is the token count for both int [B, N] and soft
        # [B, N, NTOKS] (mosaic design) restype.
        M = feats.restype.shape[1]

        atom_ref_pos = feats.ref_pos
        atom_uid = feats.ref_space_uid

        atom_feats = [
            atom_ref_pos,
            feats.ref_charge[..., None].astype(jnp.float32),
            feats.atom_pad_mask[..., None].astype(jnp.float32),
            jax.nn.one_hot(feats.ref_element, 128),
        ]
        name_chars = jax.nn.one_hot(feats.ref_atom_name_chars, 64).reshape(
            B, N, 4 * 64
        )
        name_chars = name_chars * feats.atom_is_std[..., None].astype(jnp.float32)
        atom_feats.append(name_chars)
        atom_feats.append(jax.nn.one_hot(feats.ref_hydrogens, 8))
        atom_feats = jnp.concatenate(atom_feats, axis=-1)

        c = self.embed_atom_features(atom_feats)

        W, H = self.atoms_per_window_queries, self.atoms_per_window_keys
        K = N // W
        to_keys = SingleToKeys(
            indexing_matrix=jnp.asarray(get_indexing_matrix(K, W, H)), W=W, H=H
        )

        if self.atom_encoder_depth > 0:
            atom_ref_pos_queries = atom_ref_pos.reshape(B, K, W, 1, 3)
            atom_ref_pos_keys = to_keys(atom_ref_pos).reshape(B, K, 1, H, 3)
            d = atom_ref_pos_keys - atom_ref_pos_queries
            d_norm = jnp.sum(d * d, axis=-1, keepdims=True)
            d_norm = 1 / (1 + d_norm)

            atom_mask_q = atom_mask.reshape(B, K, W, 1)
            atom_mask_k = to_keys(atom_mask[..., None].astype(jnp.float32)).reshape(
                B, K, 1, H
            ).astype(bool)
            atom_uid_q = atom_uid.reshape(B, K, W, 1)
            atom_uid_k = to_keys(atom_uid[..., None].astype(jnp.float32)).reshape(
                B, K, 1, H
            ).astype(jnp.int32)
            v = (atom_mask_q & atom_mask_k & (atom_uid_q == atom_uid_k)).astype(
                jnp.float32
            )[..., None]
            p = self.embed_atompair_ref_pos(d) * v
            p = p + self.embed_atompair_ref_dist(d_norm) * v
            p = p + self.embed_atompair_mask(v) * v
        else:
            p = None

        q0 = c

        if self.structure_prediction:
            atom_to_token = (
                jax.nn.one_hot(feats.atom_to_token, M)
                * feats.atom_pad_mask[..., None].astype(jnp.float32)
            )
            s_to_c = self.s_to_c_trans(s_trunk)
            s_to_c = atom_to_token @ s_to_c
            c = c + s_to_c

            atom_to_token_q = atom_to_token.reshape(B, K, W, atom_to_token.shape[-1])
            atom_to_token_k = to_keys(atom_to_token)
            z_to_p = self.z_to_p_trans(z)
            z_to_p = jnp.einsum(
                "bijd,bwki,bwlj->bwkld", z_to_p, atom_to_token_q, atom_to_token_k
            )
            p = p + z_to_p

        if self.atom_encoder_depth > 0:
            p = p + self.c_to_p_trans_q(c.reshape(B, K, W, 1, c.shape[-1]))
            p = p + self.c_to_p_trans_k(to_keys(c).reshape(B, K, 1, H, c.shape[-1]))
            p = p + self.p_mlp(p)
            atom_biases = self.atom_encoder.precompute_biases(p)
        else:
            atom_biases = None

        return q0, c, p, to_keys, atom_biases

    def apply_coords(self, q0, c, p, to_keys, atom_biases, r, feats,
                     multiplicity=1):
        """Per-step part of the encoder: inject coordinates ``r`` and run the
        atom transformer using the precomputed prefix."""
        atom_mask = feats.atom_pad_mask.astype(bool)
        M = feats.restype.shape[1]

        q = q0
        if self.structure_prediction:
            q = q + self.r_to_q_trans(r)

        if self.atom_encoder_depth > 0:
            q = self.atom_encoder(
                q=q, mask=atom_mask, c=c, p=p, multiplicity=multiplicity,
                to_keys=to_keys, biases=atom_biases,
            )

        q_to_a = self.atom_to_token_trans(q)
        atom_to_token = (
            jax.nn.one_hot(feats.atom_to_token, M)
            * feats.atom_pad_mask[..., None].astype(jnp.float32)
        )
        atom_to_token_mean = atom_to_token / (
            atom_to_token.sum(axis=1, keepdims=True) + 1e-6
        )
        a = jnp.swapaxes(atom_to_token_mean, 1, 2) @ q_to_a
        return a, q, c, p, to_keys

    def __call__(self, feats, s_trunk=None, z=None, r=None, multiplicity=1,
                 model_cache=None):
        q0, c, p, to_keys, atom_biases = self.precompute(feats, s_trunk, z)
        return self.apply_coords(
            q0, c, p, to_keys, atom_biases, r, feats, multiplicity
        )


# --- AtomAttentionDecoder ----------------------------------------------------
@register_from_torch("promera.model.encoders.AtomAttentionDecoder")
class AtomAttentionDecoder(AbstractFromTorch):
    a_to_q_trans: Linear
    atom_decoder: AtomTransformer
    atom_feat_to_atom_pos_update: Sequential

    def precompute_biases(self, p):
        """Per-layer pair biases for the decoder's atom transformer (constant
        across a diffusion rollout — ``p`` is the cached atom-pair rep)."""
        return self.atom_decoder.precompute_biases(p)

    def __call__(self, a, q, c, p, feats, to_keys, multiplicity=1,
                 model_cache=None, atom_biases=None):
        a_to_q = self.a_to_q_trans(a)
        atom_mask = feats.atom_pad_mask
        atom_to_token = (
            jax.nn.one_hot(feats.atom_to_token, a_to_q.shape[-2])
            * feats.atom_pad_mask[..., None].astype(jnp.float32)
        )
        a_to_q = atom_to_token @ a_to_q
        q = q + a_to_q
        q = self.atom_decoder(
            q=q, mask=atom_mask, c=c, p=p, multiplicity=multiplicity,
            to_keys=to_keys, biases=atom_biases,
        )
        return self.atom_feat_to_atom_pos_update(q)


# --- InputEmbedder -----------------------------------------------------------
@register_from_torch("promera.model.trunk.InputEmbedder")
class InputEmbedder(eqx.Module):
    atom_attention_encoder: AtomAttentionEncoder
    mask_std_feats: bool

    @staticmethod
    def from_torch(m: _trunk.InputEmbedder):
        return InputEmbedder(
            atom_attention_encoder=from_torch(m.atom_attention_encoder),
            mask_std_feats=bool(m.cfg.feature.mask_std_feats),
        )

    def __call__(self, feats):
        res_type = feats.restype
        profile = feats.profile
        deletion_mean = feats.deletion_mean[..., None]

        a, _, _, _, _ = self.atom_attention_encoder(feats)
        if self.mask_std_feats:
            a = jnp.where(feats.is_std[..., None], 0.0, a)

        is_epitope = feats.is_epitope.astype(jnp.float32)[..., None]
        # `restype` is normally int token indices [B, N] (prediction). For
        # gradient-based design (mosaic) it may instead arrive as a soft
        # distribution [B, N, NTOKS]; pass that through so it stays
        # differentiable. The integer path is unchanged.
        res_type = res_type if res_type.ndim == 3 else jax.nn.one_hot(res_type, _ntoks)
        return jnp.concatenate([a, res_type, profile, deletion_mean, is_epitope],
                               axis=-1)


# --- MSA stack ---------------------------------------------------------------
@register_from_torch("promera.model.trunk.MSALayer")
class MSALayer(AbstractFromTorch):
    msa_dropout: float
    z_dropout: float
    msa_transition: Transition
    pair_weighted_averaging: PairWeightedAveraging
    tri_mul_out: TriangleMultiplicationOutgoing
    tri_mul_in: TriangleMultiplicationIncoming
    tri_att_start: TriangleAttention
    tri_att_end: TriangleAttention
    z_transition: Transition
    outer_product_mean: OuterProductMean

    def __call__(self, z, m, token_mask, msa_mask, *, deterministic=True, key=None):
        # With ``deterministic`` (the default) every mask is all-ones and ``key``
        # is unused, so the arithmetic is identical to the no-dropout path. When
        # training, each masked residual draws its own key (PyTorch redraws the
        # mask per call to ``get_dropout_mask``).
        keys = (None,) * 5 if deterministic else jax.random.split(key, 5)
        msa_dropout = get_dropout_mask(self.msa_dropout, m, not deterministic,
                                       key=keys[0])
        m = m + msa_dropout * self.pair_weighted_averaging(m, z, token_mask)
        m = m + self.msa_transition(m)
        z = z + self.outer_product_mean(m, msa_mask)
        d = get_dropout_mask(self.z_dropout, z, not deterministic, key=keys[1])
        z = z + d * self.tri_mul_out(z, mask=token_mask)
        d = get_dropout_mask(self.z_dropout, z, not deterministic, key=keys[2])
        z = z + d * self.tri_mul_in(z, mask=token_mask)
        d = get_dropout_mask(self.z_dropout, z, not deterministic, key=keys[3])
        z = z + d * self.tri_att_start(z, mask=token_mask)
        d = get_dropout_mask(self.z_dropout, z, not deterministic, key=keys[4],
                             columnwise=True)
        z = z + d * self.tri_att_end(z, mask=token_mask)
        z = z + self.z_transition(z)
        return z, m


@register_from_torch("promera.model.trunk.MSAModule")
class MSAModule(eqx.Module):
    use_paired_feature: bool
    msa_proj: Linear
    s_proj: Linear
    stacked_parameters: MSALayer
    static: MSALayer

    @staticmethod
    def from_torch(m: _trunk.MSAModule):
        layers = [from_torch(layer) for layer in m.layers]
        _, static = eqx.partition(layers[0], eqx.is_inexact_array)
        stacked = tree.map(
            lambda *v: jnp.stack(v, 0),
            *[eqx.filter(layer, eqx.is_inexact_array) for layer in layers],
        )
        return MSAModule(
            use_paired_feature=bool(m.use_paired_feature),
            msa_proj=from_torch(m.msa_proj),
            s_proj=from_torch(m.s_proj),
            stacked_parameters=stacked,
            static=static,
        )

    @property
    def depth(self):
        # leading axis of any stacked leaf == number of scanned layers
        return tree.leaves(self.stacked_parameters)[0].shape[0]

    def __call__(self, z, emb, feats, *, deterministic=True, key=None):
        # int rows [B, S, N] (prediction) or a soft one-hot [B, S, N, NTOKS]
        # (mosaic design, where MSA row 0 carries the soft binder sequence).
        msa = feats.msa if feats.msa.ndim == 4 else jax.nn.one_hot(feats.msa, _ntoks)
        has_deletion = feats.has_deletion[..., None].astype(jnp.float32)
        deletion_value = feats.deletion_value[..., None]
        is_paired = feats.msa_paired[..., None].astype(jnp.float32)
        msa_mask = feats.msa_mask
        token_mask = feats.token_pad_mask.astype(jnp.float32)
        token_mask = token_mask[:, :, None] * token_mask[:, None, :]

        if self.use_paired_feature:
            m = jnp.concatenate([msa, has_deletion, deletion_value, is_paired], axis=-1)
        else:
            m = jnp.concatenate([msa, has_deletion, deletion_value], axis=-1)

        m = self.msa_proj(m)
        m = m + self.s_proj(emb)[:, None]

        # Per-layer keys (only used when not deterministic; the scan still
        # carries them either way so the traced graph shape is constant).
        layer_keys = (
            jnp.zeros((self.depth, 2), dtype=jnp.uint32)
            if deterministic
            else jax.random.split(key, self.depth)
        )

        @jax.checkpoint
        def body_fn(carry, layer_inputs):
            z, m = carry
            params, lkey = layer_inputs
            layer = eqx.combine(self.static, params)
            z, m = layer(z, m, token_mask, msa_mask,
                         deterministic=deterministic, key=lkey)
            return (z, m), None

        (z, m), _ = jax.lax.scan(
            body_fn, (z, m), (self.stacked_parameters, layer_keys)
        )
        s = (m * msa_mask[..., None]).sum(1) / (msa_mask.sum(1)[..., None] + 1e-5)
        return s, z


# --- Pairformer stack --------------------------------------------------------
@register_from_torch("promera.model.trunk.PairformerLayer")
class PairformerLayer(AbstractFromTorch):
    token_z: int
    dropout: float
    num_heads: int
    no_update_s: bool
    no_update_z: bool
    attention: AttentionPairBias | None
    tri_mul_out: TriangleMultiplicationOutgoing
    tri_mul_in: TriangleMultiplicationIncoming
    tri_att_start: TriangleAttention
    tri_att_end: TriangleAttention
    transition_s: Transition | None
    transition_z: Transition

    def __call__(self, s, z, mask, pair_mask, *, deterministic=True, key=None):
        # Dropout is on the pairwise (z) residuals only; the sequence (s)
        # updates are never dropped (matches the PyTorch PairformerLayer).
        keys = (None,) * 4 if deterministic else jax.random.split(key, 4)
        d = get_dropout_mask(self.dropout, z, not deterministic, key=keys[0])
        z = z + d * self.tri_mul_out(z, mask=pair_mask)
        d = get_dropout_mask(self.dropout, z, not deterministic, key=keys[1])
        z = z + d * self.tri_mul_in(z, mask=pair_mask)
        d = get_dropout_mask(self.dropout, z, not deterministic, key=keys[2])
        z = z + d * self.tri_att_start(z, mask=pair_mask)
        d = get_dropout_mask(self.dropout, z, not deterministic, key=keys[3],
                             columnwise=True)
        z = z + d * self.tri_att_end(z, mask=pair_mask)
        z = z + self.transition_z(z)
        if not self.no_update_s:
            s = s + self.attention(s, z, mask)
            s = s + self.transition_s(s)
        return s, z


@register_from_torch("promera.model.trunk.PairformerModule")
class PairformerModule(eqx.Module):
    stacked_parameters: PairformerLayer
    static: PairformerLayer

    @staticmethod
    def from_torch(m: _trunk.PairformerModule):
        layers = [from_torch(layer) for layer in m.layers]
        _, static = eqx.partition(layers[0], eqx.is_inexact_array)
        stacked = tree.map(
            lambda *v: jnp.stack(v, 0),
            *[eqx.filter(layer, eqx.is_inexact_array) for layer in layers],
        )
        return PairformerModule(stacked, static)

    @property
    def depth(self):
        # leading axis of any stacked leaf == number of scanned layers
        return tree.leaves(self.stacked_parameters)[0].shape[0]

    def __call__(self, s, z, mask, pair_mask, *, deterministic=True, key=None):
        # Per-layer keys (unused when deterministic; carried regardless so the
        # scan's traced shapes do not depend on the toggle).
        layer_keys = (
            jnp.zeros((self.depth, 2), dtype=jnp.uint32)
            if deterministic
            else jax.random.split(key, self.depth)
        )

        @jax.checkpoint
        def body_fn(carry, layer_inputs):
            s, z = carry
            params, lkey = layer_inputs
            layer = eqx.combine(self.static, params)
            s, z = layer(s, z, mask, pair_mask,
                         deterministic=deterministic, key=lkey)
            return (s, z), None

        (s, z), _ = jax.lax.scan(
            body_fn, (s, z), (self.stacked_parameters, layer_keys)
        )
        return s, z


# --- DistogramModule ---------------------------------------------------------
@register_from_torch("promera.model.trunk.DistogramModule")
class DistogramModule(AbstractFromTorch):
    distogram: Linear

    def __call__(self, z: Float[Array, "B N N D"]):
        z = z + jnp.swapaxes(z, 1, 2)
        return self.distogram(z)
