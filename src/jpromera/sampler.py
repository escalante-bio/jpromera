"""EDM (churn) diffusion sampler for Promera, translating
promera.diffusion.sampler (Sampler + EDMDiffusionStepper, edm_churn path) and
promera.model.loss.diffusion.weighted_rigid_align.
"""

import einops
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Float


def weighted_rigid_align(true_coords, pred_coords, weights, mask):
    """Align ``true_coords`` onto ``pred_coords`` (Kabsch, weighted)."""
    batch_size, num_points, dim = true_coords.shape
    weights = (mask * weights)[..., None]

    true_centroid = (true_coords * weights).sum(1, keepdims=True) / weights.sum(
        1, keepdims=True
    )
    pred_centroid = (pred_coords * weights).sum(1, keepdims=True) / weights.sum(
        1, keepdims=True
    )
    true_c = true_coords - true_centroid
    pred_c = pred_coords - pred_centroid

    cov = jnp.einsum("b n i, b n j -> b i j", weights * pred_c, true_c)
    U, _, Vh = jnp.linalg.svd(cov, full_matrices=False)
    V = jnp.swapaxes(Vh, -1, -2)

    rot = jnp.einsum("b i j, b k j -> b i k", U, V)
    det = jnp.linalg.det(rot)
    F = jnp.broadcast_to(jnp.eye(dim), (batch_size, dim, dim))
    F = F.at[:, -1, -1].set(det)
    rot = jnp.einsum("b i j, b j k, b l k -> b i l", U, F, V)

    rot = jax.lax.stop_gradient(rot)
    pred_centroid = jax.lax.stop_gradient(pred_centroid)
    aligned = jnp.einsum("b n i, b j i -> b n j", true_c, rot) + pred_centroid
    return aligned


def edm_schedule(cfg, num_steps):
    """Return (sigma_t, sigma_s, gamma) arrays of length num_steps."""
    p = cfg["rho"]
    sigma_max = cfg["sigma_max"] * cfg["sigma_data"]
    sigma_min = cfg["sigma_min"] * cfg["sigma_data"]

    def sched(t):
        return (
            sigma_min ** (1 / p)
            + (1 - t) * (sigma_max ** (1 / p) - sigma_min ** (1 / p))
        ) ** p

    lin = np.linspace(0, 1, num_steps + 1)
    sigma_t = np.array([sched(t) for t in lin[:-1]], dtype=np.float32)
    sigma_s = np.array([sched(s) for s in lin[1:]], dtype=np.float32)
    gamma = np.where(sigma_s > cfg["gamma_min"], cfg["gamma_0"], 0.0).astype(np.float32)
    return sigma_t, sigma_s, gamma


def _center(coords):
    return coords - coords.mean(axis=-2, keepdims=True)


def edm_sample(
    ad,
    conditioning: dict,
    atom_mask: Float[Array, "B M"],
    init_coords: Float[Array, "B M 3"],
    sigma_t,
    sigma_s,
    gamma,
    churn_noise,
    R,
    tr,
    *,
    step_scale: float,
    noise_scale: float,
):
    """Run the EDM-churn sampler, replaying provided per-step random draws.

    churn_noise: [S, B, M, 3]; R: [S, B, 3, 3]; tr: [S, B, 1, 3].
    Returns (final_coords, traj_x0[S], traj_noisy[S]).
    """

    @jax.checkpoint
    def body(coords, draws):
        st, ss, gm, cn, Ri, tri = draws
        t_hat = st * (1 + gm)
        coords = coords + noise_scale * jnp.sqrt(
            jnp.clip(t_hat**2 - st**2, min=0.0)
        ) * cn
        # model_func: center + random augment
        coords = _center(coords)
        coords = jnp.einsum("bmd,bds->bms", coords, Ri) + tri
        sigma = jnp.full((coords.shape[0],), t_hat)
        x0 = ad.preconditioned_network_forward(coords, sigma, conditioning)[
            "denoised_atom_coords"
        ]
        x_aligned = weighted_rigid_align(coords, x0, atom_mask, atom_mask)
        delta = (x0 - x_aligned) / t_hat
        coords_next = x_aligned + step_scale * (t_hat - ss) * delta
        return coords_next, (x0, x_aligned)

    draws = (sigma_t, sigma_s, gamma, churn_noise, R, tr)
    final, (traj, noisy) = jax.lax.scan(body, init_coords, draws)
    return final, traj, noisy
