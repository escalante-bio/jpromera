"""Loaders that build the PyTorch model and convert it to JAX."""

import os

from ..backend import from_torch
from ..config import get_config


def default_weights() -> str:
    """Path to the Promera checkpoint: ``$PROMERA_WEIGHTS`` or the HF cache."""
    p = os.environ.get("PROMERA_WEIGHTS")
    if p:
        return p
    from huggingface_hub import hf_hub_download

    return hf_hub_download("bjing-mit/promera", "promera_2606.ckpt")


def load_torch_model(weights: str | None = None):
    """Build the PyTorch PromeraModel and load weights (eval mode, CPU)."""
    from promera.model.model import PromeraModel
    from promera.utils.load_weights import load_weights

    model = PromeraModel(get_config())
    load_weights(weights or default_weights(), model)
    return model.eval()


def load_model(weights: str | None = None):
    """Load Promera weights and return the JAX/Equinox ``JPromera`` model."""
    return from_torch(load_torch_model(weights))
