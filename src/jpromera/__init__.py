"""JAX/Equinox translation of Promera.

``import jpromera`` is torch-free: it exposes the eqx model classes, the EDM
sampler, serialization and the hardcoded config — enough to load a serialized
model and run inference. Converting PyTorch weights additionally requires
``jpromera.convert`` (which imports torch + promera); the convenience loaders
below import it lazily.
"""

from . import cache  # noqa: F401  (enables JAX persistent compile cache)
from .backend import (  # noqa: F401
    AbstractFromTorch,
    Embedding,
    Identity,
    LayerNorm,
    Linear,
    Rearrange,
    Sequential,
    from_torch,
    register_from_torch,
)
from . import layers  # noqa: F401,E402
from . import trunk  # noqa: F401,E402
from . import diffusion  # noqa: F401,E402
from . import sampler  # noqa: F401,E402
from . import confidence  # noqa: F401,E402
from . import model  # noqa: F401,E402
from . import serialize  # noqa: F401,E402
from .model import JPromera  # noqa: F401,E402
from .serialize import load_pretrained, save_model  # noqa: F401,E402
from .trunk import TrunkOutput  # noqa: F401,E402
from .confidence import Confidence, Contact  # noqa: F401,E402
from .feats import Feats  # noqa: F401,E402
from .config import get_config, DIFFUSION  # noqa: F401,E402


# --- conversion loaders (lazily import jpromera.convert -> torch + promera) ---
def _convert():
    """Import the conversion module, with a clear hint if its extra is missing.

    Weight conversion needs the PyTorch source model (`jpromera[convert]`); the
    torch-free core (load a serialized model + featurize + fold/sample) doesn't.
    """
    try:
        from . import convert
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"jpromera weight conversion needs the 'convert' extra (missing: "
            f"{e.name}). Install it with:  pip install 'jpromera[convert]'  "
            f"(pulls the PyTorch promera source). Not needed to run a serialized "
            f"JAX model or to featurize."
        ) from e
    return convert


def load_model(weights=None):
    """Load Promera weights and return the JAX/Equinox ``JPromera`` model."""
    return _convert().load_model(weights)


def load_torch_model(weights=None):
    """Build the PyTorch source PromeraModel with weights loaded."""
    return _convert().load_torch_model(weights)


def default_weights():
    """Path to the Promera checkpoint (``$PROMERA_WEIGHTS`` or HF cache)."""
    return _convert().default_weights()
