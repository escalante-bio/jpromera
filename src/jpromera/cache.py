"""Enable JAX persistent compilation cache.

Import this module (``import jpromera.cache``) immediately after importing jax,
or call ``enable()`` before any compilation happens.
"""

import jax


def enable(path: str = "/jax_cache") -> None:
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


enable()
