# jpromera

JAX/Equinox translation of [Promera](https://github.com/bjing2016/promera), a
dual-purpose biomolecular generative model for structure prediction and binder
design. 

## Installation

```bash
uv add "jpromera @ git+https://github.com/escalante-bio/jpromera.git"            # core (torch-free)
uv add "jpromera[convert] @ git+https://github.com/escalante-bio/jpromera.git"   # + weight conversion
```

## Usage

```python
import jax, jpromera
from jpromera.features import featurize, save_structure

# 1UBQ — ubiquitin
schema = {"A": {"type": "protein", "entity_id": 1,
                "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQ"
                            "LEDGRTLSDYNIQKESTLHLVLRLRGG"}}

jp = jpromera.load_pretrained()                  # JAX/Equinox weights from HuggingFace
                                                 # (escalante-bio/jpromera) — no PyTorch
feats, struct = featurize(schema)                # downloads tinyprot data + MSA if absent
out = jp.fold(feats, recycling_steps=4,          # -> TrunkOutput(s, z, pdistogram, ...)
              key=jax.random.PRNGKey(0))         # key seeds per-recycle MSA subsampling
coords, traj, _ = jp.sample(                     # EDM diffusion sampling
    feats, out, num_steps=200,
    diffusion_cfg=jpromera.DIFFUSION,            # default EDM schedule
    key=jax.random.PRNGKey(0),
)
conf = jp.sm_confidence_module(feats, out, coords)    # -> Confidence(plddt, pae, ...)
contact = jp.contact_module(feats, out, coords)      # -> Contact(contact_logits, pred_dist)
save_structure(struct, coords, "pred.cif")
```

Two ways to get the model:

- `jpromera.load_pretrained()` — download pre-converted JAX/Equinox weights from
  HuggingFace ([`escalante-bio/jpromera`](https://huggingface.co/escalante-bio/jpromera)),
  cached locally. Fully torch-free; the recommended path.
- `jpromera.load_model(weights=None)` — convert a PyTorch Promera checkpoint
  yourself (resolves `$PROMERA_WEIGHTS` or the HuggingFace cache). Requires the
  `[convert]` extra (PyTorch).


## License

[MIT](LICENSE). jpromera is a JAX/Equinox translation of
[Promera](https://github.com/bjing2016/promera) (© 2026 Bowen Jing and Mihir
Bafna, MIT); the translation is © 2026 Escalante Bio. The upstream MIT notice is
preserved in `LICENSE`.
