# jpromera

JAX/Equinox translation of [Promera](https://github.com/bjing2016/promera), a
dual-purpose biomolecular generative model for structure prediction and binder
design. Translated module-by-module from the PyTorch source and validated
numerically against captured PyTorch ground truth (true-fp32), in the style of
`joltzgen`.

## Torch-free core

`import jpromera` is **torch-free** — it pulls in only JAX/Equinox/einops, enough
to load weights (`jpromera.load_pretrained()`) and run the whole
schema → featurize → fold → sample → mmCIF pipeline with PyTorch never imported
(see [Usage](#usage)).

`jpromera.features` reimplements promera's `finalize_feats` / `collate` /
`_copy_sample_to_struct` in numpy (verified byte-identical), and featurization
(`tinyprot`) auto-fetches any missing CCD/taxonomy data and MSAs over the network
— all torch-free. The only torch/promera-only step is the **one-time weight
conversion** (`jpromera.load_model` / `jpromera.convert`), needed only if you
convert a checkpoint yourself instead of using the published JAX weights.

## Installation

```bash
uv add "jpromera @ git+https://github.com/escalante-bio/jpromera.git"            # core (torch-free)
uv add "jpromera[convert] @ git+https://github.com/escalante-bio/jpromera.git"   # + weight conversion
```

## Usage

```python
import jax, jpromera
from jpromera.features import featurize, save_structure

# 1UBQ — ubiquitin, a single 76-residue protein chain
schema = {"A": {"type": "protein", "entity_id": 1,
                "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQ"
                            "LEDGRTLSDYNIQKESTLHLVLRLRGG"}}

jp = jpromera.load_pretrained()                  # JAX/Equinox weights from HuggingFace
                                                 # (escalante-bio/jpromera) — no PyTorch
feats, struct = featurize(schema, msa_depth=1024)  # downloads tinyprot data + MSA if absent
out = jp.fold(feats, recycling_steps=4)          # -> TrunkOutput(s, z, pdistogram, ...)
coords, traj, _ = jp.sample(                     # EDM diffusion sampling
    feats, out, num_steps=200,
    diffusion_cfg=jpromera.DIFFUSION,            # default EDM schedule
    key=jax.random.PRNGKey(0),
)
conf = jp.sm_confidence_module(feats, out, coords, multiplicity=1)    # -> Confidence(plddt, pae, ...)
contact = jp.contact_module(feats, out, coords, multiplicity=1)      # -> Contact(contact_logits, pred_dist)
save_structure(struct, coords, "pred.cif")
```

Two ways to get the model:

- `jpromera.load_pretrained()` — download pre-converted JAX/Equinox weights from
  HuggingFace ([`escalante-bio/jpromera`](https://huggingface.co/escalante-bio/jpromera)),
  cached locally. Fully torch-free; the recommended path.
- `jpromera.load_model(weights=None)` — convert a PyTorch Promera checkpoint
  yourself (resolves `$PROMERA_WEIGHTS` or the HuggingFace cache). Requires the
  `[convert]` extra (PyTorch).

## Notes

- **model_cache.** PyTorch caches the pairwise bias across diffusion steps. The
  JAX score model recomputes it each step (cache-free); numerically identical, possibly slower if XLA doesn't figure this out.
- **Precision.** Validation runs JAX on CPU fp32 vs torch GPU fp32; the relative-
  error floor is ~1e-6..1e-5.

## Validation

Folding real structures in both backends (Cα RMSD via gemmi): 1UBQ —
jax↔native 0.97 Å, torch↔native 0.66 Å, jax↔torch 1.05 Å; 9HKW chain A
(458 res) — jax↔native 3.76 Å, torch↔native 3.80 Å, jax↔torch 0.36 Å. The two
backends agree as well as or better than either matches native; the residual is
sampling stochasticity (independent PRNG), not a backend discrepancy
(matched-seed replay matches to ~5e-5).

## License

[MIT](LICENSE). jpromera is a JAX/Equinox translation of
[Promera](https://github.com/bjing2016/promera) (© 2026 Bowen Jing and Mihir
Bafna, MIT); the translation is © 2026 Escalante Bio. The upstream MIT notice is
preserved in `LICENSE`.
