# jpromera

JAX/Equinox translation of [Promera](https://github.com/bjing2016/promera), a
dual-purpose biomolecular generative model for structure prediction and binder
design. Translated module-by-module from the PyTorch source and validated
numerically against captured PyTorch ground truth (true-fp32), in the style of
`joltzgen`.

## Torch-free core

`import jpromera` is **torch-free** — it pulls in only JAX/Equinox/einops, enough
to load a serialized model and run inference/sampling/design with no PyTorch:

```python
import jax, jpromera
from jpromera.features import featurize, save_structure

feats, struct = featurize(schema, msa_depth=1024)      # tinyprot (torch-free) + numpy
jp = jpromera.serialize.load_model("model")            # saved .eqx + .skeleton.pkl
out = jp.fold(feats, recycling_steps=4)
coords, *_ = jp.sample(feats, out, num_steps=200,
                       diffusion_cfg=jpromera.DIFFUSION, key=jax.random.PRNGKey(0))
save_structure(struct, coords, "pred.cif")             # tinyprot mmCIF writer
# the whole schema -> .cif pipeline runs with PyTorch never imported
```

`jpromera.features` reimplements promera's `finalize_feats` / `collate` /
`_copy_sample_to_struct` in numpy (verified byte-identical), so featurization and
structure output stay torch-free. The remaining torch/promera-only steps are the
**one-time weight conversion** (`jpromera.load_model` / `jpromera.convert`) and
**MSA fetching** (`tinyprot.mmseqs2`, a network step).

Converting PyTorch weights lives in `jpromera.convert` (imports torch + promera);
`jpromera.load_model()` imports it lazily. The core declares each torch→eqx
mapping by *dotted-path string* (`backend.REGISTRY`); `convert` resolves the paths
and wires them into `from_torch`. So:

```python
jp = jpromera.load_model()          # convert promera weights -> JAX (needs torch once)
jpromera.serialize.save_model(jp, "model")   # then ship "model.*" for torch-free use
```

## Layout

- `backend.py` — **(core, torch-free)** `from_torch` dispatcher,
  `AbstractFromTorch`, the string-path `REGISTRY`, and vanilla layer classes
  (Linear, LayerNorm, Embedding, Sequential, Identity, Rearrange, activations).
- `convert/` — **(torch side)** resolves `REGISTRY` + base-type/activation
  registrations + `load_model` / `load_torch_model`.
- `layers.py` — leaf / small composites: Transition, PairWeightedAveraging,
  OuterProductMean, TriangleMultiplication{Out,In}going (cueq kernel → plain
  JAX), TriangleAttention + Attention, AttentionPairBias, AdaLN,
  ConditionedTransitionBlock, FourierEmbedding, RelativePositionEncoder.
- `trunk.py` — windowed AtomAttentionEncoder/Decoder, AtomTransformer,
  DiffusionTransformer (scanned), InputEmbedder, MSA stack, Pairformer stack,
  DistogramModule.
- `diffusion.py` — SingleConditioning, PairwiseConditioning, DiffusionModule
  (score model), AtomDiffusion (EDM preconditioning).
- `sampler.py` — `weighted_rigid_align` + the EDM-churn sampler (`edm_sample`).
- `confidence.py` — ConfidenceModule / ContactModule + heads.
- `model.py` — top-level `JPromera` (embed → recycled trunk → distogram →
  diffusion sample → confidence/contact).
- `serialize.py` — torch-free `save_model` / `load_model`.

## Installation

```bash
pip install git+https://github.com/escalante-bio/jpromera.git          # core (torch-free)
pip install 'jpromera[convert] @ git+https://github.com/escalante-bio/jpromera.git'   # + weight conversion
```

The **core** install is torch-free — JAX/Equinox + `tinyprot` (featurization) +
`gemmi`. It can load a serialized JAX model, featurize, fold, sample, and write
mmCIF with PyTorch never installed. The **`[convert]`** extra adds the PyTorch
source model (`promera`, from GitHub) for the one-time checkpoint conversion
(`jpromera.load_model`); it is not needed to run a converted model.

**Setup.** Most of upstream Promera's manual setup does **not** apply here — no
LigandMPNN (inverse folding is out of scope), no `promera` CLI/config, and MSAs
are fetched automatically (`featurize` builds + caches them via the ColabFold
server on a miss). The only remaining one-time step is `tinyprot`'s data init,
and only if you featurize from a schema:

```bash
python -m tinyprot.init --download    # CCD conformers + taxonomy LMDB (~9.5 GB)
```

JAX runs on GPU (`jax-cuda12-*`). The one-time XLA compile of the big scans
(~30–100 s) is cached in `/jax_cache`; warm `fold` is ~0.1 s and `sample`
~8 ms/diffusion step (≈250–290× speedup over cold).

## Usage

```python
import jax, jpromera
from jpromera.features import featurize, save_structure

jp = jpromera.load_pretrained()                  # JAX/Equinox weights from HuggingFace
                                                 # (escalante-bio/jpromera) — no PyTorch
feats, struct = featurize(schema, msa_depth=1024)
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

- **cueq kernels.** `triangular_mult` uses fused `cuequivariance_torch` kernels
  in PyTorch; here they are translated to the standard (numerically equivalent)
  JAX implementation and validated (rel err ~1e-5 vs the kernel).
- **model_cache.** PyTorch caches the pairwise bias across diffusion steps. The
  JAX score model recomputes it each step (cache-free); numerically identical.
- **Precision.** Validation runs JAX on CPU fp32 vs torch GPU fp32; the relative-
  error floor is ~1e-6..1e-5.

## Validation

Each translated module was checked against captured per-module PyTorch I/O
(true fp32). Representative relative errors: leaf layers 1e-6–1e-7; full trunk
`fold` ~6e-7; full score model 1.6e-6; 20-step sampler replay 4.6e-5;
confidence 3e-6; contact logits 3e-7. End-to-end the JAX pipeline folds
ubiquitin (Rg 11.6 Å, mean pLDDT 0.81 at 20 steps).

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
