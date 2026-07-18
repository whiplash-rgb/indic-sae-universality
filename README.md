# Stress-Testing Sparse-Feature Universality and the Matryoshka Hierarchy in Monolingual Indic SLMs

Code and artifacts for the BlackboxNLP 2026 Reproducibility Track submission.
Every number in the paper regenerates from this repository.

## Contents

| Path | Purpose |
|---|---|
| `slm.py`, `config.py` | 54M nanoGPT-style SLM loading; paths and config |
| `extract_real.py` | layer-3 residual extraction (shuffled random-sampling protocol) |
| `sae.py`, `train_batchtopk.py` | BatchTopK Matryoshka SAE (dict 4096, k=32) and training |
| `make_random_model.py` | seeded random-init floor models |
| `concept_probe.py` | equalized concept signatures (`--raw` for the raw-residual ladder) |
| `rsa_match.py`, `rsa_diff.py` | RSA + permutation null; paired-bootstrap floor test |
| `corpus_audit.py`, `compare_runs.py` | sampling-representativeness audit; first-slice vs shuffled comparison |
| `hierarchy_eval.py` | hierarchy diagnostics (RQ1); `--plot` builds Figure 1 |
| `diagnostic_sims.py` | zero-signal simulations (Appendix A) |
| `nnsight_check.py` | NNsight cross-validation of the extraction path |
| `matching.py` | token-profile matching instrument (Appendix A.2) |
| `save_eval_slice.py` | retains eval slices of the activation memmaps |
| `lexicons/` | generation lexicons + curated Hindi mapping |
| `results/` | shuffled-run outputs (signatures, RSA/hierarchy JSON, meta with seeds + checkpoint hashes, logs, Figure 1) |
| `results/firstslice_baseline/` | superseded file-order run (ordering-robustness baseline) |
| `RUNBOOK.txt` | exact end-to-end pipeline with the seed plan |

## Reproducing

SLM checkpoints and corpora are the public Regional-TinyStories releases
(cited in the paper). SAE checkpoints are not stored here; they retrain
end-to-end from the pinned seeds in `RUNBOOK.txt`. Point `CKPT_DIR` at the
checkpoints and follow `RUNBOOK.txt` (Steps 0–6, ~8 GPU-hours on one A40).
The statistics reproduce from `results/` alone in minutes on CPU:

```
export SAE_CKPT_DIR=results
python rsa_diff.py --ta concept_sig_hindi.npz --tb concept_sig_bengali.npz \
       --fa concept_sig_random_hindi.npz --fb concept_sig_random_bengali.npz \
       --label H-B --boots 10000 --bonferroni 3
python diagnostic_sims.py   # Appendix A tables
```

## Seeds and provenance

Shuffle seeds: Hindi 101, Marathi 202, Bengali 303 (shared within a language,
distinct across languages). Floor init seeds: 1–3 (draw A), 4–6 (draw B).
Each `meta_*.json` records the shuffle seed and the SHA-256 of the source
checkpoint.
