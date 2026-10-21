# Reproducibility and provenance

The scientific source is the private research checkout commit `7069f5a78898076b07014a3d913a1424fc02e53f`, branch `research/mithai-shop-challenge`. This public package derives from the merged private research PR #25 at `sanjos30/23n-research-desk` main commit `1dac4eced89155b746005dac0a2afd68bd142fe9`. Public packaging changes are recorded in this repository's manifest. The source checkout was not altered for publication.

## Included evidence

The v1/v2/v3 result JSON, 40-claim ledger, paper, appendix, figures, simulator and analysis code, and three cleaned inference notebooks are included. The historical source manifests document the original source tree and contain entries for archives omitted here; they are provenance records, not manifests of this upload. Use `PACKAGE.sha256` for exact public-payload hashes. The shared model adapter was extracted from the pinned source and reduced to the functions needed by this study. Unrelated research and sealed-evaluation routines were removed. `data/festivals.json` is the pinned source run's calendar compilation; it contains factual dates and source annotations, not copied web pages.

The approximately 367 MB raw `.npz` synthetic forecast arrays are omitted. The published summary JSON and verifier record the prior audit of 42 files, 241,920 rows, zero nonfinite or crossed rows and 18/18 independent metric matches. These are historical verified results; this publication did not rerun model inference. To independently recount the raw rows, regenerate the arrays with the three notebooks and the pinned inputs and weights. No model weights or non-synthetic raw datasets are included.

## Reproduction sequence

Use Python 3.11. The original CPU environment used NumPy 1.26.4, pandas 2.2.3, SciPy 1.11.4 and statsmodels 0.14.4. The notebooks pin their model packages and revisions. Run all commands below from a disposable clone of this repository; generation and analysis scripts write files.

```bash
python3.11 -m venv .venv
.venv/bin/pip install numpy==1.26.4 pandas==2.2.3 scipy==1.11.4 \
  statsmodels==0.14.4 matplotlib nbformat

# v1: local CPU baseline, then execute final/notebooks/1_core_benchmark_kaggle.ipynb
# on a compatible CUDA T4 environment with internet and publisher model access.
.venv/bin/python src/baselines.py
# Copy its /kaggle/working/mithai/ output to runs/kaggle-01/output/mithai/.
.venv/bin/python src/report.py runs/kaggle-01/output/mithai
.venv/bin/python src/extra_tables.py runs/kaggle-01/output/mithai

# v2: execute final/notebooks/2_robustness_v2_kaggle.ipynb, then copy
# /kaggle/working/v2/ to robustness/v2/runs/kaggle-v2-01/output/v2/.
cd robustness/v2
../../.venv/bin/python src/baselines_v2.py
../../.venv/bin/python src/run_v2.py
../../.venv/bin/python src/tables_v2.py
cd ../..

# v3: execute final/notebooks/3_yug_seeds_v3_kaggle.ipynb, then copy
# its v3 output to robustness/v3/runs/kaggle-v3-01/output/v3/.
cd robustness/v3
../../.venv/bin/python src/final_analysis.py
../../.venv/bin/python src/final_addon.py
../../.venv/bin/python src/final_addon_rolling.py
../../.venv/bin/python src/final_verify.py
../../.venv/bin/python src/final_figures.py
../../.venv/bin/python src/claims_check.py
cd ../..
```

The notebooks embed the simulator, calendar, model adapter and inference runner and assert v1/v2 history digests before inference. They download pinned weights from publishers under their separate terms. The reported source used a free Tesla T4 and a local CPU; v1 and v2 ETS baselines took about 25 minutes and 1.5 hours respectively on eight CPU cores. No GPU job was launched in this publication milestone. Expected verification: 241,920 forecast rows, zero nonfinite or crossed rows, 18/18 independent metric matches, and 40 claims with zero failures. The historical manifests also reference omitted source archives, so their checks require regenerated archives and the original source tree layout.

`python3 verify_release.py` checks this published package without model execution. The current claim check is `python3 robustness/v3/src/claims_check.py` on a disposable copy. See `OPERATIONS.md` for diagnostics.

Calendar dates were compiled from Office Holidays country pages, with the 2024 Diwali override cross-checked against Drik Panchang; each record has its source URL and source page hash, and `data/calendar-crosscheck.md` explains the rule. The calendar compilation is outside the Apache-2.0 and CC-BY-4.0 grants for original 23N material; see `NOTICE.md`. This release has institutional attribution only. AI tools assisted coding, analysis and manuscript preparation.
