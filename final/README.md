# The Diwali Trap / Mithai Shop Challenge

**Public release: 2026-10-21.** By **23N Labs Research Team**. This is a controlled simulation of a fictional Ahmedabad sweet shop. No genuine retailer transactions or sealed Yug-1 field-note forecasts are included. No venue or DOI is assigned.

**Citation:** 23N Labs Research Team (2026), *The Diwali Trap: Can AI Forecasting Models Help a Mithai Shop Plan for a Moving Festival?*, Mithai Shop Challenge public release, https://github.com/23nlabs/mithai-shop-challenge (accessed at the release commit).

AI tools assisted coding, analysis and manuscript preparation. This statement does not imply an independently verified human review.

- [Paper](article.md), [technical appendix](technical-appendix.md), and [six figures](figures/)
- [Claims ledger](claims-ledger.json): 40 source-linked claims
- [Core](notebooks/1_core_benchmark_kaggle.ipynb), [robustness](notebooks/2_robustness_v2_kaggle.ipynb), and [seed](notebooks/3_yug_seeds_v3_kaggle.ipynb) inference notebooks. They are unexecuted templates with pinned model and history hashes.
- [Reproduction](../REPRODUCIBILITY.md), [public validation](../VALIDATION.md), [data inventory](../DATA_INVENTORY.json), [terms](../NOTICE.md), and [operations](../OPERATIONS.md)

The original synthetic forecast archives total about 367 MB and are omitted from this repository. Source commit `7069f5a78898076b07014a3d913a1424fc02e53f` holds them privately. The published JSON summaries and ledger are verified evidence of that prior run, not a new rerun. `final_verify.py` requires the archives to recount forecast rows. Model weights are downloaded from their publishers and are not distributed here.
