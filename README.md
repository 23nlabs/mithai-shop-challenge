# The Diwali Trap / Mithai Shop Challenge

A reproducible **synthetic simulation** of festival-driven sweet-shop demand, forecasting and production decisions. This is a fictional Ahmedabad retailer: no real shop transactions, actual savings or sealed Yug-1 study forecasts are included.

**23N Labs Research Team · Public release 2026-10-21**

Read the [paper](final/article.md), [technical appendix](final/technical-appendix.md), [six original figures](final/figures/), and [40-claim ledger](final/claims-ledger.json). Reproduce with [the notebooks and code](REPRODUCIBILITY.md). The [data inventory](DATA_INVENTORY.json) distinguishes synthetic-derived results from the attributed third-party factual calendar. [Validation](VALIDATION.md), [operations](OPERATIONS.md), [notices](NOTICE.md), and the [exact file hashes](PACKAGE.sha256) complete the release.

Citation: 23N Labs Research Team (2026), *The Diwali Trap: Can AI Forecasting Models Help a Mithai Shop Plan for a Moving Festival?*, Mithai Shop Challenge public release, https://github.com/23nlabs/mithai-shop-challenge. Venue and DOI: not assigned.

AI tools assisted coding, analysis and manuscript preparation. No independent human review is asserted.

Original code uses Apache-2.0. The paper, original figures and original synthetic data use CC-BY-4.0. The Office Holidays calendar compilation with a Drik Panchang cross-check retains distinct source attribution and is outside those grants; see [NOTICE.md](NOTICE.md). Publisher model weights are not distributed.

The raw synthetic forecast archives (about 367 MB) remain at the pinned private source commit and are omitted here. The included result JSON and paper report the previously verified run; no new model inference was launched for publication. `python3 verify_release.py` and `python3 audit_public.py` check this package locally. This is a static research artifact with no Render or other hosted application deployment.
