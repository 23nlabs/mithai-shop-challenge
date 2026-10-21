# Technical Appendix — The Mithai Shop Challenge (final, v3)

> **⚠️ SYNTHETIC RETAIL DEMAND.** All data come from a simulated Ahmedabad mithai shop (`src/mithai_core.py`).
> No genuine shop transaction records are used. All rupee amounts are simulated contributions under stated
> assumptions, not realised or projected savings.

## A. Study versions and immutability

| Version | Content | Location | Status |
|---|---|---|---|
| v1 | Original design: 11 methods, 2,160 jobs, committed plan | Pinned private source; result summaries here | Historical 148-entry manifest: 143 original bytes match; five superseded documents are preserved in source history. |
| v2 | Robustness: calendar dependence (R1–R3), calibrated shopkeeper, censoring-aware regression, rolling replanning, Yug-1 seeds on the original jobs | `robustness/v2/` and pinned private source | Historical 50-entry manifest verifies at the source; summaries and code are included here. |
| v3 | Final pass: Yug-1 seeds on every v2 experiment, training-only calibration, hierarchical paired tests, independent verification | `robustness/v3/` | This release. |

## B. Pre-specified versus post-hoc (read every result with this table)

| Element | Status |
|---|---|
| Simulator, cut-offs, horizons (28/42/56), methods of v1, WAPE/MAE/wQL/coverage, actual-peak rule, surge window, raw phantom-surge metric, shop-cluster bootstrap, critical-ratio policy, hindsight reference | **Pre-specified** (before any result) |
| Committed-plan policy as primary; "forecast must have a distinct peak" rule | Changed after a **baselines-only** dry run, **before** any foundation-model output |
| Bias-adjusted phantom-surge metric | **Post-hoc** (after v1 results) |
| R1/R2/R3 scenarios; calibrated shopkeeper; censoring-aware regression; rolling replanning; Yug-1 seeds on the original jobs | **Post-hoc** (v2, subsequent review pass) |
| Yug-1 seeds on all v2 experiments; training-only bias correction and decision calibration; hierarchical paired tests | **Post-hoc** (v3, subsequent review pass) |

The headline conclusions in the article rest mainly on post-hoc robustness analyses. They are framed as
robustness findings, not confirmatory tests.

## C. Data, models and compute

**Simulator.** 2019–2026, daily, in kg.
- Latent demand and censored observed sales (capped at historical capacity).
- Corporate kaju katli orders booked 7–25 days ahead.
- Wedding seasons and bulk orders.
- Five festivals, with Diwali dates cross-checked against drikpanchang.

**Scenarios.**
- **Base:** 30 shops.
- **Stress:** early gifting, weak festival, +50% local shock; 2026 only; 20 shops each.
- **R1:** yearly-varying festival shape (lead ×e^N(0,0.35), exponent 0.8–3.5, uplift ×e^N(0,0.35), shift −6…+2 days).
- **R2:** ~3 festival-independent shocks per year (3–21 days, ×0.55–1.8).
- **R3:** both. R1–R3 use 20 shops each.

**Models (pinned and hash-verified).**

| Model | Revision |
|---|---|
| Yug-1 | `birlaailabs/yug` @ `ec53758`; package v0.1.1 @ `2ad7da71` |
| TimesFM 2.5 | @ `1d952420`; timesfm 3.0.2 @ `e51928e2` |
| Chronos-Bolt base | @ `5d9f166d`; chronos-forecasting 1.5.3 @ `fcd09fe8` |

All three run zero-shot, horizon 56, context ≤ 2,048 days, float32, on a Kaggle Tesla T4. Yug-1 uses its
default 100 sample paths with seeds 67067, 1, 2, 3 and 4.

**Forecast inventory** (raw `.npz`, counted directly by `robustness/v3/src/final_verify.py`): 42 files and
**241,920** forecast rows, with 0 non-finite and 0 crossed-quantile rows.

| Source | Forecasts |
|---|---|
| v1 foundation models | 12,960 |
| v1 local methods | 10,800 |
| v2 local methods on original jobs | 4,320 |
| v2 local methods on v2 jobs | 65,520 |
| v2 foundation models on v2 jobs | 56,160 |
| v2 Yug-1 seeds on original jobs | 17,280 |
| v3 Yug-1 seeds on v2 jobs | 74,880 |

**History identity.** Kaggle regenerated every input history, and it hash-matched the local build (v1
`d63e862b…`, v2 `757e8761…`). The numpy versions differed: 1.26.4 locally, 2.1.3 on Kaggle.

## D. Metrics and decision policy

**Accuracy** (median forecast against latent demand):
- WAPE: pooled.
- 80% coverage: q0.1–q0.9.
- wQL.
- Peak timing: 7-day sums. Scored only when the actual peak is distinct and the forecast has a distinct
  peak; the distinct-peak share is always reported alongside.
- Surge error.
- Raw and bias-adjusted phantom surge.

**Decisions.**
- Target(t) = max(forecast quantile at the critical ratio τ = margin/price, confirmed orders).
- Capacity 1.4× historical, with pre-build.
- FIFO stock, shelf lives of 10, 5 and 4 days.
- Committed 56-day plan, or weekly rolling replanning: 7-day supplier lead time, ingredients 70% of
  cost, ingredient shelf lives of 28, 14 and 7 days, MRP netting.
- Regret = hindsight reference − policy. The reference is verified feasible under identical constraints;
  it loses 0.2–0.3% of demand, to capacity at the peaks.

**Training-only calibration (v3).** Computed per method, per shop and product, from forecast windows that
ended before the cut-off:
- *bias*: b = Σ observed sales ÷ Σ median forecast.
- *bias + decision calibration*: k = empirical τ-quantile of observed ÷ forecast quantile at τ.

Notes on the calibration:
- The original ×1.10 shopkeeper is calibrated from his point forecast without the 1.10 margin.
- It learns from censored *observed* sales, which is information a real shop has.
- If no past window exists, the forecast is unchanged; this affects 4% of jobs.

**Uncertainty.** 95% intervals from 2,000 bootstrap draws. Each draw resamples simulated shops (clusters
holding all of a shop's cut-offs and products) and, for Yug-1, the five sampling seeds. A paired difference
whose interval contains 0 is reported as **inconclusive**. We make no significance claims beyond these
intervals.

## E. Results (all numbers from `robustness/v3/results/final-v3.json`, `final-v3-addon.json` and `robustness/v2/results/robustness-v2.json`)

### E1. Diwali trap, +19-day years (v2 P1)

| Method | Distinct-peak share | Median peak error (days) | Bias-adjusted phantom surge |
|---|---|---|---|
| Yug-1 | 48% | −17 | +29% (5 seeds: +28 to +29%) |
| TimesFM 2.5 | 33% | −16 | +15% |
| Chronos-Bolt | 23% | −19 | +17% |
| Seasonal-naive | 100% | −20 | +45% |
| ETS | 0% | — | +10% |
| Calendar-informed methods | 88–96% | −1 to 0 | +4 to +11% |

### E2. Calendar gain, WAPE points, paired (v2 P2)

| Scenario | Yug-1 | TimesFM 2.5 | Chronos-Bolt | ETS |
|---|---|---|---|---|
| Base | +10.3 [9.6, 11.2] | +6.2 [5.7, 6.8] | +8.0 [7.4, 8.6] | +10.4 |
| R1 | +5.6 [4.8, 6.5] | +2.9 [2.1, 3.8] | +4.2 [3.2, 5.3] | +6.7 |
| R2 | +9.4 [8.5, 10.3] | +6.0 [5.4, 6.7] | +7.3 [6.7, 8.0] | +9.6 |
| R3 | +4.7 [3.7, 5.8] | +2.6 [1.9, 3.5] | +3.7 [2.6, 4.8] | +5.9 |

### E3. Festival-season regret, ₹ per 56-day product plan, original base jobs (1,080 plans, 30 shops; Yug-1 averaged over 5 seeds)

| Method | Uncalibrated | Bias-corrected | Bias + decision-calibrated |
|---|---|---|---|
| Calendar regression (censoring-aware) | 63,812 [59,497, 68,347] | **47,602** [44,358, 50,949] | 65,797 |
| Calendar regression | 68,886 | 50,855 | 70,975 |
| TimesFM + calendar | 73,043 | 52,847 | 69,768 |
| Yug-1 + calendar | 66,275 [59,251, 73,381] | 52,882 [48,565, 57,620] | 69,791 |
| Chronos + calendar | 78,896 | 54,102 | 72,541 |
| Shopkeeper (×1.10 rule) | **50,880** | 60,564 | 71,086 |
| ETS + calendar | 72,386 | 64,709 | 75,787 |
| Shopkeeper (calibrated) | 71,870 | 65,250 | 71,154 |
| TimesFM 2.5 | 104,819 | 82,656 | 99,708 |
| Yug-1 | 91,265 [83,387, 99,134] | 87,051 | 96,830 |
| Chronos-Bolt | 118,008 | 91,129 | 108,924 |
| Seasonal-naive 52w | 109,445 | 111,298 | 109,853 |
| ETS | 133,411 | 135,663 | 136,939 |

Hindsight reference contribution: ₹327,078 per plan.

**Reading the calibration columns.**
- Bias correction helps every model-based method.
- The extra decision calibration *hurts* most methods. Calibrating to censored past sales re-learns the
  under-production that censoring causes. This is a substantive finding about censored data, not a defect
  of any model.

### E4. Paired differences after bias correction (hierarchical bootstrap; < 0 favours the first method)

| Comparison | Original | R1 | R2 | R3 | Weekly replanning |
|---|---|---|---|---|---|
| Yug-1 + cal − TimesFM + cal | +35 [−1,914, 1,603] *inconclusive* | +77 *inconcl.* | −370 *inconcl.* | −515 *inconcl.* | −202 *inconcl.* |
| Yug-1 + cal − Chronos + cal | −1,220 *inconcl.* | −860 *inconcl.* | −467 *inconcl.* | −848 *inconcl.* | −780 *inconcl.* |
| Censoring-aware regr. − Yug-1 + cal | −5,280 [−7,302, −3,208] | −4,753 | −4,715 | −3,918 | −4,530 |
| Yug-1 + cal − ETS + cal | −11,828 | −13,199 | −14,688 | −15,989 | −4,056 |
| Yug-1 + cal − calibrated shopkeeper | −12,368 | −18,392 | −14,709 | −19,159 | −13,290 |

In every family, after bias correction, censoring-aware calendar regression has significantly lower
regret than every other calendar-informed method. The three calendar-corrected foundation models are
mutually inconclusive.

### E5. Does Yug-1's history-only business advantage survive calibration? (original festival season)

| Comparison | Uncalibrated | Bias-corrected | Bias + decision-calibrated |
|---|---|---|---|
| Yug-1 − TimesFM 2.5 | −13,554 (favours Yug-1) | **+4,395 (favours TimesFM)** | −2,877 |
| Yug-1 − Chronos-Bolt | −26,743 (favours Yug-1) | −4,078 (favours Yug-1) | −12,094 |

The weekly-replanning family behaves similarly. Uncalibrated, Yug-1 − TimesFM is −10,472 [−14,870,
−6,690]. With bias plus decision calibration it is +1,484, inconclusive.

**Conclusion.** Yug-1's apparent advantage over TimesFM 2.5 was mainly a level-bias effect. It reverses
under training-only bias correction. A smaller advantage over Chronos-Bolt remains.

### E6. Yug-1 seed sensitivity

- Festival 80% coverage: 68–80% across seeds. Festival WAPE: 41.1–42.6%.
- Seed 67067, used in v1 and v2, was the most favourable seed in 15 of 16 business comparisons (fig F6).
- Plausible mechanism: the v0.1.1 `predict` call takes one seed, so every series in a run reuses the same
  uniform draws. Monte Carlo error is then correlated across the benchmark instead of averaging out.

## F. Verification performed (v3)

- Every v1 and v2 manifest entry verifies. The only differences are the five deliberately superseded
  documents, and each of their v1 versions is preserved.
- WAPE and coverage were recomputed from raw forecasts and regenerated truth, without the evaluation
  code: 18/18 checks match the published values.
- v2 recomputed every v1 headline point estimate exactly.
- An independently written planning simulator, with constraint assertions, matches the business code.
- Forecasts at the day-0 origin are identical between v1 and v2.
- A rolling-planner bug was found in v2 (perfect hindsight lost 5.5% of demand). It was fixed before any
  v2 result was used.
- Every number quoted in `article.md` and this appendix is checked against the JSON artifacts by
  `robustness/v3/src/claims_check.py`, which writes `final/claims-ledger.json`.

## G. Limitations

1. **Synthetic data.** Effect sizes are assumptions, and festival effects stay roughly calendar-anchored
   even in R1.
2. **Censoring dominates the economics.** Calibration learns from censored sales.
3. **Decision policy.** One policy family; no tuning of τ or inventory rules; one rolling-planner design.
4. **Models.** Foundation models run zero-shot with defaults. TimesFM ran in eager mode. Native covariate
   interfaces (TimesFM XReg, Chronos-2) were not tested.
5. **Post-hoc.** Most robustness conclusions are post-hoc (section B).
6. **Statistics.** Bootstrap intervals describe variation across simulated shops and model seeds, not
   real-world uncertainty.
