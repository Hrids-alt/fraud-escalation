# FraudFin: Development Notes

## Day 1: Dataset inspection and loading

**Work completed**
- Implemented `src/load_data.py` to load the Elliptic transaction features, classes and edge list.
- Mapped raw class 1 to illicit (1), raw class 2 to licit (0), and unknown to -1.
- Checked transaction IDs, feature values, edge endpoints and time-step structure.
- Added tests for data loading and label mapping.

**Dataset observations**
- 203,769 transactions and 234,355 directed edges.
- 93 local features and 72 aggregated graph features. This split comes from the dataset paper (Weber et al., 2019); it cannot be verified from the features file itself, which has no header.
- 4,545 illicit, 42,019 licit and 157,205 unlabelled transactions.
- The dataset covers time steps 1-49.

**Data handling**
- Raw dataset files are kept out of Git.
- Unlabelled transactions are retained by the loader but excluded from supervised evaluation.
- `time_step`, transaction IDs and labels must never be used as model features.

## Day 2: Chronological splits and evaluation utilities

**Work completed**
- Implemented `src/splits.py` to create chronological, labelled splits.
- Implemented `src/evaluate.py` with illicit-class metrics, per-step metrics, escalation-rate calculations, recall retention and relative-cost calculations.
- Added tests for chronological split integrity and hand-calculated evaluation metrics.

**Split definitions**
- `train_stage1`: time steps 1-28.
- `calib`: time steps 29-34.
- `train_final`: time steps 1-34.
- `test`: time steps 35-49.

**Verified split counts**
- `train_stage1`: 25,207 labelled transactions; 2,542 illicit.
- `calib`: 4,687 labelled transactions; 920 illicit.
- `train_final`: 29,894 labelled transactions; 3,462 illicit.
- `test`: 16,670 labelled transactions; 1,083 illicit.

**Important observations**
- Illicit prevalence differs across periods: approximately 10.1% in `train_stage1`, 19.6% in `calib`, 11.6% in `train_final` and 6.5% in `test`.
- Calibration and test therefore have different class prevalences. This must be reported when interpreting later cascade results.
- The test period (steps 35-49) must not be used for training, calibration, threshold selection or model selection.
- Undefined metrics should be represented as NaN where appropriate; plain accuracy must not be the headline result.

**Tests**
- Data loader tests: 13 passed.
- Split tests: 12 passed.
- Evaluation metric tests: 12 passed.

## Days 3-4: Standalone model baselines

**Work completed**
- Implemented `src/models.py` with Logistic Regression and Random Forest.
- Implemented `run_baselines.py` for standalone RQ1 comparisons.
- Added `tests/test_models.py`.
- Compared local-only features (93 columns) against all features (165 columns).
- Trained on labelled transactions from time steps 1-34 and evaluated on steps 35-49.
- Used `class_weight="balanced"` for both model families. Logistic Regression uses feature scaling.

**Initial observations**
- All features improved F1 and PR-AUC for both model families in this experiment.
- Random Forest achieved stronger overall baseline metrics than Logistic Regression.
- The everything-licit baseline achieved 93.5% accuracy but zero illicit recall, demonstrating why accuracy is unsuitable as the headline metric.
- Shuffled-label PR-AUC was 0.072 for Logistic Regression and 0.087 for Random Forest, compared with test illicit prevalence of approximately 0.065. These are diagnostic results and do not independently prove the absence of leakage.

**Generated outputs**
- `results/baselines_local.csv`
- `results/baselines_per_step_local.csv`
- `results/baselines_all.csv`
- `results/baselines_per_step_all.csv`

### E1: Local vs All feature comparison (RQ1)

Models trained on labelled transactions from time steps 1-34 and evaluated on steps 35-49. Logistic Regression uses seed 42; Random Forest reports the mean over seeds 42, 43 and 44. The decision threshold is 0.5.

| Model | Feature set | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | Local (93 features) | 0.141 | 0.861 | 0.242 | 0.252 |
| Logistic Regression | All (165 features) | 0.183 | 0.879 | 0.303 | 0.291 |
| Random Forest | Local (93 features) | 0.926 | 0.701 | 0.798 | 0.785 |
| Random Forest | All (165 features) | 0.988 | 0.696 | 0.816 | 0.796 |

**Interpretation.** Adding aggregated features improved F1 and PR-AUC for both model families in this experiment. Random Forest outperformed Logistic Regression on F1 and PR-AUC for both feature sets. However, Random Forest recall decreased slightly when aggregated features were added, from 0.701 to 0.696. These findings describe this test set and do not establish that aggregated features will always improve performance.

**Experiment setup**
- Compared Logistic Regression (LR) and Random Forest (RF) using LOCAL and ALL feature sets.
- LOCAL contains 93 local features. ALL contains 165 features: local and aggregated.
- Excluded `txId`, `time_step` and `label` from model features.
- Trained standalone RQ1 models on labelled transactions from time steps 1-34 and evaluated on 35-49.
- LR used seed 42; RF results were averaged across seeds 42, 43 and 44.
- Used `class_weight="balanced"` for both model families. LR also used feature scaling.

**Random Forest: short theory.** Random Forest is an ensemble learning algorithm that combines many decision trees. Each tree is trained using randomized samples of the training data and considers randomized subsets of features when choosing splits. Their outputs are aggregated to produce the final prediction or probability.

Advantages:
- Learns nonlinear relationships and feature interactions.
- Combines many trees, often reducing the instability of an individual tree.
- Can model complex patterns without requiring feature scaling.

In this project, `class_weight="balanced"` helps account for the imbalance between illicit and licit transactions. It does not guarantee high illicit recall or eliminate false positives.

**Interpretation of evaluation metrics**
- Precision: proportion of predicted illicit transactions that are actually illicit.
- Recall: proportion of actual illicit transactions detected.
- F1: harmonic mean of precision and recall.
- PR-AUC: average precision, summarizing precision-recall ranking performance across thresholds.

Accuracy is not used as the headline metric because a model can predict nearly everything as licit and still achieve high accuracy while detecting no illicit transactions.

## Day 5: E2, per-time-step drift analysis

**Setup.** Models trained on labelled transactions from time steps 1-34 and evaluated on test steps 35-49 using a fixed threshold of 0.5. Random Forest results are averaged over seeds 42-44; Logistic Regression uses seed 42. No test-set tuning was performed.

**Results.** Adding aggregated graph features improved pooled F1 and PR-AUC for both model families. Logistic Regression F1 increased from 0.242 (LOCAL) to 0.303 (ALL), while Random Forest F1 increased from 0.798 to 0.816. Random Forest with ALL features achieved the strongest overall result: precision 0.988, recall 0.696, F1 0.816 and PR-AUC 0.796.

**Temporal variation.** For Random Forest, per-step F1 is high on steps 35-42 for both feature sets (mean 0.88) and falls to near zero from step 43 onward (0.07 LOCAL, 0.09 ALL). Illicit recall is 0.83 (LOCAL) and 0.82 (ALL) on steps 35-42 and 0.02 on steps 43-49 (about 3 of 169 illicit transactions). ALL had higher per-step F1 in 8 of 15 steps (3 lower, 4 tied at zero); the mean per-step gain of +0.012 falls to +0.004 when steps 45-46 (fewer than 10 illicit rows) are excluded, so the pooled RF gain comes mainly from higher precision. For Logistic Regression, ALL was higher in 13 of 15 steps, with the largest gains in steps 47-49. No cause for these changes is established.

**Limitations.** Test steps 45 and 46 contain fewer than 10 illicit examples each, making per-step metrics less stable. The observed changes describe temporal performance variation; they do not establish its cause.

**Artifacts:** `results/e2_per_step.csv`, `results/e2_overall.csv`, `results/figures/e2_drift_rf.png` and `results/figures/e2_drift_lr.png`.

## Day 6: Stage 1 probability calibration

Stage 1 was trained using LOCAL features from labelled transactions in time steps 1-28. A sigmoid (Platt) calibrator was fitted using labelled transactions from steps 29-34. The test window, steps 35-49, was not used.

**Why calibration matters.** A classifier's probability scores are not necessarily reliable estimates of the actual probability of illicit activity. Calibration adjusts these scores so that predicted probabilities better align with observed frequencies. This matters for the cascade because its routing decision depends on probability thresholds: poorly calibrated scores can send too many or too few transactions to the more expensive graph-feature Stage 2 model.

**Results.** Random Forest Brier score improved from 0.0180 to 0.0149, while Logistic Regression improved from 0.1571 to 0.1086 on the calibration set. Both models also improved in the two split-half Brier checks. PR-AUC remained effectively unchanged, consistent with sigmoid calibration preserving prediction ranking.

The calibration set's illicit share was 19.63%, versus 10.08% in Stage 1 training. This prevalence difference is important when interpreting the fitted calibrators: calibration performance on steps 29-34 does not guarantee equivalent probability reliability on later time steps.

The standalone illicit-F1 thresholds selected on the calibration set were `d = 0.39` for Random Forest and `d = 0.50` for Logistic Regression. These thresholds are reserved for the standalone Stage 1 baseline. The cascade's routing threshold, `t_clear`, is selected separately to satisfy the recall-retention constraint.

Reliability plots were generated for both model families. The post-calibration reliability plots use the same steps used to fit the calibrators, so they are in-sample diagnostics; the split-half checks provide additional, though limited, evidence about calibration beyond the fitting subset.

**Saved outputs:** `results/e3_calibration.csv`, `results/e3_reliability.csv`, `results/figures/e3_reliability_rf.png` and `results/figures/e3_reliability_lr.png`. Frozen model artifacts are stored locally under `artifacts/` and must not be committed.

**Calibration notes (my words)**
- Calibration turns model scores into probabilities that match reality. If it says 0.3, about 30% of those cases should be illicit.
- The router compares `p1` to `t_clear`, so `p1` has to mean something.
- The Brier score went from 0.0180 to 0.0149 for RF and from 0.1571 to 0.1086 for LR.
- The "after" plots are in-sample (fitted on steps 29-34), so treat them as diagnostics only.
- LR is still poorly calibrated after sigmoid, so RF is the router.

**Base-rate / distribution-shift note.** The illicit share differs across the chronological windows: 10.08% in Stage 1 training (steps 1-28), 19.63% in calibration (steps 29-34) and approximately 6.50% in the final test window (steps 35-49). The calibration period is therefore not representative of the test period's class prevalence. Threshold selection and calibration results should be interpreted with this distribution shift in mind. We must report final performance on the untouched test window and avoid claiming that validation performance guarantees test performance.

## Day 7: Cascade thresholds (chosen on steps 29-34 only)

The Stage 2 decision threshold `d2` and the routing threshold `t_clear` were chosen on calibration steps 29-34 using a validation copy of Stage 2 trained on steps 1-28 (ALL features, same Random Forest family). `d` was already frozen on Day 6.

- `d = 0.39` (Stage 1, max illicit F1 on steps 29-34).
- `d2 = 0.26` (Stage 2, max illicit F1 on steps 29-34 from the validation copy).
- `t_clear = 0.39`, the largest value allowed (`t_clear <= d`) that keeps recall retention at or above 0.95. It sits at the upper bound `d`, so the sweep found no gap between the two thresholds.
- On the validation steps this escalated 19.5% of transactions and retained 99.4% of graph-always recall.
- All three values are frozen in `thresholds_rf.json`. The sweep is in `results/e4_validation_sweep.csv`.

These validation numbers come before the temporal-drift collapse after step 42 and must not be quoted as test performance.

## Day 8: RQ2 test result (steps 35-49, evaluated once)

**Setup.** Stage 1 (local features, steps 1-28) and its sigmoid calibrator were frozen on Day 6. `d = 0.39`, `d2 = 0.26` and `t_clear = 0.39` were frozen on Day 7 using steps 29-34 and a validation copy of Stage 2 (steps 1-28). Day 8 trains the final Stage 2 (ALL features, same RF family, same seed) on steps 1-34 and scores steps 35-49 once. No threshold was changed after seeing test data.

| System | Precision | Recall | F1 | Recall retention |
|---|---|---|---|---|
| Local-only (`p1 >= d`) | 0.701 | 0.732 | 0.716 | 0.997 |
| Graph-always (`p2 >= d2`) | 0.792 | 0.734 | 0.762 | 1.000 |
| Cascade (frozen `t_clear`) | 0.894 | 0.720 | 0.798 | 0.981 |
| Random escalation, same rate (mean of 5 seeds) | 0.704 | 0.733 | 0.718 | 0.998 |

Retention = recall(system) / recall(graph-always).

**Answer to RQ2.** The cascade keeps 98.1% of graph-always recall (target 95%) while sending only 6.8% of transactions to Stage 2. Under the simulated cost model (local = 1, graph = r, cascade = 1 + escalation_rate x r), the saving against graph-always is 77.7% at r = 5, 88.8% at r = 20 and 92.3% at r = 100. These costs are simulated, because the graph features are precomputed in this dataset.

**Targeted versus random escalation.** At the same escalation rate, random escalation behaves like local-only (F1 0.718), while the cascade reaches F1 0.798 (precision 0.894). So the gain comes from choosing which transactions to escalate, not from escalating some of them.

**Base-rate shift.** The test escalation rate (6.8%) is far lower than on validation (19.5%). This is consistent with the illicit share falling from 19.6% (steps 29-34) to 6.5% (steps 35-49). It is reported as observed, not corrected.

**Cost-versus-recall curve (descriptive, not used to choose anything).** The cascade curve sweeps `t_clear` from 0 to `d` on the test steps. It reaches graph-always recall at roughly 12% escalation and stays flat after that. The frozen `t_clear = 0.39` is the cheapest point on the curve, with recall 0.720 against 0.734 for graph-always. The y-axis spans only about 0.720-0.734, so all recall differences here are about 1.4 percentage points or less. The random-escalation curve is almost flat because local-only recall (0.732) is already close to graph-always recall.

**Per-step behaviour.** On steps 35-42 the cascade F1 is above graph-always on 5 of 8 steps (largest gap +0.13 on step 36) and marginally below on 3 (by at most 0.016). From step 43 onward every system collapses to F1 near 0, so this is a temporal-drift failure that the cascade does not fix (see the Day 5 drift analysis). The largest gap in the late steps is on step 46 (+0.15), but those steps have very few illicit examples, so it is a small-count effect and should not be over-read.

### Limitations (report these)
1. **Recall retention is a weak test on this split.** Local-only recall is already almost equal to graph-always recall, and the random control also retains about 99.8%. On test, the graph features add precision, not recall.
2. **The F1 advantage over graph-always has a specific cause.** `d2 = 0.26` is a low threshold, so graph-always flags many false positives. The cascade removes those in the cleared region. This is not evidence that the cascade is a better model than Stage 2.
3. **`d2` did not transfer well.** An earlier all-features baseline at threshold 0.5 reached precision 0.988, recall 0.696 and F1 0.816 on the same test steps. Both graph-always (F1 0.762) and the cascade (F1 0.798) are below that. This is consistent with thresholds chosen on a 19.6% illicit share not carrying over to a 6.5% share. `d2` was not retuned, because that would use test data.
4. The final Stage 2 is trained on steps 1-34, but `d2` was chosen from a validation copy trained on steps 1-28. The two models differ slightly.
5. One chronological split and one model family (RF). Stage 2 seed spread is reported on Day 9.
6. After step 43 the models fail for all systems, so overall test metrics are dominated by steps 35-42.

## Day 9: Cost table and seed spread

**Simulated cost (Day 8 table).** With local cost 1, graph cost r, and the cascade escalating 6.8% of transactions, the saving against graph-always is 77.7% at r = 5, 88.8% at r = 20 and 92.3% at r = 100. These are simulated, because the graph features are precomputed in this dataset.

**Seed spread (E6).** To check how much the Day 8 result depends on the random seed, the final Stage 2 (Random Forest, all features, steps 1-34) was retrained with seeds 42, 43 and 44 and scored on steps 35-49 with all thresholds unchanged (`d = 0.39`, `d2 = 0.26`, `t_clear = 0.39`). Stage 1 and its calibrator are frozen, so only Stage 2 varies. This was a descriptive measurement; no threshold or model was chosen from it, and all three seeds are reported.

| System | F1 | Recall | Precision | Recall retention |
|---|---|---|---|---|
| Local-only | 0.716 | 0.732 | 0.701 | 0.997 +/- 0.002 |
| Graph-always | 0.763 +/- 0.001 | 0.734 +/- 0.001 | 0.793 +/- 0.001 | 1.000 |
| Cascade | 0.799 +/- 0.001 | 0.721 +/- 0.002 | 0.896 +/- 0.002 | 0.982 +/- 0.001 |
| Random escalation (mean of 5 seeds) | 0.718 +/- 0.000 | 0.733 +/- 0.000 | 0.704 +/- 0.000 | 0.998 +/- 0.002 |

(mean +/- sample standard deviation over the 3 Stage 2 seeds. Local-only does not use Stage 2, so its retention varies only because the graph-always denominator does.) Seed 42 reproduces the Day 8 values exactly.

The seed has very little effect: the cascade keeps 98.1-98.2% of graph-always recall in every seed and its F1 is above both graph-always and random escalation in all three. This measures only Stage 2 seed variation. It does not capture variation from the choice of split, from a different model family or from the drift after step 42, so it should not be read as a confidence interval for future data.

**Outputs.** `results/e6_seed_spread.csv` and `web/data/results.json` (the export read by the web app; every number in it comes from `results/`).

**Dataset note.** The split of the 165 features into 93 local and 72 aggregated comes from the dataset paper. It could not be verified from the features file itself.