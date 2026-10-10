# FraudFin — Development Notes

## Day 1: Dataset inspection and loading

### Work completed
- Implemented `src/load_data.py` to load the Elliptic transaction features, classes and edge list.
- Mapped raw class `1` to illicit (`1`), raw class `2` to licit (`0`), and `unknown` to `-1`.
- Checked transaction IDs, feature values, edge endpoints and time-step structure.
- Added tests for data loading and label mapping.

### Dataset observations
- 203,769 transactions and 234,355 directed edges.
- 93 local features and 72 aggregated graph features.
- 4,545 illicit, 42,019 licit and 157,205 unlabelled transactions.
- The dataset covers time steps 1–49.

### Data handling
- Raw dataset files are kept out of Git.
- Unlabelled transactions are retained by the loader but excluded from supervised evaluation.
- `time_step`, transaction IDs and labels must never be used as model features.

## Day 2: Chronological splits and evaluation utilities

### Work completed
- Implemented `src/splits.py` to create chronological, labelled splits.
- Implemented `src/evaluate.py` with illicit-class metrics, per-step metrics, escalation-rate calculations, recall retention and relative-cost calculations.
- Added tests for chronological split integrity and hand-calculated evaluation metrics.

### Split definitions
- `train_stage1`: time steps 1–28.
- `calib`: time steps 29–34.
- `train_final`: time steps 1–34.
- `test`: time steps 35–49.

### Verified split counts
- `train_stage1`: 25,207 labelled transactions; 2,542 illicit.
- `calib`: 4,687 labelled transactions; 920 illicit.
- `train_final`: 29,894 labelled transactions; 3,462 illicit.
- `test`: 16,670 labelled transactions; 1,083 illicit.

### Important observations
- Illicit prevalence differs across periods: approximately 10.1% in `train_stage1`, 19.6% in `calib`, 11.6% in `train_final` and 6.5% in `test`.
- Calibration and test therefore have different class prevalences. This must be reported when interpreting later cascade results.
- The test period (steps 35–49) must not be used for training, calibration, threshold selection or model selection.
- Undefined metrics should be represented as `NaN` where appropriate; plain accuracy must not be the headline result.

### Tests
- Data loader tests: 13 passed.
- Split tests: 12 passed.
- Evaluation metric tests: 12 passed.

## Next steps

- Verify Git status and commit the completed Day 1–2 work.
- Test the Day 3 model utilities and baseline runner.
- Compare logistic regression and random forest on local features before repeating the comparison with all features.
- Run the everything-licit and shuffled-training-label sanity checks.
- Inspect performance by time step rather than assuming performance declines.
## Days 3–4: Standalone model baselines

### Work completed
- Implemented `src/models.py` with Logistic Regression and Random Forest.
- Implemented `run_baselines.py` for standalone RQ1 comparisons.
- Added `tests/test_models.py`.
- Compared local-only features (93 columns) against all features (165 columns).
- Trained on labelled transactions from time steps 1–34 and evaluated on steps 35–49.
- Used `class_weight="balanced"` for both model families. Logistic Regression uses feature scaling.
- Ran the complete test suite: **43 tests passed**.

### Baseline results

| Model | Features | Precision | Recall | F1 | PR-AUC |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Local | 0.141 | 0.861 | 0.242 | 0.252 |
| Logistic Regression | All | 0.183 | 0.879 | 0.303 | 0.291 |
| Random Forest | Local | 0.926 | 0.701 | 0.798 | 0.785 |
| Random Forest | All | 0.988 | 0.696 | 0.816 | 0.796 |

### Initial observations
- All features improved F1 and PR-AUC for both model families in this experiment.
- Random Forest achieved stronger overall baseline metrics than Logistic Regression.
- The everything-licit baseline achieved 93.5% accuracy but zero illicit recall, demonstrating why accuracy is unsuitable as the headline metric.
- Shuffled-label PR-AUC was 0.072 for Logistic Regression and 0.087 for Random Forest, compared with test illicit prevalence of approximately 0.065. These are diagnostic results and do not independently prove the absence of leakage.

### Generated outputs
- `results/baselines_local.csv`
- `results/baselines_per_step_local.csv`
- `results/baselines_all.csv`
- `results/baselines_per_step_all.csv`

### Next stage
Implement calibration and the cost-aware selective-escalation cascade according to `SPEC.md`. Keep the chronological splits fixed and do not use test results for model selection or threshold tuning.

## Day 4: RQ1 Feature Comparison and Random Forest

### Experiment setup
- Compared Logistic Regression (LR) and Random Forest (RF) using LOCAL and ALL feature sets.
- LOCAL contains 93 local features.
- ALL contains 165 features: local and aggregated features.
- Excluded `txId`, `time_step`, and `label` from model features.
- Trained standalone RQ1 models on labelled transactions from time steps 1–34.
- Evaluated on labelled transactions from time steps 35–49.
- LR used seed 42; RF results were averaged across seeds 42, 43, and 44.
- Used `class_weight="balanced"` for both model families. LR also used feature scaling.

### E1: Local vs All features

| Model | Features | Precision | Recall | F1 | PR-AUC |
|---|---|---:|---:|---:|---:|
| LR | Local | 0.141 | 0.861 | 0.242 | 0.252 |
| LR | All | 0.183 | 0.879 | 0.303 | 0.291 |
| RF | Local | 0.926 | 0.701 | 0.798 | 0.785 |
| RF | All | 0.988 | 0.696 | 0.816 | 0.796 |

### Findings
- Adding aggregated features improved F1 and PR-AUC for both model families in this experiment.
- RF achieved stronger F1 and PR-AUC than LR on both feature sets.
- RF recall decreased slightly from 0.701 to 0.696 when moving from LOCAL to ALL features; improvements were not uniform across every metric.
- These are test-set observations, not proof that aggregated features will always improve performance.

### Random Forest: short theory
Random Forest is an ensemble learning algorithm that combines many decision trees. Each tree is trained using randomized samples of the training data and considers randomized subsets of features when choosing splits. Their outputs are aggregated to produce the final prediction or probability.

Advantages:
- Learns nonlinear relationships and feature interactions.
- Combines many trees, often reducing the instability of an individual tree.
- Can model complex patterns without requiring feature scaling.

In this project, `class_weight="balanced"` helps account for the imbalance between illicit and licit transactions. It does not guarantee high illicit recall or eliminate false positives.

### Interpretation of evaluation metrics
- Precision: proportion of predicted illicit transactions that are actually illicit.
- Recall: proportion of actual illicit transactions detected.
- F1: harmonic mean of precision and recall.
- PR-AUC: average precision, summarizing precision-recall ranking performance across thresholds.

Accuracy is not used as the headline metric because a model can predict nearly everything as licit and still achieve high accuracy while detecting no illicit transactions.

### Next stage
Implement probability calibration and the cost-aware selective-escalation cascade. Keep test steps 35–49 out of calibration, threshold selection, and model selection.

### E1: Local vs All Feature Comparison

**Experiment:** Models trained on labelled transactions from time steps 1–34 and evaluated on steps 35–49. Logistic Regression uses seed 42; Random Forest reports the mean over seeds 42, 43, and 44. The decision threshold is 0.5.

| Model | Feature set | Precision | Recall | F1 | PR-AUC |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Local (93 features) | 0.141 | 0.861 | 0.242 | 0.252 |
| Logistic Regression | All (165 features) | 0.183 | 0.879 | 0.303 | 0.291 |
| Random Forest | Local (93 features) | 0.926 | 0.701 | 0.798 | 0.785 |
| Random Forest | All (165 features) | 0.988 | 0.696 | 0.816 | 0.796 |

**Interpretation:** Adding aggregated features improved F1 and PR-AUC for both model families in this experiment. Random Forest outperformed Logistic Regression on F1 and PR-AUC for both feature sets. However, Random Forest recall decreased slightly when aggregated features were added, from 0.701 to 0.696. These findings describe this test set and do not establish that aggregated features will always improve performance.

### E2 — Per-time-step drift analysis

**Setup:** Models trained on labelled transactions from time steps 1–34 and evaluated on test steps 35–49 using a fixed threshold of 0.5. Random Forest results are averaged over seeds 42–44; Logistic Regression uses seed 42. No test-set tuning was performed.

**Results:** Adding aggregated graph features improved pooled F1 and PR-AUC for both model families. Logistic Regression F1 increased from 0.242 (LOCAL) to 0.303 (ALL), while Random Forest F1 increased from 0.798 to 0.816. Random Forest with ALL features achieved the strongest overall result: precision 0.988, recall 0.696, F1 0.816, and PR-AUC 0.796.

Temporal variation: For Random Forest, per-step F1 is high on steps 35-42 for both
feature sets (mean 0.88) and falls to near zero from step 43 onward (0.07 LOCAL,
0.09 ALL). Illicit recall is 0.83 (LOCAL) and 0.82 (ALL) on steps 35-42 and 0.02
on steps 43-49 (about 3 of 169 illicit transactions). ALL had higher per-step F1 in
8 of 15 steps (3 lower, 4 tied at zero); the mean per-step gain of +0.012 falls to
+0.004 when steps 45-46 (fewer than 10 illicit rows) are excluded, so the pooled RF
gain comes mainly from higher precision. For Logistic Regression, ALL was higher in
13 of 15 steps, with the largest gains in steps 47-49. No cause for these changes
is established.

**Limitations:** Test steps 45 and 46 contain fewer than 10 illicit examples each, making per-step metrics less stable. The observed changes describe temporal performance variation; they do not establish its cause.

**Artifacts:** `results/e2_per_step.csv`, `results/e2_overall.csv`, `results/figures/e2_drift_rf.png`, and `results/figures/e2_drift_lr.png`.