# SPEC.md — Cost-Aware Selective Escalation for Illicit Bitcoin Transaction Detection



## 0. Rules for any AI assistant working on this project

1. Do not rename files, columns, functions or config keys defined here. If something must change, say so and wait.
2. NEVER use `time_step` as a model feature. It is only for splitting and per-step evaluation.
3. NEVER touch the test steps (35-49) for training, calibration, threshold choice or model selection. Test is evaluated once, at the end.
4. NEVER retrain Stage 1 after the calibrator and threshold are frozen.
5. Train and evaluate only on labelled nodes (illicit or licit). Unlabelled nodes are excluded from metrics.
6. Never report plain accuracy. Illicit is the positive class.
7. One module per request. Give function signatures first, then code, then a small pytest test.
8. After each module, explain it as if I must defend it in a viva, and list 5 likely examiner questions.
9. Flag any assumption you are unsure about instead of guessing.

## 1. Project in one paragraph

A two-stage fraud detector on the real Elliptic Bitcoin transaction graph. Stage 1 scores every transaction using local features only and clears the confidently licit ones. Only the rest are escalated to Stage 2, which also uses graph (neighbour-aggregated) features. We measure how much illicit recall is retained against simulated relative cost, compare against a random-escalation control, and describe how performance changes over time. Chronological evaluation only.

Fallback (only if the Day 8 checkpoint fails or the dataset cannot be used): the same cascade on the Kaggle Credit Card Fraud dataset.

## 2. Research questions

- RQ1: How much do graph features add over local features, overall and at each time step?
- RQ2: How much illicit recall is kept for how much simulated relative cost saved, and does uncertainty-based routing beat random routing at the same escalation rate?

## 3. Data (data/raw/, git-ignored)

Files (filenames confirmed in the downloaded dataset):
- `elliptic_txs_features.csv` — no header. Column 0 = txId, column 1 = time step (1-49), columns 2+ = features.
- `elliptic_txs_classes.csv` — header `txId,class`. Raw values: `unknown`, `1`, `2`.
- `elliptic_txs_edgelist.csv` — header `txId1,txId2`. Directed payment flows.

Label mapping (we use our own coding): raw `1` -> illicit = 1; raw `2` -> licit = 0; `unknown` -> -1 (unlabelled).

Expected facts (VERIFY on Day 1, do not assume):
- 203,769 transactions; 234,355 edges; 166 features per transaction (time step is one of them), so the features CSV should have 167 columns including txId.
- About 4,545 illicit, 42,019 licit, rest unlabelled.
- Expected layout: time step, then about 93 local features, then about 72 aggregated (1-hop neighbour) features. Confirm the split point.
- Edges are expected to stay within one time step. Confirm.

Our column names after loading:
- `txId`, `time_step`, `local_1..local_N` (expected N = 93), `agg_1..agg_M` (expected M = 72), `label` (1 / 0 / -1).
- Feature sets: LOCAL = all `local_*`; ALL = `local_*` + `agg_*`. Never include `time_step`, `txId` or `label` as features.

## 4. Splits (chronological only)

| Use | Time steps |
|---|---|
| Stage 1 training, and a validation copy of Stage 2 | 1-28 |
| Calibration of Stage 1, choice of `d` and `t_clear` | 29-34 |
| Final training of standalone RQ1 models and of the final Stage 2 | 1-34 |
| Test (evaluated once) | 35-49 |

A pytest test must fail if any training step is later than any validation or test step.

## 5. Models

- Two families only: logistic regression (LR, scaled features, `class_weight="balanced"`) and random forest (RF, `class_weight="balanced"`). Random seeds in config.
- Same family within every comparison. RQ1: LR(LOCAL) vs LR(ALL), RF(LOCAL) vs RF(ALL).
- Cascade: Stage 1 = family(LOCAL), Stage 2 = same family(ALL). Never mix families in one cascade. Run both families; lead with RF.
- RQ1 standalone models are trained on steps 1-34.

## 6. Training sequence for the cascade (follow in order)

1. Train Stage 1 (LOCAL) on steps 1-28.
2. Calibrate Stage 1 with sigmoid (Platt) scaling on steps 29-34. Report Brier score and a reliability plot before and after.
3. Choose the standalone decision threshold `d` for Stage 1 on steps 29-34 (maximise illicit F1). `d` is also used for the local-only baseline.
4. Train a validation copy of Stage 2 (ALL) on steps 1-28. Sweep `t_clear` on steps 29-34 (only values <= `d`). Choose the largest `t_clear` that keeps recall retention at or above the target (default 0.95, report whatever is achievable).
5. FREEZE Stage 1, the calibrator, `d` and `t_clear`.
6. Train the final Stage 2 (ALL) on steps 1-34.
7. Evaluate once on steps 35-49.
d is chosen on Day 6 (max illicit F1 on steps 29-34, calibrated Stage 1 probabilities) and frozen with Stage 1.
On Day 7 only t_clear is chosen, on steps 29-34, with t_clear ≤ d.
## 7. Cascade and baselines

Router: `p1 < t_clear` -> licit (cleared, no graph features); otherwise escalate to Stage 2, whose label is final.

Systems compared (RQ2), all on test steps:
- Local-only: frozen Stage 1 with threshold `d`.
- Graph-always: final Stage 2 on everything.
- Cascade: as above.
- Random escalation (control): for each cascade escalation rate, randomly select the same share of test transactions for Stage 2; non-selected transactions keep the Stage 1 decision (illicit if `p1 >= d`). Repeat with 5 fixed seeds; report mean and spread.

Interpretation: if uncertainty-based routing beats random routing at the same escalation rate, Stage 1 confidence carries useful information about which transactions need graph-aware analysis.

Note for the report: the RQ1 local-only model (trained on 1-34) and the RQ2 local-only baseline (frozen Stage 1, trained on 1-28) are different models. Label them clearly in tables.

### Stage 2 decision threshold (`d2`)

- Select `d2` by maximizing illicit-class F1 on calibration steps 29–34, using the validation copy of Stage 2 trained on steps 1–28 with the `ALL` feature set.
- Use the same threshold-selection rule as Stage 1's `d`: maximize illicit-class F1 on steps 29–34 using the applicable validation predictions.
- Freeze `d2` after selection. Do not adjust it using test steps 35–49.
- The recall-retention target is 0.95.
- The Stage 1 clearing threshold must satisfy `t_clear <= d`, where `d` is already frozen from Day 6. Select `t_clear` on steps 29–34 only, choosing the largest value that achieves recall retention of at least 0.95. If none achieves the target, report the best achieved retention.
- Do not use test steps 35–49 for threshold selection or retuning.
## 8. Metrics

Illicit-class precision, recall, F1; PR-AUC (`average_precision_score`); per-time-step versions; `escalation_rate` = share of transactions with `p1 >= t_clear`; `recall_retention` = recall(cascade) / recall(graph-always).

## 9. Cost model (simulated, relative)

Local-feature cost = 1. Graph-feature cost = `r` times local.

```
graph_always_cost = 1 + r
cascade_cost      = 1 + escalation_rate * r
relative_savings  = 1 - cascade_cost / graph_always_cost
```

Report r = 5, 20 and 100. This is a simulated relative cost because the dataset's graph features are precomputed. Escalation rate is reported separately and is not the cost.

## 10. Drift

Plot per-time-step illicit F1, precision and recall for local-only, graph-always and cascade. Describe only what the plot shows. Do not claim a cause that has not been tested.

## 11. Sanity checks (must pass before trusting any number)

- Split-ordering test (Section 4).
- "Predict everything licit": accuracy is high among labelled nodes, recall is 0.
- Shuffled training labels: F1 collapses.
- Precision/recall hand-computed on a 10-row toy example match the code.
- Toy cascade where the escalation count is known by hand.
- Baseline numbers are roughly in line with the original paper (Weber et al., 2019). If far off, check labels and splits first.

## 12. Repository layout

```
fraud-escalation/
  SPEC.md
  README.md
  requirements.txt
  config.yaml          # split steps, seeds, threshold grid, r values, recall target
  data/raw/            # git-ignored
  src/
    load_data.py       # load_features(), load_classes(), load_edges(), build_dataset()
    splits.py          # make_splits(df, cfg) -> dict of DataFrames; assert_chronological(splits)
    models.py          # make_model(family, seed) ; fit(model, X, y)
    calibration.py     # calibrate_sigmoid(model, X_val, y_val) ; brier(); reliability_data()
    cascade.py         # route(p1, t_clear) ; run_cascade(...) ; random_control(...)
    evaluate.py        # illicit_metrics(y, y_pred, y_score) ; per_step_metrics(...) ; relative_cost(...)
  app/streamlit_app.py # one page: threshold slider, cost-vs-recall chart, drift chart
  notebooks/           # 01_eda, 02_results
  results/             # tables and figures (precomputed for the app)
  tests/               # test_splits.py, test_cascade.py, test_metrics.py
  docs/notes.md        # my own explanations, viva prep
```

## 13. Conventions

- Python 3.10+. pandas, numpy, scikit-learn, matplotlib, plotly, streamlit, pyyaml, pytest.
- Pure functions on DataFrames/arrays; no hidden global state. Seeds and thresholds in `config.yaml`.
- Commit after every working module. Tag `v1-rq1` after RQ1 and `v2-rq2` after RQ2.

## 14. Open items to verify on Day 1

- [ ] Row counts and column counts match Section 3
- [ ] Class value counts; raw `1` = illicit, raw `2` = licit
- [ ] Split point between local and aggregated features
- [ ] Edges stay within one time step
- [ ] Illicit share per time step plotted
- [ ] Dataset licence and terms read on the download page

## 15. Current status

(update after every working module)

- Day 0: dataset download started.
- Day 1 — Complete:** Implemented `src/load_data.py` with `load_features()`, `load_classes()`, `load_edges()`, and `build_dataset()`. Verified dataset dimensions (203,769 transactions, 234,355 edges, and 167 columns in the raw feature file), class mapping (raw `1` = illicit, raw `2` = licit, `unknown` = unlabelled), and the local/aggregated feature split (93 local and 72 aggregated features). Checked transaction ID uniqueness, missing feature values, edge endpoints, and cross-time-step edges. Generated a per-time-step illicit-share plot and summary CSV. Added and ran Day 1 tests; all **13 Day 1 tests passed**.
- Day 2 — Complete: Implemented src/splits.py and configured chronological split windows in config.yaml. Added split-ordering, labelled-node filtering, configuration-validation, leakage, and expected-count tests. Verified split counts against the Day 1 diagnostics. Full test suite passed: 25 tests.


Day 3 — Evaluation and model foundations
Implemented evaluation metrics and model/feature-set utilities; tests cover metric calculations and model setup.



Day 4 — RQ1 baseline comparison
Ran LOCAL versus ALL feature baselines for Logistic Regression and Random Forest; recorded the E1 results and sanity checks.



Day 5 — Temporal drift analysis
Implemented drift metrics and plots, added tests, and documented per-time-step RQ1 results. Verify the edited numerical claims before calling the analysis final.