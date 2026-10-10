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
