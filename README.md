# Facility Usage Prediction System

A reproducible machine-learning prototype that predicts a resident's **next** facility booking, expected usage weekday/hour, and a suggested notification timestamp.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train_evaluate.py
```

Optional: regenerate the deterministic synthetic data first:

```bash
python src/generate_dataset.py
python src/train_evaluate.py
```

## Deliverables

- `data/facility_bookings.csv`: synthetic booking-event dataset.
- `src/generate_dataset.py`: fixed-seed dataset generator.
- `src/train_evaluate.py`: feature construction, chronological holdout, fitting, metrics, and prediction review.
- `outputs/prediction_review.xlsx`: row-level prediction vs actual workbook plus metrics and a data sample.
- `outputs/prediction_review.csv`: row-level holdout predictions.
- `outputs/metrics.csv`: per-output and baseline metrics.
- `outputs/error_analysis.csv`: breakdown by facility, weekday, and history depth.
- `outputs/models.joblib`: fitted model artifacts and baseline summary.
- `outputs/run_metadata.json`: split, feature and run metadata.
- `docs/technical_documentation.md`: modelling and evaluation decisions.
- `docs/data_dictionary.md`: source schema.

## Problem framing

At each resident booking event, use that event and earlier events to predict the resident's next booking event. The model predicts the next facility, usage weekday and hour. A regression model predicts days until the next usage event, and a separate model predicts days until the next booking to estimate the notification timestamp against the assignment's booked-at proxy.

The actual booking timestamp is used as a proxy for actual notification time because the synthetic dataset does not contain notification-send logs. It is not a measurement of real notification effectiveness.

## Reproducibility and leakage controls

- The generator uses a fixed seed (`20261019`).
- The training code creates next-event examples per resident; target labels are the next event, not the anchor event.
- Feature construction uses only the anchor and earlier events. The target event is never used to compute features.
- The holdout is selected by target booking timestamp, with the latest 20% of examples held out chronologically.
- Imputation and one-hot encoding are fitted within each model pipeline on training data only.
- A simple training-set mode/median baseline is included for context.
