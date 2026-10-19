# Technical Documentation — Facility Usage Prediction System

## 1. Objective and target definition
The prototype predicts a resident's next facility booking using their current and prior booking history. Four outputs are evaluated: next facility, usage weekday, usage hour, and suggested notification timestamp. A regression target estimates days from the anchor booking to the next usage event, allowing a date-level usage estimate rather than assuming the next usage is within seven days. This next-event framing avoids the circular task of predicting the facility on a booking row where that facility is already known.

## 2. Dataset
`data/facility_bookings.csv` contains 12,000 synthetic booking events for 450 residents and six facilities between 2024-01 and 2026-09. Columns are `record_ref`, `resident_ref`, `facility_ref`, `booking_timestamp`, and `usage_timestamp`. The deterministic generator models resident-level facility preferences, varying popularity, facility-specific usage hours and weekdays, weekend/seasonality effects, different lead-time distributions, long-tail advance bookings, 4.5% usage-time noise, and gradual preference/popularity drift. No real personal data is used.

## 3. Supervised examples and leakage controls
For each resident, events are sorted by booking timestamp. Each event is an anchor; the next event for that resident supplies the target. Features are calculated only from the anchor and events before it: resident ID, anchor facility and booking calendar fields, history count, number of unique facilities, historical top facility, historical mean lead time/usage hour/usage weekday, days since the previous booking, historical mean/median booking gaps, and per-facility counts. The next event's facility, usage timestamp, booking timestamp, and lead time are excluded from features.

The holdout is defined by the target event's booking timestamp: the latest 20% of supervised examples are reserved for evaluation. Feature preprocessing is fitted only on training rows. The fixed data-generation seed and model seeds make the workflow reproducible. Note: the same resident can appear in both chronological train and test, as is appropriate for forecasting returning residents; cold-start performance for unseen residents is not established.

## 4. Models and alternatives
- ExtraTreesClassifier for next facility, weekday, and hour.
- ExtraTreesRegressor for time until next usage and time until next booking. For residents with at least three observed events, historical modal usage weekday/hour are used as a personalized low-variance fallback.
- Predicted usage time is the next occurrence of the predicted weekday/hour after the anchor booking timestamp.
- Suggested notification time is predicted usage time minus predicted lead time.
- A simple training-only baseline (majority class for facility/day/hour and median lead time) is included to contextualize model performance.

Tree ensembles are a compact baseline for nonlinear interactions. Further alternatives include CatBoost/gradient boosting, per-resident empirical frequency baselines, probability ranking with top-k facility recommendations, survival/time-to-event models, and a multi-task model. Compare these on the same chronological split before choosing a production model.

## 5. Metrics
Classification outputs report exact-match accuracy and macro F1. Notification time reports MAE in hours and the proportion within 24 hours of the proxy. An overall mean output score averages facility accuracy, weekday accuracy, hour accuracy and notification-within-24-hours rate. The all-four-match rate is also reported. The workbook contains each holdout record, predicted and actual values, output-level match indicators, a metrics sheet, and grouped error analysis by actual facility, weekday, and history depth. Baseline metrics are included separately.

## 6. Error analysis
Inspect `outputs/prediction_review.xlsx` and group errors by facility, weekday, usage hour, and resident-history count. Common expected failure modes are sparse history, preference drift, random time perturbations, facility popularity imbalance, and long-tail lead times. Exact-hour accuracy is strict: an error of one hour counts as incorrect. Add ±1-hour and ±2-hour accuracy in future iterations to reflect operational tolerance.

## 7. Notification timing caveat
There is no actual notification-send timestamp in the required source schema. In line with the assignment's illustrative comparison, the booking timestamp is used as a proxy for the actual nudge time. Therefore notification metrics do **not** measure whether a real notification was sent at the right time or whether it caused a booking. A production dataset should record notification scheduling/sending, delivery, resident consent, quiet hours, opt-outs, and downstream booking outcomes.

## 8. Limitations
- Results are from synthetic data and do not establish real-world performance.
- The generator does not model facility capacity, booking conflicts, cancellations, no-shows, opening-hour constraints, or household relationships.
- Resident identity features help recurring-resident predictions; unseen-resident cold-start performance needs a separate evaluation.
- A single chronological holdout can be sensitive to the cutoff. Rolling-origin evaluation is recommended.
- Apply consent, quiet-hour, and notification-frequency rules outside the model.
- The baseline may be hard to beat for sparse/noisy targets; compare per-resident frequency baselines and calibrate any claims against actual results.

## 9. Reproduction
From the project root, install dependencies and run `python src/train_evaluate.py`. To regenerate data, run `python src/generate_dataset.py` first. Both scripts use fixed seeds. Outputs are written under `outputs/`.
