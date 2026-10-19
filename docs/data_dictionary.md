# Data Dictionary

| Column | Type | Meaning |
|---|---|---|
| `record_ref` | string | Synthetic booking event reference |
| `resident_ref` | string | Synthetic resident identifier |
| `facility_ref` | string | Facility booked |
| `booking_timestamp` | datetime | When the booking was created |
| `usage_timestamp` | datetime | Scheduled facility usage time |

The source table contains event-level facts only. Model features and next-event labels are derived by `src/train_evaluate.py`.
