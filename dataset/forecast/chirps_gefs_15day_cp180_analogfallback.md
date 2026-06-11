# Forecast Climatology Fallback

## Inputs

- forecast_features: .\dataset\forecast\chirps_gefs_15day_cp180.csv
- rainfall_features: .\dataset\rainfall\chirps_daily.csv
- missing origin dates filled: 64
- official forecast rows: 155
- fallback rows: 960
- output rows: 1115

## Fallback Rule

Fallback rows are generated only for sample origin dates whose existing forecast_context is unavailable. For each issue date, lead days 1-15 are estimated from CHIRPS rainfall strictly before the issue date. The estimator first uses same-season historical daily climatology; if there is not enough history, it falls back to a trailing historical mean.

When `--fallback_method analog` is used, each missing issue date is filled from the mean lead-day sequence of historical issue dates with similar antecedent CHIRPS rainfall. Candidate analog lead windows must be fully before the issue date, so target-period rainfall is not used.

This is a non-leaking coverage baseline, not a real weather forecast.

## Filled Origin Date Range

- start: 2020-01-14
- end: 2022-05-27
