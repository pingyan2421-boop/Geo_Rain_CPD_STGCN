import argparse
import os
from datetime import date, datetime, timedelta

import numpy as np


def parse_date_from_name(name):
    return datetime.strptime(name.replace(".npz", ""), "%Y-%m-%d").date()


def main():
    parser = argparse.ArgumentParser(description="Validate cached CHIRPS daily crop NPZ files.")
    parser.add_argument("--daily_dir", default=os.path.join("dataset", "rainfall", "chirps_crop", "daily"))
    parser.add_argument("--start_date", default="2014-10-24")
    parser.add_argument("--end_date", default="2022-07-26")
    parser.add_argument("--sample_all", action="store_true")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    expected_total = (end_date - start_date).days + 1

    if not os.path.isdir(args.daily_dir):
        raise SystemExit(f"Daily cache directory not found: {args.daily_dir}")

    names = sorted(name for name in os.listdir(args.daily_dir) if name.endswith(".npz"))
    dates = [parse_date_from_name(name) for name in names]
    date_set = set(dates)
    expected_dates = [start_date + timedelta(days=i) for i in range(expected_total)]
    cached_expected = [d for d in expected_dates if d in date_set]
    missing_before_last = []
    if dates:
        last = max(dates)
        missing_before_last = [d for d in expected_dates if start_date <= d <= last and d not in date_set]

    bad_files = []
    shapes = {}
    nan_files = 0
    negative_files = 0
    min_value = float("inf")
    max_value = float("-inf")
    sum_value = 0.0
    count_value = 0

    for name in names:
        path = os.path.join(args.daily_dir, name)
        try:
            with np.load(path) as data:
                arr = data["rain_mm"].astype(np.float32)
        except Exception as exc:
            bad_files.append((name, type(exc).__name__, str(exc)))
            continue
        shapes[arr.shape] = shapes.get(arr.shape, 0) + 1
        finite = np.isfinite(arr)
        if not finite.all():
            nan_files += 1
        finite_values = arr[finite]
        if finite_values.size:
            if float(finite_values.min()) < -1e-6:
                negative_files += 1
            min_value = min(min_value, float(finite_values.min()))
            max_value = max(max_value, float(finite_values.max()))
            sum_value += float(finite_values.sum())
            count_value += int(finite_values.size)

    first = min(dates).isoformat() if dates else "-"
    last = max(dates).isoformat() if dates else "-"
    mean_value = sum_value / count_value if count_value else float("nan")
    print(f"cache_dir={args.daily_dir}")
    print(f"cached_files={len(names)} expected_total={expected_total} cached_expected_range={len(cached_expected)}")
    print(f"first={first} last={last}")
    print(f"missing_before_last={len(missing_before_last)}")
    if missing_before_last:
        preview = ", ".join(d.isoformat() for d in missing_before_last[:10])
        print(f"missing_preview={preview}")
    print(f"bad_files={len(bad_files)}")
    if bad_files:
        for row in bad_files[:10]:
            print(f"bad_file={row[0]} error={row[1]} message={row[2]}")
    print(f"shapes={shapes}")
    print(f"nan_files={nan_files} negative_files={negative_files}")
    print(f"rain_mm_min={min_value:.6f} rain_mm_max={max_value:.6f} rain_mm_mean={mean_value:.6f}")


if __name__ == "__main__":
    main()
