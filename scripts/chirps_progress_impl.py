import argparse
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Show CHIRPS crop download progress.")
    parser.add_argument("--daily_dir", default=os.path.join("dataset", "rainfall", "chirps_crop", "daily"))
    parser.add_argument("--total", type=int, default=2833)
    parser.add_argument("--width", type=int, default=40)
    args = parser.parse_args()

    files = []
    if os.path.isdir(args.daily_dir):
        files = sorted(name for name in os.listdir(args.daily_dir) if name.endswith(".npz"))
    count = len(files)
    percent = count / max(args.total, 1)
    filled = int(round(args.width * percent))
    bar = "#" * filled + "-" * (args.width - filled)
    first = files[0].replace(".npz", "") if files else "-"
    last = files[-1].replace(".npz", "") if files else "-"
    remaining = max(args.total - count, 0)
    print(f"CHIRPS crop progress [{bar}] {count}/{args.total} ({percent * 100:.1f}%)")
    print(f"first={first} last={last} remaining_days={remaining}")
    print(f"checked_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
