#!/usr/bin/env python3
"""
Split a full hourly multi-state CSV into:
  - history_seed.csv        (first N history hours, N=168 by default)
  - streaming_source_from_hour{start_stream_index}.csv  (remaining hours starting at start_stream_index)

Usage:
  python generate_history_subset.py \
      --input /home/devpandya/data/synthetic_indian_load_data.csv \
      --output-dir /home/devpandya/data \
      --history-hours 168 \
      --start-stream-index 168 \
      --validate

If you set --no-split, the script only writes history_seed.csv and leaves the original CSV untouched.
Then you configure the producer to skip the first history-hours timestamps.
"""

import argparse, csv, os, sys
from datetime import datetime
from typing import List, Dict

REQUIRED_COLUMNS = [
    "State_Code","Timestamp_UTC","Gross_Load_MW",
    "Hour_Of_Day","Day_Of_Week","Is_Weekend","Is_Holiday_State",
    "Avg_Temp_C","Temp_Change_6H","Avg_Humidity_Pct"
]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Full yearly CSV path")
    ap.add_argument("--output-dir", required=True, help="Directory to write subset CSVs")
    ap.add_argument("--history-hours", type=int, default=168, help="Number of preceding hours to seed history")
    ap.add_argument("--start-stream-index", type=int, default=168,
                    help="Timestamp index at which streaming/producer should begin")
    ap.add_argument("--no-split", action="store_true",
                    help="If set, do not write streaming file; only write history_seed.csv")
    ap.add_argument("--validate", action="store_true", help="Perform consistency checks")
    return ap.parse_args()

def read_rows(path: str) -> List[Dict[str,str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Missing header row.")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing}")
        return list(reader)

def write_csv(path: str, fieldnames: List[str], rows: List[Dict[str,str]]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = read_rows(args.input)
    if not rows:
        print("[ERROR] Input CSV empty.", file=sys.stderr)
        sys.exit(1)

    # Gather timestamps
    timestamps = sorted(set(r["Timestamp_UTC"] for r in rows))
    # Map timestamp -> rows
    ts_to_rows = {}
    for ts in timestamps:
        ts_to_rows[ts] = [r for r in rows if r["Timestamp_UTC"] == ts]

    # Basic validation: uniform number of states per timestamp
    states_per_ts = {ts: len(rs) for ts, rs in ts_to_rows.items()}
    unique_counts = set(states_per_ts.values())
    if args.validate:
        if len(unique_counts) != 1:
            print(f"[WARN] Non-uniform record counts per timestamp: {unique_counts}")
        else:
            print(f"[INFO] Each timestamp has {unique_counts.pop()} rows.")
        # Optional: check hourly increment
        try:
            parsed = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in timestamps]
            gaps = []
            for i in range(1, len(parsed)):
                delta_h = (parsed[i] - parsed[i-1]).total_seconds()/3600.0
                if abs(delta_h - 1.0) > 1e-6:
                    gaps.append((timestamps[i-1], timestamps[i], delta_h))
            if gaps:
                print(f"[WARN] Found non-1h gaps: {gaps[:10]} (showing up to 10)")
            else:
                print("[INFO] All consecutive timestamps are 1-hour apart.")
        except Exception as e:
            print(f"[WARN] Timestamp parsing issue: {e}")

    history_hours = args.history_hours
    start_stream_index = args.start_stream_index

    if history_hours > start_stream_index:
        print(f"[ERROR] history-hours ({history_hours}) cannot exceed start-stream-index ({start_stream_index}).",
              file=sys.stderr)
        sys.exit(1)

    if start_stream_index > len(timestamps):
        print(f"[ERROR] start-stream-index {start_stream_index} beyond total timestamps {len(timestamps)}.",
              file=sys.stderr)
        sys.exit(1)

    # Build history seed subset
    history_ts = timestamps[:history_hours]    # first N timestamps
    history_rows = []
    for ts in history_ts:
        history_rows.extend(ts_to_rows[ts])

    history_out = os.path.join(args.output_dir, "history_seed.csv")
    write_csv(history_out, list(rows[0].keys()), history_rows)
    print(f"[INFO] Wrote history_seed.csv with {len(history_rows)} rows "
          f"({history_hours} hours * {len(history_rows)//history_hours} states).")

    if not args.no_split:
        stream_ts = timestamps[start_stream_index:]   # start at index 168 (or user-defined)
        streaming_rows = []
        for ts in stream_ts:
            streaming_rows.extend(ts_to_rows[ts])
        stream_out = os.path.join(args.output_dir,
                                  f"streaming_source_from_hour{start_stream_index}.csv")
        write_csv(stream_out, list(rows[0].keys()), streaming_rows)
        print(f"[INFO] Wrote streaming source CSV starting at hour index {start_stream_index}: "
              f"{len(streaming_rows)} rows.")
    else:
        print("[INFO] --no-split set: not writing streaming subset CSV.")

    print("[DONE] Split operation complete.")

if __name__ == "__main__":
    main()