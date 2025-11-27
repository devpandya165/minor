#!/usr/bin/env python3
"""
SUPER FAST Kafka producer - optimized for speed
- Uses pandas for fast CSV reading
- Logs every batch sent
- Minimal memory overhead
- Starts sending within 5 seconds
"""

from kafka import KafkaProducer
import json
import time
import os
import argparse
from typing import List, Dict
import pandas as pd

# ----------- CONFIG -----------
DEFAULT_STREAMING_CSV    = "/home/devpandya/data/streaming_source_from_hour168.csv"
DEFAULT_TOPIC            = "electricity_topic"
DEFAULT_BOOTSTRAP        = "localhost:9092"
DEFAULT_BATCH_INTERVAL   = 15
# ------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Ultra-fast Kafka electricity producer")
    p.add_argument("--streaming-csv", default=DEFAULT_STREAMING_CSV, help="Streaming CSV path.")
    p.add_argument("--topic", default=DEFAULT_TOPIC, help="Kafka topic name.")
    p.add_argument("--bootstrap", default=DEFAULT_BOOTSTRAP, help="Kafka bootstrap servers.")
    p.add_argument("--batch-interval", type=int, default=DEFAULT_BATCH_INTERVAL,
                   help="Seconds between hourly batches.")
    p.add_argument("--max-hours", type=int, default=None, help="Optional limit on hourly batches.")
    return p.parse_args()


def main():
    args = parse_args()

    print("========== ULTRA-FAST Kafka Electricity Producer ==========")
    print(f"CSV              : {args.streaming_csv}")
    print(f"Topic            : {args.topic}")
    print(f"Bootstrap        : {args.bootstrap}")
    print(f"Batch interval   : {args.batch_interval}s")
    print("===========================================================")

    # ========== FAST CSV LOADING WITH PANDAS ==========
    print("\n[CSV] Loading with pandas (super fast)...")
    csv_start = time.time()

    if not os.path.exists(args.streaming_csv):
        print(f"[ERROR] CSV not found: {args.streaming_csv}")
        return

    # Load CSV with pandas (much faster than csv.DictReader)
    df = pd.read_csv(args.streaming_csv)
    csv_time = time.time() - csv_start

    print(f"[CSV] ✅ Loaded {len(df)} rows in {csv_time:.2f}s")
    print(f"[CSV] ✅ Found {df['Timestamp_UTC'].nunique()} unique timestamps")

    # ========== CONNECT TO KAFKA ==========
    print(f"\n[KAFKA] Connecting to {args.bootstrap}...")

    try:
        producer = KafkaProducer(
            bootstrap_servers=[args.bootstrap.strip()],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
        )
        print(f"[KAFKA] ✅ Connected in {time.time() - csv_start - csv_time:.2f}s")
    except Exception as e:
        print(f"[KAFKA] ❌ {e}")
        return

    startup_time = time.time() - csv_start
    print(f"\n[STARTUP] ✅ Ready in {startup_time:.2f} seconds!")
    print(f"[PRODUCER] Starting to send data...\n")

    # ========== SEND DATA ==========

    try:
        start_time = time.time()
        hour_count = 0

        # Get unique timestamps in order
        timestamps = sorted(df['Timestamp_UTC'].unique())
        total_hours = len(timestamps)

        for ts in timestamps:
            batch = df[df['Timestamp_UTC'] == ts]

            # Send all records for this hour
            for _, row in batch.iterrows():
                record = row.to_dict()
                producer.send(args.topic, record)

            producer.flush()

            hour_count += 1
            record_count = len(batch)

            # Log EVERY batch
            pct = 100 * hour_count // total_hours
            elapsed = time.time() - start_time
            rate = hour_count / elapsed if elapsed > 0 else 0
            remaining = (total_hours - hour_count) / rate if rate > 0 else 0

            print(f"[PRODUCER] Batch #{hour_count:5d} | Time: {ts} | Records: {record_count} | "
                  f"Progress: {pct:3d}% | Rate: {rate:6.1f}h/s | ETA: {remaining:6.0f}s")

            if args.max_hours is not None and hour_count >= args.max_hours:
                print(f"\n[PRODUCER] Reached max-hours={args.max_hours}.")
                break

            # Delay between batches
            time.sleep(args.batch_interval)

        elapsed = time.time() - start_time
        total_records = hour_count * 36  # 36 states per hour

        print(f"\n{'='*80}")
        print(f"[PRODUCER] ✅ DONE!")
        print(f"{'='*80}")
        print(f"Total batches sent  : {hour_count}")
        print(f"Total records sent  : {total_records}")
        print(f"Time taken          : {elapsed:.1f}s")
        print(f"Rate                : {hour_count/elapsed:.1f} batches/second")
        print(f"Total startup time  : {startup_time:.2f}s")
        print(f"{'='*80}")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        producer.close()
        print("[PRODUCER] ✅ Closed.")


if __name__ == "__main__":
    main()
