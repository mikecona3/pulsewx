"""
Fetch NEXRAD Level II radar scans from NOAA's public AWS S3 bucket.

Bucket: unidata-nexrad-level2 (public, no credentials required). This
replaced the old noaa-nexrad-level2 bucket, which now returns Access Denied.
Layout: s3://unidata-nexrad-level2/YYYY/MM/DD/STATION/STATION_YYYYMMDD_HHMMSS_V06
(older archive files end in .gz)

Station codes: 4-letter NEXRAD site IDs, e.g. KBGM (Binghamton, NY),
KTLX (Twin Lakes / OKC), KTYX (Montague, NY / western NY / Great Lakes).
Full list: https://www.roc.noaa.gov/branches/program-branch/support-branch/wsr-88d-radar-locations
"""

import os
from datetime import datetime, timedelta, timezone

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "unidata-nexrad-level2"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _client():
    # Public bucket: unsigned requests, no AWS credentials needed.
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_available_scans(station: str, date: datetime) -> list[str]:
    """List scan keys for a station on a given UTC date."""
    prefix = f"{date:%Y/%m/%d}/{station}/"
    client = _client()
    paginator = client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Skip the *_MDM metadata files, keep actual volume scan files
            if not key.endswith("_MDM"):
                keys.append(key)
    return sorted(keys)


def scan_time(key: str) -> datetime | None:
    """Volume start time from a scan key like .../KBGM20240615_183245_V06,
    or None if the name doesn't follow that pattern."""
    fname = key.split("/")[-1]
    parts = fname.split("_")
    try:
        return datetime.strptime(parts[0][-8:] + parts[1], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except (IndexError, ValueError):
        return None


def list_recent_scans(station: str) -> list[str]:
    """Scan keys from yesterday and today (UTC), oldest first — enough to
    cover the UTC midnight rollover."""
    now = datetime.now(timezone.utc)
    return list_available_scans(station, now - timedelta(days=1)) + list_available_scans(station, now)


def fetch_latest(station: str) -> str:
    """Download the most recent available scan for a station. Returns local path."""
    keys = list_recent_scans(station)
    if not keys:
        raise RuntimeError(f"No recent scans found for station {station}")
    return download(keys[-1])


def fetch_at(station: str, when: datetime) -> str:
    """Download the scan closest to a given UTC datetime. Returns local path."""
    keys = [k for k in list_available_scans(station, when) if scan_time(k)]
    if not keys:
        raise RuntimeError(f"No scans found for {station} on {when:%Y-%m-%d}")
    closest = min(keys, key=lambda k: abs((scan_time(k) - when).total_seconds()))
    return download(closest)


def fetch_range(station: str, start: datetime, end: datetime) -> list[str]:
    """Download all scans for a station between two UTC datetimes. Returns local paths."""
    paths = []
    day = start
    while day.date() <= end.date():
        for key in list_available_scans(station, day):
            ts = scan_time(key)
            if ts and start <= ts <= end:
                paths.append(download(key))
        day += timedelta(days=1)
    return paths


def download(key: str) -> str:
    """Download one scan into DATA_DIR (skipped if already there). Returns local path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    local_path = os.path.join(DATA_DIR, key.split("/")[-1])
    if not os.path.exists(local_path):
        _client().download_file(BUCKET, key, local_path)
    return local_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch NEXRAD Level II scans from NOAA's AWS bucket")
    parser.add_argument("station", help="4-letter station code, e.g. KBGM")
    parser.add_argument("--latest", action="store_true", help="Fetch the most recent scan")
    args = parser.parse_args()

    if args.latest:
        path = fetch_latest(args.station)
        print(f"Downloaded: {path}")
