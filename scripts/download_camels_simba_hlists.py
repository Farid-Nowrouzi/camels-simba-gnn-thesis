#!/usr/bin/env python3
from __future__ import annotations

"""
download_camels_simba_hlists.py

Fast, reproducible CAMELS-SIMBA Rockstar hlist downloader.

This script downloads selected CAMELS-SIMBA Rockstar hlist snapshot files
directly from the Flatiron public data server into the thesis raw-data folder.

Remote source pattern:
    https://users.flatironinstitute.org/~camels/Rockstar/SIMBA/L25n256/LH/LH_i/hlists/hlist_snapshot.list

Local saved pattern:
    LH_i_hlist_snapshot.list

Example:
    Remote:
        .../LH_114/hlists/hlist_1.00000.list

    Local:
        data/raw/CAMELS_SIMBA_200U/LH_114_hlist_1.00000.list

Main recommended usage:
    python scripts/download_camels_simba_hlists.py \
      --start 114 \
      --end 199 \
      --output_dir data/raw/CAMELS_SIMBA_200U \
      --method aria2c \
      --parallel 10

Dry run:
    python scripts/download_camels_simba_hlists.py \
      --start 114 \
      --end 119 \
      --output_dir data/raw/CAMELS_SIMBA_200U \
      --dry_run

Verify only:
    python scripts/download_camels_simba_hlists.py \
      --start 0 \
      --end 199 \
      --output_dir data/raw/CAMELS_SIMBA_200U \
      --verify_only

Why aria2c?
-----------
aria2c is much faster than Python urlopen for this dataset because it can
download multiple files concurrently and resume partial downloads.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "https://users.flatironinstitute.org/~camels/Rockstar/SIMBA/L25n256/LH"

DEFAULT_SNAPSHOTS = [
    "0.20000",
    "0.25000",
    "0.51209",
    "0.75065",
    "1.00000",
]


# ============================================================
# Path / URL helpers
# ============================================================

def build_url(universe_id: int, snapshot: str) -> str:
    """
    Build remote URL for one CAMELS-SIMBA hlist file.
    """
    return f"{BASE_URL}/LH_{universe_id}/hlists/hlist_{snapshot}.list"


def build_output_path(output_dir: Path, universe_id: int, snapshot: str) -> Path:
    """
    Build renamed local thesis-style output path.
    """
    return output_dir / f"LH_{universe_id}_hlist_{snapshot}.list"


def expected_file_records(
    output_dir: Path,
    start: int,
    end: int,
    snapshots: List[str],
) -> List[Tuple[int, str, str, Path]]:
    """
    Build all expected download records.

    Returns list of:
        (universe_id, snapshot, url, output_path)
    """
    records = []

    for universe_id in range(start, end + 1):
        for snapshot in snapshots:
            url = build_url(universe_id, snapshot)
            output_path = build_output_path(output_dir, universe_id, snapshot)
            records.append((universe_id, snapshot, url, output_path))

    return records


# ============================================================
# Verification helpers
# ============================================================

def verify_expected_files(
    output_dir: Path,
    start: int,
    end: int,
    snapshots: List[str],
    min_size_bytes: int = 1,
) -> Dict[str, object]:
    """
    Verify that all expected renamed files exist locally.

    min_size_bytes:
        If file exists but is smaller than this, treat it as suspicious.
    """
    expected_paths = []
    missing_paths = []
    suspicious_paths = []

    for universe_id in range(start, end + 1):
        for snapshot in snapshots:
            path = build_output_path(output_dir, universe_id, snapshot)
            expected_paths.append(path)

            if not path.exists():
                missing_paths.append(path)
            else:
                size = path.stat().st_size
                if size < min_size_bytes:
                    suspicious_paths.append(path)

    found_count = len(expected_paths) - len(missing_paths)

    return {
        "expected_count": len(expected_paths),
        "found_count": found_count,
        "missing_count": len(missing_paths),
        "suspicious_count": len(suspicious_paths),
        "missing": missing_paths,
        "suspicious": suspicious_paths,
    }


def print_verification_report(report: Dict[str, object]) -> None:
    """
    Print verification summary.
    """
    print()
    print("LOCAL FILE VERIFICATION")
    print("-" * 90)
    print(f"Expected files:     {report['expected_count']}")
    print(f"Found files:        {report['found_count']}")
    print(f"Missing files:      {report['missing_count']}")
    print(f"Suspicious files:   {report['suspicious_count']}")

    missing = report["missing"]
    suspicious = report["suspicious"]

    if missing:
        print()
        print("First missing files:")
        for path in missing[:50]:
            print(f"  - {path}")
        if len(missing) > 50:
            print(f"  ... and {len(missing) - 50} more")

    if suspicious:
        print()
        print("Suspicious tiny files:")
        for path in suspicious[:50]:
            print(f"  - {path}")
        if len(suspicious) > 50:
            print(f"  ... and {len(suspicious) - 50} more")

    if report["missing_count"] == 0 and report["suspicious_count"] == 0:
        print()
        print("✅ All expected files are present.")
    else:
        print()
        print("⚠️ Some files are missing or suspicious. Re-run the same command.")


# ============================================================
# Python fallback downloader
# ============================================================

def download_file_python(
    url: str,
    output_path: Path,
    overwrite: bool = False,
    timeout: int = 120,
    retries: int = 3,
    sleep_seconds: float = 1.0,
) -> bool:
    """
    Download one file using Python urlopen.

    This is slower than aria2c but useful as fallback.
    """
    if output_path.exists() and not overwrite:
        print(f"[SKIP] Exists: {output_path}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    for attempt in range(1, retries + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 CAMELS-SIMBA thesis downloader"
                },
            )

            print(f"[GET]  {url}")

            with urlopen(request, timeout=timeout) as response:
                data = response.read()

            if len(data) == 0:
                print(f"[FAIL] Empty file downloaded: {url}")
                return False

            with temp_path.open("wb") as f:
                f.write(data)

            temp_path.rename(output_path)

            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"[OK]   Saved: {output_path} ({size_mb:.2f} MB)")
            return True

        except HTTPError as exc:
            print(
                f"[WARN] HTTP error attempt {attempt}/{retries}: "
                f"{exc.code} for {url}"
            )

        except URLError as exc:
            print(
                f"[WARN] URL/network error attempt {attempt}/{retries}: "
                f"{exc} for {url}"
            )

        except TimeoutError:
            print(f"[WARN] Timeout attempt {attempt}/{retries}: {url}")

        except Exception as exc:
            print(
                f"[WARN] Unexpected error attempt {attempt}/{retries}: "
                f"{exc} for {url}"
            )

        if attempt < retries:
            time.sleep(sleep_seconds)

    print(f"[FAIL] Could not download after {retries} attempts: {url}")
    return False


def run_python_downloader(
    records: List[Tuple[int, str, str, Path]],
    overwrite: bool,
    retries: int,
    sleep_seconds: float,
) -> Tuple[int, int]:
    """
    Sequential Python fallback downloader.
    """
    success_count = 0
    fail_count = 0

    for universe_id, snapshot, url, output_path in records:
        print()
        print(f"Universe LH_{universe_id} | snapshot {snapshot}")

        ok = download_file_python(
            url=url,
            output_path=output_path,
            overwrite=overwrite,
            retries=retries,
            sleep_seconds=sleep_seconds,
        )

        if ok:
            success_count += 1
        else:
            fail_count += 1

        time.sleep(sleep_seconds)

    return success_count, fail_count


# ============================================================
# aria2c downloader
# ============================================================

def check_aria2c_available() -> bool:
    """
    Check if aria2c is installed.
    """
    return shutil.which("aria2c") is not None


def write_aria2c_input_file(
    records: List[Tuple[int, str, str, Path]],
    input_file: Path,
    overwrite: bool = False,
) -> int:
    """
    Write aria2c input file.

    aria2c input file format:
        URL
          out=filename
          dir=folder

    Returns:
        Number of files included in aria2c job.
    """
    input_file.parent.mkdir(parents=True, exist_ok=True)

    included = 0

    with input_file.open("w", encoding="utf-8") as f:
        for _, _, url, output_path in records:
            if output_path.exists() and not overwrite:
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)

            f.write(f"{url}\n")
            f.write(f"  dir={output_path.parent}\n")
            f.write(f"  out={output_path.name}\n")

            included += 1

    return included


def run_aria2c_downloader(
    records: List[Tuple[int, str, str, Path]],
    output_dir: Path,
    overwrite: bool,
    parallel: int,
    split: int,
    retries: int,
    timeout: int,
) -> Tuple[int, int]:
    """
    Download files using aria2c.

    This is the recommended fast mode.
    """
    if not check_aria2c_available():
        raise RuntimeError(
            "aria2c is not installed. Install it with:\n"
            "  sudo apt update && sudo apt install -y aria2\n"
            "or use --method python."
        )

    input_file = output_dir / "_aria2c_download_list.txt"

    included_count = write_aria2c_input_file(
        records=records,
        input_file=input_file,
        overwrite=overwrite,
    )

    skipped_count = len(records) - included_count

    print()
    print("ARIA2C DOWNLOAD PLAN")
    print("-" * 90)
    print(f"Total expected files:     {len(records)}")
    print(f"Already existing/skipped: {skipped_count}")
    print(f"To download:              {included_count}")
    print(f"aria2c input file:        {input_file}")

    if included_count == 0:
        print()
        print("✅ Nothing to download. All requested files already exist.")
        return len(records), 0

    command = [
        "aria2c",
        "--input-file", str(input_file),
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=true" if overwrite else "--allow-overwrite=false",
        "--max-concurrent-downloads", str(parallel),
        "--split", str(split),
        "--max-connection-per-server", str(split),
        "--min-split-size", "1M",
        "--retry-wait", "2",
        "--max-tries", str(retries),
        "--timeout", str(timeout),
        "--connect-timeout", str(timeout),
        "--summary-interval", "10",
        "--console-log-level", "notice",
    ]

    print()
    print("Running aria2c:")
    print(" ".join(command))
    print("-" * 90)

    result = subprocess.run(command)

    if result.returncode != 0:
        print()
        print(f"⚠️ aria2c exited with return code {result.returncode}.")
        print("Some files may still have downloaded. Verification will decide.")

    # Final truth comes from local file verification, not aria2c logs.
    return 0, 0


# ============================================================
# Reporting helpers
# ============================================================

def print_header(args, output_dir: Path, snapshots: List[str]) -> None:
    """
    Print run configuration.
    """
    print("=" * 90)
    print("CAMELS-SIMBA HLIST DOWNLOADER")
    print("=" * 90)
    print(f"Base URL:        {BASE_URL}")
    print(f"Universe range:  LH_{args.start} to LH_{args.end}")
    print(f"Snapshots:       {snapshots}")
    print(f"Output folder:   {output_dir}")
    print(f"Method:          {args.method}")
    print(f"Overwrite:       {args.overwrite}")
    print(f"Dry run:         {args.dry_run}")
    print(f"Verify only:     {args.verify_only}")
    print(f"Parallel:        {args.parallel}")
    print(f"Split:           {args.split}")
    print("=" * 90)


def print_dry_run(records: List[Tuple[int, str, str, Path]]) -> None:
    """
    Print what would be downloaded.
    """
    for universe_id, snapshot, url, output_path in records:
        print(f"[DRY] LH_{universe_id} snapshot={snapshot}")
        print(f"      {url}")
        print(f"   -> {output_path}")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download selected CAMELS-SIMBA hlist snapshots."
    )

    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="First LH universe ID, inclusive. Example: 114",
    )

    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="Last LH universe ID, inclusive. Example: 199",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Folder where renamed files will be saved.",
    )

    parser.add_argument(
        "--snapshots",
        nargs="+",
        default=DEFAULT_SNAPSHOTS,
        help="Snapshot values to download.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="aria2c",
        choices=["aria2c", "python"],
        help="Download backend. Recommended: aria2c.",
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=10,
        help="Number of concurrent downloads for aria2c.",
    )

    parser.add_argument(
        "--split",
        type=int,
        default=4,
        help="Connections per file for aria2c.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files if they already exist.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print what would be downloaded.",
    )

    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify expected files; do not download.",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Number of retries per file.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Network timeout in seconds.",
    )

    parser.add_argument(
        "--sleep_seconds",
        type=float,
        default=0.0,
        help="Pause between Python downloads. Ignored by aria2c.",
    )

    parser.add_argument(
        "--min_size_bytes",
        type=int,
        default=1024,
        help="Minimum acceptable file size for verification.",
    )

    args = parser.parse_args()

    if args.start > args.end:
        raise ValueError("--start must be <= --end")

    if args.parallel < 1:
        raise ValueError("--parallel must be >= 1")

    if args.split < 1:
        raise ValueError("--split must be >= 1")

    output_dir = Path(args.output_dir)
    snapshots = list(args.snapshots)

    records = expected_file_records(
        output_dir=output_dir,
        start=args.start,
        end=args.end,
        snapshots=snapshots,
    )

    print_header(args=args, output_dir=output_dir, snapshots=snapshots)

    total_files = len(records)

    if args.dry_run:
        print_dry_run(records)
        print()
        print("=" * 90)
        print("DRY RUN COMPLETE")
        print("=" * 90)
        print(f"Files that would be requested: {total_files}")
        return

    if args.verify_only:
        report = verify_expected_files(
            output_dir=output_dir,
            start=args.start,
            end=args.end,
            snapshots=snapshots,
            min_size_bytes=args.min_size_bytes,
        )
        print_verification_report(report)
        print("=" * 90)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    if args.method == "aria2c":
        run_aria2c_downloader(
            records=records,
            output_dir=output_dir,
            overwrite=args.overwrite,
            parallel=args.parallel,
            split=args.split,
            retries=args.retries,
            timeout=args.timeout,
        )
    else:
        run_python_downloader(
            records=records,
            overwrite=args.overwrite,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
        )

    elapsed = time.time() - start_time

    report = verify_expected_files(
        output_dir=output_dir,
        start=args.start,
        end=args.end,
        snapshots=snapshots,
        min_size_bytes=args.min_size_bytes,
    )

    print()
    print("=" * 90)
    print("DOWNLOAD SUMMARY")
    print("=" * 90)
    print(f"Requested files:      {total_files}")
    print(f"Elapsed seconds:      {elapsed:.2f}")
    print(f"Elapsed minutes:      {elapsed / 60.0:.2f}")

    print_verification_report(report)

    print("=" * 90)

    if report["missing_count"] != 0 or report["suspicious_count"] != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()