#!/usr/bin/env python3
"""Audit valid halo counts for all U1000 five-snapshot raw catalogues."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


SNAPSHOTS = ("0.20000", "0.25000", "0.51209", "0.75065", "1.00000")
REQUIRED_ZERO_BASED_COLUMNS = (10, 17, 18, 19, 20, 21, 22)
TOP_N = 1500


def inspect_one(item: tuple[int, str, Path]) -> dict[str, object]:
    universe, snapshot, path = item
    try:
        valid = 0
        raw = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                raw += 1
                fields = line.split()
                if len(fields) <= max(REQUIRED_ZERO_BASED_COLUMNS):
                    continue
                try:
                    values = [float(fields[index]) for index in REQUIRED_ZERO_BASED_COLUMNS]
                except ValueError:
                    continue
                if all(math.isfinite(value) for value in values) and values[0] > 0:
                    valid += 1
        return {"universe_id": f"LH_{universe}", "universe_index": universe,
                "snapshot": snapshot, "path": str(path), "raw_halo_count": raw,
                "valid_real_halo_count": valid, "status": "PASS", "error": ""}
    except Exception as exc:
        return {"universe_id": f"LH_{universe}", "universe_index": universe,
                "snapshot": snapshot, "path": str(path), "raw_halo_count": "",
                "valid_real_halo_count": "", "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}"}


def describe(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    result: dict[str, float] = {
        "count": len(values), "min": float(array.min()), "max": float(array.max()),
        "mean": float(array.mean()), "population_std": float(array.std(ddof=0)),
        "sample_std": float(array.std(ddof=1)),
    }
    for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        result[f"p{percentile}"] = float(np.percentile(array, percentile))
    return result


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--raw-dir", type=Path, default=root / "data/raw/CAMELS_SIMBA_1000U")
    parser.add_argument("--report-dir", type=Path, default=root / "reports/experiment_registry")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    items = [(universe, snapshot, args.raw_dir / f"LH_{universe}_hlist_{snapshot}.list")
             for universe in range(1000) for snapshot in SNAPSHOTS]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(inspect_one, items, chunksize=10))
    successes = [row for row in rows if row["status"] == "PASS"]
    failures = [row for row in rows if row["status"] != "PASS"]
    for row in successes:
        count = int(row["valid_real_halo_count"])
        row["at_least_1500"] = count >= TOP_N
        row["padded_slots_top1500"] = max(0, TOP_N - count)
        row["padded_slot_percentage"] = 100.0 * int(row["padded_slots_top1500"]) / TOP_N
    fields = ["universe_id", "universe_index", "snapshot", "path", "status", "error",
              "raw_halo_count", "valid_real_halo_count", "at_least_1500",
              "padded_slots_top1500", "padded_slot_percentage"]
    write_csv(args.report_dir / "u1000_top1500_raw_halo_count_distribution.csv", rows, fields)
    if failures:
        decision = "NO-GO"
        counts: list[int] = []
    else:
        counts = [int(row["valid_real_halo_count"]) for row in successes]
        padded_fraction = sum(int(row["padded_slots_top1500"]) for row in successes) / (5000 * TOP_N)
        decision = "GO" if padded_fraction <= 0.10 else "CONDITIONAL GO" if padded_fraction <= 0.30 else "NO-GO"
    by_snapshot: dict[str, list[int]] = defaultdict(list)
    for row in successes:
        by_snapshot[str(row["snapshot"])].append(int(row["valid_real_halo_count"]))
    summary_rows = []
    groups = [("ALL", counts)] + [(snapshot, by_snapshot[snapshot]) for snapshot in SNAPSHOTS]
    for label, values in groups:
        if not values:
            continue
        stats = describe(values)
        matching = [row for row in successes if label == "ALL" or row["snapshot"] == label]
        stats.update({
            "scope": label,
            "at_least_1500_count": sum(int(row["at_least_1500"]) for row in matching),
            "requires_padding_count": sum(not bool(row["at_least_1500"]) for row in matching),
            "padded_slots": sum(int(row["padded_slots_top1500"]) for row in matching),
            "padded_slot_percentage": 100.0 * sum(int(row["padded_slots_top1500"]) for row in matching)
                                      / (len(matching) * TOP_N),
        })
        summary_rows.append(stats)
    summary_fields = ["scope", "count", "min", "max", "mean", "population_std", "sample_std",
                      "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99",
                      "at_least_1500_count", "requires_padding_count", "padded_slots",
                      "padded_slot_percentage"]
    write_csv(args.report_dir / "u1000_top1500_padding_summary.csv", summary_rows, summary_fields)
    affected: dict[str, dict[str, int]] = defaultdict(lambda: {"padded_slots": 0, "padded_snapshots": 0,
                                                               "minimum_real_halos": 10**9})
    for row in successes:
        if int(row["padded_slots_top1500"]):
            item = affected[str(row["universe_id"])]
            item["padded_slots"] += int(row["padded_slots_top1500"])
            item["padded_snapshots"] += 1
            item["minimum_real_halos"] = min(item["minimum_real_halos"], int(row["valid_real_halo_count"]))
    worst_universes = sorted(affected.items(), key=lambda item: (-item[1]["padded_slots"], item[0]))[:10]
    worst_snapshots = sorted(successes, key=lambda row: (int(row["valid_real_halo_count"]),
                                                          int(row["universe_index"]), str(row["snapshot"])))[:10]
    overall = summary_rows[0] if summary_rows else {}
    lines = [
        "# U1000 Top1500 raw halo-count audit", "",
        "This read-only audit inspected each raw catalogue with the production builder's validity rule: "
        "finite `Mvir`, `X/Y/Z`, and `VX/VY/VZ`, with `Mvir > 0`.", "",
        f"**Decision: {decision}.**", "",
        f"- Expected catalogues: {len(items)}", f"- Successfully inspected: {len(successes)}",
        f"- Failed catalogues: {len(failures)}",
    ]
    if overall:
        lines += [
            f"- Valid real halos: min {overall['min']:.0f}, max {overall['max']:.0f}, mean {overall['mean']:.3f}",
            f"- Population standard deviation: {overall['population_std']:.3f}",
            f"- Sample standard deviation: {overall['sample_std']:.3f}",
            "- Percentiles (1/5/10/25/50/75/90/95/99): " + ", ".join(
                f"{overall[f'p{p}']:.3f}" for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)),
            f"- Snapshots with at least 1,500 real halos: {overall['at_least_1500_count']}",
            f"- Snapshots requiring padding: {overall['requires_padding_count']}",
            f"- Total padded slots: {overall['padded_slots']}",
            f"- Padded-slot percentage: {overall['padded_slot_percentage']:.6f}%",
            f"- Universes with at least one padded snapshot: {len(affected)}", "",
            "## Snapshot-specific statistics", "",
            "| Snapshot | Min | Max | Mean | Population SD | Sample SD | >=1500 | Padded snapshots | Padded slots | Padded % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in summary_rows[1:]:
            lines.append(f"| {item['scope']} | {item['min']:.0f} | {item['max']:.0f} | {item['mean']:.3f} | "
                         f"{item['population_std']:.3f} | {item['sample_std']:.3f} | "
                         f"{item['at_least_1500_count']} | {item['requires_padding_count']} | "
                         f"{item['padded_slots']} | {item['padded_slot_percentage']:.6f}% |")
        lines += ["", "## Worst affected universes", "",
                  "| Universe | Padded snapshots | Total padded slots | Minimum real halos |",
                  "|---|---:|---:|---:|"]
        for universe, item in worst_universes:
            lines.append(f"| {universe} | {item['padded_snapshots']} | {item['padded_slots']} | {item['minimum_real_halos']} |")
        lines += ["", "## Lowest-count snapshots", "",
                  "| Universe | Snapshot | Valid real halos | Padded slots |", "|---|---:|---:|---:|"]
        for row in worst_snapshots:
            lines.append(f"| {row['universe_id']} | {row['snapshot']} | {row['valid_real_halo_count']} | {row['padded_slots_top1500']} |")
    (args.report_dir / "u1000_top1500_raw_halo_count_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{decision}: inspected={len(successes)}/5000 failed={len(failures)} "
          f"padding={overall.get('padded_slot_percentage', float('nan')):.6f}%")
    return 1 if decision == "NO-GO" else 0


if __name__ == "__main__":
    raise SystemExit(main())
