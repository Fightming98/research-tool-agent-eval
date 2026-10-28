"""用于独立校验 Agent 模拟数据分析输出的校验脚本。
使用 Python 的 csv/statistics 模块以及独立实现的分位数算法。
这是任务预期由 Agent 产出的产物，而非隐藏评分器。
"""
from pathlib import Path
import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict, Counter


def quantile_linear(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    return float(values[lo] + (values[hi] - values[lo]) * (pos - lo))


def close(a, b, tol=2e-6):
    return abs(float(a) - float(b)) <= tol


def main():
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    root = here.parent
    parser.add_argument("--input", type=Path, default=root / "data" / "microalgae_growth.csv")
    parser.add_argument("--outputs", type=Path, default=here)
    args = parser.parse_args()

    with args.input.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"culture_id", "temperature_c", "day", "od680", "measurement_date", "batch_id"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise AssertionError(f"Missing required columns: {required - set(reader.fieldnames or [])}")
        raw = list(reader)
    raw_rows = len(raw)

    # 独立于分析脚本检测完全重复的整行记录。
    seen = set(); unique = []; duplicate_count = 0
    for row in raw:
        key = tuple((k, row.get(k, "")) for k in sorted(row.keys()))
        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key); unique.append(row)

    missing_count = 0; negative_count = 0; group_values = defaultdict(list); group_all = defaultdict(list)
    key_seen = set(); key_duplicate_count = 0
    for row in unique:
        temp = int(float(row["temperature_c"]))
        day = int(float(row["day"]))
        record_key = (row["culture_id"], day)
        if record_key in key_seen:
            key_duplicate_count += 1
        key_seen.add(record_key)
        raw_v = row.get("od680", "").strip()
        if raw_v == "":
            missing_count += 1
            group_all[(temp, day)].append(None)
            continue
        val = float(raw_v)
        if val < 0:
            negative_count += 1
            group_all[(temp, day)].append(val)
            continue
        group_values[(temp, day)].append(val)
        group_all[(temp, day)].append(val)

    iqr_flagged = set()
    for key, values in group_values.items():
        q1 = quantile_linear(values, 0.25); q3 = quantile_linear(values, 0.75); iqr = q3 - q1
        low = q1 - 1.5 * iqr; high = q3 + 1.5 * iqr
        # 利用匹配的有序行识别该键值下的记录。重复行已提前删除。
        temp, day = key
        for row in unique:
            if int(float(row["temperature_c"])) != temp or int(float(row["day"])) != day:
                continue
            raw_v = row.get("od680", "").strip()
            if not raw_v:
                continue
            val = float(raw_v)
            if val >= 0 and (val < low or val > high):
                iqr_flagged.add((row["culture_id"], day))

    checks = []
    def check(name, passed, detail=""):
        checks.append((name, bool(passed), detail))

    check("input raw row count", raw_rows == 292, f"observed={raw_rows}")
    check("exact duplicate rows", duplicate_count == 4, f"observed={duplicate_count}")
    check("unique rows after dedup", len(unique) == 288, f"observed={len(unique)}")
    check("missing OD680", missing_count == 4, f"observed={missing_count}")
    check("negative OD680", negative_count == 1, f"observed={negative_count}")
    check("unique culture-day keys", len(key_seen) == 288 and key_duplicate_count == 0,
          f"unique={len(key_seen)}, remaining_duplicate_rows={key_duplicate_count}")

    quality_path = args.outputs / "quality_report.csv"
    with quality_path.open("r", newline="", encoding="utf-8-sig") as f:
        qr = {r["metric"]: r["value"] for r in csv.DictReader(f)}
    expected_qc = {
        "raw_rows": raw_rows, "expected_measurements": 288, "exact_duplicate_rows": duplicate_count,
        "unique_rows_after_exact_dedup": len(unique), "unique_culture_day_keys": len(key_seen),
        "duplicate_culture_day_key_rows": key_duplicate_count, "missing_od680_rows": missing_count,
        "negative_od680_rows": negative_count, "iqr_flagged_rows": len(iqr_flagged),
        "valid_numeric_rows": sum(len(v) for v in group_values.values()),
    }
    for metric, expected in expected_qc.items():
        check(f"quality_report:{metric}", metric in qr and int(float(qr[metric])) == int(expected),
              f"expected={expected}; observed={qr.get(metric, 'MISSING')}")

    summary_path = args.outputs / "summary.csv"
    with summary_path.open("r", newline="", encoding="utf-8-sig") as f:
        summary_rows = list(csv.DictReader(f))
    check("summary has 16 temperature-day rows", len(summary_rows) == 16, f"observed={len(summary_rows)}")
    summary_by_key = {(int(r["temperature_c"]), int(r["day"])): r for r in summary_rows}
    for key, values in group_values.items():
        rec = summary_by_key.get(key)
        if rec is None:
            check(f"summary group {key} exists", False, "missing group")
            continue
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) >= 2 else float("nan")
        check(f"summary n {key}", int(float(rec["n_valid_primary"])) == len(values),
              f"expected={len(values)}, observed={rec['n_valid_primary']}")
        check(f"summary mean {key}", close(rec["mean_all_valid"], mean),
              f"expected={mean:.8f}, observed={rec['mean_all_valid']}")
        check(f"summary sample sd {key}", close(rec["sd_all_valid"], sd),
              f"expected={sd:.8f}, observed={rec['sd_all_valid']}")

    required_files = ["quality_report.csv", "quality_flags.csv", "summary.csv", "endpoint_comparison.csv",
                      "comparison.png", "report.md", "verify.py"]
    for name in required_files:
        p = args.outputs / name
        check(f"artifact exists: {name}", p.is_file() and p.stat().st_size > 0,
              f"exists={p.exists()}, size={p.stat().st_size if p.exists() else 0}")

    # 仅使用 PNG 容器签名和尺寸检查图形的可读性与大小。
    try:
        from PIL import Image
        with Image.open(args.outputs / "comparison.png") as im:
            width, height = im.size
            check("comparison plot dimensions", width >= 1800 and height >= 600, f"{width}x{height}")
    except Exception as exc:
        check("comparison plot opens", False, str(exc))

    failures = [c for c in checks if not c[1]]
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'} | {name}" + (f" | {detail}" if detail else ""))
    print(f"\nResult: {len(checks)-len(failures)}/{len(checks)} checks passed.")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
