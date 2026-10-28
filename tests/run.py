"""针对 Agent 任务输出的隐藏式验收测试。
在真实评测中，此测试框架必须对 Agent 保持隐藏。
"""
from pathlib import Path
import argparse
import csv
import json
import math
import subprocess
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["quality_report.csv", "quality_flags.csv", "summary.csv", "endpoint_comparison.csv",
            "comparison.png", "report.md", "verify.py"]


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument("--gold", type=Path, default=ROOT / "gold")
    args = ap.parse_args()
    checks = []
    def ck(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    for name in REQUIRED:
        p = args.outputs / name
        ck(f"required artifact {name}", p.is_file() and p.stat().st_size > 0,
           "present" if p.is_file() else "missing")

    expected = json.loads((args.gold / "expected_metrics.json").read_text(encoding="utf-8"))
    actual_qc = {r["metric"]: r["value"] for r in read_csv(args.outputs / "quality_report.csv")}
    for key in ["raw_rows", "expected_measurements", "exact_duplicate_rows", "unique_rows_after_exact_dedup",
                "unique_culture_day_keys", "duplicate_culture_day_key_rows", "missing_od680_rows",
                "negative_od680_rows", "iqr_flagged_rows", "valid_numeric_rows"]:
        ck(f"QC metric {key}", key in actual_qc and int(float(actual_qc[key])) == int(expected[key]),
           f"expected={expected[key]} actual={actual_qc.get(key)}")

    actual_summary = read_csv(args.outputs / "summary.csv")
    gold_summary = read_csv(args.gold / "expected_summary.csv")
    ck("summary row count", len(actual_summary) == len(gold_summary) == 16,
       f"actual={len(actual_summary)} expected={len(gold_summary)}")
    gmap = {(int(r["temperature_c"]), int(r["day"])): r for r in gold_summary}
    for a in actual_summary:
        k = (int(a["temperature_c"]), int(a["day"]))
        g = gmap.get(k)
        ck(f"summary group exists {k}", g is not None)
        if g:
            for field in ["n_valid_primary", "n_missing_od680", "n_negative_od680", "n_iqr_flagged"]:
                ck(f"summary {k} {field}", int(float(a[field])) == int(float(g[field])))
            for field in ["mean_all_valid", "sd_all_valid", "mean_excluding_iqr", "sd_excluding_iqr"]:
                av, gv = float(a[field]), float(g[field])
                ck(f"summary {k} {field}", math.isclose(av, gv, rel_tol=1e-5, abs_tol=1e-6),
                   f"actual={av:.8f}; expected={gv:.8f}")

    endpoint = read_csv(args.outputs / "endpoint_comparison.csv")
    gold_endpoint = read_csv(args.gold / "expected_endpoint_comparison.csv")
    ck("endpoint comparison row", len(endpoint) == len(gold_endpoint) == 1)
    if endpoint and gold_endpoint:
        for field in ["n_20C", "n_25C"]:
            ck(f"endpoint {field}", int(float(endpoint[0][field])) == int(float(gold_endpoint[0][field])))
        for field in ["mean_20C", "sd_20C", "mean_25C", "sd_25C", "welch_t", "p_value_two_sided",
                      "ci95_low_difference", "ci95_high_difference"]:
            ck(f"endpoint {field}", math.isclose(float(endpoint[0][field]), float(gold_endpoint[0][field]),
                                                  rel_tol=1e-5, abs_tol=1e-6))

    try:
        with Image.open(args.outputs / "comparison.png") as im:
            ck("PNG is readable and high resolution", im.format == "PNG" and im.width >= 1800 and im.height >= 600,
               f"{im.width}x{im.height}")
    except Exception as e:
        ck("PNG is readable", False, str(e))
    report = (args.outputs / "report.md").read_text(encoding="utf-8")
    ck("report explains IQR sensitivity", "敏感性" in report)

    verifier = args.outputs / "verify.py"
    if verifier.exists():
        proc = subprocess.run([sys.executable, str(verifier), "--outputs", str(args.outputs)],
                              capture_output=True, text=True, timeout=120)
        ck("independent verifier passes", proc.returncode == 0, proc.stdout[-1800:] if proc.returncode else "verify.py passed")
    else:
        ck("independent verifier passes", False, "verify.py missing")

    failures = [x for x in checks if not x[1]]
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'} | {name}" + (f" | {detail}" if detail else ""))
    print(f"\nAcceptance: {len(checks)-len(failures)}/{len(checks)} checks passed.")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
