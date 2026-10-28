"""
参考分析程序，生成本项目 outputs 与 gold。
可在任意目录运行：python solution/run_analysis.py
"""
from pathlib import Path
import argparse
import json
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "microalgae_growth.csv"
DEFAULT_OUT = ROOT / "outputs"
SEED = 20261028


def linear_quantile(series, q):
    return float(series.quantile(q, interpolation="linear"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input)
    required = ["culture_id", "temperature_c", "day", "od680", "measurement_date", "batch_id"]
    missing_cols = [c for c in required if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")
    for col in ("temperature_c", "day", "od680"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw_rows = len(raw)
    exact_dup_mask = raw.duplicated(keep="first")
    exact_dup_rows = int(exact_dup_mask.sum())
    unique = raw.loc[~exact_dup_mask].copy().reset_index(drop=True)
    key_duplicates = int(unique.duplicated(subset=["culture_id", "day"], keep=False).sum())
    missing_mask = unique["od680"].isna()
    negative_mask = unique["od680"].lt(0).fillna(False)
    valid_mask = (~missing_mask) & (~negative_mask)

    # IQR 旗标在每个温度×观测日分组内计算，且在完成完全重复、缺失和负值处理之后进行。
    # IQR 旗标仍保留在主汇总中，仅在敏感性统计列中单独排除。
    unique["iqr_outlier"] = False
    for _, idx in unique.loc[valid_mask].groupby(["temperature_c", "day"]).groups.items():
        vals = unique.loc[idx, "od680"]
        q1, q3 = linear_quantile(vals, 0.25), linear_quantile(vals, 0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        unique.loc[idx, "iqr_outlier"] = ((vals < lo) | (vals > hi)).to_numpy()

    # 主统计使用所有去重后的非缺失、非负观测值。IQR 旗标不会被自动删除；
    # 另外报告排除旗标后的敏感性统计。
    valid = unique.loc[valid_mask].copy()
    temps = sorted(unique["temperature_c"].dropna().unique().tolist())
    days = sorted(unique["day"].dropna().unique().tolist())
    summary_rows = []
    quality_flag_rows = []
    for _, r in unique.iterrows():
        flag_list = []
        if pd.isna(r["od680"]): flag_list.append("missing_od680")
        if pd.notna(r["od680"]) and r["od680"] < 0: flag_list.append("negative_od680")
        if bool(r["iqr_outlier"]): flag_list.append("iqr_outlier")
        if flag_list:
            for flag in flag_list:
                quality_flag_rows.append({
                    "culture_id": r["culture_id"], "temperature_c": int(r["temperature_c"]),
                    "day": int(r["day"]), "od680": "" if pd.isna(r["od680"]) else r["od680"],
                    "measurement_date": r["measurement_date"], "batch_id": r["batch_id"],
                    "flag_type": flag,
                })

    for temp in temps:
        for day in days:
            grp_all = unique[(unique.temperature_c == temp) & (unique.day == day)]
            grp_valid = grp_all.loc[grp_all.od680.notna() & (grp_all.od680 >= 0), "od680"].astype(float)
            grp_iqr_valid = grp_all.loc[
                grp_all.od680.notna() & (grp_all.od680 >= 0) & (~grp_all.iqr_outlier), "od680"
            ].astype(float)
            summary_rows.append({
                "temperature_c": int(temp), "day": int(day), "n_expected": 18,
                "n_records_after_dedup": int(len(grp_all)),
                "n_valid_primary": int(len(grp_valid)),
                "n_missing_od680": int(grp_all.od680.isna().sum()),
                "n_negative_od680": int((grp_all.od680 < 0).sum()),
                "n_iqr_flagged": int(grp_all.iqr_outlier.sum()),
                "mean_all_valid": float(grp_valid.mean()) if len(grp_valid) else np.nan,
                "sd_all_valid": float(grp_valid.std(ddof=1)) if len(grp_valid) >= 2 else np.nan,
                "mean_excluding_iqr": float(grp_iqr_valid.mean()) if len(grp_iqr_valid) else np.nan,
                "sd_excluding_iqr": float(grp_iqr_valid.std(ddof=1)) if len(grp_iqr_valid) >= 2 else np.nan,
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out / "summary.csv", index=False, float_format="%.8f")
    pd.DataFrame(quality_flag_rows, columns=[
        "culture_id", "temperature_c", "day", "od680", "measurement_date", "batch_id", "flag_type"
    ]).to_csv(args.out / "quality_flags.csv", index=False)

    iqr_n = int(unique.iqr_outlier.sum())
    n_unique_keys = int(unique[["culture_id", "day"]].drop_duplicates().shape[0])
    expected_n = 2 * 18 * 8
    quality_rows = [
        ("raw_rows", raw_rows, "原始CSV行数，包含重复记录"),
        ("expected_measurements", expected_n, "设计规模：2个温度×18个培养重复×8个观测日"),
        ("exact_duplicate_rows", exact_dup_rows, "完全重复的整行记录；分析时保留首条"),
        ("unique_rows_after_exact_dedup", len(unique), "删除完全重复行后的记录数"),
        ("unique_culture_day_keys", n_unique_keys, "培养编号×观测日的唯一组合数"),
        ("duplicate_culture_day_key_rows", key_duplicates, "去除完全重复行后，仍有重复的培养编号×观测日记录数"),
        ("missing_od680_rows", int(missing_mask.sum()), "OD680缺失；不插补，不参与数值统计"),
        ("negative_od680_rows", int(negative_mask.sum()), "OD680为负；作为物理无效值标记并排除统计"),
        ("iqr_flagged_rows", iqr_n, "按温度×观测日分组、线性四分位数和1.5×IQR规则标记；主分析保留"),
        ("valid_numeric_rows", int(valid_mask.sum()), "去重后同时满足非缺失与非负条件的记录数"),
    ]
    quality = pd.DataFrame(quality_rows, columns=["metric", "value", "interpretation"])
    quality.to_csv(args.out / "quality_report.csv", index=False)

    # 终点比较：使用第 7 天的主数据及双侧 Welch t 检验。
    endpoint = valid.loc[valid.day == max(days)].copy()
    g20 = endpoint.loc[endpoint.temperature_c == 20, "od680"].astype(float).to_numpy()
    g25 = endpoint.loc[endpoint.temperature_c == 25, "od680"].astype(float).to_numpy()
    t_res = stats.ttest_ind(g25, g20, equal_var=False, alternative="two-sided")
    m20, m25 = float(np.mean(g20)), float(np.mean(g25))
    s20, s25 = float(np.std(g20, ddof=1)), float(np.std(g25, ddof=1))
    v20, v25 = s20**2 / len(g20), s25**2 / len(g25)
    se_diff = np.sqrt(v20 + v25)
    df_welch = (v20 + v25)**2 / ((v20**2 / (len(g20)-1)) + (v25**2 / (len(g25)-1)))
    diff = m25 - m20
    ci_low, ci_high = diff - stats.t.ppf(0.975, df_welch) * se_diff, diff + stats.t.ppf(0.975, df_welch) * se_diff
    endpoint_row = {
        "endpoint_day": int(max(days)), "comparison": "25C minus 20C", "n_20C": len(g20), "mean_20C": m20,
        "sd_20C": s20, "n_25C": len(g25), "mean_25C": m25, "sd_25C": s25,
        "mean_difference_25_minus_20": diff, "welch_t": float(t_res.statistic), "welch_df": float(df_welch),
        "p_value_two_sided": float(t_res.pvalue), "ci95_low_difference": ci_low, "ci95_high_difference": ci_high,
        "note": "模拟数据；差异仅演示计算流程，不构成真实生物学证据",
    }
    endpoint_df = pd.DataFrame([endpoint_row])
    endpoint_df.to_csv(args.out / "endpoint_comparison.csv", index=False)

    # 论文风格的多面板图形。
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11,
        "axes.labelsize": 9.5, "legend.fontsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 220,
    })
    colors = {20: "#2378A5", 25: "#D97732"}
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.6), gridspec_kw={"width_ratios": [1.55, 1.1, 0.9]})
    ax = axes[0]
    for temp in temps:
        sub = unique[(unique.temperature_c == temp) & unique.od680.notna() & (unique.od680 >= 0)]
        # 用细线绘制每个重复的生长轨迹，为图形提供更真实的实验背景。
        for culture, cdf in sub.sort_values("day").groupby("culture_id"):
            ax.plot(cdf.day, cdf.od680, color=colors[int(temp)], alpha=0.12, linewidth=0.65, zorder=1)
        means = summary[summary.temperature_c == temp].sort_values("day")
        ax.errorbar(means.day, means.mean_all_valid, yerr=means.sd_all_valid, color=colors[int(temp)],
                    marker="o", markersize=4, capsize=3, linewidth=2, label=f"{int(temp)} °C · all valid", zorder=4)
        ax.plot(means.day, means.mean_excluding_iqr, color=colors[int(temp)], linestyle="--", linewidth=1.35,
                alpha=0.9, label=f"{int(temp)} °C · excluding IQR flags", zorder=3)
    flagged = unique.loc[unique.iqr_outlier]
    if not flagged.empty:
        ax.scatter(flagged.day, flagged.od680, marker="D", s=58, facecolors="none", edgecolors="#B91C1C",
                   linewidths=1.6, label="IQR-flagged", zorder=6)
    ax.set_title("A. Growth time course", loc="left", weight="bold")
    ax.set_xlabel("Culture day")
    ax.set_ylabel("Optical density at 680 nm (OD680)")
    ax.set_xticks(days)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.legend(frameon=False, ncol=1, loc="upper left")

    ax = axes[1]
    vals = [g20, g25]
    bp = ax.boxplot(vals, positions=[1, 2], widths=0.46, patch_artist=True, showfliers=False,
                    medianprops={"color": "#172554", "linewidth": 1.7},
                    boxprops={"linewidth": 1.3}, whiskerprops={"linewidth": 1.2}, capprops={"linewidth": 1.2})
    for patch, temp in zip(bp["boxes"], (20, 25)):
        patch.set_facecolor(colors[temp]); patch.set_alpha(0.20); patch.set_edgecolor(colors[temp])
    jitter_rng = np.random.default_rng(SEED + 1)
    for i, (data, temp) in enumerate(zip(vals, (20, 25)), start=1):
        jitter = jitter_rng.uniform(-0.12, 0.12, size=len(data))
        ax.scatter(np.full(len(data), i) + jitter, data, s=28, color=colors[temp], alpha=0.78,
                   edgecolors="white", linewidths=0.45, zorder=3)
    ax.set_xticks([1, 2], ["20 °C", "25 °C"])
    ax.set_ylabel("OD680 on day 7")
    ax.set_title("B. Endpoint comparison", loc="left", weight="bold")
    p_txt = "p < 0.001" if t_res.pvalue < 0.001 else f"p = {t_res.pvalue:.3f}"
    ax.text(0.5, 0.97, f"Welch's t-test, two-sided\n{p_txt}\nΔ mean = {diff:.3f} [{ci_low:.3f}, {ci_high:.3f}]",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.2,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.95})
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)

    ax = axes[2]
    qclabels = ["Missing\nOD680", "Exact\nduplicates", "Negative\nOD680", "IQR\nflags"]
    qcvalues = [int(missing_mask.sum()), exact_dup_rows, int(negative_mask.sum()), iqr_n]
    qccolors = ["#64748B", "#7C3AED", "#B91C1C", "#D97732"]
    bars = ax.bar(range(4), qcvalues, color=qccolors, width=0.66, alpha=0.88)
    ax.set_xticks(range(4), qclabels)
    ax.set_ylabel("Flagged rows / records")
    ax.set_title("C. Quality-control flags", loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.set_ylim(0, max(qcvalues + [1]) * 1.35)
    for bar, val in zip(bars, qcvalues):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.06, str(val), ha="center", va="bottom", weight="bold")

    fig.suptitle("Microalgae cultivation data analysis | SYNTHETIC DATA FOR AGENT EVALUATION",
                 x=0.04, y=1.02, ha="left", fontsize=13.5, weight="bold", color="#123047")
    fig.text(0.04, -0.012,
             "Primary summaries retain non-negative IQR-flagged observations; dashed curves show sensitivity summaries excluding them. Missing and negative values are excluded from numeric summaries.",
             fontsize=8.2, color="#475569")
    fig.tight_layout(rect=[0.02, 0.035, 1, 0.96], w_pad=2.1)
    fig.savefig(args.out / "comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 任务专用的 Markdown 报告。
    growth_txt = "".join([])
    report = f"""# 微藻培养数据分析报告（模拟数据）

> **重要声明：** 本项目的数据由程序模拟生成，仅用于演示科研工具 Agent 评测流程，不是真实湿实验记录，不能据此得出生物学结论。

## 1. 研究问题与设计

比较 20 °C 与 25 °C 两种培养条件下微藻的 OD680 生长轨迹。每个温度包含 18 个模拟培养重复，每个重复计划在第 0–7 天测量一次，共 2 × 18 × 8 = {expected_n} 个唯一测量组合。CSV 中额外混入了完全重复行、缺失值、一个负 OD680 值及 IQR 候选异常值，用于测试数据质量处理能力。

## 2. 数据质量与处理规则

- 原始文件：`data/microalgae_growth.csv`；共 {raw_rows} 行。
- 完全重复行：{exact_dup_rows} 行；分析时只保留首条完全相同记录。
- 去重后记录：{len(unique)} 行；培养编号×观测日的唯一组合数为 {n_unique_keys}。
- OD680 缺失：{int(missing_mask.sum())} 行，不进行插补，不参与数值统计。
- OD680 为负：{int(negative_mask.sum())} 行，视为物理无效值并排除数值统计。
- IQR 旗标：{iqr_n} 条。按温度×观测日分组，四分位数采用线性插值，超出 Q1−1.5×IQR 或 Q3+1.5×IQR 的记录标记为候选异常。为避免仅凭统计规则机械删值，主分析保留这些非负观测，同时在 `summary.csv` 中报告排除候选异常后的敏感性统计。
- 详细计数见 `quality_report.csv`；逐条标记见 `quality_flags.csv`。

## 3. 描述统计与终点比较

`summary.csv` 提供每个温度×观测日的有效样本数、均值、样本标准差和排除 IQR 旗标后的敏感性统计。均值与标准差基于去重后非缺失、非负的 OD680 值；标准差使用样本标准差（ddof=1）。

第 7 天 20 °C 组：n={len(g20)}，均值={m20:.4f}，SD={s20:.4f}；25 °C 组：n={len(g25)}，均值={m25:.4f}，SD={s25:.4f}。采用双侧 Welch t 检验，25 °C−20 °C 均值差={diff:.4f}，t={float(t_res.statistic):.3f}，df={df_welch:.2f}，p={float(t_res.pvalue):.6g}，差值 95% CI [{ci_low:.4f}, {ci_high:.4f}]。这些数值仅说明模拟数据的计算流程。

## 4. 图形说明

`comparison.png` 包含：（A）逐重复生长轨迹、均值±样本 SD、排除 IQR 旗标后的敏感性均值；（B）第 7 天两组终点分布与 Welch 检验；（C）主要质量标记计数。图中 IQR 候选异常以空心菱形标出。

## 5. 可复核性与局限

`verify.py` 使用独立实现重新读取原始 CSV，复算质量计数、每日组别的样本数/均值/样本标准差，并检查输出文件及图像可读性。执行：`python outputs/verify.py`。本例仅模拟两个温度、有限重复和单次培养数据；未建模真实实验中的批次效应、仪器校准、时间相关性或正式的重复测量模型，不应用于真实研究推断。
"""
    (args.out / "report.md").write_text(report, encoding="utf-8")

    # 将独立校验脚本写入输出目录。
    verifier_src = ROOT / "solution" / "verify_template.py"
    (args.out / "verify.py").write_text(verifier_src.read_text(encoding="utf-8"), encoding="utf-8")

    # 保存基准产物和关键指标，供隐藏评测使用。
    gold_dir = ROOT / "gold"
    gold_dir.mkdir(exist_ok=True)
    quality.to_csv(gold_dir / "expected_quality_report.csv", index=False)
    summary.to_csv(gold_dir / "expected_summary.csv", index=False, float_format="%.8f")
    endpoint_df.to_csv(gold_dir / "expected_endpoint_comparison.csv", index=False)
    metrics = {
        "raw_rows": raw_rows,
        "expected_measurements": expected_n,
        "exact_duplicate_rows": exact_dup_rows,
        "unique_rows_after_exact_dedup": int(len(unique)),
        "unique_culture_day_keys": n_unique_keys,
        "duplicate_culture_day_key_rows": key_duplicates,
        "missing_od680_rows": int(missing_mask.sum()),
        "negative_od680_rows": int(negative_mask.sum()),
        "iqr_flagged_rows": iqr_n,
        "valid_numeric_rows": int(valid_mask.sum()),
        "endpoint_day": int(max(days)),
        "n_20C_day7": int(len(g20)),
        "n_25C_day7": int(len(g25)),
        "mean_20C_day7": m20,
        "mean_25C_day7": m25,
        "welch_p_value_day7": float(t_res.pvalue),
    }
    (gold_dir / "expected_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
