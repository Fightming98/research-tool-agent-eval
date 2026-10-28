"""生成用于 Agent 评测的可复现微藻生长模拟数据集。
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "microalgae_growth.csv"
SEED = 20261028
rng = np.random.default_rng(SEED)
rows = []
start_date = pd.Timestamp("2026-04-01")

for temp in (20, 25):
    for rep in range(1, 19):
        culture_id = f"T{temp}_R{rep:02d}"
        batch_id = f"B{((rep - 1) // 6) + 1:02d}"
        capacity_mean = 0.66 if temp == 20 else 0.88
        capacity = capacity_mean * rng.normal(1.0, 0.045)
        rate_mean = 0.76 if temp == 20 else 0.83
        rate = max(0.62, rng.normal(rate_mean, 0.035))
        midpoint = rng.normal(3.55, 0.16)
        baseline = rng.normal(0.035, 0.004)
        sample_effect = rng.normal(0.0, 0.009)
        for day in range(8):
            logistic = 1.0 / (1.0 + np.exp(-rate * (day - midpoint)))
            od = baseline + capacity * logistic + sample_effect + rng.normal(0, 0.010 + day * 0.0008)
            od = max(0.005, float(od))
            rows.append({
                "culture_id": culture_id,
                "temperature_c": temp,
                "day": day,
                "od680": round(od, 4),
                "measurement_date": (start_date + pd.Timedelta(days=day)).date().isoformat(),
                "batch_id": batch_id,
            })

df = pd.DataFrame(rows)
# 预先设定的数据质量问题：4 个缺失测量值、1 个物理无效值，
# 以及 2 个 IQR 候选异常值。编号与天数固定，确保练习可重复。
missing_cases = [("T20_R04", 3), ("T25_R08", 5), ("T20_R16", 6), ("T25_R03", 1)]
for culture, day in missing_cases:
    df.loc[(df.culture_id == culture) & (df.day == day), "od680"] = np.nan

df.loc[(df.culture_id == "T20_R11") & (df.day == 4), "od680"] = -0.0180
df.loc[(df.culture_id == "T20_R07") & (df.day == 5), "od680"] = 1.2800
df.loc[(df.culture_id == "T25_R15") & (df.day == 2), "od680"] = 0.0180

# 添加 4 条完全重复的记录，使质量审计更有意义。
duplicate_keys = [("T20_R01", 2), ("T20_R10", 7), ("T25_R05", 3), ("T25_R18", 6)]
dups = []
for culture, day in duplicate_keys:
    dup = df.loc[(df.culture_id == culture) & (df.day == day)].iloc[0].copy()
    dups.append(dup)
df = pd.concat([df, pd.DataFrame(dups)], ignore_index=True)
# 在固定随机状态下打乱顺序，使输入不会预先按组排好给 Agent。
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False, na_rep="")
print(f"Wrote {OUT} ({len(df)} rows; simulated data).")
