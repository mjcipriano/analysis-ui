"""Generate the bundled sample datasets (deterministic)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "backend" / "analysis_ui" / "sample_data"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(20240607)
index = []


def save(id, name, description, df, hint):
    df.to_csv(OUT / f"{id}.csv", index=False)
    index.append({"id": id, "name": name, "description": description, "file": f"{id}.csv", "hint": hint})


# 1 column table
n = 12
save("treatment_groups", "Treatment groups (column table)",
     "Four independent groups, one column each — t tests, one-way ANOVA, column graphs.",
     pd.DataFrame({"Control": rng.normal(100, 12, n).round(1), "Drug A": rng.normal(88, 13, n).round(1),
                   "Drug B": rng.normal(71, 11, n).round(1), "Drug C": rng.normal(96, 15, n).round(1)}),
     "column")
# 2 paired
before = rng.normal(142, 14, 15)
save("paired_bp", "Blood pressure before / after (paired)",
     "The same 15 patients measured twice — paired t test, Wilcoxon, before–after graph.",
     pd.DataFrame({"Patient": [f"P{i+1:02d}" for i in range(15)], "Before": before.round(0),
                   "After": (before - rng.normal(9, 7, 15)).round(0)}), "paired")
# 3 two-way long
rows = []
for sex in ("Female", "Male"):
    for trt, eff in (("Vehicle", 0), ("Low dose", 6), ("High dose", 15)):
        for i in range(8):
            rows.append({"Response": round(50 + eff + (4 if sex == "Male" else 0) + (5 if (sex == "Male" and trt == "High dose") else 0) + rng.normal(0, 6), 1),
                         "Treatment": trt, "Sex": sex})
save("two_way", "Treatment × sex (two-way)", "Long-format data with two factors — two-way ANOVA, grouped graphs.",
     pd.DataFrame(rows), "grouped")
# 4 dose response
conc = np.repeat([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], 3)
def dr(c, ec50, hill, top=100, bottom=3):
    return bottom + (top - bottom) / (1 + (ec50 / c) ** hill)
d = pd.concat([
    pd.DataFrame({"Concentration (µM)": conc, "Response (%)": (dr(conc, 0.12, 1.1) + rng.normal(0, 4, len(conc))).round(1), "Compound": "Compound A"}),
    pd.DataFrame({"Concentration (µM)": conc, "Response (%)": (dr(conc, 1.4, 0.9, top=92) + rng.normal(0, 4, len(conc))).round(1), "Compound": "Compound B"})])
save("dose_response", "Dose–response (two compounds)", "Triplicate responses at ten concentrations — 4PL curve fit, EC50.",
     d, "xy")
# 5 survival
rows = []
for arm, scale in (("Control", 14), ("Treatment", 24)):
    for i in range(40):
        t = rng.weibull(1.3) * scale
        c = rng.uniform(6, 40)
        rows.append({"Months": round(min(t, c), 1), "Status": int(t <= c), "Arm": arm})
save("survival", "Survival (two arms)", "Time-to-event data with censoring — Kaplan–Meier, log-rank, Cox regression.",
     pd.DataFrame(rows), "survival")
# 6 regression
age = rng.uniform(20, 75, 60)
sex = rng.choice(["Female", "Male"], 60)
bmi = rng.normal(26, 4, 60)
sbp = 95 + 0.55 * age + 1.1 * (bmi - 26) + np.where(sex == "Male", 5, 0) + rng.normal(0, 9, 60)
save("regression", "Blood pressure, age and BMI (regression)",
     "Continuous outcome with several predictors — correlation, linear and multiple regression.",
     pd.DataFrame({"Age": age.round(0), "BMI": bmi.round(1), "Sex": sex, "Systolic BP": sbp.round(0),
                   "Hypertension": np.where(sbp > 135, "Yes", "No")}), "xy")
# 7 contingency
trt = np.repeat(["Placebo", "Drug"], [60, 60])
resp = np.concatenate([rng.choice(["Responder", "Non-responder"], 60, p=[0.25, 0.75]),
                       rng.choice(["Responder", "Non-responder"], 60, p=[0.55, 0.45])])
save("contingency", "Response by treatment (contingency)",
     "Two categorical variables — chi-square, Fisher's exact test, relative risk, odds ratio.",
     pd.DataFrame({"Treatment": trt, "Outcome": resp}), "contingency")
# 8 enzyme kinetics
s = np.repeat([0.5, 1, 2, 4, 8, 16, 32, 64], 3)
save("enzyme_kinetics", "Enzyme kinetics (Michaelis–Menten)", "Velocity at eight substrate concentrations, in triplicate.",
     pd.DataFrame({"Substrate (mM)": s, "Velocity": (120 * s / (6 + s) + rng.normal(0, 4, len(s))).round(1)}), "xy")
# 9 ROC
dis = np.repeat(["Healthy", "Disease"], [50, 45])
save("diagnostic", "Diagnostic markers (ROC)", "Two biomarkers in healthy and diseased subjects — ROC curves.",
     pd.DataFrame({"Status": dis,
                   "Marker A": np.concatenate([rng.normal(10, 3, 50), rng.normal(15.5, 4, 45)]).round(2),
                   "Marker B": np.concatenate([rng.normal(40, 10, 50), rng.normal(46, 12, 45)]).round(1)}), "roc")
# 10 volcano
m = 1500
fc = rng.normal(0, 0.9, m)
hits = rng.choice(m, 90, replace=False)
fc[hits] += rng.choice([-1, 1], 90) * rng.uniform(1.2, 3.5, 90)
z = np.abs(fc) * rng.uniform(0.7, 2.1, m) + rng.exponential(0.4, m)
p = np.clip(2 * (1 - __import__("scipy.stats").stats.norm.cdf(z)), 1e-40, 1)
save("volcano", "Differential expression (volcano)", "log2 fold change and P value for 1,500 genes.",
     pd.DataFrame({"Gene": [f"GENE{i+1}" for i in range(m)], "log2FC": fc.round(3), "P value": p}), "volcano")
# 11 repeated measures
base = rng.normal(60, 9, 14)
save("repeated_measures", "Repeated measures (three time points)",
     "Each row is one subject measured at baseline, week 4 and week 8 — RM ANOVA, Friedman.",
     pd.DataFrame({"Subject": [f"S{i+1:02d}" for i in range(14)], "Baseline": base.round(1),
                   "Week 4": (base + rng.normal(5, 5, 14)).round(1), "Week 8": (base + rng.normal(11, 6, 14)).round(1)}),
     "paired")
# 12 multivariate
k = 90
sp = np.repeat(["Type I", "Type II", "Type III"], k // 3)
f1 = rng.normal(0, 1, k) + np.repeat([-1.5, 0, 1.6], k // 3)
f2 = rng.normal(0, 1, k) + np.repeat([0.8, -1.0, 0.4], k // 3)
mv = pd.DataFrame({"Subtype": sp,
                   "IL-6": (20 + 6 * f1 + rng.normal(0, 3, k)).round(1), "TNF-α": (15 + 4 * f1 + rng.normal(0, 3, k)).round(1),
                   "CRP": (8 + 2.5 * f1 + 1.5 * f2 + rng.normal(0, 2, k)).round(1), "IL-10": (12 - 3 * f2 + rng.normal(0, 2.5, k)).round(1),
                   "IFN-γ": (30 + 7 * f2 + rng.normal(0, 4, k)).round(1), "VEGF": (100 + 12 * f2 - 6 * f1 + rng.normal(0, 10, k)).round(0)})
save("biomarker_panel", "Biomarker panel (multivariate)", "Six correlated biomarkers in three tumor subtypes — correlation matrix, PCA, heat map.",
     mv, "multivariate")
# 13 method comparison
truth = rng.uniform(40, 220, 45)
save("method_comparison", "Method comparison (Bland–Altman)", "The same samples measured with two assays.",
     pd.DataFrame({"Assay A": (truth + rng.normal(0, 6, 45)).round(1), "Assay B": (truth * 1.03 + 2 + rng.normal(0, 7, 45)).round(1)}),
     "agreement")
# 14 forest
save("subgroups", "Subgroup hazard ratios (forest plot)", "Hazard ratios with 95% confidence limits for a forest plot.",
     pd.DataFrame({"Subgroup": ["Overall", "Age < 65", "Age ≥ 65", "Female", "Male", "ECOG 0", "ECOG 1", "Prior therapy", "No prior therapy"],
                   "HR": [0.72, 0.66, 0.81, 0.70, 0.74, 0.61, 0.83, 0.78, 0.65],
                   "Lower": [0.60, 0.50, 0.62, 0.52, 0.58, 0.45, 0.65, 0.59, 0.47],
                   "Upper": [0.87, 0.87, 1.06, 0.94, 0.95, 0.83, 1.07, 1.03, 0.90]}), "forest")
# 15 waterfall
arm = rng.choice(["Monotherapy", "Combination"], 36)
chg = np.where(arm == "Combination", rng.normal(-32, 30, 36), rng.normal(-8, 30, 36)).clip(-100, 80)
save("tumor_response", "Best tumor response (waterfall)", "Best percent change in tumor size per patient.",
     pd.DataFrame({"Patient": [f"{i+1:03d}" for i in range(36)], "Best change (%)": chg.round(0), "Arm": arm}), "waterfall")

(OUT / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
print(len(index), "samples")
