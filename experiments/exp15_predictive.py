"""
Experiment 15: Predictive Scaling Relation

Can we predict trapping severity from observables?

For each graph × budget combination, we have:
  - Gamma (angular compression)
  - rho(cos, ORC) (wrong gradient strength)
  - Degree CV (hub structure)
  - Clustering coefficient
  - Mean edge cosine
  - Conductance
  - Budget H

Target: cosine single-source reachability (% targets found)

If we can fit a model that predicts reachability from these features
across ALL graphs, that's a predictive scaling relation.

Also: confirm item 3 — the native-feature graphs (Cora 1433D, Amazon 767D,
DBLP 1639D) use completely different embedding spaces from PRSM (nomic 768D).
If the same structural relationships hold, that's multi-model confirmation.
"""
import sys
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

sys.stdout.reconfigure(encoding='utf-8')

OUT = Path(__file__).resolve().parent / "results"

# ══════════════════════════════════════════════════════════════════
# COLLECT ALL DATA FROM PRIOR EXPERIMENTS
# ══════════════════════════════════════════════════════════════════

# Graph structural properties (from exp7)
graph_props = {
    "PRSM": {
        "gamma": 0.249, "orc_cos_rho": -0.964, "degree_cv": 2.694,
        "cc": 0.614, "edge_cos": 0.460, "conductance": 0.756,
        "N": 39220, "D": 768, "embed_family": "nomic",
    },
    "Cora": {
        "gamma": 0.036, "orc_cos_rho": -0.743, "degree_cv": 1.341,
        "cc": 0.289, "edge_cos": 0.168, "conductance": 0.482,
        "N": 2708, "D": 1433, "embed_family": "native_bow",
    },
    "Amazon": {
        "gamma": 0.227, "orc_cos_rho": -0.873, "degree_cv": 1.941,
        "cc": 0.362, "edge_cos": 0.490, "conductance": 0.503,
        "N": 13752, "D": 767, "embed_family": "native_product",
    },
    "DBLP": {
        "gamma": 0.010, "orc_cos_rho": -0.884, "degree_cv": 1.566,
        "cc": 0.184, "edge_cos": 0.152, "conductance": 0.597,
        "N": 17716, "D": 1639, "embed_family": "native_bow",
    },
    "Synth_Iso": {
        "gamma": 0.000, "orc_cos_rho": -0.983, "degree_cv": 1.654,
        "cc": 0.046, "edge_cos": 0.096, "conductance": 0.585,
        "N": 10000, "D": 768, "embed_family": "random",
    },
    "Synth_Mix": {
        "gamma": 0.429, "orc_cos_rho": -0.824, "degree_cv": 1.343,
        "cc": 0.066, "edge_cos": 0.966, "conductance": 0.398,
        "N": 10000, "D": 768, "embed_family": "random_clustered",
    },
}

# Reachability data from experiments 9 and 13
# Format: {graph: {budget: {policy: reachability%}}}
reachability = {
    "PRSM": {
        25:  {"bfs": 2.4, "cosine": 73.2, "multi": 88.6, "bidir": 91.2, "mmr": 51.2, "ppr": 2.4},
        50:  {"bfs": 4.2, "cosine": 77.8, "multi": 94.8, "bidir": 98.8, "mmr": 59.6, "ppr": 4.2},
        100: {"bfs": 8.0, "cosine": 82.2, "multi": 97.0, "bidir": 100.0, "mmr": 68.6, "ppr": 8.0},
        200: {"bfs": 13.6, "cosine": 86.6, "multi": 99.8, "bidir": 100.0, "mmr": 74.0, "ppr": 13.6},
        500: {"bfs": 32.8, "cosine": 92.4, "multi": 100.0, "bidir": 100.0, "mmr": 80.6, "ppr": 32.8},
    },
    "Cora": {
        25:  {"bfs": 0.0, "cosine": 16.0, "multi": 17.0, "bidir": 5.0, "mmr": 9.7, "ppr": 0.0},
        50:  {"bfs": 0.7, "cosine": 28.7, "multi": 40.0, "bidir": 9.3, "mmr": 19.3, "ppr": 0.3},
        100: {"bfs": 1.3, "cosine": 48.3, "multi": 70.3, "bidir": 23.3, "mmr": 36.3, "ppr": 1.3},
        200: {"bfs": 5.3, "cosine": 69.7, "multi": 89.7, "bidir": 52.3, "mmr": 57.0, "ppr": 2.7},
    },
    "Amazon": {
        25:  {"bfs": 0.0, "cosine": 45.7, "multi": 51.7, "bidir": 2.3, "mmr": 2.3, "ppr": 0.0},
        50:  {"bfs": 0.0, "cosine": 65.3, "multi": 78.7, "bidir": 6.7, "mmr": 6.7, "ppr": 0.0},
        100: {"bfs": 0.3, "cosine": 77.0, "multi": 92.3, "bidir": 17.3, "mmr": 11.7, "ppr": 0.3},
        200: {"bfs": 0.3, "cosine": 85.7, "multi": 98.0, "bidir": 36.3, "mmr": 22.0, "ppr": 0.3},
    },
    "DBLP": {
        25:  {"bfs": 0.0, "cosine": 15.3, "multi": 12.7, "bidir": 1.0, "mmr": 7.3, "ppr": 0.0},
        50:  {"bfs": 0.0, "cosine": 28.3, "multi": 35.3, "bidir": 3.0, "mmr": 13.7, "ppr": 0.0},
        100: {"bfs": 0.0, "cosine": 39.7, "multi": 63.3, "bidir": 8.3, "mmr": 24.0, "ppr": 0.0},
        200: {"bfs": 0.7, "cosine": 49.7, "multi": 82.7, "bidir": 21.7, "mmr": 36.3, "ppr": 0.3},
    },
}

# ══════════════════════════════════════════════════════════════════
# BUILD PREDICTION DATASET
# ══════════════════════════════════════════════════════════════════

print(f"{'=' * 70}")
print("PREDICTIVE SCALING RELATION")
print(f"{'=' * 70}")

rows = []
for graph, budgets in reachability.items():
    props = graph_props[graph]
    for H, policies in budgets.items():
        row = {
            "graph": graph,
            "budget": H,
            "log_budget": np.log(H),
            "gamma": props["gamma"],
            "orc_cos_rho": abs(props["orc_cos_rho"]),  # absolute value
            "degree_cv": props["degree_cv"],
            "cc": props["cc"],
            "edge_cos": props["edge_cos"],
            "conductance": props["conductance"],
            "D": props["D"],
            "log_N": np.log(props["N"]),
            "embed_family": props["embed_family"],
            # Targets
            "bfs_reach": policies["bfs"],
            "cosine_reach": policies["cosine"],
            "multi_reach": policies["multi"],
            "bidir_reach": policies["bidir"],
            "mmr_reach": policies["mmr"],
            # Derived: trapping severity = how much cosine underperforms multi-anchor
            "trapping_severity": policies["multi"] - policies["cosine"],
            # Multi-anchor advantage over BFS
            "multi_advantage": policies["multi"] - policies["bfs"],
        }
        rows.append(row)

print(f"\n  {len(rows)} graph × budget observations from {len(reachability)} graphs")

# ══════════════════════════════════════════════════════════════════
# PREDICT COSINE REACHABILITY FROM GRAPH PROPERTIES
# ══════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("MODEL 1: Predict cosine reachability from graph properties + budget")
print(f"{'=' * 70}")

feature_names = ["gamma", "orc_cos_rho", "degree_cv", "cc", "edge_cos",
                 "conductance", "log_budget", "log_N"]
X = np.array([[r[f] for f in feature_names] for r in rows])
y_cosine = np.array([r["cosine_reach"] for r in rows])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y_cosine)
y_pred = model.predict(X_scaled)
r2 = r2_score(y_cosine, y_pred)
mae = mean_absolute_error(y_cosine, y_pred)

print(f"\n  R² = {r2:.4f}")
print(f"  MAE = {mae:.2f} percentage points")
print(f"\n  Feature coefficients:")
print(f"  {'Feature':>15s} {'Coefficient':>12s} {'|Coef|':>8s}")
print(f"  {'-'*15} {'-'*12} {'-'*8}")
coef_order = np.argsort(-np.abs(model.coef_))
for i in coef_order:
    print(f"  {feature_names[i]:>15s} {model.coef_[i]:12.3f} {abs(model.coef_[i]):8.3f}")

# Per-graph prediction accuracy
print(f"\n  Per-graph predictions (cosine reachability):")
print(f"  {'Graph':>10s} {'Budget':>7s} {'Actual':>8s} {'Predicted':>10s} {'Error':>8s}")
for r, yp in zip(rows, y_pred):
    err = yp - r["cosine_reach"]
    print(f"  {r['graph']:>10s} {r['budget']:7d} {r['cosine_reach']:7.1f}% {yp:9.1f}% {err:+7.1f}%")

# ══════════════════════════════════════════════════════════════════
# PREDICT TRAPPING SEVERITY
# ══════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("MODEL 2: Predict trapping severity (multi - cosine gap)")
print(f"{'=' * 70}")

y_trap = np.array([r["trapping_severity"] for r in rows])
model2 = LinearRegression()
model2.fit(X_scaled, y_trap)
y_pred2 = model2.predict(X_scaled)
r2_trap = r2_score(y_trap, y_pred2)
mae_trap = mean_absolute_error(y_trap, y_pred2)

print(f"\n  R² = {r2_trap:.4f}")
print(f"  MAE = {mae_trap:.2f} percentage points")
print(f"\n  Feature coefficients (what predicts trapping severity):")
print(f"  {'Feature':>15s} {'Coefficient':>12s}")
coef_order2 = np.argsort(-np.abs(model2.coef_))
for i in coef_order2:
    print(f"  {feature_names[i]:>15s} {model2.coef_[i]:12.3f}")

# ══════════════════════════════════════════════════════════════════
# MULTI-MODEL CONFIRMATION (Item 3)
# ══════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("MULTI-MODEL CONFIRMATION")
print(f"{'=' * 70}")

print(f"\n  Embedding families tested:")
families = {}
for graph, props in graph_props.items():
    fam = props["embed_family"]
    families.setdefault(fam, []).append(graph)
for fam, graphs in families.items():
    print(f"    {fam:>20s}: {', '.join(graphs)}")

print(f"\n  Policy ordering consistency across embedding families:")
print(f"  {'Graph':>10s} {'Embed':>15s} {'D':>5s} {'Multi>Cosine?':>13s} {'Cosine>BFS?':>12s} {'MMR<Cosine?':>12s}")
for graph, budgets in reachability.items():
    props = graph_props[graph]
    h100 = budgets.get(100, budgets.get(200, {}))
    multi_gt_cos = h100["multi"] > h100["cosine"]
    cos_gt_bfs = h100["cosine"] > h100["bfs"]
    mmr_lt_cos = h100["mmr"] < h100["cosine"]
    print(f"  {graph:>10s} {props['embed_family']:>15s} {props['D']:5d} "
          f"{'YES' if multi_gt_cos else 'NO':>13s} "
          f"{'YES' if cos_gt_bfs else 'NO':>12s} "
          f"{'YES' if mmr_lt_cos else 'NO':>12s}")

# Count how many orderings hold
n_graphs = len(reachability)
multi_gt_cos_count = sum(1 for g, b in reachability.items()
                        for h, p in b.items()
                        if p["multi"] > p["cosine"])
total_obs = sum(len(b) for b in reachability.values())
print(f"\n  Multi > Cosine: {multi_gt_cos_count}/{total_obs} budget-graph combinations "
      f"({multi_gt_cos_count/total_obs*100:.0f}%)")

cos_gt_bfs_count = sum(1 for g, b in reachability.items()
                      for h, p in b.items()
                      if p["cosine"] > p["bfs"])
print(f"  Cosine > BFS:   {cos_gt_bfs_count}/{total_obs} ({cos_gt_bfs_count/total_obs*100:.0f}%)")

mmr_lt_cos_count = sum(1 for g, b in reachability.items()
                      for h, p in b.items()
                      if p["mmr"] < p["cosine"])
print(f"  MMR < Cosine:   {mmr_lt_cos_count}/{total_obs} ({mmr_lt_cos_count/total_obs*100:.0f}%)")

# ══════════════════════════════════════════════════════════════════
# SCALING LAW: Does reachability follow a predictable function of H?
# ══════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("SCALING LAW: Reachability vs Budget")
print(f"{'=' * 70}")

print(f"\n  Fit: reach(H) = a * log(H) + b for each graph × policy")
print(f"  {'Graph':>10s} {'Policy':>10s} {'a (slope)':>10s} {'b (intercept)':>14s} {'R²':>6s}")
for graph in reachability:
    for policy in ["cosine", "multi", "bfs"]:
        budgets_list = sorted(reachability[graph].keys())
        log_H = np.log(budgets_list).reshape(-1, 1)
        reach = np.array([reachability[graph][h][policy] for h in budgets_list])
        if np.std(reach) < 0.1:
            continue
        lr = LinearRegression().fit(log_H, reach)
        r2_fit = lr.score(log_H, reach)
        print(f"  {graph:>10s} {policy:>10s} {lr.coef_[0]:10.2f} {lr.intercept_:14.2f} {r2_fit:6.3f}")

# ══════════════════════════════════════════════════════════════════
# SIMPLE PREDICTIVE FORMULA
# ══════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("CANDIDATE PREDICTIVE FORMULA")
print(f"{'=' * 70}")

# Try: cosine_reach ≈ f(Gamma, log(H), degree_cv)
# Minimal model with the 3 most interpretable features
X_simple = np.array([[r["gamma"], r["log_budget"], r["degree_cv"]] for r in rows])
X_simple_scaled = StandardScaler().fit_transform(X_simple)
model_simple = LinearRegression().fit(X_simple_scaled, y_cosine)
y_pred_simple = model_simple.predict(X_simple_scaled)
r2_simple = r2_score(y_cosine, y_pred_simple)

print(f"\n  Simple model: cosine_reach ~ Gamma + log(H) + degree_CV")
print(f"  R² = {r2_simple:.4f} (vs full model R² = {r2:.4f})")
print(f"  Coefficients: Gamma={model_simple.coef_[0]:.2f}, "
      f"log(H)={model_simple.coef_[1]:.2f}, "
      f"degCV={model_simple.coef_[2]:.2f}")

if r2_simple > 0.7:
    print(f"\n  The simple 3-feature model explains {r2_simple*100:.0f}% of variance.")
    print(f"  This is a candidate predictive scaling relation.")
else:
    print(f"\n  Simple model R² = {r2_simple:.3f} — needs more features or nonlinear terms.")

# Save
output = {
    "model1_r2": float(r2),
    "model1_mae": float(mae),
    "model1_coefficients": {f: float(c) for f, c in zip(feature_names, model.coef_)},
    "model2_r2_trapping": float(r2_trap),
    "model2_mae_trapping": float(mae_trap),
    "simple_model_r2": float(r2_simple),
    "multi_gt_cosine_pct": float(multi_gt_cos_count / total_obs * 100),
    "cosine_gt_bfs_pct": float(cos_gt_bfs_count / total_obs * 100),
    "mmr_lt_cosine_pct": float(mmr_lt_cos_count / total_obs * 100),
    "n_graphs": len(reachability),
    "n_observations": total_obs,
    "embedding_families": {fam: graphs for fam, graphs in families.items()},
}
with open(OUT / "exp15_predictive_scaling.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {OUT / 'exp15_predictive_scaling.json'}")
print("PREDICTIVE SCALING COMPLETE")
