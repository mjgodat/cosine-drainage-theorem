"""
Experiment 2: Local Gamma_k(v) Trapping Prediction

Tests whether local angular compression independently predicts traversal
failure after controlling for degree, graph distance, angular distance,
and local conductance.
"""
import sys
import time
import numpy as np
import torch
import json
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from math import pi

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

N_PAIRS = 2000
N_WALKS = 50
H_BUDGET = 5
K_LOCAL = 20

print(f"Device: {DEVICE}")


def compute_local_gamma(features, k=K_LOCAL):
    t = torch.tensor(features, device=DEVICE)
    t_normed = t / t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    N = len(features)
    gamma = np.zeros(N, dtype=np.float32)
    BATCH = 512
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        sims = t_normed[start:end] @ t_normed.T
        for i in range(end - start):
            sims[i, start + i] = -2
        topk_sims, _ = sims.topk(k, dim=1)
        mean_angles = torch.arccos(topk_sims.clamp(-1, 1)).mean(dim=1)
        gamma[start:end] = (1.0 - mean_angles.cpu().numpy() / (pi / 2))
    return gamma


def compute_local_conductance(adj, node):
    visited = {node}
    frontier = {node}
    for _ in range(2):
        new_frontier = set()
        for n in frontier:
            for nb in adj.get(n, []):
                if nb not in visited:
                    new_frontier.add(nb)
                    visited.add(nb)
        frontier = new_frontier
    if len(visited) < 2:
        return 0.0
    internal = 0
    boundary = 0
    for n in visited:
        for nb in adj.get(n, []):
            if nb in visited:
                internal += 1
            else:
                boundary += 1
    total = internal + boundary
    return boundary / total if total > 0 else 0.0


def run_bounded_walks(adj, source, target, n_walks=N_WALKS, budget=H_BUDGET):
    reached = 0
    for _ in range(n_walks):
        cur = source
        for _ in range(budget):
            neighbors = adj.get(cur, [])
            if not neighbors:
                break
            cur = neighbors[np.random.randint(len(neighbors))]
            if cur == target:
                reached += 1
                break
    return reached / n_walks


def run_graph_experiment(name, features, edge_index):
    N = len(features)
    print(f"\n{'=' * 70}")
    print(f"GRAPH: {name} ({N:,} nodes)")
    print(f"{'=' * 70}")

    adj = {}
    src, dst = edge_index[0], edge_index[1]
    for s, d in zip(src, dst):
        adj.setdefault(int(s), []).append(int(d))

    print("  Computing local Gamma_k...", end=" ", flush=True)
    t0 = time.time()
    gamma = compute_local_gamma(features, k=K_LOCAL)
    print(f"{time.time()-t0:.1f}s")

    degree = np.array([len(adj.get(i, [])) for i in range(N)], dtype=np.float32)

    t = torch.tensor(features, device=DEVICE)
    t_normed = t / t.norm(dim=1, keepdim=True).clamp(min=1e-8)

    print("  Sampling source-target pairs...", end=" ", flush=True)
    t0 = time.time()
    nodes_with_edges = [n for n in range(N) if len(adj.get(n, [])) > 2]

    pairs_data = []
    attempts = 0
    while len(pairs_data) < N_PAIRS and attempts < N_PAIRS * 20:
        attempts += 1
        s = nodes_with_edges[np.random.randint(len(nodes_with_edges))]
        t_node = nodes_with_edges[np.random.randint(len(nodes_with_edges))]
        if s == t_node:
            continue

        cos_st = float((t_normed[s] @ t_normed[t_node]).item())
        ang_dist = float(np.arccos(np.clip(cos_st, -1, 1)))

        visited = {s}
        frontier = {s}
        g_dist = -1
        for hop in range(1, 11):
            new_frontier = set()
            for n in frontier:
                for nb in adj.get(n, []):
                    if nb == t_node:
                        g_dist = hop
                        break
                    if nb not in visited:
                        new_frontier.add(nb)
                        visited.add(nb)
                if g_dist > 0:
                    break
            if g_dist > 0:
                break
            frontier = new_frontier
        if g_dist < 0:
            g_dist = 10

        cond = compute_local_conductance(adj, s) if len(pairs_data) < 500 else 0.0

        p_reach = run_bounded_walks(adj, s, t_node, n_walks=N_WALKS, budget=H_BUDGET)

        pairs_data.append({
            "source": s, "target": t_node,
            "gamma_local": float(gamma[s]),
            "degree_source": float(degree[s]),
            "graph_dist": g_dist,
            "angular_dist": ang_dist,
            "conductance": cond,
            "p_reach": p_reach,
            "reached": 1 if p_reach > 0 else 0,
        })

    print(f"{time.time()-t0:.1f}s, {len(pairs_data)} pairs")

    print("  Fitting logistic regression...", flush=True)

    X_names = ["gamma_local", "degree_source", "graph_dist", "angular_dist"]
    X = np.array([[p[k] for k in X_names] for p in pairs_data])
    y = np.array([p["reached"] for p in pairs_data])

    n_pos = np.sum(y)
    n_neg = len(y) - n_pos
    print(f"    Reached: {n_pos}/{len(y)} ({n_pos/len(y)*100:.1f}%)")

    if n_pos < 10 or n_neg < 10:
        print(f"    SKIPPING: insufficient class balance")
        return {"name": name, "skipped": True, "n_pos": int(n_pos), "n_neg": int(n_neg)}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    y_pred = model.predict_proba(X_scaled)[:, 1]
    auc = roc_auc_score(y, y_pred)

    print(f"\n    LOGISTIC REGRESSION RESULTS:")
    print(f"    AUC: {auc:.4f}")
    print(f"    {'Feature':>20s} {'Coefficient':>12s} {'Interpretation':>30s}")
    print(f"    {'-'*20} {'-'*12} {'-'*30}")
    for fname, coef in zip(X_names, model.coef_[0]):
        if fname == "gamma_local":
            interp = "HIGHER compression -> " + ("MORE" if coef > 0 else "LESS") + " reachable"
        elif fname == "graph_dist":
            interp = "FARTHER graph dist -> " + ("MORE" if coef > 0 else "LESS") + " reachable"
        elif fname == "degree_source":
            interp = "HIGHER degree -> " + ("MORE" if coef > 0 else "LESS") + " reachable"
        else:
            interp = ""
        sig = "***" if abs(coef) > 0.5 else "**" if abs(coef) > 0.2 else ""
        print(f"    {fname:>20s} {coef:12.4f} {interp:>30s} {sig}")

    print(f"\n    KEY QUESTION: Does Gamma_local predict trapping after controlling for confounders?")
    print(f"    beta(gamma_local) = {model.coef_[0][0]:.4f}")
    if abs(model.coef_[0][0]) > 0.1:
        print(f"    ANSWER: YES — local angular compression has independent predictive power")
    else:
        print(f"    ANSWER: WEAK — effect is small after controlling for confounders")

    print(f"\n    CORRELATIONS:")
    for i, fname in enumerate(X_names):
        rho, p = spearmanr(X[:, i], y)
        print(f"    {fname:>20s} vs reached: rho={rho:+.4f} (p={p:.2e})")

    return {
        "name": name,
        "n_pairs": len(pairs_data),
        "n_reached": int(n_pos),
        "auc": float(auc),
        "coefficients": {fname: float(c) for fname, c in zip(X_names, model.coef_[0])},
        "intercept": float(model.intercept_[0]),
    }


def main():
    t_total = time.time()
    all_results = []

    from torch_geometric.datasets import Planetoid, Amazon, CitationFull

    d = Planetoid(root='/tmp/pyg_data', name='Cora'); data = d[0]
    all_results.append(run_graph_experiment("Cora",
        data.x.numpy().astype(np.float32), data.edge_index.numpy()))

    d = Amazon(root='/tmp/pyg_data', name='Computers'); data = d[0]
    all_results.append(run_graph_experiment("Amazon Computers",
        data.x.numpy().astype(np.float32), data.edge_index.numpy()))

    d = CitationFull(root='/tmp/pyg_data', name='DBLP'); data = d[0]
    all_results.append(run_graph_experiment("DBLP",
        data.x.numpy().astype(np.float32), data.edge_index.numpy()))

    print(f"\n{'=' * 70}")
    print("CROSS-GRAPH SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Graph':>20s} {'AUC':>6s} {'b(Gamma)':>12s} {'b(g_dist)':>12s} {'b(degree)':>12s} {'b(theta)':>12s}")
    for r in all_results:
        if r.get("skipped"):
            print(f"  {r['name']:>20s} SKIPPED")
            continue
        c = r["coefficients"]
        print(f"  {r['name']:>20s} {r['auc']:6.3f} {c['gamma_local']:12.4f} "
              f"{c['graph_dist']:12.4f} {c['degree_source']:12.4f} {c['angular_dist']:12.4f}")

    with open(OUT / "exp2_local_gamma_trapping.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {OUT / 'exp2_local_gamma_trapping.json'}")
    print(f"Total time: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
