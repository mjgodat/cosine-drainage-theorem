"""
Experiment 18: Mechanism Falsification — Cross-Graph Replication

Run the same 5 falsification predictions on Cora, Amazon, DBLP,
and synthetics to determine if the "Moderate Similarity Trap"
(hubs score moderately on cosine but specialists win nearest-neighbor)
is universal or PRSM-specific.
"""
import sys
import time
import json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(__file__).resolve().parent / "results"

print(f"Device: {DEVICE}")


def run_falsification(name, features, adj_list, degrees):
    N, D = features.shape
    print(f"\n{'=' * 70}")
    print(f"GRAPH: {name} ({N:,} nodes, {D}D)")
    print(f"{'=' * 70}")

    norms = np.linalg.norm(features, axis=1)
    norm_cv = np.std(norms) / np.mean(norms) if np.mean(norms) > 0 else 0

    # L2 normalize
    emb_normed = features / np.clip(norms[:, np.newaxis], 1e-8, None)
    t_norm = torch.tensor(emb_normed, device=DEVICE)
    t_raw = torch.tensor(features, device=DEVICE)

    result = {"name": name, "N": N, "D": D, "norm_cv": float(norm_cv)}

    # P1: Cosine-to-random-target vs degree
    n_targets = min(200, N // 5)
    target_idx = np.random.choice(N, n_targets, replace=False)
    mean_cos = np.zeros(N)
    for ti in target_idx:
        sims = (t_norm @ t_norm[ti]).cpu().numpy()
        mean_cos += sims
    mean_cos /= n_targets

    mask = degrees > 0
    rho1, p1 = spearmanr(degrees[mask], mean_cos[mask])
    print(f"  P1 (cos-to-target vs degree): rho = {rho1:+.4f} (p = {p1:.2e})")
    result["P1_rho"] = float(rho1)
    result["P1_p"] = float(p1)

    # P2: Norm vs degree
    rho2, p2 = spearmanr(degrees[mask], norms[mask])
    print(f"  P2 (norm vs degree):          rho = {rho2:+.4f} (p = {p2:.2e}) CV={norm_cv:.4f}")
    result["P2_rho"] = float(rho2)

    # P3: Angular dispersion vs degree
    sample = [i for i in range(N) if degrees[i] >= 3]
    if len(sample) > 2000:
        sample = list(np.random.choice(sample, 2000, replace=False))
    dispersions = []
    sample_degs = []
    for node in sample:
        nbs = adj_list[node][:50]
        if len(nbs) < 3:
            continue
        cosines = [float(emb_normed[nb] @ emb_normed[node]) for nb in nbs]
        dispersions.append(np.var(cosines))
        sample_degs.append(degrees[node])

    if len(dispersions) > 50:
        rho3, p3 = spearmanr(sample_degs, dispersions)
    else:
        rho3, p3 = 0, 1
    print(f"  P3 (dispersion vs degree):    rho = {rho3:+.4f} (p = {p3:.2e})")
    result["P3_rho"] = float(rho3)

    # P4: Cosine-seed vs random-seed degree
    n_queries = min(200, N // 5)
    query_nodes = np.random.choice([i for i in range(N) if degrees[i] > 0], n_queries, replace=False)
    cos_seed_degs = []
    rand_seed_degs = []
    for q in query_nodes:
        sims = (t_norm[q:q+1] @ t_norm.T).squeeze(0)
        sims[q] = -2
        cos_seed = int(sims.argmax().item())
        cos_seed_degs.append(degrees[cos_seed])
        rand_seed = np.random.choice([i for i in range(N) if i != q and degrees[i] > 0])
        rand_seed_degs.append(degrees[rand_seed])

    cos_mean = np.mean(cos_seed_degs)
    rand_mean = np.mean(rand_seed_degs)
    ratio = cos_mean / rand_mean if rand_mean > 0 else 0
    print(f"  P4 (cos-seed deg / rand-seed): {cos_mean:.1f} / {rand_mean:.1f} = {ratio:.3f}")
    result["P4_cos_deg"] = float(cos_mean)
    result["P4_rand_deg"] = float(rand_mean)
    result["P4_ratio"] = float(ratio)

    # P5: L2 normalization effect
    same = 0
    raw_degs = []
    norm_degs = []
    for q in query_nodes[:100]:
        sr = (t_raw[q:q+1] @ t_raw.T).squeeze(0)
        sr[q] = -float('inf')
        raw_seed = int(sr.argmax().item())
        sn = (t_norm[q:q+1] @ t_norm.T).squeeze(0)
        sn[q] = -float('inf')
        norm_seed = int(sn.argmax().item())
        if raw_seed == norm_seed:
            same += 1
        raw_degs.append(degrees[raw_seed])
        norm_degs.append(degrees[norm_seed])

    print(f"  P5 (same seed after L2):      {same}/100  raw_deg={np.mean(raw_degs):.1f} norm_deg={np.mean(norm_degs):.1f}")
    result["P5_same_pct"] = same
    result["P5_raw_deg"] = float(np.mean(raw_degs))
    result["P5_norm_deg"] = float(np.mean(norm_degs))

    # Interpretation
    hub_attract = rho1 > 0.05
    hub_avoid_seed = ratio < 0.8
    ang_smear = rho3 > 0.1
    moderate_trap = hub_attract and hub_avoid_seed

    print(f"\n  INTERPRETATION:")
    if moderate_trap:
        print(f"  → MODERATE SIMILARITY TRAP: hubs closer on average (P1) but specialists win nearest-neighbor (P4)")
    elif hub_attract and not hub_avoid_seed:
        print(f"  → PURE HUB ATTRACTION: hubs win both average AND nearest-neighbor cosine")
    elif hub_avoid_seed and not hub_attract:
        print(f"  → PURE HUB AVOIDANCE: hubs lose on both metrics")
    else:
        print(f"  → NO CLEAR PATTERN")

    if ang_smear:
        print(f"  → ANGULAR SMEARING confirmed (hub neighbors more dispersed)")
    if norm_cv < 0.05:
        print(f"  → NORMS UNIFORM (norm pathway irrelevant)")

    result["moderate_trap"] = moderate_trap
    result["hub_attract"] = hub_attract
    result["hub_avoid_seed"] = hub_avoid_seed
    result["angular_smearing"] = ang_smear

    return result


def build_adj_list(edge_index, N):
    adj = [[] for _ in range(N)]
    for s, d in zip(edge_index[0], edge_index[1]):
        adj[int(s)].append(int(d))
    for i in range(N):
        adj[i] = list(set(adj[i]))
    return adj


# ══════════════════════════════════════════════════════════════════
# RUN ON ALL GRAPHS
# ══════════════════════════════════════════════════════════════════

all_results = []

# PyG graphs
from torch_geometric.datasets import Planetoid, Amazon, CitationFull

for cls, kwargs, name in [
    (Planetoid, {"root": "/tmp/pyg_data", "name": "Cora"}, "Cora"),
    (Amazon, {"root": "/tmp/pyg_data", "name": "Computers"}, "Amazon Computers"),
    (CitationFull, {"root": "/tmp/pyg_data", "name": "DBLP"}, "DBLP"),
]:
    d = cls(**kwargs); data = d[0]
    features = data.x.numpy().astype(np.float32)
    adj = build_adj_list(data.edge_index.numpy(), data.num_nodes)
    degrees = np.array([len(adj[i]) for i in range(data.num_nodes)])
    all_results.append(run_falsification(name, features, adj, degrees))

# Synthetic isotropic
N_SYN, D_SYN = 10000, 768
iso = np.random.randn(N_SYN, D_SYN).astype(np.float32)
t_gpu = torch.tensor(iso, device=DEVICE)
knn = torch.zeros(N_SYN, 20, dtype=torch.long, device=DEVICE)
for s in range(0, N_SYN, 512):
    e = min(s + 512, N_SYN)
    dd = torch.cdist(t_gpu[s:e], t_gpu)
    for i in range(e - s): dd[i, s + i] = float('inf')
    _, tk = dd.topk(20, dim=1, largest=False)
    knn[s:e] = tk
knn_np = knn.cpu().numpy()
adj_iso = [[] for _ in range(N_SYN)]
for i in range(N_SYN):
    for j in knn_np[i]:
        adj_iso[i].append(int(j))
        adj_iso[int(j)].append(i)
for i in range(N_SYN):
    adj_iso[i] = list(set(adj_iso[i]))
deg_iso = np.array([len(adj_iso[i]) for i in range(N_SYN)])
all_results.append(run_falsification("Synthetic Isotropic", iso, adj_iso, deg_iso))

# Synthetic mixture
mix = np.zeros((N_SYN, D_SYN), dtype=np.float32)
centers = [np.random.randn(D_SYN).astype(np.float32) * 5 for _ in range(3)]
idx = 0
for center, size in zip(centers, [7000, 2000, 1000]):
    mix[idx:idx+size] = np.random.randn(size, D_SYN).astype(np.float32) + center
    idx += size
t_gpu = torch.tensor(mix, device=DEVICE)
knn = torch.zeros(N_SYN, 20, dtype=torch.long, device=DEVICE)
for s in range(0, N_SYN, 512):
    e = min(s + 512, N_SYN)
    dd = torch.cdist(t_gpu[s:e], t_gpu)
    for i in range(e - s): dd[i, s + i] = float('inf')
    _, tk = dd.topk(20, dim=1, largest=False)
    knn[s:e] = tk
knn_np = knn.cpu().numpy()
adj_mix = [[] for _ in range(N_SYN)]
for i in range(N_SYN):
    for j in knn_np[i]:
        adj_mix[i].append(int(j))
        adj_mix[int(j)].append(i)
for i in range(N_SYN):
    adj_mix[i] = list(set(adj_mix[i]))
deg_mix = np.array([len(adj_mix[i]) for i in range(N_SYN)])
all_results.append(run_falsification("Synthetic Mixture", mix, adj_mix, deg_mix))


# ══════════════════════════════════════════════════════════════════
# CROSS-GRAPH SUMMARY
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("CROSS-GRAPH MECHANISM SUMMARY")
print(f"{'=' * 70}")

print(f"\n  {'Graph':>20s} {'P1(cos-deg)':>12s} {'P3(disp)':>10s} {'P4(ratio)':>10s} {'P5(same%)':>10s} {'Norm CV':>8s} {'Pattern':>20s}")
print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*20}")

for r in all_results:
    pattern = "MOD TRAP" if r["moderate_trap"] else "HUB ATTRACT" if r["hub_attract"] and not r["hub_avoid_seed"] else "HUB AVOID" if r["hub_avoid_seed"] else "NONE"
    print(f"  {r['name']:>20s} {r['P1_rho']:+12.4f} {r['P3_rho']:+10.4f} {r['P4_ratio']:10.3f} {r['P5_same_pct']:9d}% {r.get('norm_cv',0):8.4f} {pattern:>20s}")

# Add PRSM from exp17
print(f"  {'PRSM Crystal':>20s} {'+0.1199':>12s} {'+0.1899':>10s} {'0.469':>10s} {'81':>9s}% {'0.0318':>8s} {'MOD TRAP':>20s}")

# Count patterns
patterns = [r.get("moderate_trap", False) for r in all_results]
print(f"\n  Moderate Similarity Trap confirmed on: {sum(patterns)}/{len(all_results)} graphs (+ PRSM)")

with open(OUT / "exp18_mechanism_cross_graph.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {OUT / 'exp18_mechanism_cross_graph.json'}")
print("CROSS-GRAPH MECHANISM FALSIFICATION COMPLETE")
