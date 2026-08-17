"""
Experiment 17: Mechanism Falsification — Five predictions to discriminate
between hub entrapment, angular smearing, and dual-mechanism models.

Predictions:
  P1: Cosine-to-random-target vs degree (does cosine favor or avoid hubs?)
  P2: Norm vs degree in PRSM (does norm-degree anticorrelation exist?)
  P3: Angular dispersion vs degree (do hubs have smeared directions?)
  P4: BFS from cosine-seeds vs random-seeds (degree distribution comparison)
  P5: L2 normalization effect on seed selection (norm vs angle pathway)
"""
import sys
import time
import json
import numpy as np
import torch
from collections import deque
from pathlib import Path
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path("E:/PRSM")
OUT = Path(__file__).resolve().parent / "results"

print(f"Device: {DEVICE}")
print("Loading PRSM Crystal...", end=" ", flush=True)
t0 = time.time()

with open(ROOT / "data/g1_registry/unified_grains_with_embeddings.json", "r", encoding="utf-8") as f:
    grains_raw = json.load(f)
with open(ROOT / "data/g1_registry/relationship_registry.json", "r", encoding="utf-8") as f:
    rels = json.load(f)

concept_to_idx = {}
embeddings_list = []
seen = {}
for g in grains_raw:
    c = g.get("concept", "").lower()
    emb = g.get("embedding")
    bc = g.get("bind_count", 0)
    if c and emb and (c not in seen or bc > seen[c]):
        seen[c] = bc
        if c not in concept_to_idx:
            concept_to_idx[c] = len(concept_to_idx)
            embeddings_list.append(np.array(emb, dtype=np.float32))
        else:
            embeddings_list[concept_to_idx[c]] = np.array(emb, dtype=np.float32)

embeddings_raw = np.array(embeddings_list)
N = len(embeddings_raw)
norms_raw = np.linalg.norm(embeddings_raw, axis=1)
embeddings_normed = embeddings_raw / np.clip(norms_raw[:, np.newaxis], 1e-8, None)

adj = [[] for _ in range(N)]
gid_to_concept = {}
for g in grains_raw:
    gid = g.get("grain_id", "")
    c = g.get("concept", "").lower()
    if gid and c and c in concept_to_idx:
        gid_to_concept[gid] = c
for r in rels:
    a_c = gid_to_concept.get(r.get("grain_a_id", ""))
    b_c = gid_to_concept.get(r.get("grain_b_id", ""))
    if a_c and b_c and a_c != b_c:
        a = concept_to_idx.get(a_c)
        b = concept_to_idx.get(b_c)
        if a is not None and b is not None:
            adj[a].append(b)
            adj[b].append(a)
for i in range(N):
    adj[i] = list(set(adj[i]))

degrees = np.array([len(adj[i]) for i in range(N)])
log_degrees = np.log1p(degrees)

print(f"{N:,} grains, {time.time()-t0:.1f}s")

results = {}


# ══════════════════════════════════════════════════════════════════
# P1: COSINE-TO-RANDOM-TARGET vs DEGREE
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("P1: Does cosine-to-target favor or avoid hubs?")
print(f"{'=' * 70}")

# For 200 random targets, compute mean cosine-to-target for each node
# Then correlate with degree
t_emb = torch.tensor(embeddings_normed, device=DEVICE)
n_targets = 200
target_idx = np.random.choice(N, n_targets, replace=False)

mean_cos_to_targets = np.zeros(N)
for t_idx in target_idx:
    sims = (t_emb @ t_emb[t_idx]).cpu().numpy()
    mean_cos_to_targets += sims
mean_cos_to_targets /= n_targets

rho_cos_deg, p_cos_deg = spearmanr(degrees[degrees > 0], mean_cos_to_targets[degrees > 0])
print(f"  Spearman(degree, mean_cosine_to_random_targets): rho = {rho_cos_deg:.4f} (p = {p_cos_deg:.2e})")

if rho_cos_deg > 0.05:
    print(f"  → HUBS HAVE HIGHER COSINE to random targets (hub ATTRACTION model supported)")
elif rho_cos_deg < -0.05:
    print(f"  → HUBS HAVE LOWER COSINE to random targets (hub AVOIDANCE model supported)")
else:
    print(f"  → NO SIGNIFICANT RELATIONSHIP between degree and cosine-to-target")

# Bin by degree
deg_bins = [(0, 5), (5, 20), (20, 50), (50, 200), (200, 10000)]
print(f"\n  {'Degree Bin':>15s} {'N':>6s} {'Mean Cos-to-Target':>20s} {'Mean Norm':>10s}")
for lo, hi in deg_bins:
    mask = (degrees >= lo) & (degrees < hi)
    n = np.sum(mask)
    if n > 0:
        mc = np.mean(mean_cos_to_targets[mask])
        mn = np.mean(norms_raw[mask])
        print(f"  {f'{lo}-{hi}':>15s} {n:6d} {mc:20.6f} {mn:10.4f}")

results["P1"] = {"rho": float(rho_cos_deg), "p": float(p_cos_deg)}


# ══════════════════════════════════════════════════════════════════
# P2: NORM vs DEGREE
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("P2: Does norm correlate with degree in PRSM?")
print(f"{'=' * 70}")

mask_pos = degrees > 0
rho_norm_deg, p_norm_deg = spearmanr(degrees[mask_pos], norms_raw[mask_pos])
print(f"  Spearman(degree, L2_norm): rho = {rho_norm_deg:.4f} (p = {p_norm_deg:.2e})")
print(f"  Mean norm: {np.mean(norms_raw):.4f}, Std: {np.std(norms_raw):.4f}")
print(f"  CV of norms: {np.std(norms_raw)/np.mean(norms_raw):.4f}")

if abs(rho_norm_deg) < 0.05:
    print(f"  → Norms are effectively UNIFORM (embeddings may be pre-normalized)")
    print(f"  → Norm pathway is IRRELEVANT for cosine scoring")
elif rho_norm_deg < -0.1:
    print(f"  → HUBS HAVE SHORTER VECTORS (consistent with skip-gram literature)")
else:
    print(f"  → HUBS HAVE LONGER VECTORS (opposite of skip-gram prediction)")

results["P2"] = {"rho": float(rho_norm_deg), "p": float(p_norm_deg),
                 "norm_mean": float(np.mean(norms_raw)), "norm_cv": float(np.std(norms_raw)/np.mean(norms_raw))}


# ══════════════════════════════════════════════════════════════════
# P3: ANGULAR DISPERSION vs DEGREE
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("P3: Do hubs have more angularly dispersed neighbors?")
print(f"{'=' * 70}")

# For each node with degree >= 3, compute variance of cosine to its neighbors
sample_nodes = [i for i in range(N) if degrees[i] >= 3]
if len(sample_nodes) > 3000:
    sample_nodes = list(np.random.choice(sample_nodes, 3000, replace=False))

dispersions = []
sample_degrees = []
for node in sample_nodes:
    nbs = adj[node]
    if len(nbs) < 3:
        continue
    # Cosine of node to each neighbor
    node_vec = embeddings_normed[node]
    nb_cosines = [float(embeddings_normed[nb] @ node_vec) for nb in nbs[:50]]
    dispersions.append(np.var(nb_cosines))
    sample_degrees.append(degrees[node])

rho_disp_deg, p_disp_deg = spearmanr(sample_degrees, dispersions)
print(f"  Spearman(degree, angular_dispersion): rho = {rho_disp_deg:.4f} (p = {p_disp_deg:.2e})")

if rho_disp_deg > 0.1:
    print(f"  → HUBS HAVE MORE DISPERSED NEIGHBORS (angular smearing confirmed)")
else:
    print(f"  → No significant angular smearing with degree")

# Bin
print(f"\n  {'Degree Bin':>15s} {'N':>6s} {'Mean Dispersion':>16s}")
for lo, hi in deg_bins:
    mask = [(d >= lo and d < hi) for d in sample_degrees]
    vals = [dispersions[i] for i, m in enumerate(mask) if m]
    if vals:
        print(f"  {f'{lo}-{hi}':>15s} {len(vals):6d} {np.mean(vals):16.6f}")

results["P3"] = {"rho": float(rho_disp_deg), "p": float(p_disp_deg)}


# ══════════════════════════════════════════════════════════════════
# P4: BFS FROM COSINE-SEEDS vs RANDOM-SEEDS
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("P4: Does cosine seeding land in higher or lower degree neighborhoods?")
print(f"{'=' * 70}")

n_queries = 200
bfs_depth = 3

cos_seed_degrees = []
rand_seed_degrees = []
cos_bfs_mean_degrees = []
rand_bfs_mean_degrees = []

query_nodes = np.random.choice([i for i in range(N) if degrees[i] > 0], n_queries, replace=False)

for q in query_nodes:
    # Cosine nearest seed
    sims = (t_emb[q:q+1] @ t_emb.T).squeeze(0)
    sims[q] = -2
    cos_seed = int(sims.argmax().item())
    cos_seed_degrees.append(degrees[cos_seed])

    # Random seed
    rand_seed = np.random.choice([i for i in range(N) if i != q and degrees[i] > 0])
    rand_seed_degrees.append(degrees[rand_seed])

    # BFS 3 hops from cosine seed
    visited = {cos_seed}
    frontier = {cos_seed}
    for _ in range(bfs_depth):
        nf = set()
        for n in frontier:
            for nb in adj[n]:
                if nb not in visited:
                    visited.add(nb)
                    nf.add(nb)
        frontier = nf
    cos_bfs_mean_degrees.append(np.mean([degrees[v] for v in visited]) if visited else 0)

    # BFS 3 hops from random seed
    visited = {rand_seed}
    frontier = {rand_seed}
    for _ in range(bfs_depth):
        nf = set()
        for n in frontier:
            for nb in adj[n]:
                if nb not in visited:
                    visited.add(nb)
                    nf.add(nb)
        frontier = nf
    rand_bfs_mean_degrees.append(np.mean([degrees[v] for v in visited]) if visited else 0)

print(f"  Cosine-nearest seed mean degree: {np.mean(cos_seed_degrees):.1f}")
print(f"  Random seed mean degree:         {np.mean(rand_seed_degrees):.1f}")
print(f"  BFS from cosine seed mean deg:   {np.mean(cos_bfs_mean_degrees):.1f}")
print(f"  BFS from random seed mean deg:   {np.mean(rand_bfs_mean_degrees):.1f}")

if np.mean(cos_seed_degrees) > np.mean(rand_seed_degrees) * 1.2:
    print(f"  → COSINE SEEDS ARE HIGHER DEGREE (hub attraction confirmed)")
elif np.mean(cos_seed_degrees) < np.mean(rand_seed_degrees) * 0.8:
    print(f"  → COSINE SEEDS ARE LOWER DEGREE (hub avoidance confirmed)")
else:
    print(f"  → COSINE SEEDS HAVE SIMILAR DEGREE to random (no strong hub bias)")

results["P4"] = {
    "cos_seed_mean_deg": float(np.mean(cos_seed_degrees)),
    "rand_seed_mean_deg": float(np.mean(rand_seed_degrees)),
    "cos_bfs_mean_deg": float(np.mean(cos_bfs_mean_degrees)),
    "rand_bfs_mean_deg": float(np.mean(rand_bfs_mean_degrees)),
}


# ══════════════════════════════════════════════════════════════════
# P5: L2 NORMALIZATION EFFECT
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("P5: Does L2 normalization change seed selection?")
print(f"{'=' * 70}")

# Compare: cosine-nearest using raw embeddings vs L2-normalized embeddings
t_raw = torch.tensor(embeddings_raw, device=DEVICE)
t_norm = torch.tensor(embeddings_normed, device=DEVICE)

same_seed_count = 0
raw_seed_degrees = []
norm_seed_degrees = []

for q in query_nodes[:100]:
    # Raw cosine nearest
    sims_raw = (t_raw[q:q+1] @ t_raw.T).squeeze(0)
    sims_raw[q] = -float('inf')
    raw_seed = int(sims_raw.argmax().item())

    # Normalized cosine nearest
    sims_norm = (t_norm[q:q+1] @ t_norm.T).squeeze(0)
    sims_norm[q] = -float('inf')
    norm_seed = int(sims_norm.argmax().item())

    if raw_seed == norm_seed:
        same_seed_count += 1

    raw_seed_degrees.append(degrees[raw_seed])
    norm_seed_degrees.append(degrees[norm_seed])

print(f"  Same seed (raw vs normalized): {same_seed_count}/100 ({same_seed_count}%)")
print(f"  Raw-cosine seed mean degree:   {np.mean(raw_seed_degrees):.1f}")
print(f"  L2-cosine seed mean degree:    {np.mean(norm_seed_degrees):.1f}")

if same_seed_count > 90:
    print(f"  → Embeddings are ALREADY NORMALIZED or near-normalized")
    print(f"  → Norm pathway is irrelevant; mechanism must be angular")
else:
    print(f"  → Normalization CHANGES seed selection")
    print(f"  → Norm pathway contributes to the mechanism")

results["P5"] = {
    "same_seed_pct": same_seed_count,
    "raw_mean_deg": float(np.mean(raw_seed_degrees)),
    "norm_mean_deg": float(np.mean(norm_seed_degrees)),
}


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("MECHANISM FALSIFICATION SUMMARY")
print(f"{'=' * 70}")

print(f"\n  P1 (cosine-to-target vs degree):     rho = {results['P1']['rho']:+.4f}")
print(f"  P2 (norm vs degree):                  rho = {results['P2']['rho']:+.4f}")
print(f"  P3 (angular dispersion vs degree):    rho = {results['P3']['rho']:+.4f}")
print(f"  P4 (cosine seed deg vs random seed):  {results['P4']['cos_seed_mean_deg']:.1f} vs {results['P4']['rand_seed_mean_deg']:.1f}")
print(f"  P5 (same seed after normalization):   {results['P5']['same_seed_pct']}%")

print(f"\n  WHICH MODEL IS SUPPORTED?")
hub_attract = results['P1']['rho'] > 0.05 or results['P4']['cos_seed_mean_deg'] > results['P4']['rand_seed_mean_deg'] * 1.2
hub_avoid = results['P1']['rho'] < -0.05 or results['P4']['cos_seed_mean_deg'] < results['P4']['rand_seed_mean_deg'] * 0.8
ang_smear = results['P3']['rho'] > 0.1

if hub_attract:
    print(f"  → HUB ENTRAPMENT model supported (cosine finds hubs, gets stuck)")
if hub_avoid:
    print(f"  → HUB AVOIDANCE model supported (cosine bypasses hubs)")
if ang_smear:
    print(f"  → ANGULAR SMEARING confirmed (hubs have dispersed neighbor directions)")
if not hub_attract and not hub_avoid:
    print(f"  → NEITHER pure model supported; may be architecture-dependent")

with open(OUT / "exp17_mechanism_falsification.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT / 'exp17_mechanism_falsification.json'}")
print("MECHANISM FALSIFICATION COMPLETE")
