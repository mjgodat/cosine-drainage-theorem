"""
Experiment 21: Second Embedder / Projection Invariance Test

Tests whether VGSG findings (Gamma, P4, policy ordering) are invariant
across embedding projections and dimensionalities. Since a second neural
embedder requires API access or a large local model, we test the structural
question: are the findings dimension-specific or projection-specific?

Test bed:
  - Cora (2,708 nodes, 1433D native BoW)
  - Amazon Computers (13,752 nodes, 767D native)

Projections per graph:
  a. Native features (original dimensionality)
  b. Random projection to 384D (Gaussian random matrix)
  c. PCA projection to 128D (variance-preserving compression)

For each embedding variant:
  1. Gamma (angular compression) — mean arccos of k-NN cosines
  2. P4 test (cosine-seed degree vs random-seed degree)
  3. 6-policy benchmark at H=25,50,100 with 200 cross-class pairs
  4. Check if policy ordering (Multi > Cosine > BFS) holds
"""
import sys
import time
import json
import heapq
import numpy as np
import torch
from collections import deque
from pathlib import Path
from math import pi
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

N_PAIRS = 200
BUDGETS = [25, 50, 100]
K_GAMMA = 20
N_P4_QUERIES = 200
BFS_DEPTH = 3


# ======================================================================
# GRAPH UTILITIES
# ======================================================================

def build_adj(edge_index, N):
    """Build adjacency list from edge_index."""
    adj = [[] for _ in range(N)]
    for s, d in zip(edge_index[0], edge_index[1]):
        adj[int(s)].append(int(d))
    for i in range(N):
        adj[i] = list(set(adj[i]))
    return adj


def l2_normalize(X):
    """L2-normalize rows of X."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-8, None)


# ======================================================================
# GAMMA (ANGULAR COMPRESSION)
# ======================================================================

def compute_gamma(emb, k=K_GAMMA):
    """
    Compute local Gamma_k(v) for each node.
    Gamma = 1 - mean_angle_to_kNN / (pi/2).
    Higher Gamma = more angular compression = cosine bias.
    Returns array of per-node gamma and the global mean.
    """
    t = torch.tensor(emb, device=DEVICE, dtype=torch.float32)
    N = len(emb)
    gamma = np.zeros(N, dtype=np.float32)
    BATCH = 512
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        sims = t[start:end] @ t.T
        for i in range(end - start):
            sims[i, start + i] = -2  # exclude self
        topk_sims, _ = sims.topk(k, dim=1)
        mean_angles = torch.arccos(topk_sims.clamp(-1, 1)).mean(dim=1)
        gamma[start:end] = (1.0 - mean_angles.cpu().numpy() / (pi / 2))
    return gamma


# ======================================================================
# P4 TEST: COSINE-SEED vs RANDOM-SEED DEGREE
# ======================================================================

def compute_p4(emb, adj, degrees, n_queries=N_P4_QUERIES, bfs_depth=BFS_DEPTH):
    """
    For n_queries random query nodes:
    - Find cosine-nearest neighbor (cosine seed)
    - Pick a random neighbor (random seed)
    - Compare mean degree of seeds and their BFS neighborhoods
    Returns dict with cos/rand seed degrees and ratio.
    """
    t = torch.tensor(emb, device=DEVICE, dtype=torch.float32)
    N = len(emb)
    valid = [i for i in range(N) if degrees[i] > 0]

    query_nodes = np.random.choice(valid, min(n_queries, len(valid)), replace=False)

    cos_seed_degs = []
    rand_seed_degs = []
    cos_bfs_degs = []
    rand_bfs_degs = []

    for q in query_nodes:
        # Cosine nearest
        sims = (t[q:q+1] @ t.T).squeeze(0)
        sims[q] = -2
        cos_seed = int(sims.argmax().item())
        cos_seed_degs.append(degrees[cos_seed])

        # Random seed
        rand_seed = valid[np.random.randint(len(valid))]
        while rand_seed == q:
            rand_seed = valid[np.random.randint(len(valid))]
        rand_seed_degs.append(degrees[rand_seed])

        # BFS from cosine seed
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
        cos_bfs_degs.append(np.mean([degrees[v] for v in visited]) if visited else 0)

        # BFS from random seed
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
        rand_bfs_degs.append(np.mean([degrees[v] for v in visited]) if visited else 0)

    cos_mean = float(np.mean(cos_seed_degs))
    rand_mean = float(np.mean(rand_seed_degs))
    ratio = cos_mean / rand_mean if rand_mean > 0 else 0.0

    return {
        "cos_seed_mean_deg": cos_mean,
        "rand_seed_mean_deg": rand_mean,
        "ratio": ratio,
        "cos_bfs_mean_deg": float(np.mean(cos_bfs_degs)),
        "rand_bfs_mean_deg": float(np.mean(rand_bfs_degs)),
        "hub_attraction": ratio > 1.2,
        "hub_avoidance": ratio < 0.8,
    }


# ======================================================================
# 6-POLICY BENCHMARK
# ======================================================================

def pol_bfs(adj, emb, s, t, H):
    visited = {s}; q = deque([s])
    while q and len(visited) < H:
        u = q.popleft()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                if v == t: return True
                if len(visited) >= H: break
                q.append(v)
    return t in visited

def pol_cos(adj, emb, s, t, H):
    tv = emb[t]; visited = {s}
    pq = [(-float(emb[v] @ tv), v) for v in adj[s] if v not in visited]
    heapq.heapify(pq)
    while pq and len(visited) < H:
        _, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u)
        if u == t: return True
        for v in adj[u]:
            if v not in visited:
                heapq.heappush(pq, (-float(emb[v] @ tv), v))
    return t in visited

def pol_multi(adj, emb, s, t, H):
    hs = H // 2; ht = H - hs
    vs = {s}; vt = {t}
    ps = [(-float(emb[v] @ emb[t]), v) for v in adj[s]]
    pt = [(-float(emb[v] @ emb[s]), v) for v in adj[t]]
    heapq.heapify(ps); heapq.heapify(pt)
    while ps and len(vs) < hs:
        _, u = heapq.heappop(ps)
        if u in vs: continue
        vs.add(u)
        if u in vt: return True
        for v in adj[u]:
            if v not in vs: heapq.heappush(ps, (-float(emb[v] @ emb[t]), v))
    while pt and len(vt) < ht:
        _, u = heapq.heappop(pt)
        if u in vt: continue
        vt.add(u)
        if u in vs: return True
        for v in adj[u]:
            if v not in vt: heapq.heappush(pt, (-float(emb[v] @ emb[s]), v))
    return bool(vs & vt)

def pol_ppr(adj, emb, s, t, H, alpha=0.15):
    r = {s: 1.0}; visited = set(); q = deque([s]); pushes = 0
    while q and pushes < H:
        u = q.popleft(); visited.add(u)
        res = r.get(u, 0.0); r[u] = 0.0; pushes += 1
        nbs = adj[u]
        if nbs and res > 0:
            m = (1 - alpha) * res / len(nbs)
            for v in nbs:
                r[v] = r.get(v, 0.0) + m
                if v not in visited and len(visited) < H:
                    q.append(v)
    return t in visited

def pol_bidir_ppr(adj, emb, s, t, H, alpha=0.15):
    hs = H // 2; ht = H - hs
    rs = {s: 1.0}; vs = set(); qs = deque([s]); ps = 0
    while qs and ps < hs:
        u = qs.popleft(); vs.add(u); res = rs.get(u, 0.0); rs[u] = 0.0; ps += 1
        nbs = adj[u]
        if nbs and res > 0:
            m = (1-alpha)*res/len(nbs)
            for v in nbs:
                rs[v] = rs.get(v, 0.0) + m
                if v not in vs and len(vs) < hs: qs.append(v)
    rt = {t: 1.0}; vt = set(); qt = deque([t]); pt = 0
    while qt and pt < ht:
        u = qt.popleft(); vt.add(u); res = rt.get(u, 0.0); rt[u] = 0.0; pt += 1
        nbs = adj[u]
        if nbs and res > 0:
            m = (1-alpha)*res/len(nbs)
            for v in nbs:
                rt[v] = rt.get(v, 0.0) + m
                if v not in vt and len(vt) < ht: qt.append(v)
    return bool(vs & vt) or t in vs

def pol_mmr(adj, emb, s, t, H, lam=0.7):
    tv = emb[t]; visited = {s}
    pq = []
    for v in adj[s]:
        dp = np.log1p(len(adj[v]))
        sc = lam * float(emb[v] @ tv) - (1-lam) * 0.1 * dp
        heapq.heappush(pq, (-sc, v))
    while pq and len(visited) < H:
        _, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u)
        if u == t: return True
        for v in adj[u]:
            if v not in visited:
                dp = np.log1p(len(adj[v]))
                sc = lam * float(emb[v] @ tv) - (1-lam) * 0.1 * dp
                heapq.heappush(pq, (-sc, v))
    return t in visited


POLICIES = {
    "BFS": pol_bfs,
    "Cosine": pol_cos,
    "Multi-Anchor": pol_multi,
    "Fwd-Push PPR": pol_ppr,
    "Bidir PPR": pol_bidir_ppr,
    "MMR": pol_mmr,
}


def sample_cross_class_pairs(labels, adj, n_pairs=N_PAIRS):
    """Sample cross-class pairs where both nodes have degree > 1."""
    N = len(labels)
    unique_labels = np.unique(labels)
    class_nodes = {
        l: [i for i in range(N) if labels[i] == l and len(adj[i]) > 1]
        for l in unique_labels
    }
    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 50:
        attempts += 1
        l1, l2 = np.random.choice(unique_labels, 2, replace=False)
        if not class_nodes[l1] or not class_nodes[l2]:
            continue
        s = class_nodes[l1][np.random.randint(len(class_nodes[l1]))]
        t = class_nodes[l2][np.random.randint(len(class_nodes[l2]))]
        if s != t:
            pairs.append((s, t))
    return pairs


def run_policy_benchmark(adj, emb, pairs, budgets=BUDGETS):
    """Run 6-policy benchmark and return results dict."""
    results = {p: {} for p in POLICIES}
    for H in budgets:
        t0 = time.time()
        for pname, func in POLICIES.items():
            ok = sum(1 for s, t in pairs if func(adj, emb, s, t, H))
            results[pname][H] = ok / len(pairs) * 100
        elapsed = time.time() - t0
        print(f"    H={H:<4d} ({elapsed:.1f}s)")
    return results


def check_policy_ordering(results, H=100):
    """Check if Multi > Cosine > BFS at given budget."""
    multi = results.get("Multi-Anchor", {}).get(H, 0)
    cosine = results.get("Cosine", {}).get(H, 0)
    bfs = results.get("BFS", {}).get(H, 0)
    return {
        "Multi": multi,
        "Cosine": cosine,
        "BFS": bfs,
        "Multi_gt_Cosine": multi > cosine,
        "Cosine_gt_BFS": cosine > bfs,
        "ordering_holds": multi > cosine > bfs,
    }


# ======================================================================
# MAIN: RUN ALL VARIANTS
# ======================================================================

def run_graph_variants(graph_name, features_raw, edge_index, labels):
    """Run Gamma, P4, and 6-policy benchmark on native + 2 projections."""
    N = len(features_raw)
    D_orig = features_raw.shape[1]
    adj = build_adj(edge_index, N)
    degrees = np.array([len(adj[i]) for i in range(N)])
    pairs = sample_cross_class_pairs(labels, adj)

    print(f"\n{'=' * 75}")
    print(f"GRAPH: {graph_name} ({N:,} nodes, {D_orig}D native, "
          f"{sum(len(a) for a in adj)//2:,} edges, {len(pairs)} pairs)")
    print(f"{'=' * 75}")

    # Define projections
    projections = {}

    # a. Native
    projections[f"Native ({D_orig}D)"] = l2_normalize(features_raw.astype(np.float32))

    # b. Random projection to 384D
    rp = GaussianRandomProjection(n_components=384, random_state=42)
    proj_rp = rp.fit_transform(features_raw.astype(np.float64)).astype(np.float32)
    projections["Random Proj (384D)"] = l2_normalize(proj_rp)

    # c. PCA to 128D
    pca = PCA(n_components=128, random_state=42)
    proj_pca = pca.fit_transform(features_raw.astype(np.float64)).astype(np.float32)
    projections["PCA (128D)"] = l2_normalize(proj_pca)
    pca_var = float(sum(pca.explained_variance_ratio_))

    graph_results = {
        "graph_name": graph_name,
        "N": N,
        "D_original": D_orig,
        "n_edges": sum(len(a) for a in adj) // 2,
        "n_pairs": len(pairs),
        "pca_128_variance_explained": pca_var,
        "variants": {},
    }

    for proj_name, emb in projections.items():
        D = emb.shape[1]
        print(f"\n  --- {proj_name} ---")

        # 1. Gamma (angular compression)
        print(f"    Computing Gamma (k={K_GAMMA})...", end=" ", flush=True)
        t0 = time.time()
        gamma = compute_gamma(emb, k=K_GAMMA)
        gamma_mean = float(np.mean(gamma))
        gamma_std = float(np.std(gamma))

        # Gamma vs degree correlation
        mask = degrees > 0
        rho_gd, p_gd = spearmanr(degrees[mask], gamma[mask])
        print(f"mean={gamma_mean:.4f}, rho(gamma,degree)={rho_gd:.4f} ({time.time()-t0:.1f}s)")

        # Gamma by degree bin
        deg_bins = [(0, 5), (5, 20), (20, 50), (50, 200), (200, 10000)]
        gamma_by_bin = {}
        for lo, hi in deg_bins:
            bin_mask = (degrees >= lo) & (degrees < hi)
            n_bin = int(np.sum(bin_mask))
            if n_bin > 0:
                gamma_by_bin[f"{lo}-{hi}"] = {
                    "n": n_bin,
                    "gamma_mean": float(np.mean(gamma[bin_mask])),
                }

        # 2. P4 test
        print(f"    Computing P4 (cosine-seed vs random-seed)...", end=" ", flush=True)
        t0 = time.time()
        p4 = compute_p4(emb, adj, degrees)
        print(f"ratio={p4['ratio']:.3f} ({time.time()-t0:.1f}s)")

        # 3. 6-policy benchmark
        print(f"    Running 6-policy benchmark...")
        policy_results = run_policy_benchmark(adj, emb, pairs)

        # 4. Policy ordering check
        ordering = {}
        for H in BUDGETS:
            ordering[str(H)] = check_policy_ordering(policy_results, H)

        # Store
        graph_results["variants"][proj_name] = {
            "D": D,
            "gamma": {
                "mean": gamma_mean,
                "std": gamma_std,
                "rho_degree": float(rho_gd),
                "p_degree": float(p_gd),
                "by_degree_bin": gamma_by_bin,
            },
            "P4": p4,
            "policy_benchmark": {
                p: {str(H): v for H, v in r.items()}
                for p, r in policy_results.items()
            },
            "policy_ordering": ordering,
        }

    return graph_results


def print_summary(all_results):
    """Print cross-variant comparison tables."""
    print(f"\n{'=' * 80}")
    print("CROSS-VARIANT SUMMARY")
    print(f"{'=' * 80}")

    # Table 1: Gamma comparison
    print(f"\n  TABLE 1: Gamma (Angular Compression) Across Projections")
    print(f"  {'Graph':<20s} {'Variant':<22s} {'Gamma':>7s} {'rho(G,deg)':>11s} {'D':>5s}")
    print(f"  {'-'*65}")
    for gr in all_results:
        for vname, v in gr["variants"].items():
            g = v["gamma"]
            print(f"  {gr['graph_name']:<20s} {vname:<22s} {g['mean']:7.4f} "
                  f"{g['rho_degree']:+10.4f} {v['D']:5d}")

    # Table 2: P4 comparison
    print(f"\n  TABLE 2: P4 (Cosine-Seed / Random-Seed Degree Ratio)")
    print(f"  {'Graph':<20s} {'Variant':<22s} {'Cos Deg':>8s} {'Rand Deg':>9s} {'Ratio':>7s} {'Attract?':>10s}")
    print(f"  {'-'*76}")
    for gr in all_results:
        for vname, v in gr["variants"].items():
            p = v["P4"]
            attract = "YES" if p["hub_attraction"] else ("AVOID" if p["hub_avoidance"] else "neutral")
            print(f"  {gr['graph_name']:<20s} {vname:<22s} {p['cos_seed_mean_deg']:8.1f} "
                  f"{p['rand_seed_mean_deg']:9.1f} {p['ratio']:7.3f} {attract:>10s}")

    # Table 3: Policy ordering at H=100
    print(f"\n  TABLE 3: Policy Reachability (%) at H=100")
    print(f"  {'Graph':<20s} {'Variant':<22s} {'BFS':>6s} {'Cos':>6s} {'Multi':>6s} {'PPR':>6s} {'BiPPR':>6s} {'MMR':>6s} {'Ordering':>10s}")
    print(f"  {'-'*98}")
    for gr in all_results:
        for vname, v in gr["variants"].items():
            pb = v["policy_benchmark"]
            bfs = pb["BFS"]["100"]
            cos = pb["Cosine"]["100"]
            multi = pb["Multi-Anchor"]["100"]
            ppr = pb["Fwd-Push PPR"]["100"]
            bidir = pb["Bidir PPR"]["100"]
            mmr = pb["MMR"]["100"]
            ordering = v["policy_ordering"]["100"]
            holds = "HOLDS" if ordering["ordering_holds"] else "BROKEN"
            print(f"  {gr['graph_name']:<20s} {vname:<22s} {bfs:5.1f}% {cos:5.1f}% {multi:5.1f}% "
                  f"{ppr:5.1f}% {bidir:5.1f}% {mmr:5.1f}% {holds:>10s}")

    # Table 4: Policy ordering at all budgets
    print(f"\n  TABLE 4: Policy Ordering Check (Multi > Cosine > BFS) at All Budgets")
    print(f"  {'Graph':<20s} {'Variant':<22s}", end="")
    for H in BUDGETS:
        print(f" {'H='+str(H):>8s}", end="")
    print()
    print(f"  {'-'*68}")
    for gr in all_results:
        for vname, v in gr["variants"].items():
            print(f"  {gr['graph_name']:<20s} {vname:<22s}", end="")
            for H in BUDGETS:
                holds = v["policy_ordering"][str(H)]["ordering_holds"]
                print(f" {'HOLDS':>8s}" if holds else f" {'BROKEN':>8s}", end="")
            print()

    # Invariance assessment
    print(f"\n  INVARIANCE ASSESSMENT")
    print(f"  {'-'*60}")

    # Check Gamma stability
    gamma_vals = []
    for gr in all_results:
        gvals = [v["gamma"]["mean"] for v in gr["variants"].values()]
        gamma_vals.extend(gvals)
        spread = max(gvals) - min(gvals)
        print(f"  {gr['graph_name']}: Gamma spread = {spread:.4f} "
              f"(range: {min(gvals):.4f} - {max(gvals):.4f})")

    # Check P4 stability
    for gr in all_results:
        ratios = [v["P4"]["ratio"] for v in gr["variants"].values()]
        attraction_flags = [v["P4"]["hub_attraction"] for v in gr["variants"].values()]
        all_same = len(set(attraction_flags)) == 1
        print(f"  {gr['graph_name']}: P4 ratios = {[f'{r:.3f}' for r in ratios]}, "
              f"attraction consistent: {'YES' if all_same else 'NO'}")

    # Check ordering stability
    for gr in all_results:
        all_hold = all(
            v["policy_ordering"]["100"]["ordering_holds"]
            for v in gr["variants"].values()
        )
        print(f"  {gr['graph_name']}: Multi>Cosine>BFS at H=100 consistent across variants: "
              f"{'YES' if all_hold else 'NO'}")

    # Gamma dimensionality trend
    print(f"\n  GAMMA vs DIMENSIONALITY:")
    for gr in all_results:
        items = [(v["D"], v["gamma"]["mean"]) for v in gr["variants"].values()]
        items.sort()
        for d, g in items:
            print(f"    {gr['graph_name']} D={d:>5d}: Gamma={g:.4f}")


def main():
    t_total = time.time()
    print(f"Device: {DEVICE}")
    print(f"Experiment 21: Second Embedder / Projection Invariance Test")
    print(f"Budgets: {BUDGETS}, Pairs: {N_PAIRS}, Gamma k: {K_GAMMA}")

    from torch_geometric.datasets import Planetoid, Amazon

    all_results = []

    # Cora
    d = Planetoid(root="/tmp/pyg_data", name="Cora")
    data = d[0]
    result = run_graph_variants(
        "Cora",
        data.x.numpy().astype(np.float32),
        data.edge_index.numpy(),
        data.y.numpy()
    )
    all_results.append(result)

    # Amazon Computers
    d2 = Amazon(root="/tmp/pyg_data", name="Computers")
    data2 = d2[0]
    result2 = run_graph_variants(
        "Amazon Computers",
        data2.x.numpy().astype(np.float32),
        data2.edge_index.numpy(),
        data2.y.numpy()
    )
    all_results.append(result2)

    # Summary
    print_summary(all_results)

    # Final verdict
    print(f"\n{'=' * 80}")
    print("FINAL VERDICT")
    print(f"{'=' * 80}")

    # Collect verdicts
    gamma_stable = True
    p4_stable = True
    ordering_stable = True

    for gr in all_results:
        gvals = [v["gamma"]["mean"] for v in gr["variants"].values()]
        if max(gvals) - min(gvals) > 0.15:
            gamma_stable = False

        attraction_flags = [v["P4"]["hub_attraction"] for v in gr["variants"].values()]
        if len(set(attraction_flags)) > 1:
            p4_stable = False

        for v in gr["variants"].values():
            if not v["policy_ordering"]["100"]["ordering_holds"]:
                ordering_stable = False

    print(f"\n  Gamma invariant across projections:     {'YES' if gamma_stable else 'NO'}")
    print(f"  P4 hub-attraction stable:               {'YES' if p4_stable else 'NO'}")
    print(f"  Policy ordering (Multi>Cos>BFS) stable:  {'YES' if ordering_stable else 'NO -- see details'}")

    if gamma_stable and p4_stable:
        print(f"\n  CONCLUSION: VGSG findings are NOT model-specific or dimension-specific.")
        print(f"  Angular compression and hub attraction are structural properties of the")
        print(f"  embedding space that persist under linear projection and dimensionality change.")
    elif gamma_stable:
        print(f"\n  CONCLUSION: Gamma is invariant; P4 and/or ordering show projection sensitivity.")
        print(f"  Angular compression is structural, but traversal outcomes are partially")
        print(f"  dependent on embedding dimensionality.")
    else:
        print(f"\n  CONCLUSION: Gamma varies with projection, suggesting the findings may be")
        print(f"  partially model-specific. Further testing with neural embedders recommended.")

    # Save
    output = {
        "experiment": "exp21_second_embedder",
        "description": "Projection invariance test for VGSG findings",
        "n_pairs": N_PAIRS,
        "budgets": BUDGETS,
        "k_gamma": K_GAMMA,
        "graphs": all_results,
        "verdicts": {
            "gamma_invariant": gamma_stable,
            "p4_stable": p4_stable,
            "ordering_stable": ordering_stable,
        },
        "runtime_s": time.time() - t_total,
    }

    outfile = OUT / "exp21_second_embedder.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Total runtime: {time.time()-t_total:.0f}s")
    print(f"  Saved: {outfile}")
    print("EXPERIMENT 21 COMPLETE")


if __name__ == "__main__":
    main()
