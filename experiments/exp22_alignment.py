"""
Experiment 22: Alignment Stratification — Multi-Anchor Across Embedding-Topology Regimes

Tests whether Multi-Anchor intervention works in BOTH alignment regimes:
  - Low alignment  (embedding cosine neighbors != graph neighbors): Moderate Similarity Trap
  - High alignment (embedding cosine neighbors == graph neighbors): Hub Entrapment

Measures:
  P4 ratio: mean degree of cosine-seeded top-20 / mean degree of random top-20
  MI overlap: mutual information between cosine kNN(k=20) and graph edges
  Multi-Anchor improvement: hit rate gain over cosine-single at each budget

Combines with known NeuroCrystal baseline (P4=0.47, MI=0.2%).
"""
import sys
import time
import json
import heapq
import numpy as np
from collections import deque
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

N_PAIRS = 300
BUDGETS = [25, 50, 100]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_adj_from_edge_index(edge_index, N):
    adj = [[] for _ in range(N)]
    for s, d in zip(edge_index[0], edge_index[1]):
        adj[int(s)].append(int(d))
    for i in range(N):
        adj[i] = list(set(adj[i]))
    return adj


# ---------------------------------------------------------------------------
# Alignment measures
# ---------------------------------------------------------------------------
def compute_p4_ratio(emb, adj, N, k=20, n_samples=2000):
    """P4 = mean_degree(cosine-seeded top-k) / mean_degree(random top-k).
    Values > 1 mean cosine neighbors are higher-degree (hub-aligned).
    Values < 1 mean cosine avoids hubs."""
    degrees = np.array([len(adj[i]) for i in range(N)], dtype=np.float64)
    sample_idx = np.random.choice(N, min(n_samples, N), replace=False)

    cos_deg_sum = 0.0
    rand_deg_sum = 0.0
    count = 0

    for i in sample_idx:
        # cosine kNN
        sims = emb @ emb[i]
        sims[i] = -np.inf
        topk = np.argpartition(sims, -k)[-k:]
        cos_deg_sum += degrees[topk].mean()

        # random kNN (same k)
        rand_k = np.random.choice(N, k, replace=False)
        rand_deg_sum += degrees[rand_k].mean()
        count += 1

    p4 = (cos_deg_sum / count) / max(rand_deg_sum / count, 1e-8)
    return float(p4)


def compute_mi_overlap(emb, adj, N, k=20, n_samples=2000):
    """Network MI: fraction of cosine kNN(k=20) that are also graph neighbors.
    High MI = embedding and topology are aligned."""
    sample_idx = np.random.choice(N, min(n_samples, N), replace=False)
    overlap_total = 0
    k_total = 0

    for i in sample_idx:
        graph_nbs = set(adj[i])
        if not graph_nbs:
            continue
        sims = emb @ emb[i]
        sims[i] = -np.inf
        topk = set(np.argpartition(sims, -k)[-k:].tolist())
        overlap_total += len(topk & graph_nbs)
        k_total += k

    mi_pct = overlap_total / max(k_total, 1) * 100
    return float(mi_pct)


# ---------------------------------------------------------------------------
# Policies (same as exp13)
# ---------------------------------------------------------------------------
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
            m = (1 - alpha) * res / len(nbs)
            for v in nbs:
                rs[v] = rs.get(v, 0.0) + m
                if v not in vs and len(vs) < hs: qs.append(v)
    rt = {t: 1.0}; vt = set(); qt = deque([t]); pt = 0
    while qt and pt < ht:
        u = qt.popleft(); vt.add(u); res = rt.get(u, 0.0); rt[u] = 0.0; pt += 1
        nbs = adj[u]
        if nbs and res > 0:
            m = (1 - alpha) * res / len(nbs)
            for v in nbs:
                rt[v] = rt.get(v, 0.0) + m
                if v not in vt and len(vt) < ht: qt.append(v)
    return bool(vs & vt) or t in vs


def pol_mmr(adj, emb, s, t, H, lam=0.7):
    tv = emb[t]; visited = {s}
    pq = []
    for v in adj[s]:
        dp = np.log1p(len(adj[v]))
        sc = lam * float(emb[v] @ tv) - (1 - lam) * 0.1 * dp
        heapq.heappush(pq, (-sc, v))
    while pq and len(visited) < H:
        _, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u)
        if u == t: return True
        for v in adj[u]:
            if v not in visited:
                dp = np.log1p(len(adj[v]))
                sc = lam * float(emb[v] @ tv) - (1 - lam) * 0.1 * dp
                heapq.heappush(pq, (-sc, v))
    return t in visited


# ---------------------------------------------------------------------------
# Per-graph runner
# ---------------------------------------------------------------------------
def run_graph(name, features, edge_index, labels):
    N = len(features)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    emb = features / np.clip(norms, 1e-8, None)
    adj = build_adj_from_edge_index(edge_index, N)

    # -- Alignment measures --
    print(f"\n{'=' * 72}")
    print(f"GRAPH: {name} ({N:,} nodes, {edge_index.shape[1]:,} edges)")
    print(f"{'=' * 72}")

    t0 = time.time()
    p4 = compute_p4_ratio(emb, adj, N)
    mi = compute_mi_overlap(emb, adj, N)
    mean_deg = np.mean([len(adj[i]) for i in range(N)])
    print(f"  Alignment measures ({time.time()-t0:.1f}s):")
    print(f"    P4 ratio (cos-deg / rand-deg):  {p4:.3f}")
    print(f"    MI overlap (cos kNN vs graph):   {mi:.2f}%")
    print(f"    Mean degree:                     {mean_deg:.1f}")

    # -- Sample cross-class pairs --
    unique_labels = np.unique(labels)
    class_nodes = {int(l): [i for i in range(N) if labels[i] == l and len(adj[i]) > 1]
                   for l in unique_labels}

    pairs = []
    attempts = 0
    while len(pairs) < N_PAIRS and attempts < N_PAIRS * 50:
        attempts += 1
        l1, l2 = np.random.choice(unique_labels, 2, replace=False)
        l1, l2 = int(l1), int(l2)
        if not class_nodes[l1] or not class_nodes[l2]:
            continue
        s = class_nodes[l1][np.random.randint(len(class_nodes[l1]))]
        t = class_nodes[l2][np.random.randint(len(class_nodes[l2]))]
        if s != t:
            pairs.append((s, t))
    print(f"  Cross-class pairs: {len(pairs)}")

    # -- Run 6-policy benchmark --
    policies = {
        "BFS": lambda s, t, H: pol_bfs(adj, emb, s, t, H),
        "Cosine-Single": lambda s, t, H: pol_cos(adj, emb, s, t, H),
        "Multi-Anchor": lambda s, t, H: pol_multi(adj, emb, s, t, H),
        "Fwd-Push PPR": lambda s, t, H: pol_ppr(adj, emb, s, t, H),
        "Bidir PPR": lambda s, t, H: pol_bidir_ppr(adj, emb, s, t, H),
        "MMR": lambda s, t, H: pol_mmr(adj, emb, s, t, H),
    }

    results = {p: {} for p in policies}
    for H in BUDGETS:
        t0 = time.time()
        for pname, func in policies.items():
            ok = sum(1 for s, t in pairs if func(s, t, H))
            results[pname][H] = ok / len(pairs) * 100
        print(f"  H={H:<4d} ({time.time()-t0:.1f}s)")

    # Print table
    print(f"\n  {'Policy':<20s}", end="")
    for H in BUDGETS:
        print(f"{'H='+str(H):>8s}", end="")
    print()
    print(f"  {'-' * 20}", end="")
    for _ in BUDGETS:
        print(f"{'-' * 8}", end="")
    print()
    for pname in policies:
        print(f"  {pname:<20s}", end="")
        for H in BUDGETS:
            print(f"{results[pname][H]:7.1f}%", end="")
        print()

    # -- Multi-Anchor improvement over Cosine-Single --
    improvement = {}
    for H in BUDGETS:
        cos_val = results["Cosine-Single"][H]
        multi_val = results["Multi-Anchor"][H]
        imp_pp = multi_val - cos_val  # percentage point gain
        imp_rel = (multi_val - cos_val) / max(cos_val, 0.1) * 100  # relative gain
        improvement[H] = {"pp": imp_pp, "rel_pct": imp_rel}

    print(f"\n  Multi-Anchor improvement over Cosine-Single:")
    print(f"  {'Budget':>8s} {'Cosine':>8s} {'Multi':>8s} {'Gain pp':>8s} {'Gain %':>8s}")
    for H in BUDGETS:
        cos = results["Cosine-Single"][H]
        mul = results["Multi-Anchor"][H]
        pp = improvement[H]["pp"]
        rel = improvement[H]["rel_pct"]
        print(f"  {'H='+str(H):>8s} {cos:7.1f}% {mul:7.1f}% {pp:+7.1f}  {rel:+7.1f}%")

    return {
        "name": name,
        "N": N,
        "n_edges": int(edge_index.shape[1]),
        "mean_degree": float(mean_deg),
        "n_pairs": len(pairs),
        "alignment": {
            "p4_ratio": p4,
            "mi_overlap_pct": mi,
        },
        "results": {p: {str(H): v for H, v in r.items()} for p, r in results.items()},
        "multi_anchor_improvement": {str(H): v for H, v in improvement.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_total = time.time()
    all_results = []

    from torch_geometric.datasets import Planetoid, Amazon, CitationFull

    for cls, kwargs, name in [
        (Planetoid, {"root": "/tmp/pyg_data", "name": "Cora"}, "Cora"),
        (Amazon, {"root": "/tmp/pyg_data", "name": "Computers"}, "Amazon Computers"),
        (CitationFull, {"root": "/tmp/pyg_data", "name": "DBLP"}, "DBLP"),
    ]:
        d = cls(**kwargs)
        data = d[0]
        all_results.append(run_graph(
            name,
            data.x.numpy().astype(np.float32),
            data.edge_index.numpy(),
            data.y.numpy()
        ))

    # -- Add NeuroCrystal known results --
    neurocrystal = {
        "name": "NeuroCrystal",
        "N": 40204,
        "n_edges": 6000000,
        "mean_degree": 298.0,
        "n_pairs": 300,
        "alignment": {
            "p4_ratio": 0.47,
            "mi_overlap_pct": 0.2,
        },
        "results": {
            "Cosine-Single": {"25": 14.0, "50": 26.0, "100": 45.0},
            "Multi-Anchor": {"25": 18.0, "50": 42.0, "100": 72.0},
        },
        "multi_anchor_improvement": {
            "25": {"pp": 4.0, "rel_pct": 28.6},
            "50": {"pp": 16.0, "rel_pct": 61.5},
            "100": {"pp": 27.0, "rel_pct": 60.0},
        },
        "note": "Known baseline values from prior experiments"
    }
    all_results.append(neurocrystal)

    # -----------------------------------------------------------------------
    # ALIGNMENT STRATIFICATION TABLE
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("ALIGNMENT STRATIFICATION: Multi-Anchor Gain vs Embedding-Topology Alignment")
    print(f"{'=' * 72}")

    # Sort by MI overlap (alignment measure)
    sorted_res = sorted(all_results, key=lambda r: r["alignment"]["mi_overlap_pct"])

    print(f"\n  {'Graph':<20s} {'P4':>6s} {'MI%':>6s} {'Regime':<15s}", end="")
    for H in BUDGETS:
        print(f"  {'Multi pp@'+str(H):>12s}", end="")
    print()
    print(f"  {'-' * 20} {'-' * 6} {'-' * 6} {'-' * 15}", end="")
    for _ in BUDGETS:
        print(f"  {'-' * 12}", end="")
    print()

    for r in sorted_res:
        mi = r["alignment"]["mi_overlap_pct"]
        p4 = r["alignment"]["p4_ratio"]
        # Classify regime
        if mi < 5.0:
            regime = "Low (Sim Trap)"
        elif mi < 20.0:
            regime = "Mid"
        else:
            regime = "High (Hub Trap)"

        print(f"  {r['name']:<20s} {p4:5.2f}  {mi:5.1f}  {regime:<15s}", end="")
        for H in BUDGETS:
            imp = r["multi_anchor_improvement"].get(str(H), {})
            if isinstance(imp, dict):
                pp = imp.get("pp", 0)
            else:
                pp = 0
            print(f"  {pp:+11.1f}pp", end="")
        print()

    # -----------------------------------------------------------------------
    # KEY FINDINGS
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("KEY FINDINGS")
    print(f"{'=' * 72}")

    # Check if Multi-Anchor wins across ALL alignment levels
    all_positive = True
    for r in sorted_res:
        for H in BUDGETS:
            imp = r["multi_anchor_improvement"].get(str(H), {})
            if isinstance(imp, dict) and imp.get("pp", 0) <= 0:
                all_positive = False

    if all_positive:
        print("  [CONFIRMED] Multi-Anchor improvement is POSITIVE across ALL alignment regimes.")
    else:
        print("  [MIXED] Multi-Anchor improvement varies by alignment regime.")

    # Correlation: does gain vary with alignment?
    mi_vals = []
    gain_vals = []
    for r in sorted_res:
        mi_vals.append(r["alignment"]["mi_overlap_pct"])
        imp = r["multi_anchor_improvement"].get("100", {})
        if isinstance(imp, dict):
            gain_vals.append(imp.get("pp", 0))
        else:
            gain_vals.append(0)

    if len(mi_vals) >= 3:
        corr = np.corrcoef(mi_vals, gain_vals)[0, 1]
        print(f"  Correlation(MI%, Multi gain pp @H=100): r = {corr:+.3f}")
        if abs(corr) < 0.5:
            print("  [ROBUST] Multi-Anchor gain is approximately stable across alignment regimes.")
        elif corr > 0.5:
            print("  [TREND] Multi-Anchor gains INCREASE with alignment (helps more in Hub Trap).")
        else:
            print("  [TREND] Multi-Anchor gains DECREASE with alignment (helps more in Sim Trap).")

    # Regime-specific analysis
    print(f"\n  Regime-Specific Analysis (H=100):")
    for r in sorted_res:
        mi = r["alignment"]["mi_overlap_pct"]
        p4 = r["alignment"]["p4_ratio"]
        imp100 = r["multi_anchor_improvement"].get("100", {})
        pp = imp100.get("pp", 0) if isinstance(imp100, dict) else 0
        rel = imp100.get("rel_pct", 0) if isinstance(imp100, dict) else 0

        if mi < 5:
            failure = "Moderate Similarity Trap (cosine avoids hubs, drifts)"
        elif mi < 20:
            failure = "Mixed regime"
        else:
            failure = "Hub Entrapment (cosine finds hubs, gets stuck)"

        print(f"    {r['name']} (MI={mi:.1f}%, P4={p4:.2f}):")
        print(f"      Failure mode: {failure}")
        print(f"      Multi-Anchor gain: {pp:+.1f}pp ({rel:+.1f}% relative)")
        if pp > 0:
            print(f"      WHY IT WORKS: Target-side frontier escapes the trap from the opposite direction")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    output = {
        "experiment": "exp22_alignment_stratification",
        "description": "Multi-Anchor effectiveness stratified by embedding-topology alignment",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {"n_pairs": N_PAIRS, "budgets": BUDGETS, "seed": 42},
        "graphs": all_results,
        "alignment_summary": {
            r["name"]: {
                "p4_ratio": r["alignment"]["p4_ratio"],
                "mi_overlap_pct": r["alignment"]["mi_overlap_pct"],
                "multi_gain_pp_H100": (
                    r["multi_anchor_improvement"]["100"]["pp"]
                    if isinstance(r["multi_anchor_improvement"].get("100", {}), dict)
                    else 0
                ),
            }
            for r in sorted_res
        },
        "conclusion": {
            "multi_anchor_universal": all_positive,
            "correlation_mi_vs_gain": float(corr) if len(mi_vals) >= 3 else None,
        },
        "total_time_s": time.time() - t_total,
    }

    outpath = OUT / "exp22_alignment_stratification.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nTotal: {time.time() - t_total:.0f}s")
    print(f"Saved: {outpath}")
    print("EXPERIMENT 22 COMPLETE")


if __name__ == "__main__":
    main()
