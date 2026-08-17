"""
Experiment 13: Cross-Graph Replication — 6-Policy Benchmark on Public Graphs

Run the same 6-policy benchmark from Experiment 9 on Cora, Amazon Computers,
and DBLP. Confirms whether dual-frontier dominance holds across graph types.
"""
import sys
import time
import json
import heapq
import numpy as np
from collections import deque, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

N_PAIRS = 300
BUDGETS = [25, 50, 100, 200]


def build_adj_from_edge_index(edge_index, N):
    adj = [[] for _ in range(N)]
    for s, d in zip(edge_index[0], edge_index[1]):
        adj[int(s)].append(int(d))
    for i in range(N):
        adj[i] = list(set(adj[i]))
    return adj


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


def run_graph(name, features, edge_index, labels):
    N = len(features)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    emb = features / np.clip(norms, 1e-8, None)
    adj = build_adj_from_edge_index(edge_index, N)

    # Sample cross-class pairs
    unique_labels = np.unique(labels)
    class_nodes = {l: [i for i in range(N) if labels[i] == l and len(adj[i]) > 1] for l in unique_labels}

    pairs = []
    attempts = 0
    while len(pairs) < N_PAIRS and attempts < N_PAIRS * 50:
        attempts += 1
        l1, l2 = np.random.choice(unique_labels, 2, replace=False)
        if not class_nodes[l1] or not class_nodes[l2]: continue
        s = class_nodes[l1][np.random.randint(len(class_nodes[l1]))]
        t = class_nodes[l2][np.random.randint(len(class_nodes[l2]))]
        if s != t:
            pairs.append((s, t))

    print(f"\n{'=' * 70}")
    print(f"GRAPH: {name} ({N:,} nodes, {len(pairs)} pairs)")
    print(f"{'=' * 70}")

    policies = {
        "BFS": lambda s,t,H: pol_bfs(adj, emb, s, t, H),
        "Cosine": lambda s,t,H: pol_cos(adj, emb, s, t, H),
        "Multi-Anchor": lambda s,t,H: pol_multi(adj, emb, s, t, H),
        "Fwd-Push PPR": lambda s,t,H: pol_ppr(adj, emb, s, t, H),
        "Bidir PPR": lambda s,t,H: pol_bidir_ppr(adj, emb, s, t, H),
        "MMR": lambda s,t,H: pol_mmr(adj, emb, s, t, H),
    }

    results = {p: {} for p in policies}
    for H in BUDGETS:
        t0 = time.time()
        for pname, func in policies.items():
            ok = sum(1 for s, t in pairs if func(s, t, H))
            results[pname][H] = ok / len(pairs) * 100
        print(f"  H={H:<4d} ({time.time()-t0:.1f}s)")

    print(f"\n  {'Policy':<20s}", end="")
    for H in BUDGETS: print(f"{'H='+str(H):>8s}", end="")
    print()
    print(f"  {'-'*20}", end="")
    for _ in BUDGETS: print(f"{'-'*8}", end="")
    print()
    for pname in policies:
        print(f"  {pname:<20s}", end="")
        for H in BUDGETS: print(f"{results[pname][H]:7.1f}%", end="")
        print()

    return {"name": name, "N": N, "n_pairs": len(pairs), "results": {p: {str(H): v for H, v in r.items()} for p, r in results.items()}}


def main():
    t_total = time.time()
    all_results = []

    from torch_geometric.datasets import Planetoid, Amazon, CitationFull

    for cls, kwargs, name in [
        (Planetoid, {"root": "/tmp/pyg_data", "name": "Cora"}, "Cora"),
        (Amazon, {"root": "/tmp/pyg_data", "name": "Computers"}, "Amazon Computers"),
        (CitationFull, {"root": "/tmp/pyg_data", "name": "DBLP"}, "DBLP"),
    ]:
        d = cls(**kwargs); data = d[0]
        all_results.append(run_graph(name,
            data.x.numpy().astype(np.float32),
            data.edge_index.numpy(),
            data.y.numpy()))

    # Cross-graph summary
    print(f"\n{'=' * 70}")
    print("CROSS-GRAPH REPLICATION SUMMARY (H=100)")
    print(f"{'=' * 70}")
    print(f"  {'Graph':<20s} {'BFS':>6s} {'Cosine':>8s} {'Multi':>7s} {'BiPPR':>7s} {'MMR':>6s} {'Best':>15s}")
    for r in all_results:
        res = r["results"]
        h = "100"
        bfs = res["BFS"].get(h, 0)
        cos = res["Cosine"].get(h, 0)
        mul = res["Multi-Anchor"].get(h, 0)
        bip = res["Bidir PPR"].get(h, 0)
        mmr = res["MMR"].get(h, 0)
        vals = {"BFS": bfs, "Cosine": cos, "Multi": mul, "BiPPR": bip, "MMR": mmr}
        best = max(vals, key=vals.get)
        print(f"  {r['name']:<20s} {bfs:5.1f}% {cos:7.1f}% {mul:6.1f}% {bip:6.1f}% {mmr:5.1f}% {best:>15s}")

    with open(OUT / "exp13_cross_graph_replication.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nTotal: {time.time()-t_total:.0f}s")
    print(f"Saved: {OUT / 'exp13_cross_graph_replication.json'}")
    print("CROSS-GRAPH REPLICATION COMPLETE")


if __name__ == "__main__":
    main()
