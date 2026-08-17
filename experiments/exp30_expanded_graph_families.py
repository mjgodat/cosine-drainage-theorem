"""
Experiment 30: Expanded Graph Family Testing

Test progressive drainage and policy ordering on as many graph families
as possible. Goal: strengthen the conjecture's cross-family evidence
and find boundary conditions.

Families:
  1. Barabasi-Albert (pure PA — predicts ANTI-drainage per Hui-Wang)
  2. Erdos-Renyi (null model — weak drainage expected)
  3. Watts-Strogatz (small-world — high CC, tests tree approx)
  4. Stochastic Block Model (planted communities)
  5. Coauthor-CS (strongly assortative, real data)
  6. Coauthor-Physics (assortative, real data)
  7. Amazon Photo (different product category from Computers)
  8. PPI (protein interaction, biological)
  9. WikiCS (web hyperlinks)
  10. Actor (film co-occurrence)

For each graph:
  - Degree stats (mean, CV, max)
  - T3 test (degree drops over cosine expansion steps)
  - P4 test (cosine-seed degree vs random-seed degree)
  - 3-policy benchmark (BFS, Cosine, Multi-Anchor) at H=50,100
  - Classify: drainage / no drainage / anti-drainage
"""
import sys
import time
import json
import heapq
import numpy as np
import networkx as nx
from collections import deque, defaultdict
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

N_PAIRS = 150
WALK_LEN = 15
H_VALUES = [50, 100]


def embed_graph(features_or_N, D_target=256):
    """Ensure we have embeddings. If features exist, PCA down. If not, spectral."""
    if isinstance(features_or_N, np.ndarray):
        features = features_or_N.astype(np.float32)
        if features.shape[1] > D_target:
            pca = PCA(n_components=D_target)
            features = pca.fit_transform(features).astype(np.float32)
    else:
        # No features — use random embeddings (simulates embedding)
        N = features_or_N
        features = np.random.randn(N, D_target).astype(np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-8, None)


def build_adj(edge_index, N):
    adj = [[] for _ in range(N)]
    for s, d in zip(edge_index[0], edge_index[1]):
        adj[int(s)].append(int(d))
    for i in range(N):
        adj[i] = list(set(adj[i]))
    return adj


def test_graph(name, emb, adj, degrees, labels=None):
    N = len(emb)
    print(f"\n{'=' * 60}")
    print(f"{name} (N={N:,}, mean_deg={np.mean(degrees[degrees>0]):.1f}, "
          f"CV={np.std(degrees[degrees>0])/np.mean(degrees[degrees>0]):.3f}, "
          f"max_deg={np.max(degrees)})")
    print(f"{'=' * 60}")

    result = {
        "name": name, "N": N,
        "mean_deg": float(np.mean(degrees[degrees > 0])),
        "deg_cv": float(np.std(degrees[degrees > 0]) / max(np.mean(degrees[degrees > 0]), 1)),
        "max_deg": int(np.max(degrees)),
    }

    # T3: Degree over expansion steps
    step_degs = defaultdict(list)
    nodes_with_edges = [i for i in range(N) if degrees[i] > 1]
    if len(nodes_with_edges) < 20:
        print("  Too few nodes with edges, skipping")
        result["t3_rho"] = 0; result["drainage"] = "SKIP"
        return result

    for _ in range(min(100, len(nodes_with_edges))):
        source = nodes_with_edges[np.random.randint(len(nodes_with_edges))]
        target = nodes_with_edges[np.random.randint(len(nodes_with_edges))]
        if source == target or not adj[target]:
            continue
        tv = emb[target]; visited = {source}
        pq = [(-float(emb[v] @ tv), v) for v in adj[source] if v not in visited]
        heapq.heapify(pq)
        step = 0
        while pq and len(visited) < 50 and step < WALK_LEN:
            _, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            step_degs[step].append(degrees[u])
            for v in adj[u]:
                if v not in visited:
                    heapq.heappush(pq, (-float(emb[v] @ tv), v))
            step += 1

    if len(step_degs) > 3:
        steps = sorted(step_degs.keys())[:12]
        means = [np.mean(step_degs[s]) for s in steps]
        rho_t3, _ = spearmanr(steps, means)
        result["t3_rho"] = float(rho_t3)
        print(f"  T3 (degree over steps): rho = {rho_t3:+.4f}", end="")
        if rho_t3 < -0.3:
            print(" -> DRAINAGE")
            result["drainage"] = "YES"
        elif rho_t3 > 0.3:
            print(" -> ANTI-DRAINAGE")
            result["drainage"] = "ANTI"
        else:
            print(" -> WEAK/NONE")
            result["drainage"] = "WEAK"

        # Print first few steps
        for s in steps[:6]:
            print(f"    step {s}: mean_deg = {np.mean(step_degs[s]):.1f}")
    else:
        result["t3_rho"] = 0; result["drainage"] = "INSUFFICIENT"

    # P4: Cosine-seed degree vs random
    cos_degs, rand_degs = [], []
    for _ in range(min(100, len(nodes_with_edges))):
        q = nodes_with_edges[np.random.randint(len(nodes_with_edges))]
        sims = emb @ emb[q]
        sims[q] = -2
        cos_seed = int(np.argmax(sims))
        cos_degs.append(degrees[cos_seed])
        rand_seed = nodes_with_edges[np.random.randint(len(nodes_with_edges))]
        rand_degs.append(degrees[rand_seed])

    p4 = np.mean(cos_degs) / max(np.mean(rand_degs), 1)
    result["p4_ratio"] = float(p4)
    result["p4_cos_deg"] = float(np.mean(cos_degs))
    result["p4_rand_deg"] = float(np.mean(rand_degs))
    print(f"  P4 (cos/rand deg): {p4:.3f} (cos={np.mean(cos_degs):.1f}, rand={np.mean(rand_degs):.1f})")

    # 3-policy benchmark
    if labels is not None:
        ul = np.unique(labels)
        class_nodes = {l: [i for i in range(N) if labels[i] == l and degrees[i] > 1] for l in ul}
        pairs = []
        att = 0
        while len(pairs) < N_PAIRS and att < N_PAIRS * 20:
            att += 1
            l1, l2 = np.random.choice(ul, 2, replace=False)
            if not class_nodes[l1] or not class_nodes[l2]:
                continue
            s = class_nodes[l1][np.random.randint(len(class_nodes[l1]))]
            t = class_nodes[l2][np.random.randint(len(class_nodes[l2]))]
            if s != t:
                pairs.append((s, t))
    else:
        pairs = []
        for _ in range(N_PAIRS):
            s = nodes_with_edges[np.random.randint(len(nodes_with_edges))]
            t = nodes_with_edges[np.random.randint(len(nodes_with_edges))]
            if s != t:
                pairs.append((s, t))

    if not pairs:
        result["policy"] = {}
        return result

    for H in H_VALUES:
        bfs_r = cos_r = multi_r = 0
        for s, t in pairs:
            # BFS
            visited = {s}; q = deque([s])
            while q and len(visited) < H:
                u = q.popleft()
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        if v == t: break
                        if len(visited) >= H: break
                        q.append(v)
            if t in visited: bfs_r += 1

            # Cosine
            visited = {s}
            pq = [(-float(emb[v] @ emb[t]), v) for v in adj[s] if v not in visited]
            heapq.heapify(pq)
            while pq and len(visited) < H:
                _, u = heapq.heappop(pq)
                if u in visited: continue
                visited.add(u)
                if u == t: break
                for v in adj[u]:
                    if v not in visited:
                        heapq.heappush(pq, (-float(emb[v] @ emb[t]), v))
            if t in visited: cos_r += 1

            # Multi-anchor
            hs = H // 2; ht = H - hs
            vs = {s}; vt = {t}
            ps = [(-float(emb[v] @ emb[t]), v) for v in adj[s]]
            pt = [(-float(emb[v] @ emb[s]), v) for v in adj[t]]
            heapq.heapify(ps); heapq.heapify(pt)
            while ps and len(vs) < hs:
                _, u = heapq.heappop(ps)
                if u in vs: continue
                vs.add(u)
                if u in vt: break
                for v in adj[u]:
                    if v not in vs: heapq.heappush(ps, (-float(emb[v] @ emb[t]), v))
            while pt and len(vt) < ht:
                _, u = heapq.heappop(pt)
                if u in vt: continue
                vt.add(u)
                if u in vs: break
                for v in adj[u]:
                    if v not in vt: heapq.heappush(pt, (-float(emb[v] @ emb[s]), v))
            if vs & vt: multi_r += 1

        n = len(pairs)
        result[f"bfs_{H}"] = bfs_r / n * 100
        result[f"cos_{H}"] = cos_r / n * 100
        result[f"multi_{H}"] = multi_r / n * 100
        ordering = "Multi>Cos>BFS" if multi_r > cos_r > bfs_r else \
                   "Multi>Cos" if multi_r > cos_r else \
                   "Cos>BFS" if cos_r > bfs_r else "OTHER"
        print(f"  H={H}: BFS={bfs_r/n*100:.1f}% Cos={cos_r/n*100:.1f}% Multi={multi_r/n*100:.1f}% [{ordering}]")

    return result


# ══════════════════════════════════════════════════════════════════
# LOAD AND TEST ALL GRAPHS
# ══════════════════════════════════════════════════════════════════
all_results = []

# --- Synthetic generative models ---

# 1. Barabasi-Albert (pure preferential attachment)
print("\nGenerating Barabasi-Albert (N=5000, m=5)...")
G_ba = nx.barabasi_albert_graph(5000, 5, seed=42)
ba_edges = np.array(list(G_ba.edges())).T
ba_edges = np.hstack([ba_edges, ba_edges[::-1]])  # undirected
ba_emb = embed_graph(5000)
ba_adj = build_adj(ba_edges, 5000)
ba_deg = np.array([len(ba_adj[i]) for i in range(5000)])
all_results.append(test_graph("Barabasi-Albert", ba_emb, ba_adj, ba_deg))

# 2. Erdos-Renyi
print("\nGenerating Erdos-Renyi (N=5000, p=0.002)...")
G_er = nx.erdos_renyi_graph(5000, 0.002, seed=42)
er_edges = np.array(list(G_er.edges())).T
er_edges = np.hstack([er_edges, er_edges[::-1]])
er_emb = embed_graph(5000)
er_adj = build_adj(er_edges, 5000)
er_deg = np.array([len(er_adj[i]) for i in range(5000)])
all_results.append(test_graph("Erdos-Renyi", er_emb, er_adj, er_deg))

# 3. Watts-Strogatz (small-world)
print("\nGenerating Watts-Strogatz (N=5000, k=10, p=0.1)...")
G_ws = nx.watts_strogatz_graph(5000, 10, 0.1, seed=42)
ws_edges = np.array(list(G_ws.edges())).T
ws_edges = np.hstack([ws_edges, ws_edges[::-1]])
ws_emb = embed_graph(5000)
ws_adj = build_adj(ws_edges, 5000)
ws_deg = np.array([len(ws_adj[i]) for i in range(5000)])
all_results.append(test_graph("Watts-Strogatz", ws_emb, ws_adj, ws_deg))

# 4. Stochastic Block Model (3 communities)
print("\nGenerating SBM (3 blocks, 1500 each)...")
sizes = [1500, 1500, 2000]
p_matrix = [[0.01, 0.001, 0.001], [0.001, 0.01, 0.001], [0.001, 0.001, 0.01]]
G_sbm = nx.stochastic_block_model(sizes, p_matrix, seed=42)
sbm_edges = np.array(list(G_sbm.edges())).T
sbm_edges = np.hstack([sbm_edges, sbm_edges[::-1]])
sbm_emb = embed_graph(5000)
sbm_adj = build_adj(sbm_edges, 5000)
sbm_deg = np.array([len(sbm_adj[i]) for i in range(5000)])
sbm_labels = np.array([G_sbm.nodes[i]["block"] for i in range(5000)])
all_results.append(test_graph("Stochastic Block Model", sbm_emb, sbm_adj, sbm_deg, sbm_labels))

# --- Real-world graphs from torch_geometric ---
try:
    from torch_geometric.datasets import Planetoid, Amazon, CitationFull, Coauthor, Actor, WikiCS

    # 5. Coauthor-CS
    print("\nLoading Coauthor-CS...")
    d = Coauthor(root='/tmp/pyg_data', name='CS')
    data = d[0]
    emb = embed_graph(data.x.numpy())
    adj_l = build_adj(data.edge_index.numpy(), data.num_nodes)
    deg = np.array([len(adj_l[i]) for i in range(data.num_nodes)])
    all_results.append(test_graph("Coauthor-CS", emb, adj_l, deg, data.y.numpy()))

    # 6. Coauthor-Physics
    print("\nLoading Coauthor-Physics...")
    d = Coauthor(root='/tmp/pyg_data', name='Physics')
    data = d[0]
    emb = embed_graph(data.x.numpy())
    adj_l = build_adj(data.edge_index.numpy(), data.num_nodes)
    deg = np.array([len(adj_l[i]) for i in range(data.num_nodes)])
    all_results.append(test_graph("Coauthor-Physics", emb, adj_l, deg, data.y.numpy()))

    # 7. Amazon Photo
    print("\nLoading Amazon Photo...")
    d = Amazon(root='/tmp/pyg_data', name='Photo')
    data = d[0]
    emb = embed_graph(data.x.numpy())
    adj_l = build_adj(data.edge_index.numpy(), data.num_nodes)
    deg = np.array([len(adj_l[i]) for i in range(data.num_nodes)])
    all_results.append(test_graph("Amazon Photo", emb, adj_l, deg, data.y.numpy()))

    # 8. Actor
    print("\nLoading Actor...")
    d = Actor(root='/tmp/pyg_data')
    data = d[0]
    emb = embed_graph(data.x.numpy())
    adj_l = build_adj(data.edge_index.numpy(), data.num_nodes)
    deg = np.array([len(adj_l[i]) for i in range(data.num_nodes)])
    all_results.append(test_graph("Actor", emb, adj_l, deg, data.y.numpy()))

    # 9. WikiCS
    print("\nLoading WikiCS...")
    d = WikiCS(root='/tmp/pyg_data')
    data = d[0]
    emb = embed_graph(data.x.numpy())
    adj_l = build_adj(data.edge_index.numpy(), data.num_nodes)
    deg = np.array([len(adj_l[i]) for i in range(data.num_nodes)])
    all_results.append(test_graph("WikiCS", emb, adj_l, deg, data.y.numpy()))

except Exception as e:
    print(f"  Error loading PyG dataset: {e}")


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("CROSS-FAMILY DRAINAGE SUMMARY")
print(f"{'=' * 80}")

print(f"\n  {'Graph':>25s} {'N':>7s} {'DegCV':>7s} {'T3(drain)':>10s} {'P4':>6s} {'Cos@100':>8s} {'Multi@100':>10s} {'Drainage':>10s}")
print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*10} {'-'*6} {'-'*8} {'-'*10} {'-'*10}")

drainage_count = 0
anti_count = 0
for r in all_results:
    t3 = f"{r.get('t3_rho', 0):+.3f}"
    p4 = f"{r.get('p4_ratio', 0):.3f}"
    cos = f"{r.get('cos_100', 0):.1f}%"
    multi = f"{r.get('multi_100', 0):.1f}%"
    drain = r.get("drainage", "?")
    if drain == "YES": drainage_count += 1
    elif drain == "ANTI": anti_count += 1
    print(f"  {r['name']:>25s} {r['N']:7d} {r.get('deg_cv',0):7.3f} {t3:>10s} {p4:>6s} {cos:>8s} {multi:>10s} {drain:>10s}")

total = len([r for r in all_results if r.get("drainage") in ("YES", "ANTI", "WEAK")])
print(f"\n  Drainage: {drainage_count}/{total} graphs")
print(f"  Anti-drainage: {anti_count}/{total} graphs")
print(f"  Weak/none: {total - drainage_count - anti_count}/{total} graphs")

if anti_count > 0:
    print(f"\n  ANTI-DRAINAGE FOUND on: {[r['name'] for r in all_results if r.get('drainage') == 'ANTI']}")
    print(f"  This identifies a BOUNDARY CONDITION for the conjecture.")

# Policy ordering
multi_wins = sum(1 for r in all_results if r.get("multi_100", 0) > r.get("cos_100", 0))
print(f"\n  Multi > Cosine at H=100: {multi_wins}/{len(all_results)} graphs")

with open(OUT / "exp30_expanded_families.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved: {OUT / 'exp30_expanded_families.json'}")
print("EXPANDED FAMILY TESTING COMPLETE")
