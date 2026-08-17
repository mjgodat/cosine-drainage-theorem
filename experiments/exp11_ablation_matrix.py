"""
Experiment 11: Ablation Matrix — Separating geometry, topology, and interaction

Four ablations on PRSM Crystal, each run against the 6-policy benchmark:
  A. Vector shuffle: randomize embeddings across nodes → destroy alignment
  B. Degree-preserving rewire: randomize edges preserving degree sequence → isolate topology
  C. Whitened embeddings: remove top-3 PCA components → reduce Gamma
  D. kNN-derived graph: replace edges with cosine kNN → create geometry-aligned graph

If VGSG is geometry-topology INTERACTION:
  - Vector shuffle should eliminate cosine advantage (geometry destroyed)
  - Degree-preserving rewire should change trapping pattern (topology randomized)
  - Whitened embeddings should reduce trapping (lower Gamma)
  - kNN graph should eliminate BFS/cosine gap (geometry = topology)
"""
import sys
import time
import json
import heapq
import numpy as np
import torch
from collections import deque, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path("E:/PRSM")
OUT = Path(__file__).resolve().parent / "results"

N_PAIRS = 300
BUDGETS = [25, 100, 500]

print(f"Device: {DEVICE}")
print("Loading PRSM Crystal...", end=" ", flush=True)
t0 = time.time()

with open(ROOT / "data/g1_registry/unified_grains_with_embeddings.json", "r", encoding="utf-8") as f:
    grains_raw = json.load(f)
with open(ROOT / "data/g1_registry/relationship_registry.json", "r", encoding="utf-8") as f:
    rels = json.load(f)

concept_to_idx = {}
embeddings_list = []
concept_corpora = {}
seen = {}
for g in grains_raw:
    c = g.get("concept", "").lower()
    emb = g.get("embedding")
    bc = g.get("bind_count", 0)
    sc = g.get("source_corpora", [])
    if c and emb and (c not in seen or bc > seen[c]):
        seen[c] = bc
        if c not in concept_to_idx:
            idx = len(concept_to_idx)
            concept_to_idx[c] = idx
            embeddings_list.append(np.array(emb, dtype=np.float32))
            concept_corpora[idx] = set(sc) if isinstance(sc, list) else set()
        else:
            idx = concept_to_idx[c]
            embeddings_list[idx] = np.array(emb, dtype=np.float32)
            concept_corpora[idx] = set(sc) if isinstance(sc, list) else set()

orig_embeddings = np.array(embeddings_list)
N = len(orig_embeddings)
norms = np.linalg.norm(orig_embeddings, axis=1, keepdims=True)
orig_embeddings = orig_embeddings / np.clip(norms, 1e-8, None)

# Build original adjacency
gid_to_concept = {}
for g in grains_raw:
    gid = g.get("grain_id", "")
    c = g.get("concept", "").lower()
    if gid and c and c in concept_to_idx:
        gid_to_concept[gid] = c

orig_adj = [[] for _ in range(N)]
edge_list = []
for r in rels:
    a_c = gid_to_concept.get(r.get("grain_a_id", ""))
    b_c = gid_to_concept.get(r.get("grain_b_id", ""))
    if a_c and b_c and a_c != b_c:
        a_idx = concept_to_idx.get(a_c)
        b_idx = concept_to_idx.get(b_c)
        if a_idx is not None and b_idx is not None:
            orig_adj[a_idx].append(b_idx)
            orig_adj[b_idx].append(a_idx)
            edge_list.append((a_idx, b_idx))
for i in range(N):
    orig_adj[i] = list(set(orig_adj[i]))

corpus_nodes = defaultdict(list)
for idx, corpora in concept_corpora.items():
    for c in corpora:
        corpus_nodes[c].append(idx)
valid_corpora = [c for c, nodes in corpus_nodes.items() if len(nodes) >= 20]

print(f"{N:,} grains, {len(edge_list):,} edges, {time.time()-t0:.1f}s")

# Sample pairs (fixed across ablations)
pairs = []
attempts = 0
while len(pairs) < N_PAIRS and attempts < N_PAIRS * 50:
    attempts += 1
    c1 = valid_corpora[np.random.randint(len(valid_corpora))]
    c2 = valid_corpora[np.random.randint(len(valid_corpora))]
    if c1 == c2: continue
    s = corpus_nodes[c1][np.random.randint(len(corpus_nodes[c1]))]
    t = corpus_nodes[c2][np.random.randint(len(corpus_nodes[c2]))]
    if s == t or not orig_adj[s] or not orig_adj[t]: continue
    pairs.append((s, t))
print(f"Sampled {len(pairs)} pairs")


# Policies (simplified to 3 key ones for speed)
def run_bfs(adj_list, emb, source, target, budget):
    visited = {source}
    queue = deque([source])
    while queue and len(visited) < budget:
        u = queue.popleft()
        for v in adj_list[u]:
            if v not in visited:
                visited.add(v)
                if v == target: return True
                if len(visited) >= budget: break
                queue.append(v)
    return target in visited


def run_cosine(adj_list, emb, source, target, budget):
    target_vec = emb[target]
    visited = {source}
    pq = []
    for v in adj_list[source]:
        if v not in visited:
            heapq.heappush(pq, (-float(emb[v] @ target_vec), v))
    while pq and len(visited) < budget:
        _, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u)
        if u == target: return True
        for v in adj_list[u]:
            if v not in visited:
                heapq.heappush(pq, (-float(emb[v] @ target_vec), v))
    return target in visited


def run_bidir(adj_list, emb, source, target, budget):
    h_s = budget // 2
    h_t = budget - h_s
    visited_s = {source}
    visited_t = {target}
    pq_s = [(-float(emb[v] @ emb[target]), v) for v in adj_list[source]]
    heapq.heapify(pq_s)
    pq_t = [(-float(emb[v] @ emb[source]), v) for v in adj_list[target]]
    heapq.heapify(pq_t)
    while pq_s and len(visited_s) < h_s:
        _, u = heapq.heappop(pq_s)
        if u in visited_s: continue
        visited_s.add(u)
        if u in visited_t: return True
        for v in adj_list[u]:
            if v not in visited_s:
                heapq.heappush(pq_s, (-float(emb[v] @ emb[target]), v))
    while pq_t and len(visited_t) < h_t:
        _, u = heapq.heappop(pq_t)
        if u in visited_t: continue
        visited_t.add(u)
        if u in visited_s: return True
        for v in adj_list[u]:
            if v not in visited_t:
                heapq.heappush(pq_t, (-float(emb[v] @ emb[source]), v))
    return bool(visited_s & visited_t)


def benchmark(name, adj_list, emb):
    print(f"\n  --- {name} ---")
    results = {}
    for H in BUDGETS:
        bfs_ok = cos_ok = bid_ok = 0
        for s, t in pairs:
            if run_bfs(adj_list, emb, s, t, H): bfs_ok += 1
            if run_cosine(adj_list, emb, s, t, H): cos_ok += 1
            if run_bidir(adj_list, emb, s, t, H): bid_ok += 1
        n = len(pairs)
        results[H] = {"bfs": bfs_ok/n*100, "cosine": cos_ok/n*100, "bidir": bid_ok/n*100}
        print(f"    H={H:4d}: BFS={bfs_ok/n*100:5.1f}%  Cosine={cos_ok/n*100:5.1f}%  BiDir={bid_ok/n*100:5.1f}%")
    return results


# ── RUN ABLATIONS ──
all_results = {}

# Baseline
print(f"\n{'=' * 70}")
print("ABLATION MATRIX")
print(f"{'=' * 70}")
all_results["baseline"] = benchmark("BASELINE (original)", orig_adj, orig_embeddings)

# A: Vector shuffle
shuffled_emb = orig_embeddings.copy()
perm = np.random.permutation(N)
shuffled_emb = shuffled_emb[perm]
all_results["vector_shuffle"] = benchmark("A: VECTOR SHUFFLE (destroy alignment)", orig_adj, shuffled_emb)

# B: Degree-preserving rewire
print("\n  Building degree-preserving rewired graph...", end=" ", flush=True)
rewired_adj = [[] for _ in range(N)]
degrees = [len(orig_adj[i]) for i in range(N)]
# Simple rewire: for each edge, swap endpoints with random edge
edges = list(set((min(a,b), max(a,b)) for a, b in edge_list))
rewired_edges = list(edges)
for _ in range(len(edges) * 2):
    i = np.random.randint(len(rewired_edges))
    j = np.random.randint(len(rewired_edges))
    if i == j: continue
    a, b = rewired_edges[i]
    c, d = rewired_edges[j]
    if np.random.rand() < 0.5:
        if a != d and c != b and (min(a,d), max(a,d)) not in set(rewired_edges) and (min(c,b), max(c,b)) not in set(rewired_edges):
            rewired_edges[i] = (min(a,d), max(a,d))
            rewired_edges[j] = (min(c,b), max(c,b))
for a, b in rewired_edges:
    rewired_adj[a].append(b)
    rewired_adj[b].append(a)
for i in range(N):
    rewired_adj[i] = list(set(rewired_adj[i]))
print("done")
all_results["degree_rewire"] = benchmark("B: DEGREE-PRESERVING REWIRE (destroy topology)", rewired_adj, orig_embeddings)

# C: Whitened embeddings
print("\n  Whitening embeddings (remove top-3 PCA)...", end=" ", flush=True)
from sklearn.decomposition import PCA
pca = PCA(n_components=min(20, N))
transformed = pca.fit_transform(orig_embeddings)
# Zero out top 3 components
transformed[:, :3] = 0
whitened = pca.inverse_transform(transformed).astype(np.float32)
w_norms = np.linalg.norm(whitened, axis=1, keepdims=True)
whitened = whitened / np.clip(w_norms, 1e-8, None)
from math import pi
t_w = torch.tensor(whitened, device=DEVICE)
t_wn = t_w / t_w.norm(dim=1, keepdim=True).clamp(min=1e-8)
sample = np.random.choice(N, min(2000, N), replace=False)
sims = (t_wn[sample] @ t_wn[sample].T).cpu().numpy()
np.fill_diagonal(sims, 1.0)
triu = np.triu_indices(len(sample), k=1)
angles = np.arccos(np.clip(sims[triu], -1, 1))
gamma_whitened = 1 - np.mean(angles) / (pi / 2)
print(f"Gamma: {gamma_whitened:.4f} (was 0.249)")
all_results["whitened"] = benchmark("C: WHITENED EMBEDDINGS (reduce Gamma)", orig_adj, whitened)
all_results["whitened"]["gamma"] = float(gamma_whitened)

# D: kNN-derived graph
print("\n  Building kNN graph (k=20, GPU)...", end=" ", flush=True)
t0 = time.time()
t_emb = torch.tensor(orig_embeddings, device=DEVICE)
knn_adj = [[] for _ in range(N)]
BATCH = 512
for start in range(0, N, BATCH):
    end = min(start + BATCH, N)
    dists = torch.cdist(t_emb[start:end], t_emb)
    for i in range(end - start):
        dists[i, start + i] = float('inf')
    _, topk = dists.topk(20, dim=1, largest=False)
    for i in range(end - start):
        node = start + i
        for j in topk[i].cpu().numpy():
            knn_adj[node].append(int(j))
            knn_adj[int(j)].append(node)
for i in range(N):
    knn_adj[i] = list(set(knn_adj[i]))
print(f"{time.time()-t0:.1f}s")
all_results["knn_graph"] = benchmark("D: kNN GRAPH (geometry = topology)", knn_adj, orig_embeddings)

# Summary
print(f"\n{'=' * 70}")
print("ABLATION SUMMARY")
print(f"{'=' * 70}")
print(f"\n  {'Ablation':>30s}  {'BFS H=100':>10s} {'Cos H=100':>10s} {'BiD H=100':>10s} {'Cos-BFS':>8s}")
print(f"  {'-'*30}  {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
for name, r in all_results.items():
    h100 = r.get(100, r.get("100", {}))
    if isinstance(h100, dict):
        bfs = h100.get("bfs", 0)
        cos = h100.get("cosine", 0)
        bid = h100.get("bidir", 0)
        gap = cos - bfs
        print(f"  {name:>30s}  {bfs:9.1f}% {cos:9.1f}% {bid:9.1f}% {gap:+7.1f}%")

print(f"\n  PREDICTIONS:")
print(f"    Vector shuffle: cosine advantage should DISAPPEAR (geometry destroyed)")
print(f"    Degree rewire:  trapping pattern should CHANGE (topology randomized)")
print(f"    Whitened:       trapping should DECREASE (lower Gamma)")
print(f"    kNN graph:      BFS/cosine gap should SHRINK (geometry = topology)")

with open(OUT / "exp11_ablation_matrix.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {OUT / 'exp11_ablation_matrix.json'}")
print("ABLATION MATRIX COMPLETE")
