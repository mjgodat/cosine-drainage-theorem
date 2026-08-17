"""
Experiment 9: VGSG Policy Benchmarking Suite
Evaluates 6 expansion policies under strict node visit budgets against
target reachability on PRSM Crystal.

Policies:
  1. BFS (Topological) — isotropic, exhaustive within budget
  2. Cosine Single-Source — greedy best-first by cosine to target
  3. Multi-Anchor (50:50) — dual-frontier cosine from both ends
  4. Forward-Push PPR — local random walk with restart
  5. Bidirectional PPR — forward push from source + backward from target
  6. MMR-Regularized Cosine — cosine with degree penalty

Based on Gemini benchmark spec. Wired to PRSM Crystal data.
GPU used for embedding normalization; traversal on CPU (graph walks).
"""
import sys
import heapq
import time
import json
import numpy as np
from collections import deque, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

ROOT = Path("E:/PRSM")
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

BUDGETS = [25, 50, 100, 200, 500]
N_PAIRS = 500


# ══════════════════════════════════════════════════════════════════
# LOAD PRSM CRYSTAL
# ══════════════════════════════════════════════════════════════════
print("Loading PRSM Crystal...", end=" ", flush=True)
t0 = time.time()

with open(ROOT / "data/g1_registry/unified_grains_with_embeddings.json", "r", encoding="utf-8") as f:
    grains_raw = json.load(f)
with open(ROOT / "data/g1_registry/relationship_registry.json", "r", encoding="utf-8") as f:
    rels = json.load(f)

concept_to_idx = {}
idx_to_concept = {}
concept_corpora = {}
embeddings_list = []
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
            idx_to_concept[idx] = c
            embeddings_list.append(np.array(emb, dtype=np.float32))
            concept_corpora[idx] = set(sc) if isinstance(sc, list) else set()
        else:
            idx = concept_to_idx[c]
            embeddings_list[idx] = np.array(emb, dtype=np.float32)
            concept_corpora[idx] = set(sc) if isinstance(sc, list) else set()

embeddings = np.array(embeddings_list)
N = len(embeddings)

# L2-normalize
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / np.clip(norms, 1e-8, None)

# Build adjacency
gid_to_concept = {}
for g in grains_raw:
    gid = g.get("grain_id", "")
    c = g.get("concept", "").lower()
    if gid and c and c in concept_to_idx:
        gid_to_concept[gid] = c

adj_list = [[] for _ in range(N)]
for r in rels:
    a_c = gid_to_concept.get(r.get("grain_a_id", ""))
    b_c = gid_to_concept.get(r.get("grain_b_id", ""))
    if a_c and b_c and a_c != b_c:
        a_idx = concept_to_idx.get(a_c)
        b_idx = concept_to_idx.get(b_c)
        if a_idx is not None and b_idx is not None:
            adj_list[a_idx].append(b_idx)
            adj_list[b_idx].append(a_idx)

# Deduplicate
for i in range(N):
    adj_list[i] = list(set(adj_list[i]))

# Corpus grouping for pair sampling
corpus_nodes = defaultdict(list)
for idx, corpora in concept_corpora.items():
    for c in corpora:
        corpus_nodes[c].append(idx)
valid_corpora = [c for c, nodes in corpus_nodes.items() if len(nodes) >= 20]

print(f"{N:,} grains, {sum(len(a) for a in adj_list)//2:,} edges, {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════
# 6 TRAVERSAL POLICIES
# ══════════════════════════════════════════════════════════════════

def policy_bfs(source, target, budget_H):
    visited = {source}
    queue = deque([source])
    while queue and len(visited) < budget_H:
        u = queue.popleft()
        for v in adj_list[u]:
            if v not in visited:
                visited.add(v)
                if v == target:
                    return True, len(visited)
                if len(visited) >= budget_H:
                    break
                queue.append(v)
    return target in visited, len(visited)


def policy_cosine_single(source, target, budget_H):
    target_vec = embeddings[target]
    visited = {source}
    pq = []
    for v in adj_list[source]:
        if v not in visited:
            sim = float(np.dot(embeddings[v], target_vec))
            heapq.heappush(pq, (-sim, v))
    while pq and len(visited) < budget_H:
        _, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            return True, len(visited)
        for v in adj_list[u]:
            if v not in visited:
                sim = float(np.dot(embeddings[v], target_vec))
                heapq.heappush(pq, (-sim, v))
    return target in visited, len(visited)


def policy_multi_anchor(source, target, budget_H):
    budget_src = budget_H // 2
    budget_tgt = budget_H - budget_src
    src_vec = embeddings[source]
    tgt_vec = embeddings[target]
    visited_src = {source}
    visited_tgt = {target}
    pq_src = []
    for v in adj_list[source]:
        sim = float(np.dot(embeddings[v], tgt_vec))
        heapq.heappush(pq_src, (-sim, v))
    pq_tgt = []
    for v in adj_list[target]:
        sim = float(np.dot(embeddings[v], src_vec))
        heapq.heappush(pq_tgt, (-sim, v))
    # Source expansion
    while pq_src and len(visited_src) < budget_src:
        _, u = heapq.heappop(pq_src)
        if u in visited_src:
            continue
        visited_src.add(u)
        if u in visited_tgt:
            return True, len(visited_src) + len(visited_tgt)
        for v in adj_list[u]:
            if v not in visited_src:
                sim = float(np.dot(embeddings[v], tgt_vec))
                heapq.heappush(pq_src, (-sim, v))
    # Target expansion
    while pq_tgt and len(visited_tgt) < budget_tgt:
        _, u = heapq.heappop(pq_tgt)
        if u in visited_tgt:
            continue
        visited_tgt.add(u)
        if u in visited_src:
            return True, len(visited_src) + len(visited_tgt)
        for v in adj_list[u]:
            if v not in visited_tgt:
                sim = float(np.dot(embeddings[v], src_vec))
                heapq.heappush(pq_tgt, (-sim, v))
    return bool(visited_src & visited_tgt), len(visited_src) + len(visited_tgt)


def policy_forward_push_ppr(source, target, budget_H, alpha=0.15):
    r = {source: 1.0}
    p = {}
    visited = set()
    queue = deque([source])
    in_queue = {source}
    pushes = 0
    while queue and pushes < budget_H:
        u = queue.popleft()
        in_queue.discard(u)
        visited.add(u)
        res_u = r.get(u, 0.0)
        if res_u <= 0.0:
            continue
        p[u] = p.get(u, 0.0) + alpha * res_u
        push_mass = (1.0 - alpha) * res_u
        r[u] = 0.0
        pushes += 1
        neighbors = adj_list[u]
        if not neighbors:
            continue
        mass_per = push_mass / len(neighbors)
        for v in neighbors:
            r[v] = r.get(v, 0.0) + mass_per
            if v not in in_queue and len(visited) < budget_H:
                queue.append(v)
                in_queue.add(v)
    return target in visited, len(visited)


def policy_bidirectional_ppr(source, target, budget_H, alpha=0.15):
    budget_src = budget_H // 2
    budget_tgt = budget_H - budget_src
    # Forward
    r_s = {source: 1.0}
    visited_s = set()
    q_s = deque([source])
    pushes = 0
    while q_s and pushes < budget_src:
        u = q_s.popleft()
        visited_s.add(u)
        res = r_s.get(u, 0.0)
        r_s[u] = 0.0
        pushes += 1
        nbs = adj_list[u]
        if nbs and res > 0:
            m = (1.0 - alpha) * res / len(nbs)
            for v in nbs:
                r_s[v] = r_s.get(v, 0.0) + m
                if v not in visited_s and len(visited_s) < budget_src:
                    q_s.append(v)
    # Backward
    r_t = {target: 1.0}
    visited_t = set()
    q_t = deque([target])
    pushes = 0
    while q_t and pushes < budget_tgt:
        u = q_t.popleft()
        visited_t.add(u)
        res = r_t.get(u, 0.0)
        r_t[u] = 0.0
        pushes += 1
        nbs = adj_list[u]
        if nbs and res > 0:
            m = (1.0 - alpha) * res / len(nbs)
            for v in nbs:
                r_t[v] = r_t.get(v, 0.0) + m
                if v not in visited_t and len(visited_t) < budget_tgt:
                    q_t.append(v)
    hit = bool(visited_s & visited_t) or target in visited_s
    return hit, len(visited_s) + len(visited_t)


def policy_mmr_cosine(source, target, budget_H, lambda_param=0.7):
    target_vec = embeddings[target]
    visited = {source}
    pq = []
    for v in adj_list[source]:
        deg_penalty = np.log1p(len(adj_list[v]))
        cos_sim = float(np.dot(embeddings[v], target_vec))
        score = (lambda_param * cos_sim) - ((1.0 - lambda_param) * 0.1 * deg_penalty)
        heapq.heappush(pq, (-score, v))
    while pq and len(visited) < budget_H:
        _, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            return True, len(visited)
        for v in adj_list[u]:
            if v not in visited:
                deg_penalty = np.log1p(len(adj_list[v]))
                cos_sim = float(np.dot(embeddings[v], target_vec))
                score = (lambda_param * cos_sim) - ((1.0 - lambda_param) * 0.1 * deg_penalty)
                heapq.heappush(pq, (-score, v))
    return target in visited, len(visited)


# ══════════════════════════════════════════════════════════════════
# SAMPLE CROSS-CORPUS PAIRS
# ══════════════════════════════════════════════════════════════════
print(f"\nSampling {N_PAIRS} cross-corpus pairs...", end=" ", flush=True)
t0 = time.time()

test_pairs = []
attempts = 0
while len(test_pairs) < N_PAIRS and attempts < N_PAIRS * 50:
    attempts += 1
    c1 = valid_corpora[np.random.randint(len(valid_corpora))]
    c2 = valid_corpora[np.random.randint(len(valid_corpora))]
    if c1 == c2:
        continue
    s = corpus_nodes[c1][np.random.randint(len(corpus_nodes[c1]))]
    t = corpus_nodes[c2][np.random.randint(len(corpus_nodes[c2]))]
    if s == t or not adj_list[s] or not adj_list[t]:
        continue
    test_pairs.append((s, t))

print(f"{time.time()-t0:.1f}s, {len(test_pairs)} pairs")


# ══════════════════════════════════════════════════════════════════
# RUN BENCHMARK
# ══════════════════════════════════════════════════════════════════
policies = {
    "BFS (Topological)": lambda s, t, H: policy_bfs(s, t, H),
    "Cosine Single-Source": lambda s, t, H: policy_cosine_single(s, t, H),
    "Multi-Anchor (50:50)": lambda s, t, H: policy_multi_anchor(s, t, H),
    "Forward-Push PPR": lambda s, t, H: policy_forward_push_ppr(s, t, H),
    "Bidirectional PPR": lambda s, t, H: policy_bidirectional_ppr(s, t, H),
    "MMR-Regularized Cosine": lambda s, t, H: policy_mmr_cosine(s, t, H),
}

results = {p: {H: 0 for H in BUDGETS} for p in policies}

print(f"\n{'=' * 75}")
print(f"VGSG Policy Benchmark on PRSM Crystal ({N:,} grains, {len(test_pairs)} pairs)")
print(f"{'=' * 75}")

for H in BUDGETS:
    t0 = time.time()
    for p_name, func in policies.items():
        successes = 0
        for src, tgt in test_pairs:
            reached, _ = func(src, tgt, H)
            if reached:
                successes += 1
        results[p_name][H] = successes / len(test_pairs) * 100
    elapsed = time.time() - t0
    print(f"  Budget H={H:<4d} complete ({elapsed:.1f}s)")

# ══════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 75}")
print(f"RESULTS: Target Reachability (%) by Policy and Budget")
print(f"{'=' * 75}")

header = f"{'Policy':<27}" + "".join([f"{'H='+str(H):>9}" for H in BUDGETS])
print(f"\n{header}")
print("-" * len(header))
for p_name in policies:
    row = f"{p_name:<27}" + "".join([f"{results[p_name][H]:8.1f}%" for H in BUDGETS])
    print(row)
print("-" * len(header))

# Relative to BFS baseline
print(f"\n  Relative to BFS (percentage point difference):")
print(f"  {'Policy':<27}", end="")
for H in BUDGETS:
    print(f"{'H='+str(H):>9}", end="")
print()
for p_name in policies:
    if p_name == "BFS (Topological)":
        continue
    row = f"  {p_name:<27}"
    for H in BUDGETS:
        diff = results[p_name][H] - results["BFS (Topological)"][H]
        row += f"{diff:+8.1f}%"
    print(row)

# Key findings
print(f"\n  KEY FINDINGS:")
for H in [25, 100, 500]:
    bfs = results["BFS (Topological)"][H]
    cos = results["Cosine Single-Source"][H]
    multi = results["Multi-Anchor (50:50)"][H]
    ppr = results["Forward-Push PPR"][H]
    bi_ppr = results["Bidirectional PPR"][H]
    mmr = results["MMR-Regularized Cosine"][H]
    best_name = max(policies.keys(), key=lambda p: results[p][H])
    best_val = results[best_name][H]
    print(f"  H={H:3d}: Best={best_name} ({best_val:.1f}%), "
          f"BFS={bfs:.1f}%, Cosine={cos:.1f}%, Multi={multi:.1f}%")

# Save
output = {
    "graph": "PRSM Crystal",
    "n_grains": N,
    "n_pairs": len(test_pairs),
    "budgets": BUDGETS,
    "results": {p: {str(H): v for H, v in vals.items()} for p, vals in results.items()},
}
outfile = OUT / "exp9_policy_benchmark.json"
with open(outfile, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {outfile}")
print("POLICY BENCHMARK COMPLETE")
