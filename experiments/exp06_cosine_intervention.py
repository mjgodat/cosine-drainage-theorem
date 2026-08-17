"""
Experiment 6: Cosine-Seeded Intervention — the correct VGSG test.

BFS doesn't exhibit VGSG because it's exhaustive. VGSG is about
cosine-similarity-BIASED expansion where each hop preferentially
selects the most similar neighbors, staying in the angular basin.

Compares:
  1. Cosine-seeded single-source: greedy best-first by cosine to source
  2. Cosine-seeded multi-anchor: greedy from BOTH source and target
  3. BFS single-source (control): exhaustive, same node budget

Optimized: precomputed cosine matrix on GPU, heap-based priority queue.
"""
import sys
import time
import json
import heapq
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path("E:/PRSM")
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

N_PAIRS = 500
BUDGETS = [25, 50, 100, 200, 500, 1000]
K_SEED = 10

print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════
# LOAD CRYSTAL
# ══════════════════════════════════════════════════════════════════
print("Loading PRSM crystal...", end=" ", flush=True)
t0 = time.time()

with open(ROOT / "data" / "g1_registry" / "unified_grains_with_embeddings.json", "r", encoding="utf-8") as f:
    grains_raw = json.load(f)
with open(ROOT / "data" / "g1_registry" / "relationship_registry.json", "r", encoding="utf-8") as f:
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

# Normalized embeddings on GPU
emb_gpu = torch.tensor(embeddings, device=DEVICE)
emb_normed = emb_gpu / emb_gpu.norm(dim=1, keepdim=True).clamp(min=1e-8)

# Precompute full cosine similarity to CPU numpy for fast lookup
# This is ~39K x 39K x 4 bytes = ~5.7 GB — too large.
# Instead, precompute per-node cosine on demand via GPU batch.

# Build adjacency
gid_to_concept = {}
for g in grains_raw:
    gid = g.get("grain_id", "")
    c = g.get("concept", "").lower()
    if gid and c and c in concept_to_idx:
        gid_to_concept[gid] = c

adj = defaultdict(list)
edge_count = 0
for r in rels:
    a_c = gid_to_concept.get(r.get("grain_a_id", ""))
    b_c = gid_to_concept.get(r.get("grain_b_id", ""))
    if a_c and b_c and a_c != b_c:
        a_idx = concept_to_idx.get(a_c)
        b_idx = concept_to_idx.get(b_c)
        if a_idx is not None and b_idx is not None:
            adj[a_idx].append(b_idx)
            adj[b_idx].append(a_idx)
            edge_count += 1

# Deduplicate
for k in adj:
    adj[k] = list(set(adj[k]))

# Corpus grouping
corpus_nodes = defaultdict(list)
for idx, corpora in concept_corpora.items():
    for c in corpora:
        corpus_nodes[c].append(idx)
valid_corpora = [c for c, nodes in corpus_nodes.items() if len(nodes) >= 20]

print(f"{N:,} grains, {edge_count:,} edges, {len(valid_corpora)} corpora, {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════
# FAST COSINE LOOKUP
# ══════════════════════════════════════════════════════════════════

def get_cosine_to_anchor(anchor_idx, node_indices):
    """Batch cosine similarity of node_indices to anchor. Returns numpy array."""
    if not node_indices:
        return np.array([])
    anchor_vec = emb_normed[anchor_idx:anchor_idx+1]  # (1, D)
    node_vecs = emb_normed[node_indices]  # (len, D)
    sims = (node_vecs @ anchor_vec.T).squeeze(1)  # (len,)
    return sims.cpu().numpy()


def get_topk_cosine(anchor_idx, k):
    """Get top-k cosine neighbors of anchor (excluding self)."""
    anchor_vec = emb_normed[anchor_idx:anchor_idx+1]
    sims = (anchor_vec @ emb_normed.T).squeeze(0)
    sims[anchor_idx] = -2
    vals, idxs = sims.topk(k)
    return idxs.cpu().numpy(), vals.cpu().numpy()


# ══════════════════════════════════════════════════════════════════
# TRAVERSAL POLICIES
# ══════════════════════════════════════════════════════════════════

def cosine_seeded_single(source_idx, target_idx, budget):
    """Greedy best-first cosine-seeded expansion from source only."""
    # Seed: top-k cosine neighbors
    seed_idx, seed_sims = get_topk_cosine(source_idx, K_SEED)

    visited = {source_idx}
    # Max-heap (negate for min-heap): (-similarity, node_idx)
    heap = []
    for idx, sim in zip(seed_idx, seed_sims):
        idx = int(idx)
        visited.add(idx)
        if idx == target_idx:
            return True, len(visited)
        heapq.heappush(heap, (-sim, idx))

    while heap and len(visited) < budget:
        _, current = heapq.heappop(heap)

        # Get graph neighbors not yet visited
        neighbors = [nb for nb in adj.get(current, []) if nb not in visited]
        if not neighbors:
            continue

        # Batch cosine computation
        sims = get_cosine_to_anchor(source_idx, neighbors)

        for nb, sim in zip(neighbors, sims):
            if nb not in visited:
                visited.add(nb)
                if nb == target_idx:
                    return True, len(visited)
                heapq.heappush(heap, (-sim, nb))
                if len(visited) >= budget:
                    break

    return False, len(visited)


def cosine_seeded_multi(source_idx, target_idx, budget):
    """Greedy best-first from BOTH source and target, check overlap."""
    half = budget // 2

    # Source side seeds
    s_idx, s_sims = get_topk_cosine(source_idx, max(K_SEED // 2, 3))
    visited_s = {source_idx}
    heap_s = []
    for idx, sim in zip(s_idx, s_sims):
        idx = int(idx)
        visited_s.add(idx)
        heapq.heappush(heap_s, (-sim, idx))

    # Target side seeds
    t_idx, t_sims = get_topk_cosine(target_idx, max(K_SEED // 2, 3))
    visited_t = {target_idx}
    heap_t = []
    for idx, sim in zip(t_idx, t_sims):
        idx = int(idx)
        visited_t.add(idx)
        heapq.heappush(heap_t, (-sim, idx))

    # Check immediate overlap
    if visited_s & visited_t:
        return True, len(visited_s) + len(visited_t)

    # Alternate expansion
    while (len(visited_s) + len(visited_t)) < budget:
        expanded = False

        # Source step
        if heap_s and len(visited_s) < half:
            _, current = heapq.heappop(heap_s)
            neighbors = [nb for nb in adj.get(current, []) if nb not in visited_s]
            if neighbors:
                sims = get_cosine_to_anchor(source_idx, neighbors)
                for nb, sim in zip(neighbors, sims):
                    if nb not in visited_s:
                        visited_s.add(nb)
                        heapq.heappush(heap_s, (-sim, nb))
                        if len(visited_s) >= half:
                            break
                expanded = True

        # Target step
        if heap_t and len(visited_t) < (budget - half):
            _, current = heapq.heappop(heap_t)
            neighbors = [nb for nb in adj.get(current, []) if nb not in visited_t]
            if neighbors:
                sims = get_cosine_to_anchor(target_idx, neighbors)
                for nb, sim in zip(neighbors, sims):
                    if nb not in visited_t:
                        visited_t.add(nb)
                        heapq.heappush(heap_t, (-sim, nb))
                        if len(visited_t) >= (budget - half):
                            break
                expanded = True

        # Check overlap
        if visited_s & visited_t:
            return True, len(visited_s) + len(visited_t)

        if not expanded:
            break

    return bool(visited_s & visited_t), len(visited_s) + len(visited_t)


def bfs_single(source_idx, target_idx, budget):
    """BFS control — exhaustive, same node budget."""
    visited = {source_idx}
    frontier = [source_idx]
    while frontier and len(visited) < budget:
        next_frontier = []
        for current in frontier:
            for nb in adj.get(current, []):
                if nb not in visited:
                    visited.add(nb)
                    if nb == target_idx:
                        return True, len(visited)
                    next_frontier.append(nb)
                    if len(visited) >= budget:
                        break
            if len(visited) >= budget:
                break
        frontier = next_frontier
    return target_idx in visited, len(visited)


# ══════════════════════════════════════════════════════════════════
# SAMPLE PAIRS (fast — no BFS distance, just cross-corpus + low cosine)
# ══════════════════════════════════════════════════════════════════
print("\nSampling cross-corpus pairs (fast)...", end=" ", flush=True)
t0 = time.time()

pairs = []
attempts = 0
while len(pairs) < N_PAIRS and attempts < N_PAIRS * 50:
    attempts += 1
    c1 = valid_corpora[np.random.randint(len(valid_corpora))]
    c2 = valid_corpora[np.random.randint(len(valid_corpora))]
    if c1 == c2:
        continue
    s = corpus_nodes[c1][np.random.randint(len(corpus_nodes[c1]))]
    t_node = corpus_nodes[c2][np.random.randint(len(corpus_nodes[c2]))]
    if s == t_node or not adj.get(s) or not adj.get(t_node):
        continue

    s_corpora = concept_corpora.get(s, set())
    t_corpora = concept_corpora.get(t_node, set())
    overlap = len(s_corpora & t_corpora)

    cos_st = (emb_normed[s] @ emb_normed[t_node]).item()

    pairs.append({
        "source": s, "target": t_node,
        "source_concept": idx_to_concept[s],
        "target_concept": idx_to_concept[t_node],
        "source_corpus": c1, "target_corpus": c2,
        "corpus_overlap": overlap,
        "cosine_similarity": cos_st,
    })

print(f"{time.time()-t0:.1f}s, {len(pairs)} pairs")
print(f"  Mean cosine: {np.mean([p['cosine_similarity'] for p in pairs]):.4f}")
print(f"  Zero-overlap pairs: {sum(1 for p in pairs if p['corpus_overlap'] == 0)}/{len(pairs)}")


# ══════════════════════════════════════════════════════════════════
# RUN EXPERIMENT
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("COSINE-SEEDED INTERVENTION: Single vs Multi-Anchor vs BFS")
print(f"{'=' * 80}")

results_by_budget = {}

for budget in BUDGETS:
    cos_s_reached = 0
    cos_m_reached = 0
    bfs_reached = 0

    t0 = time.time()
    for p in pairs:
        s, t_node = p["source"], p["target"]

        r, _ = cosine_seeded_single(s, t_node, budget)
        if r: cos_s_reached += 1

        r, _ = cosine_seeded_multi(s, t_node, budget)
        if r: cos_m_reached += 1

        r, _ = bfs_single(s, t_node, budget)
        if r: bfs_reached += 1

    total = len(pairs)
    p_cs = cos_s_reached / total
    p_cm = cos_m_reached / total
    p_bf = bfs_reached / total
    m_vs_s = (p_cm - p_cs) / max(p_cs, 0.001) * 100
    b_vs_c = (p_bf - p_cs) / max(p_cs, 0.001) * 100
    elapsed = time.time() - t0

    print(f"\n  Budget={budget:5d} ({elapsed:.1f}s):")
    print(f"    Cosine single: {p_cs:.3f} ({cos_s_reached}/{total})")
    print(f"    Cosine multi:  {p_cm:.3f} ({cos_m_reached}/{total})  vs single: {m_vs_s:+.1f}%")
    print(f"    BFS control:   {p_bf:.3f} ({bfs_reached}/{total})  vs cosine: {b_vs_c:+.1f}%")

    results_by_budget[budget] = {
        "cosine_single": p_cs, "cosine_multi": p_cm, "bfs": p_bf,
        "cos_s_count": cos_s_reached, "cos_m_count": cos_m_reached,
        "bfs_count": bfs_reached, "total": total,
        "multi_vs_single_pct": m_vs_s, "bfs_vs_cosine_pct": b_vs_c,
    }


# ══════════════════════════════════════════════════════════════════
# STRATIFIED: by corpus overlap (budget=200)
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("STRATIFIED BY CORPUS OVERLAP (Budget=200)")
print(f"{'=' * 80}")

budget = 200
overlap_bins = defaultdict(lambda: {"cs": 0, "cm": 0, "bf": 0, "t": 0})
for p in pairs:
    ov = min(p["corpus_overlap"], 5)
    s, t_node = p["source"], p["target"]
    overlap_bins[ov]["t"] += 1
    r, _ = cosine_seeded_single(s, t_node, budget)
    if r: overlap_bins[ov]["cs"] += 1
    r, _ = cosine_seeded_multi(s, t_node, budget)
    if r: overlap_bins[ov]["cm"] += 1
    r, _ = bfs_single(s, t_node, budget)
    if r: overlap_bins[ov]["bf"] += 1

print(f"  {'Overlap':>8s} {'N':>5s} {'CosSingle':>10s} {'CosMulti':>10s} {'BFS':>6s} {'Multi%':>8s} {'BFS%':>8s}")
for ov in sorted(overlap_bins.keys()):
    st = overlap_bins[ov]
    cs = st["cs"]/st["t"]; cm = st["cm"]/st["t"]; bf = st["bf"]/st["t"]
    mi = (cm-cs)/max(cs,0.001)*100; bi = (bf-cs)/max(cs,0.001)*100
    print(f"  {ov:8d} {st['t']:5d} {cs:10.3f} {cm:10.3f} {bf:6.3f} {mi:+8.1f}% {bi:+8.1f}%")


# ══════════════════════════════════════════════════════════════════
# STRATIFIED: by cosine similarity (budget=200)
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("STRATIFIED BY SOURCE-TARGET COSINE (Budget=200)")
print(f"{'=' * 80}")

sim_bins = [(0.0, 0.3, "low"), (0.3, 0.5, "med"), (0.5, 0.7, "high"), (0.7, 1.0, "vhigh")]
for lo, hi, label in sim_bins:
    subset = [p for p in pairs if lo <= p["cosine_similarity"] < hi]
    if not subset:
        continue
    cs_r = cm_r = bf_r = 0
    for p in subset:
        s, t = p["source"], p["target"]
        r, _ = cosine_seeded_single(s, t, budget); cs_r += r
        r, _ = cosine_seeded_multi(s, t, budget); cm_r += r
        r, _ = bfs_single(s, t, budget); bf_r += r
    n = len(subset)
    cs = cs_r/n; cm = cm_r/n; bf = bf_r/n
    mi = (cm-cs)/max(cs,0.001)*100
    print(f"  cos {lo:.1f}-{hi:.1f} ({label:>5s}, N={n:3d}): "
          f"single={cs:.3f} multi={cm:.3f} bfs={bf:.3f} multi_imp={mi:+.1f}%")


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("SUMMARY")
print(f"{'=' * 80}")

print(f"\n  {'Budget':>8s} {'CosSingle':>10s} {'CosMulti':>10s} {'BFS':>6s} {'Multi/Single':>13s} {'BFS/Cosine':>11s}")
for b in BUDGETS:
    r = results_by_budget[b]
    print(f"  {b:8d} {r['cosine_single']:10.3f} {r['cosine_multi']:10.3f} "
          f"{r['bfs']:6.3f} {r['multi_vs_single_pct']:+13.1f}% {r['bfs_vs_cosine_pct']:+11.1f}%")

print(f"\n  KEY FINDINGS:")
r200 = results_by_budget[200]
cs, cm, bf = r200["cosine_single"], r200["cosine_multi"], r200["bfs"]

if bf > cs + 0.02:
    print(f"  1. VGSG TRAPPING DEMONSTRATED: BFS reaches {bf:.3f} vs cosine {cs:.3f} at budget 200")
    print(f"     Cosine-biased expansion gets trapped in the angular basin.")
else:
    print(f"  1. BFS ({bf:.3f}) ~ cosine single ({cs:.3f}) at budget 200")

if cm > cs + 0.02:
    print(f"  2. MULTI-ANCHOR IMPROVES: multi {cm:.3f} vs single {cs:.3f} ({r200['multi_vs_single_pct']:+.1f}%)")
    print(f"     Seeding from both ends breaks single-basin confinement.")
else:
    print(f"  2. Multi-anchor ({cm:.3f}) ~ single ({cs:.3f}) at budget 200")

# Save
output = {
    "graph": "PRSM Crystal", "n_grains": N, "n_pairs": len(pairs),
    "k_seed": K_SEED,
    "mean_cosine": float(np.mean([p["cosine_similarity"] for p in pairs])),
    "zero_overlap": sum(1 for p in pairs if p["corpus_overlap"] == 0),
    "results_by_budget": results_by_budget,
}
outfile = OUT / "exp6_cosine_seeded_intervention.json"
with open(outfile, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {outfile}")
print("COSINE-SEEDED INTERVENTION COMPLETE")
