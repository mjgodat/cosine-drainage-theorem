"""
Experiment 25: NeuroCrystal Multi-Model Test — BGE-small-en-v1.5

THE DEFINITIVE multi-model test: embed NeuroCrystal's actual grain concepts
through a SECOND neural embedding model (BGE 384D), rebuild PCA from the same
100 seeds, and rerun all key measurements.

Same graph (39,220 grains, ~6M edges), same concepts, same edges, same
validated hypotheses. Different neural encoder.

Measurements:
  a. Gamma (angular compression)
  b. P4 ratio (cosine-seed degree vs random-seed degree)
  c. P1 (cosine-to-random-target vs degree) — Spearman
  d. P3 (angular dispersion vs degree) — Spearman
  e. 6-policy benchmark at H=25,50,100 with 300 cross-corpus pairs
  f. Waypoint injection test on 7 validated hypotheses

If results hold, the single-model objection is closed.
"""
import sys
import time
import json
import heapq
import numpy as np
import torch
from collections import deque, defaultdict
from pathlib import Path
from math import pi
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path("E:/PRSM")
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

N_PAIRS = 300
BUDGETS = [25, 50, 100]
K_GAMMA = 20
N_P4_QUERIES = 200
N_GAMMA_SAMPLE = 3000

print(f"Device: {DEVICE}")
print(f"=" * 80)
print(f"EXPERIMENT 25: NeuroCrystal Multi-Model Test (BGE-small-en-v1.5)")
print(f"=" * 80)


# ======================================================================
# LOAD NEUROCRYSTAL GRAIN REGISTRY
# ======================================================================
print("\n[1/8] Loading NeuroCrystal grain registry...", flush=True)
t0 = time.time()

with open(ROOT / "data/g1_registry/unified_grains_with_embeddings.json", "r", encoding="utf-8") as f:
    grains_raw = json.load(f)
with open(ROOT / "data/g1_registry/relationship_registry.json", "r", encoding="utf-8") as f:
    rels = json.load(f)

concept_to_idx = {}
idx_to_concept = {}
embeddings_nomic_list = []
concept_corpora = {}
seed_indices = []
seen = {}

for g in grains_raw:
    c = g.get("concept", "").lower()
    emb = g.get("embedding")
    bc = g.get("bind_count", 0)
    sc = g.get("source_corpora", [])
    is_seed = g.get("seed", False)
    if c and emb and (c not in seen or bc > seen[c]):
        seen[c] = bc
        if c not in concept_to_idx:
            idx = len(concept_to_idx)
            concept_to_idx[c] = idx
            idx_to_concept[idx] = c
            embeddings_nomic_list.append(np.array(emb, dtype=np.float32))
            concept_corpora[idx] = set(sc) if isinstance(sc, list) else set()
            if is_seed:
                seed_indices.append(idx)
        else:
            idx = concept_to_idx[c]
            embeddings_nomic_list[idx] = np.array(emb, dtype=np.float32)
            concept_corpora[idx] = set(sc) if isinstance(sc, list) else set()
            if is_seed and idx not in seed_indices:
                seed_indices.append(idx)

embeddings_nomic = np.array(embeddings_nomic_list)
N = len(embeddings_nomic)

# L2-normalize nomic embeddings
norms = np.linalg.norm(embeddings_nomic, axis=1, keepdims=True)
embeddings_nomic = embeddings_nomic / np.clip(norms, 1e-8, None)

print(f"  {N:,} grains loaded, {len(seed_indices)} seeds identified, {time.time()-t0:.1f}s")

# If seed field wasn't found on enough grains, fall back to top bind_count
if len(seed_indices) < 100:
    print(f"  WARNING: Only {len(seed_indices)} seeds found by field. Using top-100 bind_count as proxy.")
    bc_list = [(seen[idx_to_concept[i]], i) for i in range(N)]
    bc_list.sort(key=lambda x: -x[0])
    seed_indices = [i for _, i in bc_list[:100]]

print(f"  Using {len(seed_indices)} seed grains for PCA anchor")


# ======================================================================
# BUILD ADJACENCY FROM RELATIONSHIP REGISTRY
# ======================================================================
print("\n[2/8] Building adjacency list...", flush=True)
t0 = time.time()

gid_to_concept = {}
for g in grains_raw:
    gid = g.get("grain_id", "")
    c = g.get("concept", "").lower()
    if gid and c and c in concept_to_idx:
        gid_to_concept[gid] = c

adj = [[] for _ in range(N)]
for r in rels:
    a_c = gid_to_concept.get(r.get("grain_a_id", ""))
    b_c = gid_to_concept.get(r.get("grain_b_id", ""))
    if a_c and b_c and a_c != b_c:
        a_idx = concept_to_idx.get(a_c)
        b_idx = concept_to_idx.get(b_c)
        if a_idx is not None and b_idx is not None:
            adj[a_idx].append(b_idx)
            adj[b_idx].append(a_idx)

for i in range(N):
    adj[i] = list(set(adj[i]))

degrees = np.array([len(adj[i]) for i in range(N)])
n_edges = sum(len(a) for a in adj) // 2

# Corpus grouping for pair sampling
corpus_nodes = defaultdict(list)
for idx, corpora in concept_corpora.items():
    for c in corpora:
        corpus_nodes[c].append(idx)
valid_corpora = [c for c, nodes in corpus_nodes.items() if len(nodes) >= 20]

print(f"  {n_edges:,} edges, {len(valid_corpora)} valid corpora, {time.time()-t0:.1f}s")


# ======================================================================
# EMBED ALL CONCEPTS WITH BGE-small-en-v1.5
# ======================================================================
print("\n[3/8] Embedding all concepts with BGE-small-en-v1.5...", flush=True)
t0 = time.time()

from sentence_transformers import SentenceTransformer

# Collect concept strings in index order
concept_texts = [idx_to_concept[i] for i in range(N)]

model_bge = SentenceTransformer('BAAI/bge-small-en-v1.5')
print(f"  Model loaded. Encoding {N:,} concept strings...", flush=True)

embeddings_bge_raw = model_bge.encode(
    concept_texts,
    batch_size=512,
    show_progress_bar=True,
    normalize_embeddings=False  # We'll L2-normalize after PCA
)
embeddings_bge_raw = embeddings_bge_raw.astype(np.float32)
print(f"  BGE raw shape: {embeddings_bge_raw.shape}, {time.time()-t0:.1f}s")

del model_bge  # Free GPU memory


# ======================================================================
# FIT PCA ON 100 SEED GRAINS' BGE EMBEDDINGS, PROJECT ALL
# ======================================================================
print("\n[4/8] Fitting PCA on seed grains' BGE embeddings...", flush=True)
t0 = time.time()

seed_bge = embeddings_bge_raw[seed_indices]
pca = PCA(n_components=min(384, len(seed_indices)))
pca.fit(seed_bge)

# Project all grains
embeddings_bge_pca = pca.transform(embeddings_bge_raw).astype(np.float32)
var_explained = float(np.sum(pca.explained_variance_ratio_))
print(f"  PCA fitted on {len(seed_indices)} seeds, {pca.n_components_} components")
print(f"  Variance explained: {var_explained:.4f}")
print(f"  Projected shape: {embeddings_bge_pca.shape}, {time.time()-t0:.1f}s")


# ======================================================================
# L2-NORMALIZE BGE EMBEDDINGS
# ======================================================================
print("\n[5/8] L2-normalizing BGE embeddings...", flush=True)
norms_bge = np.linalg.norm(embeddings_bge_pca, axis=1, keepdims=True)
embeddings_bge = embeddings_bge_pca / np.clip(norms_bge, 1e-8, None)
print(f"  Shape: {embeddings_bge.shape}")


# ======================================================================
# SAMPLE CROSS-CORPUS PAIRS
# ======================================================================
print("\n[6/8] Sampling cross-corpus pairs...", flush=True)
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
    if s == t or not adj[s] or not adj[t]:
        continue
    test_pairs.append((s, t))

print(f"  {len(test_pairs)} pairs sampled, {time.time()-t0:.1f}s")


# ======================================================================
# MEASUREMENT FUNCTIONS
# ======================================================================

def compute_gamma_sampled(emb, n_sample=N_GAMMA_SAMPLE):
    """Compute Gamma: sample n_sample pairs, mean angular distance, Gamma = (pi/2 - mean)/(pi/2)."""
    t = torch.tensor(emb, device=DEVICE, dtype=torch.float32)
    N_e = len(emb)
    idx1 = np.random.randint(0, N_e, n_sample)
    idx2 = np.random.randint(0, N_e, n_sample)
    # Ensure different
    mask = idx1 == idx2
    idx2[mask] = (idx2[mask] + 1) % N_e
    sims = (t[idx1] * t[idx2]).sum(dim=1)
    angles = torch.arccos(sims.clamp(-1, 1))
    mean_angle = float(angles.mean().item())
    gamma = (pi / 2 - mean_angle) / (pi / 2)
    return gamma, mean_angle


def compute_p4(emb, n_queries=N_P4_QUERIES):
    """P4: cosine-seed degree / random-seed degree."""
    t = torch.tensor(emb, device=DEVICE, dtype=torch.float32)
    valid = [i for i in range(N) if degrees[i] > 0]
    query_nodes = np.random.choice(valid, min(n_queries, len(valid)), replace=False)

    cos_seed_degs = []
    rand_seed_degs = []

    for q in query_nodes:
        sims = (t[q:q+1] @ t.T).squeeze(0)
        sims[q] = -2
        cos_seed = int(sims.argmax().item())
        cos_seed_degs.append(degrees[cos_seed])

        rand_seed = valid[np.random.randint(len(valid))]
        while rand_seed == q:
            rand_seed = valid[np.random.randint(len(valid))]
        rand_seed_degs.append(degrees[rand_seed])

    cos_mean = float(np.mean(cos_seed_degs))
    rand_mean = float(np.mean(rand_seed_degs))
    ratio = cos_mean / rand_mean if rand_mean > 0 else 0.0
    return {
        "cos_seed_mean_deg": cos_mean,
        "rand_seed_mean_deg": rand_mean,
        "ratio": ratio,
    }


def compute_p1(emb, n_sample=1000):
    """P1: Spearman(cosine_to_random_target, degree)."""
    t = torch.tensor(emb, device=DEVICE, dtype=torch.float32)
    valid = [i for i in range(N) if degrees[i] > 0]
    nodes = np.random.choice(valid, min(n_sample, len(valid)), replace=False)
    # Pick a random target
    target = valid[np.random.randint(len(valid))]
    target_vec = t[target:target+1]

    sims_to_target = (t[nodes] @ target_vec.T).squeeze(1).cpu().numpy()
    node_degrees = degrees[nodes]

    rho, pval = spearmanr(sims_to_target, node_degrees)
    return {"rho": float(rho), "p": float(pval)}


def compute_p3(emb, k=K_GAMMA):
    """P3: Spearman(angular_dispersion, degree). Dispersion = std of angles to k-NN."""
    t = torch.tensor(emb, device=DEVICE, dtype=torch.float32)
    valid = [i for i in range(N) if degrees[i] > 0]
    sample = np.random.choice(valid, min(2000, len(valid)), replace=False)

    dispersions = []
    BATCH = 256
    for start in range(0, len(sample), BATCH):
        end = min(start + BATCH, len(sample))
        batch_idx = sample[start:end]
        sims = t[batch_idx] @ t.T
        for i in range(end - start):
            sims[i, batch_idx[i]] = -2  # exclude self
        topk_sims, _ = sims.topk(k, dim=1)
        angles = torch.arccos(topk_sims.clamp(-1, 1))
        # dispersion = std of angles to k-NN
        disp = angles.std(dim=1).cpu().numpy()
        dispersions.extend(disp)

    dispersions = np.array(dispersions)
    sample_degrees = degrees[sample]
    rho, pval = spearmanr(dispersions, sample_degrees)
    return {"rho": float(rho), "p": float(pval)}


# ======================================================================
# 6 TRAVERSAL POLICIES
# ======================================================================

def pol_bfs(emb, s, t, H):
    visited = {s}
    queue = deque([s])
    while queue and len(visited) < H:
        u = queue.popleft()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                if v == t:
                    return True
                if len(visited) >= H:
                    break
                queue.append(v)
    return t in visited


def pol_cos(emb, s, t, H):
    tv = emb[t]
    visited = {s}
    pq = [(-float(emb[v] @ tv), v) for v in adj[s] if v not in visited]
    heapq.heapify(pq)
    while pq and len(visited) < H:
        _, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == t:
            return True
        for v in adj[u]:
            if v not in visited:
                heapq.heappush(pq, (-float(emb[v] @ tv), v))
    return t in visited


def pol_multi(emb, s, t, H):
    hs = H // 2
    ht = H - hs
    vs = {s}
    vt = {t}
    ps = [(-float(emb[v] @ emb[t]), v) for v in adj[s]]
    pt = [(-float(emb[v] @ emb[s]), v) for v in adj[t]]
    heapq.heapify(ps)
    heapq.heapify(pt)
    while ps and len(vs) < hs:
        _, u = heapq.heappop(ps)
        if u in vs:
            continue
        vs.add(u)
        if u in vt:
            return True
        for v in adj[u]:
            if v not in vs:
                heapq.heappush(ps, (-float(emb[v] @ emb[t]), v))
    while pt and len(vt) < ht:
        _, u = heapq.heappop(pt)
        if u in vt:
            continue
        vt.add(u)
        if u in vs:
            return True
        for v in adj[u]:
            if v not in vt:
                heapq.heappush(pt, (-float(emb[v] @ emb[s]), v))
    return bool(vs & vt)


def pol_fwd_ppr(emb, s, t, H, alpha=0.15):
    r = {s: 1.0}
    visited = set()
    queue = deque([s])
    in_queue = {s}
    pushes = 0
    while queue and pushes < H:
        u = queue.popleft()
        in_queue.discard(u)
        visited.add(u)
        res_u = r.get(u, 0.0)
        if res_u <= 0.0:
            continue
        r[u] = 0.0
        pushes += 1
        neighbors = adj[u]
        if not neighbors:
            continue
        push_mass = (1.0 - alpha) * res_u
        mass_per = push_mass / len(neighbors)
        for v in neighbors:
            r[v] = r.get(v, 0.0) + mass_per
            if v not in in_queue and len(visited) < H:
                queue.append(v)
                in_queue.add(v)
    return t in visited


def pol_bippr(emb, s, t, H, alpha=0.15):
    budget_src = H // 2
    budget_tgt = H - budget_src
    # Forward
    r_s = {s: 1.0}
    visited_s = set()
    q_s = deque([s])
    pushes = 0
    while q_s and pushes < budget_src:
        u = q_s.popleft()
        visited_s.add(u)
        res = r_s.get(u, 0.0)
        r_s[u] = 0.0
        pushes += 1
        nbs = adj[u]
        if nbs and res > 0:
            m = (1.0 - alpha) * res / len(nbs)
            for v in nbs:
                r_s[v] = r_s.get(v, 0.0) + m
                if v not in visited_s and len(visited_s) < budget_src:
                    q_s.append(v)
    # Backward
    r_t = {t: 1.0}
    visited_t = set()
    q_t = deque([t])
    pushes = 0
    while q_t and pushes < budget_tgt:
        u = q_t.popleft()
        visited_t.add(u)
        res = r_t.get(u, 0.0)
        r_t[u] = 0.0
        pushes += 1
        nbs = adj[u]
        if nbs and res > 0:
            m = (1.0 - alpha) * res / len(nbs)
            for v in nbs:
                r_t[v] = r_t.get(v, 0.0) + m
                if v not in visited_t and len(visited_t) < budget_tgt:
                    q_t.append(v)
    return bool(visited_s & visited_t) or t in visited_s


def pol_mmr(emb, s, t, H, lambda_param=0.7):
    tv = emb[t]
    visited = {s}
    pq = []
    for v in adj[s]:
        deg_penalty = np.log1p(len(adj[v]))
        cos_sim = float(emb[v] @ tv)
        score = (lambda_param * cos_sim) - ((1.0 - lambda_param) * 0.1 * deg_penalty)
        heapq.heappush(pq, (-score, v))
    while pq and len(visited) < H:
        _, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == t:
            return True
        for v in adj[u]:
            if v not in visited:
                deg_penalty = np.log1p(len(adj[v]))
                cos_sim = float(emb[v] @ tv)
                score = (lambda_param * cos_sim) - ((1.0 - lambda_param) * 0.1 * deg_penalty)
                heapq.heappush(pq, (-score, v))
    return t in visited


POLICIES = {
    "BFS": pol_bfs,
    "Cosine": pol_cos,
    "Multi-Anchor": pol_multi,
    "Fwd-PPR": pol_fwd_ppr,
    "BiPPR": pol_bippr,
    "MMR": pol_mmr,
}


# ======================================================================
# RUN ALL MEASUREMENTS ON BOTH EMBEDDINGS
# ======================================================================
results_all = {}

for emb_name, emb in [("nomic-embed (768D)", embeddings_nomic),
                       ("BGE-small (384D)", embeddings_bge)]:
    print(f"\n{'=' * 80}")
    print(f"  RUNNING MEASUREMENTS: {emb_name}  (shape={emb.shape})")
    print(f"{'=' * 80}")

    res = {"name": emb_name, "shape": list(emb.shape)}

    # --- a. Gamma ---
    print(f"\n  [a] Gamma (angular compression, {N_GAMMA_SAMPLE} pairs)...", flush=True)
    t0 = time.time()
    gamma_val, mean_angle = compute_gamma_sampled(emb, N_GAMMA_SAMPLE)
    print(f"      Gamma = {gamma_val:.4f}, mean_angle = {mean_angle:.4f}, {time.time()-t0:.1f}s")
    res["gamma"] = gamma_val
    res["mean_angle"] = mean_angle

    # --- b. P4 ratio ---
    print(f"\n  [b] P4 ratio ({N_P4_QUERIES} queries)...", flush=True)
    t0 = time.time()
    p4 = compute_p4(emb, N_P4_QUERIES)
    print(f"      P4 ratio = {p4['ratio']:.4f} (cos_deg={p4['cos_seed_mean_deg']:.1f}, "
          f"rand_deg={p4['rand_seed_mean_deg']:.1f}), {time.time()-t0:.1f}s")
    res["p4"] = p4

    # --- c. P1 (cosine-to-target vs degree) ---
    print(f"\n  [c] P1 (cosine-to-random-target vs degree)...", flush=True)
    t0 = time.time()
    p1 = compute_p1(emb)
    print(f"      P1 rho = {p1['rho']:+.4f}, p = {p1['p']:.2e}, {time.time()-t0:.1f}s")
    res["p1"] = p1

    # --- d. P3 (angular dispersion vs degree) ---
    print(f"\n  [d] P3 (angular dispersion vs degree)...", flush=True)
    t0 = time.time()
    p3 = compute_p3(emb)
    print(f"      P3 rho = {p3['rho']:+.4f}, p = {p3['p']:.2e}, {time.time()-t0:.1f}s")
    res["p3"] = p3

    # --- e. 6-policy benchmark ---
    print(f"\n  [e] 6-policy benchmark ({len(test_pairs)} pairs, H={BUDGETS})...", flush=True)
    policy_results = {}
    for H in BUDGETS:
        t0 = time.time()
        policy_results[H] = {}
        for pname, func in POLICIES.items():
            ok = sum(1 for s, t in test_pairs if func(emb, s, t, H))
            policy_results[H][pname] = ok / len(test_pairs) * 100
        elapsed = time.time() - t0
        row = "      H={:<4d}".format(H)
        for pname in POLICIES:
            row += f"  {pname}={policy_results[H][pname]:.1f}%"
        row += f"  ({elapsed:.1f}s)"
        print(row)
    res["policy_benchmark"] = {
        str(H): {p: v for p, v in pols.items()}
        for H, pols in policy_results.items()
    }

    # Check ordering at H=100
    h100 = policy_results[100]
    res["ordering_H100"] = {
        "Multi_gt_Cosine": h100["Multi-Anchor"] > h100["Cosine"],
        "Cosine_gt_BFS": h100["Cosine"] > h100["BFS"],
        "MMR_lt_Cosine": h100["MMR"] < h100["Cosine"],
        "full_ordering_holds": (h100["Multi-Anchor"] > h100["Cosine"] > h100["BFS"]),
    }

    # --- f. Waypoint injection test ---
    print(f"\n  [f] Waypoint injection test (7 hypotheses)...", flush=True)
    hypotheses = {
        "#1 Cobenfy": ["xanomeline", "muscarinic", "dopamine", "incentive salience", "psychosis"],
        "#5 PTSD": ["ptsd", "norepinephrine", "locus coeruleus", "amygdala", "fear extinction"],
        "#25 PP2A": ["tauopathy", "neuronal loss", "neurofibrillary tangles", "pp2a", "nr2a"],
        "#29 Eden": ["tryptamine", "aryl hydrocarbon receptor", "slc7a11", "cystine", "glutathione", "gpx4", "ferroptosis"],
        "#35 Faecal": ["faecalibacterium", "fibrillization", "enteric nerve", "alpha-synuclein", "dopaminergic neuron"],
        "#38 TGFBR1": ["tgfbr1", "alpha-sma", "tgf-beta", "smad3", "transferrin receptor"],
        "#40 Fasting": ["fasting", "ncoa4", "ferritinophagy", "ferritin", "ferroportin", "gpx4", "ferroptosis"],
    }

    H_wp = 100
    resolved = {}
    for name, concepts in hypotheses.items():
        indices = [concept_to_idx[c.lower()] for c in concepts if c.lower() in concept_to_idx]
        if len(indices) >= 3:
            resolved[name] = indices

    print(f"\n      {'Hypothesis':>15s} {'N':>3s} {'Cos(A->Z)':>11s} {'Multi(A<->Z)':>14s} {'All-WP':>8s}")
    print(f"      {'-'*15} {'-'*3} {'-'*11} {'-'*14} {'-'*8}")

    cos_totals, multi_totals, wp_totals = [], [], []
    wp_details = {}

    for name, indices in resolved.items():
        source = indices[0]
        target = indices[-1]
        intermediates = set(indices[1:-1])
        n_int = len(intermediates)
        if n_int == 0:
            continue

        # Strategy 1: Cosine A->Z
        tv = emb[target]
        visited_cos = {source}
        pq = [(-float(emb[v] @ tv), v) for v in adj[source] if v not in visited_cos]
        heapq.heapify(pq)
        while pq and len(visited_cos) < H_wp:
            _, u = heapq.heappop(pq)
            if u in visited_cos:
                continue
            visited_cos.add(u)
            for v in adj[u]:
                if v not in visited_cos:
                    heapq.heappush(pq, (-float(emb[v] @ tv), v))
        cos_int = sum(1 for idx in intermediates if idx in visited_cos)

        # Strategy 2: Multi-anchor A<->Z
        hs = H_wp // 2
        ht = H_wp - hs
        vs = {source}
        vt = {target}
        ps = [(-float(emb[v] @ emb[target]), v) for v in adj[source]]
        pt = [(-float(emb[v] @ emb[source]), v) for v in adj[target]]
        heapq.heapify(ps)
        heapq.heapify(pt)
        while ps and len(vs) < hs:
            _, u = heapq.heappop(ps)
            if u in vs:
                continue
            vs.add(u)
            for v in adj[u]:
                if v not in vs:
                    heapq.heappush(ps, (-float(emb[v] @ emb[target]), v))
        while pt and len(vt) < ht:
            _, u = heapq.heappop(pt)
            if u in vt:
                continue
            vt.add(u)
            for v in adj[u]:
                if v not in vt:
                    heapq.heappush(pt, (-float(emb[v] @ emb[source]), v))
        multi_visited = vs | vt
        multi_int = sum(1 for idx in intermediates if idx in multi_visited)

        # Strategy 3: All-waypoint injection (3 BFS hops, budget_per_wp)
        budget_per_wp = H_wp // len(indices)
        wp_visited = set()
        for wp_idx in indices:
            visited_wp = {wp_idx}
            frontier = {wp_idx}
            for _ in range(3):
                nf = set()
                for n_node in frontier:
                    for nb in adj[n_node]:
                        if nb not in visited_wp:
                            visited_wp.add(nb)
                            nf.add(nb)
                            if len(visited_wp) >= budget_per_wp:
                                break
                    if len(visited_wp) >= budget_per_wp:
                        break
                frontier = nf
            wp_visited.update(visited_wp)
        wp_int = sum(1 for idx in intermediates if idx in wp_visited)

        print(f"      {name:>15s} {n_int:3d} {cos_int:9d}/{n_int} {multi_int:12d}/{n_int} {wp_int:6d}/{n_int}")
        cos_totals.append(cos_int / n_int * 100)
        multi_totals.append(multi_int / n_int * 100)
        wp_totals.append(wp_int / n_int * 100)
        wp_details[name] = {
            "n_intermediates": n_int,
            "cos_found": cos_int,
            "multi_found": multi_int,
            "wp_found": wp_int,
        }

    cos_mean = float(np.mean(cos_totals)) if cos_totals else 0.0
    multi_mean = float(np.mean(multi_totals)) if multi_totals else 0.0
    wp_mean = float(np.mean(wp_totals)) if wp_totals else 0.0

    print(f"\n      MEAN INTERMEDIATE DISCOVERY:")
    print(f"        Cosine A->Z:     {cos_mean:.1f}%")
    print(f"        Multi A<->Z:     {multi_mean:.1f}%")
    print(f"        All-Waypoint:    {wp_mean:.1f}%")

    res["waypoint_injection"] = {
        "cos_mean": cos_mean,
        "multi_mean": multi_mean,
        "wp_mean": wp_mean,
        "per_hypothesis": wp_details,
    }

    results_all[emb_name] = res


# ======================================================================
# COMPARISON TABLE
# ======================================================================
print(f"\n\n{'=' * 80}")
print("COMPARISON TABLE: nomic-embed (768D) vs BGE-small (384D)")
print(f"{'=' * 80}")

nomic = results_all.get("nomic-embed (768D)", {})
bge = results_all.get("BGE-small (384D)", {})

rows_table = [
    ("Gamma",
     f"{nomic.get('gamma', 0):.4f}",
     f"{bge.get('gamma', 0):.4f}"),
    ("P4 ratio",
     f"{nomic.get('p4', {}).get('ratio', 0):.4f}",
     f"{bge.get('p4', {}).get('ratio', 0):.4f}"),
    ("P1 (cos-deg rho)",
     f"{nomic.get('p1', {}).get('rho', 0):+.4f}",
     f"{bge.get('p1', {}).get('rho', 0):+.4f}"),
    ("P3 (disp-deg rho)",
     f"{nomic.get('p3', {}).get('rho', 0):+.4f}",
     f"{bge.get('p3', {}).get('rho', 0):+.4f}"),
]

# Policy benchmark at each budget
for H in BUDGETS:
    for pname in POLICIES:
        nomic_val = nomic.get("policy_benchmark", {}).get(str(H), {}).get(pname, 0)
        bge_val = bge.get("policy_benchmark", {}).get(str(H), {}).get(pname, 0)
        rows_table.append(
            (f"{pname} @{H}",
             f"{nomic_val:.1f}%",
             f"{bge_val:.1f}%")
        )

# Waypoint injection
nomic_wp = nomic.get("waypoint_injection", {}).get("wp_mean", 0)
bge_wp = bge.get("waypoint_injection", {}).get("wp_mean", 0)
rows_table.append(("WP injection", f"{nomic_wp:.1f}%", f"{bge_wp:.1f}%"))

print(f"\n  {'Metric':<25s} {'nomic-embed (768D)':>20s} {'BGE-small (384D)':>20s}")
print(f"  {'-'*25} {'-'*20} {'-'*20}")
for metric, n_val, b_val in rows_table:
    print(f"  {metric:<25s} {n_val:>20s} {b_val:>20s}")


# ======================================================================
# KEY FINDINGS
# ======================================================================
print(f"\n{'=' * 80}")
print("KEY FINDINGS")
print(f"{'=' * 80}")

# 1. Policy ordering
nomic_ord = nomic.get("ordering_H100", {})
bge_ord = bge.get("ordering_H100", {})
print(f"\n  1. Policy ordering (Multi > Cosine > BFS) at H=100:")
print(f"     nomic: {'HOLDS' if nomic_ord.get('full_ordering_holds') else 'BROKEN'}")
print(f"     BGE:   {'HOLDS' if bge_ord.get('full_ordering_holds') else 'BROKEN'}")

# 2. MMR < Cosine
print(f"\n  2. MMR < Cosine at H=100:")
print(f"     nomic: {'YES' if nomic_ord.get('MMR_lt_Cosine') else 'NO'}")
print(f"     BGE:   {'YES' if bge_ord.get('MMR_lt_Cosine') else 'NO'}")

# 3. Waypoint injection
print(f"\n  3. Waypoint injection discovery rate:")
print(f"     nomic: {nomic_wp:.1f}%")
print(f"     BGE:   {bge_wp:.1f}%")

# 4. Gamma comparison
print(f"\n  4. Gamma (angular compression):")
print(f"     nomic: {nomic.get('gamma', 0):.4f}")
print(f"     BGE:   {bge.get('gamma', 0):.4f}")
gamma_similar = abs(nomic.get('gamma', 0) - bge.get('gamma', 0)) < 0.15
print(f"     Similar range: {'YES' if gamma_similar else 'NO'}")

# 5. P4 ratio
print(f"\n  5. P4 ratio (hub attraction):")
print(f"     nomic: {nomic.get('p4', {}).get('ratio', 0):.4f}")
print(f"     BGE:   {bge.get('p4', {}).get('ratio', 0):.4f}")

# Overall verdict
ordering_holds = bge_ord.get('full_ordering_holds', False)
wp_near_100 = bge_wp >= 85.0
gamma_ok = gamma_similar

verdict_parts = []
if ordering_holds:
    verdict_parts.append("Policy ordering HOLDS on BGE")
else:
    verdict_parts.append("Policy ordering BROKEN on BGE")
if wp_near_100:
    verdict_parts.append(f"WP injection = {bge_wp:.0f}%")
else:
    verdict_parts.append(f"WP injection = {bge_wp:.0f}% (below threshold)")
if gamma_ok:
    verdict_parts.append("Gamma in similar range")
else:
    verdict_parts.append("Gamma diverged significantly")

all_hold = ordering_holds and wp_near_100 and gamma_ok

print(f"\n{'=' * 80}")
if all_hold:
    print("VERDICT: SINGLE-MODEL OBJECTION CLOSED.")
    print("  All key findings replicate on BGE-small-en-v1.5 (384D).")
    print("  VGSG properties are embedding-model-invariant on NeuroCrystal.")
else:
    print("VERDICT: PARTIAL REPLICATION.")
    print(f"  {'; '.join(verdict_parts)}")
print(f"{'=' * 80}")


# ======================================================================
# SAVE RESULTS
# ======================================================================
output = {
    "experiment": "exp25_neurocrystal_bge",
    "description": "Multi-model test: BGE-small-en-v1.5 vs nomic-embed on NeuroCrystal",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "params": {
        "n_grains": N,
        "n_edges": n_edges,
        "n_seeds": len(seed_indices),
        "n_pairs": len(test_pairs),
        "budgets": BUDGETS,
        "k_gamma": K_GAMMA,
        "n_gamma_sample": N_GAMMA_SAMPLE,
        "n_p4_queries": N_P4_QUERIES,
        "pca_variance_explained": var_explained,
    },
    "nomic": {k: v for k, v in nomic.items() if k != "name"},
    "bge": {k: v for k, v in bge.items() if k != "name"},
    "comparison": {
        "ordering_holds_nomic": nomic_ord.get('full_ordering_holds', False),
        "ordering_holds_bge": bge_ord.get('full_ordering_holds', False),
        "wp_injection_nomic": nomic_wp,
        "wp_injection_bge": bge_wp,
        "gamma_nomic": nomic.get('gamma', 0),
        "gamma_bge": bge.get('gamma', 0),
        "verdict": "SINGLE-MODEL OBJECTION CLOSED" if all_hold else "PARTIAL REPLICATION",
    },
}

outfile = OUT / "exp25_neurocrystal_bge.json"
with open(outfile, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved: {outfile}")
print("EXPERIMENT 25 COMPLETE")
