"""
Experiment 19: Probing the Open Problems

Three probes on existing data:

PROBE A: Improve absolute prediction (MAE 10.6pp)
  - Add interaction term (degree_cv × gamma) for scale-free residual
  - Try log(degree_cv) and gamma² nonlinear terms
  - Does PRSM residual shrink?

PROBE B: Policy-dependent regime inside the graph ball
  - For pairs where d_G(s,t) ≤ H, how often does cosine STILL miss?
  - This is the interesting trapping regime — targets are reachable but policy fails
  - Quantify miss rate as function of angular separation at FIXED graph distance

PROBE C: Downstream reframe — PRSM waypoint injection vs single-direction
  - For validated hypotheses, compare:
    (a) Single-source cosine from endpoint A toward endpoint Z (target-directed)
    (b) Multi-anchor from both ends
    (c) PRSM-style: inject ALL waypoints, expand from each
  - Measure: what fraction of the VALIDATED intermediate concepts appear?
  - This tests whether waypoint injection (the actual PRSM architecture)
    outperforms even multi-anchor on discovery quality
"""
import sys
import time
import json
import heapq
import numpy as np
from collections import deque, defaultdict
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

ROOT = Path("E:/PRSM")
OUT = Path(__file__).resolve().parent / "results"

print("Loading PRSM Crystal...", end=" ", flush=True)
t0 = time.time()

with open(ROOT / "data/g1_registry/unified_grains_with_embeddings.json", "r", encoding="utf-8") as f:
    grains_raw = json.load(f)
with open(ROOT / "data/g1_registry/relationship_registry.json", "r", encoding="utf-8") as f:
    rels = json.load(f)

concept_to_idx = {}
idx_to_concept = {}
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
            idx_to_concept[idx] = c
            embeddings_list.append(np.array(emb, dtype=np.float32))
            concept_corpora[idx] = set(sc) if isinstance(sc, list) else set()
        else:
            idx = concept_to_idx[c]
            embeddings_list[idx] = np.array(emb, dtype=np.float32)
            concept_corpora[idx] = set(sc) if isinstance(sc, list) else set()

embeddings = np.array(embeddings_list)
N = len(embeddings)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / np.clip(norms, 1e-8, None)

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

corpus_nodes = defaultdict(list)
for idx, corpora in concept_corpora.items():
    for c in corpora:
        corpus_nodes[c].append(idx)
valid_corpora = [c for c, nodes in corpus_nodes.items() if len(nodes) >= 20]

print(f"{N:,} grains, {time.time()-t0:.1f}s")

results = {}


# ══════════════════════════════════════════════════════════════════
# PROBE A: Improve absolute prediction
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PROBE A: Can interaction terms reduce MAE on PRSM?")
print(f"{'=' * 70}")

# Data from exp15
rows = [
    {"graph": "PRSM", "gamma": 0.249, "log_H": np.log(25), "dcv": 2.694, "cos_reach": 73.2},
    {"graph": "PRSM", "gamma": 0.249, "log_H": np.log(50), "dcv": 2.694, "cos_reach": 77.8},
    {"graph": "PRSM", "gamma": 0.249, "log_H": np.log(100), "dcv": 2.694, "cos_reach": 82.2},
    {"graph": "PRSM", "gamma": 0.249, "log_H": np.log(200), "dcv": 2.694, "cos_reach": 86.6},
    {"graph": "PRSM", "gamma": 0.249, "log_H": np.log(500), "dcv": 2.694, "cos_reach": 92.4},
    {"graph": "Cora", "gamma": 0.036, "log_H": np.log(25), "dcv": 1.341, "cos_reach": 16.0},
    {"graph": "Cora", "gamma": 0.036, "log_H": np.log(50), "dcv": 1.341, "cos_reach": 28.7},
    {"graph": "Cora", "gamma": 0.036, "log_H": np.log(100), "dcv": 1.341, "cos_reach": 48.3},
    {"graph": "Cora", "gamma": 0.036, "log_H": np.log(200), "dcv": 1.341, "cos_reach": 69.7},
    {"graph": "Amazon", "gamma": 0.227, "log_H": np.log(25), "dcv": 1.941, "cos_reach": 45.7},
    {"graph": "Amazon", "gamma": 0.227, "log_H": np.log(50), "dcv": 1.941, "cos_reach": 65.3},
    {"graph": "Amazon", "gamma": 0.227, "log_H": np.log(100), "dcv": 1.941, "cos_reach": 77.0},
    {"graph": "Amazon", "gamma": 0.227, "log_H": np.log(200), "dcv": 1.941, "cos_reach": 85.7},
    {"graph": "DBLP", "gamma": 0.010, "log_H": np.log(25), "dcv": 1.566, "cos_reach": 15.3},
    {"graph": "DBLP", "gamma": 0.010, "log_H": np.log(50), "dcv": 1.566, "cos_reach": 28.3},
    {"graph": "DBLP", "gamma": 0.010, "log_H": np.log(100), "dcv": 1.566, "cos_reach": 39.7},
    {"graph": "DBLP", "gamma": 0.010, "log_H": np.log(200), "dcv": 1.566, "cos_reach": 49.7},
]

y = np.array([r["cos_reach"] for r in rows])
graphs = ["PRSM", "Cora", "Amazon", "DBLP"]

models = {
    "3-feature (baseline)": lambda r: [r["gamma"], r["log_H"], r["dcv"]],
    "+interaction (γ×dcv)": lambda r: [r["gamma"], r["log_H"], r["dcv"], r["gamma"]*r["dcv"]],
    "+γ² + log(dcv)": lambda r: [r["gamma"], r["log_H"], r["dcv"], r["gamma"]**2, np.log(r["dcv"])],
    "+γ×log(H) + dcv×log(H)": lambda r: [r["gamma"], r["log_H"], r["dcv"], r["gamma"]*r["log_H"], r["dcv"]*r["log_H"]],
    "full nonlinear": lambda r: [r["gamma"], r["log_H"], r["dcv"], r["gamma"]*r["dcv"], r["gamma"]*r["log_H"], r["dcv"]*r["log_H"], r["gamma"]**2],
}

print(f"\n  {'Model':>30s} {'R²':>6s} {'MAE':>8s} {'LOO-ρ':>8s} {'LOO-MAE':>9s} {'PRSM MAE':>10s}")
print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")

best_loo_mae = 999
best_model_name = ""

for mname, feat_fn in models.items():
    X = np.array([feat_fn(r) for r in rows])
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = LinearRegression().fit(Xs, y)
    yp = model.predict(Xs)
    r2 = r2_score(y, yp)
    mae = mean_absolute_error(y, yp)

    # Leave-one-graph-out
    all_pred, all_actual = [], []
    prsm_preds, prsm_actuals = [], []
    for held in graphs:
        train = [i for i, r in enumerate(rows) if r["graph"] != held]
        test = [i for i, r in enumerate(rows) if r["graph"] == held]
        Xtr = np.array([feat_fn(rows[i]) for i in train])
        ytr = y[train]
        Xte = np.array([feat_fn(rows[i]) for i in test])
        yte = y[test]
        sc = StandardScaler().fit(Xtr)
        m = LinearRegression().fit(sc.transform(Xtr), ytr)
        pred = m.predict(sc.transform(Xte))
        all_pred.extend(pred)
        all_actual.extend(yte)
        if held == "PRSM":
            prsm_preds.extend(pred)
            prsm_actuals.extend(yte)

    rho_loo, _ = spearmanr(all_actual, all_pred)
    mae_loo = np.mean(np.abs(np.array(all_pred) - np.array(all_actual)))
    prsm_mae = np.mean(np.abs(np.array(prsm_preds) - np.array(prsm_actuals)))

    print(f"  {mname:>30s} {r2:5.3f} {mae:7.2f} {rho_loo:7.3f} {mae_loo:8.1f} {prsm_mae:9.1f}")

    if mae_loo < best_loo_mae:
        best_loo_mae = mae_loo
        best_model_name = mname

print(f"\n  Best LOO model: {best_model_name} (MAE = {best_loo_mae:.1f} pp)")

results["probe_a"] = {"best_model": best_model_name, "best_loo_mae": float(best_loo_mae)}


# ══════════════════════════════════════════════════════════════════
# PROBE B: Policy-dependent trapping INSIDE the graph ball
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PROBE B: When d_G ≤ H, how often does cosine STILL miss?")
print(f"{'=' * 70}")

# Sample cross-corpus pairs, compute graph distance, test cosine reachability
pairs = []
attempts = 0
while len(pairs) < 300 and attempts < 15000:
    attempts += 1
    c1 = valid_corpora[np.random.randint(len(valid_corpora))]
    c2 = valid_corpora[np.random.randint(len(valid_corpora))]
    if c1 == c2: continue
    s = corpus_nodes[c1][np.random.randint(len(corpus_nodes[c1]))]
    t = corpus_nodes[c2][np.random.randint(len(corpus_nodes[c2]))]
    if s == t or not adj[s] or not adj[t]: continue

    # BFS distance
    visited = {s}
    frontier = {s}
    g_dist = -1
    for hop in range(1, 11):
        nf = set()
        for n in frontier:
            for nb in adj[n]:
                if nb == t:
                    g_dist = hop
                    break
                if nb not in visited:
                    visited.add(nb)
                    nf.add(nb)
            if g_dist > 0: break
        if g_dist > 0: break
        frontier = nf
    if g_dist < 0: g_dist = 11

    cos_st = float(embeddings[s] @ embeddings[t])
    ang_sep = float(np.arccos(np.clip(cos_st, -1, 1)))
    pairs.append({"s": s, "t": t, "g_dist": g_dist, "ang_sep": ang_sep, "cos": cos_st})

print(f"  {len(pairs)} pairs sampled")

# For each budget, check: among pairs where d_G ≤ H, does cosine find target?
for H in [50, 100, 200]:
    reachable_pairs = [p for p in pairs if p["g_dist"] <= H]
    if not reachable_pairs:
        continue

    cos_found = 0
    cos_missed = 0
    missed_ang_seps = []
    found_ang_seps = []

    for p in reachable_pairs:
        # Run cosine to target
        tv = embeddings[p["t"]]
        visited = {p["s"]}
        pq = [(-float(embeddings[v] @ tv), v) for v in adj[p["s"]] if v not in visited]
        heapq.heapify(pq)
        found = False
        while pq and len(visited) < H:
            _, u = heapq.heappop(pq)
            if u in visited: continue
            visited.add(u)
            if u == p["t"]:
                found = True
                break
            for v in adj[u]:
                if v not in visited:
                    heapq.heappush(pq, (-float(embeddings[v] @ tv), v))

        if found:
            cos_found += 1
            found_ang_seps.append(p["ang_sep"])
        else:
            cos_missed += 1
            missed_ang_seps.append(p["ang_sep"])

    total = len(reachable_pairs)
    miss_rate = cos_missed / total * 100 if total > 0 else 0

    print(f"\n  Budget H={H}: {total} pairs with d_G ≤ H")
    print(f"    Cosine found: {cos_found}/{total} ({cos_found/total*100:.1f}%)")
    print(f"    Cosine MISSED (inside ball): {cos_missed}/{total} ({miss_rate:.1f}%)")
    if missed_ang_seps and found_ang_seps:
        print(f"    Mean angular sep of FOUND:   {np.mean(found_ang_seps):.4f}")
        print(f"    Mean angular sep of MISSED:  {np.mean(missed_ang_seps):.4f}")
        rho_miss, p_miss = spearmanr(
            [p["ang_sep"] for p in reachable_pairs],
            [0 if p["t"] in {p["s"]} else 1 for p in reachable_pairs]  # placeholder
        )

    # Stratify by graph distance
    print(f"\n    Stratified by graph distance:")
    print(f"    {'d_G':>5s} {'N':>5s} {'Found':>6s} {'Missed':>7s} {'Miss%':>7s} {'Ang(miss)':>10s}")
    for d in sorted(set(p["g_dist"] for p in reachable_pairs)):
        if d > H: continue
        subset = [p for p in reachable_pairs if p["g_dist"] == d]
        n = len(subset)
        # Recompute for this subset
        f_count = 0
        m_angs = []
        for p in subset:
            tv = embeddings[p["t"]]
            visited = {p["s"]}
            pq = [(-float(embeddings[v] @ tv), v) for v in adj[p["s"]] if v not in visited]
            heapq.heapify(pq)
            found = False
            while pq and len(visited) < H:
                _, u = heapq.heappop(pq)
                if u in visited: continue
                visited.add(u)
                if u == p["t"]:
                    found = True
                    break
                for v in adj[u]:
                    if v not in visited:
                        heapq.heappush(pq, (-float(embeddings[v] @ tv), v))
            if found:
                f_count += 1
            else:
                m_angs.append(p["ang_sep"])
        miss = n - f_count
        miss_pct = miss / n * 100 if n > 0 else 0
        m_ang_str = f"{np.mean(m_angs):.4f}" if m_angs else "N/A"
        print(f"    {d:5d} {n:5d} {f_count:6d} {miss:7d} {miss_pct:6.1f}% {m_ang_str:>10s}")


# ══════════════════════════════════════════════════════════════════
# PROBE C: PRSM waypoint injection vs single-direction
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PROBE C: Does injecting ALL waypoints beat dual-frontier?")
print(f"{'=' * 70}")

hypothesis_traces = {
    "#1 Cobenfy": ["xanomeline", "muscarinic", "dopamine", "incentive salience", "psychosis"],
    "#5 PTSD": ["ptsd", "norepinephrine", "locus coeruleus", "amygdala", "fear extinction"],
    "#25 PP2A": ["tauopathy", "neuronal loss", "neurofibrillary tangles", "pp2a", "nr2a"],
    "#29 Eden": ["tryptamine", "aryl hydrocarbon receptor", "slc7a11", "cystine", "glutathione", "gpx4", "ferroptosis"],
    "#35 Faecal": ["faecalibacterium", "fibrillization", "enteric nerve", "alpha-synuclein", "dopaminergic neuron"],
    "#38 TGFBR1": ["tgfbr1", "alpha-sma", "tgf-beta", "smad3", "transferrin receptor"],
    "#40 Fasting": ["fasting", "ncoa4", "ferritinophagy", "ferritin", "ferroportin", "gpx4", "ferroptosis"],
}

# Resolve
resolved = {}
for name, concepts in hypothesis_traces.items():
    indices = [concept_to_idx[c.lower()] for c in concepts if c.lower() in concept_to_idx]
    if len(indices) >= 3:
        resolved[name] = indices

H = 100

print(f"\n  {'Hypothesis':>15s} {'N':>3s} {'Cos(A→Z)':>10s} {'Multi(A↔Z)':>12s} {'All-WP':>8s}")
print(f"  {'-'*15} {'-'*3} {'-'*10} {'-'*12} {'-'*8}")

cos_totals, multi_totals, wp_totals = [], [], []

for name, indices in resolved.items():
    source = indices[0]
    target = indices[-1]
    intermediates = set(indices[1:-1])
    n_int = len(intermediates)
    if n_int == 0: continue

    # Strategy 1: Cosine single-source A→Z
    tv = embeddings[target]
    visited_cos = {source}
    pq = [(-float(embeddings[v] @ tv), v) for v in adj[source] if v not in visited_cos]
    heapq.heapify(pq)
    while pq and len(visited_cos) < H:
        _, u = heapq.heappop(pq)
        if u in visited_cos: continue
        visited_cos.add(u)
        for v in adj[u]:
            if v not in visited_cos:
                heapq.heappush(pq, (-float(embeddings[v] @ tv), v))
    cos_int = sum(1 for idx in intermediates if idx in visited_cos)

    # Strategy 2: Multi-anchor A↔Z
    hs = H // 2; ht = H - hs
    vs = {source}; vt = {target}
    ps = [(-float(embeddings[v] @ embeddings[target]), v) for v in adj[source]]
    pt = [(-float(embeddings[v] @ embeddings[source]), v) for v in adj[target]]
    heapq.heapify(ps); heapq.heapify(pt)
    while ps and len(vs) < hs:
        _, u = heapq.heappop(ps)
        if u in vs: continue
        vs.add(u)
        for v in adj[u]:
            if v not in vs: heapq.heappush(ps, (-float(embeddings[v] @ embeddings[target]), v))
    while pt and len(vt) < ht:
        _, u = heapq.heappop(pt)
        if u in vt: continue
        vt.add(u)
        for v in adj[u]:
            if v not in vt: heapq.heappush(pt, (-float(embeddings[v] @ embeddings[source]), v))
    multi_visited = vs | vt
    multi_int = sum(1 for idx in intermediates if idx in multi_visited)

    # Strategy 3: PRSM-style — inject ALL known waypoints, expand from each
    budget_per_wp = H // len(indices)
    wp_visited = set()
    for wp_idx in indices:
        visited_wp = {wp_idx}
        # BFS from this waypoint (small budget per waypoint)
        frontier = {wp_idx}
        for _ in range(3):  # 3 hops from each waypoint
            nf = set()
            for n in frontier:
                for nb in adj[n]:
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

    print(f"  {name:>15s} {n_int:3d} {cos_int:8d}/{n_int} {multi_int:10d}/{n_int} {wp_int:6d}/{n_int}")
    cos_totals.append(cos_int / n_int * 100)
    multi_totals.append(multi_int / n_int * 100)
    wp_totals.append(wp_int / n_int * 100)

print(f"\n  MEAN INTERMEDIATE DISCOVERY RATE:")
print(f"    Cosine A→Z:     {np.mean(cos_totals):.1f}%")
print(f"    Multi A↔Z:      {np.mean(multi_totals):.1f}%")
print(f"    All-Waypoint:   {np.mean(wp_totals):.1f}%")

if np.mean(wp_totals) > np.mean(multi_totals):
    print(f"  → WAYPOINT INJECTION OUTPERFORMS DUAL-FRONTIER on intermediate discovery")
    print(f"  → The path IS the answer: placing all known coordinates finds what's between them")
else:
    print(f"  → Multi-anchor matches or beats waypoint injection")

results["probe_c"] = {
    "cos_mean": float(np.mean(cos_totals)),
    "multi_mean": float(np.mean(multi_totals)),
    "wp_mean": float(np.mean(wp_totals)),
}


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("OPEN PROBLEMS PROBE SUMMARY")
print(f"{'=' * 70}")

print(f"\n  A. Prediction improvement: best LOO MAE = {results['probe_a']['best_loo_mae']:.1f} pp (was 10.6)")
print(f"  B. Policy trapping inside graph ball: see stratified tables above")
print(f"  C. Downstream: Cos={results['probe_c']['cos_mean']:.1f}%, Multi={results['probe_c']['multi_mean']:.1f}%, "
      f"All-WP={results['probe_c']['wp_mean']:.1f}%")

with open(OUT / "exp19_open_problems.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT / 'exp19_open_problems.json'}")
print("OPEN PROBLEMS PROBE COMPLETE")
