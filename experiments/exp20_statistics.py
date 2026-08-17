"""
Experiment 20: Statistical Hygiene on Probe C (Waypoint Injection)

Verifies the headline numbers from exp19 Probe C:
  - Cosine single-source: 13.3%
  - Multi-anchor: 13.3%
  - All-waypoint injection: 100.0%

Adds:
  1. Budget accounting (total nodes visited per strategy)
  2. Bootstrap 95% CIs (1000 replicates, resampling 7 hypotheses)
  3. Leave-one-hypothesis-out stability analysis
  4. McNemar's exact test (discordant pair analysis)
"""
import sys
import time
import json
import heapq
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy.stats import binomtest  # McNemar's exact test uses binomial

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

ROOT = Path("E:/PRSM")
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("Loading PRSM Crystal...", end=" ", flush=True)
t0 = time.time()

with open(ROOT / "data/g1_registry/unified_grains_with_embeddings.json", "r", encoding="utf-8") as f:
    grains_raw = json.load(f)
with open(ROOT / "data/g1_registry/relationship_registry.json", "r", encoding="utf-8") as f:
    rels = json.load(f)

# Build concept index (deduplicate by bind_count)
concept_to_idx = {}
idx_to_concept = {}
embeddings_list = []
seen = {}
for g in grains_raw:
    c = g.get("concept", "").lower()
    emb = g.get("embedding")
    bc = g.get("bind_count", 0)
    if c and emb and (c not in seen or bc > seen[c]):
        seen[c] = bc
        if c not in concept_to_idx:
            idx = len(concept_to_idx)
            concept_to_idx[c] = idx
            idx_to_concept[idx] = c
            embeddings_list.append(np.array(emb, dtype=np.float32))
        else:
            idx = concept_to_idx[c]
            embeddings_list[idx] = np.array(emb, dtype=np.float32)

embeddings = np.array(embeddings_list)
N = len(embeddings)

# L2-normalize embeddings
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / np.clip(norms, 1e-8, None)

# Build adjacency lists (deduplicated edges)
adj = [set() for _ in range(N)]
gid_to_concept = {}
for g in grains_raw:
    gid = g.get("grain_id", "")
    c = g.get("concept", "").lower()
    if gid and c and c in concept_to_idx:
        gid_to_concept[gid] = c

edge_count = 0
for r in rels:
    a_c = gid_to_concept.get(r.get("grain_a_id", ""))
    b_c = gid_to_concept.get(r.get("grain_b_id", ""))
    if a_c and b_c and a_c != b_c:
        a = concept_to_idx.get(a_c)
        b = concept_to_idx.get(b_c)
        if a is not None and b is not None:
            if b not in adj[a]:
                edge_count += 1
            adj[a].add(b)
            adj[b].add(a)

# Convert to lists for indexing
adj = [list(s) for s in adj]

degrees = np.array([len(adj[i]) for i in range(N)])
print(f"{N:,} grains, {edge_count:,} undirected edges, {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════
# DEFINE HYPOTHESIS TRACES
# ══════════════════════════════════════════════════════════════════
hypothesis_traces = {
    "#1 Cobenfy": ["xanomeline", "muscarinic", "dopamine", "incentive salience", "psychosis"],
    "#5 PTSD": ["ptsd", "norepinephrine", "locus coeruleus", "amygdala", "fear extinction"],
    "#25 PP2A": ["tauopathy", "neuronal loss", "neurofibrillary tangles", "pp2a", "nr2a"],
    "#29 Eden": ["tryptamine", "aryl hydrocarbon receptor", "slc7a11", "cystine", "glutathione", "gpx4", "ferroptosis"],
    "#35 Faecal": ["faecalibacterium", "fibrillization", "enteric nerve", "alpha-synuclein", "dopaminergic neuron"],
    "#38 TGFBR1": ["tgfbr1", "alpha-sma", "tgf-beta", "smad3", "transferrin receptor"],
    "#40 Fasting": ["fasting", "ncoa4", "ferritinophagy", "ferritin", "ferroptosis", "gpx4", "ferroptosis"],
}

# Resolve concepts to indices (skip missing)
resolved = {}
for name, concepts in hypothesis_traces.items():
    indices = []
    for c in concepts:
        idx = concept_to_idx.get(c.lower())
        if idx is not None:
            indices.append(idx)
    if len(indices) >= 3:
        resolved[name] = indices
    else:
        print(f"  WARNING: {name} only resolved {len(indices)} concepts, skipping")

print(f"\n  Resolved {len(resolved)}/{len(hypothesis_traces)} hypotheses")
for name, indices in resolved.items():
    concepts_resolved = [idx_to_concept[i] for i in indices]
    intermediates = set(indices[1:-1])
    print(f"    {name}: {len(indices)} waypoints, {len(intermediates)} intermediates")
    print(f"      concepts: {concepts_resolved}")


# ══════════════════════════════════════════════════════════════════
# THREE STRATEGIES
# ══════════════════════════════════════════════════════════════════
H = 100

def strategy_cosine_single(source, target, intermediates, budget):
    """Cosine single-source: A->Z, greedy priority queue toward target."""
    tv = embeddings[target]
    visited = {source}
    pq = []
    for v in adj[source]:
        if v not in visited:
            heapq.heappush(pq, (-float(embeddings[v] @ tv), v))
    while pq and len(visited) < budget:
        _, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for v in adj[u]:
            if v not in visited:
                heapq.heappush(pq, (-float(embeddings[v] @ tv), v))
    found = sum(1 for idx in intermediates if idx in visited)
    return found, len(visited)


def strategy_multi_anchor(source, target, intermediates, budget):
    """Multi-anchor: A<->Z, 50:50 budget split, cosine from both ends."""
    hs = budget // 2
    ht = budget - hs

    vs = {source}
    ps = []
    for v in adj[source]:
        if v not in vs:
            heapq.heappush(ps, (-float(embeddings[v] @ embeddings[target]), v))
    while ps and len(vs) < hs:
        _, u = heapq.heappop(ps)
        if u in vs:
            continue
        vs.add(u)
        for v in adj[u]:
            if v not in vs:
                heapq.heappush(ps, (-float(embeddings[v] @ embeddings[target]), v))

    vt = {target}
    pt = []
    for v in adj[target]:
        if v not in vt:
            heapq.heappush(pt, (-float(embeddings[v] @ embeddings[source]), v))
    while pt and len(vt) < ht:
        _, u = heapq.heappop(pt)
        if u in vt:
            continue
        vt.add(u)
        for v in adj[u]:
            if v not in vt:
                heapq.heappush(pt, (-float(embeddings[v] @ embeddings[source]), v))

    combined = vs | vt
    found = sum(1 for idx in intermediates if idx in combined)
    return found, len(combined)


def strategy_waypoint_injection(indices, intermediates, budget):
    """All-waypoint injection: expand 3 BFS hops from EACH waypoint."""
    n_waypoints = len(indices)
    budget_per_wp = budget // n_waypoints
    wp_visited = set()
    for wp_idx in indices:
        local_visited = {wp_idx}
        frontier = {wp_idx}
        for _ in range(3):  # 3 hops
            nf = set()
            for n in frontier:
                for nb in adj[n]:
                    if nb not in local_visited:
                        local_visited.add(nb)
                        nf.add(nb)
                        if len(local_visited) >= budget_per_wp:
                            break
                if len(local_visited) >= budget_per_wp:
                    break
            frontier = nf
        wp_visited.update(local_visited)
    found = sum(1 for idx in intermediates if idx in wp_visited)
    return found, len(wp_visited)


# ══════════════════════════════════════════════════════════════════
# RUN ALL THREE STRATEGIES
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print(f"PROBE C VERIFICATION: Three strategies, H={H}")
print(f"{'=' * 80}")

header = (f"  {'Hypothesis':>15s} {'N_int':>5s} | "
          f"{'Cos':>5s} {'Cos_N':>6s} | "
          f"{'Multi':>5s} {'Mlt_N':>6s} | "
          f"{'WP':>5s} {'WP_N':>6s}")
print(header)
print(f"  {'-'*15} {'-'*5} + {'-'*5} {'-'*6} + {'-'*5} {'-'*6} + {'-'*5} {'-'*6}")

# Store per-hypothesis results
per_hyp = {}

for name, indices in resolved.items():
    source = indices[0]
    target = indices[-1]
    intermediates = set(indices[1:-1])
    n_int = len(intermediates)
    if n_int == 0:
        continue

    cos_found, cos_visited = strategy_cosine_single(source, target, intermediates, H)
    multi_found, multi_visited = strategy_multi_anchor(source, target, intermediates, H)
    wp_found, wp_visited = strategy_waypoint_injection(indices, intermediates, H)

    cos_rate = cos_found / n_int * 100
    multi_rate = multi_found / n_int * 100
    wp_rate = wp_found / n_int * 100

    per_hyp[name] = {
        "n_intermediates": n_int,
        "n_waypoints": len(indices),
        "cos_found": cos_found,
        "cos_visited": cos_visited,
        "cos_rate": cos_rate,
        "multi_found": multi_found,
        "multi_visited": multi_visited,
        "multi_rate": multi_rate,
        "wp_found": wp_found,
        "wp_visited": wp_visited,
        "wp_rate": wp_rate,
    }

    print(f"  {name:>15s} {n_int:5d} | "
          f"{cos_found:3d}/{n_int:<1d} {cos_visited:6d} | "
          f"{multi_found:3d}/{n_int:<1d} {multi_visited:6d} | "
          f"{wp_found:3d}/{n_int:<1d} {wp_visited:6d}")

# Summary
names = list(per_hyp.keys())
cos_rates = [per_hyp[n]["cos_rate"] for n in names]
multi_rates = [per_hyp[n]["multi_rate"] for n in names]
wp_rates = [per_hyp[n]["wp_rate"] for n in names]

cos_visits = [per_hyp[n]["cos_visited"] for n in names]
multi_visits = [per_hyp[n]["multi_visited"] for n in names]
wp_visits = [per_hyp[n]["wp_visited"] for n in names]

print(f"\n  MEAN INTERMEDIATE DISCOVERY RATE:")
print(f"    Cosine A->Z:     {np.mean(cos_rates):6.1f}%  (mean nodes visited: {np.mean(cos_visits):.0f})")
print(f"    Multi A<->Z:     {np.mean(multi_rates):6.1f}%  (mean nodes visited: {np.mean(multi_visits):.0f})")
print(f"    All-Waypoint:    {np.mean(wp_rates):6.1f}%  (mean nodes visited: {np.mean(wp_visits):.0f})")


# ══════════════════════════════════════════════════════════════════
# BUDGET ACCOUNTING
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("BUDGET ACCOUNTING: Total nodes visited by each strategy")
print(f"{'=' * 80}")

print(f"\n  {'Hypothesis':>15s} {'Cos_nodes':>10s} {'Multi_nodes':>12s} {'WP_nodes':>10s} {'WP/Cos ratio':>13s}")
print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*10} {'-'*13}")

for name in names:
    h = per_hyp[name]
    ratio = h["wp_visited"] / max(h["cos_visited"], 1)
    print(f"  {name:>15s} {h['cos_visited']:10d} {h['multi_visited']:12d} {h['wp_visited']:10d} {ratio:13.2f}x")

print(f"\n  Total across all hypotheses:")
print(f"    Cosine:     {sum(cos_visits):,d} nodes")
print(f"    Multi:      {sum(multi_visits):,d} nodes")
print(f"    Waypoint:   {sum(wp_visits):,d} nodes")
wp_cos_total_ratio = sum(wp_visits) / max(sum(cos_visits), 1)
print(f"    WP/Cos total ratio: {wp_cos_total_ratio:.2f}x")

if np.mean(wp_visits) > np.mean(cos_visits) * 1.1:
    print(f"\n  *** WARNING: Waypoint injection visits {wp_cos_total_ratio:.1f}x more nodes than cosine. ***")
    print(f"  *** The 100% discovery rate may partly reflect larger search volume. ***")
    print(f"  *** A fair comparison should equalize total budget across strategies. ***")


# ══════════════════════════════════════════════════════════════════
# BOOTSTRAP 95% CIs (1000 replicates)
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("BOOTSTRAP 95% CIs (1000 replicates, resampling 7 hypotheses w/ replacement)")
print(f"{'=' * 80}")

n_boot = 1000
n_hyp = len(names)

cos_boot_means = []
multi_boot_means = []
wp_boot_means = []

for b in range(n_boot):
    sample_idx = np.random.choice(n_hyp, size=n_hyp, replace=True)
    cos_boot_means.append(np.mean([cos_rates[i] for i in sample_idx]))
    multi_boot_means.append(np.mean([multi_rates[i] for i in sample_idx]))
    wp_boot_means.append(np.mean([wp_rates[i] for i in sample_idx]))

cos_boot_means = np.array(cos_boot_means)
multi_boot_means = np.array(multi_boot_means)
wp_boot_means = np.array(wp_boot_means)

def ci95(arr):
    lo = np.percentile(arr, 2.5)
    hi = np.percentile(arr, 97.5)
    return lo, hi

cos_ci = ci95(cos_boot_means)
multi_ci = ci95(multi_boot_means)
wp_ci = ci95(wp_boot_means)

print(f"\n  Strategy            Mean    95% CI")
print(f"  {'─'*50}")
print(f"  Cosine A->Z:     {np.mean(cos_rates):6.1f}%   [{cos_ci[0]:5.1f}%, {cos_ci[1]:5.1f}%]")
print(f"  Multi A<->Z:     {np.mean(multi_rates):6.1f}%   [{multi_ci[0]:5.1f}%, {multi_ci[1]:5.1f}%]")
print(f"  All-Waypoint:    {np.mean(wp_rates):6.1f}%   [{wp_ci[0]:5.1f}%, {wp_ci[1]:5.1f}%]")

# Also bootstrap the difference
diff_boot = wp_boot_means - cos_boot_means
diff_ci = ci95(diff_boot)
print(f"\n  Difference (WP - Cos):")
print(f"    Mean: {np.mean(diff_boot):.1f} pp")
print(f"    95% CI: [{diff_ci[0]:.1f}, {diff_ci[1]:.1f}] pp")
if diff_ci[0] > 0:
    print(f"    CI excludes zero -> SIGNIFICANT at alpha=0.05")
else:
    print(f"    CI includes zero -> NOT significant at alpha=0.05")

# Check for degenerate CI (all same value)
wp_unique = len(np.unique(wp_boot_means))
cos_unique = len(np.unique(cos_boot_means))
print(f"\n  Bootstrap distribution diagnostics:")
print(f"    Cosine boot unique values: {cos_unique}")
print(f"    WP boot unique values:     {wp_unique}")
if wp_unique == 1:
    print(f"    *** Waypoint CI is degenerate (all 100.0%) ***")
    print(f"    *** Every resample yields 100% -- CI is uninformative ***")


# ══════════════════════════════════════════════════════════════════
# LEAVE-ONE-HYPOTHESIS-OUT
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("LEAVE-ONE-HYPOTHESIS-OUT: Is 100% driven by one or two easy cases?")
print(f"{'=' * 80}")

print(f"\n  {'Left-out':>15s} {'Cos_mean':>10s} {'Multi_mean':>12s} {'WP_mean':>10s}")
print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*10}")

loo_results = {}
for i, name in enumerate(names):
    loo_cos = [cos_rates[j] for j in range(n_hyp) if j != i]
    loo_multi = [multi_rates[j] for j in range(n_hyp) if j != i]
    loo_wp = [wp_rates[j] for j in range(n_hyp) if j != i]

    loo_results[name] = {
        "cos_mean_without": float(np.mean(loo_cos)),
        "multi_mean_without": float(np.mean(loo_multi)),
        "wp_mean_without": float(np.mean(loo_wp)),
    }

    print(f"  {name:>15s} {np.mean(loo_cos):9.1f}% {np.mean(loo_multi):11.1f}% {np.mean(loo_wp):9.1f}%")

# Check stability
wp_loo_values = [v["wp_mean_without"] for v in loo_results.values()]
cos_loo_values = [v["cos_mean_without"] for v in loo_results.values()]
print(f"\n  WP LOO range: [{min(wp_loo_values):.1f}%, {max(wp_loo_values):.1f}%]")
print(f"  Cos LOO range: [{min(cos_loo_values):.1f}%, {max(cos_loo_values):.1f}%]")

if min(wp_loo_values) == max(wp_loo_values) == 100.0:
    print(f"  -> WP is 100% regardless of which hypothesis is removed (robust)")
elif min(wp_loo_values) < 100.0:
    # Find which hypothesis, when removed, changes the result
    for name in names:
        if loo_results[name]["wp_mean_without"] < 100.0:
            print(f"  -> Removing {name} drops WP below 100% -> result depends on this hypothesis")


# ══════════════════════════════════════════════════════════════════
# McNEMAR'S EXACT TEST
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("McNEMAR'S TEST: Discordant pair analysis (Cosine vs Waypoint)")
print(f"{'=' * 80}")

# For each (hypothesis, intermediate), record whether each strategy found it
# Then count discordant pairs for McNemar's test
# Concordant: both find or both miss
# Discordant: one finds, the other misses

all_intermediates = []  # list of (hyp_name, intermediate_idx)
cos_found_list = []     # boolean: cosine found this intermediate
wp_found_list = []      # boolean: waypoint found this intermediate
multi_found_list = []   # boolean: multi found this intermediate

for name, indices in resolved.items():
    source = indices[0]
    target = indices[-1]
    intermediates = set(indices[1:-1])
    if not intermediates:
        continue

    # Re-run strategies to get per-intermediate results
    # Cosine
    tv = embeddings[target]
    visited_cos = {source}
    pq = []
    for v in adj[source]:
        if v not in visited_cos:
            heapq.heappush(pq, (-float(embeddings[v] @ tv), v))
    while pq and len(visited_cos) < H:
        _, u = heapq.heappop(pq)
        if u in visited_cos:
            continue
        visited_cos.add(u)
        for v in adj[u]:
            if v not in visited_cos:
                heapq.heappush(pq, (-float(embeddings[v] @ tv), v))

    # Multi
    hs = H // 2; ht = H - hs
    vs = {source}
    ps = []
    for v in adj[source]:
        if v not in vs:
            heapq.heappush(ps, (-float(embeddings[v] @ embeddings[target]), v))
    while ps and len(vs) < hs:
        _, u = heapq.heappop(ps)
        if u in vs:
            continue
        vs.add(u)
        for v in adj[u]:
            if v not in vs:
                heapq.heappush(ps, (-float(embeddings[v] @ embeddings[target]), v))
    vt = {target}
    pt = []
    for v in adj[target]:
        if v not in vt:
            heapq.heappush(pt, (-float(embeddings[v] @ embeddings[source]), v))
    while pt and len(vt) < ht:
        _, u = heapq.heappop(pt)
        if u in vt:
            continue
        vt.add(u)
        for v in adj[u]:
            if v not in vt:
                heapq.heappush(pt, (-float(embeddings[v] @ embeddings[source]), v))
    visited_multi = vs | vt

    # Waypoint
    n_waypoints = len(indices)
    budget_per_wp = H // n_waypoints
    visited_wp = set()
    for wp_idx in indices:
        local_visited = {wp_idx}
        frontier = {wp_idx}
        for _ in range(3):
            nf = set()
            for n in frontier:
                for nb in adj[n]:
                    if nb not in local_visited:
                        local_visited.add(nb)
                        nf.add(nb)
                        if len(local_visited) >= budget_per_wp:
                            break
                if len(local_visited) >= budget_per_wp:
                    break
            frontier = nf
        visited_wp.update(local_visited)

    for inter_idx in intermediates:
        all_intermediates.append((name, idx_to_concept[inter_idx]))
        cos_found_list.append(inter_idx in visited_cos)
        multi_found_list.append(inter_idx in visited_multi)
        wp_found_list.append(inter_idx in visited_wp)

cos_found_arr = np.array(cos_found_list, dtype=bool)
multi_found_arr = np.array(multi_found_list, dtype=bool)
wp_found_arr = np.array(wp_found_list, dtype=bool)

n_total_intermediates = len(all_intermediates)
print(f"\n  Total intermediate waypoints across all hypotheses: {n_total_intermediates}")

# McNemar contingency table: Cosine vs Waypoint
#                    WP found    WP missed
# Cos found:          a            b
# Cos missed:         c            d
a = int(np.sum(cos_found_arr & wp_found_arr))
b = int(np.sum(cos_found_arr & ~wp_found_arr))
c = int(np.sum(~cos_found_arr & wp_found_arr))
d = int(np.sum(~cos_found_arr & ~wp_found_arr))

print(f"\n  McNemar contingency (Cosine vs Waypoint):")
print(f"                    WP found   WP missed")
print(f"    Cos found:      {a:8d}    {b:8d}")
print(f"    Cos missed:     {c:8d}    {d:8d}")
print(f"\n  Concordant: {a+d}, Discordant: {b+c}")
print(f"    b (Cos found, WP missed): {b}")
print(f"    c (WP found, Cos missed): {c}")

# McNemar's exact test (binomial): under H0, b/(b+c) = 0.5
discordant = b + c
if discordant > 0:
    # Two-sided binomial test: is the split b vs c different from 50/50?
    p_value = binomtest(b, discordant, 0.5).pvalue
    print(f"\n  McNemar's exact test (two-sided):")
    print(f"    n_discordant = {discordant}")
    print(f"    b = {b}, c = {c}")
    print(f"    p-value = {p_value:.6f}")
    if p_value < 0.05:
        print(f"    -> SIGNIFICANT at alpha=0.05: strategies differ")
        if c > b:
            print(f"    -> Waypoint finds {c} intermediates that cosine misses")
        else:
            print(f"    -> Cosine finds {b} intermediates that waypoint misses")
    else:
        print(f"    -> NOT significant at alpha=0.05")
else:
    p_value = 1.0
    print(f"\n  No discordant pairs -> McNemar's test not applicable")

# Same for Multi vs Waypoint
a2 = int(np.sum(multi_found_arr & wp_found_arr))
b2 = int(np.sum(multi_found_arr & ~wp_found_arr))
c2 = int(np.sum(~multi_found_arr & wp_found_arr))
d2 = int(np.sum(~multi_found_arr & ~wp_found_arr))

print(f"\n  McNemar contingency (Multi vs Waypoint):")
print(f"                    WP found   WP missed")
print(f"    Multi found:    {a2:8d}    {b2:8d}")
print(f"    Multi missed:   {c2:8d}    {d2:8d}")

disc2 = b2 + c2
if disc2 > 0:
    p_value2 = binomtest(b2, disc2, 0.5).pvalue
    print(f"\n  McNemar's exact test (two-sided):")
    print(f"    n_discordant = {disc2}")
    print(f"    b = {b2}, c = {c2}")
    print(f"    p-value = {p_value2:.6f}")
    if p_value2 < 0.05:
        print(f"    -> SIGNIFICANT at alpha=0.05")
    else:
        print(f"    -> NOT significant at alpha=0.05")
else:
    p_value2 = 1.0
    print(f"\n  No discordant pairs -> McNemar's test not applicable")

# Detail: which intermediates are found/missed by each
print(f"\n  Per-intermediate detail:")
print(f"  {'Hypothesis':>15s} {'Intermediate':>30s} {'Cos':>5s} {'Multi':>5s} {'WP':>5s}")
print(f"  {'-'*15} {'-'*30} {'-'*5} {'-'*5} {'-'*5}")
for i, (hyp_name, inter_concept) in enumerate(all_intermediates):
    c_mark = "Y" if cos_found_list[i] else "."
    m_mark = "Y" if multi_found_list[i] else "."
    w_mark = "Y" if wp_found_list[i] else "."
    print(f"  {hyp_name:>15s} {inter_concept:>30s} {c_mark:>5s} {m_mark:>5s} {w_mark:>5s}")


# ══════════════════════════════════════════════════════════════════
# COMPILE RESULTS
# ══════════════════════════════════════════════════════════════════
results = {
    "experiment": "exp20_statistical_hygiene",
    "description": "Statistical verification of Probe C headline numbers with bootstrap CIs, LOO, and McNemar's test",
    "budget_H": H,
    "n_hypotheses": len(per_hyp),
    "n_total_intermediates": n_total_intermediates,

    "headline_numbers": {
        "cos_mean_rate": float(np.mean(cos_rates)),
        "multi_mean_rate": float(np.mean(multi_rates)),
        "wp_mean_rate": float(np.mean(wp_rates)),
    },

    "budget_accounting": {
        "cos_mean_nodes": float(np.mean(cos_visits)),
        "multi_mean_nodes": float(np.mean(multi_visits)),
        "wp_mean_nodes": float(np.mean(wp_visits)),
        "cos_total_nodes": int(sum(cos_visits)),
        "multi_total_nodes": int(sum(multi_visits)),
        "wp_total_nodes": int(sum(wp_visits)),
        "wp_to_cos_ratio": float(wp_cos_total_ratio),
    },

    "bootstrap_ci_95": {
        "n_replicates": n_boot,
        "cosine": {"mean": float(np.mean(cos_boot_means)), "ci_lo": float(cos_ci[0]), "ci_hi": float(cos_ci[1])},
        "multi": {"mean": float(np.mean(multi_boot_means)), "ci_lo": float(multi_ci[0]), "ci_hi": float(multi_ci[1])},
        "waypoint": {"mean": float(np.mean(wp_boot_means)), "ci_lo": float(wp_ci[0]), "ci_hi": float(wp_ci[1])},
        "difference_wp_minus_cos": {
            "mean": float(np.mean(diff_boot)),
            "ci_lo": float(diff_ci[0]),
            "ci_hi": float(diff_ci[1]),
            "excludes_zero": bool(diff_ci[0] > 0),
        },
        "wp_ci_degenerate": bool(wp_unique == 1),
    },

    "leave_one_out": loo_results,

    "mcnemar_cos_vs_wp": {
        "concordant": a + d,
        "discordant_b_cos_found_wp_missed": b,
        "discordant_c_wp_found_cos_missed": c,
        "total_discordant": discordant,
        "p_value": float(p_value),
        "significant_alpha_05": bool(p_value < 0.05),
    },

    "mcnemar_multi_vs_wp": {
        "concordant": a2 + d2,
        "discordant_b_multi_found_wp_missed": b2,
        "discordant_c_wp_found_multi_missed": c2,
        "total_discordant": disc2,
        "p_value": float(p_value2),
        "significant_alpha_05": bool(p_value2 < 0.05),
    },

    "per_hypothesis": per_hyp,

    "per_intermediate_detail": [
        {
            "hypothesis": hyp_name,
            "intermediate": inter_concept,
            "cos_found": bool(cos_found_list[i]),
            "multi_found": bool(multi_found_list[i]),
            "wp_found": bool(wp_found_list[i]),
        }
        for i, (hyp_name, inter_concept) in enumerate(all_intermediates)
    ],
}

with open(OUT / "exp20_statistical_hygiene.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 80}")
print("EXPERIMENT 20 COMPLETE")
print(f"{'=' * 80}")
print(f"Saved: {OUT / 'exp20_statistical_hygiene.json'}")
