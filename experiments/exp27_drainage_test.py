"""
Experiment 27: The Specificity Trap — Formalizing the Micro-Mechanism

Hypothesis: In high-dimensional embedded graphs, cosine ranking preferentially
selects angularly SPECIFIC (sharp-direction) nodes over angularly DIFFUSE
(smeared-direction) nodes. Specific nodes have lower degree. Lower degree
means fewer onward edges. Fewer edges means a narrower candidate cone at
the next step. The search progressively narrows into increasingly specialized,
increasingly isolated nodes until budget is exhausted.

This is not a defect of cosine. It is cosine doing exactly what it does:
finding the most angularly aligned vector. The consequence is that angular
alignment and structural connectivity are inversely related in high-D
embedded graphs.

The Specificity Trap:
  sharp direction → high cosine to specific targets → selected by cosine
  sharp direction → few training contexts → low degree
  low degree → few onward edges → narrow next-step cone
  narrow cone → even sharper next selection → even lower degree
  → progressive budget exhaustion in increasingly isolated periphery

Tests:
  T1: Angular specificity (1 - mean_cos_to_all) inversely correlates with degree
  T2: At each expansion step, the SELECTED node has higher specificity than
      the mean of available candidates
  T3: The degree of selected nodes DECREASES over successive expansion steps
      (progressive narrowing)
  T4: The angular cone width of candidates DECREASES over steps
  T5: This pattern holds on multiple graphs (NeuroCrystal, Cora, Amazon, DBLP,
      Synthetic Isotropic)

If all 5 hold: the Specificity Trap is confirmed as the micro-mechanism.
If T1 holds but T3 doesn't: specificity exists but doesn't cause progressive narrowing.
If T1 fails: the mechanism is something else entirely.
"""
import sys
import time
import json
import heapq
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(__file__).resolve().parent / "results"

print(f"Device: {DEVICE}")


def load_neurocrystal():
    ROOT = Path("E:/PRSM")
    with open(ROOT / "data/g1_registry/unified_grains_with_embeddings.json", "r", encoding="utf-8") as f:
        grains = json.load(f)
    with open(ROOT / "data/g1_registry/relationship_registry.json", "r", encoding="utf-8") as f:
        rels = json.load(f)

    c2i = {}; embs = []; seen = {}; corpora = {}
    for g in grains:
        c = g.get("concept", "").lower()
        emb = g.get("embedding")
        bc = g.get("bind_count", 0)
        sc = g.get("source_corpora", [])
        if c and emb and (c not in seen or bc > seen[c]):
            seen[c] = bc
            if c not in c2i:
                c2i[c] = len(c2i)
                embs.append(np.array(emb, dtype=np.float32))
                corpora[c2i[c]] = set(sc) if isinstance(sc, list) else set()
            else:
                embs[c2i[c]] = np.array(emb, dtype=np.float32)
                corpora[c2i[c]] = set(sc) if isinstance(sc, list) else set()

    features = np.array(embs)
    N = len(features)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.clip(norms, 1e-8, None)

    adj = [[] for _ in range(N)]
    g2c = {}
    for g in grains:
        gid = g.get("grain_id", "")
        c = g.get("concept", "").lower()
        if gid and c and c in c2i: g2c[gid] = c
    for r in rels:
        ac = g2c.get(r.get("grain_a_id", ""))
        bc = g2c.get(r.get("grain_b_id", ""))
        if ac and bc and ac != bc:
            a, b = c2i.get(ac), c2i.get(bc)
            if a is not None and b is not None:
                adj[a].append(b); adj[b].append(a)
    for i in range(N): adj[i] = list(set(adj[i]))

    cn = defaultdict(list)
    for idx, corp in corpora.items():
        for c in corp: cn[c].append(idx)
    vc = [c for c, nodes in cn.items() if len(nodes) >= 20]

    return "NeuroCrystal", features, adj, vc, cn


def load_pyg_graph(graph_name, cls, **kwargs):
    d = cls(**kwargs); data = d[0]
    name = graph_name
    features = data.x.numpy().astype(np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.clip(norms, 1e-8, None)
    N = data.num_nodes
    adj = [[] for _ in range(N)]
    for s, d_idx in zip(data.edge_index[0].numpy(), data.edge_index[1].numpy()):
        adj[int(s)].append(int(d_idx))
    for i in range(N): adj[i] = list(set(adj[i]))
    labels = data.y.numpy()
    ul = np.unique(labels)
    cn = {str(l): [i for i in range(N) if labels[i] == l] for l in ul}
    vc = [l for l, nodes in cn.items() if len(nodes) >= 20]
    return name, features, adj, vc, cn


def load_synthetic():
    N, D = 10000, 768
    features = np.random.randn(N, D).astype(np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.clip(norms, 1e-8, None)
    # kNN adjacency
    t = torch.tensor(features, device=DEVICE)
    adj = [[] for _ in range(N)]
    for s in range(0, N, 512):
        e = min(s + 512, N)
        dd = torch.cdist(t[s:e], t)
        for i in range(e - s): dd[i, s + i] = float('inf')
        _, tk = dd.topk(20, dim=1, largest=False)
        for i in range(e - s):
            for j in tk[i].cpu().numpy():
                adj[s+i].append(int(j)); adj[int(j)].append(s+i)
    for i in range(N): adj[i] = list(set(adj[i]))
    # Pseudo-classes by quadrant
    cn = {"0": [], "1": [], "2": [], "3": []}
    for i in range(N):
        q = (1 if features[i, 0] > 0 else 0) + (2 if features[i, 1] > 0 else 0)
        cn[str(q)].append(i)
    vc = list(cn.keys())
    return "Synthetic Isotropic", features, adj, vc, cn


def run_specificity_tests(name, features, adj, valid_groups, group_nodes):
    N = len(features)
    degrees = np.array([len(adj[i]) for i in range(N)])
    t_emb = torch.tensor(features, device=DEVICE)

    print(f"\n{'=' * 70}")
    print(f"GRAPH: {name} ({N:,} nodes)")
    print(f"{'=' * 70}")

    result = {"name": name, "N": N}

    # ── T1: Specificity vs degree ──
    print("\n  T1: Angular specificity vs degree...")
    # Specificity = 1 - mean_cosine_to_sample (how sharply pointed is this node?)
    sample = np.random.choice(N, min(3000, N), replace=False)
    t_sample = t_emb[sample]
    mean_cos = (t_sample @ t_emb.T).mean(dim=1).cpu().numpy()  # mean cos to all
    specificity_sample = 1 - mean_cos  # high = sharp direction

    rho_spec_deg, p_spec_deg = spearmanr(degrees[sample], specificity_sample)
    print(f"    Spearman(degree, specificity): rho = {rho_spec_deg:+.4f} (p = {p_spec_deg:.2e})")

    if rho_spec_deg < -0.05:
        print(f"    → CONFIRMED: High-degree nodes have LOW specificity (diffuse directions)")
    elif rho_spec_deg > 0.05:
        print(f"    → REVERSED: High-degree nodes have HIGH specificity")
    else:
        print(f"    → NO RELATIONSHIP between degree and specificity")

    result["T1_rho"] = float(rho_spec_deg)
    result["T1_p"] = float(p_spec_deg)

    # Bin by degree
    print(f"    {'Degree':>10s} {'N':>5s} {'Mean Specificity':>18s} {'Mean Cos-to-All':>16s}")
    for lo, hi in [(0, 5), (5, 20), (20, 50), (50, 200), (200, 10000)]:
        mask = (degrees[sample] >= lo) & (degrees[sample] < hi)
        n = np.sum(mask)
        if n > 0:
            ms = np.mean(specificity_sample[mask])
            mc = np.mean(mean_cos[mask])
            print(f"    {f'{lo}-{hi}':>10s} {n:5d} {ms:18.6f} {mc:16.6f}")

    # ── T2: Selected node specificity vs candidate mean ──
    print("\n  T2: Does cosine SELECT more specific nodes than available?")

    n_queries = 100
    all_groups = list(group_nodes.keys())
    selected_specs = []
    candidate_specs = []
    selected_degs = []
    candidate_degs = []

    for _ in range(n_queries):
        g1, g2 = np.random.choice(all_groups, 2, replace=False)
        nodes1 = group_nodes[g1]
        nodes2 = group_nodes[g2]
        if not nodes1 or not nodes2: continue
        source = nodes1[np.random.randint(len(nodes1))]
        target = nodes2[np.random.randint(len(nodes2))]
        if not adj[source]: continue

        tv = features[target]
        candidates = adj[source]
        cos_scores = [float(features[c] @ tv) for c in candidates]
        best_idx = np.argmax(cos_scores)
        selected = candidates[best_idx]

        # Specificity of selected vs mean of candidates
        sel_spec = 1 - float(t_emb[selected] @ t_emb.mean(dim=0) /
                              (t_emb[selected].norm() * t_emb.mean(dim=0).norm() + 1e-8))
        cand_specs_local = []
        for c in candidates:
            cs = 1 - float(t_emb[c] @ t_emb.mean(dim=0) /
                           (t_emb[c].norm() * t_emb.mean(dim=0).norm() + 1e-8))
            cand_specs_local.append(cs)

        selected_specs.append(sel_spec)
        candidate_specs.append(np.mean(cand_specs_local))
        selected_degs.append(degrees[selected])
        candidate_degs.append(np.mean([degrees[c] for c in candidates]))

    mean_sel_spec = np.mean(selected_specs)
    mean_cand_spec = np.mean(candidate_specs)
    mean_sel_deg = np.mean(selected_degs)
    mean_cand_deg = np.mean(candidate_degs)

    print(f"    Selected node specificity:   {mean_sel_spec:.6f}")
    print(f"    Mean candidate specificity:  {mean_cand_spec:.6f}")
    print(f"    Selected node degree:        {mean_sel_deg:.1f}")
    print(f"    Mean candidate degree:       {mean_cand_deg:.1f}")

    if mean_sel_spec > mean_cand_spec:
        print(f"    → CONFIRMED: Cosine selects MORE specific nodes than average candidate")
    else:
        print(f"    → NOT CONFIRMED: Selected nodes are not more specific")

    result["T2_sel_spec"] = float(mean_sel_spec)
    result["T2_cand_spec"] = float(mean_cand_spec)
    result["T2_sel_deg"] = float(mean_sel_deg)
    result["T2_cand_deg"] = float(mean_cand_deg)

    # ── T3: Progressive degree decrease over expansion steps ──
    print("\n  T3: Does selected-node degree DECREASE over expansion steps?")

    step_degrees = defaultdict(list)
    step_specificities = defaultdict(list)
    step_cone_widths = defaultdict(list)

    n_traces = 100
    budget = 50

    for _ in range(n_traces):
        g1, g2 = np.random.choice(all_groups, 2, replace=False)
        nodes1 = group_nodes[g1]
        nodes2 = group_nodes[g2]
        if not nodes1 or not nodes2: continue
        source = nodes1[np.random.randint(len(nodes1))]
        target = nodes2[np.random.randint(len(nodes2))]
        if not adj[source]: continue

        tv = features[target]
        visited = {source}
        pq = []
        for v in adj[source]:
            if v not in visited:
                heapq.heappush(pq, (-float(features[v] @ tv), v))

        step = 0
        while pq and len(visited) < budget:
            _, u = heapq.heappop(pq)
            if u in visited: continue
            visited.add(u)

            step_degrees[step].append(degrees[u])

            # Specificity of selected node
            u_spec = 1 - float(t_emb[u] @ t_emb.mean(dim=0) /
                               (t_emb[u].norm() * t_emb.mean(dim=0).norm() + 1e-8))
            step_specificities[step].append(u_spec)

            # Cone width: variance of cosine-to-target among candidates
            candidates = [v for v in adj[u] if v not in visited]
            if candidates:
                cos_vals = [float(features[v] @ tv) for v in candidates]
                step_cone_widths[step].append(np.std(cos_vals))

            for v in adj[u]:
                if v not in visited:
                    heapq.heappush(pq, (-float(features[v] @ tv), v))
            step += 1

    print(f"    {'Step':>6s} {'Mean Degree':>12s} {'Mean Specificity':>17s} {'Cone Width':>11s}")
    steps_to_show = sorted(step_degrees.keys())[:15]
    step_deg_means = []
    for s in steps_to_show:
        md = np.mean(step_degrees[s])
        ms = np.mean(step_specificities[s]) if s in step_specificities else 0
        cw = np.mean(step_cone_widths[s]) if s in step_cone_widths else 0
        step_deg_means.append((s, md))
        print(f"    {s:6d} {md:12.1f} {ms:17.6f} {cw:11.6f}")

    # Correlation of step number with degree
    if len(steps_to_show) > 3:
        step_nums = [s for s, _ in step_deg_means]
        deg_means = [d for _, d in step_deg_means]
        rho_step_deg, p_step_deg = spearmanr(step_nums, deg_means)
        print(f"\n    Spearman(step, degree): rho = {rho_step_deg:+.4f} (p = {p_step_deg:.2e})")
        if rho_step_deg < -0.3:
            print(f"    → CONFIRMED: Degree DECREASES over expansion steps (progressive narrowing)")
        elif rho_step_deg > 0.3:
            print(f"    → REVERSED: Degree INCREASES (search finds hubs over time)")
        else:
            print(f"    → NO CLEAR TREND in degree over steps")
        result["T3_rho"] = float(rho_step_deg)

    # ── T4: Cone width decrease ──
    print("\n  T4: Does angular cone width DECREASE over steps?")
    if step_cone_widths:
        cw_steps = sorted(step_cone_widths.keys())[:15]
        cw_means = [np.mean(step_cone_widths[s]) for s in cw_steps]
        if len(cw_steps) > 3:
            rho_cw, p_cw = spearmanr(cw_steps, cw_means)
            print(f"    Spearman(step, cone_width): rho = {rho_cw:+.4f} (p = {p_cw:.2e})")
            if rho_cw < -0.3:
                print(f"    → CONFIRMED: Cone NARROWS over steps (progressive angular collapse)")
            else:
                print(f"    → Cone does not narrow ({rho_cw:+.4f})")
            result["T4_rho"] = float(rho_cw)

    # ── T5: FORCED SPECIALIST AVOIDANCE (causal test) ──
    print("\n  T5: Does masking top-k specialists improve reachability?")

    n_pairs = 150
    budget_test = 100
    cos_reached = 0
    masked_reached = 0
    cos_mean_deg = []
    masked_mean_deg = []

    for _ in range(n_pairs):
        g1, g2 = np.random.choice(all_groups, 2, replace=False)
        nodes1 = group_nodes[g1]
        nodes2 = group_nodes[g2]
        if not nodes1 or not nodes2: continue
        source = nodes1[np.random.randint(len(nodes1))]
        target = nodes2[np.random.randint(len(nodes2))]
        if not adj[source]: continue

        tv = features[target]

        # Standard cosine
        visited = {source}
        pq = [(-float(features[v] @ tv), v) for v in adj[source] if v not in visited]
        heapq.heapify(pq)
        found_cos = False
        degs_cos = []
        while pq and len(visited) < budget_test:
            _, u = heapq.heappop(pq)
            if u in visited: continue
            visited.add(u)
            degs_cos.append(degrees[u])
            if u == target: found_cos = True; break
            for v in adj[u]:
                if v not in visited:
                    heapq.heappush(pq, (-float(features[v] @ tv), v))
        if found_cos: cos_reached += 1
        if degs_cos: cos_mean_deg.append(np.mean(degs_cos))

        # Specialist-masked: at each step, remove top-50% cosine candidates,
        # force selection from the LOWER-cosine half (which should be higher-degree)
        visited2 = {source}
        pq2 = []
        cands = [(float(features[v] @ tv), v) for v in adj[source] if v not in visited2]
        if cands:
            cands.sort(reverse=True)
            # Keep bottom half (lower cosine = likely higher degree)
            keep = cands[len(cands)//2:]
            for score, v in keep:
                heapq.heappush(pq2, (-score, v))
        found_masked = False
        degs_masked = []
        while pq2 and len(visited2) < budget_test:
            _, u = heapq.heappop(pq2)
            if u in visited2: continue
            visited2.add(u)
            degs_masked.append(degrees[u])
            if u == target: found_masked = True; break
            cands = [(float(features[v] @ tv), v) for v in adj[u] if v not in visited2]
            if cands:
                cands.sort(reverse=True)
                keep = cands[len(cands)//2:]
                for score, v in keep:
                    heapq.heappush(pq2, (-score, v))
        if found_masked: masked_reached += 1
        if degs_masked: masked_mean_deg.append(np.mean(degs_masked))

    cos_rate = cos_reached / n_pairs * 100
    masked_rate = masked_reached / n_pairs * 100
    print(f"    Standard cosine reachability:  {cos_rate:.1f}% (mean deg {np.mean(cos_mean_deg):.1f})")
    print(f"    Specialist-masked reachability: {masked_rate:.1f}% (mean deg {np.mean(masked_mean_deg):.1f})")
    if masked_rate > cos_rate:
        print(f"    → CONFIRMED: Avoiding specialists IMPROVES reachability (+{masked_rate-cos_rate:.1f}pp)")
        print(f"    → CAUSAL EVIDENCE: the specificity trap is the mechanism")
    elif masked_rate < cos_rate:
        print(f"    → FALSIFIED: Avoiding specialists HURTS reachability ({masked_rate-cos_rate:+.1f}pp)")
    else:
        print(f"    → INCONCLUSIVE: No difference")

    result["T5_cos_rate"] = float(cos_rate)
    result["T5_masked_rate"] = float(masked_rate)
    result["T5_cos_deg"] = float(np.mean(cos_mean_deg)) if cos_mean_deg else 0
    result["T5_masked_deg"] = float(np.mean(masked_mean_deg)) if masked_mean_deg else 0

    return result


# ══════════════════════════════════════════════════════════════════
# RUN ON ALL GRAPHS
# ══════════════════════════════════════════════════════════════════

all_results = []

# NeuroCrystal
print("\nLoading NeuroCrystal...", flush=True)
name, feats, adj, vc, cn = load_neurocrystal()
all_results.append(run_specificity_tests(name, feats, adj, vc, cn))

# PyG graphs
from torch_geometric.datasets import Planetoid, Amazon, CitationFull
for cls, kwargs, gname in [
    (Planetoid, {"root": "/tmp/pyg_data", "name": "Cora"}, "Cora"),
    (Amazon, {"root": "/tmp/pyg_data", "name": "Computers"}, "Amazon Computers"),
    (CitationFull, {"root": "/tmp/pyg_data", "name": "DBLP"}, "DBLP"),
]:
    name, feats, adj, vc, cn = load_pyg_graph(gname, cls, **kwargs)
    all_results.append(run_specificity_tests(name, feats, adj, vc, cn))

# Synthetic
name, feats, adj, vc, cn = load_synthetic()
all_results.append(run_specificity_tests(name, feats, adj, vc, cn))


# ══════════════════════════════════════════════════════════════════
# CROSS-GRAPH SUMMARY
# ══════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("CROSS-GRAPH SPECIFICITY TRAP SUMMARY")
print(f"{'=' * 70}")

print(f"\n  {'Graph':>20s} {'T1(spec-deg)':>13s} {'T2(sel>cand)':>13s} {'T3(deg↓step)':>13s} {'T4(cone↓)':>10s} {'Trap?':>6s}")
print(f"  {'-'*20} {'-'*13} {'-'*13} {'-'*13} {'-'*10} {'-'*6}")

for r in all_results:
    t1 = f"{r.get('T1_rho', 0):+.4f}"
    t2 = "YES" if r.get('T2_sel_spec', 0) > r.get('T2_cand_spec', 0) else "NO"
    t3 = f"{r.get('T3_rho', 0):+.4f}"
    t4 = f"{r.get('T4_rho', 0):+.4f}"

    t1_pass = r.get('T1_rho', 0) < -0.05
    t2_pass = r.get('T2_sel_spec', 0) > r.get('T2_cand_spec', 0)
    t3_pass = r.get('T3_rho', 0) < -0.3
    t4_pass = r.get('T4_rho', 0) < -0.3
    trap = "YES" if (t1_pass and t2_pass) else "PART" if (t1_pass or t2_pass) else "NO"

    print(f"  {r['name']:>20s} {t1:>13s} {t2:>13s} {t3:>13s} {t4:>10s} {trap:>6s}")

print(f"\n  THE SPECIFICITY TRAP:")
print(f"  If T1 (specificity inversely correlates with degree) AND")
print(f"  T2 (cosine selects more specific nodes than average candidate) hold")
print(f"  across all graphs, then:")
print(f"  → Cosine does exactly what it's designed to do: find the sharpest direction")
print(f"  → The sharpest direction belongs to the most specialized node")
print(f"  → The most specialized node has the fewest connections")
print(f"  → Budget burns through a chain of increasingly isolated specialists")
print(f"  → This is a PROPERTY OF COSINE IN HIGH-D SPACE, not a defect")

with open(OUT / "exp27_specificity_trap.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved: {OUT / 'exp27_specificity_trap.json'}")
print("SPECIFICITY TRAP ANALYSIS COMPLETE")
