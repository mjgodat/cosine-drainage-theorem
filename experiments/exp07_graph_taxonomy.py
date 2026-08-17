"""
Experiment 7: Graph Taxonomy & Curvature Analysis

Phase 1: Classify each graph's structural type from measurable properties
Phase 2: Compute Ollivier-Ricci curvature (P3)
Phase 3: Compute local Cheeger conductance (P1)
Phase 4: LID-curvature correspondence (P10)
Phase 5: Network MI between cosine-kNN and graph edges (P5)
Phase 6: Cross-tabulate curvature × cosine (P3 core test)

Graphs: PRSM, Cora, Amazon Computers, DBLP, Synthetic Isotropic, Synthetic Mixture
(Hetionet/STRING from prior results referenced in summary)

All heavy computation on GPU where possible.
"""
import sys
import time
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict, Counter
from math import log2, pi
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════
# GRAPH STRUCTURE CLASSIFIER
# ══════════════════════════════════════════════════════════════════

def classify_graph_structure(name, features, edge_index, labels=None):
    """Compute structural properties and classify graph type."""
    N, D = features.shape
    print(f"\n{'=' * 70}")
    print(f"GRAPH: {name} ({N:,} nodes, {D}D)")
    print(f"{'=' * 70}")

    # Build adjacency
    adj = defaultdict(set)
    src, dst = edge_index[0], edge_index[1]
    for s, d in zip(src, dst):
        adj[int(s)].add(int(d))
        adj[int(d)].add(int(s))

    # Degree distribution
    degrees = np.array([len(adj.get(i, set())) for i in range(N)])
    deg_nonzero = degrees[degrees > 0]
    mean_deg = np.mean(deg_nonzero) if len(deg_nonzero) > 0 else 0
    max_deg = np.max(degrees)
    deg_cv = np.std(deg_nonzero) / mean_deg if mean_deg > 0 else 0

    # Power law exponent (log-log fit)
    deg_counts = Counter(degrees[degrees > 0])
    if len(deg_counts) > 5:
        dv = sorted(deg_counts.keys())
        df = [deg_counts[v] for v in dv]
        lv = np.log10([float(v) for v in dv])
        lf = np.log10([float(f) for f in df])
        alpha = -np.polyfit(lv, lf, 1)[0]
    else:
        alpha = 0.0

    # Clustering coefficient (sampled)
    sample_nodes = np.random.choice([n for n in range(N) if degrees[n] > 1],
                                     min(2000, sum(degrees > 1)), replace=False)
    cc_vals = []
    for node in sample_nodes:
        neighbors = list(adj.get(node, set()))
        k = len(neighbors)
        if k < 2:
            continue
        # Count edges among neighbors
        triangles = 0
        nb_set = set(neighbors)
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i+1:]:
                if n2 in adj.get(n1, set()):
                    triangles += 1
        possible = k * (k - 1) / 2
        cc_vals.append(triangles / possible if possible > 0 else 0)
    mean_cc = np.mean(cc_vals) if cc_vals else 0

    # Connected components (BFS)
    visited_all = set()
    n_components = 0
    largest_component = 0
    for start in range(N):
        if start in visited_all or not adj.get(start):
            continue
        component = set()
        frontier = {start}
        while frontier:
            component.update(frontier)
            next_f = set()
            for n in frontier:
                for nb in adj.get(n, set()):
                    if nb not in component:
                        next_f.add(nb)
            frontier = next_f
        visited_all.update(component)
        n_components += 1
        largest_component = max(largest_component, len(component))

    # Node type diversity
    if labels is not None:
        n_types = len(np.unique(labels))
        type_entropy = 0
        counts = np.bincount(labels)
        probs = counts[counts > 0] / len(labels)
        type_entropy = -np.sum(probs * np.log2(probs + 1e-10))
    else:
        n_types = 1
        type_entropy = 0

    # Angular compression (Gamma)
    t = torch.tensor(features, device=DEVICE)
    t_normed = t / t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    SAMPLE = min(3000, N)
    sidx = np.random.choice(N, SAMPLE, replace=False)
    sims = (t_normed[sidx] @ t_normed[sidx].T).cpu().numpy()
    np.fill_diagonal(sims, 1.0)
    triu = np.triu_indices(SAMPLE, k=1)
    angles = np.arccos(np.clip(sims[triu], -1, 1))
    gamma = 1 - np.mean(angles) / (pi / 2)

    # Edge cosine distribution
    edge_cosines = []
    edge_sample = min(50000, len(src))
    for i in np.random.choice(len(src), edge_sample, replace=False):
        s_idx, d_idx = int(src[i]), int(dst[i])
        if s_idx < N and d_idx < N:
            cos = (t_normed[s_idx] @ t_normed[d_idx]).item()
            edge_cosines.append(cos)
    edge_cos_mean = np.mean(edge_cosines) if edge_cosines else 0
    edge_cos_std = np.std(edge_cosines) if edge_cosines else 0

    # Diameter estimate (sampled BFS)
    diameters = []
    for _ in range(50):
        start = np.random.choice([n for n in range(N) if degrees[n] > 0])
        visited = {start}
        frontier = {start}
        d = 0
        while frontier:
            next_f = set()
            for n in frontier:
                for nb in adj.get(n, set()):
                    if nb not in visited:
                        visited.add(nb)
                        next_f.add(nb)
            if next_f:
                d += 1
            frontier = next_f
        diameters.append(d)
    est_diameter = max(diameters)
    avg_eccentricity = np.mean(diameters)

    # Small-world coefficient: CC / CC_random and L / L_random
    cc_random = mean_deg / N if N > 0 else 0
    sw_cc_ratio = mean_cc / cc_random if cc_random > 0 else 0

    props = {
        "name": name, "N": N, "D": D,
        "edges": len(src) // 2,
        "mean_degree": float(mean_deg),
        "max_degree": int(max_deg),
        "degree_cv": float(deg_cv),
        "power_law_alpha": float(alpha),
        "clustering_coeff": float(mean_cc),
        "n_components": n_components,
        "largest_component": largest_component,
        "est_diameter": est_diameter,
        "avg_eccentricity": float(avg_eccentricity),
        "n_types": n_types,
        "type_entropy": float(type_entropy),
        "gamma": float(gamma),
        "edge_cos_mean": float(edge_cos_mean),
        "edge_cos_std": float(edge_cos_std),
        "sw_cc_ratio": float(sw_cc_ratio),
    }

    # Classify
    if n_types > 3:
        structure = "HETEROGENEOUS"
    elif deg_cv > 2.0 or alpha > 2.0:
        structure = "SCALE-FREE"
    elif mean_cc > 0.3 and sw_cc_ratio > 10:
        structure = "SMALL-WORLD"
    elif mean_cc < 0.05 and edge_cos_mean > 0.5:
        structure = "CONTINUOUS-MANIFOLD"
    elif deg_cv < 0.3:
        structure = "REGULAR/SPATIAL"
    else:
        structure = "MODULAR"

    props["classification"] = structure

    print(f"  Nodes: {N:,}, Edges: {len(src)//2:,}")
    print(f"  Degree: mean={mean_deg:.1f}, max={max_deg}, CV={deg_cv:.3f}")
    print(f"  Power law alpha: {alpha:.3f}")
    print(f"  Clustering coefficient: {mean_cc:.4f}")
    print(f"  Components: {n_components}, Largest: {largest_component:,}")
    print(f"  Est. diameter: {est_diameter}, Avg eccentricity: {avg_eccentricity:.1f}")
    print(f"  Node types: {n_types}, Type entropy: {type_entropy:.3f}")
    print(f"  Gamma: {gamma:.4f}")
    print(f"  Edge cosine: mean={edge_cos_mean:.4f}, std={edge_cos_std:.4f}")
    print(f"  Small-world CC ratio: {sw_cc_ratio:.1f}")
    print(f"  >>> CLASSIFICATION: {structure}")

    return props, adj, t_normed


# ══════════════════════════════════════════════════════════════════
# OLLIVIER-RICCI CURVATURE (sampled)
# ══════════════════════════════════════════════════════════════════

def compute_ollivier_ricci_sampled(adj, t_normed, N, n_edges=5000):
    """Compute Ollivier-Ricci curvature on sampled edges.
    ORC(u,v) = 1 - W1(mu_u, mu_v) / d(u,v)
    where mu_u is uniform over neighbors of u, W1 is Wasserstein-1.
    Approximation: use cosine distance as metric, neighbor overlap as proxy.
    """
    print("  Computing Ollivier-Ricci curvature (sampled)...", end=" ", flush=True)
    t0 = time.time()

    # Collect edges
    all_edges = []
    for u in adj:
        for v in adj[u]:
            if u < v:
                all_edges.append((u, v))

    if len(all_edges) > n_edges:
        sample_idx = np.random.choice(len(all_edges), n_edges, replace=False)
        edges_sample = [all_edges[i] for i in sample_idx]
    else:
        edges_sample = all_edges

    curvatures = []
    edge_cosines = []

    for u, v in edges_sample:
        nb_u = list(adj.get(u, set()))
        nb_v = list(adj.get(v, set()))

        if not nb_u or not nb_v:
            continue

        # Neighbor overlap ratio (proxy for ORC)
        # High overlap = positive curvature (within community)
        # Low overlap = negative curvature (bridge edge)
        set_u = set(nb_u)
        set_v = set(nb_v)
        intersection = len(set_u & set_v)
        union = len(set_u | set_v)
        jaccard = intersection / union if union > 0 else 0

        # ORC approximation via neighbor distribution distance
        # Use mean cosine of u's neighbors to v's neighbors
        if len(nb_u) > 20:
            nb_u = list(np.random.choice(nb_u, 20, replace=False))
        if len(nb_v) > 20:
            nb_v = list(np.random.choice(nb_v, 20, replace=False))

        # Cross-neighborhood cosine (GPU batch)
        vecs_u = t_normed[nb_u]  # (|nb_u|, D)
        vecs_v = t_normed[nb_v]  # (|nb_v|, D)
        cross_cos = (vecs_u @ vecs_v.T).cpu().numpy()  # (|nb_u|, |nb_v|)

        # Wasserstein-1 approximation: optimal transport between uniform distributions
        # Use greedy assignment as proxy
        mean_cross = float(np.mean(cross_cos))

        # ORC ≈ 1 - (1 - mean_cross) / (1 - edge_cos)
        edge_cos = (t_normed[u] @ t_normed[v]).item()
        edge_dist = 1 - edge_cos
        nb_dist = 1 - mean_cross

        if edge_dist > 0.001:
            orc = 1 - nb_dist / edge_dist
        else:
            orc = 1.0  # identical nodes

        curvatures.append(orc)
        edge_cosines.append(edge_cos)

    print(f"{time.time()-t0:.1f}s, {len(curvatures)} edges")

    return np.array(curvatures), np.array(edge_cosines), edges_sample[:len(curvatures)]


# ══════════════════════════════════════════════════════════════════
# LOCAL CHEEGER CONDUCTANCE
# ══════════════════════════════════════════════════════════════════

def compute_local_conductance(adj, N, n_nodes=2000):
    """Compute local Cheeger conductance for sampled nodes.
    Conductance = boundary_edges / min(vol(S), vol(V\S))
    where S is the 2-hop neighborhood.
    """
    print("  Computing local Cheeger conductance...", end=" ", flush=True)
    t0 = time.time()

    nodes = [n for n in range(N) if len(adj.get(n, set())) > 1]
    if len(nodes) > n_nodes:
        nodes = list(np.random.choice(nodes, n_nodes, replace=False))

    conductances = []
    node_ids = []

    for node in nodes:
        # 2-hop neighborhood
        S = {node}
        frontier = {node}
        for _ in range(2):
            next_f = set()
            for n in frontier:
                for nb in adj.get(n, set()):
                    if nb not in S:
                        next_f.add(nb)
                        S.add(nb)
            frontier = next_f

        if len(S) < 3 or len(S) > N - 3:
            continue

        # Count boundary and internal edges
        vol_S = sum(len(adj.get(n, set())) for n in S)
        boundary = 0
        for n in S:
            for nb in adj.get(n, set()):
                if nb not in S:
                    boundary += 1

        vol_complement = sum(len(adj.get(n, set())) for n in range(N) if n not in S)
        min_vol = min(vol_S, vol_complement)

        if min_vol > 0:
            cond = boundary / min_vol
        else:
            cond = 0

        conductances.append(cond)
        node_ids.append(node)

    print(f"{time.time()-t0:.1f}s, {len(conductances)} nodes")
    return np.array(conductances), np.array(node_ids)


# ══════════════════════════════════════════════════════════════════
# NETWORK MI (cosine-kNN vs graph edges)
# ══════════════════════════════════════════════════════════════════

def compute_network_mi(adj, t_normed, N, k_values=[5, 10, 20, 50, 100]):
    """Mutual information between cosine-kNN graph and actual graph edges."""
    print("  Computing network MI...", end=" ", flush=True)
    t0 = time.time()

    # Build graph adjacency as set for fast lookup
    graph_edges = set()
    for u in adj:
        for v in adj[u]:
            graph_edges.add((min(u, v), max(u, v)))

    # Sample nodes for MI computation
    sample_nodes = np.random.choice(N, min(2000, N), replace=False)

    mi_values = {}
    for k in k_values:
        # Build cosine-kNN for sample nodes
        tp = fp = fn = tn = 0
        for node in sample_nodes:
            # Get cosine kNN
            sims = (t_normed[node:node+1] @ t_normed.T).squeeze(0)
            sims[node] = -2
            _, topk = sims.topk(k)
            knn_set = set(topk.cpu().numpy().tolist())

            # Get graph neighbors
            graph_nb = adj.get(int(node), set())

            # Count overlaps
            for nb in knn_set:
                edge = (min(node, nb), max(node, nb))
                if nb in graph_nb:
                    tp += 1
                else:
                    fp += 1

            for nb in graph_nb:
                if nb not in knn_set:
                    fn += 1

        # MI approximation from contingency
        total = tp + fp + fn + 1  # +1 to avoid div0
        p_knn = (tp + fp) / total
        p_graph = (tp + fn) / total
        p_both = tp / total

        if p_both > 0 and p_knn > 0 and p_graph > 0:
            mi = p_both * log2(p_both / (p_knn * p_graph + 1e-10) + 1e-10)
        else:
            mi = 0

        overlap_pct = tp / (tp + fp + fn + 1) * 100
        mi_values[k] = {"mi": float(mi), "tp": tp, "fp": fp, "fn": fn,
                        "overlap_pct": float(overlap_pct)}

    print(f"{time.time()-t0:.1f}s")
    return mi_values


# ══════════════════════════════════════════════════════════════════
# FULL ANALYSIS PER GRAPH
# ══════════════════════════════════════════════════════════════════

def analyze_graph(name, features, edge_index, labels=None):
    """Run full taxonomy + curvature + conductance + MI analysis."""
    # Phase 1: Classify structure
    props, adj, t_normed = classify_graph_structure(name, features, edge_index, labels)
    N = len(features)

    # Phase 2: Ollivier-Ricci curvature
    curvatures, edge_cos, edge_list = compute_ollivier_ricci_sampled(adj, t_normed, N)

    # Phase 3: Local conductance
    conductances, cond_nodes = compute_local_conductance(adj, N)

    # Phase 4: LID (quick estimate)
    print("  Computing LID...", end=" ", flush=True)
    t0 = time.time()
    sample = min(2000, N)
    sidx = np.random.choice(N, sample, replace=False)
    t_sample = torch.tensor(features[sidx], device=DEVICE)
    lids = []
    k_lid = 20
    BATCH = 500
    for start in range(0, sample, BATCH):
        end = min(start + BATCH, sample)
        dists = torch.cdist(t_sample[start:end], t_sample)
        for i in range(end - start):
            dists[i, start + i] = float('inf')
        topk_d, _ = dists.topk(k_lid, dim=1, largest=False)
        r_K = topk_d[:, -1:]
        log_ratios = torch.log(r_K / topk_d[:, :-1].clamp(min=1e-10))
        lid_batch = (k_lid - 1) / log_ratios.sum(dim=1).clamp(min=1e-10)
        lids.extend(lid_batch.cpu().numpy().tolist())
    lids = np.array(lids)
    print(f"{time.time()-t0:.1f}s, mean LID={np.mean(lids):.2f}")

    # Phase 5: Network MI
    mi_values = compute_network_mi(adj, t_normed, N)

    # ── CROSS-TABULATION: curvature × cosine ──
    print("\n  CURVATURE x COSINE CROSS-TABULATION:")
    if len(curvatures) > 0:
        cos_median = np.median(edge_cos)
        orc_median = np.median(curvatures)

        q1 = np.sum((edge_cos >= cos_median) & (curvatures >= orc_median))  # high cos, pos curv
        q2 = np.sum((edge_cos < cos_median) & (curvatures >= orc_median))   # low cos, pos curv
        q3 = np.sum((edge_cos < cos_median) & (curvatures < orc_median))    # low cos, neg curv
        q4 = np.sum((edge_cos >= cos_median) & (curvatures < orc_median))   # high cos, neg curv

        total_edges = len(curvatures)
        print(f"                    High Cosine    Low Cosine")
        print(f"    Pos Curvature   {q1:6d} ({q1/total_edges*100:5.1f}%)  {q2:6d} ({q2/total_edges*100:5.1f}%)")
        print(f"    Neg Curvature   {q4:6d} ({q4/total_edges*100:5.1f}%)  {q3:6d} ({q3/total_edges*100:5.1f}%)")

        # If VGSG is correct: high-cos + pos-curv should dominate (within-basin)
        # and low-cos + neg-curv should be elevated (bridge edges)
        diag_pct = (q1 + q3) / total_edges * 100
        print(f"    Diagonal dominance: {diag_pct:.1f}% (>50% supports VGSG)")

        rho_cos_orc, p_cos_orc = spearmanr(edge_cos, curvatures)
        print(f"    Spearman(cosine, ORC): rho={rho_cos_orc:.4f}, p={p_cos_orc:.2e}")

        props["orc_cos_spearman"] = float(rho_cos_orc)
        props["orc_cos_p"] = float(p_cos_orc)
        props["orc_diagonal_pct"] = float(diag_pct)
        props["orc_mean"] = float(np.mean(curvatures))
        props["orc_std"] = float(np.std(curvatures))
        props["orc_pct_negative"] = float(np.sum(curvatures < 0) / len(curvatures) * 100)

    # ── Conductance statistics ──
    if len(conductances) > 0:
        props["conductance_mean"] = float(np.mean(conductances))
        props["conductance_std"] = float(np.std(conductances))
        props["conductance_median"] = float(np.median(conductances))
        print(f"\n  CONDUCTANCE: mean={np.mean(conductances):.4f}, "
              f"median={np.median(conductances):.4f}, std={np.std(conductances):.4f}")

    # ── LID statistics ──
    props["lid_mean"] = float(np.mean(lids))
    props["lid_std"] = float(np.std(lids))

    # ── Network MI ──
    print(f"\n  NETWORK MI (cosine-kNN vs graph edges):")
    print(f"    {'k':>5s} {'Overlap%':>10s} {'MI':>10s}")
    for k, v in sorted(mi_values.items()):
        print(f"    {k:5d} {v['overlap_pct']:10.2f}% {v['mi']:10.6f}")
    props["network_mi"] = mi_values

    # ── LID-Curvature correspondence ──
    if len(curvatures) > 100 and len(lids) > 100:
        # Map edge curvature to nodes (average ORC of incident edges)
        node_orc = defaultdict(list)
        for (u, v), orc in zip(edge_list, curvatures):
            node_orc[u].append(orc)
            node_orc[v].append(orc)

        # Compute correlation for sampled nodes that have both LID and ORC
        lid_vals = []
        orc_vals = []
        for i, node_idx in enumerate(sidx):
            node_idx = int(node_idx)
            if node_idx in node_orc and i < len(lids):
                lid_vals.append(lids[i])
                orc_vals.append(np.mean(node_orc[node_idx]))

        if len(lid_vals) > 50:
            rho_lid_orc, p_lid_orc = spearmanr(lid_vals, orc_vals)
            print(f"\n  LID-CURVATURE CORRESPONDENCE:")
            print(f"    Spearman(LID, mean_ORC): rho={rho_lid_orc:.4f}, p={p_lid_orc:.2e}")
            print(f"    (Negative = low LID hubs have positive curvature = basin interiors)")
            props["lid_orc_spearman"] = float(rho_lid_orc)
            props["lid_orc_p"] = float(p_lid_orc)

    return props


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    all_results = []

    # ── PRSM Crystal ──
    print("\nLoading PRSM Crystal...")
    import pickle
    ROOT = Path("E:/PRSM")
    with open(ROOT / "data/g1_registry/unified_grains_with_embeddings.json", "r", encoding="utf-8") as f:
        grains = json.load(f)
    with open(ROOT / "data/g1_registry/relationship_registry.json", "r", encoding="utf-8") as f:
        rels = json.load(f)

    seen = {}
    feats, gids = [], []
    for g in grains:
        c = g.get("concept", "").lower()
        emb = g.get("embedding")
        bc = g.get("bind_count", 0)
        if c and emb and (c not in seen or bc > seen[c]):
            seen[c] = bc
            if c not in [x[0] for x in feats]:
                feats.append((c, np.array(emb, dtype=np.float32), g.get("grain_id", "")))

    prsm_features = np.array([f[1] for f in feats])
    gid_to_idx = {}
    concept_to_idx = {}
    for i, (c, _, gid) in enumerate(feats):
        concept_to_idx[c] = i
        gid_to_idx[gid] = i

    # Also map all grain_ids (including duplicates) to concept indices
    for g in grains:
        gid = g.get("grain_id", "")
        c = g.get("concept", "").lower()
        if gid and c and c in concept_to_idx:
            gid_to_idx[gid] = concept_to_idx[c]

    rows, cols = [], []
    for r in rels:
        a = gid_to_idx.get(r.get("grain_a_id", ""))
        b = gid_to_idx.get(r.get("grain_b_id", ""))
        if a is not None and b is not None and a != b:
            rows.extend([a, b])
            cols.extend([b, a])
    prsm_edges = np.array([rows, cols], dtype=np.int64)

    all_results.append(analyze_graph("PRSM Crystal", prsm_features, prsm_edges))

    # ── PyG Graphs ──
    from torch_geometric.datasets import Planetoid, Amazon, CitationFull

    for cls, kwargs, name in [
        (Planetoid, {"root": "/tmp/pyg_data", "name": "Cora"}, "Cora"),
        (Amazon, {"root": "/tmp/pyg_data", "name": "Computers"}, "Amazon Computers"),
        (CitationFull, {"root": "/tmp/pyg_data", "name": "DBLP"}, "DBLP"),
    ]:
        d = cls(**kwargs); data = d[0]
        all_results.append(analyze_graph(name,
            data.x.numpy().astype(np.float32),
            data.edge_index.numpy(),
            data.y.numpy()))

    # ── Synthetics ──
    N_SYN = 10000
    D_SYN = 768

    # Isotropic
    iso = np.random.randn(N_SYN, D_SYN).astype(np.float32)
    t_gpu = torch.tensor(iso, device=DEVICE)
    knn = torch.zeros(N_SYN, 20, dtype=torch.long, device=DEVICE)
    for s in range(0, N_SYN, 512):
        e = min(s + 512, N_SYN)
        dd = torch.cdist(t_gpu[s:e], t_gpu)
        for i in range(e - s): dd[i, s + i] = float('inf')
        _, tk = dd.topk(20, dim=1, largest=False)
        knn[s:e] = tk
    knn_np = knn.cpu().numpy()
    r, c = [], []
    for i in range(N_SYN):
        for j in knn_np[i]:
            r.append(i); c.append(int(j))
    iso_edges = np.array([r, c], dtype=np.int64)
    all_results.append(analyze_graph("Synthetic Isotropic", iso, iso_edges))

    # Mixture
    mix = np.zeros((N_SYN, D_SYN), dtype=np.float32)
    centers = [np.random.randn(D_SYN).astype(np.float32) * 5 for _ in range(3)]
    labels_mix = np.zeros(N_SYN, dtype=np.int32)
    idx = 0
    for ci, (center, size) in enumerate(zip(centers, [7000, 2000, 1000])):
        mix[idx:idx+size] = np.random.randn(size, D_SYN).astype(np.float32) + center
        labels_mix[idx:idx+size] = ci
        idx += size
    t_gpu = torch.tensor(mix, device=DEVICE)
    knn = torch.zeros(N_SYN, 20, dtype=torch.long, device=DEVICE)
    for s in range(0, N_SYN, 512):
        e = min(s + 512, N_SYN)
        dd = torch.cdist(t_gpu[s:e], t_gpu)
        for i in range(e - s): dd[i, s + i] = float('inf')
        _, tk = dd.topk(20, dim=1, largest=False)
        knn[s:e] = tk
    knn_np = knn.cpu().numpy()
    r, c = [], []
    for i in range(N_SYN):
        for j in knn_np[i]:
            r.append(i); c.append(int(j))
    mix_edges = np.array([r, c], dtype=np.int64)
    all_results.append(analyze_graph("Synthetic Mixture", mix, mix_edges, labels_mix))

    # ══════════════════════════════════════════════════════════════
    # CROSS-GRAPH COMPARISON
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("GRAPH TAXONOMY")
    print(f"{'=' * 80}")

    print(f"\n  {'Graph':25s} {'Type':20s} {'N':>7s} {'MeanDeg':>8s} {'DegCV':>7s} {'CC':>7s} {'Gamma':>7s} {'EdgeCos':>8s} {'ORC_mean':>9s} {'Cond':>7s}")
    print(f"  {'-'*25} {'-'*20} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*9} {'-'*7}")

    for r in all_results:
        print(f"  {r['name']:25s} {r['classification']:20s} {r['N']:7d} "
              f"{r['mean_degree']:8.1f} {r['degree_cv']:7.3f} {r['clustering_coeff']:7.4f} "
              f"{r['gamma']:7.4f} {r['edge_cos_mean']:8.4f} "
              f"{r.get('orc_mean', 0):9.4f} {r.get('conductance_mean', 0):7.4f}")

    # Cross-tabulation summary
    print(f"\n  CURVATURE x COSINE DIAGONAL DOMINANCE:")
    print(f"  (>50% = angular neighborhoods align with curvature-defined communities)")
    for r in all_results:
        diag = r.get('orc_diagonal_pct', 0)
        rho = r.get('orc_cos_spearman', 0)
        marker = " <-- VGSG SUPPORTED" if diag > 55 and rho > 0.1 else ""
        print(f"    {r['name']:25s}: diagonal={diag:.1f}%, rho(cos,ORC)={rho:.4f}{marker}")

    # LID-Curvature
    print(f"\n  LID-CURVATURE CORRESPONDENCE:")
    for r in all_results:
        rho = r.get('lid_orc_spearman', 0)
        p = r.get('lid_orc_p', 1)
        if p < 0.05:
            print(f"    {r['name']:25s}: rho={rho:.4f} (p={p:.2e}) {'SIGNIFICANT' if p < 0.001 else 'marginal'}")
        else:
            print(f"    {r['name']:25s}: rho={rho:.4f} (p={p:.2e}) ns")

    # Network MI
    print(f"\n  NETWORK MI (cosine-kNN overlap with graph at k=20):")
    for r in all_results:
        mi = r.get('network_mi', {}).get(20, {})
        print(f"    {r['name']:25s}: overlap={mi.get('overlap_pct', 0):.1f}%")

    # Prior graphs (from memory)
    print(f"\n  PRIOR GRAPH CLASSIFICATIONS (from previous experiments):")
    print(f"    Hetionet v1.0:        HETEROGENEOUS (24 types, 47K nodes)")
    print(f"    STRING-DB v12.0:      SCALE-FREE (hub-dominated, TP53=4534 edges)")
    print(f"    Grid Road:            REGULAR/SPATIAL (planar grid, 10K nodes)")

    # Save
    outfile = OUT / "exp7_graph_taxonomy_curvature.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nTotal time: {time.time()-t_total:.0f}s")
    print(f"Saved: {outfile}")
    print("GRAPH TAXONOMY & CURVATURE ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
