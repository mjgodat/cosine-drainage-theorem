"""
KTS Graph Fingerprinting — Can kinematic signatures classify graph construction geometry?

Takes every graph we've tested and computes a fingerprint vector from KTS metrics.
Then clusters the fingerprints to see if structurally similar graphs land together.

Fingerprint dimensions (per graph):
  1. kNN local momentum (feature-space smoothness)
  2. Graph walk momentum (edge topology alignment)
  3. Cross-structure momentum (boundary-crossing behavior)
  4. Random momentum (baseline — should be ~-0.50 everywhere)
  5. kNN-graph momentum gap (feature vs edge divergence)
  6. kNN local tortuosity
  7. Graph walk tortuosity
  8. Cross-structure tortuosity
  9. kNN-graph tortuosity gap
  10. kNN eccentricity
  11. Graph walk eccentricity
  12. Cross-structure eccentricity
  13. kNN saturation
  14. Graph walk saturation
  15. Cross-structure saturation

Graphs:
  - Cora (citation, 7 topics)
  - Amazon Computers (co-purchase, 10 categories)
  - DBLP (citation, academic, 4 classes)
  - Grid Road (spatial, zero semantics)
  - Synthetic Isotropic (pure Gaussian, no structure)
  - Synthetic Mixture (3 clusters)
  - Synthetic Sphere (L2-normalized, no hubness)
  - Synthetic Low-Dim (50D manifold in 768D)

GPU-accelerated.
"""
import sys
import time
import json
import numpy as np
import torch
from scipy.stats import mannwhitneyu
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

N_WALKS = 500
WALK_LEN = 7
K_NN = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════
# CORE KTS ENGINE (from previous tests, consolidated)
# ══════════════════════════════════════════════════════════════════

def build_knn_gpu(points_np, k=K_NN):
    t = torch.tensor(points_np, device=DEVICE)
    N = len(points_np)
    knn_idx = torch.zeros(N, k, dtype=torch.long, device=DEVICE)
    BATCH = 512
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        dists = torch.cdist(t[start:end], t)
        for i in range(end - start):
            dists[i, start + i] = float('inf')
        _, topk = dists.topk(k, dim=1, largest=False)
        knn_idx[start:end] = topk
    return knn_idx.cpu().numpy()


def compute_kts_batch(points_np, walks):
    t = torch.tensor(points_np, device=DEVICE)
    results = []
    for path in walks:
        if len(path) < 3:
            continue
        vecs = t[path]
        n = len(path)
        steps = vecs[1:] - vecs[:-1]
        step_norms = steps.norm(dim=1).clamp(min=1e-8)
        step_cosines = []
        for i in range(len(steps) - 1):
            cos = torch.dot(steps[i], steps[i+1]) / (step_norms[i] * step_norms[i+1])
            step_cosines.append(cos.item())
        momentum = float(np.mean(step_cosines)) if step_cosines else 0.0
        chord = (vecs[-1] - vecs[0]).norm().item()
        arc = sum((vecs[i+1] - vecs[i]).norm().item() for i in range(n - 1))
        tortuosity = chord / max(arc, 1e-8)
        vec_sum = vecs.sum(dim=0)
        tav_mag = vec_sum.norm().item() / n
        centered = vecs - vecs.mean(dim=0, keepdim=True)
        try:
            _, S, _ = torch.linalg.svd(centered, full_matrices=False)
            s = S[:3].cpu().numpy()
            eccentricity = float(s[0] / (s[1] + s[2] + 1e-8)) if len(s) >= 3 else 0.0
        except Exception:
            eccentricity = 0.0
        path_set = set(path)
        alphas_list = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        prev_nearest = -1
        sat_alpha = alphas_list[-1]
        for alpha in alphas_list:
            scaled = vec_sum * (alpha / n)
            dists = torch.norm(t - scaled.unsqueeze(0), dim=1)
            for idx in path:
                dists[idx] = float('inf')
            nearest = int(dists.argmin().item())
            if nearest == prev_nearest and prev_nearest >= 0:
                sat_alpha = alpha
                break
            prev_nearest = nearest
        results.append({
            "momentum": momentum, "tortuosity": tortuosity,
            "tav_mag": tav_mag, "eccentricity": eccentricity,
            "sat_alpha": sat_alpha,
        })
    return results


def gen_knn_walks(knn_idx, n=N_WALKS, wl=WALK_LEN):
    N = len(knn_idx)
    walks = []
    for _ in range(n):
        start = np.random.randint(N)
        path = [start]
        cur = start
        for _ in range(wl - 1):
            nb = knn_idx[cur]
            nxt = nb[np.random.randint(len(nb))]
            path.append(int(nxt))
            cur = int(nxt)
        walks.append(path)
    return walks


def gen_graph_walks(edge_index, num_nodes, n=N_WALKS, wl=WALK_LEN):
    adj = {}
    src, dst = edge_index[0], edge_index[1]
    for s, d in zip(src, dst):
        adj.setdefault(int(s), []).append(int(d))
    walks = []
    nodes_with_edges = [nd for nd in adj if len(adj[nd]) > 0]
    if not nodes_with_edges:
        return walks
    att = 0
    while len(walks) < n and att < n * 10:
        att += 1
        start = nodes_with_edges[np.random.randint(len(nodes_with_edges))]
        path = [start]
        cur = start
        for _ in range(wl - 1):
            nb = adj.get(cur, [])
            if not nb:
                break
            nxt = nb[np.random.randint(len(nb))]
            path.append(nxt)
            cur = nxt
        if len(path) == wl:
            walks.append(path)
    return walks


def gen_cross_class(labels, knn_idx, n=N_WALKS, wl=WALK_LEN):
    ul = np.unique(labels)
    ci = {l: np.where(labels == l)[0] for l in ul}
    walks = []
    for _ in range(n):
        path = []
        co = np.random.permutation(ul)
        for step in range(wl):
            c = co[step % len(co)]
            pool = ci[c]
            if path:
                last = path[-1]
                nb = knn_idx[last]
                inc = [nn for nn in nb if labels[nn] == c]
                path.append(int(inc[0]) if inc else int(np.random.choice(pool)))
            else:
                path.append(int(np.random.choice(pool)))
        walks.append(path)
    return walks


def gen_random(N, n=N_WALKS, wl=WALK_LEN):
    return [list(np.random.choice(N, wl, replace=False)) for _ in range(n)]


def mean_metric(items, key):
    vals = [x[key] for x in items]
    return float(np.mean(vals)) if vals else 0.0


def std_metric(items, key):
    vals = [x[key] for x in items]
    return float(np.std(vals)) if vals else 0.0


# ══════════════════════════════════════════════════════════════════
# GRAPH LOADERS
# ══════════════════════════════════════════════════════════════════

def load_pyg_graph(dataset_cls, **kwargs):
    d = dataset_cls(**kwargs)
    data = d[0]
    return (data.x.numpy().astype(np.float32),
            data.edge_index.numpy(),
            data.y.numpy())


def load_grid_road(grid_size=100):
    N = grid_size * grid_size
    rows, cols = [], []
    for i in range(grid_size):
        for j in range(grid_size):
            node = i * grid_size + j
            if j + 1 < grid_size:
                nb = i * grid_size + (j + 1)
                rows.extend([node, nb]); cols.extend([nb, node])
            if i + 1 < grid_size:
                nb = (i + 1) * grid_size + j
                rows.extend([node, nb]); cols.extend([nb, node])
    edges = np.array([rows, cols], dtype=np.int64)
    n_edges = len(rows)
    adj = csr_matrix((np.ones(n_edges), (rows, cols)), shape=(N, N))
    degrees = np.array(adj.sum(axis=1)).flatten()
    dis = np.zeros(N); dis[degrees > 0] = 1.0 / np.sqrt(degrees[degrees > 0])
    D = csr_matrix((dis, (range(N), range(N))), shape=(N, N))
    L = csr_matrix(np.eye(N)) - D @ adj @ D
    eigenvalues, eigenvectors = eigsh(L, k=129, which='SM', tol=1e-4)
    features = eigenvectors[:, 1:].astype(np.float32)
    labels = np.zeros(N, dtype=np.int32)
    for i in range(grid_size):
        for j in range(grid_size):
            labels[i*grid_size+j] = (0 if i < grid_size//2 else 2) + (0 if j < grid_size//2 else 1)
    return features, edges, labels


def gen_synthetic(dist_type, n_points=10000, n_dim=768):
    """Generate synthetic point cloud with kNN edges and artificial labels."""
    if dist_type == "isotropic":
        points = np.random.randn(n_points, n_dim).astype(np.float32)
        labels = np.zeros(n_points, dtype=np.int32)  # no real classes
        # Assign pseudo-labels by octant of first 3 dims
        for i in range(n_points):
            labels[i] = (1 if points[i, 0] > 0 else 0) + (2 if points[i, 1] > 0 else 0)
    elif dist_type == "mixture":
        points = np.zeros((n_points, n_dim), dtype=np.float32)
        centers = [np.random.randn(n_dim).astype(np.float32) * 5 for _ in range(3)]
        sizes = [7000, 2000, 1000]
        labels = np.zeros(n_points, dtype=np.int32)
        idx = 0
        for ci, (center, size) in enumerate(zip(centers, sizes)):
            points[idx:idx+size] = np.random.randn(size, n_dim).astype(np.float32) + center
            labels[idx:idx+size] = ci
            idx += size
    elif dist_type == "sphere":
        points = np.random.randn(n_points, n_dim).astype(np.float32)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        labels = np.zeros(n_points, dtype=np.int32)
        for i in range(n_points):
            labels[i] = (1 if points[i, 0] > 0 else 0) + (2 if points[i, 1] > 0 else 0)
    elif dist_type == "low_dim":
        low = np.random.randn(n_points, 50).astype(np.float32)
        proj = np.random.randn(50, n_dim).astype(np.float32) / np.sqrt(50)
        points = low @ proj + np.random.randn(n_points, n_dim).astype(np.float32) * 0.01
        labels = np.zeros(n_points, dtype=np.int32)
        for i in range(n_points):
            labels[i] = (1 if points[i, 0] > 0 else 0) + (2 if points[i, 1] > 0 else 0)
    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")

    # Build kNN edges as surrogate graph
    knn = build_knn_gpu(points, k=K_NN)
    rows, cols = [], []
    for i in range(n_points):
        for j in knn[i]:
            rows.append(i); cols.append(int(j))
    edges = np.array([rows, cols], dtype=np.int64)
    return points, edges, labels


# ══════════════════════════════════════════════════════════════════
# FINGERPRINT COMPUTATION
# ══════════════════════════════════════════════════════════════════

def compute_fingerprint(name, graph_type, features, edge_index, labels):
    """Run KTS on all path types and return a fingerprint vector."""
    N = len(features)
    print(f"\n  {name}: {N:,} nodes, {features.shape[1]}D, {len(np.unique(labels))} classes")

    print(f"    Building kNN...", end=" ", flush=True)
    t0 = time.time()
    knn_idx = build_knn_gpu(features, k=K_NN)
    print(f"{time.time()-t0:.1f}s")

    # Generate all path types
    paths = {
        "knn": gen_knn_walks(knn_idx),
        "graph": gen_graph_walks(edge_index, N),
        "cross": gen_cross_class(labels, knn_idx),
        "random": gen_random(N),
    }

    # Compute KTS
    kts = {}
    for ptype, walk_list in paths.items():
        print(f"    KTS {ptype}: {len(walk_list)} paths...", end=" ", flush=True)
        t0 = time.time()
        kts[ptype] = compute_kts_batch(features, walk_list)
        print(f"{time.time()-t0:.1f}s")

    # Build fingerprint vector
    metrics = ["momentum", "tortuosity", "eccentricity", "sat_alpha", "tav_mag"]
    fp = {}
    fp_vec = []

    for ptype in ["knn", "graph", "cross", "random"]:
        items = kts[ptype]
        for m in metrics:
            key = f"{ptype}_{m}"
            val = mean_metric(items, m)
            fp[key] = val
            fp_vec.append(val)

    # Computed gap features
    for m in metrics:
        gap = fp.get(f"knn_{m}", 0) - fp.get(f"graph_{m}", 0)
        fp[f"gap_knn_graph_{m}"] = gap
        fp_vec.append(gap)

        gap2 = fp.get(f"cross_{m}", 0) - fp.get(f"random_{m}", 0)
        fp[f"gap_cross_random_{m}"] = gap2
        fp_vec.append(gap2)

    fp["name"] = name
    fp["graph_type"] = graph_type
    fp["nodes"] = N
    fp["dims"] = features.shape[1]
    fp["fp_vector"] = fp_vec

    return fp


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    fingerprints = []

    # ── Real Graphs ──
    print("\n" + "=" * 80)
    print("PHASE 1: REAL-WORLD GRAPHS")
    print("=" * 80)

    from torch_geometric.datasets import Planetoid, Amazon, CitationFull

    # Cora
    f, e, l = load_pyg_graph(Planetoid, root='/tmp/pyg_data', name='Cora')
    fingerprints.append(compute_fingerprint("Cora", "citation", f, e, l))

    # Amazon Computers
    f, e, l = load_pyg_graph(Amazon, root='/tmp/pyg_data', name='Computers')
    fingerprints.append(compute_fingerprint("Amazon Computers", "co-purchase", f, e, l))

    # DBLP
    f, e, l = load_pyg_graph(CitationFull, root='/tmp/pyg_data', name='DBLP')
    fingerprints.append(compute_fingerprint("DBLP", "citation", f, e, l))

    # Grid Road
    print("\n  Generating Grid Road...", flush=True)
    f, e, l = load_grid_road(100)
    fingerprints.append(compute_fingerprint("Grid Road", "spatial", f, e, l))

    # ── Synthetic Graphs ──
    print("\n" + "=" * 80)
    print("PHASE 2: SYNTHETIC DISTRIBUTIONS")
    print("=" * 80)

    for dist in ["isotropic", "mixture", "sphere", "low_dim"]:
        print(f"\n  Generating synthetic {dist}...", flush=True)
        f, e, l = gen_synthetic(dist)
        fingerprints.append(compute_fingerprint(f"Synthetic {dist}", "synthetic", f, e, l))

    # ══════════════════════════════════════════════════════════════
    # FINGERPRINT ANALYSIS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 3: FINGERPRINT ANALYSIS")
    print("=" * 80)

    # Display fingerprint table (key metrics only)
    key_metrics = [
        ("knn_momentum", "kNN Mom"),
        ("graph_momentum", "Graph Mom"),
        ("cross_momentum", "Cross Mom"),
        ("random_momentum", "Rand Mom"),
        ("gap_knn_graph_momentum", "Gap(k-g) Mom"),
        ("knn_tortuosity", "kNN Tort"),
        ("graph_tortuosity", "Graph Tort"),
        ("gap_knn_graph_tortuosity", "Gap(k-g) Tort"),
        ("knn_sat_alpha", "kNN Sat"),
        ("graph_sat_alpha", "Graph Sat"),
    ]

    print(f"\n  {'Graph':25s} {'Type':12s}", end="")
    for _, label in key_metrics:
        print(f" {label:>12s}", end="")
    print()
    print(f"  {'-'*25} {'-'*12}", end="")
    for _ in key_metrics:
        print(f" {'-'*12}", end="")
    print()

    for fp in fingerprints:
        print(f"  {fp['name']:25s} {fp['graph_type']:12s}", end="")
        for key, _ in key_metrics:
            print(f" {fp.get(key, 0):12.4f}", end="")
        print()

    # ── Pairwise cosine similarity of fingerprint vectors ──
    print(f"\n  FINGERPRINT COSINE SIMILARITY MATRIX:")
    vecs = np.array([fp["fp_vector"] for fp in fingerprints])
    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vecs_normed = vecs / norms

    cos_matrix = vecs_normed @ vecs_normed.T
    names = [fp["name"][:20] for fp in fingerprints]

    print(f"\n  {'':22s}", end="")
    for n in names:
        print(f" {n:>12s}", end="")
    print()

    for i, n in enumerate(names):
        print(f"  {n:22s}", end="")
        for j in range(len(names)):
            val = cos_matrix[i, j]
            print(f" {val:12.3f}", end="")
        print()

    # ── Cluster detection ──
    print(f"\n  STRUCTURAL CLUSTERS (cos > 0.85):")
    visited = set()
    clusters = []
    for i in range(len(fingerprints)):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(i + 1, len(fingerprints)):
            if j not in visited and cos_matrix[i, j] > 0.85:
                cluster.append(j)
                visited.add(j)
        clusters.append(cluster)

    for ci, cluster in enumerate(clusters):
        members = [f"{fingerprints[i]['name']} ({fingerprints[i]['graph_type']})" for i in cluster]
        print(f"    Cluster {ci + 1}: {', '.join(members)}")

    # ── Graph type discrimination ──
    print(f"\n  GRAPH TYPE DISCRIMINATION:")
    print(f"  Can KTS distinguish graph construction methods?")

    types = {}
    for fp in fingerprints:
        gt = fp["graph_type"]
        types.setdefault(gt, []).append(fp)

    # For each pair of graph types, compute avg between-type vs within-type similarity
    type_names = list(types.keys())
    for i in range(len(type_names)):
        for j in range(i + 1, len(type_names)):
            t1, t2 = type_names[i], type_names[j]
            fps1 = [fp["fp_vector"] for fp in types[t1]]
            fps2 = [fp["fp_vector"] for fp in types[t2]]

            # Between-type similarities
            between = []
            for v1 in fps1:
                for v2 in fps2:
                    v1n = np.array(v1) / max(np.linalg.norm(v1), 1e-8)
                    v2n = np.array(v2) / max(np.linalg.norm(v2), 1e-8)
                    between.append(float(np.dot(v1n, v2n)))

            # Within-type similarities
            within1 = []
            for a in range(len(fps1)):
                for b in range(a + 1, len(fps1)):
                    v1n = np.array(fps1[a]) / max(np.linalg.norm(fps1[a]), 1e-8)
                    v2n = np.array(fps1[b]) / max(np.linalg.norm(fps1[b]), 1e-8)
                    within1.append(float(np.dot(v1n, v2n)))

            within2 = []
            for a in range(len(fps2)):
                for b in range(a + 1, len(fps2)):
                    v1n = np.array(fps2[a]) / max(np.linalg.norm(fps2[a]), 1e-8)
                    v2n = np.array(fps2[b]) / max(np.linalg.norm(fps2[b]), 1e-8)
                    within2.append(float(np.dot(v1n, v2n)))

            within_all = within1 + within2
            w_avg = np.mean(within_all) if within_all else float('nan')
            b_avg = np.mean(between)
            sep = w_avg - b_avg if within_all else float('nan')

            print(f"    {t1:12s} vs {t2:12s}: within={w_avg:.3f}, between={b_avg:.3f}, separation={sep:+.3f}")

    # ── Key diagnostic features ──
    print(f"\n  DIAGNOSTIC SIGNATURES:")
    for fp in fingerprints:
        gap_mom = fp.get("gap_knn_graph_momentum", 0)
        knn_mom = fp.get("knn_momentum", 0)
        graph_mom = fp.get("graph_momentum", 0)
        knn_tort = fp.get("knn_tortuosity", 0)

        if abs(gap_mom) < 0.05:
            structure = "SPATIAL (edges = feature proximity)"
        elif gap_mom > 0.3:
            structure = "CROSS-FEATURE (edges cross feature boundaries)"
        elif gap_mom > 0.1:
            structure = "MODERATE DIVERGENCE (partial feature alignment)"
        else:
            structure = "MIXED"

        if knn_mom > -0.15:
            smoothness = "ultra-smooth manifold"
        elif knn_mom > -0.35:
            smoothness = "smooth feature space"
        else:
            smoothness = "rough/clustered feature space"

        print(f"    {fp['name']:25s}: {structure} | {smoothness}")

    # Save
    # Clean fingerprints for JSON (remove numpy)
    clean_fps = []
    for fp in fingerprints:
        cfp = {k: v for k, v in fp.items() if k != "fp_vector"}
        cfp["fp_vector"] = [float(x) for x in fp["fp_vector"]]
        clean_fps.append(cfp)

    outfile = OUT / "kts_fingerprint_results.json"
    with open(outfile, "w") as f:
        json.dump({
            "fingerprints": clean_fps,
            "cosine_matrix": cos_matrix.tolist(),
            "names": [fp["name"] for fp in fingerprints],
        }, f, indent=2)

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed:.0f}s")
    print(f"Results saved: {outfile}")
    print("KTS FINGERPRINT TEST COMPLETE")


if __name__ == "__main__":
    main()
