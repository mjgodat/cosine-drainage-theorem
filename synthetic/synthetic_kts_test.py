"""
Synthetic KTS Test — Do kinematic trajectory metrics separate structured paths
from random walks on pure geometry with ZERO semantic content?

If momentum/alignment/saturation separate on featureless point clouds,
KTS measures geometric properties of path shape, not semantic content.
If they don't separate, KTS needs the semantic manifold to work —
meaning kinematics read real conceptual structure, not just high-D math.

Distributions (from synthetic_geometry_test.py):
  1. Isotropic Gaussian         (uniform, no structure)
  2. Anisotropic Gaussian       (stretched along first 50 dims)
  3. Mixture of 3 Gaussians     (70/20/10 unequal clusters)
  4. Power-law norm             (Pareto norms, uniform directions)
  5. L2-normalized unit sphere  (all on S^767)
  6. Low intrinsic dim          (50D manifold in 768D)

Path types per distribution:
  A. kNN local walks       — stay in local neighborhood (within-cluster)
  B. Cross-structure paths — deliberately cross clusters / norm regions
  C. Pure random sequences — no adjacency constraint at all

KTS metrics computed:
  - Path Momentum (K_mom)   — consecutive displacement cosine
  - Tortuosity              — chord/arc ratio
  - Covariance eccentricity — shape elongation
  - Alpha-scaling saturation — when nearest-neighbor identity stabilizes
  - TAV magnitude           — constructive interference of path vectors

All heavy computation on GPU (torch.cuda). CPU stays cold.
"""
import sys
import time
import json
import numpy as np
import torch
from scipy.stats import mannwhitneyu, spearmanr
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

N_DIM = 768
N_POINTS = 10000
K_NN = 20          # for kNN walks
N_WALKS = 500      # paths per type per distribution
WALK_LEN = 7       # path length (matches typical PRSM trace length)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Config: {N_POINTS:,} points, {N_DIM}D, {N_WALKS} walks x {WALK_LEN} steps")
print(f"Output: {OUT}")


# ══════════════════════════════════════════════════════════════════
# DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════

def generate_distributions():
    dists = {}

    # 1. Isotropic Gaussian
    dists["1_isotropic"] = np.random.randn(N_POINTS, N_DIM).astype(np.float32)

    # 2. Anisotropic Gaussian (first 50 dims 10x variance)
    aniso = np.random.randn(N_POINTS, N_DIM).astype(np.float32)
    aniso[:, :50] *= 10.0
    dists["2_anisotropic"] = aniso

    # 3. Mixture of Gaussians (3 clusters, 70/20/10)
    mix = np.zeros((N_POINTS, N_DIM), dtype=np.float32)
    centers = [np.random.randn(N_DIM).astype(np.float32) * 5 for _ in range(3)]
    sizes = [7000, 2000, 1000]
    labels = np.zeros(N_POINTS, dtype=np.int32)
    idx = 0
    for ci, (center, size) in enumerate(zip(centers, sizes)):
        mix[idx:idx+size] = np.random.randn(size, N_DIM).astype(np.float32) + center
        labels[idx:idx+size] = ci
        idx += size
    dists["3_mixture"] = mix
    # Store cluster labels for cross-cluster path generation
    dists["3_mixture_labels"] = labels

    # 4. Power-law norms
    directions = np.random.randn(N_POINTS, N_DIM).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    norms = (np.random.pareto(1.5, N_POINTS) + 1).astype(np.float32)
    dists["4_powerlaw"] = directions * norms[:, np.newaxis]
    dists["4_powerlaw_norms"] = norms

    # 5. Unit sphere
    sphere = np.random.randn(N_POINTS, N_DIM).astype(np.float32)
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    dists["5_sphere"] = sphere

    # 6. Low intrinsic dim (50D in 768D)
    low_dim = np.random.randn(N_POINTS, 50).astype(np.float32)
    projection = np.random.randn(50, N_DIM).astype(np.float32) / np.sqrt(50)
    embedded = low_dim @ projection
    embedded += np.random.randn(N_POINTS, N_DIM).astype(np.float32) * 0.01
    dists["6_low_dim"] = embedded

    return dists


# ══════════════════════════════════════════════════════════════════
# GPU kNN (builds adjacency for walks)
# ══════════════════════════════════════════════════════════════════

def build_knn_gpu(points_np, k=K_NN):
    """Build kNN adjacency on GPU. Returns (N, k) index array."""
    t0 = time.time()
    t = torch.tensor(points_np, device=DEVICE)
    N = len(points_np)
    knn_idx = torch.zeros(N, k, dtype=torch.long, device=DEVICE)

    BATCH = 512
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        # Euclidean distances
        dists = torch.cdist(t[start:end], t)  # (batch, N)
        # Mask self
        for i in range(end - start):
            dists[i, start + i] = float('inf')
        _, topk = dists.topk(k, dim=1, largest=False)
        knn_idx[start:end] = topk

    result = knn_idx.cpu().numpy()
    print(f"    kNN built: {time.time()-t0:.1f}s", flush=True)
    return result


# ══════════════════════════════════════════════════════════════════
# PATH GENERATORS
# ══════════════════════════════════════════════════════════════════

def generate_knn_walks(knn_idx, n_walks=N_WALKS, walk_len=WALK_LEN):
    """Type A: local kNN walks — stay in neighborhood."""
    N = len(knn_idx)
    walks = []
    for _ in range(n_walks):
        start = np.random.randint(N)
        path = [start]
        current = start
        for _ in range(walk_len - 1):
            neighbors = knn_idx[current]
            nxt = neighbors[np.random.randint(len(neighbors))]
            path.append(int(nxt))
            current = int(nxt)
        walks.append(path)
    return walks


def generate_cross_cluster_paths(labels, knn_idx, n_walks=N_WALKS, walk_len=WALK_LEN):
    """Type B (mixture only): paths that deliberately cross cluster boundaries."""
    unique_labels = np.unique(labels)
    cluster_indices = {l: np.where(labels == l)[0] for l in unique_labels}
    walks = []

    for _ in range(n_walks):
        path = []
        # Alternate clusters: pick a point from each cluster in rotation
        cluster_order = np.random.permutation(unique_labels)
        for step in range(walk_len):
            ci = cluster_order[step % len(cluster_order)]
            pool = cluster_indices[ci]
            if path:
                # Pick the kNN neighbor in the target cluster closest to current
                # Use kNN of last point, filter to target cluster
                last = path[-1]
                neighbors = knn_idx[last]
                in_cluster = [n for n in neighbors if labels[n] == ci]
                if in_cluster:
                    path.append(int(in_cluster[0]))
                else:
                    # No kNN neighbor in target cluster — pick random from cluster
                    path.append(int(np.random.choice(pool)))
            else:
                path.append(int(np.random.choice(pool)))
        walks.append(path)
    return walks


def generate_norm_crossing_paths(norms, n_walks=N_WALKS, walk_len=WALK_LEN):
    """Type B (power-law only): paths that cross from low-norm to high-norm."""
    sorted_idx = np.argsort(norms)
    N = len(norms)
    walks = []
    for _ in range(n_walks):
        # Pick evenly spaced quantile positions from low to high norm
        positions = np.linspace(0, N - 1, walk_len).astype(int)
        # Add jitter
        jitter = np.random.randint(-max(N // 50, 1), max(N // 50, 1) + 1, size=walk_len)
        positions = np.clip(positions + jitter, 0, N - 1)
        path = [int(sorted_idx[p]) for p in positions]
        walks.append(path)
    return walks


def generate_antipodal_paths(points_np, n_walks=N_WALKS, walk_len=WALK_LEN):
    """Type B (sphere/isotropic/anisotropic/low-dim): paths between distant points."""
    t = torch.tensor(points_np, device=DEVICE)
    t_normed = t / t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    N = len(points_np)
    walks = []

    for _ in range(n_walks):
        # Pick a random start
        start = np.random.randint(N)
        # Find a distant point (low cosine similarity)
        sims = (t_normed[start:start+1] @ t_normed.T).squeeze(0)
        # Pick from bottom 5% similarity
        bottom_k = max(N // 20, 1)
        _, far_idx = sims.topk(bottom_k, largest=False)
        end = int(far_idx[np.random.randint(len(far_idx))].item())

        # Interpolate between start and end, snapping to nearest real points
        path = [start]
        for step in range(1, walk_len - 1):
            alpha = step / (walk_len - 1)
            interp = (1 - alpha) * t[start] + alpha * t[end]
            dists = torch.norm(t - interp.unsqueeze(0), dim=1)
            # Exclude already-used points
            for used in path:
                dists[used] = float('inf')
            nearest = int(dists.argmin().item())
            path.append(nearest)
        path.append(end)
        walks.append(path)
    return walks


def generate_random_sequences(N, n_walks=N_WALKS, walk_len=WALK_LEN):
    """Type C: pure random index sequences — no adjacency, no structure."""
    walks = []
    for _ in range(n_walks):
        path = list(np.random.choice(N, walk_len, replace=False))
        walks.append(path)
    return walks


# ══════════════════════════════════════════════════════════════════
# KTS METRICS (GPU-accelerated where possible)
# ══════════════════════════════════════════════════════════════════

def compute_kts_batch(points_np, walks):
    """Compute KTS metrics for a batch of walks. GPU for vector ops."""
    t = torch.tensor(points_np, device=DEVICE)
    results = []

    for path in walks:
        if len(path) < 3:
            continue

        vecs = t[path]  # (walk_len, N_DIM) on GPU
        n = len(path)

        # ── Path Momentum ──
        steps = vecs[1:] - vecs[:-1]  # (n-1, D)
        step_norms = steps.norm(dim=1).clamp(min=1e-8)
        step_cosines = []
        for i in range(len(steps) - 1):
            cos = torch.dot(steps[i], steps[i+1]) / (step_norms[i] * step_norms[i+1])
            step_cosines.append(cos.item())
        momentum = float(np.mean(step_cosines)) if step_cosines else 0.0

        # ── Tortuosity (chord / arc) ──
        chord = (vecs[-1] - vecs[0]).norm().item()
        arc = sum((vecs[i+1] - vecs[i]).norm().item() for i in range(n - 1))
        tortuosity = chord / max(arc, 1e-8)

        # ── TAV magnitude (constructive interference) ──
        vec_sum = vecs.sum(dim=0)
        tav_mag = vec_sum.norm().item() / n

        # ── Covariance eccentricity (in full D, GPU) ──
        centered = vecs - vecs.mean(dim=0, keepdim=True)
        # Compute top 3 singular values via SVD on centered matrix
        # This is cheaper than full covariance in high-D
        try:
            _, S, _ = torch.linalg.svd(centered, full_matrices=False)
            s = S[:3].cpu().numpy()
            eccentricity = float(s[0] / (s[1] + s[2] + 1e-8)) if len(s) >= 3 else 0.0
        except Exception:
            eccentricity = 0.0

        # ── Alpha-scaling saturation ──
        # Project scaled TAV outward, find nearest point (excluding path members)
        path_set = set(path)
        alphas = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        prev_nearest = -1
        sat_alpha = alphas[-1]
        for alpha in alphas:
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
            "momentum": momentum,
            "tortuosity": tortuosity,
            "tav_mag": tav_mag,
            "eccentricity": eccentricity,
            "sat_alpha": sat_alpha,
        })

    return results


# ══════════════════════════════════════════════════════════════════
# STATISTICAL COMPARISON
# ══════════════════════════════════════════════════════════════════

def compare_groups(groups, metric_names=None):
    """Print comparison table + p-values between first group and all others."""
    if metric_names is None:
        metric_names = ["momentum", "tortuosity", "tav_mag", "eccentricity", "sat_alpha"]

    group_names = list(groups.keys())
    print(f"\n  {'Group':25s} {'n':>5s}", end="")
    for m in metric_names:
        print(f" {m:>12s}", end="")
    print()
    print(f"  {'-'*25} {'-'*5}", end="")
    for _ in metric_names:
        print(f" {'-'*12}", end="")
    print()

    for name in group_names:
        items = groups[name]
        if not items:
            print(f"  {name:25s} {0:5d}  (empty)")
            continue
        print(f"  {name:25s} {len(items):5d}", end="")
        for m in metric_names:
            vals = [x[m] for x in items]
            print(f" {np.mean(vals):7.4f}±{np.std(vals):.3f}", end="")
        print()

    # P-values: each structured group vs random
    if len(group_names) >= 2:
        baseline_name = group_names[-1]  # random is always last
        baseline = groups[baseline_name]
        if baseline:
            print(f"\n  P-VALUES vs {baseline_name}:")
            print(f"  {'Group':25s}", end="")
            for m in metric_names:
                print(f" {m:>12s}", end="")
            print()
            for name in group_names[:-1]:
                items = groups[name]
                if not items:
                    continue
                print(f"  {name:25s}", end="")
                for m in metric_names:
                    rv = [x[m] for x in items]
                    nv = [x[m] for x in baseline]
                    try:
                        _, p = mannwhitneyu(rv, nv, alternative="two-sided")
                        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                        print(f" {p:>8.2e}{sig:>3s}", end="")
                    except Exception:
                        print(f" {'err':>12s}", end="")
                print()


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("\nGenerating distributions...")
    dists = generate_distributions()

    all_results = {}

    # Define which distributions get which path types
    test_configs = [
        ("1_isotropic",   None, None),
        ("2_anisotropic", None, None),
        ("3_mixture",     "cluster", dists.get("3_mixture_labels")),
        ("4_powerlaw",    "norm",    dists.get("4_powerlaw_norms")),
        ("5_sphere",      None, None),
        ("6_low_dim",     None, None),
    ]

    for dist_name, cross_type, cross_data in test_configs:
        points = dists[dist_name]
        N = len(points)
        print(f"\n{'=' * 70}")
        print(f"DISTRIBUTION: {dist_name} ({N:,} points, {N_DIM}D)")
        print(f"{'=' * 70}")

        norms = np.linalg.norm(points, axis=1)
        print(f"  Norms: mean={norms.mean():.2f}, std={norms.std():.2f}")

        # Build kNN
        print("  Building kNN adjacency...", flush=True)
        knn_idx = build_knn_gpu(points, k=K_NN)

        # Generate paths
        print(f"  Generating {N_WALKS} paths per type...", flush=True)

        paths = {}
        paths["A: kNN local"] = generate_knn_walks(knn_idx)
        paths["C: random"] = generate_random_sequences(N)

        if cross_type == "cluster":
            paths["B: cross-cluster"] = generate_cross_cluster_paths(cross_data, knn_idx)
        elif cross_type == "norm":
            paths["B: norm-crossing"] = generate_norm_crossing_paths(cross_data)
        else:
            paths["B: antipodal"] = generate_antipodal_paths(points)

        # Compute KTS metrics
        groups = {}
        for path_type, walk_list in paths.items():
            print(f"  Computing KTS: {path_type}...", end=" ", flush=True)
            t0 = time.time()
            metrics = compute_kts_batch(points, walk_list)
            print(f"{time.time()-t0:.1f}s ({len(metrics)} paths)")
            groups[path_type] = metrics

        # Compare
        compare_groups(groups)

        # Store summary
        dist_summary = {}
        for gname, items in groups.items():
            if not items:
                continue
            dist_summary[gname] = {
                "n": len(items),
                "momentum_mean": float(np.mean([x["momentum"] for x in items])),
                "momentum_std": float(np.std([x["momentum"] for x in items])),
                "tortuosity_mean": float(np.mean([x["tortuosity"] for x in items])),
                "tortuosity_std": float(np.std([x["tortuosity"] for x in items])),
                "tav_mag_mean": float(np.mean([x["tav_mag"] for x in items])),
                "tav_mag_std": float(np.std([x["tav_mag"] for x in items])),
                "eccentricity_mean": float(np.mean([x["eccentricity"] for x in items])),
                "eccentricity_std": float(np.std([x["eccentricity"] for x in items])),
                "sat_alpha_mean": float(np.mean([x["sat_alpha"] for x in items])),
                "sat_alpha_std": float(np.std([x["sat_alpha"] for x in items])),
            }

            # P-values vs random
            random_items = groups.get("C: random", [])
            if random_items and gname != "C: random":
                pvals = {}
                for m in ["momentum", "tortuosity", "tav_mag", "eccentricity", "sat_alpha"]:
                    rv = [x[m] for x in items]
                    nv = [x[m] for x in random_items]
                    try:
                        _, p = mannwhitneyu(rv, nv, alternative="two-sided")
                        pvals[f"p_{m}_vs_random"] = float(p)
                    except Exception:
                        pvals[f"p_{m}_vs_random"] = None
                dist_summary[gname].update(pvals)

        all_results[dist_name] = dist_summary

    # ── CROSS-DISTRIBUTION SUMMARY ──
    print(f"\n{'=' * 70}")
    print("CROSS-DISTRIBUTION SUMMARY")
    print(f"{'=' * 70}")
    print(f"\nDoes KTS separate structured from random on pure geometry?")
    print(f"\n  {'Distribution':20s} {'Mom(local)':>12s} {'Mom(cross)':>12s} {'Mom(rand)':>12s} {'Sep?':>6s}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*6}")

    for dist_name in all_results:
        r = all_results[dist_name]
        local = r.get("A: kNN local", {})
        rand = r.get("C: random", {})
        # Find the cross-structure group (B: whatever)
        cross_key = [k for k in r if k.startswith("B:")][0] if any(k.startswith("B:") for k in r) else None
        cross = r.get(cross_key, {}) if cross_key else {}

        m_local = f"{local.get('momentum_mean', 0):.4f}" if local else "N/A"
        m_cross = f"{cross.get('momentum_mean', 0):.4f}" if cross else "N/A"
        m_rand = f"{rand.get('momentum_mean', 0):.4f}" if rand else "N/A"

        # Check if momentum separates (p < 0.05 for cross vs random)
        p_val = cross.get("p_momentum_vs_random") if cross else None
        sep = "YES***" if p_val and p_val < 0.001 else "YES*" if p_val and p_val < 0.05 else "no" if p_val else "?"

        print(f"  {dist_name:20s} {m_local:>12s} {m_cross:>12s} {m_rand:>12s} {sep:>6s}")

    # Save
    outfile = OUT / "synthetic_kts_results.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s")
    print(f"Results saved: {outfile}")
    print("SYNTHETIC KTS TEST COMPLETE")


if __name__ == "__main__":
    main()
