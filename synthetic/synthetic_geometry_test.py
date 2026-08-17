"""
Synthetic Geometry Test — Does hubness and radial structure emerge from pure math?

Generates controlled point clouds in R^N with prescribed properties, then measures:
- k-occurrence hubness (Nk skew)
- Radial capture (alpha-curve behavior)
- kNN density distribution
- LID distribution
- Norm distribution

Tests whether the phenomena observed in PRSM/Hetionet/STRING are properties of
high-dimensional geometry itself, not of biology, language, or any specific domain.

Distributions tested:
1. Isotropic Gaussian (uniform density, no structure)
2. Anisotropic Gaussian (stretched along some axes)
3. Mixture of Gaussians (multiple clusters with unequal weights)
4. Power-law norm distribution (some points near origin, most on periphery)
5. L2-normalized hyperspherical (all on unit sphere)
6. Low-intrinsic-dimensional manifold embedded in high N
"""
import sys
import time
import numpy as np
import torch
from scipy.stats import spearmanr, skew
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)
torch.manual_seed(42)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

N_DIM = 768       # match nomic-embed-text
N_POINTS = 10000  # manageable size
K_NN = 50         # for density
K_HUB = 10        # for hubness Nk
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}, Dim: {N_DIM}, Points: {N_POINTS:,}")


def generate_distributions():
    """Generate 6 controlled point clouds."""
    dists = {}

    # 1. Isotropic Gaussian
    dists["1_isotropic"] = np.random.randn(N_POINTS, N_DIM).astype(np.float32)

    # 2. Anisotropic Gaussian (first 50 dims have 10x variance)
    aniso = np.random.randn(N_POINTS, N_DIM).astype(np.float32)
    aniso[:, :50] *= 10.0
    dists["2_anisotropic"] = aniso

    # 3. Mixture of Gaussians (3 clusters, unequal weights: 70%, 20%, 10%)
    mix = np.zeros((N_POINTS, N_DIM), dtype=np.float32)
    centers = [np.random.randn(N_DIM) * 3 for _ in range(3)]
    sizes = [7000, 2000, 1000]
    idx = 0
    for center, size in zip(centers, sizes):
        mix[idx:idx+size] = np.random.randn(size, N_DIM) + center
        idx += size
    dists["3_mixture_unequal"] = mix

    # 4. Power-law norms (directions uniform on sphere, norms follow power law)
    directions = np.random.randn(N_POINTS, N_DIM).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # Power law: P(norm) ~ norm^(-2), so some points very near origin
    norms = (np.random.pareto(1.5, N_POINTS) + 1).astype(np.float32)
    dists["4_powerlaw_norm"] = directions * norms[:, np.newaxis]

    # 5. L2-normalized (all on unit sphere)
    sphere = np.random.randn(N_POINTS, N_DIM).astype(np.float32)
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    dists["5_unit_sphere"] = sphere

    # 6. Low intrinsic dim (50D manifold embedded in 768D)
    low_dim = np.random.randn(N_POINTS, 50).astype(np.float32)
    projection = np.random.randn(50, N_DIM).astype(np.float32) / np.sqrt(50)
    embedded = low_dim @ projection
    # Add small noise in remaining dimensions
    embedded += np.random.randn(N_POINTS, N_DIM).astype(np.float32) * 0.01
    dists["6_low_intrinsic_dim"] = embedded

    return dists


def measure_hubness(points, k=K_HUB, metric="euclidean"):
    """Compute Nk (k-occurrence) for each point. Returns Nk array and skewness."""
    t = torch.tensor(points, device=DEVICE)
    N = len(points)
    nk = np.zeros(N, dtype=np.int32)

    if metric == "cosine":
        t_normed = t / t.norm(dim=1, keepdim=True).clamp(min=1e-8)
        BATCH = 1000
        for start in range(0, N, BATCH):
            end = min(start + BATCH, N)
            sims = t_normed[start:end] @ t_normed.T
            for i in range(end - start):
                sims[i, start + i] = -float('inf')
            _, topk_idx = sims.topk(k, dim=1)
            for i in range(end - start):
                for j in topk_idx[i].cpu().numpy():
                    nk[j] += 1
    else:  # euclidean
        BATCH = 500
        for start in range(0, N, BATCH):
            end = min(start + BATCH, N)
            dists = torch.cdist(t[start:end], t)
            for i in range(end - start):
                dists[i, start + i] = float('inf')
            _, topk_idx = dists.topk(k, dim=1, largest=False)
            for i in range(end - start):
                for j in topk_idx[i].cpu().numpy():
                    nk[j] += 1
    return nk, float(skew(nk))


def measure_density_lid(points, k_density=K_NN, k_lid=20):
    """GPU-accelerated kNN density and LID."""
    t = torch.tensor(points, device=DEVICE)
    N = len(points)
    density = np.zeros(N)
    lid = np.zeros(N)

    t_normed = t / t.norm(dim=1, keepdim=True).clamp(min=1e-8)

    BATCH = 1000
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)

        # Density (cosine kNN)
        sims = t_normed[start:end] @ t_normed.T
        for i in range(end - start):
            sims[i, start + i] = -1
        topk_vals, _ = sims.topk(k_density, dim=1)
        density[start:end] = topk_vals.mean(dim=1).cpu().numpy()

        # LID (Euclidean)
        dists = torch.cdist(t[start:end], t)
        for i in range(end - start):
            dists[i, start + i] = float('inf')
        topk_dists, _ = dists.topk(k_lid, dim=1, largest=False)
        r_K = topk_dists[:, -1:]
        log_ratios = torch.log(r_K / topk_dists[:, :-1].clamp(min=1e-10))
        lid_batch = (k_lid - 1) / log_ratios.sum(dim=1).clamp(min=1e-10)
        lid[start:end] = lid_batch.cpu().numpy()

    return density, lid


def measure_alpha_capture(points, n_traces=200, alphas=[1.0, 2.0, 3.0, 5.0, 10.0]):
    """For random traces of length 5, measure which points get captured at each alpha."""
    t = torch.tensor(points, device=DEVICE)
    N = len(points)
    norms = np.linalg.norm(points, axis=1)

    capture_counts = {a: np.zeros(N, dtype=np.int32) for a in alphas}
    nearest_norms = {a: [] for a in alphas}

    for _ in range(n_traces):
        indices = np.random.choice(N, 5, replace=False)
        trace_vecs = points[indices]
        vec_sum = trace_vecs.sum(axis=0)
        n = len(indices)

        for alpha in alphas:
            scaled = vec_sum * (alpha / n)
            scaled_t = torch.tensor(scaled, device=DEVICE).unsqueeze(0)
            dists = torch.cdist(scaled_t, t).squeeze(0)
            # Exclude trace members
            for idx in indices:
                dists[idx] = float('inf')
            nearest_idx = dists.argmin().item()
            capture_counts[alpha][nearest_idx] += 1
            nearest_norms[alpha].append(norms[nearest_idx])

    return capture_counts, nearest_norms


# ══════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════
print("\nGenerating distributions...")
distributions = generate_distributions()

results = {}

for name, points in distributions.items():
    print(f"\n{'=' * 70}")
    print(f"DISTRIBUTION: {name}")
    print(f"{'=' * 70}")

    norms = np.linalg.norm(points, axis=1)
    print(f"  Norm: mean={norms.mean():.2f}, std={norms.std():.2f}, min={norms.min():.2f}, max={norms.max():.2f}")

    # Hubness
    print("  Computing hubness (Euclidean)...", end=" ", flush=True)
    t0 = time.time()
    nk_euc, skew_euc = measure_hubness(points, metric="euclidean")
    print(f"{time.time()-t0:.1f}s  Nk skew={skew_euc:.3f}")

    print("  Computing hubness (cosine)...", end=" ", flush=True)
    t0 = time.time()
    nk_cos, skew_cos = measure_hubness(points, metric="cosine")
    print(f"{time.time()-t0:.1f}s  Nk skew={skew_cos:.3f}")

    # Density + LID
    print("  Computing density + LID...", end=" ", flush=True)
    t0 = time.time()
    density, lid = measure_density_lid(points)
    print(f"{time.time()-t0:.1f}s")

    # Alpha capture
    print("  Computing alpha capture...", end=" ", flush=True)
    t0 = time.time()
    capture_counts, nearest_norms_alpha = measure_alpha_capture(points)
    print(f"{time.time()-t0:.1f}s")

    # Correlations
    print(f"\n  CORRELATIONS:")
    rho_norm_nk, _ = spearmanr(norms, nk_euc)
    rho_norm_density, _ = spearmanr(norms, density)
    rho_norm_lid, _ = spearmanr(norms, lid)
    rho_density_lid, _ = spearmanr(density, lid)
    rho_nk_density, _ = spearmanr(nk_euc, density)
    rho_nk_norm, _ = spearmanr(nk_euc, norms)
    print(f"    norm vs Nk(euc):      {rho_norm_nk:+.4f}")
    print(f"    norm vs density:      {rho_norm_density:+.4f}")
    print(f"    norm vs LID:          {rho_norm_lid:+.4f}")
    print(f"    density vs LID:       {rho_density_lid:+.4f}")
    print(f"    Nk(euc) vs density:   {rho_nk_density:+.4f}")
    print(f"    Nk(euc) vs norm:      {rho_nk_norm:+.4f}")

    # Alpha capture analysis
    print(f"\n  ALPHA CAPTURE (avg norm of captured point):")
    for alpha in [1.0, 3.0, 5.0, 10.0]:
        avg_captured_norm = np.mean(nearest_norms_alpha[alpha])
        # Top captured point
        top_captured = np.argsort(-capture_counts[alpha])[:5]
        top_norms = norms[top_captured]
        top_counts = capture_counts[alpha][top_captured]
        print(f"    a={alpha:5.1f}  avg_captured_norm={avg_captured_norm:.3f}  "
              f"top5_norms=[{', '.join(f'{n:.1f}' for n in top_norms)}]  "
              f"top5_counts=[{', '.join(str(c) for c in top_counts)}]")

    # Does norm predict alpha capture?
    total_capture = sum(capture_counts[a] for a in capture_counts)
    rho_norm_capture, p_capture = spearmanr(norms, total_capture)
    print(f"\n    norm vs total_capture: rho={rho_norm_capture:+.4f} (p={p_capture:.2e})")

    results[name] = {
        "nk_skew_euc": skew_euc,
        "nk_skew_cos": skew_cos,
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "density_mean": float(density.mean()),
        "lid_mean": float(np.mean(lid[lid > 0])),
        "rho_norm_nk": float(rho_norm_nk),
        "rho_norm_density": float(rho_norm_density),
        "rho_density_lid": float(rho_density_lid),
        "rho_nk_density": float(rho_nk_density),
        "rho_norm_capture": float(rho_norm_capture),
    }

# Summary
print(f"\n{'=' * 70}")
print("SUMMARY: Does hubness emerge from pure geometry?")
print(f"{'=' * 70}")
print(f"\n  {'Distribution':25s} {'Nk skew(E)':>10s} {'Nk skew(C)':>10s} {'norm↔Nk':>8s} {'dens↔LID':>9s} {'norm↔capt':>10s}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8} {'-'*9} {'-'*10}")
for name, r in results.items():
    print(f"  {name:25s} {r['nk_skew_euc']:10.3f} {r['nk_skew_cos']:10.3f} {r['rho_norm_nk']:+8.3f} {r['rho_density_lid']:+9.3f} {r['rho_norm_capture']:+10.3f}")

# Save
import json
with open(OUT / "synthetic_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT / 'synthetic_results.json'}")
print("SYNTHETIC GEOMETRY TEST COMPLETE")
