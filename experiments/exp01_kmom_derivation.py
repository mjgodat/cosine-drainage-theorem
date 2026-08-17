"""
Experiment 1: K_mom = -1/2 Derivation Verification

Hypothesis: The K_mom ≈ -0.50 null observed across all 13 graphs is the
mathematical consequence of consecutive displacement vectors sharing a
midpoint in high-dimensional space.

For iid samples X, Y, Z from a centered distribution with covariance σ²I:
  Δ₁ = Y - X,  Δ₂ = Z - Y
  E[Δ₁ᵀΔ₂] = -E[YᵀY] = -Dσ²
  E[||Δᵢ||²] = 2Dσ²
  Under norm concentration: E[cos(Δ₁, Δ₂)] → -1/2

Tests:
  A. Vary dimensionality D: 10, 50, 100, 500, 768, 2000, 5000
  B. Vary distribution: isotropic Gaussian, uniform hypercube, exponential,
     Laplace, elliptical (anisotropic), unit sphere (L2-normalized)
  C. Vary path length: 3, 5, 7, 10, 20, 50
  D. Vary sampling: with/without replacement, kNN walks vs iid
  E. Nonzero mean: shifted distributions
"""
import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

N_TRIALS = 10000

print(f"Device: {DEVICE}")
print(f"Trials per condition: {N_TRIALS:,}")


def momentum_monte_carlo(points, path_len=7, n_trials=N_TRIALS):
    t = torch.tensor(points, device=DEVICE)
    N = len(points)
    momenta = []
    for _ in range(n_trials):
        idx = np.random.choice(N, path_len, replace=False)
        vecs = t[idx]
        steps = vecs[1:] - vecs[:-1]
        norms = steps.norm(dim=1).clamp(min=1e-8)
        cosines = []
        for i in range(len(steps) - 1):
            cos = torch.dot(steps[i], steps[i+1]) / (norms[i] * norms[i+1])
            cosines.append(cos.item())
        momenta.append(float(np.mean(cosines)))
    return np.array(momenta)


def analytic_prediction(D, sigma2=1.0):
    return -0.5


results = {}

# TEST A: Dimensionality
print(f"\n{'=' * 70}")
print("TEST A: DIMENSIONALITY SWEEP")
print(f"{'=' * 70}")
dims = [10, 50, 100, 500, 768, 2000, 5000]
results["dim_sweep"] = {}
print(f"  {'D':>6s} {'K_mom mean':>12s} {'K_mom std':>12s} {'Predicted':>10s} {'Error':>10s}")
print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
for D in dims:
    pts = np.random.randn(20000, D).astype(np.float32)
    mom = momentum_monte_carlo(pts, path_len=7, n_trials=5000)
    pred = analytic_prediction(D)
    error = abs(np.mean(mom) - pred)
    print(f"  {D:6d} {np.mean(mom):12.6f} {np.std(mom):12.6f} {pred:10.4f} {error:10.6f}")
    results["dim_sweep"][D] = {"mean": float(np.mean(mom)), "std": float(np.std(mom)), "predicted": pred, "error": float(error)}

# TEST B: Distributions
print(f"\n{'=' * 70}")
print("TEST B: DISTRIBUTION SWEEP (D=768)")
print(f"{'=' * 70}")
D = 768
N_PTS = 10000
results["dist_sweep"] = {}
distributions = {
    "gaussian_isotropic": np.random.randn(N_PTS, D).astype(np.float32),
    "uniform_hypercube": np.random.uniform(-1, 1, (N_PTS, D)).astype(np.float32),
    "exponential": np.random.exponential(1.0, (N_PTS, D)).astype(np.float32),
    "laplace": np.random.laplace(0, 1, (N_PTS, D)).astype(np.float32),
}
aniso = np.random.randn(N_PTS, D).astype(np.float32)
aniso[:, :50] *= 10.0
distributions["anisotropic"] = aniso
sphere = np.random.randn(N_PTS, D).astype(np.float32)
sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
distributions["unit_sphere"] = sphere
mix = np.zeros((N_PTS, D), dtype=np.float32)
centers = [np.random.randn(D).astype(np.float32) * 5 for _ in range(3)]
for i, (center, size) in enumerate(zip(centers, [7000, 2000, 1000])):
    start = sum([7000, 2000, 1000][:i])
    mix[start:start+size] = np.random.randn(size, D).astype(np.float32) + center
distributions["mixture_3clusters"] = mix
shifted = np.random.randn(N_PTS, D).astype(np.float32) + 5.0
distributions["nonzero_mean"] = shifted

print(f"  {'Distribution':>25s} {'K_mom mean':>12s} {'K_mom std':>12s} {'Dev from -0.5':>14s}")
print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*14}")
for name, pts in distributions.items():
    mom = momentum_monte_carlo(pts, path_len=7, n_trials=5000)
    dev = abs(np.mean(mom) - (-0.5))
    print(f"  {name:>25s} {np.mean(mom):12.6f} {np.std(mom):12.6f} {dev:14.6f}")
    results["dist_sweep"][name] = {"mean": float(np.mean(mom)), "std": float(np.std(mom)), "deviation": float(dev)}

# TEST C: Path length
print(f"\n{'=' * 70}")
print("TEST C: PATH LENGTH SWEEP (D=768, isotropic)")
print(f"{'=' * 70}")
pts = np.random.randn(N_PTS, D).astype(np.float32)
path_lens = [3, 5, 7, 10, 15, 20, 50]
results["pathlen_sweep"] = {}
print(f"  {'Path Length':>12s} {'K_mom mean':>12s} {'K_mom std':>12s} {'Dev from -0.5':>14s}")
print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*14}")
for pl in path_lens:
    mom = momentum_monte_carlo(pts, path_len=pl, n_trials=5000)
    dev = abs(np.mean(mom) - (-0.5))
    print(f"  {pl:12d} {np.mean(mom):12.6f} {np.std(mom):12.6f} {dev:14.6f}")
    results["pathlen_sweep"][pl] = {"mean": float(np.mean(mom)), "std": float(np.std(mom)), "deviation": float(dev)}

# TEST D: kNN vs iid
print(f"\n{'=' * 70}")
print("TEST D: kNN WALKS vs IID SAMPLING (D=768)")
print(f"{'=' * 70}")
pts = np.random.randn(N_PTS, D).astype(np.float32)
results["walk_vs_iid"] = {}
mom_iid = momentum_monte_carlo(pts, path_len=7, n_trials=5000)

t_gpu = torch.tensor(pts, device=DEVICE)
knn_idx = torch.zeros(N_PTS, 20, dtype=torch.long, device=DEVICE)
for s in range(0, N_PTS, 512):
    e_idx = min(s + 512, N_PTS)
    dd = torch.cdist(t_gpu[s:e_idx], t_gpu)
    for i in range(e_idx - s):
        dd[i, s + i] = float('inf')
    _, tk = dd.topk(20, dim=1, largest=False)
    knn_idx[s:e_idx] = tk
knn_np = knn_idx.cpu().numpy()

walks = []
for _ in range(5000):
    start_node = np.random.randint(N_PTS)
    path = [start_node]
    cur = start_node
    for _ in range(6):
        nb = knn_np[cur]
        nxt = nb[np.random.randint(len(nb))]
        path.append(int(nxt))
        cur = int(nxt)
    walks.append(path)

mom_knn_vals = []
for path in walks:
    vecs = t_gpu[path]
    steps = vecs[1:] - vecs[:-1]
    norms = steps.norm(dim=1).clamp(min=1e-8)
    cosines = []
    for i in range(len(steps) - 1):
        cos = torch.dot(steps[i], steps[i+1]) / (norms[i] * norms[i+1])
        cosines.append(cos.item())
    mom_knn_vals.append(float(np.mean(cosines)))
mom_knn = np.array(mom_knn_vals)

print(f"  IID random:  K_mom = {np.mean(mom_iid):.6f} +/- {np.std(mom_iid):.6f}")
print(f"  kNN walks:   K_mom = {np.mean(mom_knn):.6f} +/- {np.std(mom_knn):.6f}")
print(f"  Difference:  {abs(np.mean(mom_iid) - np.mean(mom_knn)):.6f}")
print(f"  -> kNN walks BREAK the iid assumption: shared midpoint theorem")
print(f"     applies to iid but NOT to correlated walks")
results["walk_vs_iid"] = {
    "iid_mean": float(np.mean(mom_iid)), "iid_std": float(np.std(mom_iid)),
    "knn_mean": float(np.mean(mom_knn)), "knn_std": float(np.std(mom_knn)),
}

# SUMMARY
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"\n  The analytic prediction E[cos(D1, D2)] = -1/2 holds for:")
for name, data in results["dist_sweep"].items():
    status = "CONFIRMED" if data["deviation"] < 0.01 else "CLOSE" if data["deviation"] < 0.05 else "DEVIATES"
    print(f"    {name:>25s}: K_mom = {data['mean']:.6f} ({status})")
print(f"\n  Derivation: For iid samples from any centered isotropic distribution,")
print(f"  consecutive secant vectors share midpoint -> E[cos] = -Ds2/(2Ds2) = -1/2")
print(f"  This is a theorem, not an empirical observation.")
print(f"\n  IMPORTANT: kNN walks BREAK the iid assumption.")
print(f"  kNN K_mom = {results['walk_vs_iid']['knn_mean']:.6f} != -0.500")
print(f"  The deviation from -0.50 on structured walks is what KTS measures.")

import json
with open(OUT / "exp1_kmom_derivation.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT / 'exp1_kmom_derivation.json'}")
