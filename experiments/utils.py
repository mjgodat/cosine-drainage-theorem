"""Shared utilities for VGSG/KTS experiments."""
import numpy as np
import torch
import json
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def build_knn_gpu(points_np, k=20):
    """Build kNN adjacency on GPU. Returns (N, k) numpy index array."""
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


def compute_momentum(points_np, paths):
    """Compute K_mom for a batch of paths. GPU-accelerated."""
    t = torch.tensor(points_np, device=DEVICE)
    results = []
    for path in paths:
        if len(path) < 3:
            continue
        vecs = t[path]
        steps = vecs[1:] - vecs[:-1]
        norms = steps.norm(dim=1).clamp(min=1e-8)
        cosines = []
        for i in range(len(steps) - 1):
            cos = torch.dot(steps[i], steps[i+1]) / (norms[i] * norms[i+1])
            cosines.append(cos.item())
        results.append(float(np.mean(cosines)) if cosines else 0.0)
    return results


def compute_kts_full(points_np, paths):
    """Compute all KTS metrics for a batch of paths."""
    t = torch.tensor(points_np, device=DEVICE)
    results = []
    for path in paths:
        if len(path) < 3:
            continue
        vecs = t[path]
        n = len(path)

        # Momentum
        steps = vecs[1:] - vecs[:-1]
        norms = steps.norm(dim=1).clamp(min=1e-8)
        cosines = []
        for i in range(len(steps) - 1):
            cos = torch.dot(steps[i], steps[i+1]) / (norms[i] * norms[i+1])
            cosines.append(cos.item())
        momentum = float(np.mean(cosines)) if cosines else 0.0

        # Tortuosity
        chord = (vecs[-1] - vecs[0]).norm().item()
        arc = sum((vecs[i+1] - vecs[i]).norm().item() for i in range(n - 1))
        tortuosity = chord / max(arc, 1e-8)

        # TAV magnitude
        vec_sum = vecs.sum(dim=0)
        tav_mag = vec_sum.norm().item() / n

        # Eccentricity via SVD
        centered = vecs - vecs.mean(dim=0, keepdim=True)
        try:
            _, S, _ = torch.linalg.svd(centered, full_matrices=False)
            s = S[:3].cpu().numpy()
            eccentricity = float(s[0] / (s[1] + s[2] + 1e-8)) if len(s) >= 3 else 0.0
        except Exception:
            eccentricity = 0.0

        # Saturation
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
            "momentum": momentum, "tortuosity": tortuosity,
            "tav_mag": tav_mag, "eccentricity": eccentricity,
            "sat_alpha": sat_alpha,
        })
    return results


def gen_random_paths(N, n_paths=500, path_len=7):
    """Generate random index sequences (no adjacency constraint)."""
    return [list(np.random.choice(N, path_len, replace=False)) for _ in range(n_paths)]


def gen_knn_walks(knn_idx, n_paths=500, path_len=7):
    """Generate kNN local walks."""
    N = len(knn_idx)
    walks = []
    for _ in range(n_paths):
        start = np.random.randint(N)
        path = [start]
        cur = start
        for _ in range(path_len - 1):
            nb = knn_idx[cur]
            nxt = nb[np.random.randint(len(nb))]
            path.append(int(nxt))
            cur = int(nxt)
        walks.append(path)
    return walks


def save_results(data, filename):
    """Save results to JSON."""
    outpath = RESULTS_DIR / filename
    with open(outpath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Results saved: {outpath}")
    return outpath
