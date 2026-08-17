"""
Experiment 3: L2-Normalization Ablation

Does KTS separation survive after projecting all vectors to the unit sphere?
If YES: kinematics are purely angular (direction-based)
If NO: kinematics depend on norm variance (norm-based)
"""
import sys
import time
import numpy as np
import torch
import json
from pathlib import Path
from scipy.stats import mannwhitneyu

sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

N_WALKS = 500
WALK_LEN = 7
K_NN = 20

print(f"Device: {DEVICE}")


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
        norms = steps.norm(dim=1).clamp(min=1e-8)
        cosines = [torch.dot(steps[i], steps[i+1]) / (norms[i] * norms[i+1])
                   for i in range(len(steps) - 1)]
        momentum = float(np.mean([c.item() for c in cosines])) if cosines else 0.0
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
        results.append({"momentum": momentum, "tortuosity": tortuosity,
                        "tav_mag": tav_mag, "eccentricity": eccentricity})
    return results


def gen_knn_walks(knn_idx, n=N_WALKS, wl=WALK_LEN):
    N = len(knn_idx)
    walks = []
    for _ in range(n):
        start = np.random.randint(N)
        path = [start]; cur = start
        for _ in range(wl - 1):
            nb = knn_idx[cur]; nxt = nb[np.random.randint(len(nb))]
            path.append(int(nxt)); cur = int(nxt)
        walks.append(path)
    return walks


def gen_random(N, n=N_WALKS, wl=WALK_LEN):
    return [list(np.random.choice(N, wl, replace=False)) for _ in range(n)]


def run_ablation(name, features):
    N, D = features.shape
    print(f"\n{'=' * 70}")
    print(f"GRAPH: {name} ({N:,} nodes, {D}D)")
    print(f"{'=' * 70}")

    norms_raw = np.linalg.norm(features, axis=1, keepdims=True)
    features_l2 = features / np.clip(norms_raw, 1e-8, None)

    result = {"name": name, "N": N, "D": D}

    for label, feats in [("raw", features), ("L2-normalized", features_l2)]:
        print(f"\n  --- {label} ---")
        knn = build_knn_gpu(feats, k=K_NN)
        knn_walks = gen_knn_walks(knn)
        rand_walks = gen_random(N)

        kts_knn = compute_kts_batch(feats, knn_walks)
        kts_rand = compute_kts_batch(feats, rand_walks)

        metrics = ["momentum", "tortuosity", "tav_mag", "eccentricity"]
        print(f"  {'Metric':>15s} {'kNN mean':>10s} {'Random mean':>12s} {'p-value':>12s} {'Separates?':>12s}")
        for m in metrics:
            v_knn = [x[m] for x in kts_knn]
            v_rand = [x[m] for x in kts_rand]
            _, p = mannwhitneyu(v_knn, v_rand, alternative="two-sided")
            sep = "YES" if p < 0.001 else "marginal" if p < 0.05 else "NO"
            print(f"  {m:>15s} {np.mean(v_knn):10.4f} {np.mean(v_rand):12.4f} {p:12.2e} {sep:>12s}")
            result[f"{label}_{m}_knn"] = float(np.mean(v_knn))
            result[f"{label}_{m}_rand"] = float(np.mean(v_rand))
            result[f"{label}_{m}_p"] = float(p)

    print(f"\n  ABLATION VERDICT:")
    for m in ["momentum", "tortuosity", "eccentricity"]:
        p_raw = result.get(f"raw_{m}_p", 1)
        p_l2 = result.get(f"L2-normalized_{m}_p", 1)
        if p_raw < 0.001 and p_l2 < 0.001:
            print(f"    {m:>15s}: SURVIVES L2 normalization (angular, not norm-based)")
        elif p_raw < 0.001 and p_l2 >= 0.001:
            print(f"    {m:>15s}: LOST after L2 (was norm-dependent)")
        else:
            print(f"    {m:>15s}: raw p={p_raw:.2e}, L2 p={p_l2:.2e}")

    return result


def main():
    t_total = time.time()
    all_results = []

    from torch_geometric.datasets import Planetoid, Amazon, CitationFull

    for cls, kwargs, name in [
        (Planetoid, {"root": "/tmp/pyg_data", "name": "Cora"}, "Cora"),
        (Amazon, {"root": "/tmp/pyg_data", "name": "Computers"}, "Amazon Computers"),
        (CitationFull, {"root": "/tmp/pyg_data", "name": "DBLP"}, "DBLP"),
    ]:
        d = cls(**kwargs); data = d[0]
        all_results.append(run_ablation(name, data.x.numpy().astype(np.float32)))

    iso = np.random.randn(10000, 768).astype(np.float32)
    all_results.append(run_ablation("Synthetic Isotropic", iso))

    print(f"\n{'=' * 70}")
    print("CROSS-GRAPH L2 ABLATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Graph':>25s} {'Mom raw p':>12s} {'Mom L2 p':>12s} {'Survives?':>10s}")
    for r in all_results:
        p_raw = r.get("raw_momentum_p", 1)
        p_l2 = r.get("L2-normalized_momentum_p", 1)
        surv = "YES" if p_raw < 0.001 and p_l2 < 0.001 else "NO"
        print(f"  {r['name']:>25s} {p_raw:12.2e} {p_l2:12.2e} {surv:>10s}")

    with open(OUT / "exp3_l2_ablation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {OUT / 'exp3_l2_ablation.json'}")
    print(f"Total time: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
