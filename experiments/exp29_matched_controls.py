"""
Experiment 29: Matched Controls (reviewer-requested)
1. Norm-preserved angular shuffle
2. Random targets vs hub targets
3. Cosine vs Euclidean policy
"""
import sys, json, heapq, numpy as np
from collections import defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
np.random.seed(42)

ROOT = Path("E:/PRSM")
OUT = ROOT / "scripts/experiments/results"

print("Loading...", end=" ", flush=True)
with open(ROOT / "data/g1_registry/unified_grains_with_embeddings.json", "r", encoding="utf-8") as f:
    grains = json.load(f)
with open(ROOT / "data/g1_registry/relationship_registry.json", "r", encoding="utf-8") as f:
    rels = json.load(f)

c2i = {}; embs = []; seen = {}; corp = {}
for g in grains:
    c = g.get("concept","").lower(); emb = g.get("embedding"); bc = g.get("bind_count",0)
    sc = g.get("source_corpora", [])
    if c and emb and (c not in seen or bc > seen[c]):
        seen[c] = bc
        if c not in c2i:
            c2i[c] = len(c2i); embs.append(np.array(emb, dtype=np.float32))
            corp[c2i[c]] = set(sc) if isinstance(sc, list) else set()
        else:
            embs[c2i[c]] = np.array(emb, dtype=np.float32)
            corp[c2i[c]] = set(sc) if isinstance(sc, list) else set()

embeddings = np.array(embs); N = len(embeddings)
norms_vec = np.linalg.norm(embeddings, axis=1, keepdims=True)
emb_normed = embeddings / np.clip(norms_vec, 1e-8, None)

adj = [[] for _ in range(N)]
g2c = {}
for g in grains:
    gid = g.get("grain_id",""); c = g.get("concept","").lower()
    if gid and c and c in c2i: g2c[gid] = c
for r in rels:
    ac = g2c.get(r.get("grain_a_id","")); bc_c = g2c.get(r.get("grain_b_id",""))
    if ac and bc_c and ac != bc_c:
        a, b = c2i.get(ac), c2i.get(bc_c)
        if a is not None and b is not None: adj[a].append(b); adj[b].append(a)
for i in range(N): adj[i] = list(set(adj[i]))
degrees = np.array([len(adj[i]) for i in range(N)])

cn = defaultdict(list)
for idx, co in corp.items():
    for c in co: cn[c].append(idx)
vc = [c for c, nodes in cn.items() if len(nodes) >= 20]

pairs = []
attempts = 0
while len(pairs) < 200 and attempts < 10000:
    attempts += 1
    c1 = vc[np.random.randint(len(vc))]; c2 = vc[np.random.randint(len(vc))]
    if c1 == c2: continue
    s = cn[c1][np.random.randint(len(cn[c1]))]; t = cn[c2][np.random.randint(len(cn[c2]))]
    if s == t or not adj[s] or not adj[t]: continue
    pairs.append((s, t))

H = 100
print(f"{N:,} grains, {len(pairs)} pairs")


def run_cosine(emb_n, s, t):
    tv = emb_n[t]; visited = {s}; degs = []
    pq = [(-float(emb_n[v] @ tv), v) for v in adj[s] if v not in visited]
    heapq.heapify(pq)
    found = False
    while pq and len(visited) < H:
        _, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u); degs.append(degrees[u])
        if u == t: found = True; break
        for v in adj[u]:
            if v not in visited: heapq.heappush(pq, (-float(emb_n[v] @ tv), v))
    return found, degs


# CONTROL 1: Angular shuffle
print("\n" + "=" * 60)
print("CONTROL 1: Norm-Preserved Angular Shuffle")
print("=" * 60)

perm = np.random.permutation(N)
shuf_normed = emb_normed[perm]

orig_r, orig_d = 0, []
shuf_r, shuf_d = 0, []
for s, t in pairs:
    f, d = run_cosine(emb_normed, s, t); orig_r += f; orig_d.extend(d)
    f, d = run_cosine(shuf_normed, s, t); shuf_r += f; shuf_d.extend(d)

print(f"  Original:    reach={orig_r/len(pairs)*100:.1f}%, mean_deg={np.mean(orig_d):.1f}")
print(f"  Ang-Shuffled: reach={shuf_r/len(pairs)*100:.1f}%, mean_deg={np.mean(shuf_d):.1f}")
diff = np.mean(shuf_d) - np.mean(orig_d)
print(f"  Degree change: {diff:+.1f} ({'HIGHER with shuffle' if diff > 5 else 'similar'})")


# CONTROL 2: Hub vs peripheral targets
print("\n" + "=" * 60)
print("CONTROL 2: Hub Targets vs Peripheral Targets")
print("=" * 60)

hub_thresh = np.percentile(degrees[degrees > 0], 90)
for label, subset in [
    ("Hub targets (top 10%)", [(s,t) for s,t in pairs if degrees[t] >= hub_thresh]),
    ("Peripheral targets", [(s,t) for s,t in pairs if 0 < degrees[t] < hub_thresh]),
]:
    if not subset: print(f"  {label}: no pairs"); continue
    r_count, d_list = 0, []
    for s, t in subset[:100]:
        f, d = run_cosine(emb_normed, s, t); r_count += f; d_list.extend(d)
    n = min(len(subset), 100)
    print(f"  {label} (N={n}): reach={r_count/n*100:.1f}%, mean_deg={np.mean(d_list):.1f}")


# CONTROL 3: Cosine vs Euclidean
print("\n" + "=" * 60)
print("CONTROL 3: Cosine vs Euclidean Policy")
print("=" * 60)

euc_r, euc_d = 0, []
for s, t in pairs:
    tv = embeddings[t]; visited = {s}
    pq = [(float(np.linalg.norm(embeddings[v] - tv)), v) for v in adj[s] if v not in visited]
    heapq.heapify(pq)
    found = False
    while pq and len(visited) < H:
        _, u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u); euc_d.append(degrees[u])
        if u == t: found = True; break
        for v in adj[u]:
            if v not in visited:
                heapq.heappush(pq, (float(np.linalg.norm(embeddings[v] - tv)), v))
    euc_r += found

print(f"  Cosine:    reach={orig_r/len(pairs)*100:.1f}%, mean_deg={np.mean(orig_d):.1f}")
print(f"  Euclidean: reach={euc_r/len(pairs)*100:.1f}%, mean_deg={np.mean(euc_d):.1f}")
diff_euc = np.mean(euc_d) - np.mean(orig_d)
print(f"  Degree change: {diff_euc:+.1f} ({'HIGHER with Euclidean' if diff_euc > 5 else 'similar'})")

results = {
    "control1_orig_reach": orig_r/len(pairs)*100,
    "control1_shuf_reach": shuf_r/len(pairs)*100,
    "control1_orig_deg": float(np.mean(orig_d)),
    "control1_shuf_deg": float(np.mean(shuf_d)),
    "control3_cos_reach": orig_r/len(pairs)*100,
    "control3_euc_reach": euc_r/len(pairs)*100,
    "control3_cos_deg": float(np.mean(orig_d)),
    "control3_euc_deg": float(np.mean(euc_d)),
}
with open(OUT / "exp29_matched_controls.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT / 'exp29_matched_controls.json'}")
