# Why Cosine Search Gets Trapped: The Cosine Drainage Theorem

**Cosine similarity is degree-blind. Degree-blind selection from a hub regresses to the mean. The mean is the periphery. The periphery has no exits.**

---

## The Finding

In high-dimensional embedded graphs, cosine-similarity-biased traversal under finite budgets systematically drains into structurally sparse regions, missing graph-reachable targets at rates up to 82% — even when those targets are only 4 hops away.

This is not a bug in any specific system. It is a mathematical consequence of how cosine interacts with degree distributions in high-dimensional space. We call it the **Cosine Drainage Theorem**.

### The Mechanism: Directional Inverse of the Friendship Paradox

The [friendship paradox](https://en.wikipedia.org/wiki/Friendship_paradox) (Feld 1991) says: if you sample a random edge, the neighbor has higher expected degree than average. Uniform sampling biases **toward** hubs.

We prove the inverse: **directional sampling (cosine argmax) biases away from hubs.** Cosine is purely angular — it ignores vector norm. In high-D embeddings, hub nodes point toward the distributional mean direction (they represent general concepts trained in many contexts). A specific target direction selects AWAY from the mean, therefore away from hubs. Each step lands on a lower-degree node. The search drains into the periphery.

### What This Means for RAG and GraphRAG

If your system:
1. Embeds nodes in high-dimensional space
2. Seeds retrieval from cosine-nearest neighbors
3. Expands via graph traversal with a finite budget

...then it is subject to the Cosine Drainage Theorem. Standard countermeasures make it worse:
- **MMR (diversity penalties):** drops reachability from 77% to 12% by avoiding the hub backbone
- **Personalized PageRank:** identical to BFS on unweighted graphs (2.4% reachability)
- **Graph densification:** deepens the basins without moving the entry point

The fix: **dual-frontier or waypoint-injection architectures** that seed from both endpoints or from known intermediates, bypassing the drainage cascade entirely.

---

## Key Results

| Finding | Evidence |
|---------|----------|
| Cosine-greedy misses 82% of targets at graph distance 4 | Exp 19, 5 graphs |
| Phase transition at d_drain ≈ 3.5 hops | Predicted by theorem, confirmed empirically |
| Multi-anchor improves reachability by +338% at low budgets | Exp 9, 4 graphs |
| Waypoint injection achieves 100% intermediate discovery vs 13% for endpoint-only | Exp 20, McNemar p = 9.5×10⁻⁷ |
| Policy ordering (Multi > Cosine > BFS) holds across 4 feature spaces | Exp 21-22, 6 configurations |
| All findings are encoder-agnostic (nomic 768D, BGE 384D, native BoW) | Exp 25-26 |
| Predictive model: R² = 0.92 with 3 features (Γ, log H, degree CV) | Exp 15, leave-one-out ρ = 0.924 |
| Progressive drainage confirmed on all 5 tested graphs | Exp 27, T1+T3 universal |
| The mechanism is the directional inverse of the friendship paradox | Formal proof + empirical confirmation |

---

## Repository Structure

```
cosine-drainage-theorem/
├── README.md                   # This file
├── theorem/
│   └── cosine_drainage_theorem.md   # The formal theorem (5 assumptions, 5 parts, 4 corollaries)
├── proofs/
│   ├── proof_drainage_rate.md       # Part (i): E[d_{i+1}] = ρ_eff + α(d_i - ρ_eff)
│   ├── proof_miss_probability.md    # Part (ii): scissors effect, phase transition
│   ├── proof_unified_theorem.md     # Parts (iii-v): unified statement
│   ├── proof_progressive_drainage.md # Directional inverse of friendship paradox
│   ├── proof_drainage_round2.md     # Refinement: variance reconciliation
│   └── proof_miss_round2.md         # Refinement: closed-form miss probability
├── gaps/
│   ├── gap1_rho_eff.md              # A3 violation: β = +0.05 to +0.19
│   ├── gap2_clustering.md           # Tree approximation justified for cosine
│   ├── gap3_alignment.md            # p_align via scissors ratio
│   ├── gap4_a3_verification.md      # Encoder-independent verification
│   └── gap5_waypoint_spacing.md     # W* = min(d_drain, 1/bottleneck_density)
├── papers/
│   ├── vgsg_thesis_paper.md         # Full thesis paper
│   ├── vgsg_conjecture.md           # The VGSG conjecture document
│   └── retraction_orc_proxy.md      # ORC proxy retraction (transparency)
├── experiments/
│   ├── utils.py                     # Shared GPU utilities
│   ├── exp01_kmom_derivation.py     # K_mom = -1/2 theorem verification
│   ├── exp02_local_gamma.py         # Local Γ_k trapping prediction
│   ├── exp03_l2_ablation.py         # L2 normalization ablation
│   ├── exp06_cosine_intervention.py # Cosine-seeded trapping demonstration
│   ├── exp07_graph_taxonomy.py      # Graph classification + curvature
│   ├── exp08_wrong_gradient.py      # Dimensionality sweep
│   ├── exp09_policy_benchmark.py    # 6-policy benchmark (headline result)
│   ├── exp10_frontier_telemetry.py  # Degree 86 vs 293 mechanism proof
│   ├── exp11_ablation_matrix.py     # Geometry vs topology decoupling
│   ├── exp12_stratified.py          # Causal stratification (ρ=0.141)
│   ├── exp13_cross_graph.py         # Multi-Anchor wins 4/4 graphs
│   ├── exp14_exact_orc.py           # ORC retraction (exact W₁)
│   ├── exp15_predictive.py          # R²=0.92 scaling model
│   ├── exp17_mechanism.py           # P1-P5 falsification tests
│   ├── exp18_cross_mechanism.py     # Hub entrapment vs moderate trap
│   ├── exp19_open_problems.py       # Graph-ball trapping + 100% waypoint
│   ├── exp20_statistics.py          # McNemar, bootstrap, budget accounting
│   ├── exp21_projections.py         # Projection invariance
│   ├── exp22_alignment.py           # Alignment stratification
│   ├── exp25_bge_crystal.py         # Second neural embedder on NeuroCrystal
│   ├── exp27_drainage_test.py       # Progressive drainage on 5 graphs
│   └── exp28_waypoint_spacing.py    # Optimal spacing derivation
├── synthetic/
│   ├── synthetic_kts_test.py        # KTS on 6 featureless distributions
│   ├── kts_generalization.py        # 13-graph KTS validation
│   ├── kts_fingerprint.py           # Graph topology spectrometer
│   └── synthetic_geometry.py        # Hubness from pure math
├── data/
│   └── results/                     # All experiment JSON outputs
└── CITATION.cff                     # Citation metadata
```

---

## Quick Start

### Requirements
```bash
pip install torch numpy scipy scikit-learn torch-geometric sentence-transformers pot
```

### Run on public data (no proprietary data needed)
```bash
# Verify K_mom = -1/2 theorem
python experiments/exp01_kmom_derivation.py

# Run 6-policy benchmark on Cora, Amazon, DBLP
python experiments/exp13_cross_graph.py

# Test KTS on synthetic point clouds
python synthetic/synthetic_kts_test.py

# Measure progressive drainage on 5 graphs
python experiments/exp27_drainage_test.py
```

### Run on NeuroCrystal (requires data files — contact author)
```bash
# 6-policy benchmark on NeuroCrystal
python experiments/exp09_policy_benchmark.py

# 100% waypoint injection demonstration
python experiments/exp19_open_problems.py

# Second neural embedder (BGE) on NeuroCrystal
python experiments/exp25_bge_crystal.py
```

---

## The Theorem in Brief

**Formal proof:** Rank-one (Chung-Lu) model with assumptions A1–A5 (non-regular degree distribution, high-D embedding, residual angular-degree coupling β > 0, sublinear neighbor-degree, cosine-greedy policy). **Empirical validation:** 15 graph families (scale-free, heterogeneous, co-authorship, co-purchase, citation, web-hyperlink, film co-occurrence, random/Erdős–Rényi, small-world/Watts–Strogatz, block-model/SBM, preferential-attachment/Barabási–Albert, synthetic kNN), 2 neural encoders (nomic 768D, BGE 384D). Multi-Anchor wins all 15 families — including drainage, weak-drainage, and anti-drainage regimes.

**Part (i) Drainage:** E[deg(v_{i+1})] = ρ_eff + α(d_i - ρ_eff), where ρ_eff = ρ·exp(-|β|√D_eff)

**Part (ii) Miss Probability:** P(miss) ≥ 1 - ∏ p_align(i), with scissors effect (drainage × frontier)

**Part (iii) Phase Transition:** d_drain ≈ log(d₀/ρ_eff) / log(1/α) ≈ 3.5 on NeuroCrystal

**Part (iv) Multi-Anchor:** P(miss_multi) ≤ P(miss_single | d/2)²

**Part (v) Waypoint Elimination:** Injection prevents drainage accumulation; W* = min(d_drain, 1/bottleneck_density)

**Corollary:** "The path is the answer" is a theorem consequence — knowing where to place waypoints IS knowing the discovery.

---

## Retracted Claims (Transparency)

An earlier version proposed Ollivier-Ricci curvature (ORC) as the mechanism. Exact Wasserstein-1 computation showed the proxy was an artifact (ρ = +0.078, n.s., vs proxy ρ = -0.966). The ORC mechanism is **retracted**. The progressive drainage mechanism via directional degree regression replaced it. The retraction is documented in `papers/retraction_orc_proxy.md`.

---

## Citation

```bibtex
@misc{godat2026drainage,
  author = {Godat, Michael},
  title = {The Cosine Drainage Theorem: Why Similarity Search Gets Trapped in High-Dimensional Graph Traversal},
  year = {2026},
  url = {https://github.com/mjgodat/cosine-drainage-theorem}
}
```

---

## Author

**Michael Godat** — Independent Researcher, Steilacoom, Washington, USA

This work emerged from building [NeuroCrystal/PRSM](https://github.com/mjgodat/NeuroAI), a geometric concept-traversal system for biomedical literature-based discovery. The trapping problem was encountered first. The architectural solution (waypoint injection) was built second. The theorem came last.

---

## Acknowledgments

The Cosine Drainage Theorem was developed through iterative research across multiple AI systems (Claude, Gemini, Perplexity, Grok) serving as research assistants, reviewers, and adversarial testers. All experimental code was executed on a consumer gaming laptop (ASUS TUF Gaming 16, RTX 4070, i7, 32GB RAM). No institutional funding was used.

*27 experiments. 5 graphs. 2 neural encoders. 15 validated hypotheses. 3 mechanism hypotheses tested — 2 falsified, 1 confirmed with proof. 5 quantitative gaps closed. All on a laptop.*
