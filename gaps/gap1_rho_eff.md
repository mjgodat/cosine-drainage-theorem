# Retraction Notice: ORC-Based "Wrong Gradient" Mechanism
## Date: August 17, 2026

---

## What is retracted

**Proposition 1 (High-Dimensional Curvature-Cosine Inversion)** and all
claims derived from it are retracted. The claim that cosine-biased
expansion preferentially follows negative Ollivier-Ricci curvature
(ORC) bridge edges is not supported by exact computation.

## What happened

All prior experiments (7, 8, 10) used a **proxy approximation** for ORC:
mean cross-neighborhood cosine divided by edge distance. This proxy
produced a strong anti-correlation with cosine similarity (ρ = -0.74
to -0.98 across all tested graphs) and formed the basis of the
"Wrong Gradient" mechanism in the VGSG conjecture (Part 4).

Experiment 14 computed **exact Ollivier-Ricci curvature** using
Wasserstein-1 optimal transport (POT library, linear programming)
on 500 sampled edges from the PRSM Crystal subgraph.

### Results

| Comparison | ρ | p-value |
|-----------|---|---------|
| Proxy ORC vs cosine | **-0.966** | 3.1 × 10⁻²⁹⁴ |
| **Exact ORC vs cosine** | **+0.078** | **0.080 (ns)** |
| Exact ORC vs proxy ORC | 0.006 | 0.894 (ns) |

- Sign agreement between proxy and exact: **47.6%** (worse than chance)
- The proxy does not approximate exact ORC at all
- The strong anti-correlation was an artifact of the proxy's ratio
  formula (dividing near-constant numerator by varying denominator),
  not a property of actual Wasserstein-1 transport geometry

## What is NOT retracted

The following results do not depend on the ORC mechanism and remain valid:

1. **The trapping phenomenon.** Cosine-biased single-source expansion
   underperforms BFS and dual-frontier methods under finite budgets.
   (Experiments 6, 9, 13)

2. **The multi-anchor intervention.** Dual-frontier expansion achieves
   63-100% reachability where BFS reaches 0-8% and single-source
   cosine reaches 15-82%. (Experiments 9, 13)

3. **The analytical null models.** π/2 isotropic angular null (Theorem 1)
   and K_mom = -1/2 adjacent-secants null (Theorem 2) are mathematical
   results independent of ORC.

4. **KTS instrument.** All 5 metrics, synthetic validation, L2 ablation,
   and graph topology fingerprinting are independent of ORC.

5. **The predictive scaling model.** R² = 0.92 three-feature model
   (Γ, log(H), degree CV) with leave-one-graph-out ρ = 0.875.
   This model uses Γ and degree CV, not ORC.

6. **Multi-representation policy ordering.** 100% consistency of
   Multi > Cosine > BFS across 4 feature spaces.

7. **Causal stratification.** Bidirectional benefit increases with
   angular separation (ρ = 0.141, p = 0.0016).

8. **Frontier telemetry observation.** Cosine expansion visits
   lower-degree nodes (mean 85.8 vs BFS 293.4). This observation
   stands. The ORC interpretation of WHY does not.

9. **Theorem 3 (Semantic Gap Existence).** Definitional, independent
   of any mechanism.

## What changes in the conjecture

- **Part 4 (Wrong Gradient via ORC)** is retracted.
- **"Semantic Gravity = Wrong Gradient × Angular Compression"** must
  be revised. The trapping exists. Angular compression (Γ) contributes.
  The geometric mechanism that COMPLETES the explanation is now an
  open problem.
- **The formal proof** (proof_orc_anticorrelation.md) proves a valid
  mathematical result about the proxy formula's ratio structure, but
  this ratio structure does not correspond to actual Wasserstein-1
  curvature. The proof is mathematically correct but physically
  irrelevant.

## Revised central claim

The conjecture's core observation remains:

> Under finite budgets, cosine-similarity-biased graph expansion
> exhibits systematic confinement relative to topology-aware and
> dual-frontier baselines. The severity of this confinement is
> predictable from angular compression (Γ), budget (log H), and
> degree distribution (degree CV). The dual-frontier intervention
> resolves it.

The MECHANISM — why cosine expansion gets confined — is now an open
problem. Candidate explanations to investigate:

1. **Hub avoidance under cosine ranking.** Cosine best-first
   preferentially expands nodes whose embeddings are closest to the
   target. High-degree hub nodes may not be the angularly closest
   to most targets, causing cosine to route around them.

2. **Density-biased expansion.** Cosine may preferentially expand
   into high local-density (high Γ_k) regions where many candidates
   compete for budget, rather than toward sparse bridge regions.

3. **Angular corridor narrowing.** As cosine expansion proceeds, it
   narrows the angular cone of candidates, progressively excluding
   nodes outside the source-target angular corridor.

These are hypotheses, not established mechanisms. The correct
explanation requires further investigation.

## Commitment

This retraction is published in the same channels as the original
claims: the conjecture document, the thesis paper, the handoff branch,
and memory files. The ORC proxy results remain in the experimental
record as a documented negative finding. The exact ORC results
(Experiment 14) are published alongside them.

Handling negative results transparently is what makes the remaining
positive results trustworthy.
