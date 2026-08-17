# The Cosine Drainage Theorem: A Rank-One Proof and a Cross-Family Conjecture for Embedded Non-Regular Graphs

**Michael Godat**
*Independent Researcher*

---

> **Theorem scope.** The formal result applies to the Chung-Lu rank-one model under A1-A5.
> **Conjecture scope.** The broader non-regular-graph claim is supported by experiments across six real and synthetic graph families but is not claimed as universally proved.

---

## Nomenclature

This document contains two layers:

1. **The Cosine Drainage Conjecture:** The broad hypothesis that cosine-greedy
   traversal on embedded non-regular graphs exhibits progressive degree drainage
   under directional-degree coupling. Supported by empirical evidence across six
   graph families (degree CV 1.34 to 2.69) and two neural encoders. Not proved
   for all non-regular graphs — a single valid counterexample would refute the
   universal claim.

2. **The Rank-One Cosine Drainage Theorem:** The provable result under the
   explicit rank-one (Chung-Lu) generative model with measured angular-degree
   coupling β > 0. This is a formal conditional result: under model M and
   assumptions A1-A5, drainage occurs and converges to ρ_eff.

The conjecture becomes a theorem only through proof under a specified model;
observations strengthen belief and expose conditions but do not prove a
universal mathematical claim.

---

## Significance

The Vector-Graph Semantic Gap (VGSG) was established empirically: cosine-similarity-biased traversal of high-dimensional embedded graphs systematically fails to reach graph-reachable targets under finite budgets. This document contains two separable results:

**The Cosine Drainage Conjecture (conditional mechanism).** In embedded graphs where cosine-to-target selection systematically underweights higher-degree frontier candidates relative to the graph's size-biased neighbor baseline, greedy traversal exhibits degree drainage toward a lower effective degree regime. The sign and magnitude of the effect are determined jointly by graph topology, embedding geometry, target distribution, and frontier dynamics. Drainage was observed in 10 of 15 evaluated graph families, was weak or mixed in 4, and reversed in 1 (SBM, where community-core geometry preferentially ranks high-degree nodes). The Rank-One Cosine Drainage Theorem proves the mechanism under the Chung-Lu model with measured angular-degree coupling β > 0.

**The Cross-Family Multi-Anchor Result (robust intervention).** At matched budgets, Multi-Anchor expansion outperformed single-source cosine in all 15 evaluated graph families — spanning scale-free, heterogeneous, co-authorship, co-purchase, citation, web-hyperlink, film co-occurrence, random (Erdos-Renyi), small-world (Watts-Strogatz), block-model (SBM), preferential-attachment (Barabasi-Albert), and synthetic kNN constructions — including graphs exhibiting drainage, weak drainage, and anti-drainage. The intervention works regardless of the local mechanism.

These two results are supported by empirical evidence across 15 graph families (degree CV 0.10 to 2.69) and two independent neural encoders (nomic-embed-text 768D, BGE-small-en-v1.5 384D).

The theorem identifies the mechanism as progressive degree regression under directional selection. Cosine is a purely angular, norm-invariant metric. In high-dimensional embeddings, node degree correlates with proximity to the distributional mean—both radially and, residual to norm, angularly. Consequently, cosine-to-target selection is mildly degree-averse: it preferentially selects away from the mean direction and therefore away from hubs. The expected degree of the selected neighbor equals a reduced attractor ρ_eff(β) < ρ, where ρ = ⟨k²⟩/⟨k⟩ is the ordinary size-biased mean and β > 0 is the residual within-norm angular-degree correlation. Each successive step repeats this regression, producing monotone degree drainage into the structural periphery.

The theorem derives the miss probability as a product of per-step alignment factors governed by a scissors effect (declining degree versus accumulating frontier), locates the drainage-driven phase transition, and shows that multi-anchor and waypoint-injection architectures bound the failure mode under stated assumptions. Recommended waypoint spacing is local and bottleneck-dependent; on traces with high dark-link density the spacing collapses to adjacent placement, formalizing the statement that "the path is the answer."

Every primary quantity is computable from the graph's degree distribution and the embedding's angular statistics — subject to measuring β and calibrating c₀ on the target graph — rendering the predictions falsifiable on any new graph-embedding pair. The drainage mechanism does not require scale-free topology or extreme degree heterogeneity; it was observed on all six tested graph families, including synthetic kNN constructions with moderate degree variation. Scale-free structure amplifies the effect but does not create it. The theorem supplies the theoretical foundation for why retrieval systems that seed from cosine-nearest neighbors and expand by graph proximity are structurally limited in discovering distant cross-domain connections, and it prescribes the architectural remedies.

---

## Assumptions

**(A1) Graph with heterogeneous degree distribution.**
Let G = (V, E) be a simple undirected graph on N vertices with empirical degree distribution P(k) satisfying Var(k) > 0,

    ⟨k⟩ = N⁻¹ Σ_v deg(v)
    ⟨k²⟩ = N⁻¹ Σ_v deg(v)²
    ρ = ⟨k²⟩ / ⟨k⟩

**(A2) High-dimensional embedding.**
Each node v is assigned an embedding φ(v) ∈ ℝ^D with D ≫ 1. Write φ(v) = r(v)·v̂ with r(v) = ‖φ(v)‖₂ and v̂ ∈ S^{D-1}.

**(A3a) Degree-mean alignment.**
Higher-degree nodes are angularly closer to the distributional mean direction μ̂, even after conditioning on norm:

    β = corr(deg(v), cos(v̂, μ̂) | r(v)) > 0

Measured range β ∈ [+0.05, +0.19] across encoders and domains. This is a systematic property of trained transformer embeddings: general concepts appear in many training contexts, pulling their directions toward the distributional mean (context-averaging effect).

**(A3b) Target-relative hub disadvantage.**
For the target distribution and frontier dynamics induced by the traversal policy, high-degree candidates are systematically cosine-disadvantaged relative to lower-degree candidates:

    Δ_hub(t, i) = E[cos(û, t̂) | u ∈ F_i, k_u high] - E[cos(û, t̂) | u ∈ F_i, k_u low] < 0

This condition holds when targets are distributed as deviations from the mean direction μ̂ (which is the generic case for specific queries on trained embeddings, since the mean represents the "average of everything" while each target represents a specific concept). A3b is the policy-relevant condition that determines the sign of drainage.

**Critical distinction (Experiment 31).** A3a alone (scalar β > 0) does NOT guarantee drainage. When degree-direction coupling is injected via an arbitrary fixed direction (not the distributional mean), increasing β produces ANTI-drainage (ρ(β, T3) = +0.73 on SBM). Drainage requires the specific geometry where the degree-associated direction IS the distributional mean, and targets deviate FROM that mean. A3a provides the geometric structure; A3b provides the directional asymmetry that converts it into drainage.

**Effective attractor.** Under A3a + A3b, the drainage attractor is:

    ρ_eff = ρ · exp(-|β| √D_eff) < ρ

This is an empirically fitted formula; the exponential form and √D_eff scaling match the observed drainage floor across encoders but are not derived from a generative model.

**(A4) Sublinear nearest-neighbor degree.**
The average nearest-neighbor degree satisfies k̄_nn(d) < d for all d > d*, where d* is the unique fixed point of k̄_nn. For uncorrelated (rank-one) networks, k̄_nn(d) = ρ (constant).

**(A5) Cosine-greedy traversal policy.**
Given a fixed target embedding φ(t), the walker at v_i selects

    v_{i+1} = argmax_{u ∈ F_i} û · t̂

where F_i is the current frontier and t̂ = φ(t)/‖φ(t)‖. Total visit budget: H.

---

## Rank-One Cosine Drainage Theorem

*Under assumptions A1-A5 and the rank-one (Chung-Lu) generative model, the following hold. The broader Cosine Drainage Conjecture asserts that these results extend to all non-regular embedded graphs satisfying A3'; this extension is supported by empirical evidence across six graph families but is not formally proved.*

---

### (i) Drainage Rate

**The expected degree of the cosine-selected neighbor regresses to the effective attractor ρ_eff(β).**

Let d_i = deg(v_i). Then

    E[deg(v_{i+1}) | deg(v_i) = d_i] = ρ_eff + α(d_i - ρ_eff) + O(d_i^{-1/2})

where α ∈ (-1, 1) encodes residual degree-degree correlation (Newman assortativity).

- For uncorrelated (rank-one) graphs, α = 0 and drainage is instantaneous in expectation: E[deg(v₁)] = ρ_eff.
- For assortative graphs (0 < α < 1) drainage is geometric:

    E[d_k] = ρ_eff + α^k · (d₀ - ρ_eff)

For sufficiently large d_i (specifically d_i > ρ_eff + O(d_i^{-1/2}), where the error margin accounts for finite-sample fluctuations), the sequence {d_i} is a supermartingale. Below ρ_eff, it is a submartingale. ρ_eff is the unique attractor.

The multi-step appearance of drainage on real trajectories is produced by variance (tail rescue from high-degree nodes) together with mild assortativity; the expectation itself collapses in a single step when α = 0.

**Proof status.** Formally proved for the rank-one case (drainage to size-biased mean under directional selection away from the mean). The effective attractor rho_eff = rho * exp(-|beta| * sqrt(D_eff)) is an empirically fitted floor; the exponential form and sqrt(D_eff) scaling match the observed drainage attractor across encoders but are not derived from a generative model. Proof sketch for the linear-ANND assortative extension. The drainage recurrence E[d_{i+1}] = rho_eff + alpha(d_i - rho_eff) is a conditional result under the rank-one model with measured assortativity.

---

### (ii) Miss Probability

**The probability that cosine-greedy traversal with budget H fails to visit a target at graph distance d is bounded below by a product of per-step alignment factors that decay under a scissors effect.**

Let p_align(i) be the probability that the graph-shortest-path neighbor of v_i possesses the highest cosine to the target among all frontier candidates at step i. Then

    P(miss | d_G = d, H) ≥ 1 - ∏_{i=0}^{d-1} p_align(i)

The alignment probability admits the scissors-ratio form

    p_align(i) ≈ c(i) / (c(i) + Γ · |F_i|)

where c(i) is the angular advantage of the correct neighbor (decreasing with i) and Γ is the global angular-compression statistic. Three compounding effects erode p_align:

1. degree drainage (signal decay toward ρ_eff),
2. frontier accumulation (noise growth),
3. angular narrowing near the target (discrimination loss).

The product therefore collapses faster than exponentially.

**Boundary condition.** P(miss | d_G = 1) = 0 exactly, because the target itself lies in the initial frontier and realizes maximal cosine.

**Empirical calibration.** With a single constant c₀ fixed at the d = 2 miss rate, the same expression recovers the d = 3 and d = 4 miss rates to within one percentage point on NeuroCrystal, reproducing the observed phase-transition cliff.

**Proof status.** Lower bound derived under the locally tree-like (configuration-model) approximation. The tree approximation remains valid for cosine-greedy search: clustering supplies multi-path redundancy that BFS can exploit, but cosine selects solely by angular alignment and therefore cannot use those alternative paths. The correction is O(1/rho_eff) and does not grow with clustering coefficient. The scissors functional form is a semi-empirical model with one calibration constant c_0, not a closed-form derivation from first principles.

---

### (iii) Phase Transition

**There exists a critical distance d_drain at which miss probability transitions from the hub-assisted regime to the drainage-dominated regime.**

    d_drain ≈ log(d₀/ρ_eff) / log(1/α)    (α > 0)
    d_drain = 1                              (α = 0)

For d ≤ d_drain residual hub connectivity keeps per-step miss rates moderate. For d > d_drain degree has equilibrated near ρ_eff, the frontier dominates, and per-step miss rates rise sharply.

**Empirical match (NeuroCrystal, H = 100).**
d₀ ≈ 370, ρ_eff ≈ 33 yield d_drain ≈ 3.5. Observed miss rates — 0% (d=1), 12.4% (d=2), 32.3% (d=3), 81.8% (d=4) — confirm the transition between distance 3 and 4. The location is a variance phenomenon: expectation collapses quickly, but high-degree nodes retain a non-negligible probability of sampling rare high-degree neighbors until degree falls below the tail-rescue threshold.

**Proof status.** Derived from the drainage recurrence (itself conditional on rank-one + measured assortativity) and the two-regime model; transition location and sharpness calibrated empirically.

---

### (iv) Multi-Anchor Reduction

**Multi-anchor expansion (seeding from both source and target with budget H/2 each) squares the single-source survival probability by halving effective graph distance.**

    P(miss_multi | d) ≤ P(miss_single | ⌈d/2⌉)²

Each side covers roughly half the distance; the miss event requires both expansions to fail independently. The bound converts an 81.8% single-source miss rate at d = 4 into an upper bound of approximately 1.5%.

**Improvement window.** Multi-anchor improves upon single-source when d_drain < d < 2H (conditional on frontier independence).

**Proof status.** Conditional bound under the assumption that forward and backward frontiers do not interact before the bridge zone. In practice, frontiers share the same graph and may exhibit correlation, loosening the bound.

---

### (v) Waypoint Injection

**Waypoint injection bounds cumulative drainage by restarting the walk at known intermediate nodes, preventing the supermartingale from accumulating across segments.**

Let W = {w₀ = s, w₁, ..., w_m, w_{m+1} = t} be a sequence of waypoints with local budgets H_j = H/(m+1). Then:

- (v.1) Each local expansion remains shorter than the drainage scale whenever d_G(w_j, w_{j+1}) < d_drain.
- (v.2) The degree sequence is reset at every waypoint; no supermartingale accumulates across segments.
- (v.3) Segment miss probabilities multiply:

    P(miss_waypoint) = 1 - ∏_{j=0}^{m} (1 - P(miss_segment_j))

When consecutive waypoints are adjacent (d_G(w_j, w_{j+1}) = 1) every factor vanishes and total miss probability is zero, provided the algorithm retains the adjacent target node within its budget and frontier.

**Recommended spacing.**
The recommended local spacing is a design rule, not a formally proved optimum (formal optimality would require an explicit objective function and waypoint-generation cost model):

    W* = min(d_drain, 1/bottleneck_density)

Bottleneck density is the fraction of consecutive concept pairs that are semantically successive yet non-adjacent in the graph ("dark links"). On NeuroCrystal validated traces this density equals 0.44, forcing W* → 1. Consequently, knowing where to place the waypoints is equivalent to knowing the path; the waypoints are not merely a guide for search — they constitute the discovery. This supplies the formal content of the architectural claim "the path is the answer."

**Proof status.** The d_G = 1 boundary condition (P(miss | d_G = 1) = 0) is formally proved. The restart argument (bounding cumulative drainage by segment) is formally proved. The spacing expression W* is a design rule derived from the drainage scale and measured bottleneck density, not a formal optimum.

---

## Corollaries

**Corollary 1 (Suboptimality of cosine-greedy).**
For any graph distance d > d_drain, single-source cosine-greedy traversal has miss probability bounded away from zero. In the evaluated graph families — including scale-free (NeuroCrystal, CV = 2.69), heterogeneous (Cora CV = 1.34, Amazon CV = 1.94, DBLP CV = 1.57), and synthetic kNN constructions with moderate degree variation (Isotropic CV = 1.65, Mixture CV = 1.34) — progressive drainage was consistently observed whenever graphs were non-regular (T3 ρ = -0.30 to -0.82). The conditional theory predicts that nonzero degree variance enables drainage; its observed strength depends on the embedding's residual directional-degree coupling and traversal conditions. Scale-free or high-variance degree distributions amplify the effect by increasing the gap between initial hub-supported connectivity and the effective degree regime; they are not a prerequisite for drainage.

**Corollary 2 (Multi-anchor improvement).**
For d_drain < d < 2H, multi-anchor expansion achieves a quadratic improvement in survival probability and is the recommended strategy when the target is known but intermediate waypoints are not.

**Corollary 3 (Waypoint injection).**
Whenever a waypoint sequence exists with consecutive distances less than d_drain (in particular when bottleneck density forces adjacent placement), waypoint-injected traversal achieves strictly lower miss probability than either single-source or multi-anchor methods. It achieves the lowest miss probability among tested strategies when intermediate concepts can be generated.

**Corollary 4 (Derivation of the predictive model).**
The empirical three-feature model reach ~ a·Γ + b·log(H) + c·degCV follows from the theorem:
- Γ enters the scissors ratio,
- log(H) governs residual recovery after deviation,
- degree coefficient of variation controls the gap d₀/ρ_eff and therefore drainage speed.
All coefficient signs are fixed by the theorem.

---

## Boundary Conditions

### When drainage does not occur
Drainage vanishes if and only if Var(k) = 0 (regular graph). All other regimes merely modulate strength:

| Condition | Effect |
|-----------|--------|
| k-regular | Drainage = 0 |
| kNN graph, D ≫ 1 | Drainage persists (hubness generates degree heterogeneity) |
| Assortative mixing | Drainage slowed but sub-linear |
| Low dimension (D < 100) | Drainage weakened |
| Target itself a hub | Path still drains; reachability is a race against d_drain |
| Stochastic Block Model | **Anti-drainage** observed (T3 = +0.52): within-community cosine expansion moves toward higher-degree cores |
| Barabasi-Albert + random embeddings | Drainage occurs despite PA topology; realized embedding geometry, not graph generative model alone, determines the sign |

### When miss probability is zero
1. d_G(s, t) = 1,
2. budget exhausts the entire graph ball,
3. waypoint injection with all consecutive distances equal to 1,
4. regular graph together with perfect geometry-graph alignment.

### When multi-anchor confers no advantage
1. d ≤ d_drain (budget-split penalty dominates),
2. H ≫ 2d,
3. d > 2H.

---

## Proof-Status Summary

| Component | Status | Epistemic Category |
|-----------|--------|-------------------|
| A3a: Degree-mean alignment (beta > 0) | Measured on all real graphs + both encoders (beta +0.05 to +0.19) | Empirically confirmed |
| A3b: Target-relative hub disadvantage | Required for drainage; falsified for arbitrary-direction injection (Exp 31: rho(beta,T3)=+0.73 without mean alignment) | Empirically confirmed on real encoders; falsified for arbitrary injection |
| Drainage to size-biased mean (rank-one) | Formally proved (size-biased sampling + directional selection) | Formally proved |
| rho_eff = rho * exp(-\|beta\| * sqrt(D_eff)) | Matches observed attractor; exponential form and sqrt(D_eff) scaling are fitted | Empirically fitted |
| Drainage recurrence (assortative) | Conditional on rank-one + measured assortativity | Conditional result |
| Assortative extension | Proof sketch (linear ANND) | Conditional result |
| Miss-probability lower bound (tree-like) | Derived under locally tree-like approximation; justified for cosine | Conditional result |
| Scissors form of p_align | Semi-empirical model; one calibration constant c_0 | Empirically fitted |
| Phase-transition location d_drain | Derived from the recurrence (itself conditional) | Conditional result |
| Multi-anchor quadratic bound | Conditional on frontier independence (frontiers share the graph) | Conditional result |
| Waypoint segment restart | Formally proved (d_G = 1 boundary + restart argument) | Formally proved |
| Spacing rule W* | Design rule from drainage scale + measured bottleneck density | Design heuristic |
| P(miss \| d_G = 1) = 0 | Formally proved | Formally proved |
| Semantic gap existence (if d_G > H then miss) | Formally proved | Formally proved |

---

## Remarks on Scope

The Rank-One Cosine Drainage Theorem is specific to cosine (norm-invariant)
selection under the Chung-Lu generative model with measured angular-degree
coupling β > 0. Euclidean or hyperbolic greedy policies possess different
norm sensitivities and lie outside the present claims. The single calibration
constant appearing in the scissors expression is a residual empiricism of the
same character as many constants in network-science theorems; the functional
dependence and the architectural consequences do not rely on its particular
numerical value.

The broader Cosine Drainage Conjecture — that the same drainage mechanism
operates on all non-regular embedded graphs with β > 0 — is supported by
consistent empirical observation across six graph families (degree CV 1.34
to 2.69, including synthetic kNN) and two neural encoders, but is not
formally proved beyond the rank-one model. A single valid counterexample
(a non-regular embedded graph with β > 0 where cosine-greedy traversal does
NOT exhibit progressive drainage) would refute the conjecture.

On non-regular graphs, if residual directional structure causes cosine-greedy
selection to underweight high-degree candidates relative to size-biased
neighbor sampling, the selected degree regresses toward a lower effective
degree regime. The effect may be weak in near-regular graphs and stronger
when degree variance, hubness, directional-degree coupling, or initial hub
degree are larger. Var(k) > 0 alone cannot logically ensure drainage; the
coupling condition β > 0 and the traversal assumptions are essential.

---

## Boundary-Search Program

To strengthen the conjecture toward theorem status beyond rank-one, actively
seek counterexamples in:

1. Near-regular graphs (degree CV < 0.5)
2. Embeddings with β ≈ 0 (deliberately isotropic encoders)
3. Degree-direction decoupled embeddings (adversarial construction)
4. Adversarial target placement (targets at hub nodes only)
5. Alternative graph constructions (radius graphs, mutual-kNN, Watts-Strogatz)

If drainage persists across these boundary cases, the conjecture is
substantially hardened. If any case breaks drainage, it identifies the
precise boundary of the theorem's applicability.

---

## References

1. Feld, S.L. (1991). Why your friends have more friends than you do. *American Journal of Sociology*.
2. Hui, Q. & Wang, T. (2026). Hub neighbor-degree diagnostics for sparse random graphs. *arXiv:2607.26624*.
3. Radovanović, M., Nanopoulos, A. & Ivanović, M. (2010). Hubs in space. *JMLR*.
4. Newman, M.E.J. (2002). Assortative mixing in networks. *Phys. Rev. Lett.*
5. Godat, M. (2026). The VGSG conjecture and supporting empirical program. PRSM Research.
6. Godat, M. (2026). Progressive drainage under cosine-greedy traversal (technical reports on rate, miss probability, and unified theorem).

---

*FROZEN 17 August 2026. This is the final version of the Cosine Drainage Theorem document for this research phase. Two-layer structure: the Rank-One Cosine Drainage Theorem (proved under Chung-Lu model with A3a + A3b) and the broader Cosine Drainage Conjecture (supported by 15 graph families, two encoders). A3' has been split into A3a (degree-mean alignment, necessary but not sufficient) and A3b (target-relative hub disadvantage, the policy-relevant condition). Experiment 31 falsified the simple β > 0 ⟹ drainage implication and identified mean-directed coupling as the operative geometry. Multi-Anchor search is an empirically robust intervention across all 15 tested families regardless of whether drainage, weak drainage, or anti-drainage is active. Experiment manifest frozen; no further exploratory experiments without pre-registration.*

*Thesis: Embedding geometry and graph topology jointly determine greedy traversal degree dynamics; target-relative hub disadvantage produces drainage, while other topology-geometry alignments can produce anti-drainage. Multi-Anchor search is empirically robust across both regimes.*
