# The Cosine Drainage Theorem

**Michael Godat**
*Independent Researcher*

---

## Significance

The Vector-Graph Semantic Gap (VGSG) was established empirically: cosine-similarity-biased traversal of high-dimensional embedded graphs systematically fails to reach graph-reachable targets under finite budgets. The Cosine Drainage Theorem converts this observation into a conditional formal account supported by empirical evidence across six structurally diverse graphs — scale-free (NeuroCrystal, degree CV = 2.69), heterogeneous (Cora CV = 1.34, Amazon CV = 1.94, DBLP CV = 1.57), and synthetic kNN (Isotropic CV = 1.65, Mixture CV = 1.34) — and two independent neural encoders.

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

**(A3') Residual angular-degree correlation (revised).**
Even after conditioning on norm, a positive correlation persists between degree and angular proximity to the distributional mean direction:

    β = ρ(deg(v), cos(v̂, μ̂) | r(v)) > 0

on heterogeneous real graphs (measured range β ∈ [+0.05, +0.19] across encoders and domains). Consequently the pure conditional isotropy assumption is violated; the angular coordinate itself carries degree information. Cosine selection, being purely directional, is therefore mildly degree-averse. The effective drainage attractor is

    ρ_eff(β) = ρ · exp(-|β|√D_eff) < ρ

(The radial contribution to degree may be large or negligible according to the encoder; the residual angular coupling β is systematic.)

**(A4) Sublinear nearest-neighbor degree.**
The average nearest-neighbor degree satisfies k̄_nn(d) < d for all d > d*, where d* is the unique fixed point of k̄_nn. For uncorrelated (rank-one) networks, k̄_nn(d) = ρ (constant).

**(A5) Cosine-greedy traversal policy.**
Given a fixed target embedding φ(t), the walker at v_i selects

    v_{i+1} = argmax_{u ∈ F_i} û · t̂

where F_i is the current frontier and t̂ = φ(t)/‖φ(t)‖. Total visit budget: H.

---

## Theorem (Cosine Drainage)

*Under assumptions A1–A5 the following hold.*

---

### (i) Drainage Rate

**The expected degree of the cosine-selected neighbor regresses to the effective attractor ρ_eff(β).**

Let d_i = deg(v_i). Then

    E[deg(v_{i+1}) | deg(v_i) = d_i] = ρ_eff + α(d_i - ρ_eff) + O(d_i^{-1/2})

where α ∈ (-1, 1) encodes residual degree-degree correlation (Newman assortativity).

- For uncorrelated (rank-one) graphs, α = 0 and drainage is instantaneous in expectation: E[deg(v₁)] = ρ_eff.
- For assortative graphs (0 < α < 1) drainage is geometric:

    E[d_k] = ρ_eff + α^k · (d₀ - ρ_eff)

The sequence {d_i} is a supermartingale above ρ_eff and a submartingale below it, with ρ_eff the unique attractor.

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
| Residual angular-degree correlation beta (revised A3') | Measured, encoder-independent, systematic on real graphs | Empirically fitted |
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

The theorem is specific to cosine (norm-invariant) selection. Euclidean or hyperbolic greedy policies possess different norm sensitivities and lie outside the present claims. The single calibration constant appearing in the scissors expression is a residual empiricism of the same character as many constants in network-science theorems; the functional dependence and the architectural consequences do not rely on its particular numerical value.

---

## References

1. Feld, S.L. (1991). Why your friends have more friends than you do. *American Journal of Sociology*.
2. Hui, Q. & Wang, T. (2026). Hub neighbor-degree diagnostics for sparse random graphs. *arXiv:2607.26624*.
3. Radovanović, M., Nanopoulos, A. & Ivanović, M. (2010). Hubs in space. *JMLR*.
4. Newman, M.E.J. (2002). Assortative mixing in networks. *Phys. Rev. Lett.*
5. Godat, M. (2026). The VGSG conjecture and supporting empirical program. PRSM Research.
6. Godat, M. (2026). Progressive drainage under cosine-greedy traversal (technical reports on rate, miss probability, and unified theorem).

---

*Unified theorem, revised 17 August 2026. Incorporates systematic residual angular-degree correlation, effective attractor rho_eff(beta), justified tree bound for cosine, scissors model of alignment probability, and bottleneck-dependent waypoint spacing rule. This is a conditional formal account with a mixture of derived, fitted, and empirically calibrated quantities. Validated across multiple real graphs and embedding families (nomic, BGE, native features). All five quantitative gaps closed.*
