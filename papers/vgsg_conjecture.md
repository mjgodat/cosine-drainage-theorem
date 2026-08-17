# The VGSG Conjecture: Vector-Graph Semantic Gravity
## Formalized 2026-08-16 (v5)
## Michael Godat, Independent Researcher

---

## Preamble

Let M = S^(D-1) ⊂ ℝ^D be the (D−1)-dimensional unit hypersphere (D >> 1).
Let G = (V, E) be a graph with vertex set V embedded in M via φ: V → M,
such that v⃗_i = φ(v_i) for each v_i ∈ V.

For any two points u⃗, w⃗ ∈ M, angular distance is:

    d_θ(u⃗, w⃗) = arccos(⟨u⃗, w⃗⟩) ∈ [0, π]

Let σ_k be a cosine-similarity-based top-k seed selection policy:

    S_k(q⃗) = argmax_{S ⊆ V, |S|=k} Σ_{s ∈ S} ⟨q⃗, φ(s)⟩

Let P be a graph expansion policy (e.g., BFS, cosine-biased best-first,
beam search, Personalized PageRank) with expansion budget H ∈ ℕ⁺ measured
in total node visits. The policy-reachable set is:

    R_H^P(S_k(q⃗)) = {u ∈ V | u is visited by policy P within budget H
                       starting from seeds S_k(q⃗)}

Note: R_H^P is policy-dependent and generally a SUBSET of the graph-distance
ball {u : d_G(S_k, u) ≤ H}. An exhaustive policy (BFS) visits every node
within graph radius H. A similarity-biased policy may fail to visit nodes
within graph radius H if it preferentially expands toward angularly similar
neighbors, exhausting budget before reaching angularly distant but
graph-proximate nodes. This distinction is central to VGSG: trapping occurs
when a policy's directional bias causes R_H^P to be strictly smaller than
the graph-distance ball.

Let θ̄ denote the mean pairwise angular distance between embedded nodes:

    θ̄ = (2 / N(N-1)) Σ_{i<j} arccos(⟨φ(v_i), φ(v_j)⟩)

---

## Experimental Graph Corpus and Taxonomy

All claims in this conjecture are tested across graphs of diverse construction.
Each graph was classified by measurable structural properties:

**Classification criteria:**
- **SCALE-FREE:** degree CV > 2.0 or power-law α > 2.0 (hub-dominated)
- **HETEROGENEOUS:** multiple discrete node types (> 3 classes)
- **SMALL-WORLD:** high clustering coefficient + high CC/random ratio
- **MODULAR:** moderate clustering, no strong type or hub structure
- **REGULAR/SPATIAL:** low degree CV (< 0.3), uniform connectivity

**Graphs tested:**

    Graph               Class           Nodes   MeanDeg  DegCV   CC      Γ      EdgeCos  Cond
    ──────────────────  ──────────────  ──────  ───────  ─────  ──────  ──────  ───────  ─────
    PRSM Crystal        SCALE-FREE      39,220    26.6   2.69   0.614   0.249   0.460   0.756
    STRING-DB v12.0     SCALE-FREE      19,699    high   high   —       —       —       —
    Cora                HETEROGENEOUS    2,708     3.9   1.34   0.289   0.036   0.168   0.482
    Amazon Computers    HETEROGENEOUS   13,752    36.5   1.94   0.362   0.227   0.490   0.503
    DBLP                HETEROGENEOUS   17,716     6.0   1.57   0.184   0.010   0.152   0.597
    Hetionet v1.0       HETEROGENEOUS   47,031    multi-type    —       —       —       —
    Grid Road           REGULAR/SPATIAL 10,000     4.0   0.25   —       —       —       —
    Synth Isotropic     MODULAR         10,000    36.2   1.65   0.046   0.000   0.096   0.585
    Synth Mixture       MODULAR         10,000    35.6   1.34   0.066   0.429   0.966   0.398
    + 4 more synthetic distributions (Anisotropic, Power-law, Sphere, Low-dim)

**Key structural observations across types:**

1. SCALE-FREE graphs (PRSM, STRING-DB) have the highest clustering
   coefficients and the strongest hub structure — dense local triangles
   with a few high-degree nodes dominating connectivity.

2. HETEROGENEOUS graphs (Cora, Amazon, DBLP, Hetionet) vary widely in
   density and compression. Cora and DBLP have low Γ (sparse features);
   Amazon has high Γ (dense product embeddings).

3. MODULAR/SYNTHETIC graphs serve as controls. Isotropic (Γ = 0) provides
   the geometric baseline. Mixture (Γ = 0.429) provides extreme compression.

4. Network MI (cosine-kNN overlap with graph edges at k=20):
   PRSM = 0.2%, Cora = 4.0%, Amazon = 2.0%, DBLP = 2.3%,
   Synth Isotropic = 25.7%, Synth Mixture = 43.0%.
   Real-world graphs have near-zero overlap between cosine neighborhoods
   and graph neighborhoods. Synthetic kNN graphs have high overlap because
   the edges ARE cosine-proximity edges.

All observations below are tested against this corpus. Generalizations
are bounded by the tested types; we do not claim coverage of all possible
graph constructions.

---

## Part 1 — The Isotropic Angular Null (π/2)

In D >> 1, concentration of measure forces the mean pairwise angular distance
between uniformly random unit vectors to converge to π/2:

    lim_{D→∞} E[d_θ(u⃗, w⃗)] = π/2    for u⃗, w⃗ ~ Uniform(S^{D-1})

π/2 is the isotropic angular null: at this value, all directions are
statistically independent, no angular neighborhood is preferentially populated,
and cosine-based retrieval returns effectively random results.

**Status: PROVEN.** This follows from classical concentration of measure
(Lévy, Milman). Empirically confirmed: synthetic isotropic Gaussian in ℝ^768
yields θ̄ = 1.5708 = π/2 to four decimal places, κ = 3.00.

---

## Part 2 — Angular Compression Statistic (Γ)

When structured content is embedded in M, the mean angular distance compresses
below π/2. The angular compression statistic is:

    Γ = (π/2 − θ̄) / (π/2)    ∈ [0, 1)

Γ measures the fractional compression of angular space relative to the
isotropic null:

    Γ = 0    →  no compression (isotropic, random)
    Γ > 0    →  angular compression present

Γ > 0 implies more nodes per unit solid angle within any angular cap
C(q⃗, θ_k). This creates conditions conducive to local recurrence in
similarity-biased traversal: more intra-basin edges consumed per hop,
reducing effective traversal radius for fixed budget H.

Γ is a candidate global predictor of local trapping propensity, conditional
on graph topology and traversal policy. It does not alone determine whether
trapping occurs — that depends on the interaction of Γ with graph structure,
degree distribution, edge conductance, and the specific expansion policy.

**Local variant:**

    Γ_k(v) = 1 − (1/k) Σ_{u ∈ kNN(v)} d_θ(u⃗, v⃗) / (π/2)

Γ_k(v) operates at node level and is mechanistically closer to the actual
trapping condition than global Γ.

**Empirical values:**

    Synthetic Isotropic:   Γ = 0.000    (zero compression, control)
    DBLP:                  Γ = 0.010    (near-isotropic)
    Cora:                  Γ = 0.036    (mild compression)
    Amazon Computers:      Γ = 0.226    (strong compression)
    PRSM Crystal:          Γ = 0.249    (strong compression)
    Synthetic Mixture:     Γ = 0.439    (extreme compression)

**Status: EMPIRICAL LAW.** Observed across 13 graphs. Γ_k independently
predicts traversal confinement on 2 of 3 benchmark graphs (Cora β = −0.296,
DBLP β = −0.247) after controlling for degree, graph distance, and angular
distance. Not significant on Amazon (β = +0.003), where graph distance
alone explains reachability.

---

## Part 3 — Angular Kurtosis (κ)

The kurtosis κ of the angular distance distribution characterizes the shape
of the compression field:

    κ = E[(θ − θ̄)⁴] / σ_θ⁴

    κ = 3.0    →  Gaussian (smooth, isotropic compression)
    κ > 3.0    →  heavy-tailed (deep wells + distant outliers = escape corridors)
    κ < 3.0    →  light-tailed (uniform compression, few escape routes)

**Empirical values:**

    Synthetic Mixture:     κ =  1.03    (uniform, no escape corridors)
    Synthetic Isotropic:   κ =  3.00    (exact Gaussian, control)
    Amazon Computers:      κ =  4.84    (mildly heavy-tailed)
    PRSM Crystal:          κ =  6.14    (heavy-tailed)
    Cora:                  κ =  8.88    (very heavy-tailed)
    DBLP:                  κ = 23.45    (extreme outlier concentration)

**Γ and κ are independent axes.** DBLP has the lowest Γ (0.010) AND the
highest κ (23.45): near-isotropic mean compression with extreme local
concentration. A graph can have weak global gravity but intense local
gravitational pockets. Conversely, the synthetic mixture has high Γ (0.439)
but low κ (1.03): strong uniform compression with no escape corridors.

**Status: EMPIRICAL.** Observed across 13 graphs. Currently a descriptive
topology statistic. Not yet shown to independently predict an outcome
(e.g., escape rate) after controlling for Γ and graph structure.

---

## Part 4 — Progressive Drainage (Directional Degree Regression)

Cosine similarity is a purely angular metric that ignores vector norm.
In high-dimensional embedded graphs, node degree correlates with proximity
to the distributional mean — hubs are angularly central, specialists are
angularly peripheral [Radovanovic et al. 2010]. Cosine-to-target selection
filters a hub's neighbors by angular direction, which is approximately
independent of degree among the neighborhood. The directionally selected
neighbor therefore has expected degree equal to the size-biased population
mean ρ = E[W²]/E[W] [Hui & Wang 2026, Theorem 3.12], far below the hub's
own degree. Each step repeats this regression, producing monotone degree
drainage until the walk converges to the structural periphery.

This is the directional inverse of the friendship paradox [Feld 1991]:
where uniform edge sampling biases toward hubs, directional edge sampling
(argmax cosine to a target) biases away from hubs.

**Key prediction:**

    E[deg(v_{i+1})] ≈ ρ = E[W²]/E[W] ≪ deg(v_i)    for hubs

This inequality holds at every step, producing monotone descent.

**Three converging theoretical frameworks:**

1. **Size-biased regression** [Hui & Wang 2026]: degree-invariant neighbor
   mean under rank-one kernels. The expected degree of a directionally
   selected neighbor is the size-biased mean ρ = E[W²]/E[W], which is
   independent of the selecting node's degree for sufficiently large hubs.

2. **Zoom-in phase of greedy routing** [Boguna et al. 2009, Papadopoulos
   et al. 2012]: structural descent toward peripheral targets. In
   hyperbolic-like navigable graphs, greedy forwarding toward a target
   node descends the degree hierarchy — each hop moves closer to the
   target in the angular coordinate while dropping to lower-degree nodes
   in the radial coordinate.

3. **Directional inverse of the friendship paradox** [Feld 1991]: angular
   selection is degree-blind, so the expected degree of the selected
   neighbor regresses to the population mean rather than being biased
   upward by the selecting node's degree. This is the exact inversion
   of Feld's result, where uniform edge sampling biases the sampled
   neighbor's degree upward.

**Empirical confirmation (Experiment 27, 5 graphs):**

- T1: Hubs have diffuse angular directions. Degree-direction entropy
  correlation is negative on all graphs (ρ = −0.10 to −0.32): higher-degree
  nodes have more uniformly distributed neighbor directions, meaning
  directional selection is less likely to pick a high-degree neighbor.

- T3: Degree drops monotonically over expansion steps. The correlation
  between step index and selected-neighbor degree is strongly negative
  on all graphs (ρ = −0.30 to −0.82). On NeuroCrystal, mean degree
  drops from 370 at step 1 to 33 by step 15 — an order-of-magnitude
  drainage.

- T5: Masking specialists DESTROYS reachability. This prediction was
  tested and falsified on all 5 graphs — masking low-degree nodes
  reduces reachability but the system partially compensates through
  alternative paths. The falsification refines the model: drainage is
  the dominant mechanism but not the only path-selection force.

**Cone narrowing is a co-effect, not a cause** (Team Beta finding):
both drainage and narrowing result from radial displacement from the
distributional mean. DBLP's assortative structure provides a natural
experiment: on DBLP, the angular cone actually WIDENS over successive
expansion steps while degree drainage continues unabated. This proves
the two phenomena are decoupled. Drainage is the primary mechanism;
narrowing is a secondary geometric consequence.

**Note on ORC retraction.** An earlier version proposed an Ollivier-Ricci
curvature mechanism (proxy ORC ρ = −0.966, exact ORC ρ = +0.078 [n.s.],
proxy-exact correlation ρ = 0.006). This was retracted after exact
Wasserstein-1 computation falsified the proxy approximation. Progressive
drainage replaces ORC as the micro-level explanation.

**Proof sketch reference:** `proof_progressive_drainage.md`

**Status: EMPIRICALLY CONFIRMED** across 5 graphs. Proof sketch available
at three model levels (size-biased regression, greedy routing, friendship
paradox inversion).

---

## Part 5 — Trapping: Theorem and Empirical Regime Propositions

### Theorem 1: Existence of the Semantic Gap (FORMAL)

Let T ⊂ V be a target subgraph with bounded internal diameter
diam_G(T) = max_{t_i, t_j ∈ T} d_G(t_i, t_j) ≤ Δ_T < ∞.

If the minimum graph geodesic between the seed set and the target exceeds
the hop budget:

    d_G(S_k(q⃗), T) = min_{s ∈ S_k, t ∈ T} d_G(s, t) > H

then the policy-reachable set excludes the target for ANY policy that
traverses at most one edge per unit of budget:

    R_H^P(S_k(q⃗)) ∩ T = ∅

**Proof:** Any policy P that traverses at most one edge per budget unit
can reach at most nodes within graph distance H of the seed set. If
min_{s ∈ S_k, t ∈ T} d_G(s, t) > H, no element of T is within graph
distance H of any seed, so no policy can reach T within budget H.
Internal graph connectivity diam_G(T) < ∞ does not imply seed-conditioned
reachability under bounded expansion. ■

**Corollary (Policy-dependent trapping):** For similarity-biased policies,
a stronger condition holds: R_H^P ⊊ {u : d_G(S_k, u) ≤ H}. A target t
with d_G(S_k, t) ≤ H can STILL be missed if the policy's directional bias
causes it to visit other nodes first, exhausting budget before reaching t.
This is the operational VGSG trapping condition, validated experimentally
in Part 7.

This theorem is existential: it establishes that unreachable targets CAN
exist under bounded traversal, not that all targets ARE unreachable. Whether
trapping occurs in practice depends on the interaction of angular compression,
graph topology, and traversal policy.

**Status: PROVEN.** Formal existence result from definitions.

---

### Proposition A: Heterogeneous Dispersion Separation (EMPIRICAL REGIME)

In a multi-type manifold where entity classes form clustered sub-manifolds
{C_1, C_2, ..., C_k} in M:

A structured cross-type metapath T_meta = (v_1 ∈ C_1, v_2 ∈ C_2, ..., v_m ∈ C_k)
induces lower directional alignment than type-confined random walks:

    A(T_meta) < A(T_null)

Because uniform random walks are topologically biased to remain in the
maximum-cardinality cluster C_max, their vectors stay co-directional,
forcing A(T_null) → 1.0. Cross-type metapaths must traverse angular
boundaries between type clusters, producing directional reversals that
lower alignment.

**Status: EMPIRICAL REGIME LAW.** Validated on Hetionet v1.0
(p < 0.000001, N = 100 metapaths vs 1,000 random walks).
Alignment: curated = 0.414 ± 0.320, random = 0.883 ± 0.145.

---

### Proposition B: Scale-Free Backbone Alignment (EMPIRICAL REGIME)

In a graph dominated by scale-free hubs V_hub = {u ∈ V | deg(u) >> mean_deg}:

Valid functional cascades T_cascade preferentially traverse V_hub, occupying
the primary manifold axes:

    A(T_cascade) → 1.0
    α_sat(T_cascade) < α_sat(T_peripheral)

Random walks originating outside V_hub undergo diffusion across low-degree
peripheral nodes, requiring greater radial amplification to reach corpus-level
hubs.

**Status: EMPIRICAL REGIME LAW.** Validated on STRING-DB v12.0
(p < 0.001, N = 15 cascades vs 1,500 random walks).
Alignment: curated = 0.992 ± 0.004, random = 0.969.
Saturation: curated α_sat = 6.27 ± 2.74, random = 8.81 ± 2.37.
Average cascade degree: 1,182 ± 595 vs random walk degree: 390 ± 252.

---

### Regime Prediction Matrix

Theorem 1 and Propositions A-B produce a falsifiable prediction structure
for unseen graphs:

    If the graph has type-separated PCA clusters
        → curated cross-type paths produce LOWER alignment than random walks (Prop. A)

    If the graph has hub-density gradients without type separation
        → curated paths produce HIGHER alignment than random walks (Prop. B)

    If the graph has neither type boundaries nor hub gradients
        → kinematic separation is weak or absent; only centroid coherence separates

This prediction can be stated BEFORE KTS is run on a new graph, based solely
on the graph's node-type distribution and degree distribution.

---

## Part 6 — The K_mom Adjacent-Secants Null (FORMAL)

**Asymptotic null theorem under stated assumptions:**
For i.i.d. samples X, Y, Z drawn from a distribution with finite second
moments and approximately isotropic covariance σ²I in ℝ^D, the consecutive
displacement vectors
Δ₁ = Y − X and Δ₂ = Z − Y share the midpoint Y, creating negative
covariance:

    E[Δ₁ᵀΔ₂] = E[(Y − X)ᵀ(Z − Y)] = −E[YᵀY] = −Dσ²

Each displacement has expected squared norm:

    E[‖Δᵢ‖²] = 2Dσ²

Under norm concentration in high dimensions:

    E[cos(Δ₁, Δ₂)] ≈ −Dσ² / 2Dσ² = −1/2

The path momentum K_mom of a random i.i.d. sequence converges to −1/2
as D → ∞, under the assumptions of covariance isotropy, finite second
moments, and high-dimensional norm concentration. The result is
asymptotically independent of specific distribution shape and path length.

**Status: PROVEN.** Derived from shared-midpoint covariance.
Verified across:
- 7 dimensionalities (D = 10 to 5,000): error drops from 0.020 to 0.00002
- 7 of 8 distributions (Gaussian, uniform, exponential, Laplace, anisotropic,
  sphere, nonzero mean): all confirmed within 0.003 of −0.500
- Mixture of Gaussians deviates (K_mom = −0.44): inter-cluster sampling
  breaks the i.i.d. assumption, which is itself diagnostic
- 7 path lengths (3 to 50): mean stable, variance decreases
- kNN walks deviate to K_mom = −0.466: structured adjacency breaks i.i.d.
  The deviation from −0.500 on structured walks is what KTS measures.

---

## Part 7 — Kinematic Trajectory Spectroscopy (KTS)

KTS computes five metrics from ordered paths through M:

**(a) Path Momentum (K_mom):**

    K_mom(T) = (1/(N−2)) Σ_{i=1}^{N-2} (Δ⃗_i · Δ⃗_{i+1}) / (‖Δ⃗_i‖ · ‖Δ⃗_{i+1}‖)

    K_mom ≈  0.0  →  directional inertia (within-cluster)
    K_mom <  0.0  →  directional reversal (cross-domain tacking)
    K_mom ≈ −0.50 →  random walk null (Part 5 theorem)

**(b) Tortuosity:** τ = ‖v⃗_N − v⃗_1‖ / Σ‖v⃗_{i+1} − v⃗_i‖

**(c) Eccentricity:** Ratio of first to (second + third) singular values
of the centered path matrix. High = linear path, low = spherical.

**(d) TAV Magnitude:** ‖Σv⃗_i‖ / N — constructive interference.

**(e) Saturation (α_sat):** The radial scale at which nearest-neighbor
identity stabilizes when projecting Ψ(α) = α · v⃗_TAV outward.

Saturation is the only KTS metric that requires a populated manifold.
On the unit sphere (zero population structure), α_sat = 1.5 with zero
variance. Saturation measures basin mass — the population density along
the radial axis. Geometry without mass produces no saturation signal.

**(f) The kNN-graph momentum gap:**

    Δ_mom = K_mom(kNN walks) − K_mom(graph walks)

    Δ_mom ≈ 0    →  spatial/geometric graph (edges = feature proximity)
    Δ_mom > 0.3  →  cross-feature graph (edges cross feature boundaries)

This serves as a graph topology fingerprint that classifies construction
geometry without training.

**KTS is purely angular.** All metrics except saturation survive L2
normalization (4/4 graphs tested, all p < 0.001). The kinematic signatures
arise from directional structure, not norm variance.

**KTS operates on pure geometry.** Separation holds on featureless synthetic
point clouds with zero semantic content (6 distributions, all p < 10^-5).
The instrument measures geometric properties of path shape, not semantic
content.

**Status: EMPIRICAL.** Validated on 13 graphs (7 real-world + 6 synthetic),
all showing kinematic separation at p < 0.001 for at least one path-type
comparison.

---

## Part 8 — Intervention: Six-Policy Benchmark

### 8.1 Query-Anchored vs Target-Aware Expansion

VGSG trapping is specifically a property of **query-anchored** similarity
expansion — searching by cosine to the source/query without knowledge of
the target. When cosine expansion is directed toward a known target, it
largely escapes source-basin confinement because the gradient pulls
TOWARD the destination rather than circling the origin.

This distinction separates two retrieval paradigms:

    Query-concentric (unsupervised RAG):
        Expand by similarity to the query. No target knowledge.
        → VGSG trapping applies. Source basin confines the search.

    Target-directed (bridge finding, link prediction):
        Expand by similarity to a known target.
        → Largely escapes trapping. Cosine gradient is productive.

    Bidirectional (PRSM, multi-anchor):
        Expand from BOTH source and target.
        → Optimal. Frontiers meet at hub backbone.

### 8.2 Six-Policy Benchmark on PRSM Crystal

Tested 6 traversal policies under strict node-visit budgets on PRSM Crystal
(39,220 grains, 500 cross-corpus source-target pairs):

    Policy                      H=25    H=50   H=100   H=200   H=500
    ─────────────────────────  ──────  ──────  ──────  ──────  ──────
    Bidirectional PPR           91.2%   98.8%  100.0%  100.0%  100.0%
    Multi-Anchor (50:50)        88.6%   94.8%   97.0%   99.8%  100.0%
    Cosine Single-Source        73.2%   77.8%   82.2%   86.6%   92.4%
    MMR-Regularized Cosine      51.2%   59.6%   68.6%   74.0%   80.6%
    BFS (Topological)            2.4%    4.2%    8.0%   13.6%   32.8%
    Forward-Push PPR             2.4%    4.2%    8.0%   13.6%   32.8%

NOTE: Cosine Single-Source here expands by cosine similarity to the
TARGET (target-directed), not to the source (query-anchored). The
source-anchored version (Experiment 6) reached only 1.6% at budget 25.

### 8.3 Findings

1. **BFS is catastrophically weak on large sparse graphs.** 2.4% at
   budget 25, 32.8% at budget 500. Isotropic diffusion wastes budget
   across exponential neighborhood volume.

2. **Forward-Push PPR = BFS.** Identical at every budget. On an unweighted
   graph with uniform neighbor mass, FIFO push processes nodes in strict
   hop-distance order, producing the same expansion frontier as BFS.

3. **MMR (diversity-regularized) is WORSE than pure cosine.** 51.2% vs
   73.2% at budget 25. The degree penalty diverts search AWAY from
   high-degree hubs. On scale-free graphs (PRSM degree CV = 2.69),
   hubs are the ONLY cross-corpus transit corridors. Penalizing them
   is a direct experimental confirmation of Proposition B (Scale-Free
   Backbone Alignment): avoiding the hub backbone kills reachability.

4. **Target-directed cosine is strong but not complete.** 73.2% at budget
   25, 92.4% at budget 500. The residual 7.6-26.8% gap represents
   pairs where intermediate angular wells deflect single-source greedy
   trajectories away from the true graph bridge.

5. **Bidirectional strategies dominate.** Bidirectional PPR (91.2% at
   budget 25, 100% by budget 100) and Multi-Anchor cosine (88.6% at
   budget 25, 100% by budget 500) both exploit hub funneling: both
   frontiers rapidly ascend degree gradients into the core hub backbone,
   where they collide. Splitting the budget replaces O(b^H) search
   volume with 2·O(b^(H/2)).

### 8.4 Reconciliation with Experiment 6

Experiment 6 (source-anchored cosine) showed trapping: 1.6% at budget 25.
This benchmark (target-directed cosine) shows escape: 73.2% at budget 25.

The difference is the gradient direction:
- Cosine to SOURCE → circles the origin → trapped in source basin
- Cosine to TARGET → pulls toward destination → escapes source basin

VGSG trapping is therefore **query-anchored**: it occurs specifically when
the system searches by similarity to its starting point without knowledge
of where it needs to go. This is exactly the condition of standard RAG.

PRSM's waypoint architecture bypasses this by design: the user provides
the endpoints, and the system connects them through unknown geometry.
The path tracer is target-aware by construction.

### 8.5 Target Availability

Multi-anchor and bidirectional expansion require knowledge of the target.
This is directly applicable to:
- Known source-target bridge finding (PRSM's primary use case)
- Link prediction and relation completion
- Hypothesis evaluation between specified endpoints

It is NOT a direct substitute for open-ended RAG where the target is
unknown. For open-ended retrieval, it requires query decomposition into
multiple sub-concept anchors, or candidate target generation.

**Status: EMPIRICALLY VALIDATED.** Six policies tested, two bidirectional
strategies achieve 97-100% reachability. MMR failure confirms Proposition B.
Query-anchored trapping confirmed by comparison with Experiment 6.

---

## Part 9 — Universality

Parts 1–8 hold across all tested topological regimes:

    7 real-world graphs:  PRSM, Hetionet, STRING-DB, Cora, Amazon, DBLP, Grid Road
    6 synthetic clouds:   Isotropic, Anisotropic, Mixture, Power-law, Sphere, Low-dim

The kinematic separation is topology-dependent in its specific signature
(Propositions A and B predict the direction of separation from graph architecture)
but universal in its existence: structured paths never move like random walks
across any tested graph.

**Status: EMPIRICAL.** 13/13 graphs separate. The claim is bounded by
the tested regimes; we do not assert universality over all possible
graph-embedded metric spaces.

---

## Summary: Status of Proof

    Part 1 (π/2 null):           PROVEN       (concentration of measure)
    Part 2 (Γ):                  EMPIRICAL    (13 graphs; Γ_k significant on 2/3)
    Part 3 (κ):                  EMPIRICAL    (13 graphs; descriptive, not yet predictive)
    Part 4 (drainage):           EMPIRICALLY CONFIRMED + PROOF SKETCH  (5 graphs, 3 converging frameworks)
    Part 5, Theorem 1 (gap):     PROVEN       (formal existence from definitions)
    Part 5, Prop. A (hetero):    EMPIRICAL    (Hetionet, p < 0.000001)
    Part 5, Prop. B (hubs):      EMPIRICAL    (STRING-DB, p < 0.001)
    Part 6 (K_mom = −1/2):       PROVEN       (shared-midpoint covariance derivation)
    Part 7 (KTS):                EMPIRICAL    (13 graphs + 6 synthetic, all p < 0.001)
    Part 8 (intervention):       EMPIRICAL    (6-policy benchmark; BiPPR 91.2%, Multi 88.6% vs BFS 2.4% at H=25; MMR confirms Prop B)
    Part 9 (universality):       EMPIRICAL    (13/13 tested)

Three proven results: the π/2 null, the semantic gap existence, and the
K_mom = −1/2 null theorem.

One empirically confirmed result with proof sketch: progressive drainage
(Part 4) is confirmed across 5 graphs and supported by three converging
theoretical frameworks (size-biased regression, greedy routing zoom-in,
directional friendship paradox inversion). A formal unified proof remains
to be written; the proof sketch is available in `proof_progressive_drainage.md`.

All other parts are empirically supported across multiple independent
graphs but not formally proven.

**The conjecture's central mechanistic claim (v6):**

    SEMANTIC GRAVITY = PROGRESSIVE DRAINAGE × ANGULAR COMPRESSION
                       (under query-anchored expansion)

Progressive drainage (Part 4) causes cosine-biased expansion to descend
the degree hierarchy monotonically — the directional inverse of the
friendship paradox. Angular compression (Part 2) is a content-dependent
property of embedded semantic data that populates local basins. Neither
alone causes trapping. Together, under QUERY-ANCHORED expansion (where
the system searches by similarity to its starting point without target
knowledge), they produce VGSG: cosine search drains toward peripheral
specialists (drainage) while populated angular basins (compression)
consume the traversal budget.

Target-directed expansion (Part 8) largely escapes this trap because the
cosine gradient pulls toward the destination. Bidirectional strategies
achieve near-perfect reachability by exploiting hub-backbone funneling
from both ends simultaneously.

**Three industry assumptions falsified:**
1. "More edges = richer context" — densification increases basin trapping
2. "Diversity penalties improve coverage" — MMR collapses to 51.2% by
   avoiding the hub backbone (Proposition B confirmed)
3. "PageRank diffusion provides graph-aware expansion" — Forward-Push PPR
   = BFS (2.4%) on unweighted sparse graphs

---

## Open Problems

    1. Formal proof: unify three drainage frameworks into single theorem (would promote Part 4 to PROVEN)
    2. Formal bound on Γ as a function of D and embedding model properties
    3. Formal relationship between Γ, κ, wrong gradient, and trapping probability
    4. Whether κ independently predicts escape rate after controlling for Γ
    5. Whether Part 9 universality extends to all graph-embedded metric spaces
    6. Cosine-seeded intervention on heterogeneous graphs (Hetionet, STRING-DB)
    7. Whether non-kNN graphs (e.g., citation, co-purchase) with edges NOT derived
       from cosine proximity show the same or different curvature-cosine relationship

## Highest-Value Next Experiments

### A. Frontier Telemetry (Perplexity recommendation)
For every expansion policy and step, log:
- Candidate and selected-edge ORC distribution
- Candidate and selected-edge cosine distribution
- Cumulative graph-distance radius reached
- Whether a high-curvature exit was available but deprioritized
Show that cosine bias selects a systematically different structural edge
population from BFS and diversity-regularized controls.

### B. Ablation Matrix
    Vector shuffle:           Randomize embeddings across nodes → destroy geometry-topology alignment
    Degree-preserving rewire: Rewire edges randomly while preserving degrees → isolate topology
    Whitened embeddings:      Remove dominant PCA components → reduce Γ
    kNN-derived graph:        Build edges from cosine kNN → create geometry-aligned graph
These separate embedding-only, topology-only, and interaction effects.

### C. Stratified Interaction Test
Stratify the 6-policy benchmark by source-target cosine, Γ_k, local
conductance, degree, and graph distance. The decisive confirmatory
interaction is:

    ∂(Δ_reach) / ∂(angular separation) > 0

after conditioning on graph distance and local topology. If bidirectional
benefit increases with angular separation at fixed graph distance, the
causal chain is complete: geometry predicts failure, geometry-aware
intervention repairs it.

### D. Cross-Graph Replication
Run the 6-policy benchmark on Cora, Amazon, DBLP, and at least one
heterogeneous biomedical graph (Hetionet) and one interaction graph
(STRING-DB). Confirm that bidirectional dominance holds across graph types.

---

*Formalized by Michael Godat from empirical observations and engineering
of the PRSM system (VGSG: April 2026, KTS: August 2026). Mathematical
framework formalized August 14-16, 2026. Theorem 1 is a formal existence
result. Propositions A-B are empirically validated regime laws. The K_mom null
is a formal derivation. Intervention validated August 16, 2026. All results
tested on PRSM (39,220 nodes), Hetionet v1.0 (47,031 nodes), STRING-DB
v12.0 (19,699 nodes), Cora (2,708 nodes), Amazon Computers (13,752 nodes),
DBLP (17,716 nodes), Grid Road (10,000 nodes), and 6 synthetic distributions
(10,000 points each in ℝ^768).*
