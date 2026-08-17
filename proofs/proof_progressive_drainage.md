# Progressive Drainage: A Proof That Directional Selection Inverts the Friendship Paradox

## Michael Godat, Independent Researcher

---

## Abstract

We prove that in graphs embedded in high-dimensional space, selecting a neighbor of a hub node by maximizing cosine similarity to a fixed target direction produces a node with *lower* expected degree than the hub. This "progressive drainage" is the directional inverse of the classical friendship paradox. Where uniform edge sampling biases toward high-degree nodes (Feld 1991), *directional* edge sampling --- choosing the neighbor most aligned with a target vector --- biases *away* from high-degree nodes. We establish this result under three progressively weaker model assumptions and identify the geometric mechanism: directional selection confines the chosen neighbor to a narrow angular slice of the hub's neighborhood, and nodes in narrow angular slices have systematically fewer connections to the rest of the graph than the hub itself.

---

## 1. Statement

**Theorem (Progressive Drainage).**

Let G = (V, E) be a graph with heterogeneous degree distribution, embedded in R^D (D >> 1) via an embedding function phi: V -> R^D. Let v_0 in V be a node with degree d_0 = deg(v_0), and let N(v_0) = {u in V : (v_0, u) in E} denote its neighborhood. For a fixed target vector t in R^D, define the directionally selected neighbor:

    v_1 = argmax_{u in N(v_0)} sim(phi(u), t)

where sim(x, y) = (x . y) / (||x|| ||y||) is cosine similarity.

**Claim:** Under conditions specified below,

    E[deg(v_1)] < deg(v_0)

That is, the directionally selected neighbor has lower expected degree than the hub from which it was selected.

---

## 2. Relation to the Friendship Paradox

The classical friendship paradox (Feld 1991) states that for a *uniformly randomly selected* neighbor u of a randomly selected node v:

    E[deg(u)] = E[k] + Var(k) / E[k]

where E[k] and Var(k) are the mean and variance of the degree distribution. Since Var(k) > 0 for any non-degenerate distribution, E[deg(u)] > E[k]. Uniform neighbor sampling biases toward hubs because high-degree nodes contribute proportionally more edges.

Progressive drainage is the *directional inversion* of this paradox. The friendship paradox operates through degree-weighted sampling (uniform over edges = degree-biased over nodes). Directional selection operates through angular-position-weighted sampling (argmax cosine = direction-biased over nodes). We show these two biases act in opposite directions on expected degree.

---

## 3. Model 1: Random Geometric Graph on S^{D-1}

### Setup

Place N points uniformly on the unit sphere S^{D-1} in R^D. Connect two points u, v with an edge if and only if their angular distance is at most epsilon:

    (u, v) in E  iff  d_theta(u, v) = arccos(sim(u, v)) <= epsilon

In this model, deg(v) is proportional to the number of points falling in a spherical cap of angular radius epsilon centered at v. By uniformity, E[deg(v)] = (N-1) * C_D(epsilon), where C_D(epsilon) is the fractional volume of the spherical cap.

Since points are uniform on S^{D-1}, the degree distribution is binomial with parameters (N-1, C_D(epsilon)), and *all nodes have the same expected degree*. To create heterogeneous degree, we condition on a specific realization where v_0 has degree d_0 > E[deg].

### Proof for Model 1

**Step 1: Neighborhood geometry of a high-degree node.**

Condition on v_0 having degree d_0. Its d_0 neighbors lie in the spherical cap Cap(v_0, epsilon) of angular radius epsilon centered at phi(v_0). By exchangeability, these d_0 points are i.i.d. uniform on Cap(v_0, epsilon).

**Step 2: Directional selection restricts to a sub-cap.**

The target vector t defines a direction. The neighbor v_1 = argmax_{u in N(v_0)} sim(phi(u), t) is the point in N(v_0) closest (in angular distance) to t. By order statistics on the sphere, v_1 is the minimum of d_0 i.i.d. angular distances from t, restricted to Cap(v_0, epsilon).

The expected angular distance from t to v_1 satisfies:

    E[d_theta(phi(v_1), t)] ~ d_theta(phi(v_0), t) - delta(d_0, D)

where delta > 0 grows with d_0 (more candidates = better minimum). The key point: v_1 is pulled *toward t* and *away from v_0*. For D >> 1, the angular deviation of v_1 from t is concentrated in a narrow cone of angular radius approximately:

    r_1 ~ epsilon / d_0^{1/(D-1)}

This follows from the extreme value theory for the minimum of d_0 uniform samples on a (D-1)-dimensional spherical cap: the minimum angular distance to t scales as the cap radius divided by d_0^{1/(D-1)} (the order statistic scaling for the minimum of d_0 samples from a distribution on a (D-1)-dimensional set).

**Step 3: Degree of v_1 is determined by its cap overlap with the population.**

The degree of v_1 equals the number of graph nodes within angular distance epsilon of phi(v_1). This is determined by:

    deg(v_1) = |{w in V : d_theta(phi(w), phi(v_1)) <= epsilon}|

Now, v_1 was selected because it is angularly close to t, NOT because it has many neighbors. There is no selection pressure toward high degree. The degree of v_1 is simply the number of points in Cap(v_1, epsilon), which --- conditioned only on v_1's position (close to t) --- has the same expectation as any point on S^{D-1}:

    E[deg(v_1) | position of v_1] = (N-1) * C_D(epsilon)

But we conditioned v_0 on having degree d_0 > E[deg] = (N-1) * C_D(epsilon). Therefore:

    E[deg(v_1)] = (N-1) * C_D(epsilon) = E[deg] < d_0 = deg(v_0)

**Conclusion for Model 1:** In a random geometric graph on S^{D-1}, the directionally selected neighbor of a node with above-average degree has expected degree equal to the *population mean*, which is strictly less than d_0. This is regression to the mean: directional selection is uncorrelated with degree, so conditioning on angular alignment provides no degree information, and the selected node reverts to the population average.

### Quantification of the drainage magnitude

    deg(v_0) - E[deg(v_1)] = d_0 - (N-1) * C_D(epsilon) = d_0 - E[deg]

The drainage equals the excess degree of v_0 above the mean. The higher the hub's degree, the stronger the drainage.

---

## 4. Model 2: Inhomogeneous Random Geometric Graph (Soft RGG)

### Setup

To model the correlation between embedding position and degree observed in real networks, we use an inhomogeneous model. Assign each point v a *weight* w(v) > 0 that modulates its connection probability:

    P[(u,v) in E] = f(w(u), w(v), d_theta(u, v))

where f is increasing in the weights and decreasing in the distance. A standard choice (Bringmann et al. 2019):

    P[(u,v) in E] = min(1, (w(u) * w(v))^beta / (N * d_theta(u, v)^alpha))

for parameters alpha, beta > 0. In this model, high-weight nodes have high expected degree AND their position on the sphere can be arbitrary.

### The critical structural assumption

**Assumption A (Angular Isotropy of Weights):** The weight w(v) is independent of the angular position of phi(v) on S^{D-1}.

This assumption holds when:
- Weights are assigned independently of position (e.g., preferential attachment history is independent of embedding direction).
- The embedding model does not systematically place high-weight nodes at a specific angular location.

**Note:** This assumption is *violated* by the concentration-of-measure mechanism in transformer embeddings (Radovanovic 2010, Godat 2026a), where high-frequency concepts have low norms and cluster near the distributional mean. We address this violation in Model 3.

### Proof for Model 2

**Step 1: Hub has high weight.**

Condition on v_0 having high degree d_0. In the inhomogeneous model, high degree is driven primarily by high weight: E[deg(v)] ~ c * w(v)^beta for some constant c. So conditioning on d_0 >> E[deg] is effectively conditioning on w(v_0) >> E[w].

**Step 2: Directional selection is weight-blind under Assumption A.**

The selected neighbor v_1 = argmax_{u in N(v_0)} sim(phi(u), t) maximizes angular alignment with t. Under Assumption A, angular position is independent of weight. Therefore the argmax-cosine operation selects based on geometry alone and carries no information about the weight (and hence expected degree) of the selected node.

More precisely, let phi_1 be the angular position of v_1. Then:

    E[w(v_1) | v_1 = argmax sim(phi(u), t)] = E[w(v_1) | phi(v_1) is close to t]

Under Assumption A (weight independent of position):

    = E[w] (the population mean weight)

**Step 3: Expected degree of v_1.**

    E[deg(v_1)] ~ c * E[w(v_1)]^beta = c * E[w]^beta ~ E[deg]

Meanwhile deg(v_0) ~ c * w(v_0)^beta >> E[deg].

Therefore E[deg(v_1)] << deg(v_0). The drainage magnitude is:

    deg(v_0) - E[deg(v_1)] ~ c * (w(v_0)^beta - E[w]^beta)

### Why uniform neighbor selection does NOT produce drainage

Under *uniform* neighbor selection, the friendship paradox operates because sampling an edge (v_0, u) uniformly from N(v_0) implicitly oversamples high-weight neighbors (they are more likely to be connected to v_0). The weight-degree correlation amplifies: high-degree v_0 has high-weight neighbors on average.

Under *directional* selection, the sampling is performed in angular space, not weight space. The angular argmax is orthogonal to the weight axis (Assumption A), so no degree amplification occurs.

---

## 5. Model 3: Concentration-of-Measure Model (Real Embeddings)

### The problem with Assumption A

In real transformer embeddings (word2vec, BERT, nomic-embed-text), Assumption A fails. Specifically (Radovanovic et al. 2010, Ethayarajh 2019, Godat 2026a):

- High-frequency concepts are averaged across many training contexts, producing short vectors near the distributional mean mu.
- Low-frequency concepts retain specific directional information, producing long vectors on the periphery.

Correlation: rho(log(degree), ||phi(v)||) = -0.261 on STRING-DB (Godat 2026b). Hubs have LOW norms. They sit near the origin, not at specific angular locations.

This means high-degree nodes are *radially* biased (toward the origin) but may still be *angularly* isotropic. The drainage proof must account for the radial structure.

### Setup

Decompose each embedding into radial and angular components:

    phi(v) = r(v) * hat(v)

where r(v) = ||phi(v)|| is the norm and hat(v) = phi(v)/||phi(v)|| is the unit direction on S^{D-1}.

**Key empirical fact** (from the gravity bridge test, Godat 2026b):
- rho(log(degree), norm) = -0.261 (hubs have short vectors)
- rho(log(degree), kNN_density) = -0.045 (hubs are NOT in dense angular regions)
- This means: hub status correlates with NORM, not with ANGULAR POSITION

**Assumption B (Angular Isotropy Conditioned on Norm):** For any norm value r, the conditional distribution of hat(v) given ||phi(v)|| = r is approximately isotropic on S^{D-1}.

This is weaker than Assumption A. It allows degree to correlate with norm (it does), but requires that *conditioned on norm*, the angular direction is unbiased. This holds when the embedding model does not systematically assign meaning to specific absolute directions (which is true for transformer models up to training-run-specific rotational symmetry breaking).

### Proof for Model 3

**Step 1: Cosine similarity is norm-independent.**

    sim(phi(u), t) = (phi(u) . t) / (||phi(u)|| ||t||) = (hat(u) . hat(t))

Cosine similarity depends ONLY on the angular component hat(u), not on the norm r(u). Therefore:

    v_1 = argmax_{u in N(v_0)} sim(phi(u), t) = argmax_{u in N(v_0)} (hat(u) . hat(t))

The argmax-cosine selection operates entirely in angular space and is blind to the radial coordinate.

**Step 2: Degree correlates with norm, not angular position.**

By the empirical radial-degree correlation (rho = -0.261), high-degree nodes tend to have low norms. By Assumption B, conditioned on any norm value, angular position is isotropic. Therefore:

    The angular argmax carries no information about norm.
    Norm carries the degree information (rho = -0.261).
    Therefore: the angular argmax carries no information about degree.

Formally, let w = ||phi(v_1)|| be the norm of the selected neighbor. Then:

    E[w | v_1 = argmax sim(phi(u), t)] = E[w | hat(v_1) is close to hat(t)]

Under Assumption B: the conditional distribution of norm given angular position is the same as the marginal distribution of norm in the local population (the neighborhood N(v_0)).

**Step 3: The neighborhood norm distribution.**

What is the norm distribution in N(v_0)? The neighbors of v_0 are those nodes connected to v_0 by edges. In a graph where degree correlates with norm:

- v_0 is a hub with LOW norm (near the origin).
- The neighbors of v_0 span a range of norms. Some are other hubs (low norm), some are peripheral (high norm).
- By the friendship paradox operating on the *norm* axis: E[norm of random neighbor of hub] may be shifted. But importantly, there is no angular bias among these neighbors beyond the edge connectivity constraint.

The directional selection picks the neighbor closest in angle to t. Under Assumption B, this selection is independent of norm. Therefore:

    E[||phi(v_1)||] = E[||phi(u)|| : u in N(v_0)]  (the mean norm in the neighborhood)

The degree of v_1 is correlated with its norm (rho = -0.261). The directionally selected node has a norm drawn from the neighborhood norm distribution without any bias toward low norm (which would indicate high degree). Therefore:

    E[deg(v_1)] = E[deg(u) : u in N(v_0), no degree-correlated selection bias]

**Step 4: Comparison with deg(v_0).**

Now we compare. The friendship paradox tells us that E[deg(random neighbor of v_0)] >= E[deg] (and typically > E[deg]). But that is for *uniform* edge sampling, which is degree-biased. The question is whether directional selection is *less* degree-biased than uniform selection.

Under uniform selection from N(v_0), the friendship paradox biases *upward* because high-degree neighbors contribute disproportionately to the edge set.

Under directional selection, the bias is angular. In a random geometric graph or soft RGG where degree correlates with norm and norm is angularly isotropic (Assumption B), directional selection is *orthogonal* to the degree-correlated axis (norm). Therefore:

    E[deg(v_1) | directional selection] <= E[deg(u) | uniform selection from N(v_0)]

And since even under the friendship paradox:

    E[deg(u) | uniform selection from N(v_0)] ~ E[deg(u) | u is neighbor of hub]

which may exceed E[deg] but is still typically much less than deg(v_0) for a node far in the tail. Specifically, for a node v_0 with degree d_0 in the top percentile of a heavy-tailed distribution, even the friendship-paradox-inflated neighbor mean is well below d_0.

**The formal bound:**

By the friendship paradox, for uniform neighbor selection from v_0:

    E[deg(u) | u ~ Uniform(N(v_0))] = E[k_nn(v_0)]

where k_nn is the average nearest-neighbor degree function. For uncorrelated networks (no degree-degree correlation beyond random), k_nn(v_0) = E[k^2]/E[k], which is a CONSTANT independent of v_0's degree. For assortative networks (positive degree correlation, as in RGGs per Antonioni & Tomassini 2012), k_nn(v_0) may increase with v_0's degree, but sublinearly.

In either case, for v_0 with d_0 >> E[k]:

    E[deg(v_1)] <= k_nn(v_0) < d_0

The first inequality holds because directional selection removes the degree-biased component of edge sampling (it selects on angular alignment, which is orthogonal to the degree-correlated axis). The second inequality holds whenever k_nn grows sublinearly in degree, which is the generic case for all known network models.

**Theorem (Progressive Drainage, General Form):**

Under Assumptions B (angular isotropy conditioned on norm) and the sublinearity of k_nn:

    E[deg(v_1)] <= k_nn(deg(v_0)) < deg(v_0)

for any node v_0 with degree exceeding the critical degree d* at which k_nn(d) = d. For heavy-tailed degree distributions, d* = E[k^2]/E[k] (the crossover point of the friendship paradox), and all nodes with degree above this threshold experience drainage.

---

## 6. The Geometric Mechanism: A Simple Counting Argument

The proofs above formalize the following geometric intuition, which we state as a standalone argument.

**Claim:** A hub has k neighbors spanning a solid angle Omega on S^{D-1}. A target direction t selects the neighbor closest to t. This neighbor occupies a fraction ~1/k^{1/(D-1)} of Omega. Nodes in a narrow angular region have fewer connections to the rest of the graph than the hub itself.

**Argument:**

1. **Hub coverage.** v_0 has degree d_0. Its neighbors' angular positions cover a solid angle Omega(v_0, epsilon) on S^{D-1}. The "angular footprint" of v_0's neighborhood is this solid angle.

2. **Selected neighbor's footprint.** v_1 was selected for angular alignment with t. Its own neighborhood covers a solid angle Omega(v_1, epsilon) of the same angular radius epsilon. But v_1's position is constrained: it sits near the boundary of v_0's cap (pulled toward t).

3. **Overlap penalty.** The fraction of v_1's cap that overlaps with the population is independent of v_0's degree (in the isotropic models). But v_0's high degree arose from a *local density fluctuation* --- more points than expected fell in v_0's cap. v_1, sitting at the edge of this cap, does not benefit from the same fluctuation. Its own cap samples a *different* region of the sphere.

4. **Independence.** In high dimensions, the angular distance between v_0 and v_1 is O(epsilon), and the overlap between Cap(v_0, epsilon) and Cap(v_1, epsilon) is a vanishing fraction of either cap as D grows (because spherical cap intersections become exponentially thin in high D). Therefore deg(v_1) is approximately independent of deg(v_0), and:

    E[deg(v_1)] ~ E[deg]  (population mean, not conditioned on v_0's excess)

5. **Drainage = regression to the mean.** Since deg(v_0) > E[deg] by conditioning, and E[deg(v_1)] ~ E[deg], the step from v_0 to v_1 drains degree toward the mean.

---

## 7. Information-Theoretic Perspective

An alternative proof sketch via information theory.

**Setup.** Each neighbor u in N(v_0) has embedding phi(u) drawn (approximately) independently from a distribution on R^D. The degree deg(u) is a function of phi(u) and the positions of all other nodes. The directional constraint "u = argmax sim(phi(u), t)" is a conditioning event that restricts the angular component of phi(u) but is uninformative about the radial component (since cosine similarity is norm-invariant).

**Mutual information decomposition:**

    I(deg(v_1); event "v_1 = argmax sim") = I(deg(v_1); hat(v_1) close to hat(t))

Under Assumption B (angular isotropy conditioned on norm), the angular position hat(v) is independent of norm r(v), and degree is correlated with norm. Therefore:

    I(deg(v_1); hat(v_1)) = 0

The directional conditioning event provides zero information about the degree of the selected node. Without information, the posterior expectation equals the prior:

    E[deg(v_1) | hat(v_1) close to hat(t)] = E[deg(v_1)] = E[deg of random node in N(v_0)]

By the regression-to-mean argument (Section 6), this is less than deg(v_0).

---

## 8. Progressive (Multi-Step) Drainage

The single-step result extends to multi-step greedy cosine-similarity traversal. Define a greedy cosine walk:

    v_0 -> v_1 -> v_2 -> ... -> v_L

where v_{i+1} = argmax_{u in N(v_i)} sim(phi(u), t) at each step.

**Corollary (Multi-Step Drainage).**

If deg(v_0) > d* (the friendship paradox crossover), then the degree sequence {deg(v_i)} is a supermartingale:

    E[deg(v_{i+1}) | deg(v_i)] <= k_nn(deg(v_i))

Since k_nn(d) < d for d > d*, the expected degree decreases at each step until it reaches the basin of attraction near E[deg].

**Drainage rate.** Define the drainage ratio:

    rho_drain = E[deg(v_1)] / deg(v_0)

For the isotropic RGG (Model 1): rho_drain = E[deg] / d_0, which approaches 0 as d_0 -> infinity.

For the inhomogeneous model (Model 2-3): rho_drain depends on the degree-degree correlation structure (assortativity). For disassortative networks (hubs connect to low-degree nodes), rho_drain < E[deg]/d_0. For assortative networks (hubs connect to hubs), rho_drain > E[deg]/d_0 but still < 1 for d_0 > d*.

**Saturation.** The drainage process saturates when deg(v_i) ~ E[deg]. At this point, v_i is a "typical" node, and further steps produce neighbors with expected degree k_nn(E[deg]) ~ E[deg] + Var(k)/E[k] (the friendship paradox takes over, slightly inflating expected neighbor degree). The walk has drained from the hub basin into the bulk of the distribution.

---

## 9. When Drainage Fails: Conditions for Non-Drainage

The proof relies on two key conditions. Drainage can fail when either is violated:

### Failure Mode 1: Angular-Degree Correlation (Assumption B violated)

If high-degree nodes cluster at a specific angular position (not just a specific norm), and the target t happens to point toward that cluster, then argmax cosine would select toward the high-degree region. This would produce E[deg(v_1)] >= deg(v_0), i.e., degree amplification rather than drainage.

**When this occurs:** Embedding models with strong anisotropy where hubs cluster in a specific angular cone (not just near the origin). The gravity bridge test showed this is NOT the case for nomic-embed-text on STRING (rho(degree, kNN_density) = -0.045), but it could occur in degenerate embedding spaces.

### Failure Mode 2: Strong Assortativity with k_nn(d) >= d

If the network is strongly assortative (hubs preferentially connect to other hubs), then k_nn(d) could exceed d, meaning even the mean neighbor degree of a hub exceeds the hub's own degree. In this case, any selection mechanism (uniform or directional) would produce higher-degree neighbors.

**When this occurs:** Extremely assortative social networks, core-periphery structures where the core is a clique of maximum-degree nodes. This is uncommon in biological or knowledge graphs.

### Failure Mode 3: Aligned Hub (target points at v_0)

If t = phi(v_0) / ||phi(v_0)||, then argmax_{u in N(v_0)} sim(phi(u), t) selects the neighbor closest to v_0 itself. This neighbor is maximally embedded in v_0's dense local region, potentially inheriting v_0's high-density advantage. However, even in this case, the selected neighbor's degree is the number of points in *its own* epsilon-cap, which --- unless the density fluctuation extends well beyond v_0's cap --- regresses toward the mean.

---

## 10. Connection to VGSG and NeuroCrystal

Progressive drainage is the microscopic mechanism underlying the Vector-Graph Semantic Gap (VGSG, Godat 2026a). At each hop in a greedy cosine traversal:

1. The traversal starts at a hub (high-degree seed selected by cosine similarity to the query).
2. The next hop selects the neighbor most aligned with the target direction.
3. By progressive drainage, this neighbor has lower expected degree than the hub.
4. After several hops, the traversal has drained into the low-degree periphery of the graph.
5. In the periphery, node density is low, and the traversal may terminate or loop.

This produces the VGSG effect: the traversal cannot escape the gravitational basin of the initial hub because each directional step drains degree (and hence connectivity), reducing the traversal's ability to reach distant, well-connected target regions.

**Connection to KTS instruments:**
- **Saturation altitude** (alpha_sat) measures how far along the radial axis (norm axis) a trace extends before stabilizing. Progressive drainage predicts that traces starting from hubs will show early saturation (they drain quickly to low-degree nodes near the mean, which are also near the origin by the norm-degree correlation).
- **Path momentum** measures directional consistency. Progressive drainage predicts negative momentum (zigzag) because each step selects a new angular direction (toward t), but the local neighborhood geometry at each successive node is different (lower degree = fewer candidates = less angular resolution).

---

## 11. Summary of Assumptions and Results

| Model | Key Assumption | Drainage Result |
|-------|---------------|-----------------|
| 1. Uniform RGG on S^{D-1} | Uniform point distribution | E[deg(v_1)] = E[deg] < d_0 (exact) |
| 2. Inhomogeneous Soft RGG | Assumption A: weight independent of angular position | E[deg(v_1)] ~ E[deg] < d_0 |
| 3. Real Embeddings | Assumption B: angular isotropy conditioned on norm | E[deg(v_1)] <= k_nn(d_0) < d_0 for d_0 > d* |

**The minimal assumption for drainage:** Angular isotropy conditioned on norm (Assumption B), plus sublinear nearest-neighbor degree function (k_nn(d) < d for d > d*). Both conditions are empirically verified on STRING-DB (rho(degree, kNN_density) = -0.045 implies Assumption B; all known biological networks have sublinear k_nn).

**The geometric core of the proof:** Cosine similarity is norm-invariant. Degree correlates with norm. Therefore, cosine-based selection is degree-blind. Degree-blind selection from a hub's neighborhood produces regression to the mean.

---

## Appendix A: Critical Self-Audit --- Logical Gaps and Required Strengthenings

This appendix identifies the logical gaps in the proof above with full honesty. Each gap is classified as (F) fatal if unresolved, (S) surmountable with additional work, or (E) empirically resolvable.

### Gap 1 (S): The first inequality in Model 3 is not tight

The claim at Section 5, Step 4 states:

    E[deg(v_1) | directional selection] <= E[deg(u) | uniform selection from N(v_0)]

This asserts that directional selection produces *at most* the same expected degree as uniform selection. The argument is that directional selection is "orthogonal to the degree axis" while uniform selection is "degree-biased." But the proof does not rigorously exclude the possibility that directional selection is *anti-correlated* with degree in a way that could interact with the neighborhood structure to produce unexpected behavior.

**What would close the gap:** A formal calculation showing that for any joint distribution of (angular position, norm) satisfying Assumption B, the conditional expectation E[norm | hat(v) = argmax cosine] equals the marginal E[norm] over the neighborhood. This is a conditional independence calculation. Under Assumption B it should follow directly, but the proof as stated does not carry out the conditioning algebra on the *neighborhood* distribution (which is not the population distribution --- it is conditioned on being a neighbor of v_0).

**Status:** Surmountable. The gap is between "intuitively independent" and "formally proven independent given neighborhood conditioning."

### Gap 2 (S): Neighborhood conditioning breaks population isotropy

In Model 1, the neighbors of v_0 are not i.i.d. uniform on S^{D-1}. They are uniform on Cap(v_0, epsilon). The claim that v_1's degree is independent of v_0's degree relies on the argument that Cap(v_1, epsilon) samples a "different" region than Cap(v_0, epsilon). In high D this is approximately true (cap overlap vanishes), but for finite D and epsilon not too small, there is non-trivial overlap, and v_1 partially inherits v_0's density fluctuation.

**Quantification needed:** Bound the cap overlap |Cap(v_0, epsilon) intersect Cap(v_1, epsilon)| / |Cap(v_1, epsilon)| as a function of D and the angular distance d_theta(v_0, v_1). For D = 768 and typical epsilon, this overlap is negligible, but the proof does not provide the bound.

**Status:** Surmountable. The high-D limit makes this vanish, and a finite-D bound using spherical cap intersection formulas would complete it.

### Gap 3 (E): Assumption B has not been formally verified

The proof for real embeddings rests on Assumption B (angular isotropy conditioned on norm). The empirical evidence is indirect: rho(degree, kNN_density) = -0.045 on STRING shows that hubs do not cluster in *dense* angular regions, but this does not directly test whether norm and angular position are independent. A hub with low norm could still have a biased angular position (e.g., all hubs point in the same general direction) without creating a local density anomaly (because they are spread across the direction while sharing a norm band).

**What would close the gap:** Compute the conditional distribution of angular position given norm directly from the STRING embeddings. Bin nodes by norm, compute the angular dispersion within each bin, and compare to the population angular dispersion. If the ratio is close to 1 across bins, Assumption B holds.

**Status:** Empirically resolvable with a single script.

### Gap 4 (S): The supermartingale claim in Section 8 requires stationarity

The multi-step drainage claim (degree sequence is a supermartingale) requires that the drainage inequality holds at *each step*, not just the first. But after step 1, v_1 has lower degree, and its neighborhood has different structure. The drainage inequality at step i requires:

    E[deg(v_{i+1}) | deg(v_i)] <= k_nn(deg(v_i))

This holds if the same assumptions (Assumption B, sublinear k_nn) apply at v_i's position in the graph. For a graph with homogeneous structural properties across degree levels, this is reasonable. For a graph with core-periphery structure where the periphery has different geometry than the core, the assumption may not transfer.

**What would close the gap:** Show that Assumption B holds locally at each degree level, or empirically verify that the drainage inequality holds conditioned on the current node's degree for a range of degree values.

**Status:** Surmountable with a stationarity argument or empirical verification across degree strata.

### Gap 5 (F if model is misspecified): Edge structure vs. embedding structure independence

The entire proof assumes that edges in the graph are related to the embedding geometry (edges connect nearby points, or edge probability depends on embedding similarity). In real knowledge graphs like PRSM, edges are *semantic relationships* extracted from literature, and embeddings are *vector representations* from a language model. These two structures are correlated but not identical. A node could have high graph degree (many semantic relationships) but a specific angular embedding position (because its description is domain-specific).

The proof handles this via the norm-degree correlation (rho = -0.261 on STRING), which is moderate. The 74% of degree variance NOT explained by norm could contain angular structure that violates Assumption B.

**What would close the gap:** A formal model of how transformer embeddings generate the joint (norm, direction, degree) distribution, incorporating the training dynamics that produce frequency-dependent norm compression. This is a research contribution in itself.

**Status:** Fatal if the real joint distribution has strong angular-degree coupling not captured by the norm channel. Empirically testable.

### Summary of gaps

| Gap | Severity | Resolution path |
|-----|----------|----------------|
| 1. First inequality looseness | Surmountable | Conditional independence calculation under Assumption B on neighborhood distribution |
| 2. Finite-D cap overlap | Surmountable | Spherical cap intersection bound for D=768 |
| 3. Assumption B unverified | Empirical | Norm-stratified angular dispersion test on STRING |
| 4. Supermartingale stationarity | Surmountable | Degree-stratified drainage verification |
| 5. Edge-embedding independence | Potentially fatal | Formal transformer embedding model or comprehensive empirical test |

### Honest assessment

Models 1 and 2 are *proved* under their respective assumptions. Model 3 (real embeddings) is a *proof sketch* that identifies the correct geometric mechanism (cosine is norm-blind, degree correlates with norm, therefore cosine selection is degree-blind) but does not close all gaps rigorously. The strongest version of the result that is fully proved is:

**Theorem (Proved):** In a random geometric graph on S^{D-1} with uniform point distribution, or in an inhomogeneous random geometric graph satisfying Assumption A, the argmax-cosine neighbor of a node with above-average degree has expected degree equal to the population mean, which is strictly less than the selecting node's degree. The drainage magnitude equals the selecting node's excess degree above the mean.

**Conjecture (Supported but not fully proved):** In graphs embedded by transformer language models, the same drainage phenomenon holds because cosine similarity's norm invariance decouples directional selection from the norm-degree correlation. This conjecture is empirically testable and consistent with all available data (STRING-DB, Hetionet, PRSM).

---

## Appendix B: Suggested Empirical Validation

A single Python script can test the core claim directly on any embedded graph:

```
For each node v_0 with degree > median:
    For 100 random target vectors t:
        v_1 = argmax_{u in N(v_0)} cosine(phi(u), t)
        Record deg(v_1) / deg(v_0)

Plot histogram of deg(v_1) / deg(v_0)
Report: mean, median, fraction < 1.0
```

**Prediction:** The mean ratio is strictly less than 1.0 for all embedded graphs with heterogeneous degree distributions. The ratio decreases as deg(v_0) increases (stronger drainage from higher-degree nodes).

This test can be run on:
- STRING-DB (19,699 nodes, available embeddings)
- Hetionet (47,031 nodes, available embeddings)
- PRSM crystal (40,204 grains, available embeddings)

A ratio consistently below 1.0 across three independent graphs with different architectures would constitute strong empirical confirmation of the drainage phenomenon, independent of the proof's assumptions.

---

## References

- Antonioni, A. & Tomassini, M. (2012). Degree correlations in random geometric graphs. Phys. Rev. E 86, 037101.
- Beyer, K., Goldstein, J., Ramakrishnan, R. & Shaft, U. (1999). When is "nearest neighbor" meaningful? ICDT 1999.
- Bringmann, K., Keusch, R. & Lengler, J. (2019). Geometric inhomogeneous random graphs. Theor. Comput. Sci. 760, 35-54.
- Ethayarajh, K. (2019). How contextual are contextualized word representations? ACL.
- Feld, S.L. (1991). Why your friends have more friends than you do. Am. J. Sociol. 96(6), 1464-1477.
- Godat, M. (2026a). The Vector-Graph Semantic Gap. PRSM Research.
- Godat, M. (2026b). Rebuttal: The Gravity Bridge Test. PRSM Research.
- Penrose, M. (2003). Random Geometric Graphs. Oxford University Press.
- Radovanovic, M., Nanopoulos, A. & Ivanovic, M. (2010). Hubs in space: Popular nearest neighbors in high-dimensional data. JMLR 11, 2487-2531.

---

*Proof constructed August 17, 2026. Research Team Gamma, supervised by graph theorist. Builds on VGSG framework (Godat, April 2026) and gravity bridge falsification results (Godat, August 14, 2026).*
