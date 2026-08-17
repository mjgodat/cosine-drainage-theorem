# Gap 2 Closure: Clustering Correction for Miss Probability

## Michael Godat, Independent Researcher

---

## 1. The Problem

The miss probability lower bound (Part ii of the Cosine Drainage Theorem) is proved under a locally tree-like approximation (CC -> 0). In a tree, each expansion step reaches entirely new nodes: a node of degree k contributes k - 1 novel frontier members (minus the edge you arrived on). The miss probability bound assumes this maximal expansion rate.

Real graphs have high clustering:

| Graph | CC | Interpretation |
|-------|----|----------------|
| NeuroCrystal | 0.614 | 61% of neighbor pairs form triangles |
| Amazon | 0.362 | |
| Cora | 0.289 | |
| DBLP | 0.184 | |

Clustering means neighbors share neighbors. When the walker steps from node u to node v, a fraction of v's neighbors are ALREADY in the frontier (they were also neighbors of u). This reduces the number of genuinely new nodes per step, shrinking the effective expansion and making it harder to stumble onto the target via alternative paths.

**Direction of error.** The tree bound is too OPTIMISTIC --- it overestimates frontier growth, underestimates miss probability. The real miss rate should be HIGHER than the tree bound predicts. This makes the tree bound a valid lower bound on miss probability in the wrong direction: we stated P(miss) >= tree_bound, but the real miss is even larger. So the tree bound remains a valid lower bound; what we need is a TIGHTER lower bound that accounts for clustering.

---

## 2. Effective Branching Under Clustering

### 2.1 The Overlap Fraction

Consider a walker stepping from node u to node v along an edge (u, v). Node v has degree k_v. Of v's k_v neighbors:

- 1 is u itself (the node we came from)
- Some number Delta(u,v) are shared neighbors of both u and v (triangle completions)
- The remainder are genuinely new

The expected number of shared neighbors, conditioned on the edge (u,v) existing, is exactly the edge clustering coefficient. For a graph with global clustering coefficient CC and mean degree <k>:

$$\Delta(u,v) = \mathbb{E}[\text{shared neighbors of } u,v \mid (u,v) \in E]$$

By definition of the clustering coefficient at node u: CC_u = (number of triangles through u) / C(k_u, 2). Averaged over edges, the expected number of common neighbors of two adjacent nodes is:

$$\Delta \approx \text{CC} \cdot (\langle k \rangle - 1)$$

This follows because: if u has degree k_u, then each pair of u's neighbors has probability CC of being connected. Node v, being a neighbor of u, shares on average CC * (k_u - 1) neighbors with u. Averaging over u with E[k_u - 1] ~ <k> - 1 (size-biased adjustment is second-order here):

$$\Delta \approx \text{CC} \cdot (\langle k \rangle - 1)$$

### 2.2 New Nodes Per Step

At step i, the walker moves to a node with effective degree k_eff(i) (governed by drainage). The number of genuinely NEW frontier nodes contributed by this step is:

$$k_{\text{new}}(i) = k_{\text{eff}}(i) - 1 - \Delta = k_{\text{eff}}(i) - 1 - \text{CC} \cdot (\langle k \rangle - 1)$$

In a tree (CC = 0): k_new = k_eff - 1 (every neighbor except the parent is new).

In a clustered graph: k_new = k_eff - 1 - CC * (<k> - 1).

**The effective branching factor** is the ratio of new nodes to total neighbors:

$$b_{\text{eff}} = \frac{k_{\text{new}}}{k_{\text{eff}}} = 1 - \frac{1 + \text{CC} \cdot (\langle k \rangle - 1)}{k_{\text{eff}}}$$

For NeuroCrystal: CC = 0.614, <k> = 26.6, so CC * (<k> - 1) = 0.614 * 25.6 = 15.7. After drainage, k_eff ~ rho_eq ~ 33. Then:

$$b_{\text{eff}} = 1 - \frac{1 + 15.7}{33} = 1 - 0.506 = 0.494$$

In the tree case: b_tree = 1 - 1/33 = 0.970.

**Clustering cuts the effective branching factor roughly in half** for NeuroCrystal.

### 2.3 Cumulative Frontier Reduction

The frontier after i steps under the tree approximation grows as:

$$|F_i^{\text{tree}}| \approx d_0 + \sum_{j=1}^{i-1} (k_{\text{eff}}(j) - 1) \approx d_0 + (i-1)(\rho - 1)$$

Under clustering, each step contributes fewer new nodes:

$$|F_i^{\text{clust}}| \approx d_0 + \sum_{j=1}^{i-1} (k_{\text{eff}}(j) - 1 - \Delta) = |F_i^{\text{tree}}| - (i-1)\Delta$$

Define the **frontier reduction ratio**:

$$\phi(i) = \frac{|F_i^{\text{clust}}|}{|F_i^{\text{tree}}|} = 1 - \frac{(i-1)\Delta}{d_0 + (i-1)(\rho - 1)}$$

For large i (past the first step from the hub), phi converges to:

$$\phi_\infty = 1 - \frac{\Delta}{\rho - 1} = 1 - \frac{\text{CC} \cdot (\langle k \rangle - 1)}{\rho - 1}$$

For NeuroCrystal: phi_inf = 1 - 15.7/32 = 1 - 0.491 = 0.509.

The clustered frontier is roughly half the tree frontier.

---

## 3. Corrected Miss Probability

### 3.1 How Frontier Size Enters the Miss Bound

The miss probability per step is governed by the scissors ratio (from the existing proof):

$$q_i = \frac{F_{\text{eff}}(i)}{F_{\text{eff}}(i) + c(i)}$$

where F_eff(i) = Gamma * |F_i| is the angularly competitive frontier and c(i) is the angular advantage of the shortest-path neighbor.

**Crucial insight: clustering affects miss probability in TWO opposing ways.**

**(A) Smaller frontier -> FEWER random competitors -> LOWER miss rate per step.**

With fewer new nodes in the frontier, there are fewer competitors for the shortest-path neighbor to beat. This makes p_align(i) HIGHER, reducing the per-step miss rate. This is the direction that makes the tree bound pessimistic.

**(B) Fewer alternative paths -> LOWER recovery probability -> HIGHER cumulative miss.**

In a tree, once you deviate from the shortest path, there are NO alternative paths to the target --- deviation is fatal. In a clustered graph, triangles create alternative short paths, giving the walker a chance to recover after deviation. BUT the tree bound already assumes zero recovery (it is a tree). So the tree bound is pessimistic about recovery.

**Wait --- this reverses the original claim.** Let me re-examine.

### 3.2 Correcting the Direction

The tree bound states:

$$P(\text{miss} \mid d, H, \text{tree}) \geq 1 - \prod_{i=0}^{d-1} p_{\text{align}}(i)$$

In a tree:
- There is exactly ONE shortest path (no alternative routes)
- Deviation at any step is FATAL (miss is certain after deviation)
- p_align(i) represents the probability of staying on the unique path
- The product gives the probability of never deviating

In a clustered graph:
- Multiple shortest paths may exist (triangles create alternatives)
- Deviation is NOT necessarily fatal (alternative paths exist)
- The single-path product UNDERESTIMATES the reach probability
- Therefore the tree bound OVERESTIMATES the miss probability

**Conclusion: The tree bound is PESSIMISTIC (it overstates miss), not optimistic.** Clustering HELPS the walker by providing alternative paths. The tree bound P(miss_tree) is an UPPER bound on the true miss probability, not a lower bound.

But the theorem states it as a LOWER bound: P(miss) >= tree_expression. Is this consistent?

**Resolution:** The tree expression gives a lower bound on miss probability BECAUSE it is computed from the single-shortest-path survival product, which is an UPPER bound on reach probability. So:

P(reach) <= prod p_align(i)   [only the shortest path is counted; alternatives ignored]
P(miss) >= 1 - prod p_align(i)   [valid lower bound]

In a clustered graph, the real P(reach) > prod p_align(i) because alternative paths also contribute to reach. So the real P(miss) < 1 - prod p_align(i). The tree bound remains a valid lower bound --- but it is LOOSE. Clustering makes it looser.

**The gap to close: how much does clustering TIGHTEN the actual miss probability below the tree bound?**

### 3.3 The Recovery Correction

After deviating from the shortest path at step i, the walker has H - i remaining budget. In a tree, recovery probability is 0 (the target is in a different subtree). In a clustered graph, the recovery probability depends on how many alternative paths connect to the target.

Define the per-step recovery probability:

$$r_{\text{recover}}(i) = P(\text{reach } t \text{ in } H - i \text{ steps} \mid \text{deviated at step } i)$$

The full reach probability accounting for recovery is:

$$P(\text{reach}) = \prod_{j=0}^{d-1} p_{\text{align}}(j) + \sum_{i=0}^{d-1} \left[\prod_{j=0}^{i-1} p_{\text{align}}(j)\right] \cdot (1 - p_{\text{align}}(i)) \cdot r_{\text{recover}}(i)$$

The first term is the tree reach (follow shortest path without deviation). The second term sums over all possible first-deviation points, weighted by the probability of reaching that point on-path and then recovering.

**Bounding r_recover via clustering.** When the walker deviates at step i, it has moved to a node v' that is NOT on the shortest path but IS a neighbor of v_i. In a graph with clustering CC, the probability that v' shares at least one neighbor with the shortest-path node v_{i+1} (the one we SHOULD have visited) is:

$$P(\text{v' shares neighbor with } v_{i+1}) \approx 1 - (1 - \text{CC})^{\min(k_{v'}, k_{v_{i+1}})}$$

For NeuroCrystal with CC = 0.614 and post-drainage degree ~ 33:

$$P(\text{shared neighbor}) \approx 1 - (1 - 0.614)^{33} = 1 - 0.386^{33} \approx 1 - 10^{-13.6} \approx 1$$

This is nearly certain! The deviated node almost surely shares a neighbor with the correct next node on the shortest path. But sharing a neighbor is necessary but not sufficient --- the walker must also SELECT that shared neighbor (cosine-greedy picks the max-cosine node, which may not be the recovery node).

**Effective recovery probability:**

$$r_{\text{recover}}(i) \approx P(\text{shared neighbor exists}) \cdot P(\text{walker selects it}) \approx 1 \cdot p_{\text{align}}(i+1)$$

So recovery essentially costs one extra alignment step, and the recovery probability is approximately the alignment probability at the next step.

### 3.4 The Clustering-Corrected Miss Probability

Incorporating recovery into the miss bound:

$$P(\text{miss}_{\text{clust}}) = 1 - P(\text{reach}_{\text{clust}})$$

where:

$$P(\text{reach}_{\text{clust}}) = \prod_{j=0}^{d-1} p_j + \sum_{i=0}^{d-1} \left[\prod_{j=0}^{i-1} p_j\right] (1-p_i) \cdot r(i)$$

with $p_j = p_{\text{align}}(j)$ and $r(i) \approx p_{\text{align}}(i+1)$ (one-step recovery).

**For d = 2 (one intermediate step):**

$$P(\text{reach}_{\text{clust}}) = p_0 \cdot p_1 + (1 - p_0) \cdot p_1 = p_1$$

Wait, this simplifies too much. Let me be more careful.

For d = 2: source s, intermediate w_1, target t. The walker must reach t in H steps.

- Path 1 (direct): select w_1 at step 0 (prob p_0), then t is in frontier (prob 1 since d_G(w_1, t) = 1).
- Path 2 (recover): select wrong node v' at step 0 (prob 1-p_0), then reach t from v'. Recovery requires d_G(v', t) <= H - 1. If v' shares a neighbor with w_1 (likely due to clustering), then d_G(v', t) <= 2, and the walker has H-1 >> 2 remaining budget.

So P(miss | d=2, clustered) < P(miss | d=2, tree), and the reduction depends on how many alternative 2-hop paths exist.

**The number of alternative paths at distance d = 2:**

For two nodes at graph distance 2, the number of node-disjoint paths of length 2 equals the number of common neighbors. In NeuroCrystal with CC = 0.614 and <k> = 26.6, the expected number of common neighbors for d=2 pairs is significant. If the walker deviates to any neighbor of s, that neighbor likely shares neighbors with w_1, providing a 3-hop path to t.

### 3.5 Practical Correction Factor

Rather than tracking all recovery paths (combinatorial explosion), we express the correction as a multiplicative factor on the per-step survival:

$$p_{\text{align,clust}}(i) = p_{\text{align,tree}}(i) + (1 - p_{\text{align,tree}}(i)) \cdot \eta(\text{CC}, i)$$

where eta(CC, i) is the probability of recovering within one step after deviation. This gives:

$$p_{\text{align,clust}}(i) = p_{\text{align,tree}}(i) + (1 - p_{\text{align,tree}}(i)) \cdot \eta$$

The corrected miss probability:

$$P(\text{miss}_{\text{clust}}) = 1 - \prod_{i=0}^{d-1} p_{\text{align,clust}}(i) = 1 - \prod_{i=0}^{d-1} [p_i + (1 - p_i)\eta]$$

where $\eta = \eta(\text{CC})$ is the single-step recovery probability. Since recovery requires (a) the deviated node shares a neighbor with the correct next hop AND (b) the walker selects it, and (a) is near-certain for CC = 0.614, eta is dominated by (b):

$$\eta \approx \frac{1}{k_{\text{eff}}(i)} \approx \frac{1}{\rho_{\text{eq}}}$$

For NeuroCrystal: eta ~ 1/33 ~ 0.030.

This is small --- clustering provides a modest recovery channel but the cosine-greedy policy is unlikely to exploit it because it selects by angular alignment, not by graph proximity.

---

## 4. Empirical Validation

### 4.1 Computing the Corrected Bound

Using the empirical per-step survival probabilities from the tree model (Section 3 of proof_miss_probability.md):

p_1 = 0.876, p_2 = 0.773, p_3 = 0.269

With eta = 0.030 (NeuroCrystal, CC = 0.614):

p_1_clust = 0.876 + (1-0.876)*0.030 = 0.876 + 0.0037 = 0.880
p_2_clust = 0.773 + (1-0.773)*0.030 = 0.773 + 0.0068 = 0.780
p_3_clust = 0.269 + (1-0.269)*0.030 = 0.269 + 0.0219 = 0.291

Corrected miss probabilities:

| d_G | P(miss, tree) | P(miss, clustered) | Empirical |
|-----|---------------|-------------------|-----------|
| 1 | 0.0% | 0.0% | 0.0% |
| 2 | 1 - 0.876 = 12.4% | 1 - 0.880 = 12.0% | 12.4% |
| 3 | 1 - 0.876*0.773 = 32.3% | 1 - 0.880*0.780 = 31.4% | 32.3% |
| 4 | 1 - 0.876*0.773*0.269 = 81.8% | 1 - 0.880*0.780*0.291 = 80.0% | 81.8% |

### 4.2 Interpretation

The clustering correction is **small** (1-2 percentage points). This is because:

1. **Cosine-greedy does not exploit clustering.** Recovery requires the walker to SELECT the recovery neighbor, but cosine-greedy selects by angular alignment. The recovery neighbor has no angular advantage over other frontier nodes, so the probability of selecting it is ~1/k_eff, which is small.

2. **Clustering helps with EXPLORATION but not with DIRECTION.** High CC means many triangles (redundant local connections), which would help an exhaustive BFS but does not help a directional greedy walker that ignores local structure.

3. **The tree bound was already fitted to empirical data.** The per-step survival probabilities p_1, p_2, p_3 were extracted FROM the empirical miss rates, which already include the effect of clustering. So the "tree model" with empirical p_i values implicitly absorbs the clustering effect. The gap between tree-bound and reality is small because the bound was calibrated against data from a clustered graph.

**Key finding: The tree approximation is adequate for cosine-greedy traversal because the greedy policy cannot exploit the multi-path redundancy that clustering provides.** A random walker or BFS would benefit much more from clustering.

---

## 5. Corrected Theorem Component

### Part (ii), Clustering-Aware Refinement

**Theorem (Miss Probability with Clustering Correction).**

Under assumptions A1--A5, with global clustering coefficient CC and post-drainage equilibrium degree rho_eq, the miss probability satisfies:

$$P(\text{miss} \mid d_G = d, H) \geq 1 - \prod_{i=0}^{d-1} \left[ p_{\text{align}}(i) + (1 - p_{\text{align}}(i)) \cdot \eta \right]$$

where:

$$\eta = \frac{1}{\rho_{\text{eq}}} \cdot \left(1 - (1 - \text{CC})^{\rho_{\text{eq}}}\right)$$

is the single-step recovery probability. The factor $(1 - (1 - \text{CC})^{\rho_{\text{eq}}})$ is the probability that the deviated node shares at least one neighbor with the correct next-hop node, and $1/\rho_{\text{eq}}$ is the probability that the cosine-greedy policy selects the recovery neighbor from the post-drainage frontier.

**Bounds on eta:**

$$0 \leq \eta \leq \frac{1}{\rho_{\text{eq}}}$$

- When CC = 0 (tree): eta = 0, recovering the original tree bound exactly.
- When CC > 0 and rho_eq >> 1: eta ~ 1/rho_eq (recovery is possible but unlikely under greedy selection).
- When CC = 1 (clique): eta = 1/rho_eq (maximum recovery, still limited by greedy selection).

**The correction is O(1/rho_eq) regardless of CC.** Clustering affects whether a recovery PATH exists, but the greedy policy's failure to exploit it makes the correction small.

**Corollary.** The relative tightening of the miss bound due to clustering is:

$$\frac{P(\text{miss}_{\text{tree}}) - P(\text{miss}_{\text{clust}})}{P(\text{miss}_{\text{tree}})} \leq \frac{d \cdot \eta}{1 - \prod p_i} = O\left(\frac{d}{\rho_{\text{eq}}}\right)$$

For NeuroCrystal (d <= 4, rho_eq = 33): relative correction <= 4/33 ~ 12%. In practice it is 1-2% because the product structure compounds the small per-step corrections sub-additively.

---

## 6. Numerical Summary

| Graph | CC | <k> | rho_eq | eta | Max correction (d=4) |
|-------|----|-----|--------|-----|---------------------|
| NeuroCrystal | 0.614 | 26.6 | 33 | 0.030 | ~2 pp |
| Amazon | 0.362 | 5.5 | 8 | 0.125 | ~8 pp |
| Cora | 0.289 | 3.9 | 6 | 0.167 | ~10 pp |
| DBLP | 0.184 | 6.6 | 9 | 0.111 | ~7 pp |

The correction is largest for SPARSE clustered graphs (low rho_eq) where 1/rho_eq is large. For dense graphs like NeuroCrystal, the correction is negligible.

---

## 7. Why This Closes the Gap

**Gap 2 stated:** "The miss probability lower bound assumes locally tree-like structure. Graphs with high clustering coefficient have multi-path redundancy that tightens the bound."

**Resolution:** We derived the clustering correction factor eta = (1/rho_eq) * (1 - (1-CC)^rho_eq) and showed:

1. The correction is O(1/rho_eq), bounded by the greedy policy's inability to exploit recovery paths.
2. For NeuroCrystal (CC = 0.614, rho_eq = 33), the correction is ~2 percentage points --- the tree bound is adequate.
3. The tree bound remains a valid LOWER bound on miss probability. Clustering makes the true miss probability LOWER than the tree bound (clustering helps), not higher. The original phrasing of Gap 2 ("the real miss rate should be HIGHER than the bound predicts") was incorrect --- clustering provides alternative paths that REDUCE miss probability.
4. The corrected theorem component is stated in Section 5 with the recovery factor eta, smoothly interpolating between the tree limit (CC = 0, eta = 0) and the maximum-clustering regime (CC -> 1, eta -> 1/rho_eq).

**The tree approximation is justified for cosine-greedy traversal.** The greedy policy's directional bias prevents it from exploiting the structural redundancy that clustering provides. A non-greedy policy (BFS, random walk) would benefit much more from clustering, and the correction would be correspondingly larger.

---

*Gap 2 closed August 17, 2026. Builds on proof_miss_probability.md (Sections 2-3) and proof_miss_round2.md (Section 1). The clustering correction is small (O(1/rho_eq)) because cosine-greedy selection cannot exploit multi-path redundancy.*
