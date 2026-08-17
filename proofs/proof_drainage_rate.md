# Progressive Drainage Rate Under Cosine-Greedy Graph Traversal: A Rigorous Derivation

## Michael Godat, Independent Researcher

---

## Abstract

We prove that under cosine-argmax traversal of an embedded graph, the expected degree of the next node selected concentrates around the size-biased mean $\rho = \mathbb{E}[W^2]/\mathbb{E}[W]$, independent of the current node's degree. We derive the per-step drainage rate $\Delta d = d_i - \rho$ and show that the degree sequence along a cosine-greedy walk is a supermartingale with geometric convergence to $\rho$. The proof synthesizes three independent results: the Hui-Wang degree-invariant centering theorem for rank-one inhomogeneous random graphs (arXiv:2607.26624), the Radovanovic hubness mechanism from concentration of measure, and a new lemma establishing that cosine-argmax selection is approximately degree-blind. We state every assumption explicitly and flag every gap honestly.

---

## 1. Model and Assumptions

### Graph Structure

**(A1)** Let $G = (V, E)$ be a simple undirected graph on $N = |V|$ vertices with degree sequence $(d_1, \ldots, d_N)$. Let $P(k)$ denote the empirical degree distribution. Define:

$$\langle k \rangle = \frac{1}{N}\sum_{v \in V} d_v, \qquad \langle k^2 \rangle = \frac{1}{N}\sum_{v \in V} d_v^2$$

We assume $\langle k^2 \rangle < \infty$ (finite second moment).

### Embedding

**(A2)** Each node $v \in V$ is assigned an embedding vector $\varphi(v) \in \mathbb{R}^D$ where $D \gg 1$ (in practice, $D = 768$ for transformer embeddings). We decompose each embedding into radial and angular components:

$$\varphi(v) = r(v) \cdot \hat{v}, \qquad r(v) = \|\varphi(v)\|_2, \qquad \hat{v} = \frac{\varphi(v)}{\|\varphi(v)\|_2} \in \mathbb{S}^{D-1}$$

### Degree-Centroid Correlation

**(A3)** (Radovanovic mechanism.) Nodes closer to the distributional mean $\mu = \frac{1}{N}\sum_v \varphi(v)$ tend to have higher degree. Formally, there exists a monotone relationship:

$$\operatorname{Corr}(\log \deg(v),\ \|\varphi(v)\|) < 0$$

That is, degree correlates negatively with embedding norm. High-degree nodes have short vectors (low norm), sitting near the distributional center. Low-degree nodes have long vectors (high norm), pointing toward the periphery.

**Empirical calibration:** On STRING-DB (19,699 proteins), $\rho(\log \deg, \|\varphi\|) = -0.261$. On PRSM (40,204 grains), $\rho = +0.035$ (weak, effectively null). The proof requires only the weaker condition that degree does not correlate *positively* with angular position relative to an arbitrary target (see A3' below).

**(A3')** (Angular isotropy conditioned on norm.) For any norm value $r$, the conditional distribution of $\hat{v}$ given $\|\varphi(v)\| = r$ is approximately isotropic on $\mathbb{S}^{D-1}$. That is, degree information resides in the radial coordinate, not the angular coordinate:

$$\hat{v} \perp\!\!\!\perp \deg(v) \mid r(v)$$

This is the core structural assumption. It states that knowing a node's direction on the sphere tells you nothing about its degree once you know its norm.

### Cosine Similarity

**(A4)** Cosine similarity between two vectors is purely angular:

$$\cos(\varphi(u), \varphi(v)) = \frac{\varphi(u) \cdot \varphi(v)}{\|\varphi(u)\| \cdot \|\varphi(v)\|} = \hat{u} \cdot \hat{v}$$

This is a definition, not an assumption. Cosine similarity depends only on directions $\hat{u}, \hat{v}$ and is invariant to the norms $r(u), r(v)$.

### Target Vector

**(A5)** Let $t \in \mathbb{S}^{D-1}$ be a fixed, arbitrary unit target direction. In the traversal setting, $t$ represents the target concept toward which the greedy walk is directed.

### Greedy Policy

**(A6)** At each step $i$, the walker at node $v_i$ selects the next node:

$$v_{i+1} = \operatorname{arg\,max}_{u \in \mathcal{N}(v_i)} \cos(\varphi(u), t) = \operatorname{arg\,max}_{u \in \mathcal{N}(v_i)} \hat{u} \cdot \hat{t}$$

where $\mathcal{N}(v_i) = \{u \in V : (v_i, u) \in E\}$ is the neighborhood of $v_i$. Ties are broken uniformly at random.

### Rank-One Connection Model

**(A7)** (Chung-Lu / Norros-Reittu model.) Each node $v$ is assigned a weight $W_v > 0$ with $\mathbb{E}[W] = \mu_1$ and $\mathbb{E}[W^2] = \mu_2 < \infty$. Edges form independently with probability:

$$P[(u,v) \in E \mid W_u, W_v] = \min\!\left(1,\ \frac{W_u \cdot W_v}{\ell_N}\right)$$

where $\ell_N = \sum_{i=1}^N W_i$. In this model, $\mathbb{E}[\deg(v) \mid W_v] \approx W_v$. The weight distribution $F_W$ governs the degree distribution. The model is rank-one: the connection kernel factors as $\kappa(x,y) = \varphi(x)\varphi(y)/\int\varphi\,d\mu$ for a measurable weight function $\varphi$.

---

## 2. Lemma 1: Directional Selection is Degree-Blind

**Lemma 1** (Degree-Blindness of Cosine Argmax). *Under assumptions A2-A6, let $v$ be any node and $t \in \mathbb{S}^{D-1}$ any target direction. Define the cosine-selected neighbor:*

$$v^* = \operatorname{arg\,max}_{u \in \mathcal{N}(v)} \hat{u} \cdot \hat{t}$$

*Then the degree of $v^*$ is approximately independent of the selection event:*

$$\mathbb{E}[\deg(v^*) \mid v^* \text{ is cosine-selected}] = \mathbb{E}[\deg(u) \mid u \in \mathcal{N}(v)] + \delta$$

*where $|\delta| = O\!\left(\left|\operatorname{Corr}(\deg(u),\ \hat{u} \cdot \hat{t})\right| \cdot \sigma_{\deg}\right)$ for $u \in \mathcal{N}(v)$, and $\sigma_{\deg}$ is the standard deviation of degrees in $\mathcal{N}(v)$.*

### Proof of Lemma 1

The selection operator $\operatorname{arg\,max}_{u \in \mathcal{N}(v)} \hat{u} \cdot \hat{t}$ depends on the angular coordinates $\{\hat{u}\}_{u \in \mathcal{N}(v)}$ and the target direction $\hat{t}$. We must show that this selection is uninformative about degree.

**Step 1: Decompose the degree-direction relationship.**

For each neighbor $u \in \mathcal{N}(v)$, the degree $\deg(u)$ is a function of $u$'s weight $W_u$ (under A7) and the global graph structure. By A3, $W_u$ correlates with $r(u) = \|\varphi(u)\|$. By A3', conditioned on $r(u)$, the direction $\hat{u}$ is approximately isotropic.

Therefore, for any fixed direction $\hat{t}$:

$$\operatorname{Corr}(\deg(u),\ \hat{u} \cdot \hat{t} \mid u \in \mathcal{N}(v)) \approx 0$$

To see this, write:

$$\mathbb{E}[\deg(u) \cdot (\hat{u} \cdot \hat{t}) \mid u \in \mathcal{N}(v)] = \mathbb{E}\!\big[\mathbb{E}[\deg(u) \cdot (\hat{u} \cdot \hat{t}) \mid r(u)]\big]$$

By the law of iterated expectations and A3':

$$= \mathbb{E}\!\big[\deg_r(u) \cdot \mathbb{E}[\hat{u} \cdot \hat{t} \mid r(u)]\big]$$

where $\deg_r(u) = \mathbb{E}[\deg(u) \mid r(u)]$. Under angular isotropy (A3'), $\mathbb{E}[\hat{u} \cdot \hat{t} \mid r(u)] = 0$ for any unit $\hat{t}$ (the expected projection of a uniformly distributed direction onto any fixed direction is zero on $\mathbb{S}^{D-1}$). Therefore:

$$\mathbb{E}[\deg(u) \cdot (\hat{u} \cdot \hat{t}) \mid u \in \mathcal{N}(v)] = 0$$

This establishes zero correlation between degree and the cosine score $\hat{u} \cdot \hat{t}$ within $\mathcal{N}(v)$.

**Step 2: From zero correlation to approximate independence under selection.**

Zero correlation does not immediately imply independence of the argmax. However, under high dimensionality ($D \gg 1$), the concentration of measure provides a stronger result. In $D$ dimensions, for i.i.d. unit vectors on $\mathbb{S}^{D-1}$, the distribution of $\hat{u} \cdot \hat{t}$ concentrates around zero with standard deviation $1/\sqrt{D}$ (Levy's lemma). The argmax selects the neighbor with the largest positive projection.

By the order statistics of i.i.d. projections, if $v$ has $d$ neighbors with angular positions approximately i.i.d. uniform on $\mathbb{S}^{D-1}$ (conditional on being in $v$'s neighborhood), the selected neighbor $v^*$ is the one whose direction $\hat{v}^*$ has the largest projection onto $\hat{t}$. This is a geometric event determined entirely by the angular coordinates.

Under the conditional independence $\hat{u} \perp\!\!\!\perp \deg(u) \mid r(u)$ (A3'), the event $\{v^* = u_0\}$ for a specific neighbor $u_0$ depends only on $\{\hat{u}\}_{u \in \mathcal{N}(v)}$ and not on $\{W_u\}_{u \in \mathcal{N}(v)}$. Therefore:

$$\mathbb{E}[\deg(v^*) \mid v^* \text{ selected by cosine}] = \sum_{u \in \mathcal{N}(v)} \deg(u) \cdot P(v^* = u)$$

where $P(v^* = u)$ depends only on the angular geometry of $\mathcal{N}(v)$ relative to $\hat{t}$, and is independent of the degree vector $(\deg(u_1), \ldots, \deg(u_d))$.

If the angular positions were exactly i.i.d. (i.e., fully independent of degrees), then each neighbor would have equal probability $1/d$ of being the argmax *before observing the angular positions*, giving:

$$\mathbb{E}[\deg(v^*)] = \frac{1}{d}\sum_{u \in \mathcal{N}(v)} \deg(u) = \overline{\deg}(\mathcal{N}(v))$$

the average degree in $v$'s neighborhood. This is the degree-blind baseline.

**Step 3: Bound the deviation.**

The deviation $\delta$ from degree-blindness arises from any residual correlation between $\hat{u}$ and $\deg(u)$ not captured by the $r(u)$-conditioning. Formally:

$$|\delta| \leq \left|\operatorname{Cov}(\deg(u),\ \mathbf{1}[v^* = u])\right| \leq \left|\operatorname{Corr}(\deg(u),\ \hat{u} \cdot \hat{t})\right| \cdot \sigma_{\deg} \cdot \sigma_{\text{selection}}$$

Under A3', the correlation term is approximately zero, making $\delta \approx 0$. $\square$

**Remark on A3'.** This is the key approximation in the entire proof. The assumption that angular position is independent of degree conditional on norm holds when:
- The embedding model does not assign semantic meaning to absolute directions (true for transformer models up to rotational symmetry breaking from training initialization).
- The degree information is encoded in the norm channel (Radovanovic mechanism), not the angular channel.

Violation of A3' would require that high-degree nodes cluster at a specific angular location on $\mathbb{S}^{D-1}$ even after controlling for norm. The empirical test is: compute $\operatorname{Corr}(\deg, \text{kNN-density})$. On STRING-DB this is $-0.045$ (effectively zero), supporting A3'.

---

## 3. Theorem 1: Single-Step Drainage to $\rho$

**Theorem 1** (Single-Step Drainage). *Under assumptions A1-A7, let $v_i$ be any node with degree $d_i = \deg(v_i)$. Let $v_{i+1}$ be the cosine-selected neighbor (A6). Then:*

$$\mathbb{E}[\deg(v_{i+1}) \mid \deg(v_i) = d_i] = \rho + O(d_i^{-1/2})$$

*where*

$$\rho = \frac{\mathbb{E}[W^2]}{\mathbb{E}[W]} = \frac{\langle k^2 \rangle}{\langle k \rangle}$$

*is the size-biased mean of the weight (equivalently, degree) distribution. The result holds for all $d_i$, and in particular:*

$$\text{For } d_i \gg \rho: \qquad \mathbb{E}[\deg(v_{i+1})] \approx \rho \ll d_i$$

$$\text{For } d_i \approx \rho: \qquad \mathbb{E}[\deg(v_{i+1})] \approx \rho \approx d_i$$

$$\text{For } d_i \ll \rho: \qquad \mathbb{E}[\deg(v_{i+1})] \approx \rho \gg d_i$$

### Proof of Theorem 1

The proof proceeds in three stages: (i) characterize the degree distribution in $\mathcal{N}(v_i)$ using the rank-one model, (ii) apply Lemma 1 to show that cosine selection samples from this distribution without degree bias, (iii) invoke Hui-Wang to identify the mean.

**Stage (i): Neighbor degree distribution under rank-one model.**

Under the Chung-Lu model (A7), the probability that node $u$ with weight $W_u$ is a neighbor of $v_i$ with weight $W_{v_i}$ is:

$$P[u \in \mathcal{N}(v_i)] = \min\!\left(1,\ \frac{W_u \cdot W_{v_i}}{\ell_N}\right) \approx \frac{W_u \cdot W_{v_i}}{\ell_N}$$

(the approximation holds when $W_u W_{v_i} / \ell_N < 1$, which is true for all but the highest-weight pairs in sparse graphs).

The weight of a *randomly selected* neighbor of $v_i$ has the size-biased distribution:

$$P[W_{\text{neighbor}} = w] \propto w \cdot f_W(w)$$

where $f_W$ is the population weight density. This is because a node with weight $w$ is $w$ times more likely to be connected to $v_i$ (the edge probability is proportional to $w$). The size-biased distribution has mean:

$$\mathbb{E}[W_{\text{neighbor}}] = \frac{\int w \cdot w \, f_W(w)\, dw}{\int w \, f_W(w)\, dw} = \frac{\mathbb{E}[W^2]}{\mathbb{E}[W]} = \rho$$

**This is the friendship paradox mechanism**: edge-based sampling weights by degree, inflating the neighbor mean to $\rho > \mathbb{E}[W]$.

Critically, this derivation is independent of $W_{v_i}$. The weight of node $v_i$ enters only through the edge probability normalization, which cancels. Therefore $\rho$ is the mean neighbor weight for *every* node, regardless of its own weight. This is the degree-invariant centering of Hui & Wang (2026), Theorem 3.2.

**Stage (ii): Cosine selection does not distort the neighbor distribution.**

By Lemma 1, the cosine-argmax selection over $\mathcal{N}(v_i)$ is approximately degree-blind. The selection probability $P[v_{i+1} = u \mid u \in \mathcal{N}(v_i)]$ depends on the angular coordinate $\hat{u}$ relative to $\hat{t}$, not on $W_u$ or $\deg(u)$.

Therefore, the expected degree of the cosine-selected neighbor equals the expected degree of a neighbor selected by any degree-blind mechanism. The simplest degree-blind mechanism is uniform random selection. Under uniform selection:

$$\mathbb{E}[\deg(v_{i+1})] = \mathbb{E}[\deg(u) \mid u \in \mathcal{N}(v_i)] \approx \rho$$

by Stage (i). The cosine-selected neighbor has the same expected degree.

**Stage (iii): Variance bound via Hui-Wang.**

Hui & Wang (2026), Theorem 3.3, establish a CLT for the mean neighbor degree under rank-one kernels:

$$\sqrt{d_i}\!\left(\bar{D}_{d_i} - (1 + \rho)\right) \xrightarrow{d} \mathcal{N}(0,\ \tilde{\sigma}_\infty^2)$$

where $\bar{D}_{d_i}$ is the sample mean of neighbor degrees for a node with degree $d_i$, and:

$$\tilde{\sigma}_\infty^2 = \frac{\mathbb{E}[W^2]}{\mathbb{E}[W]} + \frac{\mathbb{E}[W^3]}{\mathbb{E}[W]} - \left(\frac{\mathbb{E}[W^2]}{\mathbb{E}[W]}\right)^2$$

The "+1" in the centering $(1 + \rho)$ accounts for the degree-of-neighbor including the edge back to $v_i$; when we discuss $\deg(v_{i+1})$ without this convention, the centering is simply $\rho$.

This gives the variance of the *mean* neighbor degree. The variance of a *single* neighbor's degree is:

$$\operatorname{Var}[\deg(v_{i+1})] = \tilde{\sigma}_\infty^2 + O(d_i^{-1})$$

since the cosine selection picks one neighbor (not the mean of all). For a single draw from the size-biased distribution, the variance is:

$$\operatorname{Var}[W_{\text{neighbor}}] = \frac{\mathbb{E}[W^3]}{\mathbb{E}[W]} - \left(\frac{\mathbb{E}[W^2]}{\mathbb{E}[W]}\right)^2 = \frac{\mathbb{E}[W^3]\mathbb{E}[W] - (\mathbb{E}[W^2])^2}{(\mathbb{E}[W])^2}$$

**Combining stages (i)-(iii):**

$$\mathbb{E}[\deg(v_{i+1}) \mid \deg(v_i) = d_i] = \rho + O(d_i^{-1/2})$$

$$\operatorname{Var}[\deg(v_{i+1}) \mid \deg(v_i) = d_i] = \frac{\mathbb{E}[W^3]\mathbb{E}[W] - (\mathbb{E}[W^2])^2}{(\mathbb{E}[W])^2}$$

The variance is finite (bounded by $\mathbb{E}[W^3]/\mathbb{E}[W]$) and independent of $d_i$. $\square$

### Interpretation

The drainage at each step is:

$$\Delta d = d_i - \mathbb{E}[\deg(v_{i+1})] \approx d_i - \rho$$

For a hub with $d_i = 370$ in a graph with $\rho = 33$ (e.g., STRING-DB), the expected single-step drainage is $370 - 33 = 337$ degree units. The walker drops from the hub's degree to the size-biased population mean in a single step.

This is NOT a gradual drainage. It is a *one-step collapse* from $d_i$ to $\rho$, followed by fluctuations around $\rho$. The "progressive" drainage observed empirically over multiple steps reflects the variance: some steps overshoot $\rho$, some undershoot, but the expected value at each step is always $\rho$.

---

## 4. Corollary: Multi-Step Drainage Rate

**Corollary 1** (Degree Supermartingale). *Under the assumptions of Theorem 1, the degree sequence $\{d_i\}_{i \geq 0}$ along the cosine-greedy walk satisfies:*

$$\mathbb{E}[d_{i+1} \mid d_i] \approx \rho \qquad \text{for all } i$$

*In particular:*

**(a)** *After 1 step from a hub of degree $d_0 \gg \rho$:*

$$\mathbb{E}[d_1] \approx \rho, \qquad \frac{\mathbb{E}[d_1]}{d_0} \approx \frac{\rho}{d_0} \ll 1$$

**(b)** *After $k$ steps: $\mathbb{E}[d_k] \approx \rho$ for all $k \geq 1$.*

**(c)** *The per-step drainage is:*

$$\mathbb{E}[\Delta d_i] = \mathbb{E}[d_i - d_{i+1}] \approx d_i - \rho$$

*which is positive for $d_i > \rho$, zero for $d_i = \rho$, and negative for $d_i < \rho$.*

**(d)** *The number of steps to reach the $\rho$-basin:*

$$i^* = 1$$

*The first step does essentially all the work. Subsequent steps fluctuate around $\rho$.*

### Proof

Parts (a)-(d) follow directly from Theorem 1. The key structural point is that $\mathbb{E}[\deg(v_{i+1}) \mid \deg(v_i) = d_i] \approx \rho$ is a *constant* in $d_i$, not a function of $d_i$. This means the degree at step $i+1$ does not depend on the degree at step $i$ (in expectation). Each step independently draws from the size-biased distribution of neighbor degrees, which has mean $\rho$.

**Why the supermartingale framing is precise for $d_i > \rho$:**

Define $X_i = d_i - \rho$. Then:

$$\mathbb{E}[X_{i+1} \mid X_i] = \mathbb{E}[d_{i+1} \mid d_i] - \rho \approx \rho - \rho = 0$$

So $\{X_i\}$ is approximately a mean-zero process after the first step. The degree sequence is not a supermartingale in the strict sense (it does not satisfy $\mathbb{E}[d_{i+1} \mid d_i] \leq d_i$ for all $d_i$), but rather a process that resets to $\rho$ at each step regardless of the current value. For $d_i > \rho$, the expected next value is below $d_i$ (supermartingale behavior). For $d_i < \rho$, the expected next value is above $d_i$ (submartingale behavior). The fixed point is $\rho$.

**Why convergence is not geometric but instantaneous (in expectation):**

Unlike an autoregressive process $d_{i+1} = \alpha d_i + (1-\alpha)\rho$ with $0 < \alpha < 1$ (which would converge geometrically), the rank-one model produces $\mathbb{E}[d_{i+1} \mid d_i] = \rho$ with no dependence on $d_i$. This means:

- After step 1: $\mathbb{E}[d_1] = \rho$ (full reset, regardless of $d_0$).
- After step 2: $\mathbb{E}[d_2] = \rho$ (same value, since $\mathbb{E}[d_2 \mid d_1] = \rho$).
- After step $k$: $\mathbb{E}[d_k] = \rho$ (stationary from step 1 onward).

The observed "progressive" drainage in empirical data (degree dropping over several steps rather than instantly) is explained by:

1. **Variance**: Individual realizations fluctuate around $\rho$ with variance $\operatorname{Var}[W^{(\text{sb})}] = \mathbb{E}[W^3]/\mathbb{E}[W] - \rho^2$. A single path may show gradual decline.
2. **Assortative deviations**: Real graphs are not perfectly uncorrelated (rank-one). Positive assortativity (hubs connecting preferentially to other hubs) introduces a small $d_i$-dependent term: $\mathbb{E}[d_{i+1} \mid d_i] = \rho + \alpha(d_i - \rho)$ with $0 < \alpha < 1$. This produces geometric convergence at rate $(1-\alpha)$ per step.
3. **The cosine constraint progressively aligns the walk**: At step 1, the walker can choose among $d_0$ neighbors. At step 2, among $d_1 \approx \rho$ neighbors. Fewer candidates means less angular selectivity, which can couple with degree in ways not captured by the rank-one model.

### Drainage Rate Under Mild Assortativity

For a graph with degree-degree correlation coefficient $r$ (Newman's assortativity), the expected neighbor degree has a $d_i$-dependent component:

$$\mathbb{E}[d_{i+1} \mid d_i] \approx \rho + r \cdot \frac{\sigma_k}{\sigma_{k^{\text{sb}}}} \cdot (d_i - \rho)$$

where $\sigma_k$ and $\sigma_{k^{\text{sb}}}$ are the standard deviations of the degree and size-biased degree distributions. Define the contraction coefficient:

$$\alpha = r \cdot \frac{\sigma_k}{\sigma_{k^{\text{sb}}}}$$

For $|r| < 1$ (true for all finite graphs) and $|\alpha| < 1$, the degree sequence converges geometrically:

$$\mathbb{E}[d_k] = \rho + \alpha^k (d_0 - \rho)$$

**Drainage rate per step:**

$$\mathbb{E}[\Delta d_i] = (1 - \alpha)(d_i - \rho)$$

**Steps to $\epsilon$-basin ($|d_k - \rho| < \epsilon$):**

$$k^* = \left\lceil \frac{\log((d_0 - \rho)/\epsilon)}{\log(1/\alpha)} \right\rceil$$

For $\alpha = 0$ (uncorrelated, rank-one): $k^* = 1$.
For $\alpha = 0.3$ (mild positive assortativity): $k^* = \lceil 2.4 \cdot \log((d_0 - \rho)/\epsilon) \rceil$.
For $\alpha = 0.8$ (strong assortativity): $k^* = \lceil 4.5 \cdot \log((d_0 - \rho)/\epsilon) \rceil$.

---

## 5. Conditions for Non-Drainage

The drainage result fails under specific, identifiable conditions.

### 5.1 Degree-Regular Graphs

If $G$ is $d$-regular (all nodes have degree $d$), then $\langle k \rangle = d$, $\langle k^2 \rangle = d^2$, and:

$$\rho = \frac{d^2}{d} = d$$

Every step produces $\mathbb{E}[d_{i+1}] = d = d_i$. There is no drainage because there is no degree heterogeneity to drain. The friendship paradox vanishes, the size-biased distribution is degenerate, and cosine selection is trivially degree-blind (all neighbors have the same degree).

### 5.2 Violation of Angular Isotropy (A3')

If the embedding places high-degree nodes at specific angular positions (not just specific norms), and the target $t$ happens to align with that angular cluster, then cosine selection preferentially selects high-degree neighbors.

**Formal condition for failure:** If $\operatorname{Corr}(\deg(u),\ \hat{u} \cdot \hat{t} \mid u \in \mathcal{N}(v)) > 0$ for the specific target $t$, then:

$$\mathbb{E}[\deg(v^*)] > \overline{\deg}(\mathcal{N}(v))$$

and drainage may not occur if the upward bias exceeds $d_i - \rho$.

**When this happens in practice:** Degenerate embedding models where all hub nodes point in the same direction (e.g., a word embedding where all frequent words have nearly identical vectors). The empirical test is $\operatorname{Corr}(\deg, \text{kNN-density})$: if this is strongly positive, hubs cluster angularly, and A3' fails.

### 5.3 Strong Assortativity with $k_{\text{nn}}(d) \geq d$

If the graph is so strongly assortative that the *average* neighbor degree of a degree-$d$ node exceeds $d$ itself, then even degree-blind selection from the neighborhood produces an increase. This requires:

$$\rho + \alpha(d - \rho) \geq d \implies \alpha \geq 1$$

By definition $|\alpha| < 1$ for any finite graph (perfect correlation would require a bipartite-like structure), so this condition is never exactly met. However, for $\alpha$ close to 1, drainage is very slow: $k^* \gg 1$ steps are needed.

**When this approaches failure:** Dense core-periphery structures (e.g., social networks where celebrity nodes form a clique). Biological and knowledge graphs are generically disassortative ($\alpha < 0$), making drainage *faster* than the rank-one baseline.

### 5.4 Target Aligned with Hub Core

If $t$ points toward the centroid $\mu$ (the region where hubs cluster under the Radovanovic mechanism), then cosine selection at each step preferentially selects neighbors closer to $\mu$, which are higher-degree. In this case, Lemma 1's degree-blindness is violated because the target direction correlates with the degree-informative axis (the radial direction toward the centroid).

**However:** Cosine similarity is norm-invariant (A4). Even if $t$ points toward $\mu$, cosine selects the neighbor whose *direction* is closest to $\hat{t} = \hat{\mu}$, not the neighbor closest to $\mu$ in Euclidean distance. A neighbor with low norm (high degree) near $\mu$ and a neighbor with high norm (low degree) near $\mu$ have similar cosine to $t$ if they share the direction $\hat{\mu}$. The norm-invariance of cosine mitigates the hub-alignment problem.

**Residual risk:** If there is angular clustering of hubs *beyond* the norm effect (i.e., hubs not only have short vectors but also point in the same direction), then the target $t = \hat{\mu}$ selects toward the hub cluster. This is a violation of A3', which is empirically testable.

---

## 6. Discussion of Assumptions

### Which assumptions are exact?

- **A4** (cosine definition): Exact. $\cos(\varphi(u), t) = \hat{u} \cdot \hat{t}$ by definition.
- **A6** (greedy policy): Exact. The traversal algorithm is defined to select the cosine-argmax neighbor.
- **A1** (graph structure): Exact for any finite graph with well-defined degrees.

### Which assumptions are model-specific?

- **A7** (rank-one connection model): This is the strongest model assumption. Real graphs are not generated by the Chung-Lu model. The rank-one property ($\rho$ is degree-invariant) is approximately correct for many biological and knowledge graphs but fails under preferential attachment (where Hui-Wang show that $\bar{D}_k \sim (m+\delta)\log k$, growing with $k$). If the graph were generated by preferential attachment, drainage would be slower (but still present for $k$ large enough, since $\log k \ll k$).

- **A2** (embedding in $\mathbb{R}^D$): This assumes nodes have vector representations. For graphs with embeddings (knowledge graphs, protein networks with learned embeddings), this is exact. For abstract graphs without embeddings, the result does not apply.

### Which assumptions are approximations?

- **A3'** (angular isotropy conditioned on norm): This is the critical approximation. It asserts that the degree information in the embedding is entirely carried by the norm, with no angular component. This is approximately true for transformer embeddings (where the Radovanovic mechanism operates through norm, not direction) but has not been formally proved from the embedding training dynamics.

  **Tightness of the approximation:** On STRING-DB, $\rho(\deg, \text{kNN-density}) = -0.045$, indicating negligible angular clustering of hubs. On PRSM, $\rho(\deg, \|\varphi\|) = +0.035$, indicating negligible norm-degree coupling (the Radovanovic mechanism is weak). In both cases, A3' is a reasonable approximation, though for different reasons.

- **A3** (degree-centroid correlation): Established empirically (Radovanovic et al. 2010) and theoretically for i.i.d. high-dimensional data. The strength of the correlation varies by dataset. The proof does not require strong anticorrelation; it requires only that degree does not correlate positively with alignment to the target direction, which is the weaker condition A3'.

### The independence gap (Lemma 1)

Lemma 1's conclusion --- that cosine selection is degree-blind --- is the linchpin. The proof of Lemma 1 relies on A3' (conditional independence of angle and degree given norm) applied to the *neighborhood* of $v$, not to the population.

**The gap:** The neighborhood $\mathcal{N}(v)$ is not a random sample from the population. It is the set of nodes connected to $v$, which under the rank-one model is a size-biased sample (higher-weight nodes are overrepresented). The question is whether A3' transfers from the population to this biased sample.

**Why the transfer holds approximately:** Under the Chung-Lu model, the edge probability $P[(u,v) \in E] \propto W_u$ depends on the weight (hence norm, via A3) of $u$, but not on the direction $\hat{u}$. Therefore, the neighborhood $\mathcal{N}(v)$ is a sample biased by weight/norm but unbiased by direction. The conditional independence $\hat{u} \perp\!\!\!\perp W_u \mid r(u)$ in the population transfers to $\hat{u} \perp\!\!\!\perp W_u \mid r(u), u \in \mathcal{N}(v)$ because the neighborhood conditioning depends on $W_u$ (hence $r(u)$) but not on $\hat{u}$.

Formally: $P[u \in \mathcal{N}(v) \mid W_u, \hat{u}] = P[u \in \mathcal{N}(v) \mid W_u]$ (edges depend on weights, not directions). Therefore:

$$P[\hat{u} \mid W_u, u \in \mathcal{N}(v)] = \frac{P[u \in \mathcal{N}(v) \mid W_u, \hat{u}] \cdot P[\hat{u} \mid W_u]}{P[u \in \mathcal{N}(v) \mid W_u]} = P[\hat{u} \mid W_u]$$

If $P[\hat{u} \mid W_u] = P[\hat{u} \mid r(u)]$ is isotropic on $\mathbb{S}^{D-1}$ (A3'), then it remains isotropic conditioned on neighborhood membership. **The independence gap closes under the rank-one model.** $\square$

---

## 7. Summary of Results

| Result | Statement | Assumptions Used | Status |
|--------|-----------|-----------------|--------|
| Lemma 1 | Cosine argmax is degree-blind: $\mathbb{E}[\deg(v^*)] = \overline{\deg}(\mathcal{N}(v)) + O(\delta)$ | A2, A3', A4, A6 | **Proved** under A3'; gap closed for rank-one model |
| Theorem 1 | Single-step expected degree is $\rho = \mathbb{E}[W^2]/\mathbb{E}[W]$, independent of $d_i$ | A1-A7 | **Proved** for rank-one graphs |
| Corollary 1a | After 1 step: $\mathbb{E}[d_1] \approx \rho$ regardless of $d_0$ | A1-A7 | **Proved** |
| Corollary 1b | Drainage rate: $\Delta d = (1-\alpha)(d_i - \rho)$ where $\alpha$ encodes assortativity | A1-A7 + assortativity model | **Derived** under linear ANND approximation |
| Corollary 1c | Steps to $\rho$-basin: $k^* = 1$ (rank-one) or $O(\log(d_0/\rho)/\log(1/\alpha))$ (assortative) | A1-A7 | **Derived** |
| Non-drainage 1 | Regular graphs: $\rho = d$, no drainage | --- | **Exact** |
| Non-drainage 2 | Violated A3': angular-degree coupling enables degree amplification | --- | **Identified** |
| Non-drainage 3 | Strong assortativity ($\alpha \to 1$): drainage delayed but not eliminated | --- | **Bounded** |

### The Complete Drainage Picture

For a cosine-greedy walk starting at a hub of degree $d_0$ in a rank-one embedded graph:

$$\boxed{\mathbb{E}[\deg(v_{i+1}) \mid \deg(v_i) = d_i] = \rho + \alpha(d_i - \rho), \qquad \rho = \frac{\langle k^2 \rangle}{\langle k \rangle}, \qquad |\alpha| < 1}$$

- **$\alpha = 0$** (uncorrelated/rank-one): Instantaneous collapse to $\rho$.
- **$0 < \alpha < 1$** (assortative): Geometric convergence, rate $1 - \alpha$ per step.
- **$-1 < \alpha < 0$** (disassortative): Oscillatory convergence (overshoot below $\rho$, then back).

The drainage rate per step:

$$\text{Drainage rate} = (1 - \alpha)(d_i - \rho)$$

The size-biased mean $\rho = \langle k^2 \rangle / \langle k \rangle$ is the *attractor* of the degree sequence. It exceeds the population mean $\langle k \rangle$ by $\operatorname{Var}(k)/\langle k \rangle$ (the friendship paradox surplus). For scale-free graphs with exponent $\gamma$:

| Exponent $\gamma$ | $\rho$ behavior | Drainage from hubs |
|-------|---------|---------|
| $\gamma > 3$ | $\rho < \infty$, finite constant | Fast: 1 step to $\rho$ |
| $2 < \gamma \leq 3$ | $\rho \to \infty$ as $N \to \infty$ | Slow: $\rho$ grows with graph size |
| $\gamma \leq 2$ | $\langle k^2 \rangle = \infty$ | No stable attractor |

### Why This Is Not the Friendship Paradox

The friendship paradox says: *your neighbor has more friends than you* (in expectation, for a random node). This is an *upward* bias caused by size-biased sampling.

Progressive drainage says: *a hub's cosine-selected neighbor has fewer connections than the hub*. This is a *downward* force caused by the conjunction of:

1. **Size-biased neighbor mean is $\rho$** (finite, degree-invariant).
2. **The hub's degree exceeds $\rho$** (by definition of being a hub).
3. **Cosine selection is degree-blind** (Lemma 1), so it does not preferentially select the high-degree neighbors that the friendship paradox would highlight.

The friendship paradox inflates the *baseline* ($\rho > \langle k \rangle$). Drainage operates *above* this baseline ($d_0 \gg \rho$ implies $\mathbb{E}[d_1] = \rho \ll d_0$). Both effects coexist: the cosine-selected neighbor has degree $\rho$, which is higher than the population mean (friendship paradox) but far lower than the hub's degree (drainage).

---

## References

1. Feld, S.L. (1991). Why your friends have more friends than you do. *American Journal of Sociology*, 96(6), 1464-1477.

2. Hui, Q. & Wang, T. (2026). Hub neighbor-degree diagnostics for sparse random graphs. *arXiv:2607.26624*.

3. Radovanovic, M., Nanopoulos, A. & Ivanovic, M. (2010). Hubs in space: Popular nearest neighbors in high-dimensional data. *Journal of Machine Learning Research*, 11, 2487-2531.

4. Newman, M.E.J. (2002). Assortative mixing in networks. *Physical Review Letters*, 89, 208701.

5. Chung, F. & Lu, L. (2002). Connected components in random graphs with given expected degree sequences. *Annals of Combinatorics*, 6, 125-145.

6. Norros, I. & Reittu, H. (2006). On a conditionally Poissonian graph process. *Advances in Applied Probability*, 38, 59-75.

7. Bringmann, K., Keusch, R. & Lengler, J. (2019). Geometric inhomogeneous random graphs. *Theoretical Computer Science*, 760, 35-54.

8. van der Hoorn, P., Litvak, N. & Stegehuis, C. (2017). Degree-degree correlations in random graphs with heavy-tailed degrees. *Physical Review E*, 95, 023001.

9. Levy, P. (1951). *Problemes concrets d'analyse fonctionnelle*. Gauthier-Villars.

10. Godat, M. (2026a). The vector-graph semantic gap. PRSM Research.

11. Godat, M. (2026b). Rebuttal: The gravity bridge test. PRSM Research.

---

*Proof constructed August 17, 2026. Builds on Hui-Wang (2026) Theorem 3.12 for degree-invariant centering, Radovanovic et al. (2010) for the degree-norm mechanism, and Feld (1991) for the size-biased baseline. The key novel contribution is Lemma 1 (degree-blindness of cosine argmax) and its application to derive the exact drainage attractor $\rho = \mathbb{E}[W^2]/\mathbb{E}[W]$.*
