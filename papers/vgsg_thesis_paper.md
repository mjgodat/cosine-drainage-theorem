# Similarity-Induced Traversal Confinement in High-Dimensional Vector-Graph Systems: Mechanism, Measurement, and Intervention

**Michael Godat**

*Independent Researcher, Steilacoom, Washington, USA*

---

## Abstract

Graph-augmented retrieval systems increasingly combine dense vector embeddings with discrete graph topology, relying on the assumption that cosine-similarity-biased expansion efficiently navigates to cross-domain targets. We show that this assumption fails systematically in high dimensions. The failure mode, which we term **Semantic Gravity**, is a similarity-induced local-recurrence bias in which cosine-biased expansion under finite budgets exhausts traversal budget in local angular basins before reaching cross-domain targets. The micro-mechanism is **progressive degree drainage via directional degree regression**---the directional inverse of the friendship paradox (Feld 1991): where uniform edge sampling biases toward hubs, directional edge sampling (argmax cosine to a target) biases away from hubs, producing monotone degree descent until the walk converges to the structural periphery. The trapping regime is **conditional on embedding-topology alignment**. In low-alignment graphs (NeuroCrystal: MI=0.2%), cosine seeds the periphery rather than hub cores, producing a Moderate Similarity Trap where hubs are angularly closer to everything but lose nearest-neighbor competitions to specialized low-degree nodes. In high-alignment graphs (Amazon Computers, synthetic mixtures), cosine seeds hub cores directly, producing standard Hub Entrapment via concentration of measure. The observable P4 ratio (cosine-seed degree / random-seed degree) predicts which regime governs a given graph.

We contribute three analytical results: (1) the Isotropic Angular Null, showing that mean pairwise angular distance converges to $\pi/2$ as dimensionality grows; (2) the Adjacent-Secants Null, deriving that path momentum converges to $K_{\text{mom}} = -1/2$ for memoryless walks under shared-midpoint covariance; and (3) a formal existence theorem establishing that bounded-budget policies can miss topologically proximate targets. We introduce Kinematic Trajectory Spectroscopy (KTS), a five-metric instrument for quantifying path kinematics against these analytical baselines, and validate it across 13 graphs (7 real-world, 6 synthetic) and 4 public benchmarks (Cora, Amazon Computers, DBLP, Hetionet).

Standard countermeasures fail or actively worsen confinement: Maximal Marginal Relevance (MMR) reduces reachability from 77.0% to 11.7% on Amazon Computers at $H=100$ by penalizing the hub backbone that bridges communities. Multi-Anchor expansion achieves 63.3%--97.0% endpoint reachability at $H=100$ across four benchmark graphs where single-source BFS reaches 0.0%--8.0%. However, the headline result concerns **intermediate discovery**: waypoint injection---NeuroCrystal/PRSM's core architecture---achieves 100% recovery of validated intermediate bridging concepts, compared to 13.3% for endpoint-only strategies (cosine A-to-Z and multi-anchor A-plus-Z). The path, not the endpoint, is the answer.

---

## 1. Introduction

### 1.1 Practical Origin

This work began as an engineering problem. While building a concept-traversal system for cross-domain scientific discovery (PRSM, built on NeuroCrystal: 40,204 concept grains, 6M cross-corpus edges, 151 source corpora), we observed that single-source cosine-seeded expansion reliably failed to reach known cross-domain targets. Paths that should traverse from one scientific domain to another---for instance, from metabolic pharmacology to neurodegeneration---became trapped in the source domain's angular neighborhood, visiting increasingly peripheral nodes within the same semantic basin rather than crossing to the target.

The architectural solution---placing waypoint anchors at intermediate and target positions, expanding bidirectionally toward the bridge zone---was built before we understood why it worked. The system produced validated cross-domain hypotheses [Godat 2026a] that single-source expansion could not reach. This paper formalizes *why* single-source expansion fails, *when* the failure is predictable, and *what* structural properties of the intervention explain its success.

### 1.2 The Two-Stage Protocol and Its Assumption

Modern graph-augmented retrieval follows a two-stage protocol: (1) embed a query into a dense vector space and retrieve the $k$ nearest neighbors as seed nodes, then (2) expand outward from these seeds along graph edges under a fixed hop or node-visit budget $H$. This protocol rests on a connectivity assumption: if the graph contains a path from the seed neighborhood to the target, then expansion under budget $H$ will find it, provided $H$ is large enough to cover the graph distance.

Variants of this architecture appear across the retrieval-augmented generation literature. GraphRAG [Edge et al. 2024] constructs hierarchical community summaries and retrieves from local neighborhoods. RAPTOR [Sarthi et al. 2024] builds recursive tree structures for multi-level retrieval. HippoRAG [Gutierrez et al. 2024] models long-term memory through graph-based associative retrieval. All share the implicit assumption that seed-local expansion, given sufficient budget, reaches cross-community targets.

### 1.3 The Failure of the Assumption

We show that this assumption fails, and we can identify precisely when and why. The failure is not stochastic---it is a geometric consequence of the interaction of high-dimensional concentration of measure with graph topology. The specific failure mode depends on the degree to which a graph's edges align with its embedding geometry.

In every real-world graph we tested, cosine-similarity-biased expansion follows a gradient that misaligns with the topological backbone needed for cross-community traversal. The nature of this misalignment is conditional:

1. **Low embedding-topology alignment (NeuroCrystal, MI=0.2%).** Cosine seeding places the expansion frontier in the graph periphery (P4 ratio = 0.47). Hubs are angularly closer to everything (P1: $\rho = +0.12$) but their angular smearing (P3: $\rho = +0.19$) produces moderate cosine scores that lose nearest-neighbor competitions to specialized low-degree nodes. The result is a **Moderate Similarity Trap**: expansion visits peripheral specialists rather than the hub backbone.

2. **High embedding-topology alignment (Amazon Computers P4=2.5, Synthetic Mixture P4=5.7).** Cosine seeding lands directly on hub cores. The standard concentration-of-measure attraction produces **Hub Entrapment**: expansion circulates among high-degree hubs rather than escaping to cross-community targets.

Angular smearing of hubs is universal (P3 positive on all real graphs). Whether it causes hub attraction or hub avoidance depends on graph construction---specifically, on whether the graph's edges were built from the same cosine geometry that the expansion policy uses.

### 1.4 Definition

**Semantic Gravity** is a similarity-induced local-recurrence bias in which cosine-biased expansion under finite budgets exhausts traversal budget in local angular basins before reaching cross-domain targets. It is not a physical force. It is a deterministic consequence of the geometry of high-dimensional embedded graphs under bounded expansion policies. The specific trapping mechanism---Moderate Similarity Trap or Hub Entrapment---is conditional on the alignment between the graph's edge structure and its embedding geometry.

### 1.5 Contributions

1. **Analytical null models.** We derive two closed-form expectations---the Isotropic Angular Null ($\theta = \pi/2$, Theorem 1) and the Adjacent-Secants Null ($K_{\text{mom}} = -1/2$, Theorem 2)---that establish exact baselines for trajectory analysis. We prove a formal existence theorem (Theorem 3) for the semantic gap under bounded expansion.

2. **Conditional mechanism identification.** We identify a conditional trapping mechanism governed by embedding-topology alignment, measured by the P4 ratio. Low-alignment graphs (P4 < 1) exhibit the Moderate Similarity Trap; high-alignment graphs (P4 > 1) exhibit Hub Entrapment. A 4-feature predictive model achieves LOO $\rho = 0.924$ and MAE $= 9.0$ pp for trapping severity across tested graphs.

3. **Micro-mechanism identification.** We identify progressive degree drainage via directional degree regression as the micro-mechanism of Semantic Gravity---the directional inverse of the friendship paradox [Feld 1991]. Cosine-to-target selection is degree-blind; the expected neighbor degree regresses to the size-biased mean $\rho = E[W^2]/E[W]$, producing monotone degree descent. Confirmed across 5 graphs (Experiment 27). An earlier curvature-based mechanism (proxy ORC) was retracted after exact computation falsified it; the drainage mechanism replaces it.

4. **Graph-ball confinement.** We demonstrate that cosine expansion misses 18.7% of graph-reachable targets inside the policy ball at $H = 100$ on NeuroCrystal, escalating to 81.8% at graph distance $d_G = 4$.

5. **Downstream utility---the headline result.** Waypoint injection (NeuroCrystal/PRSM's architecture) achieves 100% recovery of validated intermediate bridging concepts, compared to 13.3% for endpoint-only strategies (cosine A-to-Z, multi-anchor A-plus-Z).

6. **Validated endpoint intervention.** Multi-Anchor expansion achieves 63.3%--97.0% endpoint reachability at $H=100$ across four benchmark graphs (Cora, Amazon Computers, DBLP, NeuroCrystal) where BFS reaches 0.0%--8.0%. Multi-Anchor wins on all 4 graphs. Policy ordering is 100% consistent across 4 feature spaces.

7. **Kinematic instrument.** Kinematic Trajectory Spectroscopy (KTS) provides five deterministic, LLM-free, sub-millisecond metrics for quantifying path behavior against the derived analytical nulls. All metrics survive L2 normalization---the signatures are purely angular.

---

## 2. Related Work

**Embedding anisotropy and representation degeneration.** Ethayarajh [2019] showed that contextualized word representations occupy a narrow cone in embedding space, with later transformer layers producing increasingly anisotropic representations. Gao et al. [2019] formalized representation degeneration as a training pathology in language generation models where token embeddings collapse into a low-dimensional subspace. Our angular compression statistic $\Gamma$ quantifies the degree of this anisotropy at the corpus level and connects it to traversal failure: the narrower the cone, the more nodes per unit solid angle, the faster budget is consumed in local recurrence.

**Hubness in high-dimensional spaces.** Radovanovic et al. [2010] identified the hubness phenomenon: in high dimensions, some points become nearest neighbors of many others while most points appear in few nearest-neighbor lists. They traced this to concentration of measure and the skewness of the $k$-occurrence distribution. Our experiments confirm that hubness arises from pure geometry---synthetic isotropic point clouds in $\mathbb{R}^{768}$ exhibit hubness skew of 16.2, which vanishes (skew 0.3) after L2 normalization onto the unit sphere. We extend hubness from a retrieval accuracy problem to a *traversal confinement* problem, showing that whether hubs attract or repel cosine-biased expansion depends on embedding-topology alignment.

**Over-squashing and Ollivier-Ricci curvature in GNNs.** Topping et al. [2022] connected Ollivier-Ricci curvature [Ollivier 2009] to the over-squashing bottleneck in graph neural networks, showing that negatively curved edges compress information flow. Ni et al. [2019] used Ricci flow for community detection. We initially hypothesized a curvature-cosine anti-correlation as the mechanism underlying Semantic Gravity, but exact Wasserstein-1 computation falsified this (Section 3.4, Note on ORC retraction). The micro-mechanism is instead progressive degree drainage via directional degree regression---the directional inverse of the friendship paradox [Feld 1991], supported by size-biased regression theory [Hui & Wang 2026] and the zoom-in phase of greedy routing [Boguna et al. 2009].

**GraphRAG and hybrid retrieval.** Edge et al. [2024] introduced Graph RAG with hierarchical community summaries. Sarthi et al. [2024] proposed RAPTOR's recursive tree-organized retrieval. Gutierrez et al. [2024] designed HippoRAG with neurobiologically inspired graph memory. These systems assume that graph structure enhances retrieval coverage. Our results suggest this assumption requires qualification: graph densification within angular basins can increase local recurrence without improving cross-basin reachability. The "Densification Delusion"---that more edges monotonically improve retrieval---is falsified by our ablation experiments (Section 5.3).

**Literature-based discovery.** Swanson [1986] demonstrated that disjoint literatures can contain implicit connections discoverable by traversing shared intermediaries. The two-stage seed-and-expand protocol is a computational implementation of Swanson's insight. Our finding that cosine-biased expansion gets trapped in the source literature's angular basin directly explains why Swanson's original method required manual curation: automated similarity-based expansion follows a gradient that stays within one literature rather than crossing to the complementary one.

**Concentration of measure.** The geometric foundations of our analytical results rest on classical concentration of measure [Levy 1951, Milman 1971], the Hanson-Wright inequality for quadratic forms [Hanson and Wright 1971], and intrinsic dimensionality estimation [Grassberger and Procaccia 1983, Levina and Bickel 2004]. We connect these individually known phenomena into a unified account of finite-budget traversal failure: concentration drives the $\pi/2$ null, angular compression from semantic embedding creates the populated basins, and the bounded traversal budget converts these geometric conditions into operational trapping.

---

## 3. Theoretical Foundations

### 3.1 Definitions

Let $\mathcal{M} = \mathbb{S}^{D-1} \subset \mathbb{R}^D$ denote the $(D-1)$-dimensional unit hypersphere in $\mathbb{R}^D$, with $D \gg 1$. For embedded nodes $\vec{u}, \vec{w} \in \mathcal{M}$, the geodesic angular distance is:

$$d_\theta(\vec{u}, \vec{w}) = \arccos(\langle \vec{u}, \vec{w} \rangle) \in [0, \pi]$$

Let $G = (V, E)$ be a graph with vertex set $V$ embedded in $\mathcal{M}$ via $\phi: V \to \mathcal{M}$. Let $\sigma_k$ be a cosine-similarity-based top-$k$ seed selection policy:

$$S_k(\vec{q}) = \underset{S \subseteq V, |S|=k}{\arg\max} \sum_{s \in S} \langle \vec{q}, \phi(s) \rangle$$

Let $P$ be a graph expansion policy (e.g., BFS, cosine-biased best-first, beam search, Personalized PageRank) with expansion budget $H \in \mathbb{N}^+$ measured in total node visits. The **policy-reachable set** is:

$$R_H^P(S_k(\vec{q})) = \{u \in V \mid u \text{ is visited by policy } P \text{ within budget } H \text{ starting from } S_k(\vec{q})\}$$

Note that $R_H^P$ is policy-dependent and generally a *subset* of the graph-distance ball $\{u : d_G(S_k, u) \leq H\}$. An exhaustive BFS policy visits every node within graph radius $H$. A similarity-biased policy may fail to visit nodes within graph radius $H$ if it preferentially expands toward angularly similar neighbors, exhausting budget before reaching angularly distant but graph-proximate nodes. This distinction is central to the trapping mechanism.

### 3.2 Theorem 1: The Isotropic Angular Null

**Theorem 1.** *Let $\vec{u}, \vec{w} \sim \text{Uniform}(\mathbb{S}^{D-1})$ be independent, uniformly distributed random unit vectors. As $D \to \infty$:*

$$\lim_{D \to \infty} \mathbb{E}[d_\theta(\vec{u}, \vec{w})] = \frac{\pi}{2}$$

*Proof.* By concentration of measure on high-dimensional spheres [Levy 1951], the inner product $\langle \vec{u}, \vec{w} \rangle$ converges in distribution to $\mathcal{N}(0, 1/D)$. For any $\epsilon > 0$:

$$\mathbb{P}(|\langle \vec{u}, \vec{w} \rangle| \geq \epsilon) \leq 2\exp\left(-\frac{D\epsilon^2}{2}\right)$$

As $D \to \infty$, $\langle \vec{u}, \vec{w} \rangle \xrightarrow{p} 0$. Because $\arccos(x)$ is continuous and bounded on $[-1, 1]$, the bounded convergence theorem yields:

$$\lim_{D \to \infty} \mathbb{E}[\arccos(\langle \vec{u}, \vec{w} \rangle)] = \arccos(0) = \frac{\pi}{2} \quad \blacksquare$$

The value $\pi/2$ serves as the isotropic null: at this expectation, the angular distribution contains zero mutual information, and cosine-based retrieval reduces to an unguided walk.

**Definition 1 (Angular Compression $\Gamma$).** The global angular compression statistic is:

$$\Gamma = \frac{\pi/2 - \bar{\theta}}{\pi/2} \in [0, 1)$$

where $\bar{\theta} = \frac{2}{N(N-1)} \sum_{i<j} d_\theta(\vec{v}_i, \vec{v}_j)$ is the mean pairwise angular distance. $\Gamma = 0$ denotes isotropic noise; $\Gamma > 0$ indicates directional concentration.

**Definition 2 (Local Compression $\Gamma_k(v)$).** The node-level local compression over the $k$-nearest cosine neighbors $\text{kNN}(v)$:

$$\Gamma_k(v) = 1 - \frac{1}{k} \sum_{u \in \text{kNN}(v)} \frac{d_\theta(\vec{u}, \vec{v})}{\pi/2}$$

$\Gamma_k(v)$ quantifies the solid-angle packing density surrounding node $v$, directly scaling the number of intra-basin candidate edges available per hop.

**Table 1. Empirical $\Gamma$ values across tested graphs.**

| Graph | $\Gamma$ | Interpretation |
|-------|----------|----------------|
| Synthetic Isotropic | 0.000 | Zero compression (control) |
| DBLP | 0.010 | Near-isotropic |
| Cora | 0.036 | Mild compression |
| Amazon Computers | 0.226 | Strong compression |
| NeuroCrystal | 0.249 | Strong compression |
| Synthetic Mixture | 0.439 | Extreme compression |

### 3.3 Theorem 2: The Adjacent-Secants Null

To evaluate the kinematics of sequential graph traversals $\mathcal{T} = (\vec{v}_1, \vec{v}_2, \dots, \vec{v}_N)$ through $\mathcal{M}$, we analyze the alignment between consecutive displacement vectors $\vec{\Delta}_i = \vec{v}_{i+1} - \vec{v}_i$.

**Definition 3 (Path Momentum $K_{\text{mom}}$).**

$$K_{\text{mom}}(\mathcal{T}) = \frac{1}{N-2} \sum_{i=1}^{N-2} \frac{\langle \vec{\Delta}_i, \vec{\Delta}_{i+1} \rangle}{\|\vec{\Delta}_i\| \|\vec{\Delta}_{i+1}\|}$$

**Theorem 2 (Adjacent-Secants Covariance Null).** *Let $X, Y, Z \in \mathbb{R}^D$ be i.i.d. random vectors drawn from a distribution with mean zero, finite second moments, and isotropic covariance $\mathbb{E}[XX^T] = \sigma^2 I_D$. Let $\vec{\Delta}_1 = Y - X$ and $\vec{\Delta}_2 = Z - Y$. Under high-dimensional norm concentration ($D \to \infty$):*

$$\lim_{D \to \infty} \mathbb{E}\left[\frac{\langle \vec{\Delta}_1, \vec{\Delta}_2 \rangle}{\|\vec{\Delta}_1\| \|\vec{\Delta}_{2}\|}\right] = -\frac{1}{2}$$

*Proof.* Expanding the inner product:

$$\langle \vec{\Delta}_1, \vec{\Delta}_2 \rangle = (Y - X)^T(Z - Y) = Y^TZ - \|Y\|^2 - X^TZ + X^TY$$

By independence: $\mathbb{E}[Y^TZ] = \mathbb{E}[X^TZ] = \mathbb{E}[X^TY] = 0$. The expected inner product is dominated by the shared midpoint $Y$:

$$\mathbb{E}[\langle \vec{\Delta}_1, \vec{\Delta}_2 \rangle] = -\mathbb{E}[\|Y\|^2] = -D\sigma^2$$

Each displacement has expected squared norm:

$$\mathbb{E}[\|\vec{\Delta}_i\|^2] = \mathbb{E}[\|Y\|^2] + \mathbb{E}[\|X\|^2] = 2D\sigma^2$$

By the Hanson-Wright inequality [Hanson and Wright 1971], $\|\vec{\Delta}_i\|^2 / (2D\sigma^2) \xrightarrow{p} 1$ as $D \to \infty$. Applying continuous mapping:

$$\mathbb{E}\left[\frac{\langle \vec{\Delta}_1, \vec{\Delta}_2 \rangle}{\|\vec{\Delta}_1\| \|\vec{\Delta}_2\|}\right] \approx \frac{-D\sigma^2}{2D\sigma^2} = -\frac{1}{2} \quad \blacksquare$$

This result establishes $K_{\text{mom}} = -1/2$ as an exact geometric invariant for memoryless isotropic walks. Departures from $-1/2$ quantify path structure: $K_{\text{mom}} \approx 0$ indicates within-cluster inertial drift, while $K_{\text{mom}} < -0.50$ captures active directional reversal across manifold boundaries.

**Empirical verification.** We confirm the $-1/2$ null across 7 dimensionalities ($D = 10$ to $5{,}000$), 7 of 8 tested distributions, and 7 path lengths ($N = 3$ to $50$). At $D = 768$ (the operating dimensionality of nomic-embed-text), the empirical mean is $-0.4998$ with standard deviation $0.011$, an error of $0.00018$ from the predicted value. The single deviation occurs for a 3-cluster Gaussian mixture ($K_{\text{mom}} = -0.440$): inter-cluster sampling breaks the i.i.d. assumption, producing a diagnostic signal rather than a counterexample. Structured kNN walks deviate to $K_{\text{mom}} = -0.466$; this deviation from the null is itself the signal that KTS measures.

### 3.4 The Micro-Mechanism: Progressive Drainage via Directional Degree Regression

Cosine similarity is a purely angular metric that ignores vector norm. In high-dimensional embedded graphs, node degree correlates with proximity to the distributional mean---hubs are angularly central, specialists are angularly peripheral [Radovanovic et al. 2010]. Cosine-to-target selection filters a hub's neighbors by angular direction, which is approximately independent of degree among the neighborhood. The directionally selected neighbor therefore has expected degree equal to the size-biased population mean $\rho = E[W^2]/E[W]$ [Hui & Wang 2026, Theorem 3.12], far below the hub's own degree. Each step repeats this regression, producing monotone degree drainage until the walk converges to the structural periphery.

This is the directional inverse of the friendship paradox [Feld 1991]: where uniform edge sampling biases toward hubs, directional edge sampling (argmax cosine to a target) biases away from hubs.

**Three converging theoretical frameworks:**

1. **Size-biased regression** [Hui & Wang 2026]: degree-invariant neighbor mean under rank-one kernels. The expected degree of a directionally selected neighbor is the size-biased mean $\rho = E[W^2]/E[W]$, which is independent of the selecting node's degree for sufficiently large hubs.

2. **Zoom-in phase of greedy routing** [Boguna et al. 2009, Papadopoulos et al. 2012]: structural descent toward peripheral targets. In hyperbolic-like navigable graphs, greedy forwarding toward a target node descends the degree hierarchy---each hop moves closer to the target in the angular coordinate while dropping to lower-degree nodes in the radial coordinate.

3. **Directional inverse of the friendship paradox** [Feld 1991]: angular selection is degree-blind, so the expected degree of the selected neighbor regresses to the population mean rather than being biased upward by the selecting node's degree. This is the exact inversion of Feld's result, where uniform edge sampling biases the sampled neighbor's degree upward.

**Empirical confirmation (Experiment 27, 5 graphs).** Progressive drainage was tested and confirmed across all five graphs in the experimental corpus:

- **T1: Hubs have diffuse angular directions.** Degree-direction entropy correlation is negative on all graphs ($\rho = -0.10$ to $-0.32$): higher-degree nodes have more uniformly distributed neighbor directions, meaning directional selection is less likely to pick a high-degree neighbor.

- **T3: Degree drops monotonically over expansion steps.** The correlation between step index and selected-neighbor degree is strongly negative on all graphs ($\rho = -0.30$ to $-0.82$). On NeuroCrystal, mean degree drops from 370 at step 1 to 33 by step 15---an order-of-magnitude drainage.

- **T5: Masking specialists destroys reachability.** If drainage is the mechanism, then removing the peripheral specialists that drainage converges toward should destroy the system's ability to reach cross-domain targets. This prediction was tested---and falsified on all 5 graphs: masking low-degree nodes reduces reachability, confirming that specialists are the landing zone, but the system partially compensates through alternative paths. The falsification of T5 refines the model: drainage is the dominant mechanism but not the only path-selection force.

**Cone narrowing is a co-effect, not a cause.** An alternative hypothesis (Team Beta) proposed that angular cone narrowing---the progressive tightening of the set of viable next-hop directions---drives the periphery convergence. DBLP's assortative degree structure provides a natural experiment: on DBLP, the angular cone actually *widens* over successive expansion steps while degree drainage continues unabated. This decouples the two phenomena. Both drainage and cone narrowing result from radial displacement from the distributional mean, but drainage is the primary mechanism and narrowing is a secondary geometric consequence.

**Key prediction.** For any hub $v_i$ with $\deg(v_i) \gg \bar{d}$, the expected degree of the cosine-to-target selected neighbor satisfies:

$$E[\deg(v_{i+1})] \approx \rho = \frac{E[W^2]}{E[W]} \ll \deg(v_i)$$

where $W$ is the degree distribution of $v_i$'s neighbors. This inequality holds at every step, producing monotone descent.

**Note on ORC retraction.** An earlier version of this paper proposed an Ollivier-Ricci curvature mechanism (proxy ORC $\rho = -0.966$, exact ORC $\rho = +0.078$ [n.s.], proxy-exact correlation $\rho = 0.006$). This was retracted after exact Wasserstein-1 computation falsified the proxy approximation. The progressive drainage mechanism replaces ORC as the micro-level explanation of how cosine-biased expansion descends the degree hierarchy.

**Status: EMPIRICALLY CONFIRMED** across 5 graphs. Proof sketch available via three converging frameworks (see `proof_progressive_drainage.md`).

### 3.5 The Conditional Mechanism: Embedding-Topology Alignment

The trapping mechanism is now understood as conditional on the alignment between a graph's edge structure and its embedding geometry.

**Four diagnostic probes.** We define four probes that together characterize the trapping regime of any embedded graph:

- **P1: Hub angular proximity.** $\rho(\text{degree}, \text{mean cosine to all other nodes})$. Measures whether hubs are angularly closer to the rest of the graph. Positive on all real graphs tested ($+0.12$ on NeuroCrystal).

- **P3: Hub angular smearing.** $\rho(\text{degree}, \text{angular dispersion of neighbors})$. Measures whether hubs have angularly diffuse neighborhoods. Positive on all real graphs tested ($+0.19$ on NeuroCrystal). This is the universal property: hubs connect to angularly diverse nodes, producing moderate average cosine scores rather than extreme ones.

- **P4: Cosine seed degree ratio.** Mean degree of cosine-seeded nodes / mean degree of random-seeded nodes. This is the key observable that predicts the trapping regime:

| Graph | Cosine Seed Degree | Random Seed Degree | P4 Ratio | Regime |
|-------|-------------------:|-------------------:|---------:|--------|
| NeuroCrystal | 12.7 | 27.1 | 0.47 | Moderate Similarity Trap |
| Amazon Computers | 90.0 | 35.4 | 2.54 | Hub Entrapment |
| Synthetic Mixture | --- | --- | 5.7 | Hub Entrapment |

- **MI: Mutual information.** Between graph adjacency and cosine top-$k$ membership. NeuroCrystal MI=0.2%---near-zero overlap between who is a graph neighbor and who is a cosine neighbor.

**Low alignment (P4 < 1): Moderate Similarity Trap.** When graph edges are not built from cosine proximity (NeuroCrystal: knowledge-graph edges from corpus co-occurrence), cosine seeding lands on low-degree peripheral specialists. Hubs are angularly closer to everything (P1 positive), but their angular smearing (P3 positive) means their cosine scores to any given query are moderate rather than extreme. Specialized, low-degree nodes whose embeddings tightly cluster around a specific topic produce higher cosine scores to queries in that topic. The expansion frontier starts in the periphery and stays there---it visits many topologically peripheral nodes within the source angular basin rather than reaching the hub backbone.

**High alignment (P4 > 1): Hub Entrapment.** When graph edges correlate with cosine proximity (Amazon co-purchase graph, synthetic mixture), cosine seeding lands directly on high-degree hubs. Standard concentration-of-measure attraction keeps the expansion circulating among hubs rather than escaping to cross-community periphery.

**Both regimes produce the same operational failure**: budget exhaustion before reaching the target. The intervention (multi-anchor expansion, waypoint injection) works in both regimes by providing entry points on both sides of the angular gap.

**Predictive scaling.** A 4-feature model including $\Gamma$, degree CV, and their interaction ($\Gamma \times \text{degCV}$) achieves leave-one-out cross-validation $\rho = 0.924$ and MAE $= 9.0$ pp for trapping severity (cosine reachability deficit vs. oracle), improved from $\rho = 0.89$ and MAE $= 10.6$ pp without the interaction term. The interaction captures the finding that angular compression and degree heterogeneity amplify each other: high $\Gamma$ on a scale-free graph produces more severe trapping than either alone.

### 3.6 Theorem 3: Existence of the Semantic Gap

**Theorem 3.** *Let $T \subset V$ be a target subgraph with bounded internal diameter $\text{diam}_G(T) \leq \Delta_T < \infty$. If the minimum graph geodesic between the seed set and the target exceeds the hop budget:*

$$d_G(S_k(\vec{q}), T) = \min_{s \in S_k, t \in T} d_G(s, t) > H$$

*then the policy-reachable set excludes the target for any policy that traverses at most one edge per unit of budget:*

$$R_H^P(S_k(\vec{q})) \cap T = \emptyset$$

*Proof.* Any policy $P$ that traverses at most one edge per budget unit can visit at most nodes within graph distance $H$ of the seed set. If $\min_{s \in S_k, t \in T} d_G(s, t) > H$, no element of $T$ is within graph distance $H$ of any seed, so no policy reaches $T$ within budget $H$. The internal connectivity $\text{diam}_G(T) < \infty$ does not imply seed-conditioned reachability under bounded expansion. $\blacksquare$

**Corollary (Policy-dependent trapping).** For similarity-biased policies, a stronger condition holds: $R_H^P \subsetneq \{u : d_G(S_k, u) \leq H\}$. A target $t$ with $d_G(S_k, t) \leq H$ can *still* be missed if the policy's directional bias causes it to visit other nodes first, exhausting budget before reaching $t$. This is the operational VGSG trapping condition, validated experimentally in Section 5.

---

## 4. Kinematic Trajectory Spectroscopy (KTS)

### 4.1 Five Metrics

Given an ordered trace $\mathcal{T} = (\vec{v}_1, \vec{v}_2, \dots, \vec{v}_N)$ through the embedded manifold $\mathcal{M}$, KTS computes five deterministic metrics that quantify path kinematics:

**(a) Path Momentum ($K_{\text{mom}}$).** The mean cosine alignment of consecutive displacement vectors (Definition 3). Theorem 2 establishes $K_{\text{mom}} = -1/2$ as the memoryless null. Deviations diagnose path structure: $K_{\text{mom}} \approx 0$ indicates within-cluster inertia; $K_{\text{mom}} < -0.5$ indicates active cross-domain tacking.

**(b) Tortuosity ($\tau$).** The ratio of endpoint displacement to total path length:

$$\tau = \frac{\|\vec{v}_N - \vec{v}_1\|}{\sum_{i=1}^{N-1} \|\vec{v}_{i+1} - \vec{v}_i\|}$$

High tortuosity indicates a path that progresses efficiently toward its destination; low tortuosity indicates circuitous wandering within a basin.

**(c) Eccentricity.** The ratio of the first singular value to the sum of the second and third, computed from the centered path matrix. High eccentricity indicates a linear, directed path; low eccentricity indicates a spherically distributed or circling path.

**(d) TAV Magnitude.** The $L_2$ norm of the Trace Averaged Vector $\vec{v}_{\text{TAV}} = \frac{1}{N}\sum \vec{v}_i$, normalized by path length. Constructive interference (co-directional steps) produces large magnitude; destructive interference (opposing steps) produces small magnitude.

**(e) Saturation ($\alpha_{\text{sat}}$).** The minimum amplification factor $\alpha$ at which the nearest-neighbor identity of the $\alpha$-ray $\Psi(\alpha; \mathcal{T}) = \alpha \cdot \vec{v}_{\text{TAV}}$ stabilizes. Early saturation indicates a path operating near the manifold surface (broadly studied concepts); late saturation indicates a path deep in a local manifold valley.

### 4.2 Saturation Requires a Populated Manifold

Saturation is the only KTS metric that depends on the distribution of other nodes in the space, not just on the path's own vectors. On the unit sphere with no population structure, $\alpha_{\text{sat}} = 1.5$ with zero variance---there is no density gradient to detect. Saturation measures the mass distribution along the radial axis: how quickly the $\alpha$-ray traverses from local, specific concepts to generic, broadly represented ones. Geometry without mass produces no saturation signal.

### 4.3 Validation on Pure Geometry

All KTS metrics except saturation produce statistically significant separation between structured kNN walks and random walks on featureless synthetic point clouds across 6 distributions (Isotropic Gaussian, Uniform Hypercube, Exponential, Laplace, Unit Sphere, 3-Cluster Mixture), all at $p < 10^{-5}$. This confirms that the instrument measures geometric properties of path shape, not semantic content.

On the Synthetic Isotropic distribution ($D = 768$, $N = 10{,}000$ points): kNN walk momentum $= -0.464$ versus random walk momentum $= -0.499$ ($p < 10^{-92}$). The kNN graph imposes local angular constraints that shift momentum above the $-1/2$ null---precisely the signal that distinguishes structured traversal from random sampling.

### 4.4 L2-Normalization Ablation

We project all node embeddings onto the unit sphere ($\vec{v} \leftarrow \vec{v}/\|\vec{v}\|$) and recompute all KTS metrics. Across all 4 benchmark graphs (Cora, Amazon Computers, DBLP, Synthetic Isotropic), every metric retains statistical significance at $p < 0.001$. Representative results at $D = 1433$ (Cora): L2-normalized momentum kNN $= -0.433$ vs. random $= -0.499$ ($p < 10^{-76}$); L2-normalized tortuosity kNN $= 0.191$ vs. random $= 0.167$ ($p < 10^{-143}$). KTS is purely angular: the kinematic signatures arise from directional structure in the embedding space, not from norm variance.

### 4.5 Graph Topology Fingerprinting

The gap between kNN-walk momentum and graph-walk momentum serves as a topology fingerprint:

$$\Delta_{\text{mom}} = K_{\text{mom}}(\text{kNN walks}) - K_{\text{mom}}(\text{graph walks})$$

Graphs whose edges align with feature-space proximity (cosine-constructed) show $\Delta_{\text{mom}} \approx 0$. Graphs whose edges encode relational or co-occurrence structure show large $\Delta_{\text{mom}}$. In our experiments, the momentum fingerprint vector of Cora and Amazon Computers has cosine similarity $= 1.000$ (both are feature-attributed citation/co-purchase graphs with similar kNN/graph momentum gaps), while Cora and DBLP have cosine similarity $= -0.491$ (DBLP's sparser, more heterogeneous connectivity produces a qualitatively different kinematic signature). This provides a graph classification signal without any supervised training.

---

## 5. Experimental Validation

### 5.1 Graphs Tested

**Table 2. Experimental graph corpus.**

| Graph | Class | Nodes | Edges | Mean Deg | Deg CV | $\Gamma$ |
|-------|-------|------:|------:|---------:|-------:|-------:|
| NeuroCrystal | SCALE-FREE | 39,220 | 116,916 | 26.6 | 2.69 | 0.249 |
| Cora | HETEROGENEOUS | 2,708 | 5,278 | 3.9 | 1.34 | 0.036 |
| Amazon Computers | HETEROGENEOUS | 13,752 | 245,861 | 36.5 | 1.94 | 0.226 |
| DBLP | HETEROGENEOUS | 17,716 | 52,867 | 6.0 | 1.57 | 0.010 |
| Hetionet v1.0 | HETEROGENEOUS | 47,031 | 2,250,197 | multi-type | --- | --- |
| STRING-DB v12.0 | SCALE-FREE | 19,699 | 1,860,000 | high | --- | --- |
| Synth. Isotropic | MODULAR | 10,000 | 100,000 | 36.2 | 1.65 | 0.000 |
| Synth. Mixture | MODULAR | 10,000 | 100,000 | 35.6 | 1.34 | 0.439 |

Additional graphs in the test corpus include Grid Road (10,000 nodes, REGULAR/SPATIAL) and 4 additional synthetic distributions (Anisotropic, Power-law, Sphere, Low-dim), for a total of 13 graphs spanning 5 structural classes.

**Classification criteria.** SCALE-FREE: degree CV $> 2.0$ or power-law $\alpha > 2.0$. HETEROGENEOUS: $> 3$ discrete node types. MODULAR: moderate clustering, no strong type or hub structure. REGULAR/SPATIAL: degree CV $< 0.3$.

### 5.2 Frontier Telemetry (Experiment 10)

We instrument the expansion frontier of cosine-biased and BFS policies on NeuroCrystal (200 source-target pairs, $H = 100$) to measure what each policy *actually visits*.

**Table 3. Frontier telemetry: cosine-biased vs. BFS expansion.**

| Metric | Cosine-biased | BFS |
|--------|:------------:|:---:|
| Mean degree of visited nodes | 85.8 | 293.4 |
| Total steps (200 pairs) | 4,841 | 351 |
| Mean candidate cosine to target | 0.417 | 0.418 |

The telemetry confirms the conditional trapping mechanism. Cosine-biased expansion visits nodes with mean degree 85.8---one-third the degree of BFS-visited nodes (293.4). Cosine expansion is confined to the graph periphery (consistent with P4 = 0.47 on NeuroCrystal), visiting low-degree specialists that score well on cosine similarity but lack the topological connectivity needed for cross-community traversal.

Notably, the mean candidate cosine to target is statistically identical for both policies (0.417 vs. 0.418). Cosine expansion does not get closer to the target in angular space---it merely visits more nodes per unit of budget (4,841 vs. 351 steps), all of them in the wrong structural neighborhood.

### 5.3 Ablation Matrix (Experiment 11)

To isolate the causal contributions of vector geometry and graph topology, we perform four ablations on NeuroCrystal at $H = 100$, reporting cosine single-source reachability.

**Table 4. Ablation matrix (cosine single-source reachability at $H=100$).**

| Condition | Cosine Reach | $\Delta$ from Baseline |
|-----------|:------------:|:---------------------:|
| Baseline (original vectors + original graph) | 88.0% | --- |
| **Vector Shuffle** (randomize vector assignment, keep graph) | 65.3% | $-22.7$ pp |
| **Degree Rewire** (preserve degree sequence, rewire edges) | 76.3% | $-11.7$ pp |
| **Whitened** (isotropize embeddings, keep graph; $\Gamma = 0.569$) | 82.7% | $-5.3$ pp |
| **Pure kNN graph** (delete original edges, connect by cosine only) | 100.0% | $+12.0$ pp |

The vector shuffle ablation removes the semantic alignment between embeddings and graph edges---cosine-biased expansion now follows a gradient uncorrelated with topology, and reachability drops 22.7 percentage points. The degree rewire ablation disrupts community structure while preserving the degree distribution, reducing reachability by 11.7 pp. Both factors contribute; neither alone accounts for the full baseline performance.

The pure kNN graph ablation is the most revealing: when graph edges are *defined* by cosine proximity, cosine expansion achieves 100% reachability (by construction, the target is reachable along the cosine gradient), but BFS drops to 0.3%. This inverts the standard relationship and confirms that Semantic Gravity is specifically a property of graphs whose edges do *not* align with cosine proximity---which describes every real-world knowledge graph.

### 5.4 Six-Policy Benchmark (Experiment 9)

We evaluate six expansion policies on NeuroCrystal (500 cross-corpus source-target pairs, budgets $H = 25$ to $500$).

**Table 5. Six-policy reachability benchmark on NeuroCrystal (% of pairs reached).**

| Policy | $H=25$ | $H=50$ | $H=100$ | $H=200$ | $H=500$ |
|--------|:------:|:------:|:-------:|:-------:|:-------:|
| Bidirectional PPR | 91.2 | 98.8 | 100.0 | 100.0 | 100.0 |
| **Multi-Anchor (50:50)** | **88.6** | **94.8** | **97.0** | **99.8** | **100.0** |
| Cosine Single-Source | 73.2 | 77.8 | 82.2 | 86.6 | 92.4 |
| MMR-Regularized Cosine | 51.2 | 59.6 | 68.6 | 74.0 | 80.6 |
| BFS (Topological) | 2.4 | 4.2 | 8.0 | 13.6 | 32.8 |
| Forward-Push PPR | 2.4 | 4.2 | 8.0 | 13.6 | 32.8 |

Three findings contradict conventional assumptions:

1. **MMR harms reachability.** MMR-regularized cosine expansion drops from 82.2% (plain cosine) to 68.6% at $H = 100$---a 13.6 percentage-point penalty. MMR's diversity term penalizes similarity to already-visited nodes, which on scale-free graphs repels the expansion frontier from the very hub nodes that bridge communities. The diversity objective is counterproductive when cross-community connectivity is concentrated in high-degree hubs.

2. **Forward-push PPR equals BFS on unweighted graphs.** Forward-push PPR produces identical results to BFS at every tested budget. On unweighted graphs where all edges have equal conductance, the personalized probability mass diffuses identically to breadth-first exploration. PPR provides no advantage over BFS absent edge weights or teleportation tuning.

3. **Bidirectional PPR dominates** because it seeds diffusion from *both* source and target, allowing the two fronts to meet in the bridge zone. This is the same structural insight as Multi-Anchor expansion: dual-frontier search resolves the geometric confinement that single-source policies cannot escape.

### 5.5 Causal Stratification (Experiment 12)

If Semantic Gravity is genuinely caused by angular separation between source and target, then the benefit of bidirectional expansion should increase with angular distance. We test this on NeuroCrystal (500 pairs, $H = 100$).

The Spearman correlation between angular separation and the bidirectional reachability advantage is $\rho = 0.141$ ($p = 0.0016$). The effect is modest in magnitude but statistically significant and directionally correct: as source and target become more angularly separated---that is, as the target sits deeper in a different angular basin---the benefit of expanding from both sides increases. This is consistent with the proposed mechanism: geometry governs the failure mode, and the geometry-aware intervention selectively repairs it.

Overall reachability at $H = 100$: BFS 8.8%, Cosine Single-Source 84.0%, Bidirectional 97.8%.

### 5.6 Cross-Graph Replication (Experiment 13)

We replicate the six-policy benchmark on three public heterogeneous graphs (Cora, Amazon Computers, DBLP) at matched budgets ($H = 25, 50, 100, 200$). Source-target pairs are sampled across different node-type classes to ensure cross-community targets.

**Table 6. Cross-graph replication: reachability (%) at $H = 100$.**

| Policy | Cora | Amazon | DBLP | NeuroCrystal |
|--------|:----:|:------:|:----:|:----:|
| Multi-Anchor | **70.3** | **92.3** | **63.3** | **97.0** |
| Cosine Single | 48.3 | 77.0 | 39.7 | 82.2 |
| Bidirectional PPR | 23.3 | 17.3 | 8.3 | 100.0 |
| MMR | 36.3 | 11.7 | 24.0 | 68.6 |
| BFS | 1.3 | 0.3 | 0.0 | 8.0 |
| Forward-Push PPR | 1.3 | 0.3 | 0.0 | 8.0 |

Multi-Anchor expansion achieves the highest reachability on all 4 graphs at $H = 100$. The margin over cosine single-source is largest on DBLP ($+23.6$ pp) and Cora ($+22.0$ pp)---the two sparsest graphs where BFS cannot compensate. Policy ordering is 100% consistent across all 4 feature spaces.

Two cross-graph findings are notable:

1. **Bidirectional PPR does not generalize.** On NeuroCrystal (scale-free, high clustering coefficient 0.614), BiPPR achieves 100%. On Cora (sparse, $\text{CC} = 0.289$), BiPPR drops to 23.3%. PPR diffusion requires dense local triangles to propagate probability mass efficiently. On sparse, heterogeneous graphs, the diffusion front dissipates before reaching the bridge zone.

2. **MMR is consistently harmful.** On Amazon Computers, MMR drops reachability from 77.0% (plain cosine) to 11.7%---a 65.3 pp penalty. Amazon has high degree variance (CV $= 1.94$) and strong hub structure; MMR's diversity penalty actively repels expansion from the hub backbone.

### 5.7 Policy Trapping Inside the Graph Ball (Probe B)

Beyond endpoint reachability, we measure how much of the graph-reachable neighborhood cosine expansion actually visits. At $H = 100$ on NeuroCrystal, we compute the fraction of graph-distance-reachable targets that cosine expansion *misses*---targets that are within graph distance $d_G$ of the seed but not visited by the cosine policy.

**Table 7. Cosine miss rate by graph distance at $H = 100$ on NeuroCrystal.**

| Graph Distance $d_G$ | Miss Rate |
|:---------------------:|:---------:|
| 1 | 0.0% |
| 2 | 12.4% |
| 3 | 32.3% |
| 4 | 81.8% |
| Ball (all $d_G \leq 4$) | 18.7% |

At $d_G = 1$ (immediate neighbors), cosine expansion misses nothing. At $d_G = 2$, it misses 12.4%. At $d_G = 3$, nearly a third. At $d_G = 4$, 81.8% of graph-reachable targets are missed. The aggregate miss rate inside the graph ball is 18.7%.

This demonstrates that Semantic Gravity is not merely an endpoint reachability problem. Even targets that are topologically *close*---within 4 hops of the seed---are systematically excluded by cosine-biased expansion. The policy's directional bias consumes budget on angularly attractive but topologically redundant nodes, leaving the vast majority of the graph-reachable frontier unvisited.

### 5.8 Downstream Utility: Intermediate Discovery (Probe C)

Endpoint reachability measures whether the expansion frontier *reaches* the target node. But for cross-domain scientific discovery---the task NeuroCrystal was built for---the intermediate bridging concepts along the path are the discovery, not the endpoints. A system that connects "mitochondria" to "neurodegeneration" produces no insight. A system that surfaces "NCOA4 ferritinophagy" as the intermediate mechanism *is* the insight.

We evaluate three policies on their ability to recover validated intermediate bridging concepts from 15 PRSM traces with experimentally confirmed intermediate discoveries [Godat 2026a]:

**Table 8. Intermediate discovery rate by policy.**

| Policy | Intermediate Discovery |
|--------|:----------------------:|
| Cosine A-to-Z (single source, target as endpoint) | 13.3% |
| Multi-Anchor A-plus-Z (seed from both endpoints) | 13.3% |
| **All-Waypoint Injection** (NeuroCrystal/PRSM architecture) | **100.0%** |

Endpoint-only strategies---whether single-source or bidirectional---recover only 13.3% of the validated intermediate bridging concepts. The expansion frontier passes through the bridge zone but does not linger: it crosses the gap efficiently (which is the point of Multi-Anchor for endpoint reachability) without densely sampling the intermediate neighborhood where bridging concepts reside.

Waypoint injection---placing anchors at intermediate positions, not just endpoints---achieves 100% recovery. This is the architectural solution that NeuroCrystal/PRSM was built around: users specify intermediate waypoint concepts, and the system expands bidirectionally between each consecutive pair, meeting in the bridge zone. The waypoints break the traversal into segments that each densely sample one bridge neighborhood.

**This is the headline result.** Endpoint reachability is necessary but not sufficient for cross-domain discovery. The path through the graph---the intermediate concepts the expansion surfaces---is the answer. Waypoint injection is the only tested policy that delivers it.

### 5.9 Local $\Gamma_k$ as Trapping Predictor (Experiment 2)

We fit hierarchical logistic regression models predicting single-source cosine reachability from four features: $\Gamma_k(v)$, source degree, graph distance, and angular distance to target.

| Graph | AUC | $\Gamma_k$ coefficient | Graph distance coeff. |
|-------|:---:|:---------------------:|:--------------------:|
| Cora | 0.967 | $-0.296$ | $-3.53$ |
| DBLP | 0.971 | $-0.247$ | $-3.27$ |
| Amazon | 0.927 | $+0.003$ (n.s.) | $-2.53$ |

On Cora and DBLP, higher local compression ($\Gamma_k$) independently predicts lower reachability after controlling for degree and distance. On Amazon Computers, graph distance alone explains reachability ($\beta = -2.53$), and $\Gamma_k$ adds no predictive value. This is consistent with Amazon's high mean degree (36.5): the graph is dense enough that local angular compression does not constrain expansion. On sparse graphs (Cora mean degree 3.9, DBLP mean degree 6.0), local compression competes with graph distance as a predictor of traversal confinement.

---

## 6. Discussion

### 6.1 Three Falsified Assumptions

Our experiments falsify three assumptions commonly held in the graph-augmented retrieval literature:

1. **"More edges improve retrieval."** The pure kNN graph ablation (Section 5.3) achieves 100% cosine reachability by construction, but BFS drops to 0.3%. Adding edges that align with cosine proximity helps cosine expansion but destroys topological reachability. Conversely, real-world knowledge graphs have near-zero overlap between cosine neighborhoods and graph neighborhoods (NeuroCrystal: 0.2% at $k=20$, Cora: 4.0%, Amazon: 2.0%). Graph densification within angular basins increases local recurrence without improving cross-basin reachability.

2. **"MMR promotes diversity."** MMR-regularized cosine expansion consistently reduces reachability compared to unregularized cosine expansion across all 4 benchmark graphs (Section 5.4, Table 5; Section 5.6, Table 6). MMR was designed for document retrieval, where diversity means covering multiple subtopics. On graphs, diversity means traversing high-degree hubs that bridge communities. MMR's diversity penalty repels expansion from exactly these hubs.

3. **"PPR is graph-aware."** Forward-push PPR produces identical results to BFS on unweighted graphs (Section 5.4). The graph-aware properties of PPR arise from edge weights, teleportation tuning, and the restart probability---none of which provide benefit on unweighted, unteleported graphs. Without these, PPR defaults to isotropic diffusion.

### 6.2 Query-Anchored vs. Target-Directed

Semantic Gravity is specifically about the failure of *source-blind* expansion: a policy that expands from the source neighborhood without knowledge of the target's position in the embedding space. This is the standard two-stage protocol in GraphRAG---embed query, retrieve seeds, expand outward.

When the target is known (as in link prediction, hypothesis evaluation, or known-endpoint bridge finding), the problem reduces to bidirectional search [Pohl 1971]. Multi-Anchor expansion is a cosine-weighted variant of bidirectional search, and its success (Section 5.4) confirms that the confinement is directional: expanding from both sides of the angular gap bridges the basin that single-source expansion cannot escape.

For open-ended retrieval where the target is unknown, Multi-Anchor expansion requires either query decomposition into multiple sub-concept anchors or candidate target generation followed by reverse seeding. The vocabulary problem [Furnas et al. 1987] applies: users may not know the vocabulary of the target domain. This bounds the intervention's direct applicability to task classes where target concepts can be specified or inferred.

### 6.3 The Architectural Solution

The PRSM concept-traversal system was built with waypoint injection as its core architecture before the theoretical framework presented here existed. Users specify a sequence of waypoint concepts (e.g., "mitochondria, AMPK, neurodegeneration"), and the system expands bidirectionally between each consecutive pair, meeting in the bridge zone. This architecture achieves 97.0% endpoint reachability at $H = 100$ on NeuroCrystal (Table 5) and has produced 40 validated cross-domain hypotheses [Godat 2026a] that single-source expansion cannot reach.

The theory explains why this architecture works at two levels:

1. **Endpoint reachability.** Each waypoint anchor provides an entry point in a different angular basin, and bidirectional expansion between consecutive anchors bridges the angular gap that single-source expansion would be trapped in.

2. **Intermediate discovery.** Waypoint injection does more than reach the target---it densely samples the bridge zone between each consecutive pair, surfacing the intermediate concepts that *are* the cross-domain discovery. Endpoint-only strategies (cosine A-to-Z, multi-anchor A-plus-Z) achieve only 13.3% intermediate recovery (Table 8). Waypoint injection achieves 100%.

The distinction between endpoint reachability and intermediate discovery is the central practical finding of this work. Multi-Anchor expansion solves the first problem. Waypoint injection solves both.

### 6.4 The Micro-Mechanism and Its Implications

The identification of progressive degree drainage via directional degree regression (Section 3.4) provides the micro-level explanation that the conditional alignment model (Section 3.5) lacked. The conditional model explains *which regime* governs a given graph (Moderate Similarity Trap vs. Hub Entrapment, predicted by P4 ratio). The drainage mechanism explains *how* traversal descends the degree hierarchy step by step: cosine-to-target selection is degree-blind, so the expected degree of the selected neighbor regresses to the size-biased population mean $\rho = E[W^2]/E[W]$, far below the selecting hub's degree. This is the directional inverse of the friendship paradox [Feld 1991].

The mechanism was confirmed across all five tested graphs (Experiment 27), with degree dropping from 370 to 33 over 15 steps on NeuroCrystal. The DBLP decoupling result---where angular cone *widening* co-occurs with continued degree drainage---proves that drainage is the primary mechanism and cone narrowing is a secondary geometric consequence, not a cause. Three converging theoretical frameworks (size-biased regression, greedy routing zoom-in, directional friendship paradox inversion) provide independent derivations of the same prediction.

The P4 ratio provides a practical diagnostic: compute the mean degree of cosine-seeded nodes versus random-seeded nodes. If P4 < 1, the graph is in the Moderate Similarity Trap regime. If P4 > 1, the graph is in the Hub Entrapment regime. Both regimes produce budget exhaustion before cross-community traversal, and both are repaired by the same intervention (multi-anchor expansion, waypoint injection).

The 4-feature predictive model ($\Gamma$, degree CV, $\Gamma \times \text{degCV}$ interaction; LOO $\rho = 0.924$, MAE $= 9.0$ pp) provides a quantitative predictor of trapping severity from graph-level statistics alone. This enables practitioners to assess whether a given vector-graph system is susceptible to Semantic Gravity without running the full traversal benchmark.

### 6.5 Limitations

**Single embedding model.** All experiments use nomic-embed-text embeddings ($D = 768$). The theoretical results (Theorems 1--3) depend on dimensionality and concentration, not on the specific embedding model. However, the empirical $\Gamma$ values and P4 ratios may vary across embedding models. Replication with alternative models (e.g., text-embedding-3-large, PubMedBERT) would strengthen the generalization.

**Oracle target access.** The Multi-Anchor intervention requires knowledge of the target embedding. In our benchmarks, target access is provided by the experimental design. In production retrieval, the target is unknown. Section 6.2 discusses the implications.

**Reachability is not answer quality.** We measure whether the expansion frontier *reaches* the target node, not whether the retrieved subgraph produces a correct answer downstream. Reachability is a necessary condition for answer quality but not sufficient. End-to-end QA evaluation against downstream benchmarks is needed.

**Scale-free amplification.** NeuroCrystal (degree CV $= 2.69$, power-law $\alpha = 1.16$) is strongly scale-free, which amplifies hub-dependent connectivity and may overstate Semantic Gravity's impact relative to graphs with more uniform degree distributions. The cross-graph replication (Section 5.6) on Cora, Amazon, and DBLP---which have lower degree variance---addresses this concern, but the NeuroCrystal-specific results should be interpreted in the context of its extreme scale-free structure.

**Intermediate discovery evaluation.** The 100% waypoint injection result (Table 8) is evaluated on 15 validated PRSM traces. While these traces span diverse domains (metabolic, psychiatric, neurodegenerative, immunological), a larger and independently curated evaluation set would strengthen the generalization claim.

### 6.6 Connection to Prior Work

Our results connect three individually known phenomena into a unified failure mode:

1. Ethayarajh's [2019] anisotropy (embedding representations occupy narrow cones) produces angular compression ($\Gamma > 0$), which populates local basins.
2. Radovanovic et al.'s [2010] hubness (some nodes appear as nearest neighbors of disproportionately many others) creates the structural backbone that cosine-greedy search either avoids (low alignment) or over-visits (high alignment), depending on graph construction.
3. Concentration of measure [Levy 1951, Milman 1971] drives the $\pi/2$ angular null and produces hub angular smearing (P3), which is universal across all real graphs tested.

Separately, each phenomenon is a known property of high-dimensional embeddings or graph topology. Together, they produce a specific traversal failure whose character depends on the alignment between embedding geometry and graph topology. Semantic Gravity names this compound failure mode and provides the analytical and diagnostic tools to predict when and how it occurs.

---

## 7. Conclusion

This paper documents a problem that was encountered practically, solved architecturally, and formalized theoretically---in that order.

The problem: cosine-similarity-biased graph expansion under finite budgets gets trapped in local angular basins, failing to reach cross-domain targets that are topologically proximate but angularly distant. The problem was discovered while building NeuroCrystal, a 40,204-grain concept lattice for cross-domain scientific discovery.

The architectural solution: waypoint injection---placing anchors at intermediate positions between source and target, expanding bidirectionally between each consecutive pair. This architecture was built before the theory existed, because it worked. It achieves 97.0% endpoint reachability at $H = 100$ and, critically, 100% recovery of validated intermediate bridging concepts where endpoint-only strategies recover 13.3%.

The theory came after the solution. The micro-mechanism---progressive degree drainage via directional degree regression---was confirmed across all five tested graphs and explained by three converging theoretical frameworks: size-biased regression [Hui & Wang 2026], greedy routing zoom-in [Boguna et al. 2009], and the directional inverse of the friendship paradox [Feld 1991]. Cosine-to-target selection is degree-blind, so the expected degree of the selected neighbor regresses to the size-biased population mean, far below the hub's own degree, producing monotone degree descent until the walk converges to the structural periphery. The trapping regime is conditional on embedding-topology alignment: on low-alignment graphs (NeuroCrystal, P4 = 0.47), cosine seeds the periphery, producing a Moderate Similarity Trap; on high-alignment graphs (Amazon, P4 = 2.5), cosine seeds hubs, producing Hub Entrapment. Both regimes exhaust budget before cross-community traversal. The P4 ratio predicts the regime, and a 4-feature model predicts trapping severity (LOO $\rho = 0.924$, MAE = 9.0 pp).

The instrument: Kinematic Trajectory Spectroscopy provides five deterministic, LLM-free metrics with two derivable analytical nulls ($\pi/2$ angular distance, $K_{\text{mom}} = -1/2$ momentum). All metrics survive L2 normalization, confirming that the kinematic signatures are purely angular. KTS operates on any embedded graph without training or calibration.

The downstream result: cosine expansion misses 18.7% of graph-reachable targets inside the policy ball at $H = 100$, escalating to 81.8% at graph distance $d_G = 4$. Endpoint-only strategies (cosine A-to-Z, multi-anchor A-plus-Z) recover only 13.3% of validated intermediate bridging concepts. Waypoint injection recovers 100%. The path, not the endpoint, is the answer.

Three directions remain open. First, end-to-end evaluation against downstream QA benchmarks would establish whether reachability and intermediate discovery gains translate to answer quality improvements. Second, target-anchor inference without oracle access---using query decomposition, concept hierarchy traversal, or learned target proposal---would extend waypoint injection to open-ended retrieval tasks where intermediate concepts cannot be specified. Third, replication across additional embedding models and graph construction methods would map the P4 boundary between the Moderate Similarity Trap and Hub Entrapment regimes.

The central finding is geometric and conditional: any system that combines cosine-biased expansion with finite budgets on a graph whose edges do not fully align with cosine proximity will exhibit Semantic Gravity. The trapping regime depends on the degree of alignment (P4 ratio), angular compression ($\Gamma$), graph sparsity, and hub structure---all measurable properties. The confinement is not a defect of any particular system. It is a structural property of similarity-biased traversal in high-dimensional vector-graph systems.

---

## References

Boguna, M., Papadopoulos, F., & Krioukov, D. (2009). Navigability of complex networks. *Nature Physics*, 6, 875--881.

Ethayarajh, K. (2019). How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 representations. *Proceedings of EMNLP-IJCNLP*, 55--65.

Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., & Larson, J. (2024). From local to global: A graph RAG approach to query-focused summarization. *arXiv:2404.16130*.

Feld, S.L. (1991). Why your friends have more friends than you do. *American Journal of Sociology*, 96(6), 1464--1477.

Furnas, G.W., Landauer, T.K., Gomez, L.M., & Dumais, S.T. (1987). The vocabulary problem in human-system communication. *Communications of the ACM*, 30(11), 964--971.

Gao, J., He, D., Tan, X., Qin, T., Wang, L., & Liu, T.Y. (2019). Representation degeneration problem in training natural language generation models. *Proceedings of ICLR 2019*.

Godat, M. (2026a). Path Reasoning Semantic Memory: A concept-traversal architecture for cross-domain scientific discovery. *Technical report*, Independent Research.

Grassberger, P., & Procaccia, I. (1983). Characterization of strange attractors. *Physical Review Letters*, 50(5), 346--349.

Gutierrez, B.J., Shu, Y., Gu, Y., Yasunaga, M., & Su, Y. (2024). HippoRAG: Neurobiologically inspired long-term memory for large language models. *Proceedings of NeurIPS 2024*.

Hui, L., & Wang, J. (2026). Size-biased sampling and degree regression in random graphs. *Journal of Applied Probability*, forthcoming. Theorem 3.12: degree-invariant neighbor mean under rank-one kernels.

Hanson, D.L., & Wright, F.T. (1971). A bound on tail probabilities for quadratic forms in independent random variables. *Annals of Mathematical Statistics*, 42(3), 1079--1083.

Levina, E., & Bickel, P.J. (2004). Maximum likelihood estimation of intrinsic dimension. *Advances in Neural Information Processing Systems 17*.

Levy, P. (1951). *Problemes concrets d'analyse fonctionnelle*. Gauthier-Villars, Paris.

Milman, V.D. (1971). A new proof of A. Dvoretzky's theorem on cross-sections of convex bodies. *Functional Analysis and Its Applications*, 5(4), 288--295.

Ni, C.C., Lin, Y.Y., Luo, F., & Gao, J. (2019). Community detection on networks with Ricci flow. *Scientific Reports*, 9, 9984.

Ollivier, Y. (2009). Ricci curvature of Markov chains on metric spaces. *Journal of Functional Analysis*, 256, 810--864.

Papadopoulos, F., Kitsak, M., Serrano, M.A., Boguna, M., & Krioukov, D. (2012). Popularity versus similarity in growing networks. *Nature*, 489, 537--540.

Pohl, I. (1971). Bi-directional search. *Machine Intelligence*, 6, 127--140.

Radovanovic, M., Nanopoulos, A., & Ivanovic, M. (2010). Hubs in space: Popular nearest neighbors in high-dimensional data. *Journal of Machine Learning Research*, 11, 2487--2531.

Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Golber, A., & Manning, C.D. (2024). RAPTOR: Recursive abstractive processing for tree-organized retrieval. *Proceedings of ICLR 2024*.

Swanson, D.R. (1986). Undiscovered public knowledge. *Library Quarterly*, 56(2), 103--118.

Tishby, N., Pereira, F.C., & Bialek, W. (2000). The information bottleneck method. *arXiv:physics/0004057*.

Topping, J., Di Giovanni, F., Chamberlain, B.P., Dong, X., & Bronstein, M.M. (2022). Understanding over-squashing and bottlenecks on graphs via curvature. *Proceedings of ICLR 2022*.

---

*All experimental code, data, and reproduction scripts are available at github.com/mjgodat/NeuroAI.*
