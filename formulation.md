The ambulance depot problem is naturally formulated as a Wasserstein distributionally robust optimization (DRO) problem (Kuhn et al., 2019). Unlike other forms of DRO that use moment-based or f-divergences, in Wasserstein DRO we use our historical data as an empirical distribution and allow our adversary to choose any distribution with the same domain with Wasserstein distance $\leq \varepsilon$ from our empirical distribution. Intuitively, this means that the adversary can choose distributions not encountered in the past, but they cannot choose ones that are *too* different.

It is important to note why we require the $\varepsilon$ ball constraint: if our adversary were free to choose _any_ distribution, then they could choose pathological distributions that make our robust solutions unrealistically conservative. For example, consider an accident site in suburban Bangkok that very rarely has accidents, and is far away from any ambulance site. A free adversary could put all probability mass on this accident site and force us to concentrate resources on this location, which is totally unrealistic.

Thus, we proceed to formulating the problem. Suppose there are $N_a$ accident locations, $N_d$ depot locations, $N_t$ TMC locations, which define a directed graph $G = (V, E)$ where $|V| = N_a + N_d + N_t$. 

Each data point is a tuple $(A_t, C_t)$ where $A_t \in \{0, 1\}^{N_a}$ encodes whether an accident occurred at each accident location and $C_t \in \{1, 2, 3\}^{|E|}$ assigns a _congestion level_ from 1-3 to each edge.
Define $\mathcal{A}, \mathcal{C}$ as the spaces of accident and congestion scenarios, respectively. Then, our $N$ data points define an empirical distribution over $\mathcal{A} \times \mathcal{C}$, given by

$$
\hat{\mathbb{P}} = \frac{1}{N} \sum_{i=1}^N \delta_{(A_i, C_i)}
$$

where $\delta$ is the Dirac delta function. Now, let $y \in \{0, 1\}^M$ be a binary vector where $y_i=1$ if depot $i$ is used, and $0$ if not. 

Our Wasserstein DRO is written as

$$
\inf_{y \in \{0, 1\}^M} \sup_{\mathbb{Q} \in \mathbb{B}_{\varepsilon, p}(\hat{\mathbb{P}}_N)}f(\mathbb{Q}, y)
$$

where $f$ is the loss function. In our problem, we can define $f$ as the _expected response time_, so that

$$
f(\mathbb{Q}, y) = \mathbb{E}_{(a, c) \sim \mathbb{Q}}[t(y, a, c)]
$$

where $t(y,a,c)$ is the deterministic _____ response time across accidents $a$ under congestion $c$ and deployment $y$. 

If we take $t$ to be the _average response time_ so that 

$$
t(y, a, c) = \frac{1}{\sum_i a_i} \sum_{j: a_j = 1} \tau(y, j|c)
$$

where $\tau$ is the minimum travel time from any depot in $y$ to site $j$ under congestion $c$.

We can write $\tau$ as a minimization problem $\tau(y, j|c) = \min_{i: y_i = 1} T(i, j|c)$ where $T$ is the minimum travel time from depot $i$ to site $j$ under congestion $c$. This can be written as an (MI)LP to support the solution of this problem.

**Thus our optimization problem is**

$$
\min_{y \in \{0, 1\}^M} \sup_{\mathbb{Q} \in \mathbb{B}_{\varepsilon, p}(\hat{\mathbb{P}}_N)} \mathbb{E}_{(a,c) \sim \mathbb{Q}} \left[\frac{1}{\sum_i a_i} \sum_{j: a_j = 1} \min_{i: y_i = 1} T(i, j|c) \right]
$$






@incollection{kuhn2019wasserstein,
  title={Wasserstein distributionally robust optimization: Theory and applications in machine learning},
  author={Kuhn, Daniel and Esfahani, Peyman Mohajerin and Nguyen, Viet Anh and Shafieezadeh-Abadeh, Soroosh},
  booktitle={Operations research \& management science in the age of analytics},
  pages={130--166},
  year={2019},
  publisher={Informs}
}
