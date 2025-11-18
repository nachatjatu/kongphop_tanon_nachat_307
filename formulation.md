# 1. Introduction
The ambulance depot problem is naturally formulated as a Wasserstein distributionally robust optimization (DRO) problem (Kuhn et al., 2019). Unlike other forms of DRO that use moment-based or f-divergences, in Wasserstein DRO we use our historical data as an empirical distribution and allow our adversary to choose any distribution with the same domain with Wasserstein distance $\leq \varepsilon$ from our empirical distribution. Intuitively, this means that the adversary can choose distributions not encountered in the past, but they cannot choose ones that are *too* different.

It is important to note why we require the $\varepsilon$ ball constraint: if our adversary were free to choose _any_ distribution, then they could choose pathological distributions that make our robust solutions unrealistically conservative. For example, consider an accident site in suburban Bangkok that very rarely has accidents, and is far away from any ambulance site. A free adversary could put all probability mass on this accident site and force us to concentrate resources on this location, which is totally unrealistic.

# 2. Definitions and Empirical Distribution
Thus, we proceed to formulating the problem. Suppose there are $N_a$ accident locations, $N_d$ depot locations, $N_t$ TMC locations, which define a directed graph $G = (V, E)$ where $|V| = N_a + N_d + N_t$. 

Each data point is a tuple $(A_t, C_t)$ where $A_t \in \{0, 1\}^{N_a}$ encodes whether an accident occurred at each accident location and $C_t \in \{1, 2, 3\}^{|E|}$ assigns a _congestion level_ from 1-3 to each edge. Define $\mathcal{Z} = \mathcal{A}, \mathcal{C}$ as the spaces of accident and congestion scenarios, respectively. Then, our $N$ data points define an empirical distribution over $\mathcal{Z}$, given by

$$
\hat{\mathbb{P}} = \frac{1}{N} \sum_{i=1}^N \delta_{(A_i, C_i)}
$$

where $\delta$ is the Dirac delta function. 

# 3. Optimization Problem
Now, let $y \in \{0, 1\}^M$ be a binary vector where $y_i=1$ if depot $i$ is used, and $0$ if not. 

Our Wasserstein DRO is written as

$$
\inf_{y \in \{0, 1\}^M} \sup_{\mathbb{Q} \in \mathbb{B}_{\varepsilon, p}(\hat{\mathbb{P}}_N)}f(\mathbb{Q}, y)
$$

where $f$ is the loss function. 

## 3.1 Ground Metric for Wasserstein DRO
To use the Wasserstein DRO, we need to define a metric on $\mathcal{Z}$. First, note that $\mathcal{C}$ is discrete but with meaningful values, so a natural metric for it is 

$$
d_C(c_1, c_2) = \| c_2 - c_1 \|_1 = \sum_{i=1}^{|E|} |c_{2i} - c_{1i}|.
$$

Now, $\mathcal{A}$ is a little more difficult because it encodes accidents that are distributed throughout space. To do this, we can use the 1-Wasserstein distance; treat accident scenario $a$ as a distribution

$$
P_a = \frac{1}{\sum_{i: a_i=1}} \sum_{j: a_j = 1} \delta_j
$$

so that the 1-Wasserstein distance is

$$
d_A(a_1, a_2) = W_1(P_{a_1}, P_{a_2}).
$$

Thus a metric on $\mathcal{Z}$ is given by

$$
d((a_1, c_1), (a_2, c_2)) = d_A(a_1, a_2) + \lambda \|c_2 - c_1 \| 
$$

where $\lambda > 0$ is a scaling factor.

# 3.2 Loss Function
In our problem, we can define $f$ as the _expected response time_, so that

$$
f(\mathbb{Q}, y) = \mathbb{E}_{(a, c) \sim \mathbb{Q}}[t(y, a, c)]
$$

where $t(y,a,c)$ is the deterministic _____ response time across accidents $a$ under congestion $c$ and deployment $y$. 

We can take $t$ to be the _average response time_ so that 

$$
t(y, a, c) = \frac{1}{\sum_i a_i} \sum_{j: a_j = 1} \tau(y, j|c)
$$

where $\tau$ is the minimum travel time from any depot in $y$ to site $j$ under congestion $c$.

We can write $\tau$ as a minimization problem $\tau(y, j|c) = \min_{i: y_i = 1} T(i, j|c)$ where $T$ is the minimum travel time from depot $i$ to site $j$ under congestion $c$. This can be written as an (MI)LP to support the solution of this problem.

**Thus our optimization problem is given by**

$$
\min_{y \in \{0, 1\}^M} \sup_{\mathbb{Q} \in \mathbb{B}_{\varepsilon, p}(\hat{\mathbb{P}}_N)} \mathbb{E}_{(a,c) \sim \mathbb{Q}} \left[ t(y,a,c)\right]
$$

# 4. Dualizing the DRO
For convenience, write $l(z) =t(y,z)$ where $z = (a,c)$, thus we have

$$
\min_{y \in \{0, 1\}^M} \sup_{\mathbb{Q} \in \mathbb{B}_{\varepsilon, p}(\hat{\mathbb{P}}_N)} \mathbb{E}_{(a,c) \sim \mathbb{Q}} \left[ l(z)\right]
$$

By Theorem 7 in (Kuhn et al., 2019), we can re-write with Strong Duality

$$
\sup_{\mathbb{Q} \in \mathbb{B}_{\varepsilon, p}(\hat{\mathbb{P}}_N)} \mathbb{E}_{\mathbb{Q}} \left[ l(Z)\right] = \inf_{\gamma \geq 0}\mathbb{E}^{\hat{\mathbb{P}}_N}[l_\gamma(\xi_i)] + \gamma \varepsilon^p =\inf_{\gamma \geq 0}\left[\frac{1}{N} \sum_{i=1}^N l_\gamma(\xi_i) + \gamma \varepsilon^p\right]
$$

where $l_\gamma(\xi_i) = \sup_{z \in \mathbb{Z}} [l(z) - \gamma d(z, \xi_i)^p]$. Choosing $p=1$, we have the dualized DRO

$$
\min_{y \in \{0, 1\}^M, \gamma \geq 0} \left[ \frac{1}{N} \sum_{i=1}^N \sup_{z \in \mathbb{Z}} [l(z) - \gamma d(z, \xi_i)] + \gamma \varepsilon \right]
$$

# 5. Manipulaing the DRO to get single-level optimization
If we write the expression in full, we get

$$
\begin{aligned}
&\min_{y \in \{0, 1\}^M, \gamma \geq 0} \left[ \frac{1}{N} \sum_{i=1}^N \sup_{(a,c) \in \mathbb{Z}} [t(y, a, c) - \gamma d((a, c), (A_i, C_i))] + \gamma \varepsilon \right] \\
&\quad = \min_{y \in \{0, 1\}^M, \gamma \geq 0} \left[ \frac{1}{N} \sum_{i=1}^N \sup_{(a,c) \in \mathbb{Z}} [\frac{1}{|a|}\sum_{k \in A(a)} \min_{j \in D(y)} T(j, k | c) - \gamma d((a, c), (A_i, C_i))] + \gamma \varepsilon \right]  
\end{aligned}
$$

which is awful and nasty and should never be looked at. But we must. 

Observe that $D(y)$--the set of open depots under deployment $y$--is a finite set. Then, we can linearize $\min_{j \in D(y)} T(j, k | c)$ by introducing assignment variables $x_{jk}$ such that $x_{jk} = 1$ if depot $j$ serves accident site $k$, and constrain

$$
\begin{aligned}
\sum_{k}x_{jk} &= 1 \quad \text{each accident is served}\\
x_{jk} &\leq y_j \quad \text{only open depots can serve accidents}
\end{aligned}
$$

rewrite

$$
\min_{j \in D(y)} T(j, k | c) = \sum_{j} T(j, k | c) x_{jk}
$$

so we have

$$
\begin{aligned}
\min_{y \in \{0, 1\}^M, \gamma \geq 0} \left[ \frac{1}{N} \sum_{i=1}^N \sup_{(a,c) \in \mathbb{Z}} [\frac{1}{|a|}\sum_{k \in A(a)} \sum_{\text{depots }j} T(j, k | c) x_{jk} - \gamma d((a, c), (A_i, C_i))] + \gamma \varepsilon \right]  
\end{aligned}
$$

Now, observe that $T(j, k | c)$ is the solution to the shortest path minimization problem where "length" is the travel time for each edge. Thus we can write

$$
\begin{aligned}
T(j, k | c) &= \min_{\text{paths}} \sum_{e} t_e = \max_{\pi} \pi(k) - \pi(j) \quad \text{ s.t. } \pi(v) - \pi(u) \leq t(u, v), \quad \forall u, v \in V
\end{aligned}
$$

Thus we obtain

$$
\begin{aligned}
\min_{y \in \{0, 1\}^M, \gamma \geq 0} \left[ \frac{1}{N} \sum_{i=1}^N \max_{(a,c) \in \mathbb{Z}, \pi} [\frac{1}{|a|}\sum_{k \in A(a)} \sum_{\text{depots }j} (\pi(k) - \pi(j)) x_{jk} - \gamma d((a, c), (A_i, C_i))] + \gamma \varepsilon \right]  
\end{aligned}
$$

@incollection{kuhn2019wasserstein,
  title={Wasserstein distributionally robust optimization: Theory and applications in machine learning},
  author={Kuhn, Daniel and Esfahani, Peyman Mohajerin and Nguyen, Viet Anh and Shafieezadeh-Abadeh, Soroosh},
  booktitle={Operations research \& management science in the age of analytics},
  pages={130--166},
  year={2019},
  publisher={Informs}
}
