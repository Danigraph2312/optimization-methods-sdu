# Optimization Methods — Python

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-scientific-013243?style=flat&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-optimization-8CAAE6?style=flat&logo=scipy&logoColor=white)
![Grade](https://img.shields.io/badge/Grade-12%2F12_·_30%2F30_cum_laude-brightgreen?style=flat)
![University](https://img.shields.io/badge/SDU_Odense-Erasmus_Exchange-red?style=flat)

**Course:** AI505 – Optimization  
**Institution:** University of Southern Denmark (SDU), Odense — Erasmus Exchange  
**Year:** Spring 2025  
**Grade:** 12/12 (Danish scale) · converted to 30/30 cum laude

---

## Overview

Four assignments covering the theory and practice of mathematical optimization, from classical calculus-based differentiation to large-scale metaheuristics for combinatorial problems. All implementations are in **Python**, with a strong emphasis on both analytical understanding and computational experimentation. Each assignment is accompanied by a **LaTeX report** with full mathematical derivations.

---

## Assignment 1 — Differentiation

Comparative study of three differentiation paradigms applied to scalar and multivariate functions.

**Analytical differentiation** of three functions:

$$f_1(x) = x^3 - 2x^2 + 3x - 5, \quad f_2(x) = e^x \sin(x), \quad f_3(x) = \frac{1}{1+x^2}$$

**Numerical differentiation** — forward, central, and backward finite differences at $x = 1$ with step size $h = 10^{-5}$. Error analysis as a function of $h$ on a log-log scale, showing the trade-off between truncation error and floating-point cancellation.

**Automatic differentiation (AD)** — implemented via computational graphs and the chain rule. Results match analytical values exactly (errors on the order of machine precision), confirming AD's superiority over numerical methods.

**Multivariate and higher-order differentiation** of $f_4(x, y) = x^2 y + e^x \sin(y)$: gradient $\nabla f_4$ and Hessian $H(f_4)$ computed analytically and verified via AD at $(x, y) = (3, 5)$.

**Descent Direction Iteration Methods** — five algorithms benchmarked on the COCO platform across dimensions 2D to 40D: Nelder-Mead Simplex, Powell's Method, L-BFGS, BFGS, and Conjugate Gradient. Key finding: derivative-based methods (BFGS, L-BFGS) maintain consistent performance as dimensionality increases, while derivative-free methods (Powell, Nelder-Mead) degrade sharply beyond 5D.

---

## Assignment 2 — Optimization for Machine Learning

Eight gradient-based optimizers implemented and benchmarked on **LeNet-5** trained on MNIST, across three loss functions: Cross-Entropy Loss (CEL), Regularized CEL (RegCEL), and Sparse-weight CEL (SparseCEL).

| Optimizer | Type | Update Rule |
|---|---|---|
| SGD | First-order stochastic | $w_{k+1} \leftarrow w_k - \alpha_k \nabla f_{i_k}(w_k)$ |
| Mini-batch SGD | First-order stochastic | Averaged gradient over random subset $S_k$ |
| Batch SGD (Steepest Descent) | First-order batch | Full-gradient descent |
| SGD + Momentum (Heavy Ball) | First-order w/ memory | $w_{k+1} \leftarrow w_k - \alpha_k \nabla f_{i_k}(w_k) + \beta_k(w_k - w_{k-1})$ |
| SGD + Nesterov | First-order w/ lookahead | Gradient evaluated at $\tilde{w}_k = w_k + \beta_k(w_k - w_{k-1})$ |
| Natural Gradient | Second-order (Fisher) | $w_{k+1} \leftarrow w_k - \alpha_k G^{-1}(w_k) \nabla F(w_k)$ |
| L-BFGS | Quasi-Newton | Approximate inverse Hessian via limited memory |
| Iterate Averaging | Noise reduction | Running average of iterates $\tilde{w}_k = \frac{1}{k+1}\sum_{j=1}^{k+1} w_j$ |
| Adam | Adaptive moments | Bias-corrected decaying momentum + squared gradient |

**Key findings:** Adam and L-BFGS consistently achieve the highest accuracy and fastest convergence across CEL and RegCEL. Under SparseCEL, iterate averaging proves most robust due to its trajectory-smoothing effect. Batch SGD provides the cleanest loss curves but at the cost of per-epoch speed.

---

## Assignment 3 — Game Theory & Linear Programming

A zero-sum game between two commanders (Antigonus and Brasidas) allocating legions across mountain passes, solved via linear programming.

**Problem setup:** Each player distributes $n$ legions across 3 passes. A player wins a pass if they allocate strictly more legions to it. The overall winner controls the majority of passes. With 7 legions, the number of valid allocations per player is:

$$\binom{n + k - 1}{k - 1} = \binom{9}{2} = 36 \quad \Rightarrow \quad P \in \mathbb{R}^{36 \times 36}$$

**Payoff matrix:** $P_{ij} \in \{-1, 0, 1\}$ depending on whether Antigonus wins, ties, or loses against Brasidas' strategy $j$. Visualised as a heatmap revealing diagonal block structure.

**LP formulation (Antigonus — row player):**

$$\max_{x, v} \; v \quad \text{s.t.} \quad \sum_i x_i P_{ij} \geq v \;\; \forall j, \quad \sum_i x_i = 1, \quad x_i \geq 0$$

**LP formulation (Brasidas — column player, dual):**

$$\min_{y, w} \; w \quad \text{s.t.} \quad \sum_j y_j P_{ij} \leq w \;\; \forall i, \quad \sum_j y_j = 1, \quad y_j \geq 0$$

Solved using `scipy.optimize.linprog` with the HiGHS backend.

**Results:** Both players' optimal strategies are mixed, assigning probability $\frac{1}{3}$ uniformly over 3 pure strategies each. The game value is $v = 0$, confirming that the game is **perfectly fair** under optimal play. This result holds across all tested legion counts (5–9): the support size varies but the expected payoff remains near zero, and a pure strategy is never optimal.

---

## Assignment 4 — Combinatorial Optimization: Candle Race

A custom NP-hard combinatorial problem combining routing (TSP-like) and scheduling, solved via Variable Neighborhood Search (VNS) under a 60-second time budget.

**Problem definition:** A player starts at position $V_0$ at time $t = 0$. Each village $V_i$ has a candle with height $h_i$ and burning rate $b_i$. The candle score upon arrival at time $t_{v_i}$ is:

$$s_i = \max(0,\; h_i - b_i \cdot t_{v_i})$$

Travel time between villages is measured using Manhattan distance. The goal is to find an ordered subset $v = \{v_1, \ldots, v_k\} \subseteq V$ maximising total score $S(v) = \sum_{i=1}^{k} s_i$.

**Implementation** follows the ROAR-NET specification with four abstractions: `Problem`, `Solution`, `Move`, `Neighborhood`.

Two neighbourhood structures are defined: **Insert** (relocate village $i$ to position $j$) and **Swap** (exchange positions of two villages). Initial solutions are generated via a greedy heuristic (maximise immediate score gain at each step) or randomly.

**VNS algorithm:** alternates between Insert and Swap local search (first-improvement), then applies a double-bridge 4-opt perturbation to escape local optima. Distance matrix is precomputed in $O(n^2)$; move evaluation is $O(1)$ via incremental delta scoring.

**Computational results:**

| Instance | Villages | Visited | Score |
|---|---|---|---|
| `sample.txt` | 4 | 2 | 778 |
| `berlin52_1.txt` | 51 | 6 | 123,140 |
| `d493_2.txt` | 492 | 148 | 150,487,264 |
| `d1291_3.txt` | 1,290 | 300 | 763,033,142 |

The algorithm scales to instances with $n > 1000$ villages within the time limit.

---

## Repository Structure

```
optimization-methods-sdu/
├── assignment1-differentiation/
│   ├── opt_assign1.ipynb
│   └── opt_assignment1.pdf
├── assignment2-ml-optimization/
│   ├── asg_[2]_.py
│   └── asg_2_.pdf
├── assignment3-game-theory/
│   ├── asg_[3].py
│   └── asg_3_.pdf
├── assignment4-candle-race/
│   ├── candle_race.py
│   └── asg_4_.pdf
└── README.md
```

---

## Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | All implementations |
| NumPy / SciPy | Linear algebra, LP solving (`linprog` / HiGHS) |
| PyTorch | LeNet-5 training (Assignment 2) |
| Matplotlib / Seaborn | Visualisation |
| LaTeX | Reports with full mathematical derivations |
