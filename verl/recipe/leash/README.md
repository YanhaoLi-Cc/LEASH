# LEASH: Adaptive Length Penalty and Reward Shaping

This module implements **LEASH** (adaptive **LE**ngth pen**A**lty and reward **SH**aping), an RL-based adaptive length control method for efficient reasoning in LLMs. LEASH formulates length control as a constrained optimization problem and employs a Lagrangian primal-dual method to dynamically adjust the penalty coefficient.

## Algorithm Overview

LEASH extends the DAPO framework by introducing a Lagrangian constraint mechanism to control the length of generated sequences while optimizing task performance.

### 1. Problem Formulation

The goal is to maximize the expected task reward while keeping the average generation length under a target $L_t$:

$$\max_\theta \; J_R(\theta), \quad \text{s.t.} \quad J_P(\theta) \leq 0,$$

where:

$$J_R(\theta) = \mathbb{E}_{x \sim D,\, y \sim \pi_\theta(\cdot|x)}[r(x, y)],$$

$$J_P(\theta) = \mathbb{E}_{x \sim D,\, y \sim \pi_\theta(\cdot|x)}\left[\frac{L(y)}{L_t} - 1\right].$$

### 2. Lagrangian Formulation

We convert the constrained problem into an unconstrained saddle-point problem:

$$\mathcal{L}(\theta, \lambda) = J_R(\theta) - \lambda \cdot J_P(\theta),$$

where $\lambda \geq 0$ is the Lagrange multiplier (dual variable).

### 3. One-Sided Penalized Reward

To avoid incentivizing overly short outputs, LEASH uses a **one-sided** penalty that only penalizes over-length generations:

$$r''(x, y) = r(x, y) - \lambda \cdot \max\left(0,\; \frac{L(y)}{L_t} - 1\right).$$

The reward is further clipped to $[-1, 1]$ for training stability:

$$r'''(x, y) = \text{clip}\left(r''(x, y),\; -1,\; 1\right).$$

At the token level, the penalty is applied at the last response token $T$:

$$\tilde{r}_t = r_t - \lambda \cdot \mathbb{1}[t = T] \cdot \max\left(0,\; \frac{T}{L_t} - 1\right).$$

### 4. Dual Variable (Lambda) Update

After each policy update, $\lambda$ is adjusted based on the batch-level constraint violation:

$$\hat{J}_P = \frac{1}{BG} \sum_{b,i} \left(\frac{L(y_{b,i})}{L_t} - 1\right),$$

$$\lambda_{k+1} = \text{clip}\left(\lambda_k + \alpha_\lambda \cdot \hat{J}_P,\; \lambda_{\min},\; \lambda_{\max}\right).$$

- If average length exceeds target ($\hat{J}_P > 0$), $\lambda$ increases to intensify the penalty.
- If average length is below target ($\hat{J}_P < 0$), $\lambda$ decreases to relax the penalty.

Which can be equivalently written as:

$$\hat{J}_P = \frac{\mathbb{E}[L(y)]}{L_t} - 1$$

