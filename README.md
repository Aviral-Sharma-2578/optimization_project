# Optimization Algorithms for Supply Chain and Facility Location Problems

This repository contains Python implementations of optimization algorithms and their applications to supply chain and facility location problems. Each file corresponds to a specific problem formulation and its associated solution approach.

---

## Problem Statement for `subgradient.py` and `surrogate.py`

### Problem 1: Subgradient Method for Lagrangian Dual

Given a Lagrangian multiplier \( \lambda^{(k)} \) at iteration \( k \), we aim to calculate the updated multiplier \( \lambda^{(k+1)} \) by following the subgradient direction with a properly selected step length \( \alpha_k \):
\[
\lambda^{(k+1)} = \max \{ 0, \lambda^{(k)} - \alpha_k (b - Ax^{(k)}) \},
\]
where \( x^{(k)} \) is the optimal solution of the primal subproblem \( Z_L(\lambda^{(k)}) \). 

**Problem Formulation:**
Here's the problem statement and formulation for Problem 1 (Fisher, 1985):

\[
Z_P = \max \{ 16x_1 + 10x_2 + 4x_4 \},
\]
subject to:
\[
8x_1 + 2x_2 + x_3 + 4x_4 \leq 10, \\
x_1 + x_2 \leq 1, \\
x_3 + x_4 \leq 1, \\
0 \leq x \leq 1 \text{ and integer}.
\]

**Dual Problem:**
If we remove the first constraint and associate it with multiplier \( \lambda \geq 0 \), the Lagrangian dual becomes:
\[
Z_L(\lambda) = \max \{ (16 - 8\lambda)x_1 + (10 - 2\lambda)x_2 + (0 - \lambda)x_3 + (4 - 4\lambda)x_4 + 10\lambda \},
\]
subject to:
\[
x_1 + x_2 \leq 1, \\
x_3 + x_4 \leq 1, \\
0 \leq x \leq 1 \text{ and integer}.
\]

We solve this problem using the subgradient method with varying \( \alpha_k \): constant \( \alpha_k \), and diminishing \( \alpha_k = \alpha_{k-1}/2, \alpha_{k-1}/3 \).

---

### Problem 2: Surrogate Subgradient Method

The surrogate subgradient method differs from the regular subgradient method in that it only approximately solves **one subproblem** at each iteration. Specifically, the surrogate subgradient method follows these steps:

1. Perform an approximate optimization to obtain \( x^{k+1} \) such that:
\[
\tilde{L}(\lambda^{k+1}, x^{k+1}) < \tilde{L}(\lambda^{k+1}, x^k),
\]
where:
\[
\tilde{L}(\lambda, x) = \sum_{i=1}^I J_i(x_i) + \lambda^T (Ax - b).
\]
2. Update \( \lambda \) using the subgradient:
\[
\lambda^{(k+1)} = \max \{ 0, \lambda^{(k)} - \alpha_k (b - Ax^{(k)}) \}.
\]

---

## Problem Statement for `tabu_search.py`

### Problem 3: Capacitated Plant Location Problem (CPL)

A company producing cattle forage is looking to supply seven farms with daily forage. To achieve this, the company has identified six potential silo locations to store the forage. Each farm has a daily demand, and each silo has a maximum daily capacity. The goal is to minimize the total cost (transportation + facility costs) while ensuring all demands are satisfied and capacity constraints are met.

**Problem Description:**
- **Farms:** Each farm \( j \) has a daily forage demand (in quintals) of:
  \( \{ 36, 42, 34, 50, 27, 30, 43 \} \).
- **Silos:** Each potential silo \( i \) has a maximum daily throughput of:
  \( \{ 80, 90, 110, 120, 100, 120 \} \).

**Costs:**
1. **Facility Fixed Costs:** The daily cost of activating silo \( i \) is:
   \[
   \text{Daily Cost} = \frac{\text{Total Cost for 4 Years}}{1461 \text{ days}}.
   \]
2. **Transportation Costs:** For each pair \( (i, j) \), the transportation cost is calculated as:
   \[
   c_{ij} = 0.06 \times 2 \times \text{Distance}(i, j) \times \text{Demand}_j + 0.15 \times \text{Demand}_j.
   \]

**Mathematical Model:**

**Decision Variables:**
- \( y_i \): Binary decision variable indicating whether silo \( i \) is activated.
- \( x_{ij} \): Fraction of demand from farm \( j \) satisfied by silo \( i \).

**Objective Function:**
\[
\min \sum_{i=1}^6 \sum_{j=1}^7 c_{ij} x_{ij} + \sum_{i=1}^6 \text{Daily Facility Cost}_i y_i.
\]

**Constraints:**
1. **Demand Satisfaction:**
   \[
   \sum_{i=1}^6 x_{ij} = 1, \quad \forall j \in \{1, ..., 7\}.
   \]
2. **Capacity Limits:**
   \[
   \sum_{j=1}^7 \text{Throughput}_{ij} \leq \text{Capacity}_i y_i, \quad \forall i \in \{1, ..., 6\}.
   \]
3. **Variable Domains:**
   \[
   x_{ij} \geq 0, \quad y_i \in \{0, 1\}.
   \]

**Output:**
The optimal solution ensures facilities 1, 5, and 6 are capitalized, and the total daily cost is minimized to **1218.08**.

---

## Files in This Repository

1. `subgradient.py`: Implements the subgradient method for solving the Lagrangian dual problem.
2. `surrogate.py`: Implements the surrogate subgradient method for solving a modified Lagrangian dual.
3. `tabu_search.py`: Implements the Tabu Search algorithm to solve the capacitated plant location problem.

---
