# Chapter 186: VQE Portfolio Optimization

## Introduction — Variational Quantum Eigensolver for Portfolio Optimization

Portfolio optimization sits at the heart of quantitative finance. Since Harry Markowitz introduced Modern Portfolio Theory in 1952, practitioners have sought to find the optimal allocation of assets that maximizes return for a given level of risk. The classical mean-variance optimization framework works well for small portfolios, but as the number of assets grows, the combinatorial complexity of the problem explodes — particularly when introducing discrete constraints such as cardinality limits (maximum number of assets held), minimum lot sizes, or binary inclusion/exclusion decisions.

The **Variational Quantum Eigensolver (VQE)** is a hybrid quantum-classical algorithm originally designed for finding ground-state energies of molecular Hamiltonians in quantum chemistry. However, its structure maps naturally onto combinatorial optimization problems, including portfolio optimization. VQE belongs to the family of **Variational Quantum Algorithms (VQAs)**, which are considered among the most promising candidates for achieving practical quantum advantage on near-term noisy intermediate-scale quantum (NISQ) devices.

The key idea is straightforward: encode the portfolio optimization problem as a Hamiltonian — a matrix whose lowest eigenvalue (ground state energy) corresponds to the optimal portfolio. Then use a parameterized quantum circuit (ansatz) to prepare trial states, measure the expected energy, and iteratively adjust the parameters using a classical optimizer until convergence.

In this chapter, we implement a **classical simulation** of the VQE algorithm in Rust, apply it to cryptocurrency portfolio optimization using real market data from Bybit, and compare the results against traditional optimization approaches.

## Mathematical Foundation

### The Portfolio Optimization Problem

Given $n$ assets with expected return vector $\boldsymbol{\mu} \in \mathbb{R}^n$ and covariance matrix $\Sigma \in \mathbb{R}^{n \times n}$, the classical Markowitz mean-variance optimization seeks weights $\mathbf{w} \in \mathbb{R}^n$ that solve:

$$\min_{\mathbf{w}} \quad \gamma \, \mathbf{w}^T \Sigma \, \mathbf{w} - \mathbf{w}^T \boldsymbol{\mu}$$

subject to $\sum_i w_i = 1$ and $w_i \geq 0$, where $\gamma$ is the risk-aversion parameter.

### QUBO Formulation

To map this onto a quantum-compatible form, we discretize the weights. Each weight $w_i$ is encoded using $K$ binary variables:

$$w_i = \sum_{k=0}^{K-1} \frac{2^k}{2^K - 1} x_{i,k}, \quad x_{i,k} \in \{0, 1\}$$

This transforms the continuous optimization into a **Quadratic Unconstrained Binary Optimization (QUBO)** problem:

$$\min_{\mathbf{x}} \quad \mathbf{x}^T Q \, \mathbf{x}$$

where $Q$ encodes the objective function and constraints (via penalty terms). The QUBO matrix $Q$ is constructed as:

$$Q = \gamma \, \tilde{\Sigma} - \text{diag}(\tilde{\boldsymbol{\mu}}) + \lambda \left( A^T A - 2 \, \mathbf{1}^T A \right)$$

Here $\tilde{\Sigma}$ and $\tilde{\boldsymbol{\mu}}$ are the covariance and return vectors re-expressed in the binary encoding, and $\lambda$ is a penalty multiplier enforcing the budget constraint $\sum_i w_i = 1$.

### QUBO to Ising Hamiltonian

The QUBO is converted to an Ising Hamiltonian by substituting $x_i = (1 - z_i)/2$ where $z_i \in \{-1, +1\}$ are spin variables. The resulting Hamiltonian takes the form:

$$H = \sum_{i} h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j + C$$

where $Z_i$ are Pauli-Z operators, $h_i$ are local fields, $J_{ij}$ are coupling strengths, and $C$ is a constant offset. The ground state of $H$ — the eigenstate with the lowest eigenvalue — corresponds to the optimal binary assignment and thus the optimal portfolio.

### VQE Ansatz

The VQE algorithm uses a parameterized quantum circuit $U(\boldsymbol{\theta})$ to prepare a trial state $|\psi(\boldsymbol{\theta})\rangle = U(\boldsymbol{\theta}) |0\rangle^{\otimes n}$. We use a **hardware-efficient ansatz** consisting of layers of single-qubit rotation gates ($R_y$, $R_z$) followed by entangling CNOT gates:

$$U(\boldsymbol{\theta}) = \prod_{\ell=1}^{L} \left[ \prod_{i} R_y(\theta_{i,\ell}^y) R_z(\theta_{i,\ell}^z) \cdot \text{CNOT-layer} \right]$$

The VQE optimization loop:

1. Prepare $|\psi(\boldsymbol{\theta})\rangle$ using the ansatz circuit
2. Measure the expectation value $\langle \psi(\boldsymbol{\theta}) | H | \psi(\boldsymbol{\theta}) \rangle$
3. Feed the energy to a classical optimizer (e.g., COBYLA, Nelder-Mead)
4. Update $\boldsymbol{\theta}$ and repeat until convergence

## Quantum Advantage Beyond Markowitz

The classical Markowitz optimization with continuous weights can be solved efficiently via quadratic programming. So where does quantum help?

### Combinatorial Explosion with Discrete Constraints

Real-world portfolio optimization often involves constraints that make the problem NP-hard:

- **Cardinality constraints**: Hold at most $k$ out of $n$ assets
- **Minimum position sizes**: If you hold an asset, hold at least some minimum fraction
- **Sector exposure limits**: No more than a specified percentage in any sector
- **Transaction cost thresholds**: Binary buy/sell decisions with fixed costs

With $n$ assets and $K$-bit discretization, the search space is $2^{nK}$ — exponential in both the number of assets and the precision. For 50 assets with 3-bit precision, that is $2^{150} \approx 10^{45}$ possibilities.

### Why VQE Is Promising

1. **Quantum superposition** allows the ansatz to explore exponentially many portfolio combinations simultaneously.
2. **Entanglement** enables the circuit to capture correlations between asset allocations efficiently.
3. **Variational structure** makes VQE robust to noise — a critical advantage on near-term hardware.
4. **Hybrid design** leverages classical optimizers for parameter tuning while quantum hardware handles the exponential state space.

Current quantum hardware is limited to tens of noisy qubits, so a genuine speedup on portfolio problems is still aspirational. However, classical simulations of VQE — like the one we implement here — serve as quantum-inspired heuristics that can outperform naive brute-force approaches on discrete optimization problems.

## Implementation Walkthrough

Our Rust implementation simulates the VQE algorithm classically. The core components are:

### 1. Complex Number Arithmetic

Since quantum states live in complex Hilbert space, we define a `Complex` type storing real and imaginary parts as `f64` pairs. We implement addition, multiplication, conjugation, and modulus operations.

### 2. Pauli Matrices and Quantum Gates

The single-qubit Pauli matrices $I, X, Y, Z$ are represented as 2x2 complex matrices. Rotation gates are built from matrix exponentials:

- $R_x(\theta) = \cos(\theta/2) \, I - i \sin(\theta/2) \, X$
- $R_y(\theta) = \cos(\theta/2) \, I - i \sin(\theta/2) \, Y$
- $R_z(\theta) = \cos(\theta/2) \, I - i \sin(\theta/2) \, Z$

### 3. State Vector Simulation

We represent the $n$-qubit quantum state as a $2^n$-dimensional complex vector. Gate application is performed by constructing the full $2^n \times 2^n$ matrix via tensor products and matrix-vector multiplication.

### 4. QUBO to Ising Conversion

The `qubo_to_ising` function takes a QUBO matrix and returns the local fields $h_i$, couplings $J_{ij}$, and constant offset. This is done by substituting the binary-to-spin transformation and collecting terms.

### 5. Hamiltonian Expectation

Given the Ising parameters and a state vector, the `expectation_value` function computes $\langle \psi | H | \psi \rangle$ by summing contributions from each Pauli-Z term and Z-Z coupling term.

### 6. VQE Optimization Loop

We use a **Nelder-Mead** (downhill simplex) optimizer — a gradient-free method suitable for noisy objective functions. The optimizer repeatedly:

- Proposes new ansatz parameters
- Runs the simulated circuit
- Evaluates the Hamiltonian expectation
- Adjusts parameters based on the simplex geometry

### 7. Portfolio Construction

After VQE converges, the optimal binary string is extracted by measuring the final state (taking the basis state with the highest probability amplitude). The binary string is decoded back into portfolio weights.

## Bybit Crypto Portfolio Optimization

We apply the VQE-based optimizer to a portfolio of four major cryptocurrencies traded on Bybit:

| Asset | Ticker | Role |
|-------|--------|------|
| Bitcoin | BTC/USDT | Store of value, low relative volatility |
| Ethereum | ETH/USDT | Smart contract platform |
| Solana | SOL/USDT | High-performance L1 |
| BNB | BNB/USDT | Exchange token |

### Data Pipeline

1. Fetch daily kline (candlestick) data from Bybit's public API
2. Compute daily log returns for each asset
3. Estimate the mean return vector and covariance matrix
4. Construct the QUBO matrix with risk-aversion parameter $\gamma = 0.5$

### VQE Configuration

- **Qubits**: 8 (2 bits per asset weight for 4 assets)
- **Ansatz layers**: 2
- **Parameters**: 32 rotation angles
- **Optimizer**: Nelder-Mead, 500 iterations
- **Penalty multiplier**: $\lambda = 2.0$ for budget constraint

## Results Comparison: Classical vs Quantum-Inspired

We compare three strategies:

### 1. Equal Weight Benchmark
Each asset receives 25% allocation. This is the naive baseline.

### 2. Classical Mean-Variance (Analytical)
Solve the continuous Markowitz optimization using matrix inversion. This gives the theoretically optimal continuous weights.

### 3. VQE Quantum-Inspired (Discrete)
Run the VQE simulation with 2-bit discretization. Weights are restricted to multiples of $1/3$ (values: 0, 1/3, 2/3, 1).

### Typical Results

| Strategy | BTC | ETH | SOL | BNB | Return | Risk | Sharpe |
|----------|-----|-----|-----|-----|--------|------|--------|
| Equal Weight | 0.25 | 0.25 | 0.25 | 0.25 | Baseline | Baseline | Baseline |
| Classical MV | Varies | Varies | Varies | Varies | Higher | Lower | Best (continuous) |
| VQE Discrete | 0.33 | 0.33 | 0.00 | 0.33 | Comparable | Slightly higher | Good (discrete) |

The VQE approach finds good discrete allocations that closely approximate the continuous optimum. The discretization naturally imposes a form of regularization — preventing extreme allocations that may arise from estimation error in the covariance matrix.

### Advantages of the VQE Approach

1. **Handles discrete constraints natively** — no need for rounding heuristics
2. **Scales to cardinality-constrained problems** where classical QP fails
3. **Provides a natural framework** for adding binary constraints
4. **Quantum-ready** — the same formulation runs on actual quantum hardware when available

### Current Limitations

1. **Exponential classical simulation cost** — simulating $n$ qubits requires $O(2^n)$ memory
2. **Limited precision** — few bits per weight means coarse allocations
3. **Barren plateaus** — deep ansatze can have vanishing gradients
4. **No provable speedup yet** for portfolio-sized instances on current hardware

## Key Takeaways

1. **VQE maps naturally to portfolio optimization** via QUBO encoding of discretized mean-variance problems. The ground state of the corresponding Ising Hamiltonian encodes the optimal portfolio weights.

2. **The QUBO formulation handles discrete constraints natively**, making it ideal for real-world portfolio problems with cardinality limits, minimum position sizes, and binary inclusion/exclusion decisions.

3. **Classical VQE simulation serves as a quantum-inspired heuristic** that, while not faster than specialized classical solvers for small problems, validates the approach and provides a migration path to quantum hardware.

4. **The ansatz design matters enormously** — too shallow and the circuit cannot express the optimal state; too deep and optimization suffers from barren plateaus. Hardware-efficient ansatze with 1-3 layers work well for small instances.

5. **For crypto portfolios on Bybit**, the VQE approach produces sensible discrete allocations that capture the essential structure of the continuous optimum while avoiding the overfitting risks inherent in continuous optimization with noisy return estimates.

6. **Rust provides an excellent platform** for quantum simulation — its performance characteristics and type system make it well-suited for the numerically intensive linear algebra operations at the core of state-vector simulation.

7. **The gap between quantum promise and current reality remains significant**, but the algorithmic framework is mature. When fault-tolerant quantum computers become available, the same QUBO/VQE formulations will transfer directly, potentially providing genuine speedups for large, highly-constrained portfolio optimization problems.
