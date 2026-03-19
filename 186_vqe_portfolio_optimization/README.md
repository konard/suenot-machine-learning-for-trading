# Chapter 186: VQE Portfolio Optimization

## 1. Introduction

The **Variational Quantum Eigensolver (VQE)** is a hybrid quantum-classical algorithm originally designed to find the ground state energy of molecular Hamiltonians. In recent years, researchers have recognized that the same mathematical framework maps elegantly onto combinatorial optimization problems in finance, and portfolio optimization in particular.

Portfolio optimization --- the task of selecting asset weights that maximize expected return for a given level of risk --- is one of the most fundamental problems in quantitative finance. Harry Markowitz formalized this in 1952 as the **mean-variance optimization** problem. While classical solvers handle modest portfolios efficiently, the combinatorial explosion that arises when we add integer constraints (e.g., selecting exactly *k* assets from a universe of *N*), sector constraints, or transaction cost thresholds makes the problem NP-hard in the general case. This is precisely the regime where quantum computing promises an advantage.

VQE is especially attractive because it is designed for **Noisy Intermediate-Scale Quantum (NISQ)** devices. Unlike algorithms that require deep circuits (e.g., Quantum Phase Estimation), VQE uses shallow parameterized circuits whose parameters are optimized by a classical computer. This makes it one of the most practical quantum algorithms available today.

In this chapter, we will:

- Derive the mathematical mapping from Markowitz optimization to a Quadratic Unconstrained Binary Optimization (QUBO) problem.
- Show how the QUBO is encoded as a Hamiltonian whose ground state corresponds to the optimal portfolio.
- Walk through the VQE algorithm and its quantum-classical feedback loop.
- Implement a full VQE portfolio optimizer in Rust, including a statevector simulator.
- Integrate real cryptocurrency market data from the Bybit exchange API.

## 2. Mathematical Foundation

### 2.1 Markowitz Mean-Variance Optimization

The classical portfolio optimization problem seeks a weight vector **w** that minimizes portfolio variance subject to a target return:

```
minimize:    w^T * Sigma * w
subject to:  mu^T * w >= r_target
             sum(w_i) = 1
             w_i >= 0
```

where **Sigma** is the covariance matrix of asset returns, **mu** is the vector of expected returns, and `r_target` is the desired minimum return.

### 2.2 QUBO Formulation

To map this problem onto a quantum computer, we discretize the portfolio weights. For *N* assets, we introduce binary decision variables. In the simplest formulation, each binary variable `x_i in {0, 1}` indicates whether asset *i* is included in the portfolio (with equal weight `1/k` where *k* is the number of selected assets).

The objective function becomes a **Quadratic Unconstrained Binary Optimization (QUBO)** problem:

```
minimize:  x^T * Q * x
```

where the QUBO matrix **Q** encodes both the objective and penalty terms for constraint violations:

```
Q_ij = lambda_risk * Sigma_ij / k^2
     - lambda_return * mu_i * delta_ij
     + lambda_budget * (1 - 2*k*delta_ij + 2*(1 - delta_ij))
```

More precisely, the QUBO matrix is constructed as:

```
Q = lambda_risk * Sigma_normalized
  - lambda_return * diag(mu)
  + lambda_budget * (ones_matrix - 2*k*I + k^2*I)
```

Here:
- `lambda_risk` penalizes portfolio variance.
- `lambda_return` rewards expected return.
- `lambda_budget` enforces the constraint that exactly *k* assets are selected.

### 2.3 Hamiltonian Encoding

The QUBO objective maps to an Ising Hamiltonian by substituting `x_i = (1 - Z_i) / 2`, where `Z_i` is the Pauli-Z operator acting on qubit *i*. The resulting Hamiltonian is:

```
H = sum_ij Q_ij * (1 - Z_i)(1 - Z_j) / 4
```

The ground state of this Hamiltonian corresponds to the binary string that minimizes the QUBO objective, which in turn gives us the optimal asset selection.

### 2.4 VQE Algorithm

VQE finds the ground state energy of Hamiltonian **H** using the **variational principle**:

```
E_0 <= <psi(theta)| H |psi(theta)>
```

for any parameterized quantum state `|psi(theta)>`. The algorithm:

1. Prepare a parameterized quantum state (ansatz) `|psi(theta)>` on the quantum computer.
2. Measure the expectation value `<H>` with respect to this state.
3. Feed the energy value to a classical optimizer.
4. The classical optimizer updates **theta** to minimize the energy.
5. Repeat until convergence.

The ansatz circuit typically consists of layers of single-qubit rotation gates (RY, RZ) followed by entangling CNOT gates. This structure is called a **hardware-efficient ansatz**.

## 3. Quantum-Classical Hybrid Loop

The VQE algorithm is the canonical example of a **variational quantum-classical algorithm**. Understanding the interplay between the quantum and classical components is essential.

### 3.1 The Quantum Component

The quantum processor's job is to:

1. **State Preparation**: Apply the parameterized ansatz circuit to the initial state `|00...0>`. The ansatz for *N* qubits with *L* layers consists of:
   - A layer of RY(theta_i) gates on each qubit for initial rotation.
   - For each layer *l*:
     - RY(theta) and RZ(theta) rotations on each qubit.
     - A ladder of CNOT gates connecting adjacent qubits.

2. **Measurement**: Measure the expectation value of the Hamiltonian. Since the Hamiltonian is a sum of Pauli terms, each term can be measured independently, and the results are combined classically.

### 3.2 The Classical Component

The classical optimizer receives the energy expectation value and updates the circuit parameters. Common choices include:

- **COBYLA** (Constrained Optimization BY Linear Approximation): A gradient-free method well-suited to noisy objective functions.
- **Nelder-Mead**: Another gradient-free simplex method.
- **SPSA** (Simultaneous Perturbation Stochastic Approximation): Designed for noisy optimization with only two function evaluations per iteration.

In our implementation, we use a simple yet effective approach: **coordinate descent with random restarts**. For each parameter, we evaluate the cost function at several candidate values and select the best. This is repeated across multiple random initializations to escape local minima.

### 3.3 Convergence

The VQE loop terminates when one of the following conditions is met:
- The energy change between iterations falls below a threshold.
- A maximum number of iterations is reached.
- The optimizer reports convergence.

In practice, the energy landscape for portfolio optimization QUBOs is relatively smooth (compared to, say, molecular chemistry), and convergence is typically achieved within a few hundred iterations.

## 4. Trading Application

### 4.1 Mapping Portfolio Optimization to VQE

Consider a crypto portfolio with *N* candidate assets. The workflow is:

1. **Data Collection**: Fetch historical OHLCV data from an exchange (Bybit in our case).
2. **Return Estimation**: Compute log returns and estimate expected returns (mean) and covariances.
3. **QUBO Construction**: Build the QUBO matrix from the covariance matrix and expected returns, with appropriate penalty weights.
4. **VQE Execution**: Run the VQE algorithm to find the optimal binary assignment.
5. **Portfolio Construction**: Convert the binary solution to portfolio weights.

### 4.2 Asset Selection as Combinatorial Optimization

When the number of assets is small (3-8), the problem is tractable for classical brute-force search, making it an ideal testbed for verifying quantum results. As the universe grows to 20-50+ assets, the exponential scaling of the search space (`2^N` possible selections) makes classical exact methods infeasible, and quantum approaches become increasingly relevant.

### 4.3 Practical Considerations

- **Number of qubits**: Each asset requires one qubit in the simplest formulation. More sophisticated encodings (e.g., multi-bit weight discretization) require multiple qubits per asset.
- **Circuit depth**: Deeper ansatze (more layers) can express more complex states but are harder to optimize and more susceptible to noise.
- **Penalty tuning**: The Lagrange multipliers (`lambda_risk`, `lambda_return`, `lambda_budget`) must be carefully tuned. If the budget penalty is too low, the optimizer may violate the constraint; too high, and it dominates the objective.

## 5. Implementation Walkthrough

Our Rust implementation consists of several modules:

### 5.1 Statevector Simulator

We simulate the quantum circuit classically using a **statevector simulator**. For *N* qubits, the state is a complex vector of dimension `2^N`. Gates are applied by constructing and multiplying the appropriate unitary matrices.

Key gate implementations:
- **RY(theta)**: Rotation around the Y-axis. Matrix: `[[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]`.
- **RZ(theta)**: Rotation around the Z-axis. Matrix: `[[e^(-it/2), 0], [0, e^(it/2)]]`.
- **CNOT**: Controlled-NOT gate. Flips the target qubit if the control qubit is `|1>`.

### 5.2 QUBO Matrix Construction

The `build_qubo_matrix` function takes the covariance matrix, expected returns, and penalty weights as inputs and returns the QUBO matrix. The diagonal terms encode the linear coefficients, and the off-diagonal terms encode the quadratic interactions.

### 5.3 Cost Function Evaluation

Given a statevector, we compute the expectation value of the QUBO Hamiltonian. For a diagonal Hamiltonian (which our QUBO Hamiltonian is after the Ising encoding), this reduces to:

```
<H> = sum_x |alpha_x|^2 * f(x)
```

where `alpha_x` is the amplitude of basis state `|x>` and `f(x)` is the QUBO objective value for binary string *x*.

### 5.4 Classical Optimizer

Our optimizer uses **coordinate descent**: for each parameter in turn, evaluate the cost function at several candidate values (current value +/- small perturbations) and keep the best. This is wrapped in multiple random restarts to improve the chance of finding the global minimum.

### 5.5 Bybit Data Integration

The `fetch_bybit_klines` function queries the Bybit V5 API for historical candlestick data. We fetch daily candles for multiple trading pairs, compute log returns, and estimate the covariance matrix and mean returns.

## 6. Bybit Data Integration

### 6.1 API Endpoint

We use the Bybit V5 market API:

```
GET https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval=D&limit=100
```

This returns up to 100 daily candles, which is sufficient for estimating short-term correlations.

### 6.2 Computing Returns and Covariances

From the closing prices `P_t`, we compute log returns:

```
r_t = ln(P_t / P_{t-1})
```

The covariance matrix is estimated using the standard sample covariance estimator:

```
Sigma_ij = (1 / (T-1)) * sum_t (r_it - mu_i)(r_jt - mu_j)
```

### 6.3 Correlation Structure in Crypto Markets

Cryptocurrency markets exhibit several distinctive correlation patterns:

- **High intra-market correlation**: Most major cryptos are highly correlated with BTC, especially during market-wide selloffs.
- **Regime-dependent correlations**: Correlations increase during bear markets (correlation breakdown).
- **Altcoin diversification**: Some altcoins with distinct use cases (e.g., DeFi vs. infrastructure) can provide partial diversification.

These patterns make portfolio optimization particularly valuable in crypto: naive equal-weight portfolios carry hidden concentration risk due to high correlations.

## 7. Key Takeaways

1. **VQE is practical for NISQ devices**: Unlike many quantum algorithms, VQE uses shallow circuits that can run on near-term quantum hardware. The classical optimizer compensates for limited circuit expressivity.

2. **Portfolio optimization maps naturally to QUBO**: The Markowitz framework, with its quadratic objective and linear constraints, translates directly to a QUBO problem suitable for quantum optimization.

3. **Hybrid algorithms leverage both paradigms**: The quantum computer explores the exponentially large Hilbert space, while the classical optimizer navigates the parameter landscape efficiently.

4. **Classical simulation is essential for validation**: For small portfolios (< 20 assets), classical statevector simulation lets us verify quantum results exactly. Our Rust implementation demonstrates this approach.

5. **Penalty tuning is critical**: The balance between risk minimization, return maximization, and budget constraint enforcement determines solution quality. Cross-validation or Bayesian optimization can automate this tuning.

6. **Real market data matters**: Synthetic examples with uncorrelated assets miss the challenge of real-world correlation structures. Integrating Bybit data reveals the high correlation regime typical of crypto markets.

7. **Scalability path is clear**: While our 3-asset example is classically trivial, the same VQE framework scales to larger universes where quantum advantage emerges. The key bottleneck is qubit count and circuit noise, both of which are improving rapidly.

8. **Rust provides performance**: The statevector simulation, which involves `2^N`-dimensional complex vector operations, benefits enormously from Rust's zero-cost abstractions and memory safety. For production use, this simulator could be parallelized across multiple cores or replaced with a GPU backend.
