# Chapter 199: Quantum Annealing Trading

## 1. Introduction

Quantum annealing is a metaheuristic optimization technique that exploits quantum mechanical phenomena to solve combinatorial optimization problems. In the context of algorithmic trading, many critical decisions -- portfolio construction, index tracking, order routing, and trade lot sizing -- reduce to NP-hard combinatorial problems that classical solvers struggle with as the problem size grows.

D-Wave Systems pioneered commercial quantum annealing hardware, offering processors with thousands of qubits specifically designed for optimization. Unlike universal gate-based quantum computers that execute arbitrary quantum circuits, D-Wave's quantum annealers are purpose-built to find low-energy states of Ising spin systems. This specialization makes them natural candidates for financial optimization problems that can be mapped onto quadratic unconstrained binary optimization (QUBO) formulations.

The central idea is straightforward: encode a trading optimization problem into a QUBO matrix, submit it to a quantum annealer (or a classical simulator), and interpret the resulting binary solution as a trading decision. The quantum advantage, when it exists, comes from the ability of quantum tunneling to escape local minima that trap classical optimizers.

This chapter covers the mathematical foundations of quantum annealing, demonstrates how to formulate common trading problems as QUBO instances, compares quantum annealing with classical simulated annealing, and provides a complete Rust implementation that fetches live data from Bybit and solves portfolio optimization problems.

## 2. Mathematical Foundation

### 2.1 The Ising Model

The Ising model originates from statistical mechanics and describes a system of interacting binary spins. Each spin variable $s_i \in \{-1, +1\}$ represents a binary degree of freedom. The energy (Hamiltonian) of the system is:

$$H_{Ising} = -\sum_{i<j} J_{ij} s_i s_j - \sum_i h_i s_i$$

where $J_{ij}$ are coupling strengths between spins $i$ and $j$, and $h_i$ are local magnetic fields acting on individual spins. The optimization task is to find the spin configuration $\{s_i\}$ that minimizes $H_{Ising}$.

### 2.2 QUBO Formulation

Quadratic Unconstrained Binary Optimization uses binary variables $x_i \in \{0, 1\}$ instead of spin variables. The objective function is:

$$\min_x \; x^T Q x = \sum_{i} Q_{ii} x_i + \sum_{i<j} Q_{ij} x_i x_j$$

where $Q$ is an upper-triangular matrix encoding linear terms on the diagonal and quadratic interactions on the off-diagonal entries. The Ising and QUBO formulations are mathematically equivalent via the substitution $x_i = (s_i + 1) / 2$.

For trading problems, the art lies in constructing the $Q$ matrix so that low-energy binary solutions correspond to profitable or risk-minimized trading decisions.

### 2.3 Annealing Schedule

Quantum annealing operates by slowly evolving a quantum system from an easy-to-prepare initial state to the ground state of a problem Hamiltonian. The time-dependent Hamiltonian is:

$$H(t) = A(t) H_{driver} + B(t) H_{problem}$$

where $A(t)$ decreases from a large value to zero and $B(t)$ increases from zero to a large value over the annealing time $T$. Common schedules include:

- **Linear schedule**: $A(t) = 1 - t/T$, $B(t) = t/T$
- **Exponential schedule**: $A(t) = e^{-\alpha t/T}$, $B(t) = 1 - e^{-\alpha t/T}$

The exponential schedule often performs better in practice because it spends more time in the critical region where the energy gap between ground and first excited states is smallest.

### 2.4 Transverse Field and Tunneling

The driver Hamiltonian in quantum annealing is typically a transverse field:

$$H_{driver} = -\sum_i \sigma_i^x$$

where $\sigma_i^x$ is the Pauli-X operator. This transverse field induces quantum superposition among spin states, enabling **quantum tunneling** through energy barriers. Unlike classical thermal fluctuations (used in simulated annealing) that must go over barriers, quantum tunneling allows the system to pass through thin, tall barriers -- a potential advantage for rugged energy landscapes common in financial optimization.

### 2.5 The Adiabatic Theorem

The adiabatic theorem guarantees that if the Hamiltonian changes sufficiently slowly, the system remains in its instantaneous ground state. Specifically, the annealing time $T$ must satisfy:

$$T \gg \frac{\max_t |\langle 1(t) | \dot{H}(t) | 0(t) \rangle|}{(\Delta_{min})^2}$$

where $\Delta_{min}$ is the minimum energy gap between the ground state $|0(t)\rangle$ and first excited state $|1(t)\rangle$ during the anneal. For practical problems, this gap can be exponentially small, which limits the theoretical guarantee. Nevertheless, heuristic quantum annealing with finite annealing times often finds good solutions even without rigorous adiabatic guarantees.

## 3. Trading Applications

### 3.1 Portfolio Optimization with Cardinality Constraints

The classical Markowitz mean-variance portfolio problem becomes NP-hard when cardinality constraints are added (e.g., "select exactly $K$ assets from a universe of $N$"). The QUBO formulation is:

$$\min_x \; \lambda \sum_{i,j} \Sigma_{ij} x_i x_j - \sum_i \mu_i x_i + P \left(\sum_i x_i - K\right)^2$$

where $x_i \in \{0,1\}$ indicates whether asset $i$ is selected, $\Sigma$ is the covariance matrix, $\mu$ is the expected return vector, $\lambda$ is the risk aversion parameter, and $P$ is a penalty coefficient enforcing the cardinality constraint. The penalty term expands to quadratic and linear terms that integrate naturally into the QUBO matrix.

### 3.2 Index Tracking

Index tracking seeks a sparse subset of assets that replicates an index's returns. The QUBO formulation minimizes tracking error:

$$\min_x \; \sum_t \left(r_t^{index} - \sum_i x_i w_i r_{i,t}\right)^2$$

subject to cardinality constraints. When weights $w_i$ are pre-determined (e.g., equal weight among selected assets), this reduces to a pure binary selection problem that maps directly to QUBO.

### 3.3 Trade Lot Optimization

For large orders that must be broken into discrete lots, quantum annealing can optimize the allocation across venues and time slots. Binary variables encode whether a specific lot is assigned to a specific venue at a specific time, with the objective function penalizing market impact and transaction costs.

### 3.4 Order Routing

Multi-venue order routing with discrete routing decisions can be expressed as a QUBO. The objective minimizes expected execution cost (spread + market impact + fees) subject to the constraint that the total routed quantity matches the order size. Each binary variable $x_{iv}$ indicates whether lot $i$ is routed to venue $v$.

## 4. D-Wave vs Gate-Based Quantum Computing

| Aspect | Quantum Annealing (D-Wave) | Gate-Based (IBM, Google) |
|--------|---------------------------|--------------------------|
| **Problem type** | Optimization (QUBO/Ising) | General quantum algorithms |
| **Qubit count** | ~5000+ (Advantage) | ~1000+ (noisy) |
| **Connectivity** | Pegasus graph topology | All-to-all (with overhead) |
| **Error correction** | None (analog process) | Partial / in development |
| **Programming model** | Define $Q$ matrix | Design quantum circuits |
| **Best for trading** | Combinatorial selection problems | Quantum Monte Carlo, amplitude estimation |
| **Availability** | Cloud API (Leap) | Cloud API (Qiskit, Cirq) |

**When to use annealing**: Problems that naturally decompose into binary selection with quadratic interactions -- portfolio selection, index tracking, binary allocation. The QUBO formulation must be sparse enough to embed on the hardware graph.

**When to use gate-based**: Problems requiring quantum speedup on continuous optimization, simulation of quantum systems for derivatives pricing, or amplitude estimation for risk analysis.

For most practical trading applications today, quantum annealing is more immediately applicable because trading decisions often involve discrete choices that map naturally to binary variables.

## 5. Simulated vs Quantum Annealing

Simulated annealing (SA) is a well-established classical metaheuristic that serves as the primary benchmark for quantum annealing. Understanding the differences is critical for evaluating quantum advantage claims.

### 5.1 Simulated Annealing (Classical)

SA uses thermal fluctuations controlled by a temperature parameter $T$ that decreases over time. At each step, a random spin flip is proposed. The flip is accepted with probability:

$$P(\text{accept}) = \min\left(1, \; e^{-\Delta E / T}\right)$$

where $\Delta E$ is the energy change. At high temperature, most flips are accepted (exploration); at low temperature, only improving flips are accepted (exploitation).

### 5.2 Simulated Quantum Annealing (SQA)

SQA approximates quantum annealing on classical hardware using the Suzuki-Trotter decomposition. It simulates $P$ replicas (Trotter slices) of the system coupled along an imaginary time dimension. The transverse field strength $\Gamma(t)$ decreases during the anneal, mimicking the quantum tunneling effect.

The effective Hamiltonian for SQA is:

$$H_{SQA} = \sum_{k=1}^P H_{problem}(\{s_i^k\}) - J_\perp \sum_{k=1}^P \sum_i s_i^k s_i^{k+1}$$

where $J_\perp = -\frac{T}{2} \ln \tanh\left(\frac{\Gamma}{PT}\right)$ is the inter-replica coupling.

### 5.3 Comparative Advantages

- **SA strengths**: Simple to implement, well-understood convergence, no hardware limitations.
- **SQA strengths**: Can tunnel through thin barriers more efficiently than SA, often finds better solutions on problems with rugged energy landscapes.
- **Hardware QA strengths**: Massive parallelism from physical quantum superposition, constant-time per anneal regardless of problem structure (within hardware limits).

In practice, for small-to-medium trading problems (N < 100 assets), classical SA and SQA are competitive with hardware QA. The potential quantum advantage emerges for larger problems with complex constraint structures.

## 6. Implementation Walkthrough

Our Rust implementation provides three core components:

### 6.1 QUBO Representation

The `QuboProblem` struct stores the upper-triangular $Q$ matrix using `ndarray`. Linear terms occupy the diagonal, and quadratic terms occupy the upper triangle. The `energy()` method computes $x^T Q x$ for a given binary solution.

```rust
pub struct QuboProblem {
    pub q_matrix: Array2<f64>,
    pub num_variables: usize,
}
```

### 6.2 Solvers

Two solvers are implemented:

- `SimulatedAnnealingSolver`: Classical SA with configurable temperature schedule (linear or exponential decay), number of sweeps, and initial temperature.
- `SimulatedQuantumAnnealingSolver`: SQA with Trotter slices, transverse field schedule, and inter-replica coupling. Each Trotter slice evolves independently under the problem Hamiltonian while coupled to neighboring slices.

Both solvers return a `SolverResult` containing the best solution, its energy, and the energy history for convergence analysis.

### 6.3 Portfolio QUBO Encoding

The `PortfolioQuboBuilder` takes a covariance matrix, expected returns, and parameters (risk aversion $\lambda$, target cardinality $K$, penalty strength $P$) and constructs the QUBO matrix:

```rust
pub fn build_qubo(&self) -> QuboProblem {
    // Diagonal: -mu_i + lambda * Sigma_ii + P * (1 - 2K)
    // Off-diagonal: lambda * Sigma_ij + 2P
    // Constant offset (ignored in optimization): P * K^2
}
```

### 6.4 Energy Landscape Analysis

The `EnergyLandscapeAnalyzer` provides tools for understanding the optimization landscape: computing the energy distribution over random samples, estimating the fraction of local minima, and measuring barrier heights between solutions.

## 7. Bybit Data Integration

The implementation fetches live market data from the Bybit API v5 to construct realistic portfolio optimization problems.

### 7.1 Data Pipeline

1. **Fetch tickers**: Query `/v5/market/tickers` for spot market data to get latest prices and 24h volume.
2. **Fetch klines**: Query `/v5/market/kline` for historical OHLCV data at daily intervals.
3. **Compute returns**: Calculate log returns from closing prices.
4. **Build covariance matrix**: Estimate the sample covariance matrix from historical returns.
5. **Compute expected returns**: Use mean historical returns as a simple estimator.

### 7.2 API Integration

```rust
pub async fn fetch_klines(symbol: &str, interval: &str, limit: u32)
    -> Result<Vec<KlineData>>
{
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    // Parse response and return structured data
}
```

The example fetches data for multiple crypto assets (BTC, ETH, SOL, etc.), builds the covariance matrix, and formulates a portfolio selection QUBO targeting a specific number of assets.

## 8. Key Takeaways

1. **Natural fit**: Many trading optimization problems (portfolio selection, index tracking, order routing) are combinatorial and map naturally to QUBO formulations that quantum annealers can solve.

2. **QUBO encoding is the bottleneck**: The hardest part is not running the annealer -- it is constructing a QUBO matrix that faithfully represents the trading problem with appropriate penalty terms for constraints.

3. **Penalty tuning matters**: The penalty coefficient $P$ for constraint violations must be large enough to enforce constraints but not so large that it dominates the objective and creates a flat energy landscape. A common heuristic is $P \approx \max(|Q_{ij}|)$.

4. **Classical baselines are strong**: For problems with fewer than ~100 binary variables, well-tuned simulated annealing often matches or exceeds quantum annealing performance. Always benchmark against classical solvers.

5. **Quantum tunneling advantage**: SQA and hardware QA show potential advantages on problems with tall, thin energy barriers -- which can arise in highly constrained portfolio problems with many near-degenerate solutions.

6. **Hardware limitations**: Current quantum annealers have limited connectivity (Pegasus graph), requiring minor embedding that increases the effective number of qubits needed. Dense QUBO matrices (like full covariance matrices) require significant embedding overhead.

7. **Hybrid approaches**: The most practical near-term approach combines quantum annealing for the combinatorial core (asset selection) with classical optimization for continuous parameters (portfolio weights).

8. **Scalability potential**: As quantum annealing hardware improves (more qubits, better connectivity, lower noise), the approach becomes increasingly viable for large-scale trading problems that overwhelm classical solvers.

9. **Simulated quantum annealing**: Even without quantum hardware, SQA on classical computers provides a useful middle ground -- it captures some tunneling benefits and often outperforms standard SA on rugged landscapes.

10. **Practical pipeline**: The complete workflow (data ingestion, QUBO construction, solving, interpretation) can be implemented efficiently in Rust, with the solver as a drop-in component that can be swapped between classical SA, SQA, and hardware QA via cloud APIs.
