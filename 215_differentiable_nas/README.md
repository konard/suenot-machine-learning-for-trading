# Chapter 215: Differentiable NAS (DARTS)

## 1. Introduction - Differentiable NAS (DARTS): Making Architecture Search Continuous and Differentiable

Neural Architecture Search (NAS) has emerged as a powerful paradigm for automating the design of neural network architectures. Traditional NAS methods, such as reinforcement-learning-based or evolutionary approaches, require training thousands of candidate architectures from scratch, consuming enormous computational budgets. For trading applications, where rapid iteration on model design is critical due to shifting market regimes, this expense is prohibitive.

Differentiable Architecture Search (DARTS), introduced by Liu, Simonyan, and Yang in 2019, fundamentally changed the NAS landscape by reformulating the discrete architecture selection problem as a continuous optimization problem. Instead of searching over a discrete set of architectures one at a time, DARTS relaxes the search space to be continuous, enabling the use of gradient descent to jointly optimize both the architecture and its weights. This reduces the search cost from thousands of GPU-days to a single GPU-day, making architecture search practical for domains like algorithmic trading where model freshness and adaptability matter.

The core insight of DARTS is elegant: rather than choosing one operation per edge in a computation graph, assign a continuous weight to every candidate operation and optimize those weights using standard backpropagation. At the end of the search, discretize by keeping the highest-weighted operation on each edge. This chapter walks through the mathematical foundations of DARTS, its algorithm, trading-specific applications, notable variants, and a complete Rust implementation integrated with live Bybit market data.

## 2. Mathematical Foundation

### 2.1 Continuous Relaxation of Discrete Architecture Choices

Consider a directed acyclic graph (DAG) where each node represents a latent representation and each edge represents a candidate operation (e.g., convolution, skip connection, dense layer). In the discrete formulation, each edge selects exactly one operation from a candidate set O = {o_1, o_2, ..., o_K}. The total number of possible architectures grows combinatorially with the number of edges and operations, making exhaustive search intractable.

DARTS replaces the discrete choice with a continuous relaxation. For each edge (i, j) in the DAG, instead of selecting a single operation, DARTS computes a weighted mixture of all candidate operations:

```
f_{i,j}(x) = sum_{k=1}^{K} p_k * o_k(x)
```

where the mixing weights p_k are derived from learnable architecture parameters alpha via a softmax:

```
p_k = exp(alpha_k) / sum_{m=1}^{K} exp(alpha_m)
```

### 2.2 Architecture Parameters alpha

The architecture parameters alpha = {alpha_{i,j,k}} form a separate set of learnable parameters distinct from the network weights w. Each alpha_{i,j,k} controls the contribution of operation k on edge (i, j). Initially, all alpha values are set equally (or randomly), meaning all operations contribute equally. During search, gradient descent on alpha pushes the distribution toward favoring certain operations.

The softmax ensures that the mixing weights form a valid probability distribution, and the temperature of the softmax can be adjusted to control the sharpness of the distribution. As training progresses, the distribution typically sharpens, with one operation dominating per edge.

### 2.3 Bi-Level Optimization

DARTS frames architecture search as a bi-level optimization problem:

```
min_{alpha}  L_val(w*(alpha), alpha)
subject to   w*(alpha) = argmin_{w} L_train(w, alpha)
```

The outer objective minimizes the validation loss with respect to the architecture parameters alpha, while the inner objective trains the network weights w on the training loss. This formulation ensures that the architecture is optimized for generalization (validation performance) rather than merely fitting the training data.

In practice, the bi-level optimization is approximated with alternating gradient descent steps:

1. **Step 1 (Weight update):** Take one gradient step on w using the training loss:
   ```
   w' = w - eta_w * grad_w L_train(w, alpha)
   ```

2. **Step 2 (Architecture update):** Take one gradient step on alpha using the validation loss:
   ```
   alpha' = alpha - eta_alpha * grad_alpha L_val(w', alpha)
   ```

This alternation is efficient because it avoids fully training the network at each architecture update step. The first-order approximation (using w' instead of fully converged w*) introduces some bias but works well in practice.

### 2.4 Second-Order Approximation

For more accurate architecture gradients, DARTS supports a second-order approximation that accounts for the dependency of w* on alpha:

```
grad_alpha L_val(w*(alpha), alpha) ≈ grad_alpha L_val(w', alpha) - eta_w * grad^2_{alpha,w} L_train(w, alpha) * grad_w L_val(w', alpha)
```

The second term involves a Hessian-vector product, which can be computed efficiently using finite differences. This provides a better estimate of the true architecture gradient but at additional computational cost.

## 3. DARTS Algorithm

### 3.1 Mixed Operation

The mixed operation is the atomic building block of the DARTS search space. Each mixed operation maintains:
- A set of K candidate operations {o_1, ..., o_K}
- Architecture parameters alpha = (alpha_1, ..., alpha_K)
- Softmax weights p = softmax(alpha)

During the forward pass, all operations are applied to the input and their outputs are combined using the softmax weights. This is what makes the architecture differentiable: the gradient flows through all operations simultaneously, weighted by their architecture parameters.

### 3.2 Architecture Gradient

The architecture gradient grad_alpha L_val tells us which operations should be strengthened (positive contribution to reducing validation loss) and which should be weakened. For a mixed operation with operations {o_k} and weights {p_k}:

```
dL/d(alpha_k) = sum_{m} (dL/dp_m) * (dp_m/d(alpha_k))
```

The Jacobian of softmax dp_m/d(alpha_k) = p_m * (delta_{mk} - p_k) ensures that increasing alpha_k increases the weight of operation k while decreasing the weights of other operations.

### 3.3 Discretization

After the search phase converges, the continuous architecture must be discretized into a concrete architecture for final evaluation. The standard discretization strategy is:

1. For each edge (i, j), select the operation with the highest architecture weight:
   ```
   o*_{i,j} = argmax_k alpha_{i,j,k}
   ```
2. For each node, retain the top-k incoming edges (typically k=2) based on the maximum edge weight.
3. Remove zero operations and prune unused edges.

### 3.4 Search and Evaluation Phases

The DARTS workflow consists of two distinct phases:

**Search Phase:**
- Train a small proxy network with mixed operations
- Alternate between weight and architecture parameter updates
- Typically runs for 50-100 epochs on a reduced dataset
- Output: optimized architecture parameters alpha

**Evaluation Phase:**
- Discretize the architecture from alpha
- Build a full-sized network with the discovered architecture
- Train from scratch on the complete dataset
- Evaluate on held-out test data

## 4. Trading Applications

### 4.1 Efficiently Finding Optimal Architectures for Time-Series Prediction

Financial time-series prediction presents unique architectural challenges. Unlike image classification where convolutional architectures dominate, the optimal architecture for trading signals is not obvious. Different market regimes (trending, mean-reverting, volatile) may benefit from different computational primitives.

DARTS enables efficient exploration of this design space. A trading-specific search space might include:
- **Identity/Skip connections:** For markets where recent raw features are most predictive
- **Dense layers:** For learning complex nonlinear feature interactions
- **1D convolution-like operations:** For detecting local temporal patterns (e.g., candlestick formations)
- **Exponential moving average operations:** For capturing trend information at various timescales
- **Difference operations:** For learning from returns rather than prices
- **Attention-like operations:** For selectively focusing on relevant time steps

By running DARTS on historical market data, we can discover which combination of these primitives works best for a specific asset or trading strategy. The search can be re-run periodically as market conditions change, automatically adapting the model architecture.

### 4.2 Operation Selection for Financial Feature Processing

In feature engineering for trading, practitioners often debate which transformations to apply to raw market data. DARTS can automate this selection:

- **Input features:** OHLCV data, order book snapshots, funding rates
- **Candidate operations per feature:** normalization, log-transform, differencing, windowed statistics, skip (pass-through)
- **DARTS selects:** The optimal processing pipeline for each feature independently

This approach has several advantages over manual feature engineering:
1. It removes human bias in feature selection
2. It can discover non-obvious feature transformations
3. It adapts to new market regimes when re-run
4. The search is computationally cheap (hours, not days)

### 4.3 Multi-Asset Architecture Search

For portfolio-level trading systems, DARTS can search for architectures that process multiple asset streams jointly. The search space includes operations for cross-asset interaction (correlation layers, attention across assets) alongside single-asset processing operations. This enables discovering architectures that capture lead-lag relationships and contagion effects across assets.

## 5. DARTS Variants

### 5.1 PC-DARTS (Partially Connected DARTS)

PC-DARTS addresses the memory inefficiency of standard DARTS, which requires storing all candidate operations in memory simultaneously. Key innovations:
- **Partial channel connections:** Only sample a subset of channels for the mixed operation during search, reducing memory by 1/K
- **Edge normalization:** Add explicit normalization of edge weights to stabilize search
- PC-DARTS can search on larger proxy networks, leading to better architectures

For trading applications, PC-DARTS enables searching with larger feature sets and longer lookback windows that would otherwise exceed memory limits.

### 5.2 FairDARTS

Standard DARTS suffers from an unfair advantage for parameter-free operations (skip connections, zero) because they are easier to optimize. FairDARTS addresses this with:
- **Sigmoid relaxation:** Replace softmax with independent sigmoid activations per operation, removing the competitive exclusion effect
- **Zero operation handling:** Treat the zero operation separately with a threshold-based pruning mechanism
- This produces more balanced architectures with richer computational graphs

In trading, this matters because skip connections (which correspond to "use raw features directly") would otherwise dominate, potentially missing important nonlinear transformations.

### 5.3 DARTS+ (Improved Robustness)

DARTS+ introduces early stopping of the architecture search based on the eigenvalues of the Hessian of the validation loss. When the dominant eigenvalue becomes too large, the search has become unstable and should stop. Additional improvements include:
- **Auxiliary skip connections:** Added to every intermediate node to reduce the unfair advantage of skip connections in the search
- **Search space regularization:** L2 regularization on architecture parameters to prevent premature convergence

### 5.4 Progressive DARTS (P-DARTS)

P-DARTS gradually increases the depth of the search network during the search process:
- Start with a shallow network (fewer cells)
- Progressively increase depth while pruning operations
- This bridges the depth gap between search and evaluation networks
- Particularly relevant for trading where the optimal network depth is unknown

## 6. Implementation Walkthrough with Rust Code

Our Rust implementation provides a complete DARTS system suitable for trading applications. The implementation is organized around several key components:

### 6.1 Operation Primitives

Each candidate operation implements a common interface that takes an input vector and produces an output vector. Our search space includes:
- **Identity:** Pass input unchanged (skip connection)
- **Zero:** Output zeros (equivalent to dropping this edge)
- **Dense:** Fully connected linear transformation with learnable weights
- **Conv1D-like:** Sliding window convolution for temporal pattern detection
- **Skip Connection:** Residual connection for gradient flow

```rust
// Each operation transforms input -> output
// See rust/src/lib.rs for full implementation
```

### 6.2 Mixed Operations and Softmax Mixing

The MixedOperation struct holds architecture parameters alpha for each candidate operation and computes the softmax-weighted mixture during forward passes. The softmax temperature can be annealed during search to gradually sharpen the architecture distribution.

### 6.3 Bi-Level Optimization Loop

The search procedure alternates between:
1. Sampling a training batch and updating network weights w
2. Sampling a validation batch and updating architecture parameters alpha

Both updates use simple gradient descent with configurable learning rates. The architecture gradient is computed using the first-order approximation for efficiency.

### 6.4 Architecture Discretization

After search completes, the `discretize()` method extracts the final architecture by selecting the highest-weighted operation per edge. The resulting discrete architecture can be evaluated independently.

## 7. Bybit Data Integration

The implementation includes a complete Bybit API integration for fetching real market data:

```rust
// Fetches OHLCV kline data from Bybit's v5 API
// Converts to feature vectors for DARTS search
// Supports multiple timeframes and symbols
```

The data pipeline:
1. **Fetch:** Query Bybit's `/v5/market/kline` endpoint for BTCUSDT candles
2. **Parse:** Extract open, high, low, close, volume from JSON response
3. **Normalize:** Apply z-score normalization to each feature
4. **Window:** Create sliding window samples for time-series prediction
5. **Split:** Divide into training and validation sets for bi-level optimization

## 8. Key Takeaways

1. **DARTS makes NAS practical** by replacing discrete search with continuous optimization. Architecture search that previously required thousands of GPU-days now completes in hours, making it feasible for trading applications where models must be frequently updated.

2. **Continuous relaxation is the key insight.** By replacing hard operation selection with softmax-weighted mixtures, DARTS enables gradient-based optimization of architecture choices alongside network weights.

3. **Bi-level optimization ensures generalization.** Optimizing architecture parameters on validation data while training weights on training data helps discover architectures that generalize rather than overfit.

4. **Trading benefits from automated architecture design.** Financial markets exhibit regime changes that may require different network architectures. DARTS enables rapid, automated architecture adaptation without manual intervention.

5. **Operation search spaces should be domain-specific.** For trading, include operations that capture relevant financial dynamics: temporal convolutions for pattern detection, skip connections for raw feature access, and dense layers for nonlinear feature interaction.

6. **Variants address practical limitations.** PC-DARTS reduces memory requirements, FairDARTS removes bias toward parameter-free operations, and DARTS+ improves search stability. Choose the variant that matches your computational constraints and data characteristics.

7. **Discretization is a critical step.** The gap between the continuous relaxed architecture and the discretized final architecture can affect performance. Techniques like progressive search (P-DARTS) and careful edge selection help bridge this gap.

8. **Rust implementation provides performance.** By implementing DARTS in Rust, we achieve the memory safety and performance characteristics needed for production trading systems, while maintaining the ability to integrate with live market data feeds from exchanges like Bybit.
