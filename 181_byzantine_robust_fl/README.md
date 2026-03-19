# Chapter 181: Byzantine-Robust Federated Learning for Trading

## 1. Introduction: Byzantine Fault Tolerance in Distributed Trading Systems

Federated learning (FL) has emerged as a compelling paradigm for collaborative model training in financial markets. Multiple trading firms, hedge funds, or even individual algorithmic traders can jointly train a predictive model without sharing their proprietary data. Each participant trains a local model on their private dataset and shares only gradient updates or model parameters with a central aggregation server. The server combines these updates to produce a globally improved model.

However, this collaborative setup introduces a critical vulnerability: **Byzantine faults**. In distributed systems theory, a Byzantine fault refers to any arbitrary or malicious behavior by a participant. Unlike simple crash faults where a node merely stops responding, Byzantine faults encompass the full spectrum of adversarial behavior -- a participant may send corrupted gradients, deliberately misleading model updates, or even coordinate with other malicious participants to manipulate the global model.

In trading, the stakes are particularly high. Consider a federated learning system where ten quantitative trading firms collaboratively train a price prediction model. If two of these firms are adversarial -- perhaps seeking to manipulate the model to create predictable mispricings they can exploit -- the resulting model could generate systematically biased predictions that benefit the attackers at the expense of honest participants.

The classical Federated Averaging (FedAvg) algorithm, which simply averages all client updates, is catastrophically vulnerable to Byzantine attacks. A single malicious client sending arbitrarily large gradient values can completely dominate the aggregated update. This chapter explores robust aggregation mechanisms that provide provable resilience against Byzantine participants, specifically in the context of trading model training.

The fundamental question we address is: **How can we aggregate model updates from multiple participants such that the result remains close to what honest participants would produce, even when a fraction of participants are adversarial?**

## 2. Mathematical Foundations

### 2.1 Problem Formulation

Consider $n$ clients participating in federated learning, of which at most $f < n/2$ are Byzantine. In each round $t$, each client $i$ computes a gradient update $g_i^{(t)} \in \mathbb{R}^d$ based on their local data. Honest clients compute genuine stochastic gradients, while Byzantine clients may submit arbitrary vectors.

The goal is to design an aggregation function $\text{Agg}(g_1, g_2, \ldots, g_n)$ that approximates the true mean of the honest gradients, regardless of the values submitted by Byzantine clients.

### 2.2 Krum Algorithm

The **Krum** algorithm (Blanchard et al., 2017) selects the single gradient update that is most "central" among all submissions. The intuition is that honest gradients should cluster together, while Byzantine gradients are likely to be outliers.

For each gradient $g_i$, compute the sum of squared distances to its $n - f - 2$ nearest neighbors:

$$s(i) = \sum_{j \in \mathcal{N}_i} \|g_i - g_j\|^2$$

where $\mathcal{N}_i$ denotes the set of $n - f - 2$ closest gradients to $g_i$ (excluding $g_i$ itself).

**Krum** selects the gradient with the minimum score:

$$\text{Krum}(g_1, \ldots, g_n) = g_{i^*}, \quad \text{where } i^* = \arg\min_i s(i)$$

**Multi-Krum** extends this by selecting the $m$ gradients with the smallest scores and averaging them:

$$\text{Multi-Krum}(g_1, \ldots, g_n) = \frac{1}{m} \sum_{j=1}^{m} g_{i_j^*}$$

where $i_1^*, \ldots, i_m^*$ are the indices of the $m$ smallest scores.

**Theoretical guarantee:** If the number of Byzantine workers $f < (n - 2) / 2$, then Krum converges to a stationary point of the loss function of honest workers.

### 2.3 Trimmed Mean

The **coordinate-wise trimmed mean** operates independently on each dimension of the gradient vector. For each coordinate $j \in \{1, \ldots, d\}$:

1. Sort the values $g_{1,j}, g_{2,j}, \ldots, g_{n,j}$
2. Remove the $\beta$ largest and $\beta$ smallest values (where $\beta \geq f$)
3. Average the remaining $n - 2\beta$ values

$$\text{TrMean}_j = \frac{1}{n - 2\beta} \sum_{i \in \mathcal{S}_j} g_{i,j}$$

where $\mathcal{S}_j$ is the set of indices remaining after trimming dimension $j$.

This method is robust because no matter what values Byzantine clients submit for any coordinate, those extreme values will be trimmed away. The trimmed mean has a breakdown point of $\beta/n$, meaning it remains reliable as long as the fraction of Byzantine clients does not exceed $\beta/n$.

### 2.4 Coordinate-Wise Median

The simplest robust aggregation method takes the **median** of each coordinate independently:

$$\text{Med}_j = \text{median}(g_{1,j}, g_{2,j}, \ldots, g_{n,j})$$

The coordinate-wise median has a breakdown point of 50%, making it the most robust among these methods. However, it can introduce bias when the honest gradients are not symmetrically distributed.

**Convergence rate comparison:**
- FedAvg (no robustness): $O(1/\sqrt{T})$ convergence, breaks under any Byzantine attack
- Krum: $O(1/\sqrt{T})$ convergence with up to $f < (n-2)/2$ Byzantine workers
- Trimmed Mean: $O(1/\sqrt{T})$ convergence with up to $f < n/2$ Byzantine workers
- Median: $O(1/\sqrt{T})$ convergence with up to $f < n/2$ Byzantine workers, but with larger constants

## 3. Threat Model: Types of Byzantine Attacks

### 3.1 Gradient Poisoning (Additive Noise Attack)

The attacker adds a carefully crafted perturbation to the true gradient:

$$g_{\text{malicious}} = g_{\text{true}} + \delta$$

where $\delta$ is chosen to shift the model in a direction favorable to the attacker. In trading, this could mean biasing the model to predict upward moves for an asset the attacker holds long positions in.

**Scaling attack variant:** The attacker scales the true gradient by a large factor:

$$g_{\text{malicious}} = \lambda \cdot g_{\text{true}}, \quad \lambda \gg 1$$

This is particularly dangerous against FedAvg because a single scaled gradient can dominate the average.

### 3.2 Model Replacement Attack

Rather than perturbing gradients, the attacker computes a gradient that, when aggregated, replaces the global model with one of their choosing:

$$g_{\text{malicious}} = \frac{n}{\eta}(\theta_{\text{target}} - \theta_{\text{global}})$$

where $\theta_{\text{target}}$ is the model the attacker wants to install, $\theta_{\text{global}}$ is the current global model, and $\eta$ is the learning rate. When averaged with $n-1$ honest zero-ish updates, this effectively sets the new global model to $\theta_{\text{target}}$.

### 3.3 Label Flipping Attack

In the trading context, a Byzantine participant trains their local model on intentionally mislabeled data. For example:
- Labeling price increases as decreases and vice versa
- Shifting regression targets by a systematic bias
- Randomly permuting labels to inject noise

The resulting gradients are valid in structure but point in harmful directions. This attack is subtle because the gradients appear statistically normal in magnitude, making detection harder.

### 3.4 Collusion Attack

Multiple Byzantine clients coordinate to submit similar malicious gradients. If $f$ colluding attackers all submit the same malicious gradient, they create a "fake cluster" that can fool proximity-based defenses like Krum. This is the most challenging attack scenario and requires $f < (n-3)/4$ for Krum to remain provably secure against colluding adversaries.

## 4. Implementation Walkthrough

Our Rust implementation provides a complete Byzantine-robust federated learning framework with the following components:

### 4.1 Core Data Structures

We represent gradient updates as `ndarray` vectors. Each client is either `Honest` or `Byzantine`, and the system supports multiple aggregation strategies: `FedAvg`, `Krum`, `TrimmedMean`, and `CoordinateMedian`.

### 4.2 Aggregation Implementations

**Krum:** For each gradient, we compute distances to all other gradients, sort them, sum the closest $n - f - 2$ distances, and select the gradient with the minimum score. The time complexity is $O(n^2 d)$ where $d$ is the gradient dimension.

**Trimmed Mean:** We iterate over each coordinate, collect all values, sort them, trim the top and bottom $\beta$ values, and average the rest. Time complexity is $O(n d \log n)$ due to sorting.

**Coordinate-Wise Median:** For each coordinate, we sort all values and pick the middle element. We use the standard sort-based median which runs in $O(n d \log n)$.

### 4.3 Byzantine Client Simulation

Our framework simulates Byzantine behavior by:
1. Generating honest gradients as noisy versions of a "true" gradient (simulating stochastic gradient descent on different local datasets)
2. Generating malicious gradients using configurable attack strategies: random noise injection, gradient scaling, or sign-flipping
3. Running the aggregation and measuring the angular similarity between the aggregated result and the true gradient

### 4.4 Trading Model Integration

The implementation includes a `BybitClient` that fetches real OHLCV data for any trading pair. We compute simple features (returns, moving average ratios) and use them to generate realistic gradient distributions that reflect actual market data characteristics.

## 5. Bybit Data Integration

Our implementation fetches historical kline (candlestick) data from the Bybit V5 API:

```
GET https://api.bybit.com/v5/market/klines?category=linear&symbol=BTCUSDT&interval=60&limit=200
```

The data pipeline works as follows:

1. **Fetch OHLCV data** for the specified trading pair and interval
2. **Compute features** from raw price data:
   - Log returns: $r_t = \ln(p_t / p_{t-1})$
   - Normalized volume: $v_t / \bar{v}$
   - Price relative to moving average: $p_t / \text{MA}_k(p)$
   - Volatility estimate from high-low range: $(h_t - l_t) / p_t$
3. **Generate synthetic gradients** based on feature statistics, simulating what local SGD updates would look like if each client had a subset of the market data
4. **Apply Byzantine attacks** to a fraction of the gradients
5. **Aggregate** using each robust method and compare results

This approach demonstrates how real market microstructure translates into the gradient distributions that our robust aggregation methods must handle.

## 6. Key Takeaways

1. **FedAvg is fundamentally broken under adversarial conditions.** A single Byzantine participant can arbitrarily manipulate the aggregated model. Never use simple averaging in adversarial federated learning settings, especially in high-stakes domains like trading.

2. **Krum provides strong single-point selection but wastes information.** By selecting only one gradient per round, Krum discards useful information from honest participants. Multi-Krum offers a practical middle ground between robustness and efficiency.

3. **Trimmed mean offers the best practical trade-off.** It uses information from all non-trimmed gradients, handles high-dimensional settings well, and provides strong theoretical guarantees. It is the recommended default for trading applications.

4. **Coordinate-wise median is the most robust but introduces bias.** With a 50% breakdown point, it tolerates the most Byzantine clients, but the coordinate-wise operation can distort the gradient direction in high dimensions.

5. **The threat model matters enormously.** Gradient scaling attacks are easy to detect but devastating against naive aggregation. Collusion attacks are much harder to defend against and require stronger assumptions on the ratio of honest to Byzantine participants.

6. **Real market data has heavy-tailed gradient distributions.** Financial returns are leptokurtic, which means honest gradient updates naturally have larger variance and occasional extreme values. This makes distinguishing honest outliers from Byzantine attacks more challenging than in typical ML settings. Robust aggregation methods must be calibrated to the natural variability of financial gradient distributions.

7. **Defense in depth is essential.** Combine robust aggregation with gradient clipping, anomaly detection on submitted updates, and reputation systems that track participant reliability over time. No single defense mechanism is sufficient against a sophisticated adversary in financial markets.

8. **Computational overhead is manageable.** Krum adds $O(n^2 d)$ overhead per round, and trimmed mean adds $O(n d \log n)$. For typical federated learning setups with $n < 100$ participants and models with $d < 10^6$ parameters, this overhead is negligible compared to the local training cost.

## References

- Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine learning with adversaries: Byzantine tolerant gradient descent. *NeurIPS*.
- Yin, D., Chen, Y., Kannan, R., & Bartlett, P. (2018). Byzantine-robust distributed learning: Towards optimal statistical rates. *ICML*.
- Fang, M., Cao, X., Jia, J., & Gong, N. (2020). Local model poisoning attacks to Byzantine-robust federated learning. *USENIX Security*.
- Baruch, M., Baruch, G., & Goldberg, Y. (2019). A little is enough: Circumventing defenses for distributed learning. *NeurIPS*.
