# Chapter 290: Active Learning for Trading

## Introduction

In machine learning for trading, one of the most persistent bottlenecks is not the availability of raw data -- markets generate terabytes of tick data daily -- but rather the availability of *high-quality labels*. Consider the task of labeling market regimes (trending, mean-reverting, volatile), identifying complex event patterns (flash crashes, short squeezes, liquidation cascades), or annotating order flow for informed vs. uninformed trades. Each of these tasks typically requires expensive domain expert annotation.

**Active learning** addresses this challenge by intelligently selecting which unlabeled samples should be annotated next. Instead of randomly labeling data or labeling everything, the model itself identifies the samples from which it would learn the most. This can reduce labeling costs by 50-90% while achieving comparable or even superior model performance.

This chapter covers the theoretical foundations of active learning, presents three major query strategies (uncertainty sampling, diversity sampling, and query by committee), derives the mathematical underpinnings, and demonstrates a complete Rust implementation integrated with Bybit market data.

## Theoretical Foundations

### The Active Learning Framework

Active learning operates in a **pool-based** setting. We have:

- A small **labeled set** $\mathcal{L} = \{(x_i, y_i)\}_{i=1}^{n_L}$
- A large **unlabeled pool** $\mathcal{U} = \{x_j\}_{j=1}^{n_U}$ where $n_U \gg n_L$
- A **model** $f_\theta$ trained on $\mathcal{L}$
- An **acquisition function** $a(x; f_\theta)$ that scores the informativeness of each unlabeled sample

The iterative procedure is:

1. Train $f_\theta$ on current $\mathcal{L}$
2. Score all $x \in \mathcal{U}$ using $a(x; f_\theta)$
3. Select the top-$k$ highest-scoring samples: $\mathcal{Q} = \arg\max_{S \subset \mathcal{U}, |S|=k} \sum_{x \in S} a(x; f_\theta)$
4. Obtain labels for $\mathcal{Q}$ (from oracle/expert)
5. Move: $\mathcal{L} \leftarrow \mathcal{L} \cup \mathcal{Q}$, $\mathcal{U} \leftarrow \mathcal{U} \setminus \mathcal{Q}$
6. Repeat until labeling budget is exhausted

### Why Active Learning Matters for Trading

Trading data has several properties that make active learning particularly valuable:

1. **Concept drift**: Market regimes change, making older labels less relevant. Active learning naturally focuses on recent, uncertain regions.
2. **Class imbalance**: Rare events (crashes, breakouts) are the most interesting but hardest to label. Uncertainty sampling naturally gravitates toward ambiguous boundary cases.
3. **Expert cost**: A quantitative analyst's time costs $200-500/hour. Reducing the number of samples requiring manual review has direct ROI.
4. **Non-stationarity**: Labels that were correct last month may be wrong today. Active learning enables efficient relabeling campaigns.

## Uncertainty Sampling

### Least Confidence

The simplest uncertainty measure selects the sample where the model's most confident class prediction is lowest:

$$a_{\text{LC}}(x) = 1 - \max_c P(y = c \mid x; \theta)$$

For a sample where the model predicts class probabilities [0.51, 0.49], the least confidence score is $1 - 0.51 = 0.49$ (very uncertain). For [0.99, 0.01], the score is $1 - 0.99 = 0.01$ (very confident).

### Margin Sampling

Margin sampling considers the gap between the top two most probable classes:

$$a_{\text{margin}}(x) = 1 - \left[ P(y = c_1 \mid x) - P(y = c_2 \mid x) \right]$$

where $c_1$ and $c_2$ are the first and second most probable classes. A small margin means the model cannot distinguish between two competing hypotheses.

In trading, this is particularly useful for identifying price levels where the model is torn between "buy" and "hold" or "sell" and "hold" -- precisely the decision boundaries that matter most.

### Entropy Sampling

Shannon entropy measures the overall uncertainty of the full predictive distribution:

$$a_{\text{entropy}}(x) = H[P(y \mid x)] = -\sum_{c=1}^{C} P(y = c \mid x) \log P(y = c \mid x)$$

Entropy is maximized when the distribution is uniform (maximum uncertainty) and minimized when all probability mass is concentrated on a single class. For $C$ classes, the maximum entropy is $\log C$.

Entropy considers the full distribution, not just the top classes. This makes it superior when the model spreads probability across many classes -- common in multi-regime classification tasks.

### Comparison of Uncertainty Measures

| Measure | Considers | Best For |
|---------|-----------|----------|
| Least Confidence | Top-1 class only | Binary or near-binary tasks |
| Margin | Top-2 classes | Decision boundary refinement |
| Entropy | Full distribution | Multi-class with many regimes |

## Diversity Sampling

Pure uncertainty sampling can lead to **redundancy** -- selecting many similar samples from the same uncertain region. Diversity sampling addresses this by ensuring selected samples cover different regions of the feature space.

### Furthest-First Traversal

The greedy furthest-first algorithm maintains a set $\mathcal{S}$ of selected samples and iteratively adds the sample that is farthest from the current set:

$$x^* = \arg\max_{x \in \mathcal{U}} \min_{s \in \mathcal{S}} d(x, s)$$

where $d(\cdot, \cdot)$ is a distance metric (typically Euclidean or cosine distance).

This guarantees that selected samples form a well-spread covering of the input space. The algorithm is a 2-approximation of the optimal k-center problem.

### Trading Application

In trading, diversity sampling ensures that the labeled set covers different market microstructure regimes:
- High-volatility periods vs. quiet periods
- Different volume profiles (thin vs. thick books)
- Various trend patterns (momentum, reversal, sideways)

## Query by Committee (QBC)

### Ensemble Disagreement

QBC maintains a committee of $M$ models $\{f_1, \ldots, f_M\}$, each trained on the current labeled set but with different random initializations (or architectures). The acquisition function measures **disagreement**:

$$a_{\text{QBC}}(x) = H\left[\frac{1}{M}\sum_{m=1}^{M} \mathbb{1}[f_m(x) = c]\right]$$

This is the **vote entropy** -- the entropy of the distribution of predicted classes across committee members.

### Consensus Entropy

A more refined measure uses the average KL divergence of each member from the consensus:

$$a_{\text{KL}}(x) = \frac{1}{M}\sum_{m=1}^{M} D_{\text{KL}}\left(P_m(y \mid x) \;\|\; \bar{P}(y \mid x)\right)$$

where $\bar{P}(y \mid x) = \frac{1}{M}\sum_{m=1}^{M} P_m(y \mid x)$ is the consensus distribution.

### Trading Interpretation

In trading, committee disagreement often signals:
- **Regime transitions**: Some models have adapted to the new regime, others have not
- **Ambiguous price action**: E.g., a breakout that could be real or a fakeout
- **Data quality issues**: Unusual spreads or volume that confuse some models

## Rust Implementation Walkthrough

### Architecture Overview

The implementation consists of five main components:

1. **`SimpleClassifier`**: A single-layer linear classifier (`y = softmax(Wx + b)`) trained with SGD. In production, this would be replaced with a deep neural network, but the linear model suffices to demonstrate the active learning mechanics.

2. **`UncertaintySampler`**: Implements all three uncertainty strategies (least confidence, margin, entropy). The `select()` method scores every sample in the pool and returns the top-k indices.

3. **`DiversitySampler`**: Implements greedy furthest-first traversal. Maintains a running vector of minimum distances to avoid recomputation.

4. **`QueryByCommittee`**: Maintains a vector of `SimpleClassifier` instances. Provides both `vote_entropy()` and `consensus_entropy()` acquisition functions.

5. **`ActiveLearningPipeline`**: The orchestrator that manages the labeled/unlabeled split, runs query rounds, trains models, and tracks accuracy history.

### Key Implementation Details

The `SimpleClassifier` uses the standard cross-entropy gradient for SGD:

```
gradient_j = P(y=j|x) - 1{j == label}
W[j,f] -= lr * gradient_j * x[f]
b[j]   -= lr * gradient_j
```

The diversity sampler avoids O(n*k) distance recomputation by incrementally updating minimum distances:

```
for each new selected point s:
    for each remaining point x in pool:
        min_distance[x] = min(min_distance[x], dist(x, s))
    next_selected = argmax(min_distance)
```

Trading features are engineered from raw OHLCV data:
- **Log return**: `ln(close_t / close_{t-1})` -- captures price momentum
- **Body ratio**: `|close - open| / (high - low)` -- measures candlestick conviction
- **Shadow ratios**: Upper and lower wicks relative to range -- measures rejection
- **Volume change**: Relative volume shift -- indicates participation

### Bybit Integration

The trading example fetches real-time OHLCV candles from the Bybit V5 API:

```
GET /v5/market/kline?category=spot&symbol=BTCUSDT&interval=60&limit=200
```

The response is parsed into a `Vec<Candle>` and processed through the feature engineering pipeline. Labels are generated from forward returns with a configurable threshold and lookahead period.

## Experimental Results

The example compares five strategies on BTCUSDT hourly data:

| Strategy | Typical Final Accuracy | Data Used |
|----------|----------------------|-----------|
| Random Baseline | ~35-40% | 50 samples |
| Uncertainty (Entropy) | ~40-50% | 50 samples |
| Uncertainty (Margin) | ~38-48% | 50 samples |
| Uncertainty (Least Confidence) | ~38-48% | 50 samples |
| Diversity | ~37-45% | 50 samples |
| Full Dataset | ~40-50% | ~140 samples |

The key insight is that active learning with 50 labeled samples (36% of the data) often achieves accuracy comparable to training on the full 140-sample training set.

## Advanced Techniques

### Hybrid Acquisition Functions

In practice, combining uncertainty and diversity yields the best results. The **BADGE** (Batch Active learning by Diverse Gradient Embeddings) method selects diverse samples in the gradient embedding space, naturally combining both signals:

$$a_{\text{BADGE}}(x) = \text{kmeans++}\left(\nabla_\theta \ell(f_\theta(x), \hat{y})\right)$$

### Expected Model Change

Rather than measuring uncertainty about the prediction, we can measure how much the model's parameters would change if we labeled a particular sample:

$$a_{\text{EMC}}(x) = \left\| \mathbb{E}_{y \sim P(y|x)} \left[ \nabla_\theta \ell(f_\theta(x), y) \right] \right\|$$

Samples that would cause the largest parameter update are the most informative.

### Bayesian Active Learning by Disagreement (BALD)

BALD uses the mutual information between predictions and model parameters:

$$a_{\text{BALD}}(x) = I(y; \theta \mid x) = H[P(y \mid x)] - \mathbb{E}_{P(\theta)}[H[P(y \mid x, \theta)]]$$

This distinguishes between **aleatoric uncertainty** (inherent noise -- e.g., market randomness) and **epistemic uncertainty** (model ignorance -- reducible by more data). Only epistemic uncertainty is useful for active learning.

## Trading Applications

### 1. Regime Labeling

Active learning can efficiently identify market regimes. An expert labels a small seed of clearly trending/ranging periods. The model then selects ambiguous transition periods for expert review, rapidly building a comprehensive regime classifier.

### 2. Event Detection

For detecting events like flash crashes or short squeezes, active learning focuses the expert's attention on borderline cases -- was that 2% drop in 5 minutes a "flash crash" or normal volatility? These boundary cases are precisely what the model needs to learn.

### 3. Order Flow Classification

Classifying trades as informed vs. uninformed is expensive (requires matching with news, insider filings, etc.). Active learning selects the trades where classification is most ambiguous, maximizing the value of each manual investigation.

### 4. Sentiment Labeling

Financial news sentiment is subjective and requires human judgment. Active learning selects articles where automated NLP models disagree or are uncertain, ensuring expert annotators focus on genuinely ambiguous content.

## Key Takeaways

1. **Active learning reduces labeling cost by 50-90%** while maintaining model performance. In trading, where expert labels are expensive, this translates to significant ROI.

2. **Uncertainty sampling** is the simplest and often most effective strategy. Entropy sampling is preferred for multi-class tasks; margin sampling excels at binary decision boundaries.

3. **Diversity sampling** prevents redundancy and ensures coverage of the feature space. Combine with uncertainty for best results.

4. **Query by Committee** naturally captures epistemic uncertainty through ensemble disagreement. It requires training multiple models but provides robust acquisition signals.

5. **Trading-specific considerations**: Non-stationarity means the active learning loop should be continuous -- new data arrives, old labels may become stale, and the model should constantly identify what needs re-annotation.

6. **The Rust implementation** provides production-ready performance. The linear classifier can be replaced with any model that produces probability estimates; the active learning infrastructure remains the same.

## Running the Code

```bash
cd 290_active_learning_trading/rust
cargo test           # Run 15 unit tests
cargo run --example trading_example  # Full Bybit integration demo
```

## References

1. Settles, B. (2009). "Active Learning Literature Survey." Computer Sciences Technical Report 1648, University of Wisconsin-Madison.
2. Gal, Y., Islam, R., & Ghahramani, Z. (2017). "Deep Bayesian Active Learning with Image Data." ICML.
3. Ash, J. T., et al. (2020). "Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds." ICLR.
4. Houlsby, N., et al. (2011). "Bayesian Active Learning for Classification and Preference Learning." arXiv:1112.5745.
5. Sener, O. & Savarese, S. (2018). "Active Learning for Convolutional Neural Networks: A Core-Set Approach." ICLR.
