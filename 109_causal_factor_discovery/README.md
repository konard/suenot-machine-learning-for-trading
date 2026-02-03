# Chapter 109: Causal Factor Discovery

## Overview

Causal Factor Discovery is a machine learning approach that goes beyond correlation-based analysis to identify factors that have genuine causal relationships with asset returns. Traditional factor models (like Fama-French) rely on empirically observed correlations, which may be spurious, time-varying, or confounded. Causal discovery methods aim to uncover the underlying causal structure from observational data, enabling more robust and interpretable trading strategies.

The key insight is that while correlation can break down during regime changes, true causal relationships remain stable. This chapter covers causal discovery algorithms (PC, FCI, GES, NOTEARS), their application to financial time series, and practical implementations in both Python and Rust for systematic trading.

## Table of Contents

1. [Introduction to Causal Discovery](#introduction-to-causal-discovery)
2. [From Correlation to Causation](#from-correlation-to-causation)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Causal Discovery Algorithms](#causal-discovery-algorithms)
5. [Applications to Trading](#applications-to-trading)
6. [Implementation in Python](#implementation-in-python)
7. [Implementation in Rust](#implementation-in-rust)
8. [Practical Examples with Stock and Crypto Data](#practical-examples-with-stock-and-crypto-data)
9. [Backtesting Framework](#backtesting-framework)
10. [Performance Evaluation](#performance-evaluation)
11. [Future Directions](#future-directions)
12. [References](#references)

---

## Introduction to Causal Discovery

### Why Causation Matters in Trading

Consider a classic example: ice cream sales and drowning deaths are highly correlated. But eating ice cream doesn't cause drowning — both are caused by a common factor: hot weather.

In financial markets, similar issues arise:
- Two stocks may be correlated because they're in the same sector (confounding)
- A factor may appear predictive only because it's correlated with the true driver
- Correlations can reverse during market stress (correlation breakdown)

Causal discovery aims to identify the true causal structure:

```
Spurious correlation:       Causal relationship:
   Ice Cream                    Interest Rates
      ↑                              ↓
      |  correlated                  |  causes
      ↓                              ↓
   Drownings                    Stock Returns

(Wrong: No causal link)      (Right: Direct effect)
```

### Structural Causal Models (SCMs)

A Structural Causal Model consists of:
1. **Endogenous variables** V = {V₁, V₂, ..., Vₙ}: The variables we observe
2. **Exogenous variables** U = {U₁, U₂, ..., Uₙ}: Unobserved noise/disturbances
3. **Structural equations**: Vᵢ = fᵢ(PAᵢ, Uᵢ), where PAᵢ are the parents (causes) of Vᵢ

For example:
```
Market Sentiment → Volatility → Returns
       ↓
    Volume

Structural equations:
Volatility = f₁(Sentiment, U₁)
Returns = f₂(Volatility, U₂)
Volume = f₃(Sentiment, U₃)
```

### Causal Graphs (DAGs)

The causal structure is represented as a Directed Acyclic Graph (DAG):
- Nodes = Variables
- Edges = Direct causal relationships
- No cycles (cause precedes effect)

```
Example DAG for factor returns:

  Market         Interest Rates
     ↓                  ↓
  Beta Factor    Bond Factor
     ↓                  ↓
     └──────→ Stock Return ←──────┘
                    ↑
              Size Factor
```

---

## From Correlation to Causation

### The Fundamental Problem

**Correlation**: P(Y | X) ≠ P(Y)
- X and Y are statistically dependent
- Could be: X → Y, Y → X, X ← Z → Y (confounding), or selection bias

**Causation**: P(Y | do(X)) — the effect of *intervening* on X
- What happens to Y if we forcibly set X to a value?
- do(X) removes all incoming edges to X in the causal graph

### The do-calculus

Pearl's do-calculus allows us to compute causal effects from observational data when certain conditions are met:

**Rule 1 (Insertion/deletion of observations):**
P(Y | do(X), Z, W) = P(Y | do(X), W) if Y ⊥ Z | X, W in the manipulated graph

**Rule 2 (Action/observation exchange):**
P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W) if Y ⊥ Z | X, W in the manipulated graph

**Rule 3 (Insertion/deletion of actions):**
P(Y | do(X), do(Z), W) = P(Y | do(X), W) if Y ⊥ Z | X, W in the manipulated graph

### Identifying Causal Effects

Key graphical criteria:

**Backdoor Criterion**: A set of variables Z satisfies the backdoor criterion for (X, Y) if:
1. No node in Z is a descendant of X
2. Z blocks all backdoor paths from X to Y

If Z satisfies the backdoor criterion:
```
P(Y | do(X)) = Σ_z P(Y | X, Z=z) P(Z=z)
```

**Frontdoor Criterion**: When backdoor adjustment isn't possible but there's a mediating variable:
```
X → M → Y  (with unobserved confounding between X and Y)
P(Y | do(X)) = Σ_m P(M=m | X) Σ_x' P(Y | M=m, X=x') P(X=x')
```

---

## Mathematical Foundation

### Conditional Independence Testing

Causal discovery algorithms rely heavily on conditional independence (CI) tests:

**Definition**: X ⊥ Y | Z means X and Y are independent given Z:
```
P(X, Y | Z) = P(X | Z) P(Y | Z)
```

**Common CI Tests**:

1. **Partial Correlation Test** (linear, Gaussian):
```
ρ(X,Y|Z) = (ρ(X,Y) - ρ(X,Z)ρ(Y,Z)) / √((1-ρ(X,Z)²)(1-ρ(Y,Z)²))

Test statistic: z = (1/2)ln((1+ρ)/(1-ρ)) √(n-|Z|-3)
```

2. **Kernel-based Tests** (nonlinear):
- HSIC (Hilbert-Schmidt Independence Criterion)
- Kernel Conditional Independence Test (KCI)

3. **Mutual Information Tests**:
```
I(X; Y | Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)
```

### Markov Equivalence Classes

Multiple DAGs can encode the same conditional independencies:

```
DAG 1: X → Y → Z    }
DAG 2: X ← Y ← Z    }  Same independencies: X ⊥ Z | Y
DAG 3: X ← Y → Z    }

Different: X → Y ← Z  (V-structure: X ⊥ Z but X ⊬ Z | Y)
```

Causal discovery from observational data can only identify the **Markov equivalence class**, represented by a CPDAG (Completed Partially Directed Acyclic Graph).

### Faithfulness Assumption

The faithfulness assumption states that all conditional independencies in the data are entailed by the causal graph (no "accidental" cancellations).

**Violation example**: X → Y, X → Z → Y where the direct and indirect effects cancel:
```
Y = αX + βZ, Z = γX
Y = αX + βγX = (α + βγ)X

If α = -βγ, then Y ⊥ X despite X → Y!
```

---

## Causal Discovery Algorithms

### PC Algorithm (Peter-Clark)

The PC algorithm learns a CPDAG through conditional independence testing:

**Phase 1: Skeleton Learning**
```
Start with complete undirected graph
For each edge X — Y:
    For conditioning set sizes k = 0, 1, 2, ...
        For each subset S of neighbors of X or Y with |S| = k:
            If X ⊥ Y | S:
                Remove edge X — Y
                Record S as separation set
```

**Phase 2: Orient V-structures**
```
For each triple X — Z — Y where X and Y are not adjacent:
    If Z ∉ SepSet(X, Y):
        Orient as X → Z ← Y (V-structure)
```

**Phase 3: Orient remaining edges** (using Meek rules):
```
R1: If X → Y — Z and X, Z non-adjacent: orient Y → Z
R2: If X → Y → Z and X — Z: orient X → Z
R3: If X — Y → Z, X — W → Z, X — Z, Y, W non-adjacent: orient X → Z
R4: If X — Y → Z → W and X — W: orient X → W
```

### FCI Algorithm (Fast Causal Inference)

FCI extends PC to handle latent confounders:

- Outputs a PAG (Partial Ancestral Graph)
- Uses additional edge types: ◦→ (possibly directed), ◦—◦ (unknown), ↔ (bidirected, indicating latent confounder)

```
Example PAG:
X ◦→ Y means: X causes Y, or there's a latent confounder, or both
X ↔ Y means: latent confounder between X and Y (no direct edge)
```

### GES Algorithm (Greedy Equivalence Search)

Score-based algorithm that searches over equivalence classes:

**Forward Phase**:
```
Start with empty graph
Repeat:
    Find edge addition that maximally increases BIC score
    Add edge if score improves
Until no improving addition exists
```

**Backward Phase**:
```
Repeat:
    Find edge deletion that maximally increases BIC score
    Delete edge if score improves
Until no improving deletion exists
```

**BIC Score**:
```
BIC(G, D) = Σᵢ [log P(Xᵢ | PAᵢ) - (|θᵢ|/2) log n]
```

### NOTEARS (Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian for Structure learning)

Continuous optimization approach:

**Key insight**: Acyclicity can be expressed as a smooth constraint:
```
h(W) = tr(exp(W ◦ W)) - d = 0
```
where W is the adjacency matrix and ◦ is element-wise product.

**Optimization problem**:
```
minimize    (1/2n)||X - XW||²_F + λ||W||₁
subject to  h(W) = 0
```

Solved via augmented Lagrangian method:
```
L(W, α, ρ) = (1/2n)||X - XW||²_F + λ||W||₁ + α·h(W) + (ρ/2)h(W)²
```

---

## Applications to Trading

### Causal Factor Discovery for Alpha

**Goal**: Identify factors that causally influence returns (not just correlated)

**Process**:
1. Collect potential factors (technical, fundamental, alternative data)
2. Apply causal discovery to identify factor → return edges
3. Estimate causal effects using backdoor/frontdoor adjustment
4. Build trading signals from causally validated factors

```
Traditional approach:
Factor_1, Factor_2, ..., Factor_k → All used if correlated with returns

Causal approach:
Factor_1 → Returns (causal)           → Use
Factor_2 ← Confounder → Returns       → Adjust for confounder
Factor_3 ←─────────────── Returns     → Discard (reverse causation)
Factor_4 ... no edge ... Returns      → Discard (spurious)
```

### Regime-Robust Factors

Causal relationships should be stable across regimes:

```
Regime 1 (Bull market):
  Momentum → Returns (strong positive effect)

Regime 2 (Bear market):
  Momentum → Returns (negative effect, but causal link persists)

vs. Spurious factor:
  Regime 1: Factor X correlated with returns
  Regime 2: Correlation disappears (was confounded)
```

### Causal Portfolio Construction

Use causal structure for portfolio optimization:
1. Identify causal drivers of each asset's returns
2. Diversify based on causal factors, not just correlations
3. More robust to correlation breakdown during stress

```python
# Traditional: correlation-based
weights = optimize(returns.cov())

# Causal: factor-based with causal adjustment
causal_factors = discover_causal_factors(data)
factor_exposures = estimate_causal_effects(returns, causal_factors)
weights = optimize_causal(factor_exposures, factor_cov)
```

---

## Implementation in Python

### Causal Discovery Model

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from itertools import combinations

@dataclass
class CausalGraph:
    """Represents a causal graph (DAG or CPDAG)."""
    nodes: List[str]
    edges: Dict[Tuple[str, str], str]  # (from, to): edge_type

    def add_edge(self, source: str, target: str, edge_type: str = "directed"):
        self.edges[(source, target)] = edge_type

    def remove_edge(self, source: str, target: str):
        self.edges.pop((source, target), None)
        self.edges.pop((target, source), None)

    def get_parents(self, node: str) -> List[str]:
        return [s for (s, t), e in self.edges.items() if t == node and e == "directed"]

    def get_children(self, node: str) -> List[str]:
        return [t for (s, t), e in self.edges.items() if s == node and e == "directed"]

    def get_neighbors(self, node: str) -> Set[str]:
        neighbors = set()
        for (s, t), e in self.edges.items():
            if s == node:
                neighbors.add(t)
            if t == node:
                neighbors.add(s)
        return neighbors


class PartialCorrelationTest:
    """Partial correlation test for conditional independence."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def test(self, data: pd.DataFrame, x: str, y: str, z: List[str]) -> Tuple[bool, float]:
        """
        Test if X ⊥ Y | Z using partial correlation.
        Returns: (is_independent, p_value)
        """
        if len(z) == 0:
            # Simple correlation
            r, p = stats.pearsonr(data[x], data[y])
            return p > self.alpha, p

        # Partial correlation via regression residuals
        from sklearn.linear_model import LinearRegression

        # Residualize X on Z
        reg_x = LinearRegression().fit(data[z], data[x])
        res_x = data[x] - reg_x.predict(data[z])

        # Residualize Y on Z
        reg_y = LinearRegression().fit(data[z], data[y])
        res_y = data[y] - reg_y.predict(data[z])

        # Correlation of residuals
        r, p = stats.pearsonr(res_x, res_y)

        # Fisher z-transform for proper p-value
        n = len(data)
        z_stat = 0.5 * np.log((1 + r) / (1 - r + 1e-10))
        se = 1 / np.sqrt(n - len(z) - 3)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat / se)))

        return p_value > self.alpha, p_value


class PCAlgorithm:
    """PC Algorithm for causal discovery."""

    def __init__(self, alpha: float = 0.05, max_cond_size: int = 4):
        self.alpha = alpha
        self.max_cond_size = max_cond_size
        self.ci_test = PartialCorrelationTest(alpha)
        self.sep_sets: Dict[Tuple[str, str], Set[str]] = {}

    def fit(self, data: pd.DataFrame) -> CausalGraph:
        """Learn causal structure from data."""
        nodes = list(data.columns)

        # Phase 1: Learn skeleton
        graph = self._learn_skeleton(data, nodes)

        # Phase 2: Orient v-structures
        graph = self._orient_v_structures(graph, nodes)

        # Phase 3: Apply Meek rules
        graph = self._apply_meek_rules(graph)

        return graph

    def _learn_skeleton(self, data: pd.DataFrame, nodes: List[str]) -> CausalGraph:
        """Learn the undirected skeleton."""
        # Start with complete graph
        graph = CausalGraph(nodes=nodes, edges={})
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                graph.add_edge(n1, n2, "undirected")
                graph.add_edge(n2, n1, "undirected")

        # Remove edges based on CI tests
        for cond_size in range(self.max_cond_size + 1):
            edges_to_check = list(graph.edges.keys())

            for (x, y) in edges_to_check:
                if (x, y) not in graph.edges and (y, x) not in graph.edges:
                    continue

                # Get potential conditioning sets
                neighbors_x = graph.get_neighbors(x) - {y}
                neighbors_y = graph.get_neighbors(y) - {x}
                potential_cond = neighbors_x | neighbors_y

                if len(potential_cond) < cond_size:
                    continue

                # Test all conditioning sets of this size
                for cond_set in combinations(potential_cond, cond_size):
                    cond_list = list(cond_set)
                    is_independent, p_val = self.ci_test.test(data, x, y, cond_list)

                    if is_independent:
                        graph.remove_edge(x, y)
                        self.sep_sets[(x, y)] = set(cond_set)
                        self.sep_sets[(y, x)] = set(cond_set)
                        break

        return graph

    def _orient_v_structures(self, graph: CausalGraph, nodes: List[str]) -> CausalGraph:
        """Orient v-structures (X → Z ← Y)."""
        for z in nodes:
            neighbors = list(graph.get_neighbors(z))
            for i, x in enumerate(neighbors):
                for y in neighbors[i+1:]:
                    # Check if x and y are non-adjacent
                    if y not in graph.get_neighbors(x):
                        # Check if z is NOT in separation set
                        sep_set = self.sep_sets.get((x, y), set())
                        if z not in sep_set:
                            # Orient as v-structure: X → Z ← Y
                            graph.remove_edge(x, z)
                            graph.remove_edge(z, x)
                            graph.remove_edge(y, z)
                            graph.remove_edge(z, y)
                            graph.add_edge(x, z, "directed")
                            graph.add_edge(y, z, "directed")

        return graph

    def _apply_meek_rules(self, graph: CausalGraph) -> CausalGraph:
        """Apply Meek's orientation rules."""
        changed = True
        while changed:
            changed = False

            for (x, y), edge_type in list(graph.edges.items()):
                if edge_type != "undirected":
                    continue

                # Rule 1: X → Y — Z, X and Z non-adjacent → Y → Z
                parents_y = graph.get_parents(y)
                for p in parents_y:
                    if p not in graph.get_neighbors(x):
                        graph.remove_edge(x, y)
                        graph.remove_edge(y, x)
                        graph.add_edge(y, x, "directed")
                        changed = True
                        break

        return graph


class CausalEffectEstimator:
    """Estimate causal effects using adjustment formulas."""

    def __init__(self, graph: CausalGraph):
        self.graph = graph

    def get_backdoor_adjustment_set(self, treatment: str, outcome: str) -> Optional[Set[str]]:
        """Find a valid backdoor adjustment set."""
        parents_treatment = set(self.graph.get_parents(treatment))

        # Simple heuristic: use all parents of treatment
        # (More sophisticated: find minimal sufficient set)
        if len(parents_treatment) > 0:
            return parents_treatment
        return set()

    def estimate_ate(self, data: pd.DataFrame, treatment: str, outcome: str) -> Dict[str, float]:
        """
        Estimate Average Treatment Effect using backdoor adjustment.
        """
        adjustment_set = self.get_backdoor_adjustment_set(treatment, outcome)

        if adjustment_set is None:
            raise ValueError(f"Cannot identify causal effect of {treatment} on {outcome}")

        from sklearn.linear_model import LinearRegression

        if len(adjustment_set) == 0:
            # No confounders, simple regression
            reg = LinearRegression().fit(data[[treatment]], data[outcome])
            ate = reg.coef_[0]

            # Bootstrap confidence interval
            bootstrap_ates = []
            for _ in range(1000):
                idx = np.random.choice(len(data), len(data), replace=True)
                boot_data = data.iloc[idx]
                boot_reg = LinearRegression().fit(boot_data[[treatment]], boot_data[outcome])
                bootstrap_ates.append(boot_reg.coef_[0])

            ci_low, ci_high = np.percentile(bootstrap_ates, [2.5, 97.5])
        else:
            # Backdoor adjustment via regression
            features = [treatment] + list(adjustment_set)
            reg = LinearRegression().fit(data[features], data[outcome])
            ate = reg.coef_[0]  # Coefficient on treatment

            # Bootstrap confidence interval
            bootstrap_ates = []
            for _ in range(1000):
                idx = np.random.choice(len(data), len(data), replace=True)
                boot_data = data.iloc[idx]
                boot_reg = LinearRegression().fit(boot_data[features], boot_data[outcome])
                bootstrap_ates.append(boot_reg.coef_[0])

            ci_low, ci_high = np.percentile(bootstrap_ates, [2.5, 97.5])

        return {
            'ate': ate,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'adjustment_set': list(adjustment_set)
        }


class CausalFactorModel:
    """
    Causal Factor Model for trading.
    Discovers causal factors and estimates their effects on returns.
    """

    def __init__(self, alpha: float = 0.05, min_effect_size: float = 0.01):
        self.alpha = alpha
        self.min_effect_size = min_effect_size
        self.graph: Optional[CausalGraph] = None
        self.causal_factors: List[str] = []
        self.factor_effects: Dict[str, float] = {}

    def fit(self, features: pd.DataFrame, returns: pd.Series) -> 'CausalFactorModel':
        """
        Discover causal factors from features and estimate their effects.

        Args:
            features: DataFrame of potential factor values
            returns: Series of asset returns
        """
        # Combine features and returns
        data = features.copy()
        data['returns'] = returns.values

        # Discover causal structure
        pc = PCAlgorithm(alpha=self.alpha)
        self.graph = pc.fit(data)

        # Identify factors that cause returns
        estimator = CausalEffectEstimator(self.graph)

        for factor in features.columns:
            # Check if there's a path from factor to returns
            try:
                effect = estimator.estimate_ate(data, factor, 'returns')

                # Keep factor if effect is significant and large enough
                if effect['ci_low'] * effect['ci_high'] > 0:  # Same sign
                    if abs(effect['ate']) >= self.min_effect_size:
                        self.causal_factors.append(factor)
                        self.factor_effects[factor] = effect['ate']
            except ValueError:
                continue

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate return predictions from causal factors."""
        if not self.causal_factors:
            return np.zeros(len(features))

        predictions = np.zeros(len(features))
        for factor in self.causal_factors:
            if factor in features.columns:
                predictions += self.factor_effects[factor] * features[factor].values

        return predictions

    def get_factor_summary(self) -> pd.DataFrame:
        """Get summary of discovered causal factors."""
        return pd.DataFrame({
            'factor': self.causal_factors,
            'causal_effect': [self.factor_effects[f] for f in self.causal_factors]
        }).sort_values('causal_effect', key=abs, ascending=False)
```

### Trading Data Pipeline

```python
import pandas as pd
import numpy as np
from typing import Dict, List
import requests

class FactorDataPipeline:
    """Pipeline for preparing factor data for causal discovery."""

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def compute_factors(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute potential causal factors from price data.

        Args:
            prices: DataFrame with 'open', 'high', 'low', 'close', 'volume'
        """
        factors = pd.DataFrame(index=prices.index)

        close = prices['close']
        volume = prices['volume'] if 'volume' in prices else pd.Series(1, index=prices.index)

        # Momentum factors
        factors['momentum_5'] = close.pct_change(5)
        factors['momentum_20'] = close.pct_change(20)
        factors['momentum_60'] = close.pct_change(60)

        # Mean reversion factors
        factors['ma_deviation_20'] = (close - close.rolling(20).mean()) / close.rolling(20).std()
        factors['ma_deviation_50'] = (close - close.rolling(50).mean()) / close.rolling(50).std()

        # Volatility factors
        returns = close.pct_change()
        factors['volatility_20'] = returns.rolling(20).std()
        factors['volatility_ratio'] = returns.rolling(5).std() / returns.rolling(20).std()

        # Volume factors
        factors['volume_ma_ratio'] = volume / volume.rolling(20).mean()
        factors['volume_trend'] = volume.rolling(5).mean() / volume.rolling(20).mean()

        # Price range factors
        factors['atr'] = self._compute_atr(prices, 14)
        factors['range_pct'] = (prices['high'] - prices['low']) / close

        # RSI
        factors['rsi'] = self._compute_rsi(close, 14)

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        factors['macd'] = (ema_12 - ema_26) / close

        # Lagged returns (for Granger-style causality)
        factors['return_lag1'] = returns.shift(1)
        factors['return_lag2'] = returns.shift(2)
        factors['return_lag5'] = returns.shift(5)

        return factors.dropna()

    def _compute_atr(self, prices: pd.DataFrame, period: int) -> pd.Series:
        """Compute Average True Range."""
        high, low, close = prices['high'], prices['low'], prices['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _compute_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta).where(delta < 0, 0).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))


def fetch_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch stock data using yfinance."""
    import yfinance as yf
    data = yf.download(symbol, start=start, end=end)
    data.columns = [c.lower() for c in data.columns]
    return data


def fetch_bybit_data(symbol: str = "BTCUSDT", interval: str = "D", limit: int = 1000) -> pd.DataFrame:
    """Fetch cryptocurrency data from Bybit API."""
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    resp = requests.get(url, params=params).json()
    records = resp['result']['list']

    df = pd.DataFrame(records, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'].astype(int), unit='ms')
    df = df.sort_values('open_time').reset_index(drop=True)
    df.set_index('open_time', inplace=True)

    return df
```

### Backtesting Framework

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class BacktestResult:
    """Results from backtesting a causal factor strategy."""
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    factor_summary: pd.DataFrame


class CausalFactorBacktester:
    """Backtesting framework for causal factor strategies."""

    def __init__(
        self,
        model: CausalFactorModel,
        threshold: float = 0.0,
        transaction_cost: float = 0.001,
        position_sizing: str = 'equal'
    ):
        self.model = model
        self.threshold = threshold
        self.transaction_cost = transaction_cost
        self.position_sizing = position_sizing

    def run(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
        train_window: int = 252,
        retrain_every: int = 21
    ) -> BacktestResult:
        """
        Run walk-forward backtest with periodic model retraining.
        """
        signals = []
        actual_returns = []

        for i in range(train_window, len(features) - 1, retrain_every):
            # Training window
            train_end = i
            train_start = max(0, train_end - train_window)

            train_features = features.iloc[train_start:train_end]
            train_returns = returns.iloc[train_start:train_end]

            # Fit model on training data
            self.model.fit(train_features, train_returns)

            # Generate predictions for test period
            test_end = min(i + retrain_every, len(features) - 1)
            test_features = features.iloc[i:test_end]
            test_returns = returns.iloc[i+1:test_end+1]

            # Generate signals
            predictions = self.model.predict(test_features)

            for j, pred in enumerate(predictions):
                if j < len(test_returns):
                    if pred > self.threshold:
                        signals.append(1)
                    elif pred < -self.threshold:
                        signals.append(-1)
                    else:
                        signals.append(0)
                    actual_returns.append(test_returns.iloc[j])

        signals = np.array(signals)
        actual_returns = np.array(actual_returns)

        # Compute strategy returns
        position_changes = np.diff(signals, prepend=0)
        costs = np.abs(position_changes) * self.transaction_cost
        strategy_returns = signals * actual_returns - costs

        # Compute metrics
        metrics = self._compute_metrics(strategy_returns, signals)

        return BacktestResult(
            total_return=metrics['total_return'],
            annual_return=metrics['annual_return'],
            annual_volatility=metrics['annual_volatility'],
            sharpe_ratio=metrics['sharpe_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            max_drawdown=metrics['max_drawdown'],
            win_rate=metrics['win_rate'],
            num_trades=metrics['num_trades'],
            factor_summary=self.model.get_factor_summary()
        )

    def _compute_metrics(self, returns: np.ndarray, signals: np.ndarray) -> Dict[str, float]:
        """Compute performance metrics."""
        # Total return
        total_return = np.exp(np.sum(np.log1p(returns))) - 1

        # Annualized metrics (assuming daily data)
        n_days = len(returns)
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        annual_volatility = np.std(returns) * np.sqrt(252)

        # Sharpe ratio
        sharpe_ratio = annual_return / (annual_volatility + 1e-10)

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1e-10
        sortino_ratio = annual_return / downside_vol

        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        # Win rate
        winning_days = np.sum(returns > 0)
        trading_days = np.sum(returns != 0)
        win_rate = winning_days / (trading_days + 1e-10)

        # Number of trades
        position_changes = np.diff(signals, prepend=0)
        num_trades = np.sum(np.abs(position_changes) > 0)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': int(num_trades)
        }
```

---

## Implementation in Rust

### Project Structure

```
109_causal_factor_discovery/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── graph.rs
│   ├── discovery/
│   │   ├── mod.rs
│   │   ├── pc.rs
│   │   └── ci_test.rs
│   ├── estimation/
│   │   ├── mod.rs
│   │   └── backdoor.rs
│   ├── data/
│   │   ├── mod.rs
│   │   ├── factors.rs
│   │   └── bybit.rs
│   └── trading/
│       ├── mod.rs
│       ├── strategy.rs
│       └── backtest.rs
└── examples/
    ├── discover_factors.rs
    ├── crypto_causal.rs
    └── backtest_strategy.rs
```

### Cargo.toml

```toml
[package]
name = "causal_factor_discovery"
version = "0.1.0"
edition = "2021"

[dependencies]
ndarray = "0.15"
ndarray-stats = "0.5"
statrs = "0.16"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
reqwest = { version = "0.11", features = ["json"] }
tokio = { version = "1.0", features = ["full"] }
itertools = "0.12"

[dev-dependencies]
criterion = "0.5"
```

### Causal Graph Implementation (Rust)

```rust
// src/graph.rs
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq)]
pub enum EdgeType {
    Directed,
    Undirected,
    Bidirected,  // For latent confounders
}

#[derive(Debug, Clone)]
pub struct CausalGraph {
    pub nodes: Vec<String>,
    pub edges: HashMap<(String, String), EdgeType>,
    node_set: HashSet<String>,
}

impl CausalGraph {
    pub fn new(nodes: Vec<String>) -> Self {
        let node_set: HashSet<String> = nodes.iter().cloned().collect();
        CausalGraph {
            nodes,
            edges: HashMap::new(),
            node_set,
        }
    }

    pub fn add_edge(&mut self, source: &str, target: &str, edge_type: EdgeType) {
        self.edges.insert((source.to_string(), target.to_string()), edge_type);
    }

    pub fn remove_edge(&mut self, source: &str, target: &str) {
        self.edges.remove(&(source.to_string(), target.to_string()));
        self.edges.remove(&(target.to_string(), source.to_string()));
    }

    pub fn has_edge(&self, source: &str, target: &str) -> bool {
        self.edges.contains_key(&(source.to_string(), target.to_string()))
            || self.edges.contains_key(&(target.to_string(), source.to_string()))
    }

    pub fn get_parents(&self, node: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter_map(|((s, t), e)| {
                if t == node && *e == EdgeType::Directed {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn get_children(&self, node: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter_map(|((s, t), e)| {
                if s == node && *e == EdgeType::Directed {
                    Some(t.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn get_neighbors(&self, node: &str) -> HashSet<String> {
        let mut neighbors = HashSet::new();
        for ((s, t), _) in &self.edges {
            if s == node {
                neighbors.insert(t.clone());
            }
            if t == node {
                neighbors.insert(s.clone());
            }
        }
        neighbors
    }

    pub fn add_complete_undirected(&mut self) {
        for i in 0..self.nodes.len() {
            for j in (i + 1)..self.nodes.len() {
                let n1 = &self.nodes[i];
                let n2 = &self.nodes[j];
                self.add_edge(n1, n2, EdgeType::Undirected);
                self.add_edge(n2, n1, EdgeType::Undirected);
            }
        }
    }
}
```

### Conditional Independence Test (Rust)

```rust
// src/discovery/ci_test.rs
use ndarray::{Array1, Array2, Axis};
use statrs::distribution::{ContinuousCDF, Normal};

pub struct PartialCorrelationTest {
    pub alpha: f64,
}

impl PartialCorrelationTest {
    pub fn new(alpha: f64) -> Self {
        PartialCorrelationTest { alpha }
    }

    /// Test X ⊥ Y | Z using partial correlation
    pub fn test(
        &self,
        data: &Array2<f64>,
        x_idx: usize,
        y_idx: usize,
        z_indices: &[usize],
    ) -> (bool, f64) {
        let n = data.nrows() as f64;

        if z_indices.is_empty() {
            // Simple correlation
            let x = data.column(x_idx);
            let y = data.column(y_idx);
            let r = pearson_correlation(&x.to_owned(), &y.to_owned());
            let p_value = correlation_p_value(r, n as usize);
            return (p_value > self.alpha, p_value);
        }

        // Compute partial correlation via residuals
        let x = data.column(x_idx).to_owned();
        let y = data.column(y_idx).to_owned();

        // Collect conditioning variables
        let z: Array2<f64> = Array2::from_shape_fn((data.nrows(), z_indices.len()), |(i, j)| {
            data[[i, z_indices[j]]]
        });

        // Residualize X on Z
        let res_x = residualize(&x, &z);

        // Residualize Y on Z
        let res_y = residualize(&y, &z);

        // Correlation of residuals
        let r = pearson_correlation(&res_x, &res_y);

        // Fisher z-transform
        let z_stat = 0.5 * ((1.0 + r) / (1.0 - r + 1e-10)).ln();
        let se = 1.0 / (n - z_indices.len() as f64 - 3.0).sqrt();

        let normal = Normal::new(0.0, 1.0).unwrap();
        let p_value = 2.0 * (1.0 - normal.cdf((z_stat / se).abs()));

        (p_value > self.alpha, p_value)
    }
}

fn pearson_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.mean().unwrap();
    let mean_y = y.mean().unwrap();

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    cov / (var_x.sqrt() * var_y.sqrt() + 1e-10)
}

fn correlation_p_value(r: f64, n: usize) -> f64 {
    if n < 3 {
        return 1.0;
    }
    let t = r * ((n as f64 - 2.0) / (1.0 - r * r + 1e-10)).sqrt();
    let normal = Normal::new(0.0, 1.0).unwrap();
    2.0 * (1.0 - normal.cdf(t.abs()))
}

fn residualize(y: &Array1<f64>, x: &Array2<f64>) -> Array1<f64> {
    // Simple OLS: y = X * beta + residual
    // beta = (X'X)^-1 X'y

    let xt = x.t();
    let xtx = xt.dot(x);
    let xty = xt.dot(y);

    // Solve using simple method (for small dimensions)
    let beta = solve_linear_system(&xtx, &xty);

    let predicted = x.dot(&beta);
    y - &predicted
}

fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    // Simple Gauss-Jordan elimination for small matrices
    let n = a.nrows();
    let mut aug = Array2::zeros((n, n + 1));

    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        for j in 0..=n {
            let temp = aug[[i, j]];
            aug[[i, j]] = aug[[max_row, j]];
            aug[[max_row, j]] = temp;
        }

        // Eliminate
        for k in (i + 1)..n {
            if aug[[i, i]].abs() > 1e-10 {
                let factor = aug[[k, i]] / aug[[i, i]];
                for j in i..=n {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        x[i] = aug[[i, n]];
        for j in (i + 1)..n {
            x[i] -= aug[[i, j]] * x[j];
        }
        if aug[[i, i]].abs() > 1e-10 {
            x[i] /= aug[[i, i]];
        }
    }

    x
}
```

### PC Algorithm (Rust)

```rust
// src/discovery/pc.rs
use crate::graph::{CausalGraph, EdgeType};
use crate::discovery::ci_test::PartialCorrelationTest;
use ndarray::Array2;
use std::collections::{HashMap, HashSet};
use itertools::Itertools;

pub struct PCAlgorithm {
    alpha: f64,
    max_cond_size: usize,
    ci_test: PartialCorrelationTest,
    sep_sets: HashMap<(String, String), HashSet<String>>,
}

impl PCAlgorithm {
    pub fn new(alpha: f64, max_cond_size: usize) -> Self {
        PCAlgorithm {
            alpha,
            max_cond_size,
            ci_test: PartialCorrelationTest::new(alpha),
            sep_sets: HashMap::new(),
        }
    }

    pub fn fit(&mut self, data: &Array2<f64>, node_names: &[String]) -> CausalGraph {
        let mut graph = CausalGraph::new(node_names.to_vec());
        graph.add_complete_undirected();

        // Create node name to index mapping
        let name_to_idx: HashMap<&str, usize> = node_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();

        // Phase 1: Learn skeleton
        self.learn_skeleton(data, &mut graph, &name_to_idx);

        // Phase 2: Orient v-structures
        self.orient_v_structures(&mut graph);

        // Phase 3: Apply Meek rules
        self.apply_meek_rules(&mut graph);

        graph
    }

    fn learn_skeleton(
        &mut self,
        data: &Array2<f64>,
        graph: &mut CausalGraph,
        name_to_idx: &HashMap<&str, usize>,
    ) {
        for cond_size in 0..=self.max_cond_size {
            let edges: Vec<(String, String)> = graph.edges.keys().cloned().collect();

            for (x, y) in edges {
                if !graph.has_edge(&x, &y) {
                    continue;
                }

                let neighbors_x = graph.get_neighbors(&x);
                let neighbors_y = graph.get_neighbors(&y);
                let mut potential_cond: HashSet<String> = neighbors_x.union(&neighbors_y).cloned().collect();
                potential_cond.remove(&x);
                potential_cond.remove(&y);

                if potential_cond.len() < cond_size {
                    continue;
                }

                let potential_vec: Vec<String> = potential_cond.into_iter().collect();

                for cond_set in potential_vec.iter().combinations(cond_size) {
                    let cond_indices: Vec<usize> = cond_set
                        .iter()
                        .filter_map(|n| name_to_idx.get(n.as_str()).copied())
                        .collect();

                    let x_idx = name_to_idx[x.as_str()];
                    let y_idx = name_to_idx[y.as_str()];

                    let (is_independent, _p_val) = self.ci_test.test(data, x_idx, y_idx, &cond_indices);

                    if is_independent {
                        graph.remove_edge(&x, &y);
                        let sep: HashSet<String> = cond_set.iter().map(|s| (*s).clone()).collect();
                        self.sep_sets.insert((x.clone(), y.clone()), sep.clone());
                        self.sep_sets.insert((y.clone(), x.clone()), sep);
                        break;
                    }
                }
            }
        }
    }

    fn orient_v_structures(&mut self, graph: &mut CausalGraph) {
        let nodes = graph.nodes.clone();

        for z in &nodes {
            let neighbors: Vec<String> = graph.get_neighbors(z).into_iter().collect();

            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let x = &neighbors[i];
                    let y = &neighbors[j];

                    // Check if x and y are non-adjacent
                    if !graph.has_edge(x, y) {
                        // Check if z is NOT in separation set
                        let sep_set = self.sep_sets
                            .get(&(x.clone(), y.clone()))
                            .cloned()
                            .unwrap_or_default();

                        if !sep_set.contains(z) {
                            // Orient as v-structure: X → Z ← Y
                            graph.remove_edge(x, z);
                            graph.remove_edge(y, z);
                            graph.add_edge(x, z, EdgeType::Directed);
                            graph.add_edge(y, z, EdgeType::Directed);
                        }
                    }
                }
            }
        }
    }

    fn apply_meek_rules(&self, graph: &mut CausalGraph) {
        let mut changed = true;

        while changed {
            changed = false;
            let edges: Vec<((String, String), EdgeType)> = graph.edges.iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();

            for ((x, y), edge_type) in edges {
                if edge_type != EdgeType::Undirected {
                    continue;
                }

                // Rule 1: If there exists P → Y and P, X non-adjacent, orient X → Y
                let parents_y = graph.get_parents(&y);
                for p in parents_y {
                    if !graph.has_edge(&p, &x) {
                        graph.remove_edge(&x, &y);
                        graph.add_edge(&x, &y, EdgeType::Directed);
                        changed = true;
                        break;
                    }
                }
            }
        }
    }
}
```

### Causal Effect Estimation (Rust)

```rust
// src/estimation/backdoor.rs
use crate::graph::CausalGraph;
use ndarray::{Array1, Array2};
use std::collections::HashSet;

pub struct CausalEffectEstimator<'a> {
    graph: &'a CausalGraph,
}

#[derive(Debug, Clone)]
pub struct CausalEffect {
    pub ate: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub adjustment_set: Vec<String>,
}

impl<'a> CausalEffectEstimator<'a> {
    pub fn new(graph: &'a CausalGraph) -> Self {
        CausalEffectEstimator { graph }
    }

    pub fn get_backdoor_adjustment_set(&self, treatment: &str) -> HashSet<String> {
        // Simple: use parents of treatment
        self.graph.get_parents(treatment).into_iter().collect()
    }

    pub fn estimate_ate(
        &self,
        data: &Array2<f64>,
        node_names: &[String],
        treatment: &str,
        outcome: &str,
    ) -> Option<CausalEffect> {
        let adjustment_set = self.get_backdoor_adjustment_set(treatment);

        let treatment_idx = node_names.iter().position(|n| n == treatment)?;
        let outcome_idx = node_names.iter().position(|n| n == outcome)?;

        let adjustment_indices: Vec<usize> = adjustment_set
            .iter()
            .filter_map(|n| node_names.iter().position(|name| name == n))
            .collect();

        let n = data.nrows();

        // Build design matrix
        let n_features = 1 + adjustment_indices.len();
        let mut x = Array2::zeros((n, n_features));
        let y = data.column(outcome_idx).to_owned();

        for i in 0..n {
            x[[i, 0]] = data[[i, treatment_idx]];
            for (j, &adj_idx) in adjustment_indices.iter().enumerate() {
                x[[i, j + 1]] = data[[i, adj_idx]];
            }
        }

        // OLS estimation
        let beta = ols_regression(&x, &y)?;
        let ate = beta[0];

        // Bootstrap confidence interval
        let mut bootstrap_ates = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let indices: Vec<usize> = (0..n).map(|_| rand_usize(n)).collect();

            let mut x_boot = Array2::zeros((n, n_features));
            let mut y_boot = Array1::zeros(n);

            for (i, &idx) in indices.iter().enumerate() {
                for j in 0..n_features {
                    x_boot[[i, j]] = x[[idx, j]];
                }
                y_boot[i] = y[idx];
            }

            if let Some(beta_boot) = ols_regression(&x_boot, &y_boot) {
                bootstrap_ates.push(beta_boot[0]);
            }
        }

        bootstrap_ates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ci_low = bootstrap_ates.get(25).copied().unwrap_or(ate);
        let ci_high = bootstrap_ates.get(975).copied().unwrap_or(ate);

        Some(CausalEffect {
            ate,
            ci_low,
            ci_high,
            adjustment_set: adjustment_set.into_iter().collect(),
        })
    }
}

fn ols_regression(x: &Array2<f64>, y: &Array1<f64>) -> Option<Array1<f64>> {
    let xt = x.t();
    let xtx = xt.dot(x);
    let xty = xt.dot(y);

    // Simple solve (for small matrices)
    let n = xtx.nrows();
    let mut aug = Array2::zeros((n, n + 1));

    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = xtx[[i, j]];
        }
        aug[[i, n]] = xty[i];
    }

    // Gauss-Jordan
    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                max_row = k;
            }
        }

        for j in 0..=n {
            let temp = aug[[i, j]];
            aug[[i, j]] = aug[[max_row, j]];
            aug[[max_row, j]] = temp;
        }

        if aug[[i, i]].abs() < 1e-10 {
            return None;
        }

        for k in (i + 1)..n {
            let factor = aug[[k, i]] / aug[[i, i]];
            for j in i..=n {
                aug[[k, j]] -= factor * aug[[i, j]];
            }
        }
    }

    let mut beta = Array1::zeros(n);
    for i in (0..n).rev() {
        beta[i] = aug[[i, n]];
        for j in (i + 1)..n {
            beta[i] -= aug[[i, j]] * beta[j];
        }
        beta[i] /= aug[[i, i]];
    }

    Some(beta)
}

fn rand_usize(max: usize) -> usize {
    // Simple LCG random (for demonstration)
    use std::time::{SystemTime, UNIX_EPOCH};
    static mut SEED: u64 = 0;
    unsafe {
        if SEED == 0 {
            SEED = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
        }
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED as usize) % max
    }
}
```

### Bybit Data Fetcher (Rust)

```rust
// src/data/bybit.rs
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct BybitKline {
    pub open_time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Deserialize)]
struct BybitResponse {
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<BybitKline>, Box<dyn std::error::Error>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let resp: BybitResponse = reqwest::get(&url).await?.json().await?;

    let klines: Vec<BybitKline> = resp
        .result
        .list
        .iter()
        .filter_map(|arr| {
            Some(BybitKline {
                open_time: arr.get(0)?.parse().ok()?,
                open: arr.get(1)?.parse().ok()?,
                high: arr.get(2)?.parse().ok()?,
                low: arr.get(3)?.parse().ok()?,
                close: arr.get(4)?.parse().ok()?,
                volume: arr.get(5)?.parse().ok()?,
            })
        })
        .collect();

    Ok(klines)
}
```

### Trading Strategy (Rust)

```rust
// src/trading/strategy.rs
use crate::graph::CausalGraph;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Signal {
    Long,
    Short,
    Neutral,
}

pub struct CausalFactorStrategy {
    pub causal_factors: Vec<String>,
    pub factor_effects: HashMap<String, f64>,
    pub threshold: f64,
}

impl CausalFactorStrategy {
    pub fn new(
        causal_factors: Vec<String>,
        factor_effects: HashMap<String, f64>,
        threshold: f64,
    ) -> Self {
        CausalFactorStrategy {
            causal_factors,
            factor_effects,
            threshold,
        }
    }

    pub fn generate_signal(&self, features: &HashMap<String, f64>) -> Signal {
        let mut prediction = 0.0;

        for factor in &self.causal_factors {
            if let (Some(&value), Some(&effect)) = (features.get(factor), self.factor_effects.get(factor)) {
                prediction += value * effect;
            }
        }

        if prediction > self.threshold {
            Signal::Long
        } else if prediction < -self.threshold {
            Signal::Short
        } else {
            Signal::Neutral
        }
    }
}

#[derive(Debug)]
pub struct BacktestMetrics {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub num_trades: usize,
}

pub fn backtest_strategy(
    strategy: &CausalFactorStrategy,
    features_series: &[HashMap<String, f64>],
    returns: &[f64],
    transaction_cost: f64,
) -> BacktestMetrics {
    let mut signals = Vec::new();
    let mut strategy_returns = Vec::new();

    for (features, &ret) in features_series.iter().zip(returns.iter()) {
        let signal = strategy.generate_signal(features);
        let position = match signal {
            Signal::Long => 1.0,
            Signal::Short => -1.0,
            Signal::Neutral => 0.0,
        };
        signals.push(position);
    }

    // Compute returns with transaction costs
    let mut prev_position = 0.0;
    for (i, &position) in signals.iter().enumerate() {
        if i < returns.len() {
            let cost = (position - prev_position).abs() * transaction_cost;
            let ret = position * returns[i] - cost;
            strategy_returns.push(ret);
            prev_position = position;
        }
    }

    // Compute metrics
    let total_return: f64 = strategy_returns.iter()
        .map(|&r| (1.0 + r).ln())
        .sum::<f64>()
        .exp() - 1.0;

    let mean_return = strategy_returns.iter().sum::<f64>() / strategy_returns.len() as f64;
    let variance: f64 = strategy_returns.iter()
        .map(|&r| (r - mean_return).powi(2))
        .sum::<f64>() / strategy_returns.len() as f64;
    let std_dev = variance.sqrt();
    let sharpe_ratio = mean_return / (std_dev + 1e-10) * (252.0_f64).sqrt();

    // Max drawdown
    let mut cumulative = 1.0;
    let mut running_max = 1.0;
    let mut max_drawdown = 0.0;
    for &ret in &strategy_returns {
        cumulative *= 1.0 + ret;
        running_max = running_max.max(cumulative);
        let drawdown = (cumulative - running_max) / running_max;
        max_drawdown = max_drawdown.min(drawdown);
    }

    // Win rate
    let wins = strategy_returns.iter().filter(|&&r| r > 0.0).count();
    let total = strategy_returns.iter().filter(|&&r| r != 0.0).count();
    let win_rate = wins as f64 / (total as f64 + 1e-10);

    // Number of trades
    let mut num_trades = 0;
    let mut prev = 0.0;
    for &s in &signals {
        if (s - prev).abs() > 0.0 {
            num_trades += 1;
        }
        prev = s;
    }

    BacktestMetrics {
        total_return,
        sharpe_ratio,
        max_drawdown,
        win_rate,
        num_trades,
    }
}
```

---

## Practical Examples with Stock and Crypto Data

### Example 1: Stock Factor Discovery (Python)

```python
import yfinance as yf
import pandas as pd

# Download stock data
data = yf.download('AAPL', start='2018-01-01', end='2024-01-01')
data.columns = [c.lower() for c in data.columns]

# Compute factors
pipeline = FactorDataPipeline()
factors = pipeline.compute_factors(data)
returns = data['close'].pct_change().shift(-1).dropna()

# Align data
common_idx = factors.index.intersection(returns.index)
factors = factors.loc[common_idx]
returns = returns.loc[common_idx]

# Discover causal factors
model = CausalFactorModel(alpha=0.05, min_effect_size=0.001)
model.fit(factors, returns)

print("Discovered Causal Factors:")
print(model.get_factor_summary())

# Backtest
backtester = CausalFactorBacktester(model, threshold=0.001)
result = backtester.run(factors, returns)

print(f"\nBacktest Results:")
print(f"Annual Return: {result.annual_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
print(f"Win Rate: {result.win_rate:.2%}")
```

### Example 2: Crypto Causal Analysis (Python)

```python
# Fetch BTC and ETH data
btc = fetch_bybit_data("BTCUSDT", "D", 1000)
eth = fetch_bybit_data("ETHUSDT", "D", 1000)

# Compute factors for BTC
pipeline = FactorDataPipeline()
btc_factors = pipeline.compute_factors(btc)
btc_returns = btc['close'].pct_change().shift(-1).dropna()

# Align
common_idx = btc_factors.index.intersection(btc_returns.index)
btc_factors = btc_factors.loc[common_idx]
btc_returns = btc_returns.loc[common_idx]

# Discover causal factors
btc_model = CausalFactorModel(alpha=0.1, min_effect_size=0.005)
btc_model.fit(btc_factors, btc_returns)

print("BTC Causal Factors:")
print(btc_model.get_factor_summary())

# Test on ETH (cross-asset validation)
eth_factors = pipeline.compute_factors(eth)
eth_returns = eth['close'].pct_change().shift(-1).dropna()
common_idx = eth_factors.index.intersection(eth_returns.index)
eth_factors = eth_factors.loc[common_idx]
eth_returns = eth_returns.loc[common_idx]

# Apply BTC model to ETH
predictions = btc_model.predict(eth_factors)
correlation = pd.Series(predictions).corr(eth_returns)
print(f"\nBTC model applied to ETH - Prediction correlation: {correlation:.3f}")
```

### Example 3: Rust Causal Discovery

```rust
// examples/discover_factors.rs
use causal_factor_discovery::discovery::pc::PCAlgorithm;
use causal_factor_discovery::estimation::backdoor::CausalEffectEstimator;
use causal_factor_discovery::data::bybit::fetch_bybit_klines;
use ndarray::Array2;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Fetch BTC data
    let klines = fetch_bybit_klines("BTCUSDT", "D", 200).await?;
    println!("Fetched {} klines", klines.len());

    // Compute features
    let mut data_vec = Vec::new();
    let mut prev_close = klines[0].close;

    for kline in &klines[1..] {
        let log_ret = (kline.close / prev_close).ln();
        let vol_ratio = kline.volume / 1e6; // Normalized
        let range = (kline.high - kline.low) / kline.close;
        let body = (kline.close - kline.open) / kline.close;

        data_vec.push(vec![log_ret, vol_ratio, range, body]);
        prev_close = kline.close;
    }

    // Add next return as target
    let n = data_vec.len() - 1;
    let mut features_with_target = Vec::new();
    for i in 0..n {
        let mut row = data_vec[i].clone();
        row.push(data_vec[i + 1][0]); // Next return
        features_with_target.push(row);
    }

    // Create ndarray
    let data = Array2::from_shape_vec(
        (features_with_target.len(), 5),
        features_with_target.into_iter().flatten().collect()
    )?;

    let node_names: Vec<String> = vec![
        "log_return".to_string(),
        "volume".to_string(),
        "range".to_string(),
        "body".to_string(),
        "next_return".to_string(),
    ];

    // Run PC algorithm
    let mut pc = PCAlgorithm::new(0.05, 2);
    let graph = pc.fit(&data, &node_names);

    println!("\nDiscovered Causal Graph:");
    for ((s, t), e) in &graph.edges {
        println!("  {} -> {} ({:?})", s, t, e);
    }

    // Estimate causal effects
    let estimator = CausalEffectEstimator::new(&graph);

    for factor in &["log_return", "volume", "range", "body"] {
        if let Some(effect) = estimator.estimate_ate(&data, &node_names, factor, "next_return") {
            println!(
                "\nCausal effect of {} on next_return: {:.4} [{:.4}, {:.4}]",
                factor, effect.ate, effect.ci_low, effect.ci_high
            );
        }
    }

    Ok(())
}
```

---

## Backtesting Framework

### Walk-Forward Validation

```python
def walk_forward_causal_backtest(
    data: pd.DataFrame,
    train_period: int = 504,  # 2 years
    test_period: int = 63,    # 3 months
    retrain_every: int = 21,  # Monthly
):
    """Walk-forward backtest with causal factor discovery."""

    pipeline = FactorDataPipeline()
    results = []

    for i in range(train_period, len(data) - test_period, retrain_every):
        # Training data
        train_data = data.iloc[i - train_period:i]
        train_factors = pipeline.compute_factors(train_data)
        train_returns = train_data['close'].pct_change().shift(-1).dropna()

        # Align
        common_idx = train_factors.index.intersection(train_returns.index)
        train_factors = train_factors.loc[common_idx]
        train_returns = train_returns.loc[common_idx]

        # Discover causal factors
        model = CausalFactorModel(alpha=0.05, min_effect_size=0.001)
        model.fit(train_factors, train_returns)

        # Test data
        test_data = data.iloc[i:i + test_period]
        test_factors = pipeline.compute_factors(test_data)
        test_returns = test_data['close'].pct_change().shift(-1).dropna()

        common_idx = test_factors.index.intersection(test_returns.index)
        test_factors = test_factors.loc[common_idx]
        test_returns = test_returns.loc[common_idx]

        # Generate predictions
        predictions = model.predict(test_factors)

        # Evaluate
        correlation = pd.Series(predictions).corr(test_returns)
        ic = np.corrcoef(predictions[:-1], test_returns.values[:-1])[0, 1]

        results.append({
            'period_start': data.index[i],
            'n_causal_factors': len(model.causal_factors),
            'factors': model.causal_factors,
            'correlation': correlation,
            'ic': ic
        })

    return pd.DataFrame(results)
```

### Comparison with Correlation-Based Models

```python
def compare_causal_vs_correlation(data: pd.DataFrame):
    """Compare causal factor model vs. correlation-based model."""

    pipeline = FactorDataPipeline()
    factors = pipeline.compute_factors(data)
    returns = data['close'].pct_change().shift(-1).dropna()

    common_idx = factors.index.intersection(returns.index)
    factors = factors.loc[common_idx]
    returns = returns.loc[common_idx]

    # Split data
    train_size = int(0.7 * len(factors))
    train_factors, test_factors = factors.iloc[:train_size], factors.iloc[train_size:]
    train_returns, test_returns = returns.iloc[:train_size], returns.iloc[train_size:]

    # Causal model
    causal_model = CausalFactorModel(alpha=0.05)
    causal_model.fit(train_factors, train_returns)
    causal_pred = causal_model.predict(test_factors)
    causal_corr = np.corrcoef(causal_pred, test_returns.values)[0, 1]

    # Correlation-based model (select top correlated factors)
    correlations = train_factors.apply(lambda x: x.corr(train_returns))
    top_factors = correlations.abs().nlargest(5).index.tolist()

    from sklearn.linear_model import LinearRegression
    corr_model = LinearRegression().fit(train_factors[top_factors], train_returns)
    corr_pred = corr_model.predict(test_factors[top_factors])
    corr_corr = np.corrcoef(corr_pred, test_returns.values)[0, 1]

    print(f"Causal Model - Test correlation: {causal_corr:.4f}")
    print(f"Causal factors: {causal_model.causal_factors}")
    print(f"\nCorrelation Model - Test correlation: {corr_corr:.4f}")
    print(f"Selected factors: {top_factors}")

    return {
        'causal_corr': causal_corr,
        'corr_corr': corr_corr,
        'causal_factors': causal_model.causal_factors,
        'corr_factors': top_factors
    }
```

---

## Performance Evaluation

### Metrics Summary

| Metric | Description | Target |
|--------|-------------|--------|
| Information Coefficient (IC) | Correlation between predictions and returns | > 0.03 |
| IC Information Ratio | IC mean / IC std | > 0.5 |
| Sharpe Ratio | Risk-adjusted return | > 1.0 |
| Sortino Ratio | Downside risk-adjusted return | > 1.5 |
| Max Drawdown | Largest peak-to-trough decline | > -20% |
| Factor Stability | % of factors retained across retraining | > 50% |

### Advantages of Causal Factor Discovery

1. **Robustness**: Causal factors should remain predictive across market regimes
2. **Interpretability**: Causal relationships have clear economic meaning
3. **Avoids Overfitting**: Fewer spurious factors selected
4. **Transfer Learning**: Causal factors may transfer across assets/markets

### Key Considerations

1. **Sample Size**: Causal discovery requires substantial data for reliable CI tests
2. **Faithfulness Violations**: Real markets may violate faithfulness assumptions
3. **Non-stationarity**: Causal structures may change over time
4. **Hidden Confounders**: Unobserved factors can bias causal estimates

---

## Future Directions

1. **Time-Varying Causal Structures**: Detect regime changes in causal relationships
2. **Deep Causal Discovery**: Neural network-based causal discovery for nonlinear relationships
3. **Counterfactual Trading**: "What-if" analysis using causal models
4. **Multi-Asset Causal Models**: Cross-asset causal relationships for portfolio construction
5. **Real-Time Causal Monitoring**: Online causal discovery for adaptive strategies
6. **Causal Reinforcement Learning**: Combine causal inference with RL for trading

---

## References

1. Pearl, J. (2009). Causality: Models, Reasoning, and Inference. Cambridge University Press.
2. Spirtes, P., Glymour, C., & Scheines, R. (2000). Causation, Prediction, and Search. MIT Press.
3. Peters, J., Janzing, D., & Schölkopf, B. (2017). Elements of Causal Inference. MIT Press.
4. Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous Optimization for Structure Learning. NeurIPS 2018.
5. Vowels, M. J., Camgoz, N. C., & Mayol-Cuevas, R. (2022). D'ya Like DAGs? A Survey on Structure Learning and Causal Discovery. ACM Computing Surveys.
6. Nauta, M., Bucur, D., & Seifert, C. (2019). Causal Discovery with Attention-Based Convolutional Neural Networks. Machine Learning and Knowledge Extraction.
7. Runge, J., et al. (2019). Detecting and Quantifying Causal Associations in Large Nonlinear Time Series Datasets. Science Advances.
8. Hoyer, P. O., et al. (2008). Nonlinear Causal Discovery with Additive Noise Models. NeurIPS 2008.

---

## Running the Examples

### Python

```bash
cd 109_causal_factor_discovery/python
pip install -r requirements.txt
python model.py           # Test causal discovery
python backtest.py        # Run backtesting
```

### Rust

```bash
cd 109_causal_factor_discovery
cargo build
cargo run --example discover_factors
cargo run --example crypto_causal
cargo run --example backtest_strategy
```
