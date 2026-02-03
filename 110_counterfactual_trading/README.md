# Chapter 110: Counterfactual Trading

## Overview

Counterfactual Trading is an advanced approach to algorithmic trading that leverages counterfactual reasoning from causal inference to evaluate trading decisions. Unlike traditional backtesting that only considers what happened, counterfactual analysis asks: "What WOULD have happened if I had taken a different action?"

This methodology enables traders to:
- Evaluate the true impact of trading decisions
- Identify optimal actions in hindsight with proper causal adjustment
- Build more robust trading strategies by understanding causal mechanisms
- Avoid confounding factors that plague traditional performance attribution

The key insight is that we can estimate counterfactual outcomes using structural causal models, allowing us to answer questions like "What would my P&L have been if I had NOT executed this trade?" or "What if I had used a different position size?"

## Table of Contents

1. [Introduction to Counterfactual Reasoning](#introduction-to-counterfactual-reasoning)
2. [Counterfactuals in Trading Context](#counterfactuals-in-trading-context)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Counterfactual Estimation Methods](#counterfactual-estimation-methods)
5. [Applications to Trading](#applications-to-trading)
6. [Implementation in Python](#implementation-in-python)
7. [Implementation in Rust](#implementation-in-rust)
8. [Practical Examples with Stock and Crypto Data](#practical-examples-with-stock-and-crypto-data)
9. [Backtesting Framework](#backtesting-framework)
10. [Performance Evaluation](#performance-evaluation)
11. [Future Directions](#future-directions)
12. [References](#references)

---

## Introduction to Counterfactual Reasoning

### What Are Counterfactuals?

A counterfactual is a "what if" statement about an alternative scenario that didn't actually happen:

- **Factual**: "I bought stock X and made $100"
- **Counterfactual**: "What would I have made if I had NOT bought stock X?"

```
Timeline:
─────────────────────────────────────────────────────────
    t=0              t=1              t=2
    │                │                │
    ▼                ▼                ▼
 Decision:       Outcome:         Final:
 Buy Stock X     Price +5%        P&L = +$100

Counterfactual World:
─────────────────────────────────────────────────────────
    t=0              t=1              t=2
    │                │                │
    ▼                ▼                ▼
 Decision:       Outcome:         Final:
 Don't Buy       Price +5%        P&L = $0
                 (same price)     (no position)
```

### The Fundamental Problem of Causal Inference

We can never observe both outcomes for the same unit at the same time — this is called the "fundamental problem of causal inference." In trading:

- If you executed a trade, you can't know what would have happened without it
- If you didn't trade, you can't know what would have happened if you did

### Potential Outcomes Framework

The Neyman-Rubin potential outcomes framework formalizes counterfactuals:

- **Y(1)**: Outcome if treated (e.g., executed trade)
- **Y(0)**: Outcome if not treated (e.g., no trade)
- **Individual Treatment Effect (ITE)**: τᵢ = Yᵢ(1) - Yᵢ(0)

We observe:
```
Yᵢ = Tᵢ · Yᵢ(1) + (1 - Tᵢ) · Yᵢ(0)

where Tᵢ ∈ {0, 1} is the treatment indicator
```

The challenge: We only observe one potential outcome per unit!

---

## Counterfactuals in Trading Context

### Trading as Treatment Assignment

In trading, we can frame decisions as treatment assignments:

| Trading Concept | Causal Concept |
|-----------------|----------------|
| Trade execution | Treatment |
| No trade | Control |
| Returns | Outcome |
| Market conditions | Covariates |
| Trading strategy | Treatment policy |

### Key Counterfactual Questions in Trading

1. **Trade Attribution**: "How much of my return was due to THIS specific trade vs. market movement?"

2. **Strategy Evaluation**: "What would my portfolio have returned if I had used Strategy B instead of Strategy A?"

3. **Position Sizing**: "What if I had doubled my position size?"

4. **Timing**: "What if I had entered one day earlier/later?"

5. **Risk Management**: "What would have happened if I had not used stop-losses?"

### Why Traditional Backtesting Fails

Traditional backtesting suffers from:

```
Problem 1: Confounding
─────────────────────────────────────────────
Market Sentiment
      │
      ├──────────────┐
      ▼              ▼
  My Trade        Returns

"My trade caused positive returns"
Reality: Market sentiment caused BOTH!


Problem 2: Selection Bias
─────────────────────────────────────────────
Only evaluate trades we actually made
→ Miss information about trades we didn't make
→ Biased performance estimates


Problem 3: Survivorship Bias
─────────────────────────────────────────────
Strategy looks good because bad versions
were abandoned → Overstated performance
```

---

## Mathematical Foundation

### Structural Causal Models (SCMs)

An SCM M = (U, V, F) consists of:
- **U**: Exogenous (external) variables
- **V**: Endogenous (internal) variables
- **F**: Structural equations Vᵢ = fᵢ(PAᵢ, Uᵢ)

### The Three Levels of Causal Hierarchy

Pearl's Causal Hierarchy (Ladder of Causation):

```
Level 3: COUNTERFACTUALS (Imagining)
─────────────────────────────────────────────
"What would Y have been if X had been different?"
P(Yₓ | X=x', Y=y')
Requires: Full SCM specification


Level 2: INTERVENTIONS (Doing)
─────────────────────────────────────────────
"What happens if I do X?"
P(Y | do(X))
Requires: Causal graph + data


Level 1: ASSOCIATIONS (Seeing)
─────────────────────────────────────────────
"What if I see X?"
P(Y | X)
Requires: Data only
```

### Computing Counterfactuals: Three Steps

**Step 1: Abduction** — Use evidence to determine values of exogenous variables U

```
Given: X = x, Y = y (observed)
Find: U such that fₓ(U) = x and f_y(X, U) = y
```

**Step 2: Action** — Modify the model according to the intervention

```
Replace: X = fₓ(U) with X = x' (counterfactual value)
```

**Step 3: Prediction** — Compute the counterfactual outcome

```
Calculate: Y_{X=x'} = f_y(x', U)
```

### Counterfactual Formulas

For a linear SCM:
```
Y = αX + βZ + U_Y
X = γZ + U_X

Counterfactual Y_{X=x'} given observed (X=x, Y=y, Z=z):

Step 1 (Abduction):
U_Y = y - αx - βz

Step 2 (Action):
Set X = x'

Step 3 (Prediction):
Y_{X=x'} = αx' + βz + U_Y
         = αx' + βz + (y - αx - βz)
         = y + α(x' - x)
```

---

## Counterfactual Estimation Methods

### Method 1: Twin Networks

Create a "twin" of the observed unit in the counterfactual world:

```
Observed World:           Counterfactual World:
     U                         U (same!)
     │                         │
     ▼                         ▼
     X ─────→ Y               X' ─────→ Y'
     │
 (observed)                (counterfactual)
```

The key insight: Exogenous variables U are shared between worlds.

### Method 2: Matching

Find similar units that received different treatments:

```python
# Propensity Score Matching for counterfactuals
def estimate_counterfactual(unit, treatment_value, data):
    # Find similar units with opposite treatment
    similar_units = find_similar(unit, data)
    opposite_treated = similar_units[similar_units.treatment == treatment_value]

    # Estimate counterfactual as weighted average
    counterfactual = weighted_average(opposite_treated.outcome)
    return counterfactual
```

### Method 3: Outcome Regression

Model the outcome as a function of treatment and covariates:

```
Y = f(T, X) + ε

Counterfactual:
Ŷ(t') = f(t', X) for any treatment value t'
```

### Method 4: Doubly Robust Estimation

Combines propensity scores and outcome regression:

```
τ̂_DR = (1/n) Σ [(Tᵢ·Yᵢ)/e(Xᵢ) - ((Tᵢ - e(Xᵢ))/e(Xᵢ))·μ₁(Xᵢ)]
      - (1/n) Σ [((1-Tᵢ)·Yᵢ)/(1-e(Xᵢ)) + ((Tᵢ - e(Xᵢ))/(1-e(Xᵢ)))·μ₀(Xᵢ)]

where:
- e(X) = P(T=1|X) is the propensity score
- μ₁(X) = E[Y|T=1, X] is outcome model for treated
- μ₀(X) = E[Y|T=0, X] is outcome model for control
```

---

## Applications to Trading

### 1. Counterfactual Trade Attribution

Decompose returns into causal components:

```
Total Return = Market Effect + Strategy Effect + Residual

Where:
- Market Effect = E[Y | do(Market), no trade]
- Strategy Effect = Y_{observed} - Y_{counterfactual: no trade}
- Residual = Unexplained variation
```

### 2. Optimal Policy Learning

Use counterfactual outcomes to learn optimal trading policies:

```
π*(x) = argmax_a E[Y(a) | X = x]

Estimated via:
π̂*(x) = argmax_a Σᵢ ω(Xᵢ, x, Aᵢ, a) · Yᵢ
```

### 3. What-If Analysis

Evaluate alternative strategies:

```
Scenario: "What if I had used 2x leverage?"

Counterfactual Model:
Position_cf = 2 × Position_observed
Return_cf = Position_cf × Price_change - Cost_cf

Compare: Return_cf vs Return_observed
```

### 4. Regret Minimization

Minimize counterfactual regret:

```
Regret(t) = max_a Y_t(a) - Y_t(A_t)

where:
- Y_t(a) = counterfactual return under action a
- A_t = action actually taken
```

---

## Implementation in Python

### Counterfactual Estimator

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors


@dataclass
class CounterfactualResult:
    """Result of counterfactual estimation."""
    observed_outcome: float
    counterfactual_outcome: float
    treatment_effect: float
    confidence_interval: Tuple[float, float]
    method: str


class CounterfactualEstimator:
    """
    Estimates counterfactual outcomes for trading decisions.

    Supports multiple estimation methods:
    - Outcome regression
    - Propensity score matching
    - Doubly robust estimation
    """

    def __init__(self, method: str = 'doubly_robust'):
        self.method = method
        self.outcome_model = None
        self.propensity_model = None
        self.fitted = False

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray):
        """
        Fit the counterfactual model.

        Args:
            X: Covariates (market features)
            treatment: Treatment indicator (1 = traded, 0 = no trade)
            outcome: Observed outcomes (returns)
        """
        # Fit outcome model
        self.outcome_model_treated = LinearRegression()
        self.outcome_model_control = LinearRegression()

        treated_mask = treatment == 1
        control_mask = treatment == 0

        if np.sum(treated_mask) > 0:
            self.outcome_model_treated.fit(X[treated_mask], outcome[treated_mask])
        if np.sum(control_mask) > 0:
            self.outcome_model_control.fit(X[control_mask], outcome[control_mask])

        # Fit propensity model
        self.propensity_model = LogisticRegression(max_iter=1000)
        self.propensity_model.fit(X, treatment)

        self.fitted = True
        return self

    def estimate_counterfactual(
        self,
        X: np.ndarray,
        treatment: int,
        observed_outcome: float
    ) -> CounterfactualResult:
        """
        Estimate what the outcome would have been under opposite treatment.

        Args:
            X: Covariates for this unit
            treatment: Actual treatment received (0 or 1)
            observed_outcome: Actual observed outcome

        Returns:
            CounterfactualResult with counterfactual outcome
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        X = X.reshape(1, -1) if X.ndim == 1 else X

        # Estimate counterfactual outcome
        if treatment == 1:
            # Unit was treated, estimate control outcome
            cf_outcome = self.outcome_model_control.predict(X)[0]
        else:
            # Unit was not treated, estimate treated outcome
            cf_outcome = self.outcome_model_treated.predict(X)[0]

        # Treatment effect
        if treatment == 1:
            effect = observed_outcome - cf_outcome
        else:
            effect = cf_outcome - observed_outcome

        # Bootstrap confidence interval
        ci_low, ci_high = self._bootstrap_ci(X, treatment, observed_outcome)

        return CounterfactualResult(
            observed_outcome=observed_outcome,
            counterfactual_outcome=cf_outcome,
            treatment_effect=effect,
            confidence_interval=(ci_low, ci_high),
            method=self.method
        )

    def _bootstrap_ci(
        self,
        X: np.ndarray,
        treatment: int,
        observed_outcome: float,
        n_bootstrap: int = 1000,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for treatment effect."""
        effects = []

        for _ in range(n_bootstrap):
            # Add noise to estimate uncertainty
            noise = np.random.normal(0, 0.01)
            if treatment == 1:
                cf = self.outcome_model_control.predict(X)[0] + noise
                effect = observed_outcome - cf
            else:
                cf = self.outcome_model_treated.predict(X)[0] + noise
                effect = cf - observed_outcome
            effects.append(effect)

        return np.percentile(effects, [100*alpha/2, 100*(1-alpha/2)])

    def estimate_ate(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray
    ) -> Dict[str, float]:
        """
        Estimate Average Treatment Effect using doubly robust estimation.
        """
        n = len(outcome)

        # Propensity scores
        propensity = self.propensity_model.predict_proba(X)[:, 1]
        propensity = np.clip(propensity, 0.01, 0.99)  # Avoid extreme weights

        # Outcome predictions
        mu1 = self.outcome_model_treated.predict(X)
        mu0 = self.outcome_model_control.predict(X)

        # Doubly robust estimator
        treated_term = (treatment * outcome / propensity -
                       (treatment - propensity) / propensity * mu1)
        control_term = ((1 - treatment) * outcome / (1 - propensity) +
                       (treatment - propensity) / (1 - propensity) * mu0)

        ate = np.mean(treated_term) - np.mean(control_term)

        # Standard error via influence function
        influence = treated_term - control_term - ate
        se = np.std(influence) / np.sqrt(n)

        return {
            'ate': ate,
            'se': se,
            'ci_low': ate - 1.96 * se,
            'ci_high': ate + 1.96 * se
        }


class TwinNetworkEstimator:
    """
    Twin Network approach for counterfactual estimation.
    Uses the same exogenous noise for counterfactual prediction.
    """

    def __init__(self):
        self.structural_model = None
        self.noise_distribution = None

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray):
        """Fit the structural causal model."""
        # Fit outcome model: Y = f(T, X) + U
        features = np.column_stack([treatment, X])
        self.structural_model = LinearRegression()
        self.structural_model.fit(features, outcome)

        # Estimate noise distribution
        predictions = self.structural_model.predict(features)
        residuals = outcome - predictions
        self.noise_mean = np.mean(residuals)
        self.noise_std = np.std(residuals)

        return self

    def estimate_counterfactual(
        self,
        X: np.ndarray,
        treatment: int,
        observed_outcome: float
    ) -> float:
        """
        Estimate counterfactual using twin network approach.

        The key insight: we infer the noise term U from the observed outcome,
        then use the SAME noise for the counterfactual prediction.
        """
        X = X.reshape(1, -1) if X.ndim == 1 else X

        # Step 1: Abduction - infer noise term
        features_observed = np.column_stack([[treatment], X])
        predicted_observed = self.structural_model.predict(features_observed)[0]
        noise_u = observed_outcome - predicted_observed

        # Step 2: Action - set counterfactual treatment
        cf_treatment = 1 - treatment

        # Step 3: Prediction - compute counterfactual with same noise
        features_cf = np.column_stack([[cf_treatment], X])
        cf_outcome = self.structural_model.predict(features_cf)[0] + noise_u

        return cf_outcome


class PropensityScoreMatching:
    """
    Matching-based counterfactual estimation.
    Finds similar units with opposite treatment.
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.treated_nn = None
        self.control_nn = None
        self.data = None

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray):
        """Fit nearest neighbor models for matching."""
        self.data = {
            'X': X,
            'treatment': treatment,
            'outcome': outcome
        }

        treated_mask = treatment == 1
        control_mask = treatment == 0

        self.treated_nn = NearestNeighbors(n_neighbors=min(self.n_neighbors, np.sum(treated_mask)))
        self.control_nn = NearestNeighbors(n_neighbors=min(self.n_neighbors, np.sum(control_mask)))

        if np.sum(treated_mask) > 0:
            self.treated_nn.fit(X[treated_mask])
            self.treated_indices = np.where(treated_mask)[0]
        if np.sum(control_mask) > 0:
            self.control_nn.fit(X[control_mask])
            self.control_indices = np.where(control_mask)[0]

        return self

    def estimate_counterfactual(
        self,
        X: np.ndarray,
        treatment: int,
        observed_outcome: float
    ) -> float:
        """Estimate counterfactual by matching to opposite-treated units."""
        X = X.reshape(1, -1) if X.ndim == 1 else X

        if treatment == 1:
            # Find similar control units
            distances, indices = self.control_nn.kneighbors(X)
            matched_indices = self.control_indices[indices[0]]
        else:
            # Find similar treated units
            distances, indices = self.treated_nn.kneighbors(X)
            matched_indices = self.treated_indices[indices[0]]

        # Inverse distance weighting
        weights = 1 / (distances[0] + 1e-6)
        weights = weights / np.sum(weights)

        cf_outcome = np.sum(weights * self.data['outcome'][matched_indices])
        return cf_outcome
```

### Counterfactual Trading Strategy

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TradeDecision:
    """Represents a trading decision with counterfactual analysis."""
    timestamp: pd.Timestamp
    action: int  # 1 = buy, -1 = sell, 0 = hold
    observed_return: float
    counterfactual_return: float
    treatment_effect: float
    confidence: float


class CounterfactualTradingStrategy:
    """
    Trading strategy that uses counterfactual reasoning for:
    1. Evaluating past decisions
    2. Making better future decisions
    3. Understanding true strategy performance
    """

    def __init__(
        self,
        base_strategy,
        counterfactual_estimator: CounterfactualEstimator,
        lookback: int = 100
    ):
        self.base_strategy = base_strategy
        self.cf_estimator = counterfactual_estimator
        self.lookback = lookback
        self.decision_history: List[TradeDecision] = []

    def compute_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute market features for counterfactual estimation."""
        features = pd.DataFrame(index=prices.index)

        close = prices['close']

        # Price-based features
        features['return_1d'] = close.pct_change(1)
        features['return_5d'] = close.pct_change(5)
        features['return_20d'] = close.pct_change(20)

        # Volatility
        features['volatility'] = features['return_1d'].rolling(20).std()

        # Momentum
        features['momentum'] = close / close.rolling(20).mean() - 1

        # Volume (if available)
        if 'volume' in prices.columns:
            features['volume_ma_ratio'] = prices['volume'] / prices['volume'].rolling(20).mean()

        return features.dropna()

    def evaluate_decision(
        self,
        features: np.ndarray,
        action: int,
        observed_return: float
    ) -> TradeDecision:
        """
        Evaluate a trading decision using counterfactual analysis.

        Args:
            features: Market features at decision time
            action: Action taken (1, -1, or 0)
            observed_return: Actual return achieved
        """
        # Treatment: did we trade?
        treatment = 1 if action != 0 else 0

        # Estimate counterfactual
        cf_result = self.cf_estimator.estimate_counterfactual(
            features, treatment, observed_return
        )

        decision = TradeDecision(
            timestamp=pd.Timestamp.now(),
            action=action,
            observed_return=observed_return,
            counterfactual_return=cf_result.counterfactual_outcome,
            treatment_effect=cf_result.treatment_effect,
            confidence=1 - (cf_result.confidence_interval[1] - cf_result.confidence_interval[0])
        )

        self.decision_history.append(decision)
        return decision

    def compute_strategy_attribution(self) -> Dict[str, float]:
        """
        Decompose total returns into:
        - Market component (what we would have earned anyway)
        - Strategy component (added value from trading decisions)
        """
        if not self.decision_history:
            return {'market': 0, 'strategy': 0, 'total': 0}

        total_return = sum(d.observed_return for d in self.decision_history)
        cf_return = sum(d.counterfactual_return for d in self.decision_history)
        strategy_return = sum(d.treatment_effect for d in self.decision_history)

        return {
            'total_return': total_return,
            'market_component': cf_return,
            'strategy_component': strategy_return,
            'strategy_contribution_pct': strategy_return / (abs(total_return) + 1e-10) * 100
        }

    def identify_best_counterfactual_decisions(self, top_n: int = 10) -> List[TradeDecision]:
        """
        Identify decisions where we made the right call
        (observed return much better than counterfactual).
        """
        sorted_decisions = sorted(
            self.decision_history,
            key=lambda d: d.treatment_effect,
            reverse=True
        )
        return sorted_decisions[:top_n]

    def identify_worst_counterfactual_decisions(self, top_n: int = 10) -> List[TradeDecision]:
        """
        Identify decisions where we made the wrong call
        (counterfactual return would have been better).
        """
        sorted_decisions = sorted(
            self.decision_history,
            key=lambda d: d.treatment_effect,
            reverse=False
        )
        return sorted_decisions[:top_n]

    def compute_regret(self) -> Dict[str, float]:
        """
        Compute counterfactual regret metrics.

        Regret = max(0, counterfactual_return - observed_return)
        """
        regrets = [
            max(0, d.counterfactual_return - d.observed_return)
            for d in self.decision_history
        ]

        return {
            'total_regret': sum(regrets),
            'mean_regret': np.mean(regrets),
            'max_regret': max(regrets) if regrets else 0,
            'regret_frequency': sum(1 for r in regrets if r > 0) / len(regrets) if regrets else 0
        }


class CounterfactualPolicyOptimizer:
    """
    Learns optimal trading policy using counterfactual outcomes.
    """

    def __init__(self, cf_estimator: CounterfactualEstimator):
        self.cf_estimator = cf_estimator
        self.policy_model = None

    def estimate_policy_value(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        policy: callable
    ) -> float:
        """
        Estimate the value of a given policy using inverse propensity weighting.

        V(π) = E[Y(π(X))]
        """
        n = len(outcome)
        propensity = self.cf_estimator.propensity_model.predict_proba(X)[:, 1]
        propensity = np.clip(propensity, 0.01, 0.99)

        # Policy recommendations
        policy_actions = np.array([policy(x) for x in X])

        # IPW estimator
        weights = np.where(
            treatment == policy_actions,
            1 / np.where(treatment == 1, propensity, 1 - propensity),
            0
        )

        policy_value = np.sum(weights * outcome) / np.sum(weights)
        return policy_value

    def learn_optimal_policy(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray
    ) -> callable:
        """
        Learn the optimal trading policy that maximizes expected returns.

        Uses doubly robust policy learning.
        """
        n = len(outcome)

        # Estimate counterfactual outcomes for all units
        mu1 = self.cf_estimator.outcome_model_treated.predict(X)
        mu0 = self.cf_estimator.outcome_model_control.predict(X)

        propensity = self.cf_estimator.propensity_model.predict_proba(X)[:, 1]
        propensity = np.clip(propensity, 0.01, 0.99)

        # Doubly robust pseudo-outcomes
        gamma1 = mu1 + treatment / propensity * (outcome - mu1)
        gamma0 = mu0 + (1 - treatment) / (1 - propensity) * (outcome - mu0)

        # CATE estimates
        cate = gamma1 - gamma0

        # Learn policy: trade if CATE > 0
        from sklearn.ensemble import GradientBoostingClassifier
        policy_labels = (cate > 0).astype(int)
        self.policy_model = GradientBoostingClassifier(n_estimators=100)
        self.policy_model.fit(X, policy_labels)

        def optimal_policy(x):
            x = x.reshape(1, -1) if x.ndim == 1 else x
            return self.policy_model.predict(x)[0]

        return optimal_policy
```

### Trading Data Pipeline

```python
import pandas as pd
import numpy as np
import requests
from typing import Optional


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


def prepare_counterfactual_dataset(
    prices: pd.DataFrame,
    strategy_signals: pd.Series,
    forward_return_periods: int = 1
) -> pd.DataFrame:
    """
    Prepare dataset for counterfactual analysis.

    Args:
        prices: OHLCV data
        strategy_signals: Trading signals (1, -1, 0)
        forward_return_periods: Periods for computing forward returns
    """
    df = pd.DataFrame(index=prices.index)

    # Features
    close = prices['close']
    df['return_1d'] = close.pct_change(1)
    df['return_5d'] = close.pct_change(5)
    df['return_20d'] = close.pct_change(20)
    df['volatility'] = df['return_1d'].rolling(20).std()
    df['momentum'] = close / close.rolling(20).mean() - 1
    df['rsi'] = compute_rsi(close, 14)

    if 'volume' in prices.columns:
        df['volume_ratio'] = prices['volume'] / prices['volume'].rolling(20).mean()

    # Treatment (trading signal)
    df['treatment'] = (strategy_signals != 0).astype(int)
    df['signal'] = strategy_signals

    # Outcome (forward return)
    df['forward_return'] = close.pct_change(forward_return_periods).shift(-forward_return_periods)

    # Observed return (signal * forward_return if traded, else 0)
    df['observed_return'] = df['signal'] * df['forward_return']

    return df.dropna()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta).where(delta < 0, 0).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))
```

### Backtesting with Counterfactuals

```python
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd


@dataclass
class CounterfactualBacktestResult:
    """Results from counterfactual backtesting."""
    total_return: float
    counterfactual_return: float
    strategy_alpha: float
    sharpe_ratio: float
    counterfactual_sharpe: float
    regret: float
    attribution: Dict[str, float]
    decision_analysis: pd.DataFrame


class CounterfactualBacktester:
    """
    Backtesting framework with counterfactual analysis.
    """

    def __init__(
        self,
        cf_estimator: CounterfactualEstimator,
        transaction_cost: float = 0.001
    ):
        self.cf_estimator = cf_estimator
        self.transaction_cost = transaction_cost

    def run(
        self,
        prices: pd.DataFrame,
        strategy_signals: pd.Series,
        train_ratio: float = 0.5
    ) -> CounterfactualBacktestResult:
        """
        Run backtest with counterfactual analysis.
        """
        # Prepare data
        data = prepare_counterfactual_dataset(prices, strategy_signals)

        feature_cols = ['return_1d', 'return_5d', 'return_20d', 'volatility', 'momentum']
        if 'volume_ratio' in data.columns:
            feature_cols.append('volume_ratio')

        X = data[feature_cols].values
        treatment = data['treatment'].values
        outcome = data['observed_return'].values

        # Split train/test
        train_size = int(len(data) * train_ratio)

        X_train, X_test = X[:train_size], X[train_size:]
        treatment_train, treatment_test = treatment[:train_size], treatment[train_size:]
        outcome_train, outcome_test = outcome[:train_size], outcome[train_size:]

        # Fit counterfactual model
        self.cf_estimator.fit(X_train, treatment_train, outcome_train)

        # Estimate counterfactuals for test period
        cf_outcomes = []
        for i in range(len(X_test)):
            cf = self.cf_estimator.estimate_counterfactual(
                X_test[i], treatment_test[i], outcome_test[i]
            )
            cf_outcomes.append(cf.counterfactual_outcome)

        cf_outcomes = np.array(cf_outcomes)

        # Compute metrics
        total_return = np.sum(outcome_test)
        cf_return = np.sum(cf_outcomes)
        strategy_alpha = total_return - cf_return

        # Sharpe ratios
        sharpe = self._compute_sharpe(outcome_test)
        cf_sharpe = self._compute_sharpe(cf_outcomes)

        # Regret
        regret = np.sum(np.maximum(0, cf_outcomes - outcome_test))

        # Attribution
        attribution = {
            'total_return': total_return,
            'market_component': cf_return,
            'strategy_alpha': strategy_alpha,
            'alpha_contribution_pct': strategy_alpha / (abs(total_return) + 1e-10) * 100
        }

        # Decision analysis
        decision_df = pd.DataFrame({
            'observed': outcome_test,
            'counterfactual': cf_outcomes,
            'treatment_effect': outcome_test - cf_outcomes,
            'treatment': treatment_test
        }, index=data.index[train_size:])

        return CounterfactualBacktestResult(
            total_return=total_return,
            counterfactual_return=cf_return,
            strategy_alpha=strategy_alpha,
            sharpe_ratio=sharpe,
            counterfactual_sharpe=cf_sharpe,
            regret=regret,
            attribution=attribution,
            decision_analysis=decision_df
        )

    def _compute_sharpe(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Compute annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
```

---

## Implementation in Rust

### Project Structure

```
110_counterfactual_trading/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── estimator/
│   │   ├── mod.rs
│   │   ├── outcome_regression.rs
│   │   ├── propensity.rs
│   │   ├── doubly_robust.rs
│   │   └── twin_network.rs
│   ├── strategy/
│   │   ├── mod.rs
│   │   ├── trading.rs
│   │   └── policy.rs
│   ├── data/
│   │   ├── mod.rs
│   │   ├── features.rs
│   │   └── bybit.rs
│   └── backtest/
│       ├── mod.rs
│       └── metrics.rs
└── examples/
    ├── stock_counterfactual.rs
    ├── crypto_counterfactual.rs
    └── policy_optimization.rs
```

### Cargo.toml

```toml
[package]
name = "counterfactual_trading"
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

[dev-dependencies]
criterion = "0.5"
```

### Counterfactual Estimator (Rust)

```rust
// src/estimator/outcome_regression.rs
use ndarray::{Array1, Array2, Axis};

/// Outcome regression model for counterfactual estimation
pub struct OutcomeRegression {
    /// Coefficients for treated outcome model
    pub coef_treated: Array1<f64>,
    /// Coefficients for control outcome model
    pub coef_control: Array1<f64>,
    /// Intercept for treated model
    pub intercept_treated: f64,
    /// Intercept for control model
    pub intercept_control: f64,
    /// Flag indicating if model is fitted
    pub fitted: bool,
}

impl OutcomeRegression {
    pub fn new() -> Self {
        OutcomeRegression {
            coef_treated: Array1::zeros(0),
            coef_control: Array1::zeros(0),
            intercept_treated: 0.0,
            intercept_control: 0.0,
            fitted: false,
        }
    }

    /// Fit outcome models for treated and control groups
    pub fn fit(
        &mut self,
        x: &Array2<f64>,
        treatment: &Array1<f64>,
        outcome: &Array1<f64>,
    ) {
        let n = x.nrows();
        let p = x.ncols();

        // Separate treated and control
        let mut x_treated = Vec::new();
        let mut y_treated = Vec::new();
        let mut x_control = Vec::new();
        let mut y_control = Vec::new();

        for i in 0..n {
            if treatment[i] > 0.5 {
                x_treated.push(x.row(i).to_owned());
                y_treated.push(outcome[i]);
            } else {
                x_control.push(x.row(i).to_owned());
                y_control.push(outcome[i]);
            }
        }

        // Fit treated model
        if !x_treated.is_empty() {
            let x_t = stack_rows(&x_treated);
            let y_t = Array1::from_vec(y_treated);
            let (coef, intercept) = ols_regression(&x_t, &y_t);
            self.coef_treated = coef;
            self.intercept_treated = intercept;
        }

        // Fit control model
        if !x_control.is_empty() {
            let x_c = stack_rows(&x_control);
            let y_c = Array1::from_vec(y_control);
            let (coef, intercept) = ols_regression(&x_c, &y_c);
            self.coef_control = coef;
            self.intercept_control = intercept;
        }

        self.fitted = true;
    }

    /// Predict outcome under treatment
    pub fn predict_treated(&self, x: &Array1<f64>) -> f64 {
        x.dot(&self.coef_treated) + self.intercept_treated
    }

    /// Predict outcome under control
    pub fn predict_control(&self, x: &Array1<f64>) -> f64 {
        x.dot(&self.coef_control) + self.intercept_control
    }

    /// Estimate counterfactual outcome
    pub fn estimate_counterfactual(
        &self,
        x: &Array1<f64>,
        treatment: f64,
        observed_outcome: f64,
    ) -> CounterfactualResult {
        let cf_outcome = if treatment > 0.5 {
            // Was treated, estimate control outcome
            self.predict_control(x)
        } else {
            // Was not treated, estimate treated outcome
            self.predict_treated(x)
        };

        let treatment_effect = if treatment > 0.5 {
            observed_outcome - cf_outcome
        } else {
            cf_outcome - observed_outcome
        };

        CounterfactualResult {
            observed_outcome,
            counterfactual_outcome: cf_outcome,
            treatment_effect,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    pub observed_outcome: f64,
    pub counterfactual_outcome: f64,
    pub treatment_effect: f64,
}

/// OLS regression via normal equations
fn ols_regression(x: &Array2<f64>, y: &Array1<f64>) -> (Array1<f64>, f64) {
    let n = x.nrows();
    let p = x.ncols();

    // Add intercept column
    let mut x_aug = Array2::ones((n, p + 1));
    for i in 0..n {
        for j in 0..p {
            x_aug[[i, j + 1]] = x[[i, j]];
        }
    }

    // Normal equations: (X'X)^-1 X'y
    let xt = x_aug.t();
    let xtx = xt.dot(&x_aug);
    let xty = xt.dot(y);

    let beta = solve_linear_system(&xtx, &xty);

    let intercept = beta[0];
    let coef = beta.slice(ndarray::s![1..]).to_owned();

    (coef, intercept)
}

fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = a.nrows();
    let mut aug = Array2::zeros((n, n + 1));

    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Gauss-Jordan elimination
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
        if aug[[i, i]].abs() > 1e-10 {
            for k in (i + 1)..n {
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

fn stack_rows(rows: &[Array1<f64>]) -> Array2<f64> {
    let n = rows.len();
    let p = rows[0].len();
    let mut result = Array2::zeros((n, p));
    for (i, row) in rows.iter().enumerate() {
        for j in 0..p {
            result[[i, j]] = row[j];
        }
    }
    result
}
```

### Doubly Robust Estimator (Rust)

```rust
// src/estimator/doubly_robust.rs
use ndarray::{Array1, Array2};
use crate::estimator::outcome_regression::OutcomeRegression;
use crate::estimator::propensity::PropensityModel;

/// Doubly robust estimator for counterfactual inference
pub struct DoublyRobustEstimator {
    outcome_model: OutcomeRegression,
    propensity_model: PropensityModel,
    fitted: bool,
}

impl DoublyRobustEstimator {
    pub fn new() -> Self {
        DoublyRobustEstimator {
            outcome_model: OutcomeRegression::new(),
            propensity_model: PropensityModel::new(),
            fitted: false,
        }
    }

    /// Fit both outcome and propensity models
    pub fn fit(
        &mut self,
        x: &Array2<f64>,
        treatment: &Array1<f64>,
        outcome: &Array1<f64>,
    ) {
        self.outcome_model.fit(x, treatment, outcome);
        self.propensity_model.fit(x, treatment);
        self.fitted = true;
    }

    /// Estimate Average Treatment Effect using doubly robust estimator
    pub fn estimate_ate(
        &self,
        x: &Array2<f64>,
        treatment: &Array1<f64>,
        outcome: &Array1<f64>,
    ) -> ATEResult {
        let n = x.nrows();

        // Get propensity scores
        let propensity: Vec<f64> = (0..n)
            .map(|i| {
                let p = self.propensity_model.predict(&x.row(i).to_owned());
                p.clamp(0.01, 0.99)
            })
            .collect();

        // Get outcome predictions
        let mu1: Vec<f64> = (0..n)
            .map(|i| self.outcome_model.predict_treated(&x.row(i).to_owned()))
            .collect();
        let mu0: Vec<f64> = (0..n)
            .map(|i| self.outcome_model.predict_control(&x.row(i).to_owned()))
            .collect();

        // Doubly robust estimator
        let mut treated_sum = 0.0;
        let mut control_sum = 0.0;

        for i in 0..n {
            let t = treatment[i];
            let y = outcome[i];
            let e = propensity[i];

            // Treated term
            treated_sum += t * y / e - (t - e) / e * mu1[i];

            // Control term
            control_sum += (1.0 - t) * y / (1.0 - e) + (t - e) / (1.0 - e) * mu0[i];
        }

        let ate = treated_sum / n as f64 - control_sum / n as f64;

        // Standard error via influence function
        let mut influence = Vec::with_capacity(n);
        for i in 0..n {
            let t = treatment[i];
            let y = outcome[i];
            let e = propensity[i];

            let treated_term = t * y / e - (t - e) / e * mu1[i];
            let control_term = (1.0 - t) * y / (1.0 - e) + (t - e) / (1.0 - e) * mu0[i];

            influence.push(treated_term - control_term - ate);
        }

        let variance: f64 = influence.iter().map(|x| x * x).sum::<f64>() / n as f64;
        let se = (variance / n as f64).sqrt();

        ATEResult {
            ate,
            se,
            ci_low: ate - 1.96 * se,
            ci_high: ate + 1.96 * se,
        }
    }

    /// Estimate individual counterfactual outcome
    pub fn estimate_counterfactual(
        &self,
        x: &Array1<f64>,
        treatment: f64,
        observed_outcome: f64,
    ) -> CounterfactualResult {
        self.outcome_model.estimate_counterfactual(x, treatment, observed_outcome)
    }
}

#[derive(Debug, Clone)]
pub struct ATEResult {
    pub ate: f64,
    pub se: f64,
    pub ci_low: f64,
    pub ci_high: f64,
}

use crate::estimator::outcome_regression::CounterfactualResult;
```

### Propensity Score Model (Rust)

```rust
// src/estimator/propensity.rs
use ndarray::{Array1, Array2};

/// Logistic regression for propensity score estimation
pub struct PropensityModel {
    pub coef: Array1<f64>,
    pub intercept: f64,
    pub fitted: bool,
}

impl PropensityModel {
    pub fn new() -> Self {
        PropensityModel {
            coef: Array1::zeros(0),
            intercept: 0.0,
            fitted: false,
        }
    }

    /// Fit logistic regression for propensity scores
    pub fn fit(&mut self, x: &Array2<f64>, treatment: &Array1<f64>) {
        let n = x.nrows();
        let p = x.ncols();

        // Initialize coefficients
        let mut beta = Array1::zeros(p + 1);
        let learning_rate = 0.01;
        let max_iter = 1000;

        // Gradient descent for logistic regression
        for _ in 0..max_iter {
            let mut gradient = Array1::zeros(p + 1);

            for i in 0..n {
                let xi = &x.row(i);
                let yi = treatment[i];

                // Linear combination
                let mut z = beta[0];
                for j in 0..p {
                    z += beta[j + 1] * xi[j];
                }

                // Sigmoid
                let prob = 1.0 / (1.0 + (-z).exp());

                // Gradient
                let error = prob - yi;
                gradient[0] += error;
                for j in 0..p {
                    gradient[j + 1] += error * xi[j];
                }
            }

            // Update
            for j in 0..=p {
                beta[j] -= learning_rate * gradient[j] / n as f64;
            }
        }

        self.intercept = beta[0];
        self.coef = beta.slice(ndarray::s![1..]).to_owned();
        self.fitted = true;
    }

    /// Predict propensity score P(T=1|X)
    pub fn predict(&self, x: &Array1<f64>) -> f64 {
        let z = self.intercept + x.dot(&self.coef);
        1.0 / (1.0 + (-z).exp())
    }
}
```

### Trading Strategy (Rust)

```rust
// src/strategy/trading.rs
use ndarray::Array1;
use crate::estimator::doubly_robust::DoublyRobustEstimator;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct TradeDecision {
    pub timestamp: i64,
    pub action: i32,  // 1 = buy, -1 = sell, 0 = hold
    pub observed_return: f64,
    pub counterfactual_return: f64,
    pub treatment_effect: f64,
}

pub struct CounterfactualTradingStrategy {
    cf_estimator: DoublyRobustEstimator,
    decision_history: VecDeque<TradeDecision>,
    max_history: usize,
}

impl CounterfactualTradingStrategy {
    pub fn new(cf_estimator: DoublyRobustEstimator, max_history: usize) -> Self {
        CounterfactualTradingStrategy {
            cf_estimator,
            decision_history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    /// Evaluate a trading decision using counterfactual analysis
    pub fn evaluate_decision(
        &mut self,
        features: &Array1<f64>,
        action: i32,
        observed_return: f64,
        timestamp: i64,
    ) -> TradeDecision {
        let treatment = if action != 0 { 1.0 } else { 0.0 };

        let cf_result = self.cf_estimator.estimate_counterfactual(
            features,
            treatment,
            observed_return,
        );

        let decision = TradeDecision {
            timestamp,
            action,
            observed_return,
            counterfactual_return: cf_result.counterfactual_outcome,
            treatment_effect: cf_result.treatment_effect,
        };

        // Maintain history
        if self.decision_history.len() >= self.max_history {
            self.decision_history.pop_front();
        }
        self.decision_history.push_back(decision.clone());

        decision
    }

    /// Compute strategy attribution
    pub fn compute_attribution(&self) -> StrategyAttribution {
        let total_return: f64 = self.decision_history.iter()
            .map(|d| d.observed_return)
            .sum();

        let cf_return: f64 = self.decision_history.iter()
            .map(|d| d.counterfactual_return)
            .sum();

        let strategy_alpha = total_return - cf_return;

        let alpha_contribution = if total_return.abs() > 1e-10 {
            strategy_alpha / total_return.abs() * 100.0
        } else {
            0.0
        };

        StrategyAttribution {
            total_return,
            market_component: cf_return,
            strategy_alpha,
            alpha_contribution_pct: alpha_contribution,
        }
    }

    /// Compute counterfactual regret
    pub fn compute_regret(&self) -> RegretMetrics {
        let regrets: Vec<f64> = self.decision_history.iter()
            .map(|d| (d.counterfactual_return - d.observed_return).max(0.0))
            .collect();

        let total_regret: f64 = regrets.iter().sum();
        let mean_regret = total_regret / regrets.len() as f64;
        let max_regret = regrets.iter().cloned().fold(0.0, f64::max);
        let regret_frequency = regrets.iter().filter(|&&r| r > 0.0).count() as f64
            / regrets.len() as f64;

        RegretMetrics {
            total_regret,
            mean_regret,
            max_regret,
            regret_frequency,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StrategyAttribution {
    pub total_return: f64,
    pub market_component: f64,
    pub strategy_alpha: f64,
    pub alpha_contribution_pct: f64,
}

#[derive(Debug, Clone)]
pub struct RegretMetrics {
    pub total_regret: f64,
    pub mean_regret: f64,
    pub max_regret: f64,
    pub regret_frequency: f64,
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

### Backtest Metrics (Rust)

```rust
// src/backtest/metrics.rs
use ndarray::Array1;

#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub total_return: f64,
    pub counterfactual_return: f64,
    pub strategy_alpha: f64,
    pub sharpe_ratio: f64,
    pub counterfactual_sharpe: f64,
    pub regret: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
}

pub fn compute_sharpe_ratio(returns: &[f64], periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance: f64 = returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / returns.len() as f64;
    let std_dev = variance.sqrt();

    if std_dev < 1e-10 {
        return 0.0;
    }

    mean / std_dev * periods_per_year.sqrt()
}

pub fn compute_max_drawdown(returns: &[f64]) -> f64 {
    let mut cumulative = 1.0;
    let mut running_max = 1.0;
    let mut max_dd = 0.0;

    for &ret in returns {
        cumulative *= 1.0 + ret;
        running_max = running_max.max(cumulative);
        let dd = (cumulative - running_max) / running_max;
        max_dd = max_dd.min(dd);
    }

    max_dd
}

pub fn compute_win_rate(returns: &[f64]) -> f64 {
    let wins = returns.iter().filter(|&&r| r > 0.0).count();
    let total = returns.iter().filter(|&&r| r != 0.0).count();

    if total == 0 {
        return 0.0;
    }

    wins as f64 / total as f64
}

pub fn compute_regret(observed: &[f64], counterfactual: &[f64]) -> f64 {
    observed.iter()
        .zip(counterfactual.iter())
        .map(|(o, c)| (c - o).max(0.0))
        .sum()
}
```

---

## Practical Examples with Stock and Crypto Data

### Example 1: Stock Counterfactual Analysis (Python)

```python
import yfinance as yf
import numpy as np
import pandas as pd

# Download stock data
data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
data.columns = [c.lower() for c in data.columns]

# Generate simple momentum strategy signals
returns = data['close'].pct_change()
signals = np.where(returns.rolling(5).mean() > 0, 1, -1)
signals = pd.Series(signals, index=data.index)

# Prepare dataset
dataset = prepare_counterfactual_dataset(data, signals)

feature_cols = ['return_1d', 'return_5d', 'return_20d', 'volatility', 'momentum']
X = dataset[feature_cols].values
treatment = dataset['treatment'].values
outcome = dataset['observed_return'].values

# Fit counterfactual estimator
cf_estimator = CounterfactualEstimator(method='doubly_robust')
cf_estimator.fit(X[:500], treatment[:500], outcome[:500])

# Analyze specific trading decisions
for i in range(500, 510):
    result = cf_estimator.estimate_counterfactual(X[i], treatment[i], outcome[i])
    print(f"Day {i}:")
    print(f"  Observed: {result.observed_outcome:.4f}")
    print(f"  Counterfactual: {result.counterfactual_outcome:.4f}")
    print(f"  Treatment Effect: {result.treatment_effect:.4f}")
    print()

# Estimate ATE
ate_result = cf_estimator.estimate_ate(X[500:], treatment[500:], outcome[500:])
print(f"Average Treatment Effect: {ate_result['ate']:.4f}")
print(f"95% CI: [{ate_result['ci_low']:.4f}, {ate_result['ci_high']:.4f}]")
```

### Example 2: Crypto Counterfactual Trading (Python)

```python
# Fetch BTC data
btc_data = fetch_bybit_data("BTCUSDT", "D", 1000)

# RSI-based strategy
rsi = compute_rsi(btc_data['close'], 14)
signals = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
signals = pd.Series(signals, index=btc_data.index)

# Prepare dataset
dataset = prepare_counterfactual_dataset(btc_data, signals)

# Split and fit
train_size = 500
feature_cols = ['return_1d', 'return_5d', 'return_20d', 'volatility', 'momentum']
X_train = dataset[feature_cols].values[:train_size]
X_test = dataset[feature_cols].values[train_size:]
treatment_train = dataset['treatment'].values[:train_size]
treatment_test = dataset['treatment'].values[train_size:]
outcome_train = dataset['observed_return'].values[:train_size]
outcome_test = dataset['observed_return'].values[train_size:]

# Fit and evaluate
cf_estimator = CounterfactualEstimator()
cf_estimator.fit(X_train, treatment_train, outcome_train)

# Counterfactual backtest
backtester = CounterfactualBacktester(cf_estimator)
result = backtester.run(btc_data, signals)

print(f"Total Return: {result.total_return:.2%}")
print(f"Counterfactual Return: {result.counterfactual_return:.2%}")
print(f"Strategy Alpha: {result.strategy_alpha:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Regret: {result.regret:.2%}")
```

### Example 3: Rust Counterfactual Analysis

```rust
// examples/crypto_counterfactual.rs
use counterfactual_trading::data::bybit::fetch_bybit_klines;
use counterfactual_trading::estimator::doubly_robust::DoublyRobustEstimator;
use counterfactual_trading::strategy::trading::CounterfactualTradingStrategy;
use ndarray::{Array1, Array2};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Fetch BTC data
    let klines = fetch_bybit_klines("BTCUSDT", "D", 500).await?;
    println!("Fetched {} klines", klines.len());

    // Compute features and returns
    let mut features = Vec::new();
    let mut returns = Vec::new();

    for i in 20..klines.len() - 1 {
        let close = klines[i].close;
        let prev_close = klines[i - 1].close;

        // Simple features
        let ret_1d = (close / prev_close).ln();
        let ret_5d = (close / klines[i - 5].close).ln();
        let ret_20d = (close / klines[i - 20].close).ln();

        // Volatility (simplified)
        let mut vol_sum = 0.0;
        for j in 0..20 {
            let r = (klines[i - j].close / klines[i - j - 1].close).ln();
            vol_sum += r * r;
        }
        let volatility = (vol_sum / 20.0).sqrt();

        features.push(vec![ret_1d, ret_5d, ret_20d, volatility]);

        // Forward return
        let fwd_ret = (klines[i + 1].close / close).ln();
        returns.push(fwd_ret);
    }

    // Create signals (simple momentum)
    let mut treatments = Vec::new();
    let mut outcomes = Vec::new();

    for i in 0..features.len() {
        let signal = if features[i][0] > 0.0 { 1.0 } else { 0.0 };
        treatments.push(signal);
        outcomes.push(if signal > 0.5 { returns[i] } else { 0.0 });
    }

    // Convert to arrays
    let x = Array2::from_shape_vec(
        (features.len(), 4),
        features.into_iter().flatten().collect()
    )?;
    let treatment = Array1::from_vec(treatments);
    let outcome = Array1::from_vec(outcomes);

    // Fit estimator
    let mut estimator = DoublyRobustEstimator::new();
    let train_size = 300;

    let x_train = x.slice(ndarray::s![..train_size, ..]).to_owned();
    let treatment_train = treatment.slice(ndarray::s![..train_size]).to_owned();
    let outcome_train = outcome.slice(ndarray::s![..train_size]).to_owned();

    estimator.fit(&x_train, &treatment_train, &outcome_train);

    // Estimate ATE
    let x_test = x.slice(ndarray::s![train_size.., ..]).to_owned();
    let treatment_test = treatment.slice(ndarray::s![train_size..]).to_owned();
    let outcome_test = outcome.slice(ndarray::s![train_size..]).to_owned();

    let ate = estimator.estimate_ate(&x_test, &treatment_test, &outcome_test);

    println!("\nAverage Treatment Effect: {:.4}", ate.ate);
    println!("95% CI: [{:.4}, {:.4}]", ate.ci_low, ate.ci_high);

    // Individual counterfactuals
    println!("\nSample Counterfactual Analysis:");
    for i in 0..5 {
        let cf = estimator.estimate_counterfactual(
            &x_test.row(i).to_owned(),
            treatment_test[i],
            outcome_test[i],
        );
        println!(
            "  Observed: {:.4}, Counterfactual: {:.4}, Effect: {:.4}",
            cf.observed_outcome, cf.counterfactual_outcome, cf.treatment_effect
        );
    }

    Ok(())
}
```

---

## Performance Evaluation

### Metrics Summary

| Metric | Description | Target |
|--------|-------------|--------|
| Strategy Alpha | Return attributable to trading decisions | > 0 |
| Alpha t-statistic | Statistical significance of alpha | > 2.0 |
| Counterfactual Regret | How much better we could have done | Minimize |
| Sharpe Ratio | Risk-adjusted return | > 1.0 |
| Decision Win Rate | % of decisions with positive treatment effect | > 50% |
| Attribution Accuracy | How well we separate market vs strategy | Higher is better |

### Advantages of Counterfactual Trading

1. **True Attribution**: Separates market luck from trading skill
2. **Better Decision Making**: Understand what actions actually cause returns
3. **Reduced Overfitting**: Counterfactual reasoning is more robust
4. **Policy Learning**: Can learn optimal trading rules from data
5. **Regret Analysis**: Understand opportunity costs of decisions

### Limitations

1. **Strong Assumptions**: Requires valid causal model
2. **Unconfoundedness**: Assumes no unmeasured confounders
3. **Positivity**: Needs overlap in covariate distributions
4. **Model Misspecification**: Results depend on model quality

---

## Future Directions

1. **Deep Counterfactual Models**: Neural networks for complex counterfactual estimation
2. **Time-Varying Treatment Effects**: Heterogeneous effects across market regimes
3. **Multi-Action Counterfactuals**: Beyond binary treatment (position sizing)
4. **Counterfactual Reinforcement Learning**: Combine with RL for optimal execution
5. **Real-Time Counterfactual Monitoring**: Online estimation during live trading
6. **Causal Bandits**: Explore-exploit with counterfactual reasoning

---

## References

1. Pearl, J. (2009). Causality: Models, Reasoning, and Inference. Cambridge University Press.
2. Imbens, G. W., & Rubin, D. B. (2015). Causal Inference for Statistics, Social, and Biomedical Sciences. Cambridge University Press.
3. Athey, S., & Imbens, G. W. (2017). The State of Applied Econometrics: Causality and Policy Evaluation. Journal of Economic Perspectives.
4. Chernozhukov, V., et al. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. The Econometrics Journal.
5. Kennedy, E. H. (2016). Semiparametric Theory and Empirical Processes in Causal Inference. Statistical Science.
6. Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. Journal of the American Statistical Association.
7. Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of Regression Coefficients When Some Regressors are not Always Observed. Journal of the American Statistical Association.
8. Bang, H., & Robins, J. M. (2005). Doubly Robust Estimation in Missing Data and Causal Inference Models. Biometrics.

---

## Running the Examples

### Python

```bash
cd 110_counterfactual_trading/python
pip install -r requirements.txt
python model.py            # Test counterfactual estimation
python backtest.py         # Run counterfactual backtest
jupyter notebook examples.ipynb  # Interactive examples
```

### Rust

```bash
cd 110_counterfactual_trading
cargo build --release
cargo run --example stock_counterfactual
cargo run --example crypto_counterfactual
cargo run --example policy_optimization
```
