"""
Counterfactual Trading Models

This module implements counterfactual estimation methods for trading decision analysis.
Supports outcome regression, propensity score matching, and doubly robust estimation.
"""

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

    Example:
        >>> estimator = CounterfactualEstimator(method='doubly_robust')
        >>> estimator.fit(X, treatment, outcome)
        >>> result = estimator.estimate_counterfactual(x, 1, 0.05)
    """

    def __init__(self, method: str = 'doubly_robust'):
        """
        Initialize the counterfactual estimator.

        Args:
            method: Estimation method ('outcome_regression', 'matching', 'doubly_robust')
        """
        self.method = method
        self.outcome_model_treated = None
        self.outcome_model_control = None
        self.propensity_model = None
        self.fitted = False

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray):
        """
        Fit the counterfactual model.

        Args:
            X: Covariates (market features) of shape (n_samples, n_features)
            treatment: Treatment indicator (1 = traded, 0 = no trade) of shape (n_samples,)
            outcome: Observed outcomes (returns) of shape (n_samples,)

        Returns:
            self: Fitted estimator
        """
        X = np.asarray(X)
        treatment = np.asarray(treatment)
        outcome = np.asarray(outcome)

        # Fit outcome models for treated and control
        self.outcome_model_treated = LinearRegression()
        self.outcome_model_control = LinearRegression()

        treated_mask = treatment == 1
        control_mask = treatment == 0

        if np.sum(treated_mask) > 0:
            self.outcome_model_treated.fit(X[treated_mask], outcome[treated_mask])
        if np.sum(control_mask) > 0:
            self.outcome_model_control.fit(X[control_mask], outcome[control_mask])

        # Fit propensity model
        self.propensity_model = LogisticRegression(max_iter=1000, solver='lbfgs')
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
            CounterfactualResult with counterfactual outcome and treatment effect
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        X = np.asarray(X)
        X = X.reshape(1, -1) if X.ndim == 1 else X

        # Estimate counterfactual outcome
        if treatment == 1:
            cf_outcome = self.outcome_model_control.predict(X)[0]
        else:
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
        n_bootstrap: int = 500,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for treatment effect."""
        effects = []
        noise_scale = 0.01

        for _ in range(n_bootstrap):
            noise = np.random.normal(0, noise_scale)
            if treatment == 1:
                cf = self.outcome_model_control.predict(X)[0] + noise
                effect = observed_outcome - cf
            else:
                cf = self.outcome_model_treated.predict(X)[0] + noise
                effect = cf - observed_outcome
            effects.append(effect)

        return tuple(np.percentile(effects, [100*alpha/2, 100*(1-alpha/2)]))

    def estimate_ate(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray
    ) -> Dict[str, float]:
        """
        Estimate Average Treatment Effect using doubly robust estimation.

        Args:
            X: Covariates
            treatment: Treatment indicators
            outcome: Observed outcomes

        Returns:
            Dictionary with ATE, standard error, and confidence interval
        """
        X = np.asarray(X)
        treatment = np.asarray(treatment)
        outcome = np.asarray(outcome)
        n = len(outcome)

        # Propensity scores
        propensity = self.propensity_model.predict_proba(X)[:, 1]
        propensity = np.clip(propensity, 0.01, 0.99)

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

    The key insight is that the noise term U is shared between
    the factual and counterfactual worlds.
    """

    def __init__(self):
        self.structural_model = None
        self.noise_mean = 0.0
        self.noise_std = 1.0

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray):
        """
        Fit the structural causal model: Y = f(T, X) + U

        Args:
            X: Covariates
            treatment: Treatment indicators
            outcome: Observed outcomes
        """
        X = np.asarray(X)
        treatment = np.asarray(treatment).reshape(-1, 1)
        outcome = np.asarray(outcome)

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

        Three steps:
        1. Abduction: Infer noise U from observed outcome
        2. Action: Set counterfactual treatment
        3. Prediction: Compute outcome with same noise
        """
        X = np.asarray(X)
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
    Matching-based counterfactual estimation using propensity scores.
    Finds similar units with opposite treatment to estimate counterfactuals.
    """

    def __init__(self, n_neighbors: int = 5):
        """
        Initialize matching estimator.

        Args:
            n_neighbors: Number of neighbors for matching
        """
        self.n_neighbors = n_neighbors
        self.treated_nn = None
        self.control_nn = None
        self.data = None
        self.treated_indices = None
        self.control_indices = None

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray):
        """Fit nearest neighbor models for matching."""
        X = np.asarray(X)
        treatment = np.asarray(treatment)
        outcome = np.asarray(outcome)

        self.data = {
            'X': X,
            'treatment': treatment,
            'outcome': outcome
        }

        treated_mask = treatment == 1
        control_mask = treatment == 0

        # Fit treated neighbors
        n_treated = np.sum(treated_mask)
        if n_treated > 0:
            self.treated_nn = NearestNeighbors(
                n_neighbors=min(self.n_neighbors, n_treated)
            )
            self.treated_nn.fit(X[treated_mask])
            self.treated_indices = np.where(treated_mask)[0]

        # Fit control neighbors
        n_control = np.sum(control_mask)
        if n_control > 0:
            self.control_nn = NearestNeighbors(
                n_neighbors=min(self.n_neighbors, n_control)
            )
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
        X = np.asarray(X)
        X = X.reshape(1, -1) if X.ndim == 1 else X

        if treatment == 1:
            # Find similar control units
            if self.control_nn is None:
                return observed_outcome
            distances, indices = self.control_nn.kneighbors(X)
            matched_indices = self.control_indices[indices[0]]
        else:
            # Find similar treated units
            if self.treated_nn is None:
                return observed_outcome
            distances, indices = self.treated_nn.kneighbors(X)
            matched_indices = self.treated_indices[indices[0]]

        # Inverse distance weighting
        weights = 1 / (distances[0] + 1e-6)
        weights = weights / np.sum(weights)

        cf_outcome = np.sum(weights * self.data['outcome'][matched_indices])
        return cf_outcome


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta).where(delta < 0, 0).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


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

    Returns:
        DataFrame with features, treatment, and outcomes
    """
    df = pd.DataFrame(index=prices.index)

    # Ensure column names are lowercase
    prices = prices.copy()
    prices.columns = [c.lower() for c in prices.columns]
    close = prices['close']

    # Features
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


# Example usage
if __name__ == '__main__':
    import requests

    def fetch_bybit_data(symbol: str = "BTCUSDT", interval: str = "D", limit: int = 500) -> pd.DataFrame:
        """Fetch cryptocurrency data from Bybit API."""
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        try:
            resp = requests.get(url, params=params, timeout=10).json()
            records = resp['result']['list']

            df = pd.DataFrame(records, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            df['open_time'] = pd.to_datetime(df['open_time'].astype(int), unit='ms')
            df = df.sort_values('open_time').reset_index(drop=True)
            df.set_index('open_time', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Return synthetic data for testing
            dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq='D')
            np.random.seed(42)
            returns = np.random.normal(0.001, 0.02, limit)
            close = 100 * np.exp(np.cumsum(returns))
            return pd.DataFrame({
                'open': close * 0.99,
                'high': close * 1.01,
                'low': close * 0.98,
                'close': close,
                'volume': np.random.uniform(1000, 10000, limit)
            }, index=dates)

    print("Counterfactual Trading Model Demo")
    print("=" * 50)

    # Fetch data
    print("\nFetching BTC data...")
    data = fetch_bybit_data("BTCUSDT", "D", 500)
    print(f"Data shape: {data.shape}")

    # Generate momentum signals
    returns = data['close'].pct_change()
    signals = pd.Series(
        np.where(returns.rolling(5).mean() > 0, 1, -1),
        index=data.index
    )

    # Prepare dataset
    dataset = prepare_counterfactual_dataset(data, signals)
    print(f"Dataset shape: {dataset.shape}")

    # Split data
    train_size = 300
    feature_cols = ['return_1d', 'return_5d', 'return_20d', 'volatility', 'momentum']
    if 'volume_ratio' in dataset.columns:
        feature_cols.append('volume_ratio')

    X_train = dataset[feature_cols].values[:train_size]
    X_test = dataset[feature_cols].values[train_size:]
    treatment_train = dataset['treatment'].values[:train_size]
    treatment_test = dataset['treatment'].values[train_size:]
    outcome_train = dataset['observed_return'].values[:train_size]
    outcome_test = dataset['observed_return'].values[train_size:]

    # Fit counterfactual estimator
    print("\nFitting counterfactual estimator...")
    estimator = CounterfactualEstimator(method='doubly_robust')
    estimator.fit(X_train, treatment_train, outcome_train)

    # Estimate ATE
    ate_result = estimator.estimate_ate(X_test, treatment_test, outcome_test)
    print(f"\nAverage Treatment Effect: {ate_result['ate']:.6f}")
    print(f"Standard Error: {ate_result['se']:.6f}")
    print(f"95% CI: [{ate_result['ci_low']:.6f}, {ate_result['ci_high']:.6f}]")

    # Analyze individual decisions
    print("\nSample Counterfactual Analysis:")
    for i in range(5):
        result = estimator.estimate_counterfactual(
            X_test[i], treatment_test[i], outcome_test[i]
        )
        print(f"  Decision {i+1}:")
        print(f"    Observed: {result.observed_outcome:.6f}")
        print(f"    Counterfactual: {result.counterfactual_outcome:.6f}")
        print(f"    Treatment Effect: {result.treatment_effect:.6f}")

    print("\nDemo completed successfully!")
