"""
Counterfactual Backtesting Framework

This module implements backtesting with counterfactual analysis to evaluate
trading strategies and decompose returns into market and strategy components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

from model import (
    CounterfactualEstimator,
    TwinNetworkEstimator,
    PropensityScoreMatching,
    prepare_counterfactual_dataset,
    compute_rsi
)


@dataclass
class TradeDecision:
    """Represents a trading decision with counterfactual analysis."""
    timestamp: pd.Timestamp
    action: int  # 1 = buy, -1 = sell, 0 = hold
    observed_return: float
    counterfactual_return: float
    treatment_effect: float
    confidence: float


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
    max_drawdown: float
    win_rate: float


class CounterfactualTradingStrategy:
    """
    Trading strategy that uses counterfactual reasoning for:
    1. Evaluating past decisions
    2. Making better future decisions
    3. Understanding true strategy performance
    """

    def __init__(
        self,
        counterfactual_estimator: CounterfactualEstimator,
        lookback: int = 100
    ):
        self.cf_estimator = counterfactual_estimator
        self.lookback = lookback
        self.decision_history: List[TradeDecision] = []

    def evaluate_decision(
        self,
        features: np.ndarray,
        action: int,
        observed_return: float,
        timestamp: pd.Timestamp = None
    ) -> TradeDecision:
        """
        Evaluate a trading decision using counterfactual analysis.

        Args:
            features: Market features at decision time
            action: Action taken (1, -1, or 0)
            observed_return: Actual return achieved
            timestamp: Decision timestamp

        Returns:
            TradeDecision with counterfactual analysis
        """
        treatment = 1 if action != 0 else 0

        cf_result = self.cf_estimator.estimate_counterfactual(
            features, treatment, observed_return
        )

        ci_width = cf_result.confidence_interval[1] - cf_result.confidence_interval[0]
        confidence = 1 - min(ci_width, 1)

        decision = TradeDecision(
            timestamp=timestamp or pd.Timestamp.now(),
            action=action,
            observed_return=observed_return,
            counterfactual_return=cf_result.counterfactual_outcome,
            treatment_effect=cf_result.treatment_effect,
            confidence=confidence
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
            'strategy_contribution_pct': (
                strategy_return / (abs(total_return) + 1e-10) * 100
            )
        }

    def identify_best_decisions(self, top_n: int = 10) -> List[TradeDecision]:
        """
        Identify decisions where we made the right call
        (observed return much better than counterfactual).
        """
        return sorted(
            self.decision_history,
            key=lambda d: d.treatment_effect,
            reverse=True
        )[:top_n]

    def identify_worst_decisions(self, top_n: int = 10) -> List[TradeDecision]:
        """
        Identify decisions where we made the wrong call
        (counterfactual return would have been better).
        """
        return sorted(
            self.decision_history,
            key=lambda d: d.treatment_effect,
            reverse=False
        )[:top_n]

    def compute_regret(self) -> Dict[str, float]:
        """
        Compute counterfactual regret metrics.

        Regret = max(0, counterfactual_return - observed_return)
        """
        regrets = [
            max(0, d.counterfactual_return - d.observed_return)
            for d in self.decision_history
        ]

        if not regrets:
            return {'total_regret': 0, 'mean_regret': 0, 'max_regret': 0, 'regret_frequency': 0}

        return {
            'total_regret': sum(regrets),
            'mean_regret': np.mean(regrets),
            'max_regret': max(regrets),
            'regret_frequency': sum(1 for r in regrets if r > 0) / len(regrets)
        }


class CounterfactualBacktester:
    """
    Backtesting framework with counterfactual analysis.
    """

    def __init__(
        self,
        cf_estimator: CounterfactualEstimator = None,
        transaction_cost: float = 0.001
    ):
        self.cf_estimator = cf_estimator or CounterfactualEstimator()
        self.transaction_cost = transaction_cost

    def run(
        self,
        prices: pd.DataFrame,
        strategy_signals: pd.Series,
        train_ratio: float = 0.5
    ) -> CounterfactualBacktestResult:
        """
        Run backtest with counterfactual analysis.

        Args:
            prices: OHLCV data
            strategy_signals: Trading signals
            train_ratio: Ratio of data for training

        Returns:
            CounterfactualBacktestResult with comprehensive metrics
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

        # Max drawdown
        max_dd = self._compute_max_drawdown(outcome_test)

        # Win rate
        win_rate = self._compute_win_rate(outcome_test - cf_outcomes)

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
            decision_analysis=decision_df,
            max_drawdown=max_dd,
            win_rate=win_rate
        )

    def _compute_sharpe(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Compute annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)

    def _compute_max_drawdown(self, returns: np.ndarray) -> float:
        """Compute maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return np.min(drawdowns) if len(drawdowns) > 0 else 0

    def _compute_win_rate(self, effects: np.ndarray) -> float:
        """Compute win rate of treatment effects."""
        if len(effects) == 0:
            return 0.0
        return np.sum(effects > 0) / len(effects)


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

        if np.sum(weights) == 0:
            return 0.0

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
        """
        from sklearn.ensemble import GradientBoostingClassifier

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
        policy_labels = (cate > 0).astype(int)
        self.policy_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.policy_model.fit(X, policy_labels)

        def optimal_policy(x):
            x = np.asarray(x)
            x = x.reshape(1, -1) if x.ndim == 1 else x
            return self.policy_model.predict(x)[0]

        return optimal_policy


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

    print("Counterfactual Backtesting Demo")
    print("=" * 50)

    # Fetch data
    print("\nFetching BTC data...")
    data = fetch_bybit_data("BTCUSDT", "D", 500)
    print(f"Data shape: {data.shape}")

    # Generate RSI-based signals
    rsi = compute_rsi(data['close'], 14)
    signals = pd.Series(
        np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0)),
        index=data.index
    )

    # Run counterfactual backtest
    print("\nRunning counterfactual backtest...")
    backtester = CounterfactualBacktester(transaction_cost=0.001)
    result = backtester.run(data, signals, train_ratio=0.5)

    # Print results
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Counterfactual Return: {result.counterfactual_return:.2%}")
    print(f"Strategy Alpha: {result.strategy_alpha:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Counterfactual Sharpe: {result.counterfactual_sharpe:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Total Regret: {result.regret:.2%}")

    print("\n" + "=" * 50)
    print("ATTRIBUTION ANALYSIS")
    print("=" * 50)
    for key, value in result.attribution.items():
        if 'pct' in key:
            print(f"{key}: {value:.1f}%")
        else:
            print(f"{key}: {value:.4f}")

    print("\n" + "=" * 50)
    print("DECISION ANALYSIS (First 10 Periods)")
    print("=" * 50)
    print(result.decision_analysis.head(10).to_string())

    # Test policy optimizer
    print("\n" + "=" * 50)
    print("POLICY OPTIMIZATION")
    print("=" * 50)

    dataset = prepare_counterfactual_dataset(data, signals)
    feature_cols = ['return_1d', 'return_5d', 'return_20d', 'volatility', 'momentum']

    X = dataset[feature_cols].values
    treatment = dataset['treatment'].values
    outcome = dataset['observed_return'].values

    train_size = int(len(X) * 0.5)
    X_train = X[:train_size]
    treatment_train = treatment[:train_size]
    outcome_train = outcome[:train_size]

    cf_estimator = CounterfactualEstimator()
    cf_estimator.fit(X_train, treatment_train, outcome_train)

    optimizer = CounterfactualPolicyOptimizer(cf_estimator)
    optimal_policy = optimizer.learn_optimal_policy(X_train, treatment_train, outcome_train)

    # Evaluate optimal policy
    X_test = X[train_size:]
    treatment_test = treatment[train_size:]
    outcome_test = outcome[train_size:]

    optimal_value = optimizer.estimate_policy_value(
        X_test, treatment_test, outcome_test, optimal_policy
    )
    print(f"Optimal Policy Value: {optimal_value:.4f}")

    # Compare with original policy
    original_policy = lambda x: 1 if treatment_test[0] == 1 else 0
    original_value = np.mean(outcome_test)
    print(f"Original Policy Value: {original_value:.4f}")
    print(f"Policy Improvement: {(optimal_value - original_value):.4f}")

    print("\nDemo completed successfully!")
