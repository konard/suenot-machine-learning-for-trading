"""
SHAP-based trading model interpretability.

This module implements SHAP (SHapley Additive exPlanations) for explaining
trading model predictions. Based on the paper:
"A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017)
arXiv:1705.07874

SHAP values provide:
- Local explanations: Why did the model make this specific prediction?
- Global explanations: Which features are most important overall?
- Feature interactions: How do features combine to affect predictions?
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SHAPExplanation:
    """Container for SHAP explanation results."""
    shap_values: np.ndarray
    base_value: float
    feature_names: List[str]
    prediction: float

    def top_contributors(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N features by absolute SHAP value."""
        idx = np.argsort(np.abs(self.shap_values))[::-1][:n]
        return [(self.feature_names[i], self.shap_values[i]) for i in idx]

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary of feature contributions."""
        return dict(zip(self.feature_names, self.shap_values))


class SHAPExplainer:
    """
    SHAP explainer for trading models.

    Wraps the shap library to provide model-agnostic explanations
    for any tree-based or gradient boosting model.
    """

    def __init__(
        self,
        model: Any,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model (tree-based recommended).
            background_data: Background dataset for computing expected values.
            feature_names: Names of input features.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("shap library is required. Install with: pip install shap")

        self.model = model
        self.feature_names = feature_names or [f"f{i}" for i in range(background_data.shape[1])]

        # Try TreeExplainer first (faster for tree models), fall back to KernelExplainer
        try:
            self.explainer = shap.TreeExplainer(model, background_data)
            self.explainer_type = "tree"
            logger.info("Using TreeExplainer for fast SHAP computation")
        except Exception:
            self.explainer = shap.KernelExplainer(model.predict_proba, background_data)
            self.explainer_type = "kernel"
            logger.info("Using KernelExplainer for model-agnostic SHAP computation")

    def explain(self, x: np.ndarray) -> SHAPExplanation:
        """
        Compute SHAP values for a single instance.

        Args:
            x: Input features (1D array).

        Returns:
            SHAPExplanation with SHAP values and metadata.
        """
        x = np.atleast_2d(x)
        shap_values = self.explainer.shap_values(x)

        # Handle multi-class output (use positive class for binary)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        # Get base value
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = 0.5

        prediction = float(self.model.predict_proba(x)[0, 1])

        return SHAPExplanation(
            shap_values=shap_values[0],
            base_value=base_value,
            feature_names=self.feature_names,
            prediction=prediction,
        )

    def explain_batch(self, X: np.ndarray) -> List[SHAPExplanation]:
        """Compute SHAP values for multiple instances."""
        return [self.explain(x) for x in X]

    def global_importance(self, X: np.ndarray) -> Dict[str, float]:
        """
        Compute global feature importance as mean(|SHAP|).

        Args:
            X: Input features (2D array).

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        importance = np.abs(shap_values).mean(axis=0)
        return dict(zip(self.feature_names, importance))


class TradingSHAPModel:
    """
    Trading model with built-in SHAP interpretability.

    Combines a gradient boosting classifier with SHAP explanations
    for transparent trading signal generation.
    """

    FEATURE_NAMES = [
        "returns_1", "returns_5", "returns_10", "returns_20",
        "volatility_5", "volatility_10", "volatility_20",
        "rsi_14", "macd", "macd_signal",
        "bb_upper", "bb_lower", "bb_width",
        "volume_ma_ratio", "price_ma_ratio_5", "price_ma_ratio_20",
    ]

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        use_xgboost: bool = True,
    ):
        """
        Initialize trading model.

        Args:
            n_estimators: Number of boosting iterations.
            max_depth: Maximum tree depth.
            learning_rate: Learning rate for boosting.
            use_xgboost: Use XGBoost if available (faster).
        """
        if use_xgboost and XGBOOST_AVAILABLE:
            self.model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            )
            logger.info("Using XGBoost classifier")
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
            )
            logger.info("Using sklearn GradientBoostingClassifier")

        self.scaler = StandardScaler()
        self.explainer: Optional[SHAPExplainer] = None
        self.is_fitted = False

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trading features from OHLCV data.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume].

        Returns:
            DataFrame with computed features.
        """
        features = pd.DataFrame(index=df.index)

        close = df["close"]
        volume = df["volume"]

        # Returns at different horizons
        features["returns_1"] = close.pct_change(1)
        features["returns_5"] = close.pct_change(5)
        features["returns_10"] = close.pct_change(10)
        features["returns_20"] = close.pct_change(20)

        # Volatility (rolling std of returns)
        ret = close.pct_change()
        features["volatility_5"] = ret.rolling(5).std()
        features["volatility_10"] = ret.rolling(10).std()
        features["volatility_20"] = ret.rolling(20).std()

        # RSI
        features["rsi_14"] = self._compute_rsi(close, 14)

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        features["macd"] = ema12 - ema26
        features["macd_signal"] = features["macd"].ewm(span=9).mean()

        # Bollinger Bands
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        features["bb_upper"] = (close - (ma20 + 2 * std20)) / close
        features["bb_lower"] = (close - (ma20 - 2 * std20)) / close
        features["bb_width"] = (4 * std20) / ma20

        # Volume relative to moving average
        vol_ma = volume.rolling(20).mean()
        features["volume_ma_ratio"] = volume / vol_ma

        # Price relative to moving averages
        features["price_ma_ratio_5"] = close / close.rolling(5).mean()
        features["price_ma_ratio_20"] = close / close.rolling(20).mean()

        return features

    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def compute_labels(
        self, df: pd.DataFrame, horizon: int = 5, threshold: float = 0.0
    ) -> pd.Series:
        """
        Compute target labels for classification.

        Args:
            df: DataFrame with close prices.
            horizon: Forward-looking horizon for returns.
            threshold: Minimum return for positive label.

        Returns:
            Binary labels (1 = positive forward return, 0 = negative).
        """
        forward_return = df["close"].pct_change(horizon).shift(-horizon)
        return (forward_return > threshold).astype(int)

    def fit(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        test_size: float = 0.2,
    ) -> Dict[str, float]:
        """
        Fit the trading model.

        Args:
            df: DataFrame with OHLCV data.
            horizon: Forward-looking horizon for labels.
            test_size: Fraction of data for testing.

        Returns:
            Dictionary with training metrics.
        """
        # Compute features and labels
        features = self.compute_features(df)
        labels = self.compute_labels(df, horizon=horizon)

        # Combine and drop NaN rows
        data = features.copy()
        data["label"] = labels
        data = data.dropna()

        X = data[self.FEATURE_NAMES].values
        y = data["label"].values

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Fit model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Create SHAP explainer with background data
        background = X_train_scaled[np.random.choice(len(X_train_scaled), min(100, len(X_train_scaled)), replace=False)]
        self.explainer = SHAPExplainer(
            self.model, background, feature_names=self.FEATURE_NAMES
        )

        # Compute metrics
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)

        logger.info(f"Model trained. Train acc: {train_acc:.3f}, Test acc: {test_acc:.3f}")

        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for new data.

        Args:
            df: DataFrame with OHLCV data.

        Returns:
            Array of predicted probabilities (positive class).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        features = self.compute_features(df)
        X = features[self.FEATURE_NAMES].dropna().values
        X_scaled = self.scaler.transform(X)

        return self.model.predict_proba(X_scaled)[:, 1]

    def explain_signal(self, df: pd.DataFrame, idx: int = -1) -> SHAPExplanation:
        """
        Explain a specific trading signal.

        Args:
            df: DataFrame with OHLCV data.
            idx: Index of the row to explain (-1 for latest).

        Returns:
            SHAPExplanation with feature contributions.
        """
        if self.explainer is None:
            raise ValueError("Model must be fitted before explaining")

        features = self.compute_features(df)
        X = features[self.FEATURE_NAMES].dropna().values
        X_scaled = self.scaler.transform(X)

        return self.explainer.explain(X_scaled[idx])

    def rolling_feature_importance(
        self, df: pd.DataFrame, window: int = 100
    ) -> pd.DataFrame:
        """
        Compute rolling feature importance over time.

        Args:
            df: DataFrame with OHLCV data.
            window: Rolling window size.

        Returns:
            DataFrame with importance scores over time.
        """
        if self.explainer is None:
            raise ValueError("Model must be fitted before computing importance")

        features = self.compute_features(df)
        X = features[self.FEATURE_NAMES].dropna()
        X_scaled = self.scaler.transform(X.values)

        importance_over_time = []
        for i in range(window, len(X_scaled)):
            window_data = X_scaled[i - window:i]
            importance = self.explainer.global_importance(window_data)
            importance["timestamp"] = X.index[i]
            importance_over_time.append(importance)

        return pd.DataFrame(importance_over_time)


def demo():
    """Run a demonstration of the SHAP trading model."""
    from data_loader import BybitDataLoader

    # Load data
    loader = BybitDataLoader()
    data = loader.fetch_klines("BTCUSDT", interval="60", limit=500)

    print(f"\nData source: {data.source}")
    print(f"Data shape: {data.df.shape}")

    # Create and fit model
    model = TradingSHAPModel(n_estimators=50, max_depth=4)
    metrics = model.fit(data.df)

    print(f"\nTraining metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Explain latest signal
    explanation = model.explain_signal(data.df)

    print(f"\nLatest Signal Explanation:")
    print(f"  Prediction: {explanation.prediction:.3f}")
    print(f"  Base value: {explanation.base_value:.3f}")
    print(f"\n  Top contributing features:")
    for name, value in explanation.top_contributors(5):
        direction = "+" if value > 0 else ""
        print(f"    {name}: {direction}{value:.4f}")

    # Global feature importance
    features = model.compute_features(data.df)
    X = features[model.FEATURE_NAMES].dropna().values
    X_scaled = model.scaler.transform(X)
    importance = model.explainer.global_importance(X_scaled[-100:])

    print(f"\n  Global feature importance (last 100 samples):")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for name, value in sorted_importance[:5]:
        print(f"    {name}: {value:.4f}")


if __name__ == "__main__":
    demo()
