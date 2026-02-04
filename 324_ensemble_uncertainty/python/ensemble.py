"""
Ensemble Models module for trading with uncertainty quantification.

This module provides implementations of various ensemble methods
including Random Forest, Gradient Boosting, and Stacking.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.model_selection import cross_val_predict
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleModel(ABC):
    """
    Abstract base class for ensemble models with uncertainty quantification.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleModel':
        """Fit the ensemble model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        pass


class RandomForestEnsemble(EnsembleModel):
    """
    Random Forest ensemble with uncertainty quantification.

    Provides:
    - Prediction variance from tree disagreement
    - Out-of-bag (OOB) uncertainty estimates
    - Quantile predictions for confidence intervals
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float] = 'sqrt',
        n_jobs: int = -1,
        random_state: int = 42,
        task: str = 'regression'
    ):
        """
        Initialize Random Forest ensemble.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
            max_features: Number of features to consider
            n_jobs: Number of parallel jobs
            random_state: Random seed
            task: 'regression' or 'classification'
        """
        self.n_estimators = n_estimators
        self.task = task
        self.random_state = random_state

        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'n_jobs': n_jobs,
            'random_state': random_state,
            'oob_score': True,
        }

        if task == 'regression':
            self.model = RandomForestRegressor(**params)
        else:
            self.model = RandomForestClassifier(**params)

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestEnsemble':
        """
        Fit the Random Forest model.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Self
        """
        logger.info(f"Fitting Random Forest with {self.n_estimators} trees...")
        self.model.fit(X, y)
        self.is_fitted = True

        if hasattr(self.model, 'oob_score_'):
            logger.info(f"OOB Score: {self.model.oob_score_:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates from tree variance.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (mean_predictions, uncertainty_std)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get predictions from all trees
        tree_predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ])

        # Mean prediction
        mean_pred = np.mean(tree_predictions, axis=0)

        # Uncertainty as standard deviation
        uncertainty = np.std(tree_predictions, axis=0)

        return mean_pred, uncertainty

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ) -> Dict[float, np.ndarray]:
        """
        Predict quantiles for confidence intervals.

        Args:
            X: Feature matrix
            quantiles: List of quantiles to compute

        Returns:
            Dictionary mapping quantile to predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get predictions from all trees
        tree_predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ])

        # Compute quantiles
        results = {}
        for q in quantiles:
            results[q] = np.percentile(tree_predictions, q * 100, axis=0)

        return results

    def get_oob_predictions(self) -> Optional[np.ndarray]:
        """Get out-of-bag predictions if available."""
        if hasattr(self.model, 'oob_prediction_'):
            return self.model.oob_prediction_
        return None

    def feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get feature importances.

        Args:
            feature_names: Optional list of feature names

        Returns:
            DataFrame with feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        importances = self.model.feature_importances_

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return df


class GradientBoostingEnsemble(EnsembleModel):
    """
    Gradient Boosting ensemble with uncertainty quantification.

    Supports quantile regression for uncertainty estimation.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        task: str = 'regression',
        use_quantile: bool = True
    ):
        """
        Initialize Gradient Boosting ensemble.

        Args:
            n_estimators: Number of boosting stages
            learning_rate: Shrinkage parameter
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
            random_state: Random seed
            task: 'regression' or 'classification'
            use_quantile: Whether to train quantile models for uncertainty
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.task = task
        self.random_state = random_state
        self.use_quantile = use_quantile

        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state,
        }

        if task == 'regression':
            self.model = GradientBoostingRegressor(**params)

            # Quantile models for uncertainty
            if use_quantile:
                self.model_q10 = GradientBoostingRegressor(
                    **{**params, 'loss': 'quantile', 'alpha': 0.1}
                )
                self.model_q90 = GradientBoostingRegressor(
                    **{**params, 'loss': 'quantile', 'alpha': 0.9}
                )
        else:
            self.model = GradientBoostingClassifier(**params)

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingEnsemble':
        """
        Fit the Gradient Boosting model.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Self
        """
        logger.info(f"Fitting Gradient Boosting with {self.n_estimators} stages...")
        self.model.fit(X, y)

        if self.use_quantile and self.task == 'regression':
            logger.info("Fitting quantile models for uncertainty...")
            self.model_q10.fit(X, y)
            self.model_q90.fit(X, y)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty from quantile models.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (predictions, uncertainty_std)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions = self.model.predict(X)

        if self.use_quantile and self.task == 'regression':
            q10 = self.model_q10.predict(X)
            q90 = self.model_q90.predict(X)
            # Estimate std from IQR (assuming normal distribution)
            uncertainty = (q90 - q10) / 2.56  # 2.56 = 2 * 1.28 (80% CI)
        else:
            # Fallback: use staged predict for variance estimate
            staged_preds = list(self.model.staged_predict(X))
            uncertainty = np.std(staged_preds[-10:], axis=0)

        return predictions, uncertainty

    def predict_interval(
        self,
        X: np.ndarray,
        confidence: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence interval.

        Args:
            X: Feature matrix
            confidence: Confidence level (0.8 for 80% CI)

        Returns:
            Tuple of (median, lower_bound, upper_bound)
        """
        if not self.is_fitted or not self.use_quantile:
            raise ValueError("Model not fitted with quantile regression.")

        median = self.model.predict(X)
        lower = self.model_q10.predict(X)
        upper = self.model_q90.predict(X)

        return median, lower, upper


class StackingEnsemble(EnsembleModel):
    """
    Stacking ensemble that combines multiple base models.

    Provides uncertainty from model disagreement.
    """

    def __init__(
        self,
        base_models: Optional[List[EnsembleModel]] = None,
        meta_model: Optional[EnsembleModel] = None,
        use_probabilities: bool = False,
        random_state: int = 42
    ):
        """
        Initialize Stacking ensemble.

        Args:
            base_models: List of base models
            meta_model: Meta-learner model
            use_probabilities: Whether to use probabilities (for classification)
            random_state: Random seed
        """
        if base_models is None:
            base_models = [
                RandomForestEnsemble(n_estimators=50, random_state=random_state),
                GradientBoostingEnsemble(n_estimators=50, random_state=random_state),
            ]

        if meta_model is None:
            meta_model = RandomForestEnsemble(n_estimators=50, random_state=random_state)

        self.base_models = base_models
        self.meta_model = meta_model
        self.use_probabilities = use_probabilities
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble using cross-validation.

        Args:
            X: Feature matrix
            y: Target values
            cv: Number of cross-validation folds

        Returns:
            Self
        """
        logger.info(f"Fitting Stacking ensemble with {len(self.base_models)} base models...")

        # Get out-of-fold predictions from base models
        meta_features = []

        for i, model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}...")
            model.fit(X, y)

            # Cross-validation predictions for meta-features
            if hasattr(model.model, 'predict'):
                oof_preds = cross_val_predict(
                    model.model, X, y, cv=cv, method='predict'
                )
                meta_features.append(oof_preds)

        # Stack meta-features
        X_meta = np.column_stack(meta_features)

        # Fit meta-model
        logger.info("Training meta-model...")
        self.meta_model.fit(X_meta, y)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the stacking ensemble.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get base model predictions
        base_preds = np.column_stack([
            model.predict(X) for model in self.base_models
        ])

        # Meta-model prediction
        return self.meta_model.predict(base_preds)

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty from model disagreement.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (predictions, uncertainty_std)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get base model predictions
        base_preds = np.array([model.predict(X) for model in self.base_models])

        # Uncertainty from base model disagreement
        base_uncertainty = np.std(base_preds, axis=0)

        # Stack for meta-model
        X_meta = base_preds.T

        # Meta-model prediction with its own uncertainty
        meta_pred, meta_uncertainty = self.meta_model.predict_with_uncertainty(X_meta)

        # Combine uncertainties
        total_uncertainty = np.sqrt(base_uncertainty**2 + meta_uncertainty**2)

        return meta_pred, total_uncertainty


def main():
    """Example usage of ensemble models."""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5] * [0.3, 0.2, 0.15, 0.1, 0.25], axis=1) + \
        np.random.randn(n_samples) * 0.5

    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print("=" * 60)
    print("Random Forest Ensemble")
    print("=" * 60)

    rf = RandomForestEnsemble(n_estimators=100)
    rf.fit(X_train, y_train)

    # Predict with uncertainty
    pred_rf, uncertainty_rf = rf.predict_with_uncertainty(X_test)
    print(f"\nPredictions shape: {pred_rf.shape}")
    print(f"Mean uncertainty: {uncertainty_rf.mean():.4f}")
    print(f"Uncertainty range: [{uncertainty_rf.min():.4f}, {uncertainty_rf.max():.4f}]")

    # Quantile predictions
    quantiles = rf.predict_quantiles(X_test, [0.1, 0.5, 0.9])
    print(f"\nQuantile predictions computed for: {list(quantiles.keys())}")

    # Feature importance
    print(f"\nTop 5 feature importances:")
    print(rf.feature_importance().head())

    print("\n" + "=" * 60)
    print("Gradient Boosting Ensemble")
    print("=" * 60)

    gb = GradientBoostingEnsemble(n_estimators=100, use_quantile=True)
    gb.fit(X_train, y_train)

    # Predict with uncertainty
    pred_gb, uncertainty_gb = gb.predict_with_uncertainty(X_test)
    print(f"\nMean uncertainty: {uncertainty_gb.mean():.4f}")

    # Confidence interval
    median, lower, upper = gb.predict_interval(X_test)
    print(f"80% CI width (mean): {(upper - lower).mean():.4f}")

    print("\n" + "=" * 60)
    print("Stacking Ensemble")
    print("=" * 60)

    stack = StackingEnsemble()
    stack.fit(X_train, y_train, cv=3)

    # Predict with uncertainty
    pred_stack, uncertainty_stack = stack.predict_with_uncertainty(X_test)
    print(f"\nMean uncertainty: {uncertainty_stack.mean():.4f}")

    # Compare predictions
    from sklearn.metrics import mean_squared_error, r2_score

    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)

    for name, preds in [('Random Forest', pred_rf),
                         ('Gradient Boosting', pred_gb),
                         ('Stacking', pred_stack)]:
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"{name}: MSE={mse:.4f}, R2={r2:.4f}")


if __name__ == '__main__':
    main()
