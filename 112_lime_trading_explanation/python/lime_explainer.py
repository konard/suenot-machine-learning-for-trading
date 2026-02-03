"""
LIME Explainer for Trading Models

Provides LIME (Local Interpretable Model-agnostic Explanations) for
explaining predictions from trading models.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass


@dataclass
class LimeExplanation:
    """
    Container for LIME explanation results.

    Attributes:
        instance: The original instance being explained
        prediction: Model's prediction for the instance
        prediction_proba: Prediction probability
        feature_contributions: Dict mapping feature names to their contributions
        intercept: Intercept of the local linear model
        local_model_score: R² score of the local linear model
        feature_names: List of feature names
    """
    instance: np.ndarray
    prediction: int
    prediction_proba: float
    feature_contributions: Dict[str, float]
    intercept: float
    local_model_score: float
    feature_names: List[str]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert feature contributions to a DataFrame."""
        df = pd.DataFrame({
            'feature': list(self.feature_contributions.keys()),
            'contribution': list(self.feature_contributions.values())
        })
        df = df.sort_values('contribution', key=abs, ascending=False)
        return df

    def get_top_features(self, n: int = 5) -> Dict[str, float]:
        """Get top n features by absolute contribution."""
        sorted_features = sorted(
            self.feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return dict(sorted_features[:n])

    def __str__(self) -> str:
        """String representation of the explanation."""
        lines = [
            f"Prediction: {'UP' if self.prediction == 1 else 'DOWN'} "
            f"(probability: {self.prediction_proba:.2%})",
            "",
            "Feature Contributions:"
        ]

        df = self.to_dataframe()
        for _, row in df.iterrows():
            sign = '+' if row['contribution'] >= 0 else ''
            lines.append(f"  {sign}{row['contribution']:.4f}  {row['feature']}")

        lines.append(f"\nLocal model R²: {self.local_model_score:.4f}")
        return '\n'.join(lines)


class LimeExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for trading models.

    This implementation generates explanations by:
    1. Creating perturbations around the instance to explain
    2. Getting predictions from the model for all perturbations
    3. Weighting perturbations by proximity to the original instance
    4. Fitting a local linear model on the weighted perturbations
    5. Using the linear model coefficients as feature contributions
    """

    def __init__(
        self,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: List[str],
        training_data: Optional[np.ndarray] = None,
        num_samples: int = 5000,
        kernel_width: Optional[float] = None,
        random_state: int = 42
    ):
        """
        Initialize the LIME explainer.

        Args:
            model_predict_fn: Function that takes features and returns probabilities
            feature_names: List of feature names
            training_data: Training data used for computing statistics (for perturbations)
            num_samples: Number of perturbed samples to generate
            kernel_width: Width of the exponential kernel (default: sqrt(num_features) * 0.75)
            random_state: Random seed for reproducibility
        """
        self.model_predict_fn = model_predict_fn
        self.feature_names = feature_names
        self.num_features = len(feature_names)
        self.num_samples = num_samples
        self.random_state = random_state

        np.random.seed(random_state)

        if kernel_width is None:
            self.kernel_width = np.sqrt(self.num_features) * 0.75
        else:
            self.kernel_width = kernel_width

        if training_data is not None:
            self.feature_means = np.mean(training_data, axis=0)
            self.feature_stds = np.std(training_data, axis=0)
            self.feature_stds[self.feature_stds == 0] = 1.0
        else:
            self.feature_means = None
            self.feature_stds = None

    def _generate_perturbations(self, instance: np.ndarray) -> np.ndarray:
        """
        Generate perturbed samples around the instance.

        Args:
            instance: The original instance (1D array)

        Returns:
            Array of perturbed samples (num_samples x num_features)
        """
        perturbations = np.zeros((self.num_samples, self.num_features))
        perturbations[0] = instance

        if self.feature_stds is not None:
            noise = np.random.normal(0, 1, (self.num_samples - 1, self.num_features))
            perturbations[1:] = instance + noise * self.feature_stds * 0.5
        else:
            std_estimate = np.abs(instance) * 0.1 + 0.01
            noise = np.random.normal(0, 1, (self.num_samples - 1, self.num_features))
            perturbations[1:] = instance + noise * std_estimate

        return perturbations

    def _compute_distances(self, instance: np.ndarray, perturbations: np.ndarray) -> np.ndarray:
        """
        Compute distances between instance and perturbations.

        Args:
            instance: Original instance
            perturbations: Perturbed samples

        Returns:
            Array of distances
        """
        if self.feature_stds is not None:
            scaled_instance = instance / self.feature_stds
            scaled_perturbations = perturbations / self.feature_stds
        else:
            scaled_instance = instance
            scaled_perturbations = perturbations

        distances = np.sqrt(np.sum((scaled_perturbations - scaled_instance) ** 2, axis=1))
        return distances

    def _compute_weights(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute kernel weights from distances.

        Args:
            distances: Array of distances

        Returns:
            Array of weights
        """
        weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))
        return weights

    def _fit_local_model(
        self,
        perturbations: np.ndarray,
        predictions: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Fit a weighted linear regression model.

        Args:
            perturbations: Perturbed samples
            predictions: Model predictions for perturbations
            weights: Sample weights

        Returns:
            Tuple of (coefficients, intercept, R² score)
        """
        try:
            from sklearn.linear_model import Ridge
            from sklearn.metrics import r2_score
        except ImportError:
            raise ImportError("Please install scikit-learn")

        model = Ridge(alpha=1.0, fit_intercept=True)
        model.fit(perturbations, predictions, sample_weight=weights)

        predicted = model.predict(perturbations)
        weighted_r2 = 1 - np.sum(weights * (predictions - predicted) ** 2) / \
                         np.sum(weights * (predictions - np.average(predictions, weights=weights)) ** 2)

        return model.coef_, model.intercept_, weighted_r2

    def explain(
        self,
        instance: np.ndarray,
        class_to_explain: int = 1
    ) -> LimeExplanation:
        """
        Generate a LIME explanation for a single instance.

        Args:
            instance: Feature values for the instance to explain (1D array)
            class_to_explain: Which class probability to explain (default: 1 = UP)

        Returns:
            LimeExplanation object
        """
        if isinstance(instance, pd.DataFrame):
            instance = instance.values.flatten()
        elif isinstance(instance, pd.Series):
            instance = instance.values

        instance = np.array(instance).flatten()

        perturbations = self._generate_perturbations(instance)

        proba = self.model_predict_fn(perturbations)
        if len(proba.shape) == 2:
            predictions = proba[:, class_to_explain]
        else:
            predictions = proba

        distances = self._compute_distances(instance, perturbations)
        weights = self._compute_weights(distances)

        coefficients, intercept, r2_score = self._fit_local_model(
            perturbations, predictions, weights
        )

        feature_contributions = {
            name: coef * instance[i]
            for i, (name, coef) in enumerate(zip(self.feature_names, coefficients))
        }

        original_proba = self.model_predict_fn(instance.reshape(1, -1))
        if len(original_proba.shape) == 2:
            prediction_proba = original_proba[0, class_to_explain]
            prediction = int(original_proba[0, 1] > 0.5)
        else:
            prediction_proba = original_proba[0]
            prediction = int(prediction_proba > 0.5)

        return LimeExplanation(
            instance=instance,
            prediction=prediction,
            prediction_proba=prediction_proba,
            feature_contributions=feature_contributions,
            intercept=intercept,
            local_model_score=r2_score,
            feature_names=self.feature_names
        )

    def explain_batch(
        self,
        instances: np.ndarray,
        class_to_explain: int = 1
    ) -> List[LimeExplanation]:
        """
        Generate LIME explanations for multiple instances.

        Args:
            instances: Feature values for instances to explain (2D array)
            class_to_explain: Which class probability to explain

        Returns:
            List of LimeExplanation objects
        """
        if isinstance(instances, pd.DataFrame):
            instances = instances.values

        explanations = []
        for i in range(len(instances)):
            explanation = self.explain(instances[i], class_to_explain)
            explanations.append(explanation)

        return explanations


class LimeAnalyzer:
    """
    Utilities for analyzing LIME explanations.
    """

    @staticmethod
    def aggregate_explanations(
        explanations: List[LimeExplanation]
    ) -> pd.DataFrame:
        """
        Aggregate feature contributions across multiple explanations.

        Args:
            explanations: List of LimeExplanation objects

        Returns:
            DataFrame with aggregated statistics per feature
        """
        all_contributions = {name: [] for name in explanations[0].feature_names}

        for exp in explanations:
            for name, contrib in exp.feature_contributions.items():
                all_contributions[name].append(contrib)

        results = []
        for name, contribs in all_contributions.items():
            results.append({
                'feature': name,
                'mean_contribution': np.mean(contribs),
                'std_contribution': np.std(contribs),
                'mean_abs_contribution': np.mean(np.abs(contribs)),
                'positive_count': sum(1 for c in contribs if c > 0),
                'negative_count': sum(1 for c in contribs if c < 0)
            })

        df = pd.DataFrame(results)
        df = df.sort_values('mean_abs_contribution', ascending=False)
        return df

    @staticmethod
    def explanation_consistency(
        explanations: List[LimeExplanation],
        top_n: int = 5
    ) -> float:
        """
        Measure consistency of top features across explanations.

        Args:
            explanations: List of LimeExplanation objects
            top_n: Number of top features to consider

        Returns:
            Consistency score (0 to 1)
        """
        top_features_lists = []
        for exp in explanations:
            top = list(exp.get_top_features(top_n).keys())
            top_features_lists.append(set(top))

        total_overlap = 0
        comparisons = 0

        for i in range(len(top_features_lists)):
            for j in range(i + 1, len(top_features_lists)):
                overlap = len(top_features_lists[i] & top_features_lists[j])
                total_overlap += overlap / top_n
                comparisons += 1

        if comparisons == 0:
            return 1.0

        return total_overlap / comparisons

    @staticmethod
    def filter_by_confidence(
        explanations: List[LimeExplanation],
        min_r2: float = 0.5
    ) -> List[LimeExplanation]:
        """
        Filter explanations by local model quality.

        Args:
            explanations: List of LimeExplanation objects
            min_r2: Minimum R² score for the local model

        Returns:
            Filtered list of explanations
        """
        return [exp for exp in explanations if exp.local_model_score >= min_r2]


def create_lime_explainer_for_model(
    model,
    X_train: pd.DataFrame
) -> LimeExplainer:
    """
    Factory function to create a LIME explainer for a trading model.

    Args:
        model: Trading model with predict_proba method
        X_train: Training data DataFrame

    Returns:
        LimeExplainer instance
    """
    def predict_fn(X):
        return model.predict_proba(X)

    return LimeExplainer(
        model_predict_fn=predict_fn,
        feature_names=list(X_train.columns),
        training_data=X_train.values
    )
