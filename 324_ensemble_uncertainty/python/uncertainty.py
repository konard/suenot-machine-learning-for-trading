"""
Uncertainty Quantification module.

This module provides methods for quantifying and analyzing
uncertainty in ensemble predictions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UncertaintyQuantifier:
    """
    Quantifies uncertainty from ensemble predictions.

    Supports:
    - Epistemic uncertainty (model uncertainty)
    - Aleatoric uncertainty (data uncertainty)
    - Total uncertainty decomposition
    - Various uncertainty metrics
    """

    def __init__(self):
        """Initialize the uncertainty quantifier."""
        pass

    def prediction_variance(
        self,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Compute variance of predictions across ensemble members.

        Args:
            predictions: Array of shape (n_models, n_samples) or (n_models, n_samples, n_outputs)

        Returns:
            Variance array of shape (n_samples,) or (n_samples, n_outputs)
        """
        return np.var(predictions, axis=0)

    def prediction_std(
        self,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Compute standard deviation of predictions across ensemble members.

        Args:
            predictions: Array of shape (n_models, n_samples)

        Returns:
            Standard deviation array
        """
        return np.std(predictions, axis=0)

    def coefficient_of_variation(
        self,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Compute coefficient of variation (relative uncertainty).

        Args:
            predictions: Array of shape (n_models, n_samples)

        Returns:
            CV array (std / |mean|)
        """
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        return std_pred / (np.abs(mean_pred) + 1e-8)

    def interquartile_range(
        self,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Compute interquartile range of predictions.

        Args:
            predictions: Array of shape (n_models, n_samples)

        Returns:
            IQR array (Q75 - Q25)
        """
        q25 = np.percentile(predictions, 25, axis=0)
        q75 = np.percentile(predictions, 75, axis=0)
        return q75 - q25

    def confidence_interval(
        self,
        predictions: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute confidence interval from ensemble predictions.

        Args:
            predictions: Array of shape (n_models, n_samples)
            confidence: Confidence level (0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound) arrays
        """
        alpha = 1 - confidence
        lower = np.percentile(predictions, alpha/2 * 100, axis=0)
        upper = np.percentile(predictions, (1 - alpha/2) * 100, axis=0)
        return lower, upper

    def prediction_entropy(
        self,
        probabilities: np.ndarray
    ) -> np.ndarray:
        """
        Compute entropy of averaged probability predictions (classification).

        Args:
            probabilities: Array of shape (n_models, n_samples, n_classes)

        Returns:
            Entropy array of shape (n_samples,)
        """
        mean_probs = np.mean(probabilities, axis=0)
        # Clip to avoid log(0)
        mean_probs = np.clip(mean_probs, 1e-10, 1 - 1e-10)
        entropy = -np.sum(mean_probs * np.log(mean_probs), axis=-1)
        return entropy

    def mutual_information(
        self,
        probabilities: np.ndarray
    ) -> np.ndarray:
        """
        Compute mutual information (epistemic uncertainty for classification).

        I(y; w | x) = H(y | x) - E_w[H(y | x, w)]

        Args:
            probabilities: Array of shape (n_models, n_samples, n_classes)

        Returns:
            Mutual information array of shape (n_samples,)
        """
        # Total entropy
        total_entropy = self.prediction_entropy(probabilities)

        # Expected entropy (average of individual model entropies)
        probs_clipped = np.clip(probabilities, 1e-10, 1 - 1e-10)
        individual_entropies = -np.sum(probs_clipped * np.log(probs_clipped), axis=-1)
        expected_entropy = np.mean(individual_entropies, axis=0)

        # Mutual information
        return total_entropy - expected_entropy

    def decompose_uncertainty(
        self,
        predictions: np.ndarray,
        leaf_variances: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Decompose total uncertainty into epistemic and aleatoric components.

        Args:
            predictions: Array of shape (n_models, n_samples)
            leaf_variances: Optional per-tree leaf variances for aleatoric estimate

        Returns:
            Dictionary with 'epistemic', 'aleatoric', and 'total' uncertainties
        """
        # Total variance
        total_var = np.var(predictions, axis=0)

        if leaf_variances is not None:
            # Aleatoric from leaf variances
            aleatoric_var = np.mean(leaf_variances, axis=0)
            # Epistemic = Total - Aleatoric
            epistemic_var = np.maximum(total_var - aleatoric_var, 0)
        else:
            # Estimate epistemic as prediction variance
            epistemic_var = total_var
            # Aleatoric estimated from residual spread (if we had targets)
            aleatoric_var = np.zeros_like(total_var)

        return {
            'epistemic': np.sqrt(epistemic_var),
            'aleatoric': np.sqrt(aleatoric_var),
            'total': np.sqrt(total_var),
        }

    def compute_all_metrics(
        self,
        predictions: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute all uncertainty metrics.

        Args:
            predictions: Array of shape (n_models, n_samples)

        Returns:
            Dictionary with all uncertainty metrics
        """
        mean_pred = np.mean(predictions, axis=0)
        lower, upper = self.confidence_interval(predictions)

        return {
            'mean': mean_pred,
            'std': self.prediction_std(predictions),
            'variance': self.prediction_variance(predictions),
            'cv': self.coefficient_of_variation(predictions),
            'iqr': self.interquartile_range(predictions),
            'ci_lower': lower,
            'ci_upper': upper,
            'ci_width': upper - lower,
        }


class OOBUncertaintyEstimator:
    """
    Estimates uncertainty using Out-of-Bag (OOB) samples.
    """

    def __init__(self):
        """Initialize the OOB uncertainty estimator."""
        pass

    def compute_oob_uncertainty(
        self,
        trees: List,
        X_train: np.ndarray,
        bootstrap_indices: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute OOB predictions and uncertainty.

        Args:
            trees: List of trained trees
            X_train: Training feature matrix
            bootstrap_indices: List of bootstrap sample indices per tree

        Returns:
            Tuple of (oob_predictions, oob_uncertainty)
        """
        n_samples = X_train.shape[0]
        n_trees = len(trees)

        # Track which trees each sample is OOB for
        oob_predictions = [[] for _ in range(n_samples)]

        for tree_idx, (tree, indices) in enumerate(zip(trees, bootstrap_indices)):
            # Get OOB samples for this tree
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[indices] = False

            # Predict OOB samples
            oob_samples = np.where(oob_mask)[0]
            if len(oob_samples) > 0:
                preds = tree.predict(X_train[oob_samples])
                for sample_idx, pred in zip(oob_samples, preds):
                    oob_predictions[sample_idx].append(pred)

        # Compute mean and std
        oob_mean = np.array([
            np.mean(preds) if len(preds) > 0 else np.nan
            for preds in oob_predictions
        ])
        oob_std = np.array([
            np.std(preds) if len(preds) > 1 else np.nan
            for preds in oob_predictions
        ])

        return oob_mean, oob_std


class UncertaintyAnalyzer:
    """
    Analyzes and visualizes uncertainty in predictions.
    """

    def __init__(self, quantifier: Optional[UncertaintyQuantifier] = None):
        """Initialize the analyzer."""
        self.quantifier = quantifier or UncertaintyQuantifier()

    def uncertainty_summary(
        self,
        predictions: np.ndarray,
        actuals: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Generate summary statistics for uncertainty.

        Args:
            predictions: Array of shape (n_models, n_samples)
            actuals: Optional actual values for error analysis

        Returns:
            Summary DataFrame
        """
        metrics = self.quantifier.compute_all_metrics(predictions)

        summary = {
            'Metric': [],
            'Mean': [],
            'Std': [],
            'Min': [],
            'Max': [],
        }

        for name, values in metrics.items():
            if name != 'mean':
                summary['Metric'].append(name)
                summary['Mean'].append(np.nanmean(values))
                summary['Std'].append(np.nanstd(values))
                summary['Min'].append(np.nanmin(values))
                summary['Max'].append(np.nanmax(values))

        df = pd.DataFrame(summary)

        # Add error analysis if actuals provided
        if actuals is not None:
            mean_pred = metrics['mean']
            errors = actuals - mean_pred
            uncertainties = metrics['std']

            # Coverage: % of actuals within 2 std
            coverage = np.mean(np.abs(errors) <= 2 * uncertainties)

            # Correlation between uncertainty and error
            correlation = np.corrcoef(uncertainties, np.abs(errors))[0, 1]

            extra_metrics = pd.DataFrame({
                'Metric': ['Coverage (2 std)', 'Uncertainty-Error Corr'],
                'Mean': [coverage, correlation],
                'Std': [np.nan, np.nan],
                'Min': [np.nan, np.nan],
                'Max': [np.nan, np.nan],
            })
            df = pd.concat([df, extra_metrics], ignore_index=True)

        return df

    def stratify_by_uncertainty(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = 5
    ) -> pd.DataFrame:
        """
        Stratify predictions by uncertainty level and analyze.

        Args:
            predictions: Array of shape (n_models, n_samples)
            actuals: Actual values
            n_bins: Number of uncertainty bins

        Returns:
            DataFrame with stratified analysis
        """
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        errors = actuals - mean_pred

        # Create bins
        bins = np.percentile(uncertainty, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(uncertainty, bins[1:-1])

        results = []
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if mask.sum() > 0:
                results.append({
                    'Uncertainty Bin': bin_idx + 1,
                    'Count': mask.sum(),
                    'Mean Uncertainty': uncertainty[mask].mean(),
                    'Mean Abs Error': np.abs(errors[mask]).mean(),
                    'RMSE': np.sqrt(np.mean(errors[mask]**2)),
                    'Coverage': np.mean(np.abs(errors[mask]) <= 2 * uncertainty[mask]),
                })

        return pd.DataFrame(results)


def main():
    """Example usage of uncertainty quantification."""
    np.random.seed(42)

    # Simulate ensemble predictions
    n_models = 10
    n_samples = 100

    # True values
    true_values = np.random.randn(n_samples) * 2

    # Simulated model predictions with varying uncertainty
    predictions = np.array([
        true_values + np.random.randn(n_samples) * (0.3 + 0.2 * np.abs(true_values))
        for _ in range(n_models)
    ])

    print("=" * 60)
    print("Uncertainty Quantification")
    print("=" * 60)

    quantifier = UncertaintyQuantifier()

    # Compute metrics
    metrics = quantifier.compute_all_metrics(predictions)
    print("\nComputed metrics:")
    for name, values in metrics.items():
        print(f"  {name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

    # Decompose uncertainty
    decomposition = quantifier.decompose_uncertainty(predictions)
    print("\nUncertainty decomposition:")
    for name, values in decomposition.items():
        print(f"  {name}: mean={np.mean(values):.4f}")

    # Confidence intervals
    lower, upper = quantifier.confidence_interval(predictions)
    coverage = np.mean((true_values >= lower) & (true_values <= upper))
    print(f"\n95% CI coverage: {coverage:.2%}")

    print("\n" + "=" * 60)
    print("Uncertainty Analysis")
    print("=" * 60)

    analyzer = UncertaintyAnalyzer()

    # Summary
    summary = analyzer.uncertainty_summary(predictions, true_values)
    print("\nUncertainty Summary:")
    print(summary.to_string())

    # Stratified analysis
    print("\nStratified Analysis by Uncertainty:")
    stratified = analyzer.stratify_by_uncertainty(predictions, true_values, n_bins=5)
    print(stratified.to_string())

    print("\n" + "=" * 60)
    print("Classification Uncertainty (Simulated)")
    print("=" * 60)

    # Simulate probability predictions
    n_classes = 3
    prob_predictions = np.random.dirichlet(
        np.ones(n_classes),
        size=(n_models, n_samples)
    )

    # Entropy
    entropy = quantifier.prediction_entropy(prob_predictions)
    print(f"\nMean entropy: {np.mean(entropy):.4f}")

    # Mutual information
    mi = quantifier.mutual_information(prob_predictions)
    print(f"Mean mutual information (epistemic): {np.mean(mi):.4f}")


if __name__ == '__main__':
    main()
