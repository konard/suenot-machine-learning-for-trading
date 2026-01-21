"""
Evaluation metrics for regime classification.

This module provides tools to evaluate the accuracy and usefulness
of regime classification models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

from .classifier import MarketRegime, RegimeResult


@dataclass
class ClassificationMetrics:
    """Metrics for regime classification evaluation."""
    accuracy: float
    precision: Dict[MarketRegime, float]
    recall: Dict[MarketRegime, float]
    f1_score: Dict[MarketRegime, float]
    confusion_matrix: pd.DataFrame
    regime_distribution: Dict[MarketRegime, float]


@dataclass
class TimingMetrics:
    """Metrics for regime transition timing."""
    avg_detection_delay: float  # Days
    false_transitions: int
    missed_transitions: int
    correct_transitions: int
    transition_accuracy: float


class RegimeEvaluator:
    """
    Evaluate regime classification performance.

    Compares predicted regimes against ground truth or
    evaluates based on forward returns.
    """

    def __init__(self):
        """Initialize evaluator."""
        self.regimes = list(MarketRegime)

    def evaluate_vs_ground_truth(
        self,
        predictions: List[RegimeResult],
        ground_truth: List[MarketRegime]
    ) -> ClassificationMetrics:
        """
        Evaluate predictions against ground truth labels.

        Args:
            predictions: Predicted regime results
            ground_truth: True regime labels

        Returns:
            ClassificationMetrics with evaluation results
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground_truth must have same length")

        pred_regimes = [p.regime for p in predictions]

        # Calculate accuracy
        correct = sum(1 for p, g in zip(pred_regimes, ground_truth) if p == g)
        accuracy = correct / len(predictions)

        # Calculate per-regime metrics
        precision = {}
        recall = {}
        f1_score = {}

        for regime in self.regimes:
            # True positives, false positives, false negatives
            tp = sum(1 for p, g in zip(pred_regimes, ground_truth)
                     if p == regime and g == regime)
            fp = sum(1 for p, g in zip(pred_regimes, ground_truth)
                     if p == regime and g != regime)
            fn = sum(1 for p, g in zip(pred_regimes, ground_truth)
                     if p != regime and g == regime)

            # Precision
            precision[regime] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # Recall
            recall[regime] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # F1 Score
            if precision[regime] + recall[regime] > 0:
                f1_score[regime] = 2 * (precision[regime] * recall[regime]) / \
                                   (precision[regime] + recall[regime])
            else:
                f1_score[regime] = 0.0

        # Confusion matrix
        confusion = self._build_confusion_matrix(pred_regimes, ground_truth)

        # Regime distribution
        regime_counts = Counter(pred_regimes)
        total = len(predictions)
        distribution = {r: regime_counts.get(r, 0) / total for r in self.regimes}

        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            confusion_matrix=confusion,
            regime_distribution=distribution
        )

    def evaluate_forward_returns(
        self,
        predictions: List[RegimeResult],
        returns: pd.Series,
        forward_window: int = 5
    ) -> Dict:
        """
        Evaluate regimes based on forward returns.

        Checks if regime classifications align with subsequent
        market performance.

        Args:
            predictions: Predicted regime results
            returns: Return series
            forward_window: Days to look forward

        Returns:
            Dictionary with return analysis by regime
        """
        results = {regime: [] for regime in self.regimes}

        for i, pred in enumerate(predictions):
            # Get forward returns
            if i + forward_window < len(returns):
                fwd_return = returns.iloc[i:i+forward_window].sum()
                results[pred.regime].append(fwd_return)

        # Calculate statistics
        analysis = {}
        for regime in self.regimes:
            regime_returns = results[regime]
            if regime_returns:
                analysis[regime] = {
                    'count': len(regime_returns),
                    'avg_forward_return': np.mean(regime_returns),
                    'std_forward_return': np.std(regime_returns),
                    'positive_pct': sum(1 for r in regime_returns if r > 0) / len(regime_returns),
                    'avg_confidence': np.mean([p.confidence for p in predictions
                                               if p.regime == regime])
                }
            else:
                analysis[regime] = {
                    'count': 0,
                    'avg_forward_return': 0.0,
                    'std_forward_return': 0.0,
                    'positive_pct': 0.0,
                    'avg_confidence': 0.0
                }

        # Check if regimes make sense
        # Bull should have positive returns, Bear negative
        regime_validity = {}
        if analysis[MarketRegime.BULL]['count'] > 0:
            regime_validity['bull_correct'] = analysis[MarketRegime.BULL]['avg_forward_return'] > 0
        if analysis[MarketRegime.BEAR]['count'] > 0:
            regime_validity['bear_correct'] = analysis[MarketRegime.BEAR]['avg_forward_return'] < 0

        return {
            'by_regime': analysis,
            'validity_check': regime_validity
        }

    def evaluate_transition_timing(
        self,
        predictions: List[RegimeResult],
        ground_truth: List[MarketRegime],
        tolerance_days: int = 5
    ) -> TimingMetrics:
        """
        Evaluate timing of regime transitions.

        Args:
            predictions: Predicted regime results
            ground_truth: True regime labels
            tolerance_days: Acceptable delay in detection

        Returns:
            TimingMetrics with transition analysis
        """
        # Find transitions in ground truth
        true_transitions = []
        for i in range(1, len(ground_truth)):
            if ground_truth[i] != ground_truth[i-1]:
                true_transitions.append((i, ground_truth[i-1], ground_truth[i]))

        # Find transitions in predictions
        pred_regimes = [p.regime for p in predictions]
        pred_transitions = []
        for i in range(1, len(pred_regimes)):
            if pred_regimes[i] != pred_regimes[i-1]:
                pred_transitions.append((i, pred_regimes[i-1], pred_regimes[i]))

        # Match transitions
        delays = []
        correct = 0
        missed = 0
        false_pos = 0

        for true_idx, true_from, true_to in true_transitions:
            # Find matching prediction
            matched = False
            for pred_idx, pred_from, pred_to in pred_transitions:
                if pred_from == true_from and pred_to == true_to:
                    delay = pred_idx - true_idx
                    if abs(delay) <= tolerance_days:
                        delays.append(delay)
                        correct += 1
                        matched = True
                        break
            if not matched:
                missed += 1

        # Count false transitions
        for pred_idx, pred_from, pred_to in pred_transitions:
            matched = False
            for true_idx, true_from, true_to in true_transitions:
                if pred_from == true_from and pred_to == true_to:
                    if abs(pred_idx - true_idx) <= tolerance_days:
                        matched = True
                        break
            if not matched:
                false_pos += 1

        avg_delay = np.mean(delays) if delays else 0.0
        total_true = len(true_transitions)
        accuracy = correct / total_true if total_true > 0 else 0.0

        return TimingMetrics(
            avg_detection_delay=avg_delay,
            false_transitions=false_pos,
            missed_transitions=missed,
            correct_transitions=correct,
            transition_accuracy=accuracy
        )

    def _build_confusion_matrix(
        self,
        predictions: List[MarketRegime],
        ground_truth: List[MarketRegime]
    ) -> pd.DataFrame:
        """Build confusion matrix."""
        matrix = np.zeros((len(self.regimes), len(self.regimes)))

        for pred, true in zip(predictions, ground_truth):
            pred_idx = self.regimes.index(pred)
            true_idx = self.regimes.index(true)
            matrix[true_idx, pred_idx] += 1

        regime_names = [r.value for r in self.regimes]
        return pd.DataFrame(
            matrix,
            index=[f"True_{r}" for r in regime_names],
            columns=[f"Pred_{r}" for r in regime_names]
        )


class ConfidenceCalibrator:
    """
    Calibrate and evaluate confidence scores.

    Checks if confidence scores are well-calibrated
    (i.e., 80% confidence should be correct 80% of the time).
    """

    def __init__(self, num_bins: int = 10):
        """
        Initialize calibrator.

        Args:
            num_bins: Number of bins for calibration
        """
        self.num_bins = num_bins

    def evaluate_calibration(
        self,
        predictions: List[RegimeResult],
        ground_truth: List[MarketRegime]
    ) -> Dict:
        """
        Evaluate confidence calibration.

        Args:
            predictions: Predicted regime results
            ground_truth: True regime labels

        Returns:
            Dictionary with calibration metrics
        """
        # Bin predictions by confidence
        bin_edges = np.linspace(0, 1, self.num_bins + 1)
        bins = {i: {'correct': 0, 'total': 0, 'confidences': []}
                for i in range(self.num_bins)}

        for pred, true in zip(predictions, ground_truth):
            conf = pred.confidence
            bin_idx = min(int(conf * self.num_bins), self.num_bins - 1)

            bins[bin_idx]['total'] += 1
            bins[bin_idx]['confidences'].append(conf)
            if pred.regime == true:
                bins[bin_idx]['correct'] += 1

        # Calculate calibration metrics
        calibration_data = []
        for i in range(self.num_bins):
            if bins[i]['total'] > 0:
                avg_confidence = np.mean(bins[i]['confidences'])
                accuracy = bins[i]['correct'] / bins[i]['total']
                calibration_data.append({
                    'bin': i,
                    'bin_range': (bin_edges[i], bin_edges[i+1]),
                    'avg_confidence': avg_confidence,
                    'accuracy': accuracy,
                    'count': bins[i]['total'],
                    'calibration_error': abs(avg_confidence - accuracy)
                })

        # Expected Calibration Error (ECE)
        total_samples = len(predictions)
        ece = sum(d['count'] / total_samples * d['calibration_error']
                  for d in calibration_data) if calibration_data else 0

        # Maximum Calibration Error (MCE)
        mce = max(d['calibration_error'] for d in calibration_data) if calibration_data else 0

        return {
            'calibration_data': calibration_data,
            'expected_calibration_error': ece,
            'max_calibration_error': mce
        }


def generate_report(
    predictions: List[RegimeResult],
    prices: pd.Series,
    ground_truth: Optional[List[MarketRegime]] = None
) -> str:
    """
    Generate a comprehensive evaluation report.

    Args:
        predictions: Predicted regime results
        prices: Price series
        ground_truth: Optional ground truth labels

    Returns:
        Formatted report string
    """
    evaluator = RegimeEvaluator()
    returns = prices.pct_change().dropna()

    report = []
    report.append("=" * 60)
    report.append("REGIME CLASSIFICATION EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")

    # Basic statistics
    regime_counts = Counter([p.regime for p in predictions])
    total = len(predictions)

    report.append("REGIME DISTRIBUTION:")
    report.append("-" * 40)
    for regime in MarketRegime:
        count = regime_counts.get(regime, 0)
        pct = count / total * 100 if total > 0 else 0
        report.append(f"  {regime.value:15s}: {count:5d} ({pct:5.1f}%)")
    report.append("")

    # Confidence statistics
    confidences = [p.confidence for p in predictions]
    report.append("CONFIDENCE STATISTICS:")
    report.append("-" * 40)
    report.append(f"  Mean:   {np.mean(confidences):.3f}")
    report.append(f"  Std:    {np.std(confidences):.3f}")
    report.append(f"  Min:    {np.min(confidences):.3f}")
    report.append(f"  Max:    {np.max(confidences):.3f}")
    report.append("")

    # Forward return analysis
    if len(returns) > 5:
        fwd_analysis = evaluator.evaluate_forward_returns(
            predictions[:len(returns)-5], returns, forward_window=5
        )

        report.append("FORWARD RETURN ANALYSIS (5-day):")
        report.append("-" * 40)
        for regime in MarketRegime:
            data = fwd_analysis['by_regime'][regime]
            if data['count'] > 0:
                report.append(f"  {regime.value}:")
                report.append(f"    Count:      {data['count']}")
                report.append(f"    Avg Return: {data['avg_forward_return']:.4f}")
                report.append(f"    Positive:   {data['positive_pct']:.1%}")
        report.append("")

    # Ground truth comparison if available
    if ground_truth is not None:
        metrics = evaluator.evaluate_vs_ground_truth(predictions, ground_truth)

        report.append("ACCURACY METRICS:")
        report.append("-" * 40)
        report.append(f"  Overall Accuracy: {metrics.accuracy:.3f}")
        report.append("")
        report.append("  Per-Regime F1 Scores:")
        for regime in MarketRegime:
            report.append(f"    {regime.value:15s}: {metrics.f1_score[regime]:.3f}")
        report.append("")

        timing = evaluator.evaluate_transition_timing(predictions, ground_truth)
        report.append("TRANSITION TIMING:")
        report.append("-" * 40)
        report.append(f"  Correct:    {timing.correct_transitions}")
        report.append(f"  Missed:     {timing.missed_transitions}")
        report.append(f"  False:      {timing.false_transitions}")
        report.append(f"  Avg Delay:  {timing.avg_detection_delay:.1f} days")
        report.append("")

    report.append("=" * 60)

    return "\n".join(report)
