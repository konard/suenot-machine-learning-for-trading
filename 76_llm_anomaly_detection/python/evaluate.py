"""
Evaluation metrics and visualization for anomaly detection.

This module provides tools to evaluate anomaly detection performance
and visualize results.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import logging

import numpy as np
import pandas as pd

try:
    from .detector import AnomalyResult, AnomalyType
except ImportError:
    from detector import AnomalyResult, AnomalyType

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for anomaly detection."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "confusion_matrix": self.confusion_matrix,
        }

    def print_summary(self) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 40)
        print("ANOMALY DETECTION EVALUATION")
        print("=" * 40)
        print(f"Accuracy:  {self.accuracy:.4f}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall:    {self.recall:.4f}")
        print(f"F1 Score:  {self.f1_score:.4f}")
        print(f"ROC AUC:   {self.roc_auc:.4f}")
        print(f"PR AUC:    {self.pr_auc:.4f}")
        print("\nConfusion Matrix:")
        print(f"  TP: {self.confusion_matrix['tp']:5d}  FP: {self.confusion_matrix['fp']:5d}")
        print(f"  FN: {self.confusion_matrix['fn']:5d}  TN: {self.confusion_matrix['tn']:5d}")
        print("=" * 40 + "\n")


class AnomalyEvaluator:
    """
    Evaluate anomaly detection performance.

    Supports:
    - Classification metrics (precision, recall, F1)
    - ROC and PR curves
    - Threshold analysis
    - Per-type evaluation
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize evaluator.

        Args:
            threshold: Score threshold for anomaly classification
        """
        self.threshold = threshold

    def evaluate(
        self,
        predictions: List[AnomalyResult],
        ground_truth: List[bool],
    ) -> EvaluationMetrics:
        """
        Evaluate anomaly detection results.

        Args:
            predictions: List of AnomalyResult from detector
            ground_truth: List of true labels (True = anomaly)

        Returns:
            EvaluationMetrics with performance metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        # Extract scores and predictions
        scores = np.array([p.score for p in predictions])
        pred_labels = np.array([p.is_anomaly for p in predictions])
        true_labels = np.array(ground_truth)

        # Confusion matrix
        tp = np.sum(pred_labels & true_labels)
        fp = np.sum(pred_labels & ~true_labels)
        fn = np.sum(~pred_labels & true_labels)
        tn = np.sum(~pred_labels & ~true_labels)

        # Metrics
        accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # ROC AUC
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            roc_auc = roc_auc_score(true_labels, scores) if len(np.unique(true_labels)) > 1 else 0.5
            pr_auc = average_precision_score(true_labels, scores) if len(np.unique(true_labels)) > 1 else 0
        except ImportError:
            logger.warning("sklearn not available, using simple AUC calculation")
            roc_auc = self._simple_auc(true_labels, scores)
            pr_auc = precision  # Approximation

        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            confusion_matrix={"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        )

    def _simple_auc(self, y_true: np.ndarray, scores: np.ndarray) -> float:
        """Simple AUC calculation without sklearn."""
        # Sort by score descending
        sorted_indices = np.argsort(-scores)
        y_sorted = y_true[sorted_indices]

        # Calculate AUC using trapezoidal rule
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr_prev, fpr_prev = 0, 0
        auc = 0

        for i, y in enumerate(y_sorted):
            if y:
                tpr = (i + 1) / n_pos
                auc += (tpr - tpr_prev) * (1 - fpr_prev)
                tpr_prev = tpr
            else:
                fpr = (i + 1 - np.sum(y_sorted[:i+1])) / n_neg
                fpr_prev = fpr

        return auc

    def find_optimal_threshold(
        self,
        predictions: List[AnomalyResult],
        ground_truth: List[bool],
        metric: str = "f1_score",
    ) -> Tuple[float, EvaluationMetrics]:
        """
        Find optimal threshold for the specified metric.

        Args:
            predictions: List of AnomalyResult
            ground_truth: True labels
            metric: Metric to optimize

        Returns:
            Tuple of (optimal threshold, metrics at that threshold)
        """
        scores = np.array([p.score for p in predictions])
        true_labels = np.array(ground_truth)

        thresholds = np.linspace(0, 1, 101)
        best_threshold = 0.5
        best_score = 0
        best_metrics = None

        for threshold in thresholds:
            # Create predictions at this threshold
            pred_at_threshold = [
                AnomalyResult(
                    is_anomaly=p.score >= threshold,
                    score=p.score,
                    anomaly_type=p.anomaly_type,
                    confidence=p.confidence,
                    explanation=p.explanation,
                    details=p.details,
                )
                for p in predictions
            ]

            metrics = self.evaluate(pred_at_threshold, ground_truth)
            score = getattr(metrics, metric, 0)

            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = metrics

        return best_threshold, best_metrics

    def evaluate_by_type(
        self,
        predictions: List[AnomalyResult],
        ground_truth: List[bool],
        ground_truth_types: Optional[List[AnomalyType]] = None,
    ) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate performance by anomaly type.

        Args:
            predictions: List of AnomalyResult
            ground_truth: True labels
            ground_truth_types: True anomaly types (optional)

        Returns:
            Dictionary of metrics per anomaly type
        """
        # Group predictions by predicted type
        type_results: Dict[AnomalyType, List[Tuple[AnomalyResult, bool]]] = {}

        for pred, true_label in zip(predictions, ground_truth):
            at = pred.anomaly_type
            if at not in type_results:
                type_results[at] = []
            type_results[at].append((pred, true_label))

        # Evaluate each type
        metrics_by_type = {}
        for anomaly_type, results in type_results.items():
            preds = [r[0] for r in results]
            labels = [r[1] for r in results]

            if len(preds) > 0:
                metrics = self.evaluate(preds, labels)
                metrics_by_type[anomaly_type.value] = metrics

        return metrics_by_type


def generate_synthetic_anomalies(
    data: pd.DataFrame,
    anomaly_ratio: float = 0.05,
    anomaly_types: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[bool]]:
    """
    Generate synthetic anomalies for testing.

    Args:
        data: Normal OHLCV data
        anomaly_ratio: Fraction of anomalies to inject
        anomaly_types: Types of anomalies to inject

    Returns:
        Tuple of (data with anomalies, labels)
    """
    data = data.copy()
    n_anomalies = int(len(data) * anomaly_ratio)
    labels = [False] * len(data)

    anomaly_types = anomaly_types or ["price_spike", "volume_spike", "flash_crash"]

    # Select random indices for anomalies
    np.random.seed(42)
    anomaly_indices = np.random.choice(
        range(10, len(data) - 10),  # Avoid edges
        size=min(n_anomalies, len(data) - 20),
        replace=False,
    )

    for idx in anomaly_indices:
        anomaly_type = np.random.choice(anomaly_types)

        if anomaly_type == "price_spike":
            # Inject sudden price spike
            spike = np.random.uniform(0.05, 0.15) * (1 if np.random.random() > 0.5 else -1)
            data.loc[data.index[idx], "close"] *= (1 + spike)
            data.loc[data.index[idx], "high"] *= (1 + abs(spike))
            if spike < 0:
                data.loc[data.index[idx], "low"] *= (1 + spike)

        elif anomaly_type == "volume_spike":
            # Inject volume spike
            data.loc[data.index[idx], "volume"] *= np.random.uniform(5, 20)

        elif anomaly_type == "flash_crash":
            # Inject flash crash pattern
            crash_pct = np.random.uniform(0.10, 0.25)
            data.loc[data.index[idx], "low"] *= (1 - crash_pct)
            data.loc[data.index[idx], "close"] *= (1 - crash_pct * 0.8)

        labels[idx] = True

    logger.info(f"Injected {len(anomaly_indices)} synthetic anomalies")
    return data, labels


def plot_anomalies(
    data: pd.DataFrame,
    anomaly_results: List[AnomalyResult],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot price chart with anomalies highlighted.

    Args:
        data: OHLCV DataFrame
        anomaly_results: List of anomaly detection results
        save_path: Path to save plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Price chart
    ax1 = axes[0]
    ax1.plot(data.index, data["close"], label="Close Price", color="blue", alpha=0.7)

    # Highlight anomalies
    anomaly_mask = [r.is_anomaly for r in anomaly_results]
    # Align mask with data
    if len(anomaly_mask) < len(data):
        anomaly_mask = [False] * (len(data) - len(anomaly_mask)) + anomaly_mask

    anomaly_indices = [i for i, is_anom in enumerate(anomaly_mask) if is_anom]
    if anomaly_indices:
        ax1.scatter(
            [data.index[i] for i in anomaly_indices if i < len(data)],
            [data["close"].iloc[i] for i in anomaly_indices if i < len(data)],
            color="red",
            s=100,
            marker="^",
            label="Anomaly",
            zorder=5,
        )

    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.set_title("Price with Anomalies")

    # Anomaly scores
    ax2 = axes[1]
    scores = [r.score for r in anomaly_results]
    if len(scores) < len(data):
        scores = [0] * (len(data) - len(scores)) + scores
    ax2.fill_between(data.index[:len(scores)], scores, alpha=0.5, color="orange")
    ax2.axhline(y=0.5, color="red", linestyle="--", label="Threshold")
    ax2.set_ylabel("Anomaly Score")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.set_title("Anomaly Scores")

    # Volume
    ax3 = axes[2]
    ax3.bar(data.index, data["volume"], alpha=0.5, color="green")
    ax3.set_ylabel("Volume")
    ax3.set_xlabel("Time")
    ax3.set_title("Trading Volume")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark: Optional[pd.Series] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot equity curve from backtest.

    Args:
        equity_curve: Equity series from backtest
        benchmark: Optional benchmark for comparison
        save_path: Path to save plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Equity curve
    ax1 = axes[0]
    ax1.plot(equity_curve.index, equity_curve.values, label="Strategy", color="blue")
    if benchmark is not None:
        ax1.plot(benchmark.index, benchmark.values, label="Benchmark", color="gray", alpha=0.7)
    ax1.set_ylabel("Equity")
    ax1.legend()
    ax1.set_title("Equity Curve")
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color="red")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Time")
    ax2.set_title("Drawdown")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def create_evaluation_report(
    predictions: List[AnomalyResult],
    ground_truth: List[bool],
    backtest_result: Optional[Any] = None,
) -> str:
    """
    Create a text report of evaluation results.

    Args:
        predictions: Anomaly detection results
        ground_truth: True labels
        backtest_result: Optional backtest results

    Returns:
        Formatted report string
    """
    evaluator = AnomalyEvaluator()
    metrics = evaluator.evaluate(predictions, ground_truth)

    report = []
    report.append("=" * 60)
    report.append("ANOMALY DETECTION EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")

    # Detection metrics
    report.append("Detection Performance:")
    report.append("-" * 40)
    report.append(f"  Accuracy:  {metrics.accuracy:.4f}")
    report.append(f"  Precision: {metrics.precision:.4f}")
    report.append(f"  Recall:    {metrics.recall:.4f}")
    report.append(f"  F1 Score:  {metrics.f1_score:.4f}")
    report.append(f"  ROC AUC:   {metrics.roc_auc:.4f}")
    report.append(f"  PR AUC:    {metrics.pr_auc:.4f}")
    report.append("")

    # Confusion matrix
    report.append("Confusion Matrix:")
    report.append("-" * 40)
    cm = metrics.confusion_matrix
    report.append(f"                 Predicted")
    report.append(f"              Neg      Pos")
    report.append(f"  Actual Neg  {cm['tn']:5d}    {cm['fp']:5d}")
    report.append(f"         Pos  {cm['fn']:5d}    {cm['tp']:5d}")
    report.append("")

    # Anomaly type distribution
    type_counts: Dict[str, int] = {}
    for p in predictions:
        if p.is_anomaly:
            at = p.anomaly_type.value
            type_counts[at] = type_counts.get(at, 0) + 1

    if type_counts:
        report.append("Detected Anomaly Types:")
        report.append("-" * 40)
        for atype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            report.append(f"  {atype}: {count}")
        report.append("")

    # Backtest results
    if backtest_result is not None:
        report.append("Backtest Performance:")
        report.append("-" * 40)
        report.append(f"  Total Return: {backtest_result.total_return_pct:.2f}%")
        report.append(f"  Sharpe Ratio: {backtest_result.sharpe_ratio:.2f}")
        report.append(f"  Max Drawdown: {backtest_result.max_drawdown_pct:.2f}%")
        report.append(f"  Win Rate:     {backtest_result.win_rate:.1f}%")
        report.append(f"  Num Trades:   {backtest_result.num_trades}")
        report.append("")

    report.append("=" * 60)

    return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    from data_loader import load_sample_data
    from detector import StatisticalAnomalyDetector

    print("Loading sample data...")
    data = load_sample_data(source="bybit")

    if not data.empty:
        print(f"Loaded {len(data)} rows")

        # Generate synthetic anomalies
        print("\nGenerating synthetic anomalies...")
        data_with_anomalies, labels = generate_synthetic_anomalies(
            data, anomaly_ratio=0.05
        )

        # Run detection
        print("Running anomaly detection...")
        detector = StatisticalAnomalyDetector(z_threshold=2.5)
        detector.fit(data_with_anomalies.iloc[:100])
        predictions = detector.detect(data_with_anomalies.iloc[100:])

        # Align labels
        labels = labels[100:]

        # Evaluate
        if len(predictions) == len(labels):
            evaluator = AnomalyEvaluator()
            metrics = evaluator.evaluate(predictions, labels)
            metrics.print_summary()

            # Find optimal threshold
            opt_threshold, opt_metrics = evaluator.find_optimal_threshold(
                predictions, labels
            )
            print(f"Optimal threshold: {opt_threshold:.2f}")
            print(f"F1 at optimal threshold: {opt_metrics.f1_score:.4f}")

            # Create report
            report = create_evaluation_report(predictions, labels)
            print(report)
        else:
            print(f"Length mismatch: {len(predictions)} predictions, {len(labels)} labels")
    else:
        print("Could not load sample data")
