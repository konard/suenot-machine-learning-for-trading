"""
Evaluation metrics for fine-tuned financial LLMs.

This module provides comprehensive evaluation for:
- Classification metrics (accuracy, F1, precision, recall)
- Trading metrics (Sharpe, Sortino, drawdown)
- Signal quality metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from dataclasses import dataclass


@dataclass
class ClassificationMetrics:
    """Metrics for classification evaluation."""
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1: Dict[str, float]
    macro_f1: float
    weighted_f1: float
    confusion_matrix: np.ndarray
    support: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class_precision": self.precision,
            "per_class_recall": self.recall,
            "per_class_f1": self.f1
        }


@dataclass
class TradingMetrics:
    """Metrics for trading strategy evaluation."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float
    volatility: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "expectancy": self.expectancy,
            "volatility": self.volatility
        }


def evaluate_sentiment(
    predictions: List[int],
    labels: List[int],
    label_names: Optional[List[str]] = None
) -> ClassificationMetrics:
    """
    Evaluate sentiment classification performance.

    Args:
        predictions: Predicted labels
        labels: True labels
        label_names: Optional names for labels

    Returns:
        ClassificationMetrics with comprehensive evaluation
    """
    if label_names is None:
        label_names = ["Bearish", "Neutral", "Bullish"]

    # Convert to numpy
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Accuracy
    accuracy = np.mean(predictions == labels)

    # Per-class metrics
    precision = {}
    recall = {}
    f1 = {}
    support = {}

    unique_labels = sorted(set(labels) | set(predictions))

    for label in unique_labels:
        name = label_names[label] if label < len(label_names) else str(label)

        # True positives, false positives, false negatives
        tp = np.sum((predictions == label) & (labels == label))
        fp = np.sum((predictions == label) & (labels != label))
        fn = np.sum((predictions != label) & (labels == label))

        # Precision
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision[name] = prec

        # Recall
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall[name] = rec

        # F1
        f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1[name] = f1_score

        # Support (count of true labels)
        support[name] = int(np.sum(labels == label))

    # Macro F1 (unweighted average)
    macro_f1 = np.mean(list(f1.values()))

    # Weighted F1 (weighted by support)
    total_support = sum(support.values())
    weighted_f1 = sum(
        f1[name] * support[name] / total_support
        for name in f1
    )

    # Confusion matrix
    num_classes = len(unique_labels)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for pred, true in zip(predictions, labels):
        confusion[true, pred] += 1

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        confusion_matrix=confusion,
        support=support
    )


def evaluate_trading(
    returns: List[float],
    benchmark_returns: Optional[List[float]] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> TradingMetrics:
    """
    Evaluate trading strategy performance.

    Args:
        returns: List of period returns
        benchmark_returns: Optional benchmark returns for comparison
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        TradingMetrics with comprehensive evaluation
    """
    returns = np.array(returns)

    # Basic statistics
    total_return = np.prod(1 + returns) - 1
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    # Annualized metrics
    annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    volatility = std_return * np.sqrt(periods_per_year)

    # Risk-adjusted returns
    excess_return = mean_return - risk_free_rate / periods_per_year

    # Sharpe ratio
    sharpe_ratio = (excess_return * periods_per_year) / volatility if volatility > 0 else 0

    # Sortino ratio (downside deviation)
    negative_returns = returns[returns < 0]
    downside_std = np.std(negative_returns) if len(negative_returns) > 0 else std_return
    sortino_ratio = (excess_return * periods_per_year) / (downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0

    # Drawdown analysis
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max

    max_drawdown = abs(np.min(drawdowns))

    # Max drawdown duration
    in_drawdown = drawdowns < 0
    drawdown_periods = []
    current_period = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_period += 1
        else:
            if current_period > 0:
                drawdown_periods.append(current_period)
            current_period = 0
    max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0

    # Calmar ratio
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

    # Win/loss statistics
    winners = returns[returns > 0]
    losers = returns[returns < 0]

    win_rate = len(winners) / len(returns) if len(returns) > 0 else 0
    avg_win = np.mean(winners) if len(winners) > 0 else 0
    avg_loss = np.mean(losers) if len(losers) > 0 else 0

    # Profit factor
    gross_profit = np.sum(winners)
    gross_loss = abs(np.sum(losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Expectancy
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    return TradingMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=expectancy,
        volatility=volatility
    )


def evaluate_signal_quality(
    signals: List[int],
    returns: List[float],
    confidence: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Evaluate signal quality against actual returns.

    Args:
        signals: Trading signals (-1=sell, 0=hold, 1=buy)
        returns: Actual returns following signals
        confidence: Optional signal confidence scores

    Returns:
        Dictionary with signal quality metrics
    """
    signals = np.array(signals)
    returns = np.array(returns)

    # Signal accuracy
    predicted_direction = signals
    actual_direction = np.sign(returns)

    # Directional accuracy (excluding holds)
    non_hold_mask = signals != 0
    if np.sum(non_hold_mask) > 0:
        directional_accuracy = np.mean(
            predicted_direction[non_hold_mask] == actual_direction[non_hold_mask]
        )
    else:
        directional_accuracy = 0.0

    # Signal profitability
    signal_returns = signals * returns
    total_signal_return = np.sum(signal_returns)
    avg_signal_return = np.mean(signal_returns[non_hold_mask]) if np.sum(non_hold_mask) > 0 else 0

    # Information coefficient (correlation between signal and returns)
    if np.std(signals) > 0 and np.std(returns) > 0:
        ic = np.corrcoef(signals, returns)[0, 1]
    else:
        ic = 0.0

    # Hit rate by signal type
    buy_signals = signals == 1
    sell_signals = signals == -1

    buy_hit_rate = np.mean(returns[buy_signals] > 0) if np.sum(buy_signals) > 0 else 0
    sell_hit_rate = np.mean(returns[sell_signals] < 0) if np.sum(sell_signals) > 0 else 0

    metrics = {
        "directional_accuracy": directional_accuracy,
        "total_signal_return": total_signal_return,
        "avg_signal_return": avg_signal_return,
        "information_coefficient": ic,
        "buy_hit_rate": buy_hit_rate,
        "sell_hit_rate": sell_hit_rate,
        "num_signals": len(signals),
        "num_trades": int(np.sum(non_hold_mask)),
        "trade_ratio": np.sum(non_hold_mask) / len(signals)
    }

    # Confidence-weighted metrics
    if confidence is not None:
        confidence = np.array(confidence)

        # Confidence-weighted return
        weighted_return = np.sum(signal_returns * confidence) / np.sum(confidence[non_hold_mask])
        metrics["confidence_weighted_return"] = weighted_return

        # Average confidence of correct vs incorrect signals
        correct_mask = (predicted_direction == actual_direction) & non_hold_mask
        incorrect_mask = (predicted_direction != actual_direction) & non_hold_mask

        metrics["avg_confidence_correct"] = np.mean(confidence[correct_mask]) if np.sum(correct_mask) > 0 else 0
        metrics["avg_confidence_incorrect"] = np.mean(confidence[incorrect_mask]) if np.sum(incorrect_mask) > 0 else 0

    return metrics


def compute_feature_importance(
    model,
    test_data: pd.DataFrame,
    feature_names: List[str],
    method: str = "permutation"
) -> Dict[str, float]:
    """
    Compute feature importance for the model.

    Args:
        model: Trained model
        test_data: Test dataset
        feature_names: Names of features
        method: Importance method (permutation, gradient, etc.)

    Returns:
        Dictionary mapping feature names to importance scores
    """
    # Placeholder implementation
    # In practice, implement permutation importance or gradient-based methods
    importance = {name: np.random.random() for name in feature_names}

    # Normalize
    total = sum(importance.values())
    importance = {k: v / total for k, v in importance.items()}

    return importance


def compute_calibration(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10
) -> Dict[str, Any]:
    """
    Compute calibration metrics for probability predictions.

    Args:
        predictions: Predicted probabilities
        labels: True labels (binary)
        num_bins: Number of calibration bins

    Returns:
        Calibration metrics including ECE and reliability diagram data
    """
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Bin predictions
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(predictions, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Compute calibration
    bin_counts = []
    bin_accuracies = []
    bin_confidences = []

    for i in range(num_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_counts.append(np.sum(mask))
            bin_accuracies.append(np.mean(labels[mask]))
            bin_confidences.append(np.mean(predictions[mask]))
        else:
            bin_counts.append(0)
            bin_accuracies.append(0)
            bin_confidences.append((bin_edges[i] + bin_edges[i+1]) / 2)

    # Expected Calibration Error (ECE)
    total_samples = len(predictions)
    ece = sum(
        (count / total_samples) * abs(acc - conf)
        for count, acc, conf in zip(bin_counts, bin_accuracies, bin_confidences)
    )

    # Maximum Calibration Error (MCE)
    mce = max(
        abs(acc - conf)
        for acc, conf in zip(bin_accuracies, bin_confidences)
        if conf > 0  # Skip empty bins
    )

    return {
        "ece": ece,
        "mce": mce,
        "bin_edges": bin_edges.tolist(),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts
    }


def generate_evaluation_report(
    classification_metrics: ClassificationMetrics,
    trading_metrics: TradingMetrics,
    signal_quality: Dict[str, float]
) -> str:
    """
    Generate comprehensive evaluation report.

    Args:
        classification_metrics: Classification evaluation results
        trading_metrics: Trading strategy evaluation results
        signal_quality: Signal quality metrics

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "FINE-TUNED LLM EVALUATION REPORT",
        "=" * 60,
        "",
        "CLASSIFICATION METRICS",
        "-" * 40,
        f"Accuracy: {classification_metrics.accuracy:.4f}",
        f"Macro F1: {classification_metrics.macro_f1:.4f}",
        f"Weighted F1: {classification_metrics.weighted_f1:.4f}",
        "",
        "Per-Class Performance:",
    ]

    for name in classification_metrics.f1:
        lines.append(
            f"  {name}: P={classification_metrics.precision[name]:.3f}, "
            f"R={classification_metrics.recall[name]:.3f}, "
            f"F1={classification_metrics.f1[name]:.3f} "
            f"(n={classification_metrics.support[name]})"
        )

    lines.extend([
        "",
        "TRADING METRICS",
        "-" * 40,
        f"Total Return: {trading_metrics.total_return:.2%}",
        f"Annualized Return: {trading_metrics.annualized_return:.2%}",
        f"Sharpe Ratio: {trading_metrics.sharpe_ratio:.2f}",
        f"Sortino Ratio: {trading_metrics.sortino_ratio:.2f}",
        f"Calmar Ratio: {trading_metrics.calmar_ratio:.2f}",
        f"Max Drawdown: {trading_metrics.max_drawdown:.2%}",
        f"Win Rate: {trading_metrics.win_rate:.2%}",
        f"Profit Factor: {trading_metrics.profit_factor:.2f}",
        f"Expectancy: {trading_metrics.expectancy:.4f}",
        "",
        "SIGNAL QUALITY",
        "-" * 40,
        f"Directional Accuracy: {signal_quality['directional_accuracy']:.2%}",
        f"Information Coefficient: {signal_quality['information_coefficient']:.4f}",
        f"Buy Hit Rate: {signal_quality['buy_hit_rate']:.2%}",
        f"Sell Hit Rate: {signal_quality['sell_hit_rate']:.2%}",
        f"Trade Ratio: {signal_quality['trade_ratio']:.2%}",
        "",
        "=" * 60
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    print("Testing evaluation metrics...")

    # Classification evaluation
    predictions = [0, 1, 2, 1, 0, 2, 1, 2, 0, 1]
    labels = [0, 1, 2, 1, 1, 2, 1, 2, 0, 0]

    class_metrics = evaluate_sentiment(predictions, labels)
    print(f"\nClassification Metrics:")
    print(f"Accuracy: {class_metrics.accuracy:.4f}")
    print(f"Macro F1: {class_metrics.macro_f1:.4f}")
    print(f"Per-class F1: {class_metrics.f1}")

    # Trading evaluation
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01

    trading_metrics = evaluate_trading(returns)
    print(f"\nTrading Metrics:")
    print(f"Total Return: {trading_metrics.total_return:.2%}")
    print(f"Sharpe Ratio: {trading_metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {trading_metrics.max_drawdown:.2%}")

    # Signal quality
    signals = np.random.choice([-1, 0, 1], size=252)
    signal_metrics = evaluate_signal_quality(signals, returns)
    print(f"\nSignal Quality:")
    print(f"Directional Accuracy: {signal_metrics['directional_accuracy']:.2%}")
    print(f"Information Coefficient: {signal_metrics['information_coefficient']:.4f}")

    # Full report
    report = generate_evaluation_report(class_metrics, trading_metrics, signal_metrics)
    print("\n" + report)
