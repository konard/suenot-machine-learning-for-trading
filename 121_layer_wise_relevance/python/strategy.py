"""
Trading Strategy and Backtesting with LRP Insights

This module provides trading strategy implementation and backtesting
with Layer-wise Relevance Propagation explanations.

Features:
    - LRP-based position sizing (confidence-weighted)
    - Signal analysis with feature attribution
    - Comprehensive backtesting metrics
    - Feature importance tracking

Example:
    >>> results = backtest_with_lrp(model, test_loader, feature_names)
    >>> print(f"Sharpe: {results['sharpe_ratio']:.2f}")
    >>> print(f"Top feature: {results['feature_importance'][0]}")
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

from .model import LRPNetwork


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting strategy.

    Attributes:
        initial_capital: Starting capital
        transaction_cost: Trading fee as fraction (0.001 = 0.1%)
        confidence_threshold: Minimum confidence to trade
        max_position: Maximum position size (1.0 = 100%)
        stop_loss: Stop loss threshold (0.02 = 2%)
        take_profit: Take profit threshold (0.04 = 4%)
        use_lrp_sizing: Whether to use LRP coherence for position sizing
    """
    initial_capital: float = 100000
    transaction_cost: float = 0.001
    confidence_threshold: float = 0.6
    max_position: float = 1.0
    stop_loss: float = 0.02
    take_profit: float = 0.04
    use_lrp_sizing: bool = True


def analyze_trading_signal(
    model: LRPNetwork,
    sample: torch.Tensor,
    feature_names: List[str],
    threshold: float = 0.05
) -> Dict:
    """
    Analyze a single trading signal with LRP explanation.

    Args:
        model: Trained LRP model
        sample: Single input sample [1, seq_len, features] or [1, features]
        feature_names: Names of features
        threshold: Minimum relevance percentage to include in explanation

    Returns:
        Dictionary containing:
            - signal: "BUY" or "SELL/HOLD"
            - confidence: Prediction confidence (0-1)
            - explanations: List of feature contributions
            - raw_relevance: Raw relevance values
    """
    model.eval()

    with torch.no_grad():
        # Get prediction
        output = model(sample)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

        # Get LRP explanation
        relevance = model.explain(sample, target_class=pred_class)

        # Aggregate relevance over time dimension if present
        if len(relevance.shape) == 3:
            relevance = relevance.mean(dim=1)  # [1, features]

        relevance = relevance[0].numpy()

        # Normalize to percentages
        total = np.abs(relevance).sum()
        if total > 0:
            relevance_pct = relevance / total * 100
        else:
            relevance_pct = np.zeros_like(relevance)

        # Handle feature names for flattened input
        if len(relevance_pct) > len(feature_names):
            # Assume multiple time steps
            n_features = len(feature_names)
            n_timesteps = len(relevance_pct) // n_features

            # Aggregate over time steps
            relevance_by_feature = np.zeros(n_features)
            for i in range(n_timesteps):
                relevance_by_feature += np.abs(relevance_pct[i * n_features:(i + 1) * n_features])
            relevance_pct = relevance_by_feature / n_timesteps
        elif len(relevance_pct) < len(feature_names):
            feature_names = feature_names[:len(relevance_pct)]

        # Create explanation entries
        explanations = []
        for name, rel in zip(feature_names, relevance_pct):
            if abs(rel) >= threshold * 100:
                direction = "supports" if rel > 0 else "contradicts"
                explanations.append({
                    "feature": name,
                    "relevance_pct": float(rel),
                    "direction": direction
                })

        # Sort by absolute relevance
        explanations.sort(key=lambda x: abs(x["relevance_pct"]), reverse=True)

    signal = "BUY" if pred_class == 1 else "SELL/HOLD"

    return {
        "signal": signal,
        "confidence": confidence,
        "explanations": explanations,
        "raw_relevance": relevance,
        "feature_names": feature_names
    }


def compute_feature_importance(
    model: LRPNetwork,
    data_loader: DataLoader,
    feature_names: List[str]
) -> List[Tuple[str, float]]:
    """
    Compute average feature importance using LRP across dataset.

    Args:
        model: Trained LRP model
        data_loader: DataLoader with test data
        feature_names: List of feature names

    Returns:
        List of (feature_name, importance) sorted by importance
    """
    model.eval()
    total_relevance = None
    count = 0

    with torch.no_grad():
        for batch_x, _ in data_loader:
            relevance = model.explain(batch_x)

            # Average over batch
            batch_relevance = torch.abs(relevance).mean(dim=0)

            # Flatten if needed
            if len(batch_relevance.shape) > 1:
                batch_relevance = batch_relevance.mean(dim=0)

            if total_relevance is None:
                total_relevance = batch_relevance
            else:
                total_relevance = total_relevance + batch_relevance
            count += 1

    avg_relevance = total_relevance / count

    # Handle dimension mismatch
    n_features = len(feature_names)
    if len(avg_relevance) > n_features:
        # Reshape and aggregate over time
        reshaped = avg_relevance.view(-1, n_features)
        avg_relevance = reshaped.mean(dim=0)
    elif len(avg_relevance) < n_features:
        feature_names = feature_names[:len(avg_relevance)]

    # Normalize
    avg_relevance = avg_relevance / avg_relevance.sum()

    importance = list(zip(feature_names, avg_relevance.tolist()))
    importance.sort(key=lambda x: x[1], reverse=True)

    return importance


def calculate_coherence(relevance: np.ndarray) -> float:
    """
    Calculate explanation coherence score.

    High coherence = few dominant features (focused explanation)
    Low coherence = spread across many features (diffuse explanation)

    Args:
        relevance: Raw relevance values

    Returns:
        Coherence score (0-1)
    """
    abs_rel = np.abs(relevance)
    total = abs_rel.sum()

    if total == 0:
        return 0.0

    probs = abs_rel / total

    # Entropy (lower = more focused)
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    max_entropy = np.log(len(relevance))

    # Coherence = 1 - normalized entropy
    coherence = 1 - (entropy / max_entropy) if max_entropy > 0 else 0

    return float(coherence)


def backtest_with_lrp(
    model: LRPNetwork,
    test_loader: DataLoader,
    feature_names: List[str],
    config: Optional[BacktestConfig] = None
) -> Dict:
    """
    Backtest trading strategy with LRP-based position sizing.

    The strategy uses model confidence and LRP coherence to adjust
    position sizes. Higher confidence and more focused explanations
    lead to larger positions.

    Args:
        model: Trained LRP model
        test_loader: Test DataLoader
        feature_names: Feature names for explanation
        config: Backtest configuration

    Returns:
        Dictionary with backtest results and metrics
    """
    config = config or BacktestConfig()
    model.eval()

    capital = config.initial_capital
    position = 0.0

    history = {
        "capital": [capital],
        "positions": [],
        "returns": [],
        "signals": [],
        "explanations": []
    }

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            for i in range(len(batch_x)):
                sample = batch_x[i:i + 1]
                actual_direction = batch_y[i].item()

                # Get prediction
                output = model(sample)
                probs = torch.softmax(output, dim=1)
                pred_class = output.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()

                # Get LRP explanation
                relevance = model.explain(sample, target_class=pred_class)
                if len(relevance.shape) == 3:
                    relevance = relevance.mean(dim=1)
                relevance = relevance[0].numpy()

                # Calculate coherence
                coherence = calculate_coherence(relevance)

                # Determine target position
                if confidence >= config.confidence_threshold:
                    target_position = 1.0 if pred_class == 1 else -1.0

                    # Scale by confidence and coherence if enabled
                    if config.use_lrp_sizing:
                        scaling = confidence * (0.5 + 0.5 * coherence)
                        target_position *= min(scaling, config.max_position)
                else:
                    target_position = 0.0

                # Calculate trading costs
                position_change = abs(target_position - position)
                costs = position_change * config.transaction_cost * capital

                # Simulate return (simplified: +1% if correct, -1% if wrong)
                correct = (pred_class == 1) == (actual_direction == 1)
                actual_return = 0.01 if actual_direction == 1 else -0.01

                # Calculate PnL
                pnl = position * actual_return * capital - costs
                capital += pnl

                # Update position
                position = target_position

                # Record
                history["capital"].append(capital)
                history["positions"].append(position)
                history["returns"].append(pnl / (capital - pnl) if capital != pnl else 0)
                history["signals"].append({
                    "pred": pred_class,
                    "actual": actual_direction,
                    "confidence": confidence,
                    "coherence": coherence,
                    "correct": correct
                })
                history["explanations"].append(relevance)

    # Calculate metrics
    returns = np.array(history["returns"])

    results = {
        "total_return": (capital - config.initial_capital) / config.initial_capital,
        "final_capital": capital,
        "sharpe_ratio": _calculate_sharpe(returns),
        "sortino_ratio": _calculate_sortino(returns),
        "max_drawdown": _calculate_max_drawdown(history["capital"]),
        "win_rate": _calculate_win_rate(history["signals"]),
        "profit_factor": _calculate_profit_factor(returns),
        "avg_confidence": np.mean([s["confidence"] for s in history["signals"]]),
        "avg_coherence": np.mean([s["coherence"] for s in history["signals"]]),
        "n_trades": len([p for p in history["positions"] if p != 0]),
        "history": history,
        "feature_importance": _aggregate_explanations(
            history["explanations"], feature_names
        )
    }

    return results


def _calculate_sharpe(returns: np.ndarray, rf: float = 0) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    excess = returns - rf / 252
    return float(np.sqrt(252) * excess.mean() / (excess.std() + 1e-9))


def _calculate_sortino(returns: np.ndarray, target: float = 0) -> float:
    """Calculate Sortino ratio (downside risk adjusted)."""
    if len(returns) == 0:
        return 0.0
    downside = returns[returns < target]
    downside_std = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 1e-9
    return float(np.sqrt(252) * (returns.mean() - target) / downside_std)


def _calculate_max_drawdown(capital_history: List[float]) -> float:
    """Calculate maximum drawdown."""
    peak = capital_history[0]
    max_dd = 0

    for capital in capital_history:
        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak
        max_dd = max(max_dd, dd)

    return max_dd


def _calculate_win_rate(signals: List[Dict]) -> float:
    """Calculate prediction accuracy."""
    if not signals:
        return 0.0
    correct = sum(1 for s in signals if s["correct"])
    return correct / len(signals)


def _calculate_profit_factor(returns: np.ndarray) -> float:
    """Calculate profit factor (gross profit / gross loss)."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return float(gains / (losses + 1e-9))


def _aggregate_explanations(
    explanations: List[np.ndarray],
    feature_names: List[str]
) -> List[Tuple[str, float]]:
    """Aggregate feature importance from backtest explanations."""
    if not explanations:
        return []

    # Stack and average
    stacked = np.stack([np.abs(e).flatten() for e in explanations])
    avg = stacked.mean(axis=0)

    # Handle dimension mismatch
    n_features = len(feature_names)
    if len(avg) > n_features:
        reshaped = avg.reshape(-1, n_features)
        avg = reshaped.mean(axis=0)
    elif len(avg) < n_features:
        feature_names = feature_names[:len(avg)]

    # Normalize
    total = avg.sum()
    if total > 0:
        avg = avg / total

    importance = list(zip(feature_names, avg.tolist()))
    importance.sort(key=lambda x: x[1], reverse=True)

    return importance


def visualize_backtest_results(results: Dict, save_path: Optional[str] = None):
    """
    Visualize backtest results.

    Args:
        results: Output from backtest_with_lrp
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Capital curve
    ax1 = axes[0, 0]
    ax1.plot(results["history"]["capital"], linewidth=1)
    ax1.set_title("Capital Over Time")
    ax1.set_xlabel("Trade")
    ax1.set_ylabel("Capital")
    ax1.grid(True, alpha=0.3)

    # Returns distribution
    ax2 = axes[0, 1]
    returns = np.array(results["history"]["returns"])
    ax2.hist(returns, bins=50, edgecolor="black", alpha=0.7)
    ax2.axvline(x=0, color="red", linestyle="--")
    ax2.set_title("Returns Distribution")
    ax2.set_xlabel("Return")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    # Feature importance
    ax3 = axes[1, 0]
    importance = results["feature_importance"][:10]  # Top 10
    features = [f[0] for f in importance]
    values = [f[1] * 100 for f in importance]
    ax3.barh(features[::-1], values[::-1], color="steelblue", alpha=0.7)
    ax3.set_title("Top 10 Feature Importance (LRP)")
    ax3.set_xlabel("Importance (%)")
    ax3.grid(True, alpha=0.3)

    # Metrics summary
    ax4 = axes[1, 1]
    ax4.axis("off")
    metrics_text = f"""
    Backtest Results Summary
    ========================

    Total Return:    {results['total_return']:.2%}
    Sharpe Ratio:    {results['sharpe_ratio']:.2f}
    Sortino Ratio:   {results['sortino_ratio']:.2f}
    Max Drawdown:    {results['max_drawdown']:.2%}
    Win Rate:        {results['win_rate']:.2%}
    Profit Factor:   {results['profit_factor']:.2f}

    Avg Confidence:  {results['avg_confidence']:.2%}
    Avg Coherence:   {results['avg_coherence']:.2f}
    Total Trades:    {results['n_trades']}
    """
    ax4.text(0.1, 0.5, metrics_text, family="monospace", fontsize=11,
             verticalalignment="center", transform=ax4.transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def generate_trading_report(
    model: LRPNetwork,
    data_loader: DataLoader,
    feature_names: List[str]
) -> str:
    """
    Generate a markdown trading report with LRP insights.

    Args:
        model: Trained model
        data_loader: Data to analyze
        feature_names: Feature names

    Returns:
        Markdown-formatted report string
    """
    model.eval()

    buy_signals = []
    sell_signals = []

    with torch.no_grad():
        for batch_x, _ in data_loader:
            for i in range(min(len(batch_x), 100)):  # Limit for speed
                sample = batch_x[i:i + 1]
                analysis = analyze_trading_signal(model, sample, feature_names)

                if analysis["signal"] == "BUY":
                    buy_signals.append(analysis)
                else:
                    sell_signals.append(analysis)

    report = "# LRP Trading Signal Analysis Report\n\n"

    report += "## Summary\n\n"
    report += f"- Total signals analyzed: {len(buy_signals) + len(sell_signals)}\n"
    report += f"- BUY signals: {len(buy_signals)}\n"
    report += f"- SELL/HOLD signals: {len(sell_signals)}\n\n"

    if buy_signals:
        report += "## BUY Signal Analysis\n\n"
        report += f"Average confidence: {np.mean([s['confidence'] for s in buy_signals]):.2%}\n\n"
        report += "### Top Contributing Features\n\n"

        # Aggregate buy signal explanations
        feature_scores = {}
        for analysis in buy_signals:
            for exp in analysis["explanations"][:5]:
                feat = exp["feature"]
                feature_scores[feat] = feature_scores.get(feat, 0) + abs(exp["relevance_pct"])

        for feat, score in sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
            report += f"- **{feat}**: {score / len(buy_signals):.1f}% avg\n"

    if sell_signals:
        report += "\n## SELL/HOLD Signal Analysis\n\n"
        report += f"Average confidence: {np.mean([s['confidence'] for s in sell_signals]):.2%}\n\n"
        report += "### Top Contributing Features\n\n"

        feature_scores = {}
        for analysis in sell_signals:
            for exp in analysis["explanations"][:5]:
                feat = exp["feature"]
                feature_scores[feat] = feature_scores.get(feat, 0) + abs(exp["relevance_pct"])

        for feat, score in sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
            report += f"- **{feat}**: {score / len(sell_signals):.1f}% avg\n"

    return report
