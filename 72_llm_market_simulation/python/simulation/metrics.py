"""
Performance Metrics for LLM Market Simulation

Calculates various performance and risk metrics.
"""

from typing import Dict, List, Any
import numpy as np


def calculate_performance_metrics(
    price_history: List[float],
    fundamental_history: List[float] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics

    Args:
        price_history: List of prices over time
        fundamental_history: List of fundamental values (optional)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Dictionary of performance metrics
    """
    prices = np.array(price_history)
    metrics = {}

    if len(prices) < 2:
        return metrics

    # Returns
    returns = np.diff(prices) / prices[:-1]

    # Total return
    total_return = (prices[-1] - prices[0]) / prices[0]
    metrics["total_return"] = float(total_return)
    metrics["total_return_pct"] = float(total_return * 100)

    # Annualized return (CAGR)
    num_periods = len(prices) - 1
    years = num_periods / periods_per_year
    if years > 0:
        cagr = (prices[-1] / prices[0]) ** (1 / years) - 1
        metrics["cagr"] = float(cagr)
        metrics["cagr_pct"] = float(cagr * 100)

    # Volatility
    volatility = np.std(returns) * np.sqrt(periods_per_year)
    metrics["volatility"] = float(volatility)
    metrics["volatility_pct"] = float(volatility * 100)

    # Sharpe Ratio
    excess_return = metrics.get("cagr", 0) - risk_free_rate
    if volatility > 0:
        sharpe = excess_return / volatility
        metrics["sharpe_ratio"] = float(sharpe)

    # Sortino Ratio (downside deviation)
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0:
        downside_vol = np.std(negative_returns) * np.sqrt(periods_per_year)
        if downside_vol > 0:
            sortino = excess_return / downside_vol
            metrics["sortino_ratio"] = float(sortino)

    # Maximum Drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    metrics["max_drawdown"] = float(max_drawdown)
    metrics["max_drawdown_pct"] = float(max_drawdown * 100)

    # Calmar Ratio
    if max_drawdown < 0:
        calmar = metrics.get("cagr", 0) / abs(max_drawdown)
        metrics["calmar_ratio"] = float(calmar)

    # VaR (Value at Risk) at 95%
    var_95 = np.percentile(returns, 5)
    metrics["var_95_daily"] = float(var_95)
    metrics["var_95_daily_pct"] = float(var_95 * 100)

    # Win rate
    positive_returns = returns[returns > 0]
    win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
    metrics["win_rate"] = float(win_rate)
    metrics["win_rate_pct"] = float(win_rate * 100)

    # Average win / average loss
    if len(positive_returns) > 0:
        avg_win = np.mean(positive_returns)
        metrics["avg_win"] = float(avg_win)
        metrics["avg_win_pct"] = float(avg_win * 100)

    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0:
        avg_loss = np.mean(negative_returns)
        metrics["avg_loss"] = float(avg_loss)
        metrics["avg_loss_pct"] = float(avg_loss * 100)

    # Profit factor
    if len(positive_returns) > 0 and len(negative_returns) > 0:
        total_wins = np.sum(positive_returns)
        total_losses = abs(np.sum(negative_returns))
        if total_losses > 0:
            profit_factor = total_wins / total_losses
            metrics["profit_factor"] = float(profit_factor)

    # Price discovery metrics (if fundamental history provided)
    if fundamental_history is not None and len(fundamental_history) == len(prices):
        fundamentals = np.array(fundamental_history)
        deviations = prices - fundamentals

        metrics["tracking_error"] = float(np.std(deviations))
        metrics["mean_absolute_deviation"] = float(np.mean(np.abs(deviations)))
        metrics["final_deviation"] = float(deviations[-1])
        metrics["final_deviation_pct"] = float(deviations[-1] / fundamentals[-1] * 100)

        # Correlation with fundamental
        if np.std(prices) > 0 and np.std(fundamentals) > 0:
            correlation = np.corrcoef(prices, fundamentals)[0, 1]
            metrics["fundamental_correlation"] = float(correlation)

    return metrics


def calculate_agent_metrics(
    trade_history: List[Dict[str, Any]],
    price_history: List[float],
    initial_value: float
) -> Dict[str, float]:
    """
    Calculate performance metrics for a single agent

    Args:
        trade_history: List of trade records
        price_history: Price history for calculating returns
        initial_value: Initial portfolio value

    Returns:
        Dictionary of agent-specific metrics
    """
    metrics = {}

    if not trade_history:
        return {"num_trades": 0}

    metrics["num_trades"] = len(trade_history)

    # Calculate PnL from trades
    total_pnl = 0
    winning_trades = 0
    losing_trades = 0
    total_win = 0
    total_loss = 0

    for i, trade in enumerate(trade_history):
        action = trade.get("action", "")
        qty = trade.get("quantity", 0)
        price = trade.get("price", 0)

        # Simplified PnL calculation
        if action == "sell" and i > 0:
            # Find matching buy
            for prev_trade in reversed(trade_history[:i]):
                if prev_trade.get("action") == "buy":
                    pnl = (price - prev_trade.get("price", price)) * qty
                    total_pnl += pnl
                    if pnl > 0:
                        winning_trades += 1
                        total_win += pnl
                    else:
                        losing_trades += 1
                        total_loss += abs(pnl)
                    break

    metrics["total_pnl"] = float(total_pnl)
    metrics["winning_trades"] = winning_trades
    metrics["losing_trades"] = losing_trades

    if winning_trades + losing_trades > 0:
        metrics["trade_win_rate"] = float(winning_trades / (winning_trades + losing_trades))

    if winning_trades > 0:
        metrics["avg_winning_trade"] = float(total_win / winning_trades)

    if losing_trades > 0:
        metrics["avg_losing_trade"] = float(total_loss / losing_trades)

    if total_loss > 0:
        metrics["trade_profit_factor"] = float(total_win / total_loss)

    return metrics


def detect_bubble(
    price_history: List[float],
    fundamental_history: List[float],
    bubble_threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Detect bubble formation in price series

    Args:
        price_history: Historical prices
        fundamental_history: Fundamental values
        bubble_threshold: Percentage above fundamental to qualify as bubble

    Returns:
        Dictionary with bubble analysis
    """
    prices = np.array(price_history)
    fundamentals = np.array(fundamental_history)

    if len(prices) != len(fundamentals):
        return {"error": "Price and fundamental arrays must have same length"}

    deviation = (prices - fundamentals) / fundamentals

    # Find bubble periods (price > fundamental by threshold)
    bubble_mask = deviation > bubble_threshold

    if not any(bubble_mask):
        return {"bubble_detected": False}

    # Find bubble periods
    bubble_periods = []
    in_bubble = False
    start_idx = 0

    for i, is_bubble in enumerate(bubble_mask):
        if is_bubble and not in_bubble:
            in_bubble = True
            start_idx = i
        elif not is_bubble and in_bubble:
            in_bubble = False
            peak_idx = start_idx + np.argmax(prices[start_idx:i])
            bubble_periods.append({
                "start": int(start_idx),
                "peak": int(peak_idx),
                "end": int(i),
                "peak_deviation": float(deviation[peak_idx]),
                "duration": int(i - start_idx)
            })

    # Handle bubble at end
    if in_bubble:
        peak_idx = start_idx + np.argmax(prices[start_idx:])
        bubble_periods.append({
            "start": int(start_idx),
            "peak": int(peak_idx),
            "end": int(len(prices) - 1),
            "peak_deviation": float(deviation[peak_idx]),
            "duration": int(len(prices) - 1 - start_idx)
        })

    return {
        "bubble_detected": True,
        "num_bubbles": len(bubble_periods),
        "bubble_periods": bubble_periods,
        "max_deviation": float(np.max(deviation)),
        "time_in_bubble_pct": float(np.mean(bubble_mask) * 100)
    }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate sample price series
    prices = [100.0]
    for _ in range(250):
        change = np.random.normal(0.0002, 0.02)
        prices.append(prices[-1] * (1 + change))

    # Generate fundamental value series (less volatile)
    fundamentals = [100.0]
    for _ in range(250):
        change = np.random.normal(0.0002, 0.01)
        fundamentals.append(fundamentals[-1] * (1 + change))

    # Calculate metrics
    metrics = calculate_performance_metrics(prices, fundamentals)

    print("Performance Metrics:")
    print("=" * 40)
    for key, value in sorted(metrics.items()):
        if "pct" in key:
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value:.4f}")

    # Detect bubbles
    bubble_info = detect_bubble(prices, fundamentals)
    print("\nBubble Analysis:")
    print("=" * 40)
    for key, value in bubble_info.items():
        print(f"{key}: {value}")
