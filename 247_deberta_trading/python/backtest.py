"""
Backtesting framework for DeBERTa sentiment-based trading strategies.

Provides:
- Sentiment-triggered event-driven backtesting
- Performance metrics calculation (Sharpe, Sortino, Max Drawdown)
- Risk management with position sizing
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from python.model import DeBERTaSentimentModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Result of a single trade."""
    entry_date: str
    exit_date: str
    symbol: str
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    return_pct: float
    sentiment_score: float
    holding_days: int


@dataclass
class BacktestResults:
    """Results from backtesting."""
    trades: List[TradeResult]
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int
    equity_curve: np.ndarray


class SentimentBacktester:
    """
    Backtester for DeBERTa sentiment-based trading strategies.

    The strategy generates signals based on sentiment analysis of
    financial text, then enters and exits positions based on
    configurable thresholds and risk parameters.

    Example:
        >>> from python.model import DeBERTaSentimentModel
        >>> from python.backtest import SentimentBacktester
        >>> model = DeBERTaSentimentModel()
        >>> backtester = SentimentBacktester(model=model)
        >>> results = backtester.run(price_data=df, headlines=headlines)
        >>> print(f"Sharpe: {results.sharpe_ratio:.4f}")
    """

    def __init__(
        self,
        model: Optional[DeBERTaSentimentModel] = None,
        initial_capital: float = 100000.0,
        position_pct: float = 0.1,
        entry_threshold: float = 0.6,
        stop_loss: float = 0.05,
        take_profit: float = 0.10,
        max_holding_days: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize sentiment backtester.

        Args:
            model: DeBERTa sentiment model instance
            initial_capital: Starting capital in USD
            position_pct: Fraction of capital per trade
            entry_threshold: Minimum sentiment confidence for entry
            stop_loss: Stop loss percentage (e.g., 0.05 = 5%)
            take_profit: Take profit percentage (e.g., 0.10 = 10%)
            max_holding_days: Maximum days to hold a position
            verbose: Enable verbose logging
        """
        self.model = model or DeBERTaSentimentModel(use_transformer=False)
        self.initial_capital = initial_capital
        self.position_pct = position_pct
        self.entry_threshold = entry_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_holding_days = max_holding_days
        self.verbose = verbose

    def run(
        self,
        price_data: pd.DataFrame,
        headlines: pd.DataFrame,
        symbol: str = "ASSET",
    ) -> BacktestResults:
        """
        Run backtest with sentiment signals.

        Args:
            price_data: DataFrame with columns [timestamp, open, high, low, close, volume]
            headlines: DataFrame with columns [timestamp, text]
            symbol: Asset symbol for logging

        Returns:
            BacktestResults with performance metrics
        """
        capital = self.initial_capital
        equity_curve = [capital]
        trades: List[TradeResult] = []
        position = None

        price_data = price_data.sort_values("timestamp").reset_index(drop=True)

        for i in range(1, len(price_data)):
            current_date = price_data.iloc[i]["timestamp"]
            current_price = price_data.iloc[i]["close"]
            prev_price = price_data.iloc[i - 1]["close"]

            # Check if we have headlines for this date
            day_headlines = headlines[
                pd.to_datetime(headlines["timestamp"]).dt.date
                == pd.to_datetime(current_date).date()
            ]

            # Manage existing position
            if position is not None:
                days_held = i - position["entry_idx"]
                price_change = (current_price - position["entry_price"]) / position["entry_price"]
                position_return = price_change * position["direction"]

                # Check exit conditions
                should_exit = False
                if position_return <= -self.stop_loss:
                    should_exit = True
                    if self.verbose:
                        logger.info(f"Stop loss hit at {current_date}")
                elif position_return >= self.take_profit:
                    should_exit = True
                    if self.verbose:
                        logger.info(f"Take profit hit at {current_date}")
                elif days_held >= self.max_holding_days:
                    should_exit = True
                    if self.verbose:
                        logger.info(f"Max holding period reached at {current_date}")

                if should_exit:
                    pnl = position["size"] * position_return
                    capital += position["size"] + pnl

                    trades.append(TradeResult(
                        entry_date=str(position["entry_date"]),
                        exit_date=str(current_date),
                        symbol=symbol,
                        direction=position["direction"],
                        entry_price=position["entry_price"],
                        exit_price=current_price,
                        position_size=position["size"],
                        pnl=pnl,
                        return_pct=position_return,
                        sentiment_score=position["sentiment"],
                        holding_days=days_held,
                    ))
                    position = None

            # Generate new signal if no position
            if position is None and len(day_headlines) > 0:
                texts = day_headlines["text"].tolist()
                signal_data = self.model.get_trading_signal(
                    texts, threshold=self.entry_threshold
                )

                if abs(signal_data["signal"]) > 0:
                    direction = 1 if signal_data["signal"] > 0 else -1
                    size = capital * self.position_pct * signal_data["confidence"]
                    capital -= size

                    position = {
                        "entry_date": current_date,
                        "entry_price": current_price,
                        "entry_idx": i,
                        "direction": direction,
                        "size": size,
                        "sentiment": signal_data["signal"],
                    }

                    if self.verbose:
                        dir_str = "LONG" if direction > 0 else "SHORT"
                        logger.info(
                            f"{current_date}: {dir_str} {symbol} @ {current_price:.2f} "
                            f"(sentiment: {signal_data['signal']:.3f})"
                        )

            equity_curve.append(capital + (
                position["size"] if position else 0.0
            ))

        # Close any remaining position
        if position is not None:
            final_price = price_data.iloc[-1]["close"]
            price_change = (final_price - position["entry_price"]) / position["entry_price"]
            position_return = price_change * position["direction"]
            pnl = position["size"] * position_return
            capital += position["size"] + pnl

            trades.append(TradeResult(
                entry_date=str(position["entry_date"]),
                exit_date=str(price_data.iloc[-1]["timestamp"]),
                symbol=symbol,
                direction=position["direction"],
                entry_price=position["entry_price"],
                exit_price=final_price,
                position_size=position["size"],
                pnl=pnl,
                return_pct=position_return,
                sentiment_score=position["sentiment"],
                holding_days=len(price_data) - 1 - position["entry_idx"],
            ))

        equity = np.array(equity_curve)
        metrics = self._compute_metrics(trades, equity)

        return BacktestResults(
            trades=trades,
            total_return=metrics["total_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"],
            max_drawdown=metrics["max_drawdown"],
            win_rate=metrics["win_rate"],
            profit_factor=metrics["profit_factor"],
            n_trades=len(trades),
            equity_curve=equity,
        )

    def _compute_metrics(
        self, trades: List[TradeResult], equity: np.ndarray
    ) -> Dict[str, float]:
        """Compute performance metrics from trade results."""
        if len(trades) == 0:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
            }

        returns = np.array([t.return_pct for t in trades])
        total_return = (equity[-1] - equity[0]) / equity[0]

        # Sharpe Ratio (annualized, assuming daily)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino Ratio
        downside = returns[returns < 0]
        if len(downside) > 0 and np.std(downside) > 0:
            sortino = np.mean(returns) / np.std(downside) * np.sqrt(252)
        else:
            sortino = sharpe

        # Maximum Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        # Win Rate
        wins = sum(1 for t in trades if t.pnl > 0)
        win_rate = wins / len(trades)

        # Profit Factor
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }
