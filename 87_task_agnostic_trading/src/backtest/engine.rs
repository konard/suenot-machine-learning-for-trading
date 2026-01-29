//! Backtesting engine for evaluating task-agnostic trading strategies

use crate::strategy::TradingSignal;
use serde::{Deserialize, Serialize};

/// Configuration for the backtesting engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Transaction cost as a fraction (e.g., 0.001 = 0.1%)
    pub transaction_cost: f64,
    /// Slippage as a fraction
    pub slippage: f64,
    /// Maximum leverage allowed
    pub max_leverage: f64,
    /// Risk-free rate for Sharpe ratio (annualized)
    pub risk_free_rate: f64,
    /// Trading periods per year (252 for stocks, 365 for crypto)
    pub periods_per_year: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            transaction_cost: 0.001,
            slippage: 0.0005,
            max_leverage: 1.0,
            risk_free_rate: 0.05,
            periods_per_year: 365.0, // Crypto default
        }
    }
}

/// Performance metrics from a backtest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return as a percentage
    pub total_return_pct: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized)
    pub sortino_ratio: f64,
    /// Maximum drawdown as a percentage
    pub max_drawdown_pct: f64,
    /// Calmar ratio (annualized return / max drawdown)
    pub calmar_ratio: f64,
    /// Win rate (percentage of profitable trades)
    pub win_rate: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Number of trades
    pub total_trades: usize,
    /// Average trade return
    pub avg_trade_return: f64,
    /// Average winning trade
    pub avg_win: f64,
    /// Average losing trade
    pub avg_loss: f64,
    /// Final equity
    pub final_equity: f64,
}

/// Result of a backtest run
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Equity curve (portfolio value over time)
    pub equity_curve: Vec<f64>,
    /// Returns per period
    pub returns: Vec<f64>,
    /// Drawdown curve
    pub drawdowns: Vec<f64>,
    /// Trade log
    pub trades: Vec<TradeRecord>,
}

/// Record of a single trade
#[derive(Debug, Clone)]
pub struct TradeRecord {
    /// Entry index
    pub entry_idx: usize,
    /// Exit index
    pub exit_idx: usize,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position direction (1 = long, -1 = short)
    pub direction: f64,
    /// Position size
    pub size: f64,
    /// Profit/loss
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
}

/// Backtesting engine
pub struct BacktestEngine {
    config: BacktestConfig,
}

impl BacktestEngine {
    /// Create a new backtesting engine
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run a backtest using trading signals and price data
    pub fn run(
        &self,
        signals: &[TradingSignal],
        prices: &[f64],
    ) -> BacktestResult {
        assert_eq!(signals.len(), prices.len(), "Signals and prices must have same length");

        let n = signals.len();
        let mut equity = self.config.initial_capital;
        let mut equity_curve = vec![equity];
        let mut returns = Vec::new();
        let mut trades: Vec<TradeRecord> = Vec::new();

        let mut current_position: f64 = 0.0; // -1 to 1
        let mut entry_price: f64 = 0.0;
        let mut entry_idx: usize = 0;

        for i in 0..n {
            let target_position = signals[i].signal_type.direction() * signals[i].position_size;

            // Check if position change is needed
            if (target_position - current_position).abs() > 0.01 {
                // Close existing position
                if current_position.abs() > 0.01 && i > 0 {
                    let exit_price = prices[i] * (1.0 - self.config.slippage * current_position.signum());
                    let trade_return = current_position * (exit_price - entry_price) / entry_price;
                    let cost = self.config.transaction_cost * current_position.abs();
                    let net_return = trade_return - cost;

                    equity *= 1.0 + net_return;

                    trades.push(TradeRecord {
                        entry_idx,
                        exit_idx: i,
                        entry_price,
                        exit_price,
                        direction: current_position.signum(),
                        size: current_position.abs(),
                        pnl: equity * net_return,
                        return_pct: net_return * 100.0,
                    });
                }

                // Open new position
                if target_position.abs() > 0.01 {
                    entry_price = prices[i] * (1.0 + self.config.slippage * target_position.signum());
                    entry_idx = i;
                    let cost = self.config.transaction_cost * target_position.abs();
                    equity *= 1.0 - cost;
                }

                current_position = target_position;
            }

            // Track equity
            if current_position.abs() > 0.01 && i > 0 {
                let unrealized = current_position * (prices[i] - entry_price) / entry_price;
                let period_equity = equity * (1.0 + unrealized);
                equity_curve.push(period_equity);
            } else {
                equity_curve.push(equity);
            }

            // Period return
            if equity_curve.len() >= 2 {
                let prev = equity_curve[equity_curve.len() - 2];
                let curr = equity_curve[equity_curve.len() - 1];
                if prev > 0.0 {
                    returns.push((curr - prev) / prev);
                } else {
                    returns.push(0.0);
                }
            }
        }

        // Close final position
        if current_position.abs() > 0.01 {
            let exit_price = prices[n - 1];
            let trade_return = current_position * (exit_price - entry_price) / entry_price;
            let cost = self.config.transaction_cost * current_position.abs();
            let net_return = trade_return - cost;
            equity *= 1.0 + net_return;

            trades.push(TradeRecord {
                entry_idx,
                exit_idx: n - 1,
                entry_price,
                exit_price,
                direction: current_position.signum(),
                size: current_position.abs(),
                pnl: equity * net_return,
                return_pct: net_return * 100.0,
            });

            *equity_curve.last_mut().unwrap() = equity;
        }

        // Compute drawdowns
        let mut peak = self.config.initial_capital;
        let drawdowns: Vec<f64> = equity_curve
            .iter()
            .map(|&eq| {
                if eq > peak {
                    peak = eq;
                }
                (peak - eq) / peak
            })
            .collect();

        let metrics = self.compute_metrics(&equity_curve, &returns, &drawdowns, &trades);

        BacktestResult {
            metrics,
            equity_curve,
            returns,
            drawdowns,
            trades,
        }
    }

    /// Compute performance metrics
    fn compute_metrics(
        &self,
        equity_curve: &[f64],
        returns: &[f64],
        drawdowns: &[f64],
        trades: &[TradeRecord],
    ) -> PerformanceMetrics {
        let initial = self.config.initial_capital;
        let final_equity = equity_curve.last().copied().unwrap_or(initial);

        // Total return
        let total_return_pct = (final_equity - initial) / initial * 100.0;

        // Annualized return
        let n_periods = returns.len() as f64;
        let periods_per_year = self.config.periods_per_year;
        let years = n_periods / periods_per_year;
        let annualized_return = if years > 0.0 {
            ((final_equity / initial).powf(1.0 / years) - 1.0) * 100.0
        } else {
            0.0
        };

        // Sharpe ratio
        let mean_return = if returns.is_empty() {
            0.0
        } else {
            returns.iter().sum::<f64>() / n_periods
        };
        let std_return = if returns.len() > 1 {
            let var: f64 = returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / (n_periods - 1.0);
            var.sqrt()
        } else {
            0.0
        };

        let daily_rf = self.config.risk_free_rate / periods_per_year;
        let sharpe_ratio = if std_return > 1e-10 {
            (mean_return - daily_rf) / std_return * periods_per_year.sqrt()
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = returns
            .iter()
            .filter(|&&r| r < daily_rf)
            .map(|&r| (r - daily_rf).powi(2))
            .collect();
        let downside_std = if !downside_returns.is_empty() {
            (downside_returns.iter().sum::<f64>() / downside_returns.len() as f64).sqrt()
        } else {
            0.0
        };
        let sortino_ratio = if downside_std > 1e-10 {
            (mean_return - daily_rf) / downside_std * periods_per_year.sqrt()
        } else {
            0.0
        };

        // Maximum drawdown
        let max_drawdown_pct = drawdowns
            .iter()
            .cloned()
            .fold(0.0f64, f64::max)
            * 100.0;

        // Calmar ratio
        let calmar_ratio = if max_drawdown_pct > 0.01 {
            annualized_return / max_drawdown_pct
        } else {
            0.0
        };

        // Trade statistics
        let winning_trades: Vec<&TradeRecord> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<&TradeRecord> = trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let win_rate = if !trades.is_empty() {
            winning_trades.len() as f64 / trades.len() as f64 * 100.0
        } else {
            0.0
        };

        let gross_profit: f64 = winning_trades.iter().map(|t| t.pnl).sum();
        let gross_loss: f64 = losing_trades.iter().map(|t| t.pnl.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_trade_return = if !trades.is_empty() {
            trades.iter().map(|t| t.return_pct).sum::<f64>() / trades.len() as f64
        } else {
            0.0
        };

        let avg_win = if !winning_trades.is_empty() {
            winning_trades.iter().map(|t| t.return_pct).sum::<f64>() / winning_trades.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losing_trades.is_empty() {
            losing_trades.iter().map(|t| t.return_pct).sum::<f64>() / losing_trades.len() as f64
        } else {
            0.0
        };

        PerformanceMetrics {
            total_return_pct,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown_pct,
            calmar_ratio,
            win_rate,
            profit_factor,
            total_trades: trades.len(),
            avg_trade_return,
            avg_win,
            avg_loss,
            final_equity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::{SignalConfig, SignalGenerator};

    #[test]
    fn test_backtest_basic() {
        let engine = BacktestEngine::new(BacktestConfig::default());
        let gen = SignalGenerator::new(SignalConfig::default());

        // Simulate uptrending market with buy signals
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.5).collect();
        let signals: Vec<TradingSignal> = prices
            .iter()
            .map(|_| gen.generate_single(0.5, 0.8, "Trending", 0.2))
            .collect();

        let result = engine.run(&signals, &prices);

        assert!(result.metrics.total_return_pct > 0.0);
        assert!(!result.equity_curve.is_empty());
        assert!(result.metrics.total_trades > 0);
    }

    #[test]
    fn test_hold_signals() {
        let engine = BacktestEngine::new(BacktestConfig::default());
        let gen = SignalGenerator::new(SignalConfig::default());

        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let signals: Vec<TradingSignal> = prices
            .iter()
            .map(|_| gen.generate_single(0.0, 0.3, "Calm", 0.1))
            .collect();

        let result = engine.run(&signals, &prices);

        // Should have no trades (all hold signals)
        assert_eq!(result.metrics.total_trades, 0);
    }
}
