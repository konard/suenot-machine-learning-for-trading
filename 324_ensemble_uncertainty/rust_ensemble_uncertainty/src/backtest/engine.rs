//! Backtesting engine.

use crate::strategy::signal::{Signal, SignalType};
use crate::strategy::uncertainty_strategy::UncertaintyStrategy;
use super::report::BacktestReport;

/// Trade record
#[derive(Debug, Clone)]
pub struct Trade {
    pub symbol: String,
    pub entry_time: i64,
    pub exit_time: i64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub position_size: f64,
    pub direction: SignalType,
    pub prediction: f64,
    pub uncertainty: f64,
    pub confidence: f64,
    pub pnl: f64,
    pub pnl_pct: f64,
    pub exit_reason: String,
}

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub transaction_cost: f64,
    pub slippage: f64,
    pub risk_free_rate: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100000.0,
            transaction_cost: 0.001,
            slippage: 0.0005,
            risk_free_rate: 0.02,
        }
    }
}

/// Backtesting engine
pub struct BacktestEngine {
    config: BacktestConfig,
}

impl BacktestEngine {
    /// Create new backtest engine
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest
    pub fn run(
        &self,
        timestamps: &[i64],
        prices: &[f64],
        predictions: &[f64],
        uncertainties: &[f64],
        strategy: &UncertaintyStrategy,
        hold_periods: usize,
    ) -> BacktestReport {
        let n_periods = prices.len();
        let mut capital = self.config.initial_capital;
        let mut equity: Vec<f64> = vec![capital];
        let mut trades: Vec<Trade> = Vec::new();
        let mut current_trade: Option<(usize, Signal, f64)> = None; // (entry_idx, signal, entry_price)

        for i in 0..n_periods.saturating_sub(hold_periods) {
            // Check if we need to close existing position
            if let Some((entry_idx, ref signal, entry_price)) = current_trade {
                let periods_held = i - entry_idx;
                if periods_held >= hold_periods {
                    // Close position
                    let exit_price = prices[i] * (1.0 - self.config.slippage);

                    let pnl_pct = match signal.signal_type {
                        SignalType::Long => (exit_price - entry_price) / entry_price,
                        SignalType::Short => (entry_price - exit_price) / entry_price,
                        _ => 0.0,
                    } - self.config.transaction_cost;

                    let pnl = capital * signal.position_size * pnl_pct;
                    capital += pnl;

                    trades.push(Trade {
                        symbol: signal.symbol.clone(),
                        entry_time: timestamps[entry_idx],
                        exit_time: timestamps[i],
                        entry_price,
                        exit_price,
                        position_size: signal.position_size,
                        direction: signal.signal_type,
                        prediction: signal.prediction,
                        uncertainty: signal.uncertainty,
                        confidence: signal.confidence,
                        pnl,
                        pnl_pct,
                        exit_reason: "hold_period".to_string(),
                    });

                    current_trade = None;
                }
            }

            // Open new position if none active
            if current_trade.is_none() {
                let signal = strategy.generate_signal(
                    "TEST",
                    predictions[i],
                    uncertainties[i],
                    Some(prices[i]),
                    None,
                );

                if signal.signal_type != SignalType::Hold && signal.position_size > 0.0 {
                    let entry_price = prices[i] * (1.0 + self.config.slippage + self.config.transaction_cost);
                    current_trade = Some((i, signal, entry_price));
                }
            }

            equity.push(capital);
        }

        // Close any remaining position
        if let Some((entry_idx, signal, entry_price)) = current_trade {
            let exit_idx = n_periods - 1;
            let exit_price = prices[exit_idx];

            let pnl_pct = match signal.signal_type {
                SignalType::Long => (exit_price - entry_price) / entry_price,
                SignalType::Short => (entry_price - exit_price) / entry_price,
                _ => 0.0,
            } - self.config.transaction_cost;

            let pnl = capital * signal.position_size * pnl_pct;
            capital += pnl;

            trades.push(Trade {
                symbol: signal.symbol.clone(),
                entry_time: timestamps[entry_idx],
                exit_time: timestamps[exit_idx],
                entry_price,
                exit_price,
                position_size: signal.position_size,
                direction: signal.signal_type,
                prediction: signal.prediction,
                uncertainty: signal.uncertainty,
                confidence: signal.confidence,
                pnl,
                pnl_pct,
                exit_reason: "end".to_string(),
            });
        }

        equity.push(capital);

        // Calculate metrics
        self.calculate_report(equity, trades, &self.config)
    }

    /// Calculate backtest report from results
    fn calculate_report(
        &self,
        equity: Vec<f64>,
        trades: Vec<Trade>,
        config: &BacktestConfig,
    ) -> BacktestReport {
        let total_return = (equity.last().unwrap_or(&config.initial_capital) - config.initial_capital)
            / config.initial_capital;

        // Returns
        let returns: Vec<f64> = equity
            .windows(2)
            .map(|w| if w[0] > 0.0 { (w[1] - w[0]) / w[0] } else { 0.0 })
            .collect();

        // Sharpe ratio
        let mean_return = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
        let std_return = (returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
            / returns.len().max(1) as f64)
            .sqrt();
        let excess_return = mean_return - config.risk_free_rate / 252.0;
        let sharpe_ratio = if std_return > 0.0 {
            (252.0_f64).sqrt() * excess_return / std_return
        } else {
            0.0
        };

        // Sortino ratio
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
        let downside_std = if !downside_returns.is_empty() {
            (downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                / downside_returns.len() as f64)
                .sqrt()
        } else {
            1e-10
        };
        let sortino_ratio = (252.0_f64).sqrt() * mean_return / downside_std;

        // Max drawdown
        let mut max_equity = config.initial_capital;
        let mut max_drawdown = 0.0;
        for &e in &equity {
            max_equity = max_equity.max(e);
            let drawdown = (max_equity - e) / max_equity;
            max_drawdown = max_drawdown.max(drawdown);
        }

        // Trade statistics
        let winning_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let win_rate = if !trades.is_empty() {
            winning_trades.len() as f64 / trades.len() as f64
        } else {
            0.0
        };

        let total_wins: f64 = winning_trades.iter().map(|t| t.pnl).sum();
        let total_losses: f64 = losing_trades.iter().map(|t| t.pnl.abs()).sum();
        let profit_factor = if total_losses > 0.0 {
            total_wins / total_losses
        } else {
            0.0
        };

        let avg_trade_pnl = if !trades.is_empty() {
            trades.iter().map(|t| t.pnl).sum::<f64>() / trades.len() as f64
        } else {
            0.0
        };

        BacktestReport {
            total_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown: -max_drawdown,
            win_rate,
            profit_factor,
            total_trades: trades.len(),
            avg_trade_pnl,
            equity_curve: equity,
            trades,
        }
    }
}

impl Default for BacktestEngine {
    fn default() -> Self {
        Self::new(BacktestConfig::default())
    }
}
