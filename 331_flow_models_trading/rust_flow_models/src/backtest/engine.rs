//! Backtesting engine

use crate::backtest::report::BacktestReport;
use crate::strategy::flow_strategy::FlowTradingStrategy;
use crate::strategy::signal::{Signal, SignalType};
use ndarray::{Array1, Array2};
use chrono::{DateTime, Utc};

/// Trade record
#[derive(Debug, Clone)]
pub struct Trade {
    pub timestamp: DateTime<Utc>,
    pub signal_type: SignalType,
    pub price: f64,
    pub size: f64,
    pub pnl: f64,
}

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Position size as fraction of capital
    pub position_size: f64,
    /// Transaction cost in basis points
    pub transaction_cost_bps: f64,
    /// Slippage in basis points
    pub slippage_bps: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            position_size: 0.1,
            transaction_cost_bps: 10.0,
            slippage_bps: 5.0,
        }
    }
}

/// Backtesting engine
pub struct BacktestEngine {
    pub config: BacktestConfig,
    pub capital: f64,
    pub position: f64,
    pub trades: Vec<Trade>,
    pub equity_curve: Vec<f64>,
}

impl BacktestEngine {
    /// Create new backtest engine
    pub fn new(config: BacktestConfig) -> Self {
        Self {
            capital: config.initial_capital,
            position: 0.0,
            trades: Vec::new(),
            equity_curve: Vec::new(),
            config,
        }
    }

    /// Reset engine state
    pub fn reset(&mut self) {
        self.capital = self.config.initial_capital;
        self.position = 0.0;
        self.trades.clear();
        self.equity_curve.clear();
    }

    /// Run backtest
    pub fn run(
        &mut self,
        strategy: &mut FlowTradingStrategy,
        features: &Array2<f64>,
        prices: &Array1<f64>,
        timestamps: &[DateTime<Utc>],
    ) -> BacktestReport {
        self.reset();

        let n_samples = features.nrows();
        log::info!("Running backtest on {} samples", n_samples);

        for i in 0..n_samples {
            let current_features = features.row(i).to_owned();
            let current_price = prices[i];
            let current_time = timestamps[i];

            // Generate signal
            let signal = strategy.generate_signal(&current_features);

            // Execute signal
            self.execute_signal(&signal, current_price, current_time);

            // Update equity
            let equity = self.capital + self.position * current_price;
            self.equity_curve.push(equity);
        }

        // Close any remaining position
        if self.position.abs() > 1e-10 {
            let last_price = prices[n_samples - 1];
            let last_time = timestamps[n_samples - 1];
            self.close_position(last_price, last_time);
        }

        // Generate report
        self.generate_report(prices, timestamps)
    }

    /// Execute trading signal
    fn execute_signal(&mut self, signal: &Signal, price: f64, timestamp: DateTime<Utc>) {
        match signal.signal_type {
            SignalType::ReduceExposure => {
                if self.position.abs() > 1e-10 {
                    self.close_position(price, timestamp);
                }
            }
            SignalType::Long => {
                if self.position <= 0.0 {
                    if self.position < 0.0 {
                        self.close_position(price, timestamp);
                    }
                    self.open_position(1, price, timestamp, signal.confidence);
                }
            }
            SignalType::Short => {
                if self.position >= 0.0 {
                    if self.position > 0.0 {
                        self.close_position(price, timestamp);
                    }
                    self.open_position(-1, price, timestamp, signal.confidence);
                }
            }
            SignalType::Neutral => {
                // Do nothing
            }
        }
    }

    /// Open a new position
    fn open_position(&mut self, direction: i32, price: f64, timestamp: DateTime<Utc>, confidence: f64) {
        // Apply slippage
        let slippage = self.config.slippage_bps / 10_000.0;
        let execution_price = if direction > 0 {
            price * (1.0 + slippage)
        } else {
            price * (1.0 - slippage)
        };

        // Calculate position size
        let size = self.capital * self.config.position_size * confidence / execution_price;
        let size = size * direction as f64;

        // Apply transaction cost
        let cost = (size * execution_price).abs() * self.config.transaction_cost_bps / 10_000.0;
        self.capital -= cost;

        self.position = size;

        self.trades.push(Trade {
            timestamp,
            signal_type: if direction > 0 { SignalType::Long } else { SignalType::Short },
            price: execution_price,
            size: size.abs(),
            pnl: 0.0,
        });
    }

    /// Close current position
    fn close_position(&mut self, price: f64, timestamp: DateTime<Utc>) {
        if self.position.abs() < 1e-10 {
            return;
        }

        // Apply slippage (opposite direction)
        let slippage = self.config.slippage_bps / 10_000.0;
        let execution_price = if self.position > 0.0 {
            price * (1.0 - slippage)
        } else {
            price * (1.0 + slippage)
        };

        // Calculate P&L
        let trade_value = self.position * execution_price;
        let cost = trade_value.abs() * self.config.transaction_cost_bps / 10_000.0;

        self.capital += trade_value - cost;

        // Update last trade P&L
        if let Some(last_trade) = self.trades.last_mut() {
            let entry_price = last_trade.price;
            last_trade.pnl = (execution_price - entry_price) * self.position - cost;
        }

        self.position = 0.0;
    }

    /// Generate backtest report
    fn generate_report(&self, prices: &Array1<f64>, timestamps: &[DateTime<Utc>]) -> BacktestReport {
        let equity: Array1<f64> = self.equity_curve.iter().cloned().collect();

        // Calculate returns
        let returns: Array1<f64> = (1..equity.len())
            .map(|i| (equity[i] - equity[i - 1]) / equity[i - 1])
            .collect();

        // Total return
        let total_return = (equity[equity.len() - 1] - self.config.initial_capital)
            / self.config.initial_capital;

        // Annualized metrics
        let n_days = if timestamps.len() >= 2 {
            (timestamps.last().unwrap().signed_duration_since(*timestamps.first().unwrap()))
                .num_days()
                .max(1) as f64
        } else {
            1.0
        };

        let annual_factor = 365.0 / n_days;
        let annual_return = (1.0 + total_return).powf(annual_factor) - 1.0;

        let returns_std = if returns.len() > 1 {
            returns.std(0.0)
        } else {
            0.0
        };
        let annual_vol = returns_std * (252.0_f64).sqrt();

        // Sharpe ratio
        let sharpe = if annual_vol > 0.0 {
            annual_return / annual_vol
        } else {
            0.0
        };

        // Sortino ratio
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_vol = if !downside_returns.is_empty() {
            let mean: f64 = downside_returns.iter().sum::<f64>() / downside_returns.len() as f64;
            let variance: f64 = downside_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / downside_returns.len() as f64;
            variance.sqrt() * (252.0_f64).sqrt()
        } else {
            0.0
        };
        let sortino = if downside_vol > 0.0 {
            annual_return / downside_vol
        } else {
            0.0
        };

        // Maximum drawdown
        let mut peak = equity[0];
        let mut max_drawdown = 0.0;
        for &e in equity.iter() {
            peak = peak.max(e);
            let drawdown = (peak - e) / peak;
            max_drawdown = max_drawdown.max(drawdown);
        }

        // Trade statistics
        let wins = self.trades.iter().filter(|t| t.pnl > 0.0).count();
        let win_rate = if !self.trades.is_empty() {
            wins as f64 / self.trades.len() as f64
        } else {
            0.0
        };

        let avg_pnl = if !self.trades.is_empty() {
            self.trades.iter().map(|t| t.pnl).sum::<f64>() / self.trades.len() as f64
        } else {
            0.0
        };

        BacktestReport {
            total_return,
            annual_return,
            annual_volatility: annual_vol,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown,
            win_rate,
            num_trades: self.trades.len(),
            avg_trade_pnl: avg_pnl,
            final_equity: equity[equity.len() - 1],
            equity_curve: equity.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_backtest_engine_creation() {
        let config = BacktestConfig::default();
        let engine = BacktestEngine::new(config.clone());

        assert_eq!(engine.capital, config.initial_capital);
        assert_eq!(engine.position, 0.0);
        assert!(engine.trades.is_empty());
    }
}
