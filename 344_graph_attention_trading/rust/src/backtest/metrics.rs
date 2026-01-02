//! Backtesting metrics and evaluation
//!
//! Compute performance metrics for trading strategies.

use crate::api::Candle;
use crate::features::FeatureExtractor;
use crate::gat::GraphAttentionNetwork;
use crate::graph::SparseGraph;
use crate::trading::{Portfolio, SignalGenerator, TradingStrategy};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Backtesting engine
pub struct Backtester {
    /// Initial capital
    initial_capital: f64,
    /// Trading fee (as fraction)
    fee: f64,
    /// Slippage (as fraction)
    slippage: f64,
}

impl Default for Backtester {
    fn default() -> Self {
        Self::new(10000.0)
    }
}

impl Backtester {
    /// Create new backtester
    pub fn new(initial_capital: f64) -> Self {
        Self {
            initial_capital,
            fee: 0.001,     // 0.1% fee
            slippage: 0.0005, // 0.05% slippage
        }
    }

    /// Set fees
    pub fn with_fee(mut self, fee: f64) -> Self {
        self.fee = fee;
        self
    }

    /// Set slippage
    pub fn with_slippage(mut self, slippage: f64) -> Self {
        self.slippage = slippage;
        self
    }

    /// Run backtest
    pub fn run(
        &self,
        candles_multi: &[Vec<Candle>],
        symbols: &[&str],
        gat: &GraphAttentionNetwork,
        graph: &SparseGraph,
        strategy: &TradingStrategy,
    ) -> BacktestResult {
        let feature_extractor = FeatureExtractor::new();
        let signal_generator = SignalGenerator::new();
        let mut portfolio = Portfolio::new(self.initial_capital);

        let min_len = candles_multi.iter().map(|c| c.len()).min().unwrap_or(0);
        let warmup = 50; // Warmup period for indicators

        let mut equity_curve = Vec::new();
        let mut returns = Vec::new();
        let mut prev_signals = Array1::zeros(symbols.len());

        for t in warmup..min_len {
            // Get current candles up to time t
            let current_candles: Vec<Vec<Candle>> = candles_multi
                .iter()
                .map(|c| c[..=t].to_vec())
                .collect();

            // Extract features
            let features = feature_extractor.extract_multi(&current_candles);

            // Generate signals
            let raw_signals = signal_generator.generate_raw(gat, &features, graph);
            let smoothed_signals = signal_generator.smooth_signals(
                &raw_signals,
                &prev_signals,
                strategy.signal_smoothing,
            );
            prev_signals = smoothed_signals.clone();

            // Get current prices
            let prices: HashMap<String, f64> = symbols
                .iter()
                .enumerate()
                .map(|(i, s)| (s.to_string(), candles_multi[i][t].close))
                .collect();

            // Update portfolio
            portfolio.update_prices(&prices);

            // Compute target weights
            let mut target_weights: HashMap<String, f64> = HashMap::new();
            for (i, &symbol) in symbols.iter().enumerate() {
                let signal = smoothed_signals[i];
                let weight = signal.clamp(-1.0, 1.0) * strategy.max_position;
                if weight.abs() > 0.01 {
                    target_weights.insert(symbol.to_string(), weight.max(0.0)); // Long only for simplicity
                }
            }

            // Rebalance if needed
            let current_weights = portfolio.weights();
            let needs_rebalance = target_weights.iter().any(|(symbol, &target)| {
                let current = current_weights.get(symbol).copied().unwrap_or(0.0);
                (current - target).abs() > strategy.rebalance_threshold
            });

            if needs_rebalance {
                portfolio.rebalance(&target_weights, &prices, candles_multi[0][t].timestamp);
            }

            // Record equity
            let equity = portfolio.total_value();
            equity_curve.push(equity);

            if equity_curve.len() > 1 {
                let prev_equity = equity_curve[equity_curve.len() - 2];
                returns.push((equity - prev_equity) / prev_equity);
            }
        }

        // Compute metrics
        let metrics = PerformanceMetrics::from_returns(&returns, &equity_curve);
        let trade_stats = portfolio.trade_stats();

        BacktestResult {
            metrics,
            equity_curve,
            returns,
            final_portfolio: portfolio,
            trade_stats,
        }
    }

    /// Run simple buy-and-hold benchmark
    pub fn benchmark_buy_hold(&self, candles: &[Candle]) -> BacktestResult {
        let mut equity_curve = Vec::new();
        let mut returns = Vec::new();

        let initial_price = candles.first().map(|c| c.close).unwrap_or(1.0);
        let shares = self.initial_capital / initial_price;

        for candle in candles {
            let equity = shares * candle.close;
            equity_curve.push(equity);

            if equity_curve.len() > 1 {
                let prev = equity_curve[equity_curve.len() - 2];
                returns.push((equity - prev) / prev);
            }
        }

        let metrics = PerformanceMetrics::from_returns(&returns, &equity_curve);

        BacktestResult {
            metrics,
            equity_curve,
            returns,
            final_portfolio: Portfolio::new(self.initial_capital),
            trade_stats: Default::default(),
        }
    }
}

/// Backtest result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub metrics: PerformanceMetrics,
    pub equity_curve: Vec<f64>,
    pub returns: Vec<f64>,
    pub final_portfolio: Portfolio,
    pub trade_stats: crate::trading::portfolio::TradeStats,
}

/// Performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annual_return: f64,
    /// Annualized volatility
    pub volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Win rate (from returns)
    pub win_rate: f64,
    /// Average win
    pub avg_win: f64,
    /// Average loss
    pub avg_loss: f64,
}

impl PerformanceMetrics {
    /// Compute metrics from returns and equity curve
    pub fn from_returns(returns: &[f64], equity_curve: &[f64]) -> Self {
        if returns.is_empty() || equity_curve.is_empty() {
            return Self::default();
        }

        // Assume hourly data, 24 * 365 periods per year
        let periods_per_year = 24.0 * 365.0;

        // Total return
        let initial = equity_curve.first().copied().unwrap_or(1.0);
        let final_val = equity_curve.last().copied().unwrap_or(1.0);
        let total_return = (final_val - initial) / initial;

        // Annualized return
        let years = returns.len() as f64 / periods_per_year;
        let annual_return = if years > 0.0 {
            (1.0 + total_return).powf(1.0 / years) - 1.0
        } else {
            0.0
        };

        // Volatility
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
        let volatility = variance.sqrt() * periods_per_year.sqrt();

        // Sharpe ratio (assuming 0 risk-free rate)
        let sharpe_ratio = if volatility > 0.0 {
            annual_return / volatility
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let negative_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
        let downside_variance = if !negative_returns.is_empty() {
            negative_returns.iter().map(|r| r.powi(2)).sum::<f64>() / negative_returns.len() as f64
        } else {
            0.0
        };
        let downside_std = downside_variance.sqrt() * periods_per_year.sqrt();
        let sortino_ratio = if downside_std > 0.0 {
            annual_return / downside_std
        } else {
            0.0
        };

        // Maximum drawdown
        let max_drawdown = Self::compute_max_drawdown(equity_curve);

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            annual_return / max_drawdown
        } else {
            0.0
        };

        // Win rate
        let positive_returns: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).copied().collect();
        let win_rate = positive_returns.len() as f64 / returns.len() as f64;

        // Average win/loss
        let avg_win = if !positive_returns.is_empty() {
            positive_returns.iter().sum::<f64>() / positive_returns.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !negative_returns.is_empty() {
            negative_returns.iter().sum::<f64>() / negative_returns.len() as f64
        } else {
            0.0
        };

        Self {
            total_return,
            annual_return,
            volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            win_rate,
            avg_win,
            avg_loss,
        }
    }

    /// Compute maximum drawdown
    fn compute_max_drawdown(equity_curve: &[f64]) -> f64 {
        let mut max_dd = 0.0;
        let mut peak = equity_curve.first().copied().unwrap_or(1.0);

        for &equity in equity_curve {
            if equity > peak {
                peak = equity;
            }
            let dd = (peak - equity) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        max_dd
    }

    /// Format metrics as string
    pub fn summary(&self) -> String {
        format!(
            "Performance Metrics:\n\
             Total Return: {:.2}%\n\
             Annual Return: {:.2}%\n\
             Volatility: {:.2}%\n\
             Sharpe Ratio: {:.3}\n\
             Sortino Ratio: {:.3}\n\
             Max Drawdown: {:.2}%\n\
             Calmar Ratio: {:.3}\n\
             Win Rate: {:.2}%\n\
             Avg Win: {:.4}%\n\
             Avg Loss: {:.4}%",
            self.total_return * 100.0,
            self.annual_return * 100.0,
            self.volatility * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.calmar_ratio,
            self.win_rate * 100.0,
            self.avg_win * 100.0,
            self.avg_loss * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 110.0, 105.0, 120.0, 90.0, 100.0];
        let dd = PerformanceMetrics::compute_max_drawdown(&equity);

        // Max drawdown is from 120 to 90 = 25%
        assert!((dd - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_metrics_from_returns() {
        let returns = vec![0.01, -0.02, 0.03, 0.01, -0.01, 0.02];
        let equity: Vec<f64> = returns
            .iter()
            .scan(100.0, |state, &r| {
                *state *= 1.0 + r;
                Some(*state)
            })
            .collect();

        let metrics = PerformanceMetrics::from_returns(&returns, &equity);

        assert!(metrics.total_return > 0.0);
        assert!(metrics.win_rate > 0.0);
    }
}
