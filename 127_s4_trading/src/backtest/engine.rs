//! Backtesting engine implementation.

use crate::data::bybit::MarketData;
use crate::trading::strategy::{TradingStrategy, StrategyConfig};
use crate::model::s4::S4Config;
use serde::{Deserialize, Serialize};

/// Configuration for backtesting.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital.
    pub initial_capital: f64,
    /// Trading fee (percentage).
    pub trading_fee: f64,
    /// Slippage (percentage).
    pub slippage: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            trading_fee: 0.001,
            slippage: 0.0005,
        }
    }
}

/// Results from backtesting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Total return percentage.
    pub total_return: f64,
    /// Annualized return percentage.
    pub annual_return: f64,
    /// Sharpe ratio (annualized).
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized).
    pub sortino_ratio: f64,
    /// Maximum drawdown percentage.
    pub max_drawdown: f64,
    /// Calmar ratio.
    pub calmar_ratio: f64,
    /// Win rate percentage.
    pub win_rate: f64,
    /// Profit factor.
    pub profit_factor: f64,
    /// Total number of trades.
    pub total_trades: usize,
    /// Final equity.
    pub final_equity: f64,
    /// Equity curve.
    pub equity_curve: Vec<f64>,
}

impl BacktestResult {
    /// Print a summary of the results.
    pub fn summary(&self) -> String {
        format!(
            r#"
Backtest Results:
-----------------
Total Return:    {:.2}%
Annual Return:   {:.2}%
Sharpe Ratio:    {:.2}
Sortino Ratio:   {:.2}
Max Drawdown:    {:.2}%
Calmar Ratio:    {:.2}
Win Rate:        {:.2}%
Profit Factor:   {:.2}
Total Trades:    {}
Final Equity:    ${:.2}
"#,
            self.total_return * 100.0,
            self.annual_return * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.calmar_ratio,
            self.win_rate * 100.0,
            self.profit_factor,
            self.total_trades,
            self.final_equity,
        )
    }
}

/// Backtesting engine for S4 strategies.
pub struct BacktestEngine {
    config: BacktestConfig,
    strategy: TradingStrategy,
}

impl BacktestEngine {
    /// Create a new backtest engine.
    pub fn new(
        backtest_config: BacktestConfig,
        model_config: S4Config,
        strategy_config: StrategyConfig,
    ) -> Self {
        Self {
            config: backtest_config,
            strategy: TradingStrategy::new(model_config, strategy_config),
        }
    }

    /// Create with default configurations.
    pub fn with_defaults() -> Self {
        Self::new(
            BacktestConfig::default(),
            S4Config::default(),
            StrategyConfig::default(),
        )
    }

    /// Run backtest on market data.
    pub fn run(&mut self, data: &MarketData) -> BacktestResult {
        self.strategy.reset();

        let n = data.candles.len();
        if n < 2 {
            return self.empty_result();
        }

        let mut equity = self.config.initial_capital;
        let mut equity_curve = vec![equity];
        let mut position = 0.0;
        let mut entry_price = 0.0;

        let mut returns: Vec<f64> = Vec::new();
        let mut trades: Vec<f64> = Vec::new(); // P&L per trade
        let mut total_trades = 0;

        let lookback = 256.min(n / 2);

        for i in lookback..n {
            // Create a slice of data up to current point
            let slice = MarketData {
                symbol: data.symbol.clone(),
                interval: data.interval.clone(),
                candles: data.candles[..=i].to_vec(),
                source: data.source.clone(),
            };

            let current_price = data.candles[i].close;
            let prev_price = data.candles[i - 1].close;

            // Update equity based on position
            if position.abs() > 0.001 {
                let position_return = (current_price - prev_price) / prev_price;
                let position_pnl = equity * position * position_return;
                equity += position_pnl;
            }

            // Record daily return
            if !equity_curve.is_empty() {
                let daily_return = (equity - equity_curve.last().unwrap()) / equity_curve.last().unwrap();
                returns.push(daily_return);
            }
            equity_curve.push(equity);

            // Generate signal and update position
            let (signal, position_change) = self.strategy.on_data(&slice);

            if position_change.abs() > 0.001 {
                // Apply trading costs
                let trade_cost = equity * position_change.abs() * (self.config.trading_fee + self.config.slippage);
                equity -= trade_cost;

                // Track trade P&L
                if position.abs() > 0.001 && position_change.signum() != position.signum() {
                    // Closing trade
                    let trade_pnl = (current_price - entry_price) / entry_price * position;
                    trades.push(trade_pnl);
                }

                // Update position
                let old_position = position;
                position += position_change;

                if position.abs() > 0.001 && old_position.abs() < 0.001 {
                    // Opening new trade
                    entry_price = current_price;
                }

                total_trades += 1;

                tracing::debug!(
                    "t={}: {} | Position: {:.4} -> {:.4} | Equity: ${:.2}",
                    i, signal, old_position, position, equity
                );
            }
        }

        self.calculate_metrics(equity_curve, returns, trades, total_trades)
    }

    /// Calculate performance metrics.
    fn calculate_metrics(
        &self,
        equity_curve: Vec<f64>,
        returns: Vec<f64>,
        trades: Vec<f64>,
        total_trades: usize,
    ) -> BacktestResult {
        let initial = self.config.initial_capital;
        let final_equity = *equity_curve.last().unwrap_or(&initial);

        // Total return
        let total_return = (final_equity - initial) / initial;

        // Annualize (assuming hourly data, ~8760 hours per year)
        let n_periods = returns.len().max(1);
        let periods_per_year = 8760.0;
        let annual_factor = periods_per_year / n_periods as f64;
        let annual_return = (1.0 + total_return).powf(annual_factor) - 1.0;

        // Sharpe ratio
        let mean_return: f64 = if returns.is_empty() {
            0.0
        } else {
            returns.iter().sum::<f64>() / returns.len() as f64
        };

        let std_return: f64 = if returns.len() < 2 {
            0.001
        } else {
            let variance: f64 = returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / (returns.len() - 1) as f64;
            variance.sqrt().max(0.001)
        };

        let sharpe_ratio = mean_return / std_return * periods_per_year.sqrt();

        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = returns
            .iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();

        let downside_std: f64 = if downside_returns.len() < 2 {
            std_return
        } else {
            let variance: f64 = downside_returns
                .iter()
                .map(|r| r.powi(2))
                .sum::<f64>()
                / downside_returns.len() as f64;
            variance.sqrt().max(0.001)
        };

        let sortino_ratio = mean_return / downside_std * periods_per_year.sqrt();

        // Maximum drawdown
        let mut max_equity = initial;
        let mut max_drawdown = 0.0;

        for &eq in &equity_curve {
            max_equity = max_equity.max(eq);
            let drawdown = (max_equity - eq) / max_equity;
            max_drawdown = max_drawdown.max(drawdown);
        }

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 0.001 {
            annual_return / max_drawdown
        } else {
            0.0
        };

        // Win rate and profit factor
        let winning_trades: Vec<f64> = trades.iter().filter(|&&t| t > 0.0).copied().collect();
        let losing_trades: Vec<f64> = trades.iter().filter(|&&t| t < 0.0).copied().collect();

        let win_rate = if !trades.is_empty() {
            winning_trades.len() as f64 / trades.len() as f64
        } else {
            0.0
        };

        let gross_profit: f64 = winning_trades.iter().sum();
        let gross_loss: f64 = losing_trades.iter().map(|t| t.abs()).sum();

        let profit_factor = if gross_loss > 0.001 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        BacktestResult {
            total_return,
            annual_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            win_rate,
            profit_factor,
            total_trades,
            final_equity,
            equity_curve,
        }
    }

    /// Return empty result for insufficient data.
    fn empty_result(&self) -> BacktestResult {
        BacktestResult {
            total_return: 0.0,
            annual_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            calmar_ratio: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            total_trades: 0,
            final_equity: self.config.initial_capital,
            equity_curve: vec![self.config.initial_capital],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::generate_synthetic_data;

    #[test]
    fn test_backtest_engine() {
        let mut engine = BacktestEngine::with_defaults();
        let data = generate_synthetic_data("BTCUSDT", "60", 1000);
        let result = engine.run(&data);

        assert!(result.equity_curve.len() > 0);
        assert!(result.final_equity > 0.0);
        println!("{}", result.summary());
    }

    #[test]
    fn test_metrics_calculation() {
        let mut engine = BacktestEngine::with_defaults();
        let data = generate_synthetic_data("ETHUSDT", "60", 500);
        let result = engine.run(&data);

        assert!(result.max_drawdown >= 0.0);
        assert!(result.win_rate >= 0.0 && result.win_rate <= 1.0);
    }
}
