//! Backtesting engine for HiPPO trading strategies.

use crate::trading::strategy::TradingStrategy;

/// Performance metrics for a trading strategy.
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub annual_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub num_trades: usize,
}

impl std::fmt::Display for PerformanceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Performance Metrics:")?;
        writeln!(f, "  Total Return:   {:>8.2}%", self.total_return * 100.0)?;
        writeln!(f, "  Annual Return:  {:>8.2}%", self.annual_return * 100.0)?;
        writeln!(f, "  Sharpe Ratio:   {:>8.2}", self.sharpe_ratio)?;
        writeln!(f, "  Sortino Ratio:  {:>8.2}", self.sortino_ratio)?;
        writeln!(f, "  Max Drawdown:   {:>8.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "  Calmar Ratio:   {:>8.2}", self.calmar_ratio)?;
        writeln!(f, "  Win Rate:       {:>8.2}%", self.win_rate * 100.0)?;
        writeln!(f, "  Profit Factor:  {:>8.2}", self.profit_factor)?;
        writeln!(f, "  Num Trades:     {:>8}", self.num_trades)
    }
}

/// Backtesting engine.
pub struct BacktestEngine {
    /// Annualization factor (252 for daily, 8760 for hourly).
    annual_factor: f64,
}

impl BacktestEngine {
    /// Create a new backtest engine.
    pub fn new(annual_factor: f64) -> Self {
        Self { annual_factor }
    }

    /// Run a backtest with a strategy on price data.
    pub fn run_with_strategy(
        &self,
        prices: &[f64],
        strategy: &TradingStrategy,
    ) -> PerformanceMetrics {
        let positions = strategy.get_positions(prices);
        self.run_with_positions(prices, &positions)
    }

    /// Run a backtest with pre-computed positions.
    pub fn run_with_positions(
        &self,
        prices: &[f64],
        positions: &[f64],
    ) -> PerformanceMetrics {
        if prices.len() < 2 {
            return PerformanceMetrics {
                total_return: 0.0,
                annual_return: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown: 0.0,
                calmar_ratio: 0.0,
                win_rate: 0.0,
                profit_factor: 0.0,
                num_trades: 0,
            };
        }

        // Compute returns
        let returns: Vec<f64> = (1..prices.len())
            .map(|i| (prices[i] - prices[i - 1]) / prices[i - 1])
            .collect();

        // Strategy returns (use lagged positions to avoid lookahead)
        let n = returns.len().min(positions.len() - 1);
        let strategy_returns: Vec<f64> = (0..n)
            .map(|i| returns[i] * positions[i])
            .collect();

        self.compute_metrics(&strategy_returns, positions)
    }

    fn compute_metrics(
        &self,
        strategy_returns: &[f64],
        positions: &[f64],
    ) -> PerformanceMetrics {
        if strategy_returns.is_empty() {
            return PerformanceMetrics {
                total_return: 0.0,
                annual_return: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown: 0.0,
                calmar_ratio: 0.0,
                win_rate: 0.0,
                profit_factor: 0.0,
                num_trades: 0,
            };
        }

        // Cumulative returns
        let mut cumulative = Vec::with_capacity(strategy_returns.len());
        let mut cum = 1.0;
        for &r in strategy_returns {
            cum *= 1.0 + r;
            cumulative.push(cum);
        }

        let total_return = cum - 1.0;
        let n_periods = strategy_returns.len() as f64;
        let annual_return = (1.0 + total_return).powf(self.annual_factor / n_periods) - 1.0;

        // Sharpe ratio
        let mean_ret: f64 = strategy_returns.iter().sum::<f64>() / n_periods;
        let var: f64 = strategy_returns
            .iter()
            .map(|&r| (r - mean_ret).powi(2))
            .sum::<f64>()
            / n_periods;
        let std_ret = var.sqrt();
        let sharpe = if std_ret > 1e-10 {
            mean_ret / std_ret * self.annual_factor.sqrt()
        } else {
            0.0
        };

        // Sortino ratio
        let downside: Vec<f64> = strategy_returns
            .iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();
        let downside_std = if !downside.is_empty() {
            let ds_var: f64 =
                downside.iter().map(|&r| r.powi(2)).sum::<f64>() / downside.len() as f64;
            ds_var.sqrt()
        } else {
            1e-10
        };
        let sortino = if downside_std > 1e-10 {
            mean_ret / downside_std * self.annual_factor.sqrt()
        } else {
            0.0
        };

        // Maximum drawdown
        let mut max_cum = 0.0_f64;
        let mut max_drawdown = 0.0_f64;
        for &c in &cumulative {
            max_cum = max_cum.max(c);
            let dd = (c - max_cum) / max_cum;
            max_drawdown = max_drawdown.min(dd);
        }

        // Calmar ratio
        let calmar = if max_drawdown.abs() > 1e-10 {
            annual_return / max_drawdown.abs()
        } else {
            0.0
        };

        // Win rate and profit factor
        let non_zero: Vec<f64> = strategy_returns
            .iter()
            .filter(|&&r| r.abs() > 1e-15)
            .copied()
            .collect();
        let wins: Vec<f64> = non_zero.iter().filter(|&&r| r > 0.0).copied().collect();
        let losses: Vec<f64> = non_zero.iter().filter(|&&r| r < 0.0).copied().collect();
        let win_rate = if !non_zero.is_empty() {
            wins.len() as f64 / non_zero.len() as f64
        } else {
            0.0
        };
        let gross_profit: f64 = wins.iter().sum();
        let gross_loss: f64 = losses.iter().map(|r| r.abs()).sum::<f64>();
        let profit_factor = if gross_loss > 1e-10 {
            gross_profit / gross_loss
        } else {
            0.0
        };

        // Count trades (position changes)
        let num_trades = positions
            .windows(2)
            .filter(|w| (w[1] - w[0]).abs() > 1e-10)
            .count();

        PerformanceMetrics {
            total_return,
            annual_return,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown,
            calmar_ratio: calmar,
            win_rate,
            profit_factor,
            num_trades,
        }
    }
}

impl Default for BacktestEngine {
    fn default() -> Self {
        Self::new(8760.0) // Hourly data by default
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::generate_synthetic_data;
    use crate::trading::strategy::{StrategyConfig, TradingStrategy};

    #[test]
    fn test_backtest_with_strategy() {
        let config = StrategyConfig {
            n_hippo: 8,
            dt: 0.1,
            trend_threshold: 0.0,
            vol_filter: false,
            max_position: 1.0,
        };
        let strategy = TradingStrategy::new(config);
        let data = generate_synthetic_data("BTCUSDT", 200, 50000.0, 0.02, 42);
        let prices = data.close_prices();

        let engine = BacktestEngine::new(8760.0);
        let metrics = engine.run_with_strategy(&prices, &strategy);

        // Basic sanity checks
        assert!(metrics.max_drawdown <= 0.0);
        assert!(metrics.win_rate >= 0.0 && metrics.win_rate <= 1.0);
    }

    #[test]
    fn test_backtest_buy_and_hold() {
        let data = generate_synthetic_data("BTCUSDT", 100, 50000.0, 0.02, 42);
        let prices = data.close_prices();
        let positions = vec![1.0; prices.len()];

        let engine = BacktestEngine::new(8760.0);
        let metrics = engine.run_with_positions(&prices, &positions);

        assert!(metrics.num_trades == 0);
    }
}
