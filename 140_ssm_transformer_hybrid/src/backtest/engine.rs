//! Backtesting engine for evaluating trading strategies.
//!
//! Simulates trading with transaction costs, position sizing,
//! and computes performance metrics.

use crate::model::hybrid::SSMTransformerModel;
use crate::data::bybit::Kline;
use crate::trading::strategy::{TradingStrategy, StrategyConfig};
use crate::trading::signals::SignalDirection;

/// Backtest performance metrics.
#[derive(Debug, Clone)]
pub struct BacktestResults {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub calmar_ratio: f64,
    pub n_trades: usize,
    pub equity_curve: Vec<f64>,
}

impl std::fmt::Display for BacktestResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BacktestResults {{ return: {:.2}%, sharpe: {:.3}, sortino: {:.3}, \
             max_dd: {:.2}%, win_rate: {:.1}%, profit_factor: {:.2}, trades: {} }}",
            self.total_return * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.win_rate * 100.0,
            self.profit_factor,
            self.n_trades,
        )
    }
}

/// Backtesting engine.
#[derive(Debug)]
pub struct BacktestEngine {
    pub initial_capital: f64,
    pub transaction_cost: f64,
}

impl BacktestEngine {
    pub fn new(initial_capital: f64, transaction_cost: f64) -> Self {
        Self {
            initial_capital,
            transaction_cost,
        }
    }

    /// Run backtest with the given model on kline data.
    ///
    /// # Arguments
    /// * `model` - The SSM-Transformer model
    /// * `klines` - Historical kline data
    ///
    /// # Returns
    /// BacktestResults with performance metrics
    pub fn run(
        &self,
        model: &SSMTransformerModel,
        klines: &[Kline],
    ) -> crate::Result<BacktestResults> {
        let config = StrategyConfig {
            lookback: 50,
            ..Default::default()
        };
        let strategy = TradingStrategy::new(model.clone(), config);

        let signals = strategy.generate_all_signals(klines);
        if signals.is_empty() {
            return Err(crate::HybridError::BacktestError(
                "No signals generated".to_string(),
            ));
        }

        self.compute_results(&signals)
    }

    /// Run backtest with custom strategy configuration.
    pub fn run_with_config(
        &self,
        model: &SSMTransformerModel,
        klines: &[Kline],
        config: StrategyConfig,
    ) -> crate::Result<BacktestResults> {
        let strategy = TradingStrategy::new(model.clone(), config);
        let signals = strategy.generate_all_signals(klines);

        if signals.is_empty() {
            return Err(crate::HybridError::BacktestError(
                "No signals generated".to_string(),
            ));
        }

        self.compute_results(&signals)
    }

    fn compute_results(
        &self,
        signals: &[(crate::trading::signals::TradingSignal, f64)],
    ) -> crate::Result<BacktestResults> {
        let n = signals.len();
        let mut strategy_returns = Vec::with_capacity(n);
        let mut prev_position = 0.0_f64;

        for (signal, actual_return) in signals {
            let position = match signal.direction {
                SignalDirection::Long => signal.position_size,
                SignalDirection::Short => -signal.position_size,
                SignalDirection::Flat => 0.0,
            };

            let cost = (position - prev_position).abs() * self.transaction_cost;
            let ret = position * actual_return - cost;
            strategy_returns.push(ret);
            prev_position = position;
        }

        // Equity curve
        let mut equity = Vec::with_capacity(n + 1);
        equity.push(self.initial_capital);
        for &ret in &strategy_returns {
            let last = *equity.last().unwrap();
            equity.push(last * (1.0 + ret));
        }

        let final_equity = *equity.last().unwrap();
        let total_return = final_equity / self.initial_capital - 1.0;

        // Sharpe ratio (annualized, hourly data = 8760 hours/year)
        let mean_ret = strategy_returns.iter().sum::<f64>() / n as f64;
        let std_ret = {
            let var: f64 = strategy_returns
                .iter()
                .map(|r| (r - mean_ret).powi(2))
                .sum::<f64>()
                / n as f64;
            var.sqrt()
        };
        let sharpe = if std_ret > 1e-10 {
            mean_ret / std_ret * (8760.0_f64).sqrt()
        } else {
            0.0
        };

        // Sortino ratio
        let downside: Vec<f64> = strategy_returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let sortino = if !downside.is_empty() {
            let downside_std = {
                let mean_d = downside.iter().sum::<f64>() / downside.len() as f64;
                let var: f64 = downside.iter().map(|r| (r - mean_d).powi(2)).sum::<f64>()
                    / downside.len() as f64;
                var.sqrt()
            };
            if downside_std > 1e-10 {
                mean_ret / downside_std * (8760.0_f64).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Max drawdown
        let mut peak = equity[0];
        let mut max_dd = 0.0_f64;
        for &e in &equity {
            if e > peak {
                peak = e;
            }
            let dd = (e - peak) / peak;
            if dd < max_dd {
                max_dd = dd;
            }
        }

        // Win rate and profit factor
        let winning: Vec<f64> = strategy_returns.iter().filter(|&&r| r > 0.0).cloned().collect();
        let losing: Vec<f64> = strategy_returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let active_count = strategy_returns.iter().filter(|&&r| r.abs() > 1e-10).count();

        let win_rate = if active_count > 0 {
            winning.len() as f64 / active_count as f64
        } else {
            0.0
        };

        let gross_profit: f64 = winning.iter().sum();
        let gross_loss: f64 = losing.iter().map(|r| r.abs()).sum();
        let profit_factor = if gross_loss > 1e-10 {
            gross_profit / gross_loss
        } else {
            0.0
        };

        let n_trades = signals
            .windows(2)
            .filter(|w| {
                let d0 = match w[0].0.direction { SignalDirection::Flat => 0, _ => 1 };
                let d1 = match w[1].0.direction { SignalDirection::Flat => 0, _ => 1 };
                d0 != d1
            })
            .count();

        let calmar = if max_dd.abs() > 1e-10 {
            total_return / max_dd.abs()
        } else {
            0.0
        };

        Ok(BacktestResults {
            total_return,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown: max_dd,
            win_rate,
            profit_factor,
            calmar_ratio: calmar,
            n_trades,
            equity_curve: equity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::generate_synthetic_klines;
    use crate::model::hybrid::SSMTransformerModel;

    #[test]
    fn test_backtest_engine() {
        let model = SSMTransformerModel::new(15, 16, 2, 1, 8, 4);
        let klines = generate_synthetic_klines(300);
        let engine = BacktestEngine::new(100_000.0, 0.001);

        let config = StrategyConfig {
            lookback: 20,
            ..Default::default()
        };

        let results = engine.run_with_config(&model, &klines, config);
        assert!(results.is_ok());

        let r = results.unwrap();
        assert!(r.equity_curve.len() > 1);
        println!("{}", r);
    }
}
