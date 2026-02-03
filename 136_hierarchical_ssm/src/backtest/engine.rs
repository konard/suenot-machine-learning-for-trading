//! Backtesting engine for evaluating HiSS trading strategies.

use crate::data::bybit::Candle;
use crate::data::features::{compute_features, normalize_features};
use crate::model::{HiSSConfig, HierarchicalSSM};
use crate::trading::strategy::{StrategyConfig, TradingStrategy};

/// Results of a backtest run.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub num_trades: usize,
    pub equity_curve: Vec<f64>,
}

/// Engine for running backtests on historical data.
pub struct BacktestEngine {
    model: HierarchicalSSM,
    strategy_config: StrategyConfig,
    window_size: usize,
    initial_capital: f64,
}

impl BacktestEngine {
    pub fn new(
        model_config: HiSSConfig,
        strategy_config: StrategyConfig,
        window_size: usize,
        initial_capital: f64,
    ) -> Self {
        Self {
            model: HierarchicalSSM::new(model_config),
            strategy_config,
            window_size,
            initial_capital,
        }
    }

    /// Run backtest on historical candle data.
    pub fn run(&self, candles: &[Candle]) -> BacktestResult {
        let features = compute_features(candles);
        let normalized = normalize_features(&features);

        if normalized.len() < self.window_size + 1 {
            return BacktestResult {
                total_return: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown: 0.0,
                win_rate: 0.0,
                profit_factor: 0.0,
                num_trades: 0,
                equity_curve: vec![self.initial_capital],
            };
        }

        let mut strategy =
            TradingStrategy::new(self.strategy_config.clone(), self.initial_capital);
        let mut equity_curve = vec![self.initial_capital];
        let mut returns: Vec<f64> = Vec::new();
        let mut peak = self.initial_capital;
        let mut max_drawdown: f64 = 0.0;
        let mut wins = 0;
        let mut losses = 0;
        let mut gross_profit = 0.0;
        let mut gross_loss = 0.0;

        // We need prices aligned with features; features start at index ~26 of candles
        let price_offset = candles.len() - features.len();

        for i in self.window_size..normalized.len() {
            let window = &normalized[i - self.window_size..i];
            let prediction = self.model.predict(window);

            let candle_idx = price_offset + i;
            if candle_idx >= candles.len() {
                break;
            }
            let price = candles[candle_idx].close;

            let pnl = strategy.step(&prediction, price);

            if pnl != 0.0 {
                returns.push(pnl / self.initial_capital);
                if pnl > 0.0 {
                    wins += 1;
                    gross_profit += pnl;
                } else {
                    losses += 1;
                    gross_loss += pnl.abs();
                }
            }

            let val = strategy.portfolio_value();
            equity_curve.push(val);
            peak = peak.max(val);
            let dd = (peak - val) / peak;
            max_drawdown = max_drawdown.max(dd);
        }

        let num_trades = wins + losses;
        let win_rate = if num_trades > 0 {
            wins as f64 / num_trades as f64
        } else {
            0.0
        };
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let total_return =
            (equity_curve.last().unwrap_or(&self.initial_capital) - self.initial_capital)
                / self.initial_capital;

        let sharpe = compute_sharpe(&returns);
        let sortino = compute_sortino(&returns);

        BacktestResult {
            total_return,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown,
            win_rate,
            profit_factor,
            num_trades,
            equity_curve,
        }
    }
}

fn compute_sharpe(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let var: f64 =
        returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std = var.sqrt();
    if std > 0.0 {
        mean / std * (252.0_f64).sqrt()
    } else {
        0.0
    }
}

fn compute_sortino(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
    if downside.is_empty() {
        return if mean > 0.0 { f64::INFINITY } else { 0.0 };
    }
    let downside_var: f64 =
        downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64;
    let downside_std = downside_var.sqrt();
    if downside_std > 0.0 {
        mean / downside_std * (252.0_f64).sqrt()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::generate_synthetic_candles;

    #[test]
    fn test_backtest_runs() {
        let model_config = HiSSConfig {
            input_dim: 8,
            hidden_dim: 16,
            num_levels: 2,
            downsample_factors: vec![4],
            ssm_state_dim: 4,
            ..Default::default()
        };
        let strategy_config = StrategyConfig::default();
        let engine = BacktestEngine::new(model_config, strategy_config, 32, 100000.0);

        let candles = generate_synthetic_candles(200, 50000.0);
        let result = engine.run(&candles);

        assert!(!result.equity_curve.is_empty());
        assert!(result.max_drawdown >= 0.0 && result.max_drawdown <= 1.0);
    }
}
