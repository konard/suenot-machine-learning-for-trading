//! Backtesting engine for Meta-RL trading strategies.

use crate::agent::meta_rl::MetaRLAgent;
use crate::data::bybit::Kline;
use crate::data::features::FeatureGenerator;
use crate::env::trading_env::TradingEnvironment;

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub transaction_cost: f64,
    pub adaptation_episodes: usize,
    pub episode_length: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            transaction_cost: 0.001,
            adaptation_episodes: 2,
            episode_length: 200,
        }
    }
}

/// Backtest results
#[derive(Debug, Clone)]
pub struct BacktestResults {
    pub initial_capital: f64,
    pub final_capital: f64,
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub num_steps: usize,
    pub equity_curve: Vec<f64>,
}

impl BacktestResults {
    pub fn summary(&self) -> String {
        format!(
            "=== Meta-RL Backtest Results ===\n\
             Capital: ${:.2} -> ${:.2}\n\
             Total Return: {:.2}%\n\
             Sharpe Ratio: {:.3}\n\
             Sortino Ratio: {:.3}\n\
             Max Drawdown: {:.2}%\n\
             Steps: {}",
            self.initial_capital, self.final_capital,
            self.total_return * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.num_steps
        )
    }
}

/// Backtesting engine
pub struct BacktestEngine {
    config: BacktestConfig,
    feature_generator: FeatureGenerator,
}

impl BacktestEngine {
    /// Create a new backtest engine
    pub fn new(config: BacktestConfig) -> Self {
        Self {
            config,
            feature_generator: FeatureGenerator::default_window(),
        }
    }

    /// Run backtest on historical data
    pub fn run(
        &self,
        agent: &mut MetaRLAgent,
        klines: &[Kline],
    ) -> BacktestResults {
        let features = self.feature_generator.compute_features(klines);
        if features.is_empty() {
            return self.empty_results();
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let offset = closes.len() - features.len();
        let prices: Vec<f64> = closes[offset..].to_vec();

        let mut env = TradingEnvironment::new(
            prices,
            features,
            self.config.initial_capital,
            self.config.transaction_cost,
            self.config.episode_length,
        );

        // Adaptation phase
        agent.reset_hidden();
        let mut prev_action = 0;
        let mut prev_reward = 0.0;
        let mut prev_done = false;

        for _ep in 0..self.config.adaptation_episodes {
            let mut state = env.reset();
            let mut done = false;

            while !done {
                let (action, _value) = agent.get_action(
                    &state, prev_action, prev_reward, prev_done,
                );
                let result = env.step(action);
                prev_action = action;
                prev_reward = result.reward;
                prev_done = result.done;
                state = result.state;
                done = result.done;
            }
        }

        // Evaluation phase
        let mut state = env.reset();
        let mut done = false;
        let mut step_returns = Vec::new();
        let mut equity_curve = vec![self.config.initial_capital];

        while !done {
            let (action, _value) = agent.get_action(
                &state, prev_action, prev_reward, prev_done,
            );
            let result = env.step(action);
            step_returns.push(result.reward);
            equity_curve.push(result.capital);

            prev_action = action;
            prev_reward = result.reward;
            prev_done = result.done;
            state = result.state;
            done = result.done;
        }

        let final_capital = *equity_curve.last().unwrap_or(&self.config.initial_capital);
        let total_return = (final_capital / self.config.initial_capital) - 1.0;

        // Sharpe ratio
        let mean_ret = if step_returns.is_empty() { 0.0 }
            else { step_returns.iter().sum::<f64>() / step_returns.len() as f64 };
        let std_ret = if step_returns.len() < 2 { 1.0 }
            else {
                (step_returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
                    / step_returns.len() as f64).sqrt()
            };
        let sharpe = if std_ret > 1e-10 {
            (8760.0_f64).sqrt() * mean_ret / std_ret
        } else { 0.0 };

        // Sortino ratio
        let downside: Vec<f64> = step_returns.iter().filter(|r| **r < 0.0).copied().collect();
        let downside_std = if downside.len() < 2 { 1.0 }
            else {
                let dm = downside.iter().sum::<f64>() / downside.len() as f64;
                (downside.iter().map(|r| (r - dm).powi(2)).sum::<f64>()
                    / downside.len() as f64).sqrt()
            };
        let sortino = if downside_std > 1e-10 {
            (8760.0_f64).sqrt() * mean_ret / downside_std
        } else { 0.0 };

        // Max drawdown
        let max_drawdown = self.compute_max_drawdown(&equity_curve);

        BacktestResults {
            initial_capital: self.config.initial_capital,
            final_capital,
            total_return,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown,
            num_steps: step_returns.len(),
            equity_curve,
        }
    }

    fn compute_max_drawdown(&self, equity: &[f64]) -> f64 {
        let mut peak = equity[0];
        let mut max_dd = 0.0;
        for &e in equity {
            if e > peak { peak = e; }
            let dd = (peak - e) / peak;
            if dd > max_dd { max_dd = dd; }
        }
        max_dd
    }

    fn empty_results(&self) -> BacktestResults {
        BacktestResults {
            initial_capital: self.config.initial_capital,
            final_capital: self.config.initial_capital,
            total_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            num_steps: 0,
            equity_curve: vec![self.config.initial_capital],
        }
    }
}
