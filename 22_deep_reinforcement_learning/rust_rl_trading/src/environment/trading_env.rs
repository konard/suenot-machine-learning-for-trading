//! Trading environment implementation.

use crate::data::MarketData;
use crate::environment::{TradingAction, TradingState};
use ndarray::Array1;
use rand::Rng;

/// Trading environment configuration
#[derive(Debug, Clone)]
pub struct EnvConfig {
    /// Number of trading days per episode
    pub episode_length: usize,
    /// Trading cost in basis points (0.001 = 0.1%)
    pub trading_cost_bps: f64,
    /// Time cost for holding position (cost of capital)
    pub time_cost_bps: f64,
    /// Starting capital
    pub initial_capital: f64,
    /// Maximum drawdown before episode ends
    pub max_drawdown: f64,
    /// Reward scaling factor
    pub reward_scale: f64,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            episode_length: 252, // One trading year
            trading_cost_bps: 0.001, // 0.1%
            time_cost_bps: 0.0001, // 0.01%
            initial_capital: 10000.0,
            max_drawdown: 0.3, // 30%
            reward_scale: 100.0,
        }
    }
}

/// Step result returned by the environment
#[derive(Debug)]
pub struct StepResult {
    /// Next state observation
    pub state: TradingState,
    /// Reward for the action
    pub reward: f64,
    /// Whether episode is done
    pub done: bool,
    /// Additional info
    pub info: StepInfo,
}

/// Additional information about the step
#[derive(Debug, Clone)]
pub struct StepInfo {
    /// Current Net Asset Value
    pub nav: f64,
    /// Current position
    pub position: f64,
    /// Trade executed this step
    pub trade: f64,
    /// Trading costs this step
    pub costs: f64,
    /// Market return this step
    pub market_return: f64,
    /// Strategy return this step
    pub strategy_return: f64,
    /// Buy and hold NAV (benchmark)
    pub benchmark_nav: f64,
}

/// Trading environment for reinforcement learning
pub struct TradingEnvironment {
    /// Market data
    data: MarketData,
    /// Configuration
    config: EnvConfig,
    /// Current step in data
    current_step: usize,
    /// Starting step for current episode
    start_step: usize,
    /// Current position (-1, 0, 1)
    position: f64,
    /// Entry price for current position
    entry_price: f64,
    /// Current NAV
    nav: f64,
    /// Peak NAV (for drawdown calculation)
    peak_nav: f64,
    /// Benchmark NAV (buy and hold)
    benchmark_nav: f64,
    /// Time in current position
    time_in_position: usize,
    /// Episode history
    history: Vec<StepInfo>,
}

impl TradingEnvironment {
    /// Create a new trading environment
    pub fn new(data: MarketData, config: EnvConfig) -> Self {
        Self {
            data,
            config,
            current_step: 0,
            start_step: 0,
            position: 0.0,
            entry_price: 0.0,
            nav: 0.0,
            peak_nav: 0.0,
            benchmark_nav: 0.0,
            time_in_position: 0,
            history: Vec::new(),
        }
    }

    /// Create with default config
    pub fn with_default_config(data: MarketData) -> Self {
        Self::new(data, EnvConfig::default())
    }

    /// Get the state size
    pub fn state_size(&self) -> usize {
        TradingState::size(MarketData::state_size())
    }

    /// Get the action size
    pub fn action_size(&self) -> usize {
        TradingAction::count()
    }

    /// Reset the environment for a new episode
    pub fn reset(&mut self) -> TradingState {
        let mut rng = rand::thread_rng();

        // Random starting point (leave room for episode)
        let max_start = self.data.len().saturating_sub(self.config.episode_length + 50);
        self.start_step = if max_start > 50 {
            rng.gen_range(50..max_start)
        } else {
            50
        };
        self.current_step = self.start_step;

        // Reset state
        self.position = 0.0;
        self.entry_price = 0.0;
        self.nav = self.config.initial_capital;
        self.peak_nav = self.nav;
        self.benchmark_nav = self.config.initial_capital;
        self.time_in_position = 0;
        self.history.clear();

        self.get_state()
    }

    /// Take a step in the environment
    pub fn step(&mut self, action: TradingAction) -> StepResult {
        let current_candle = self.data.get_candle(self.current_step).unwrap();
        let current_price = current_candle.close;

        // Get market return
        let market_return = self.data.returns()[self.current_step];

        // Calculate new position
        let new_position = action.position();
        let position_change = new_position - self.position;

        // Calculate trading costs
        let trade_cost = position_change.abs() * self.config.trading_cost_bps * self.nav;
        let time_cost = if position_change == 0.0 {
            self.config.time_cost_bps * self.nav
        } else {
            0.0
        };
        let total_cost = trade_cost + time_cost;

        // Calculate strategy return
        let strategy_return = self.position * market_return;
        let pnl = strategy_return * self.nav - total_cost;

        // Update NAV
        self.nav += pnl;
        self.peak_nav = self.peak_nav.max(self.nav);

        // Update benchmark (buy and hold)
        self.benchmark_nav *= 1.0 + market_return;

        // Update position tracking
        if position_change != 0.0 {
            self.position = new_position;
            self.entry_price = current_price;
            self.time_in_position = 0;
        } else {
            self.time_in_position += 1;
        }

        // Calculate reward
        let reward = self.calculate_reward(pnl, strategy_return, market_return);

        // Record history
        let info = StepInfo {
            nav: self.nav,
            position: self.position,
            trade: position_change,
            costs: total_cost,
            market_return,
            strategy_return,
            benchmark_nav: self.benchmark_nav,
        };
        self.history.push(info.clone());

        // Move to next step
        self.current_step += 1;

        // Check if done
        let steps_taken = self.current_step - self.start_step;
        let drawdown = (self.peak_nav - self.nav) / self.peak_nav;
        let done = steps_taken >= self.config.episode_length
            || self.current_step >= self.data.len() - 1
            || drawdown >= self.config.max_drawdown
            || self.nav <= 0.0;

        StepResult {
            state: self.get_state(),
            reward,
            done,
            info,
        }
    }

    /// Get current state observation
    fn get_state(&self) -> TradingState {
        let market_features = self
            .data
            .get_state(self.current_step)
            .unwrap_or_else(|| Array1::zeros(MarketData::state_size()));

        let unrealized_pnl = if self.position != 0.0 && self.entry_price > 0.0 {
            let current_price = self.data.get_candle(self.current_step).unwrap().close;
            self.position * (current_price - self.entry_price) / self.entry_price
        } else {
            0.0
        };

        let time_normalized = (self.time_in_position as f64 / 20.0).min(1.0);
        let episode_progress =
            (self.current_step - self.start_step) as f64 / self.config.episode_length as f64;

        TradingState::new(
            market_features,
            self.position,
            unrealized_pnl,
            time_normalized,
            episode_progress,
        )
    }

    /// Calculate reward
    fn calculate_reward(&self, pnl: f64, strategy_return: f64, market_return: f64) -> f64 {
        // Primary reward: PnL normalized by initial capital
        let pnl_reward = pnl / self.config.initial_capital;

        // Bonus for beating the market
        let alpha = strategy_return - market_return;

        // Penalty for excessive drawdown
        let drawdown = (self.peak_nav - self.nav) / self.peak_nav;
        let drawdown_penalty = if drawdown > 0.1 {
            -drawdown * 0.5
        } else {
            0.0
        };

        (pnl_reward + alpha * 0.5 + drawdown_penalty) * self.config.reward_scale
    }

    /// Get episode statistics
    pub fn get_episode_stats(&self) -> EpisodeStats {
        if self.history.is_empty() {
            return EpisodeStats::default();
        }

        let total_return = (self.nav - self.config.initial_capital) / self.config.initial_capital;
        let benchmark_return =
            (self.benchmark_nav - self.config.initial_capital) / self.config.initial_capital;

        let returns: Vec<f64> = self.history.iter().map(|h| h.strategy_return).collect();
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
            / returns.len() as f64;
        let volatility = variance.sqrt() * (252.0_f64).sqrt(); // Annualized

        let sharpe = if volatility > 0.0 {
            (mean_return * 252.0) / volatility
        } else {
            0.0
        };

        let max_drawdown = self
            .history
            .iter()
            .scan(self.config.initial_capital, |peak, info| {
                *peak = peak.max(info.nav);
                Some((*peak - info.nav) / *peak)
            })
            .fold(0.0_f64, |max, dd| max.max(dd));

        let trades: usize = self
            .history
            .iter()
            .filter(|h| h.trade.abs() > 0.0)
            .count();
        let total_costs: f64 = self.history.iter().map(|h| h.costs).sum();

        EpisodeStats {
            total_return,
            benchmark_return,
            sharpe_ratio: sharpe,
            max_drawdown,
            volatility,
            num_trades: trades,
            total_costs,
            final_nav: self.nav,
        }
    }
}

/// Episode statistics
#[derive(Debug, Clone, Default)]
pub struct EpisodeStats {
    pub total_return: f64,
    pub benchmark_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub volatility: f64,
    pub num_trades: usize,
    pub total_costs: f64,
    pub final_nav: f64,
}

impl std::fmt::Display for EpisodeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Return: {:.2}% | Benchmark: {:.2}% | Sharpe: {:.2} | MaxDD: {:.2}% | Trades: {} | Costs: ${:.2}",
            self.total_return * 100.0,
            self.benchmark_return * 100.0,
            self.sharpe_ratio,
            self.max_drawdown * 100.0,
            self.num_trades,
            self.total_costs
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Candle;
    use chrono::Utc;

    fn create_test_data(n: usize) -> MarketData {
        let mut candles = Vec::with_capacity(n);
        let mut price = 100.0;

        for i in 0..n {
            let change = (i as f64 * 0.1).sin() * 2.0;
            price = (price + change).max(1.0);
            candles.push(Candle::new(
                Utc::now(),
                "TEST".to_string(),
                price - 0.5,
                price + 1.0,
                price - 1.0,
                price,
                1000.0,
                price * 1000.0,
            ));
        }

        MarketData::from_candles(candles)
    }

    #[test]
    fn test_env_reset() {
        let data = create_test_data(500);
        let mut env = TradingEnvironment::with_default_config(data);

        let state = env.reset();
        assert_eq!(state.position, 0.0);
        assert_eq!(state.unrealized_pnl, 0.0);
    }

    #[test]
    fn test_env_step() {
        let data = create_test_data(500);
        let mut env = TradingEnvironment::with_default_config(data);

        env.reset();

        // Take a long position
        let result = env.step(TradingAction::Long);
        assert!(!result.done);
        assert_eq!(result.info.position, 1.0);

        // Hold
        let result = env.step(TradingAction::Long);
        assert!(!result.done);

        // Go short
        let result = env.step(TradingAction::Short);
        assert!(!result.done);
        assert_eq!(result.info.position, -1.0);
    }

    #[test]
    fn test_episode_completion() {
        let data = create_test_data(500);
        let config = EnvConfig {
            episode_length: 10,
            ..Default::default()
        };
        let mut env = TradingEnvironment::new(data, config);

        env.reset();

        let mut done = false;
        let mut steps = 0;

        while !done {
            let result = env.step(TradingAction::Hold);
            done = result.done;
            steps += 1;

            if steps > 100 {
                panic!("Episode did not end");
            }
        }

        assert!(steps >= 10);
    }
}
