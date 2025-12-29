//! Multi-agent trading environment

use std::collections::HashMap;

use rand::Rng;
use rand_distr::{Distribution, Normal};

use super::{EpisodeResult, MultiAgentConfig, Observation, StepInfo, StepResult};
use crate::agents::{Agent, AgentAction, AgentId, AgentState};
use crate::orderbook::{Order, OrderBook, OrderSide};

/// Multi-agent trading environment
pub struct MultiAgentEnv {
    config: MultiAgentConfig,
    order_book: OrderBook,
    agent_states: HashMap<AgentId, AgentState>,
    current_step: usize,
    price_history: Vec<f64>,
    return_history: Vec<f64>,
    historical_prices: Option<Vec<f64>>,
}

impl MultiAgentEnv {
    /// Create a new environment
    pub fn new(config: MultiAgentConfig) -> Self {
        let order_book = OrderBook::new(config.initial_price);

        let mut agent_states = HashMap::new();
        for i in 0..config.n_agents {
            agent_states.insert(i, AgentState::new(config.initial_cash));
        }

        Self {
            config,
            order_book,
            agent_states,
            current_step: 0,
            price_history: Vec::new(),
            return_history: Vec::new(),
            historical_prices: None,
        }
    }

    /// Set historical prices for replay
    pub fn set_historical_prices(&mut self, prices: Vec<f64>) {
        self.historical_prices = Some(prices);
    }

    /// Reset the environment
    pub fn reset(&mut self) -> HashMap<AgentId, Observation> {
        self.order_book.reset(self.config.initial_price);
        self.current_step = 0;
        self.price_history.clear();
        self.return_history.clear();

        for (_, state) in self.agent_states.iter_mut() {
            state.reset(self.config.initial_cash);
        }

        // Initial observations
        self.get_observations()
    }

    /// Get current observations for all agents
    fn get_observations(&self) -> HashMap<AgentId, Observation> {
        let mid_price = self.order_book.mid_price();
        let spread = self.order_book.spread();
        let bid_depth = self.order_book.bid_depth();
        let ask_depth = self.order_book.ask_depth();

        let volume = if self.price_history.len() > 1 {
            self.order_book.total_volume()
        } else {
            0.0
        };

        let volatility = self.calculate_volatility(20);

        let mut observations = HashMap::new();
        for agent_id in 0..self.config.n_agents {
            observations.insert(
                agent_id,
                Observation::new(
                    mid_price,
                    spread,
                    bid_depth,
                    ask_depth,
                    volume,
                    volatility,
                    self.current_step,
                ),
            );
        }

        observations
    }

    /// Calculate recent volatility
    fn calculate_volatility(&self, period: usize) -> f64 {
        if self.return_history.len() < period {
            return self.config.volatility;
        }

        let recent: Vec<f64> = self.return_history
            .iter()
            .rev()
            .take(period)
            .copied()
            .collect();

        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance = recent.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / recent.len() as f64;

        variance.sqrt()
    }

    /// Execute one step of the environment
    pub fn step(&mut self, actions: &HashMap<AgentId, AgentAction>) -> StepResult {
        let prev_price = self.order_book.mid_price();

        // Convert agent actions to orders
        let orders = self.actions_to_orders(actions);

        // Submit orders to order book
        let match_result = self.order_book.submit_orders(orders);

        // Process trades and update agent states
        for trade in &match_result.trades {
            // Update buyer
            if let Some(buyer_state) = self.agent_states.get_mut(&trade.buyer_id) {
                buyer_state.record_buy(trade.price, trade.quantity);
            }

            // Update seller
            if let Some(seller_state) = self.agent_states.get_mut(&trade.seller_id) {
                seller_state.record_sell(trade.price, trade.quantity);
            }
        }

        // Apply price dynamics (if not using historical)
        if self.historical_prices.is_none() {
            self.apply_price_dynamics(&match_result);
        } else {
            self.apply_historical_price();
        }

        let current_price = self.order_book.mid_price();

        // Record price history
        self.price_history.push(current_price);
        if prev_price > 0.0 {
            self.return_history.push((current_price - prev_price) / prev_price);
        }

        // Update agent price observations
        for (_, state) in self.agent_states.iter_mut() {
            state.observe_price(current_price);
            state.update_unrealized_pnl(current_price);
        }

        self.current_step += 1;

        // Calculate rewards
        let rewards = self.calculate_rewards(prev_price, current_price);

        // Check if done
        let done = self.current_step >= self.config.max_steps
            || self.check_early_termination();

        // Build result
        let observations = self.get_observations();

        let info = StepInfo {
            price: current_price,
            n_trades: match_result.trade_count(),
            volume: match_result.volume,
            spread: self.order_book.spread(),
            step: self.current_step,
        };

        let mut result = StepResult::new(observations, rewards, done);
        result.info = info;

        result
    }

    /// Convert agent actions to orders
    fn actions_to_orders(&self, actions: &HashMap<AgentId, AgentAction>) -> Vec<Order> {
        let mut orders = Vec::new();

        for (&agent_id, action) in actions {
            match action {
                AgentAction::Hold => {}
                AgentAction::MarketBuy { quantity } => {
                    orders.push(Order::market_buy(agent_id, *quantity));
                }
                AgentAction::MarketSell { quantity } => {
                    orders.push(Order::market_sell(agent_id, *quantity));
                }
                AgentAction::LimitBuy { price, quantity } => {
                    orders.push(Order::limit_buy(agent_id, *price, *quantity));
                }
                AgentAction::LimitSell { price, quantity } => {
                    orders.push(Order::limit_sell(agent_id, *price, *quantity));
                }
                AgentAction::Quote {
                    bid_price,
                    ask_price,
                    quantity,
                } => {
                    orders.push(Order::limit_buy(agent_id, *bid_price, *quantity));
                    orders.push(Order::limit_sell(agent_id, *ask_price, *quantity));
                }
                AgentAction::CancelAll => {
                    self.order_book.cancel_all_for_agent(agent_id);
                }
            }
        }

        orders
    }

    /// Apply price dynamics based on order flow
    fn apply_price_dynamics(&mut self, _match_result: &crate::orderbook::MatchResult) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, self.config.volatility).unwrap();

        let current_price = self.order_book.mid_price();
        let fundamental = self.config.fundamental_value;

        // Random walk + mean reversion
        let random_shock = normal.sample(&mut rng);
        let mean_reversion = self.config.mean_reversion * (fundamental - current_price) / fundamental;

        // Order book imbalance effect
        let imbalance_effect = self.order_book.imbalance() * self.config.price_impact;

        let price_change = current_price * (random_shock + mean_reversion + imbalance_effect);
        let new_price = (current_price + price_change).max(0.01);

        // Update order book with new fundamental value
        self.order_book.set_fundamental_value(new_price);
    }

    /// Apply historical price
    fn apply_historical_price(&mut self) {
        if let Some(ref prices) = self.historical_prices {
            if self.current_step < prices.len() {
                let new_price = prices[self.current_step];
                self.order_book.set_fundamental_value(new_price);
            }
        }
    }

    /// Calculate rewards for all agents
    fn calculate_rewards(&self, prev_price: f64, current_price: f64) -> HashMap<AgentId, f64> {
        let mut rewards = HashMap::new();

        for (&agent_id, state) in &self.agent_states {
            let reward = match self.config.reward_type {
                super::config::RewardType::PnL => {
                    // Simple PnL change
                    let prev_value = state.cash + state.inventory * prev_price;
                    let curr_value = state.cash + state.inventory * current_price;
                    curr_value - prev_value
                }
                super::config::RewardType::LogReturns => {
                    let prev_value = state.cash + state.inventory * prev_price;
                    let curr_value = state.cash + state.inventory * current_price;
                    if prev_value > 0.0 && curr_value > 0.0 {
                        (curr_value / prev_value).ln()
                    } else {
                        0.0
                    }
                }
                super::config::RewardType::SharpeAdjusted => {
                    let pnl = {
                        let prev_value = state.cash + state.inventory * prev_price;
                        let curr_value = state.cash + state.inventory * current_price;
                        curr_value - prev_value
                    };
                    let vol = self.calculate_volatility(20);
                    if vol > 0.0 {
                        pnl / vol
                    } else {
                        pnl
                    }
                }
                _ => {
                    // Default to PnL
                    let prev_value = state.cash + state.inventory * prev_price;
                    let curr_value = state.cash + state.inventory * current_price;
                    curr_value - prev_value
                }
            };

            rewards.insert(agent_id, reward);
        }

        // Apply zero-sum or ranking if needed
        if self.config.reward_type == super::config::RewardType::ZeroSum {
            let mean_reward: f64 = rewards.values().sum::<f64>() / rewards.len() as f64;
            for reward in rewards.values_mut() {
                *reward -= mean_reward;
            }
        }

        rewards
    }

    /// Check for early termination (e.g., bankruptcy)
    fn check_early_termination(&self) -> bool {
        for state in self.agent_states.values() {
            let value = state.portfolio_value(self.order_book.mid_price());
            if value < 0.0 {
                return true;
            }
        }
        false
    }

    /// Run a complete episode with given agents
    pub fn run_episode(
        &mut self,
        agents: &mut [Box<dyn Agent>],
        max_steps: Option<usize>,
    ) -> EpisodeResult {
        let max_steps = max_steps.unwrap_or(self.config.max_steps);

        // Reset environment
        let mut observations = self.reset();

        // Set agent IDs
        for (i, agent) in agents.iter_mut().enumerate() {
            agent.set_id(i);
            agent.reset();
        }

        let mut total_rewards: HashMap<AgentId, f64> = HashMap::new();
        for i in 0..agents.len() {
            total_rewards.insert(i, 0.0);
        }

        // Run episode
        for _ in 0..max_steps {
            // Get actions from all agents
            let mut actions = HashMap::new();
            for (i, agent) in agents.iter_mut().enumerate() {
                let obs = observations.get(&i).unwrap();
                let state = self.agent_states.get(&i).unwrap();
                let action = agent.act(obs, state);
                actions.insert(i, action);
            }

            // Step environment
            let result = self.step(&actions);

            // Update agents (learning)
            for (i, agent) in agents.iter_mut().enumerate() {
                let reward = result.reward(i);
                let obs = result.observations.get(&i).unwrap();
                agent.learn(reward, obs, result.done);

                *total_rewards.get_mut(&i).unwrap() += reward;
            }

            observations = result.observations;

            if result.done {
                break;
            }
        }

        // Build episode result
        let current_price = self.order_book.mid_price();
        let mut episode_result = EpisodeResult::new();

        for (&agent_id, state) in &self.agent_states {
            let final_value = state.portfolio_value(current_price);
            episode_result.final_values.insert(agent_id, final_value);
            episode_result.total_pnl.insert(
                agent_id,
                final_value - self.config.initial_cash,
            );
            episode_result.trade_counts.insert(agent_id, state.trade_count);
        }

        episode_result.total_rewards = total_rewards;
        episode_result.price_history = self.price_history.clone();
        episode_result.n_steps = self.current_step;

        episode_result
    }

    /// Get current price
    pub fn current_price(&self) -> f64 {
        self.order_book.mid_price()
    }

    /// Get agent state
    pub fn agent_state(&self, agent_id: AgentId) -> Option<&AgentState> {
        self.agent_states.get(&agent_id)
    }

    /// Get order book reference
    pub fn order_book(&self) -> &OrderBook {
        &self.order_book
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::{NoiseTrader, TrendFollowingAgent};

    #[test]
    fn test_environment_creation() {
        let config = MultiAgentConfig::default();
        let env = MultiAgentEnv::new(config.clone());

        assert_eq!(env.current_step, 0);
        assert_eq!(env.agent_states.len(), config.n_agents);
    }

    #[test]
    fn test_environment_reset() {
        let config = MultiAgentConfig::default();
        let mut env = MultiAgentEnv::new(config.clone());

        let obs = env.reset();

        assert_eq!(obs.len(), config.n_agents);
        assert_eq!(env.current_step, 0);
    }

    #[test]
    fn test_environment_step() {
        let config = MultiAgentConfig::default().with_agents(2);
        let mut env = MultiAgentEnv::new(config);

        env.reset();

        let mut actions = HashMap::new();
        actions.insert(0, AgentAction::Hold);
        actions.insert(1, AgentAction::Hold);

        let result = env.step(&actions);

        assert_eq!(result.observations.len(), 2);
        assert!(!result.done);
    }

    #[test]
    fn test_run_episode() {
        let config = MultiAgentConfig::default()
            .with_agents(2);

        let mut env = MultiAgentEnv::new(config);

        let mut agents: Vec<Box<dyn Agent>> = vec![
            Box::new(NoiseTrader::default()),
            Box::new(TrendFollowingAgent::default()),
        ];

        let result = env.run_episode(&mut agents, Some(100));

        assert_eq!(result.final_values.len(), 2);
        assert!(result.n_steps <= 100);

        println!("Episode summary: {}", result.summary());
    }
}
