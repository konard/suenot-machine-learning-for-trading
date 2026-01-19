//! Simulation Engine
//!
//! Orchestrates multi-agent market simulation.

use crate::agents::{Agent, AgentDecision, Action};
use crate::agents::base::MarketState;
use crate::market::{OrderBook, Order, OrderType, Side};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result for a single agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    /// Agent ID
    pub agent_id: String,
    /// Agent type
    pub agent_type: String,
    /// Final cash balance
    pub final_cash: f64,
    /// Final shares
    pub final_shares: i64,
    /// Final portfolio value
    pub final_value: f64,
    /// Number of trades
    pub num_trades: u64,
    /// Return percentage
    pub return_pct: f64,
}

/// Simulation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    /// Price history
    pub price_history: Vec<f64>,
    /// Fundamental value history
    pub fundamental_history: Vec<f64>,
    /// Results per agent
    pub agent_results: HashMap<String, AgentResult>,
    /// Total trades executed
    pub total_trades: u64,
    /// Final price
    pub final_price: f64,
    /// Number of steps completed
    pub steps_completed: u64,
}

/// Simulation Engine
pub struct SimulationEngine {
    /// Order book
    order_book: OrderBook,
    /// Trading agents
    agents: HashMap<String, Box<dyn Agent>>,
    /// Agent order for round-robin execution
    agent_order: Vec<String>,
    /// Fundamental value
    fundamental_value: f64,
    /// Price volatility per step
    volatility: f64,
    /// Price history
    price_history: Vec<f64>,
    /// Fundamental history
    fundamental_history: Vec<f64>,
    /// Trade counter
    trade_count: u64,
    /// Current step
    current_step: u64,
    /// Initial price
    initial_price: f64,
}

impl SimulationEngine {
    /// Create a new simulation engine
    pub fn new(initial_price: f64, fundamental_value: f64, volatility: f64) -> Self {
        Self {
            order_book: OrderBook::new(initial_price),
            agents: HashMap::new(),
            agent_order: Vec::new(),
            fundamental_value,
            volatility,
            price_history: vec![initial_price],
            fundamental_history: vec![fundamental_value],
            trade_count: 0,
            current_step: 0,
            initial_price,
        }
    }

    /// Add an agent to the simulation
    pub fn add_agent(&mut self, agent: Box<dyn Agent>) {
        let id = agent.id().to_string();
        self.agent_order.push(id.clone());
        self.agents.insert(id, agent);
    }

    /// Get current market state for agents
    fn get_market_state(&self) -> MarketState {
        MarketState {
            current_price: self.order_book.last_price,
            best_bid: self.order_book.best_bid(),
            best_ask: self.order_book.best_ask(),
            price_history: self.price_history.clone(),
            fundamental_value: self.fundamental_value,
            step: self.current_step,
        }
    }

    /// Process an agent's decision
    fn process_decision(&mut self, agent_id: &str, decision: &AgentDecision) {
        if matches!(decision.action, Action::Hold) {
            return;
        }

        let side = match decision.action {
            Action::Buy => Side::Buy,
            Action::Sell => Side::Sell,
            Action::Hold => return,
        };

        let price = decision.limit_price.unwrap_or(self.order_book.last_price);

        let order = Order::new(
            agent_id.to_string(),
            side,
            decision.order_type,
            price,
            decision.quantity,
        );

        let result = self.order_book.submit_order(order);

        // Update agent positions for filled orders
        if result.filled_quantity > 0 {
            self.trade_count += 1;

            let agent = self.agents.get_mut(agent_id).unwrap();
            let value = result.average_price * result.filled_quantity as f64;

            match side {
                Side::Buy => {
                    agent.update_position(-value, result.filled_quantity as i64);
                }
                Side::Sell => {
                    agent.update_position(value, -(result.filled_quantity as i64));
                }
            }
        }
    }

    /// Update fundamental value with random walk
    fn update_fundamental(&mut self) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, self.volatility * 0.5).unwrap();
        let change = normal.sample(&mut rng);
        self.fundamental_value *= 1.0 + change;
        self.fundamental_history.push(self.fundamental_value);
    }

    /// Add noise to price (represents external factors)
    fn add_price_noise(&mut self) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, self.volatility).unwrap();
        let noise = normal.sample(&mut rng);

        // Price reverts slightly toward fundamental
        let reversion = (self.fundamental_value - self.order_book.last_price) * 0.01;

        let new_price = self.order_book.last_price * (1.0 + noise) + reversion;
        self.order_book.last_price = new_price.max(1.0); // Prevent negative prices
    }

    /// Run one simulation step
    fn step(&mut self) {
        self.current_step += 1;

        // Update fundamental value
        self.update_fundamental();

        // Get market state
        let state = self.get_market_state();

        // Each agent makes a decision
        let agent_ids: Vec<String> = self.agent_order.clone();
        for agent_id in agent_ids {
            let agent = self.agents.get_mut(&agent_id).unwrap();
            let decision = agent.make_decision(&state);
            drop(agent); // Release borrow

            self.process_decision(&agent_id, &decision);
        }

        // Add price noise and record
        self.add_price_noise();
        self.price_history.push(self.order_book.last_price);

        // Clear old orders periodically
        if self.current_step % 10 == 0 {
            self.order_book.clear();
        }
    }

    /// Run simulation for specified number of steps
    pub fn run(&mut self, num_steps: u64) -> SimulationResult {
        for _ in 0..num_steps {
            self.step();
        }

        self.get_results()
    }

    /// Run simulation with progress callback
    pub fn run_with_progress<F>(&mut self, num_steps: u64, mut callback: F) -> SimulationResult
    where
        F: FnMut(u64, f64),
    {
        for i in 0..num_steps {
            self.step();
            if i % 10 == 0 {
                callback(i, self.order_book.last_price);
            }
        }

        self.get_results()
    }

    /// Get simulation results
    fn get_results(&self) -> SimulationResult {
        let mut agent_results = HashMap::new();

        for (id, agent) in &self.agents {
            let final_value = agent.portfolio_value(self.order_book.last_price);
            let initial_value = self.initial_price * 100.0 + 100000.0; // Rough estimate

            agent_results.insert(id.clone(), AgentResult {
                agent_id: id.clone(),
                agent_type: agent.agent_type().to_string(),
                final_cash: agent.cash(),
                final_shares: agent.shares(),
                final_value,
                num_trades: 0, // Would need to track per agent
                return_pct: (final_value / initial_value - 1.0) * 100.0,
            });
        }

        SimulationResult {
            price_history: self.price_history.clone(),
            fundamental_history: self.fundamental_history.clone(),
            agent_results,
            total_trades: self.trade_count,
            final_price: self.order_book.last_price,
            steps_completed: self.current_step,
        }
    }

    /// Reset simulation
    pub fn reset(&mut self) {
        self.order_book = OrderBook::new(self.initial_price);
        self.price_history = vec![self.initial_price];
        self.fundamental_history = vec![self.fundamental_value];
        self.trade_count = 0;
        self.current_step = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::{ValueInvestor, MomentumTrader, MarketMaker};

    #[test]
    fn test_simulation_runs() {
        let mut engine = SimulationEngine::new(100.0, 100.0, 0.02);

        engine.add_agent(Box::new(ValueInvestor::new(
            "value_1".to_string(),
            100000.0,
            100,
            100.0,
        )));

        engine.add_agent(Box::new(MarketMaker::new(
            "mm_1".to_string(),
            200000.0,
            200,
        )));

        let result = engine.run(100);

        assert_eq!(result.steps_completed, 100);
        assert!(result.price_history.len() > 100);
        assert_eq!(result.agent_results.len(), 2);
    }

    #[test]
    fn test_price_changes() {
        let mut engine = SimulationEngine::new(100.0, 100.0, 0.05);

        engine.add_agent(Box::new(MomentumTrader::new(
            "momentum_1".to_string(),
            100000.0,
            50,
        )));

        let result = engine.run(50);

        // Price should change from initial
        let min_price = result.price_history.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_price = result.price_history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        assert!(max_price > min_price);
    }
}
