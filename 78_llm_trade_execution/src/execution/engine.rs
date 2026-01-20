//! Execution engine for managing order lifecycle.

use crate::data::OrderBook;
use crate::execution::{
    ChildOrder, ExecutionState, ExecutionStateMachine, LlmAdapter, LlmDecision, LlmError,
    OrderId, ParentOrder, ParentOrderStatus, Side, StateTransition,
};
use crate::impact::MarketImpactEstimator;
use crate::strategy::ExecutionStrategy;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Execution-related errors
#[derive(Error, Debug)]
pub enum ExecutionError {
    #[error("Order not found: {0}")]
    OrderNotFound(String),

    #[error("Invalid order: {0}")]
    InvalidOrder(String),

    #[error("Strategy error: {0}")]
    Strategy(String),

    #[error("Market data error: {0}")]
    MarketData(String),

    #[error("Exchange error: {0}")]
    Exchange(String),

    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    #[error("State transition error: {0}")]
    StateTransition(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Timeout: {0}")]
    Timeout(String),
}

/// Execution engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// Minimum slice size (base currency)
    pub min_slice_size: f64,
    /// Maximum slice size (base currency)
    pub max_slice_size: f64,
    /// Minimum time between slices (milliseconds)
    pub min_slice_interval_ms: u64,
    /// Maximum time between slices (milliseconds)
    pub max_slice_interval_ms: u64,
    /// Use LLM for execution decisions
    pub use_llm: bool,
    /// LLM decision interval (milliseconds)
    pub llm_interval_ms: u64,
    /// Maximum market impact allowed (bps)
    pub max_impact_bps: f64,
    /// Maximum participation rate
    pub max_participation_rate: f64,
    /// Enable adaptive slice sizing
    pub adaptive_sizing: bool,
    /// Enable real-time market data
    pub real_time_data: bool,
    /// Order timeout in seconds
    pub order_timeout_secs: u64,
    /// Retry count for failed orders
    pub retry_count: u32,
    /// Enable verbose logging
    pub verbose: bool,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            min_slice_size: 0.001,
            max_slice_size: 1.0,
            min_slice_interval_ms: 1000,
            max_slice_interval_ms: 60000,
            use_llm: false,
            llm_interval_ms: 5000,
            max_impact_bps: 50.0,
            max_participation_rate: 0.25,
            adaptive_sizing: true,
            real_time_data: true,
            order_timeout_secs: 30,
            retry_count: 3,
            verbose: false,
        }
    }
}

impl ExecutionConfig {
    /// Create config for aggressive execution
    pub fn aggressive() -> Self {
        Self {
            max_participation_rate: 0.35,
            max_impact_bps: 100.0,
            min_slice_interval_ms: 500,
            ..Default::default()
        }
    }

    /// Create config for passive execution
    pub fn passive() -> Self {
        Self {
            max_participation_rate: 0.10,
            max_impact_bps: 20.0,
            min_slice_interval_ms: 5000,
            ..Default::default()
        }
    }

    /// Create config for LLM-assisted execution
    pub fn with_llm() -> Self {
        Self {
            use_llm: true,
            adaptive_sizing: true,
            ..Default::default()
        }
    }
}

/// Execution result metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Parent order ID
    pub order_id: OrderId,
    /// Symbol
    pub symbol: String,
    /// Side
    pub side: Side,
    /// Total quantity requested
    pub total_quantity: f64,
    /// Total quantity filled
    pub filled_quantity: f64,
    /// Number of child orders
    pub child_order_count: u32,
    /// Average fill price
    pub average_price: f64,
    /// Arrival price
    pub arrival_price: f64,
    /// Market VWAP during execution
    pub market_vwap: f64,
    /// Implementation shortfall (bps)
    pub implementation_shortfall: f64,
    /// VWAP slippage (bps)
    pub vwap_slippage: f64,
    /// Participation rate achieved
    pub participation_rate: f64,
    /// Execution start time
    pub start_time: DateTime<Utc>,
    /// Execution end time
    pub end_time: DateTime<Utc>,
    /// Execution duration (seconds)
    pub duration_secs: u64,
    /// Final status
    pub status: ParentOrderStatus,
    /// State transition history
    pub state_history: Vec<StateTransition>,
    /// LLM decisions made
    pub llm_decisions: Vec<LlmDecision>,
}

impl ExecutionResult {
    /// Calculate execution cost in quote currency
    pub fn execution_cost(&self) -> f64 {
        self.filled_quantity * self.average_price
    }

    /// Calculate theoretical cost at arrival price
    pub fn theoretical_cost(&self) -> f64 {
        self.filled_quantity * self.arrival_price
    }

    /// Calculate absolute slippage
    pub fn absolute_slippage(&self) -> f64 {
        let sign = match self.side {
            Side::Buy => 1.0,
            Side::Sell => -1.0,
        };
        (self.average_price - self.arrival_price) * self.filled_quantity * sign
    }

    /// Check if execution was successful
    pub fn is_successful(&self) -> bool {
        self.status == ParentOrderStatus::Completed && self.filled_quantity >= self.total_quantity * 0.99
    }
}

/// Execution engine
pub struct ExecutionEngine {
    config: ExecutionConfig,
    state_machine: ExecutionStateMachine,
    parent_orders: HashMap<OrderId, ParentOrder>,
    child_orders: HashMap<OrderId, Vec<ChildOrder>>,
    llm_adapter: Option<LlmAdapter>,
    impact_estimator: Option<MarketImpactEstimator>,
    market_data_cache: HashMap<String, OrderBook>,
    llm_decisions: Vec<LlmDecision>,
}

impl ExecutionEngine {
    /// Create a new execution engine
    pub fn new(config: ExecutionConfig) -> Self {
        Self {
            config,
            state_machine: ExecutionStateMachine::new(),
            parent_orders: HashMap::new(),
            child_orders: HashMap::new(),
            llm_adapter: None,
            impact_estimator: None,
            market_data_cache: HashMap::new(),
            llm_decisions: Vec::new(),
        }
    }

    /// Set the LLM adapter
    pub fn with_llm_adapter(mut self, adapter: LlmAdapter) -> Self {
        self.llm_adapter = Some(adapter);
        self
    }

    /// Set the impact estimator
    pub fn with_impact_estimator(mut self, estimator: MarketImpactEstimator) -> Self {
        self.impact_estimator = Some(estimator);
        self
    }

    /// Validate a parent order
    fn validate_order(&self, order: &ParentOrder) -> Result<(), ExecutionError> {
        if order.total_quantity <= 0.0 {
            return Err(ExecutionError::InvalidOrder("Quantity must be positive".into()));
        }

        if order.time_horizon == 0 {
            return Err(ExecutionError::InvalidOrder("Time horizon must be positive".into()));
        }

        if order.symbol.is_empty() {
            return Err(ExecutionError::InvalidOrder("Symbol is required".into()));
        }

        if order.max_participation <= 0.0 || order.max_participation > 1.0 {
            return Err(ExecutionError::InvalidOrder(
                "Participation rate must be between 0 and 1".into(),
            ));
        }

        Ok(())
    }

    /// Execute a parent order with a given strategy
    pub async fn execute(
        &mut self,
        mut order: ParentOrder,
        strategy: Box<dyn ExecutionStrategy>,
    ) -> Result<ExecutionResult, ExecutionError> {
        // Validate the order
        self.state_machine
            .transition(ExecutionState::Validating, "Order received")
            .map_err(|e| ExecutionError::StateTransition(e))?;

        self.validate_order(&order)?;

        // Schedule execution
        self.state_machine
            .transition(ExecutionState::Scheduling, "Validation passed")
            .map_err(|e| ExecutionError::StateTransition(e))?;

        // Get arrival price (in real implementation, would fetch from exchange)
        let arrival_price = self.get_current_price(&order.symbol).await?;
        order.start(arrival_price);

        // Store the order
        let order_id = order.id.clone();
        self.parent_orders.insert(order_id.clone(), order.clone());
        self.child_orders.insert(order_id.clone(), Vec::new());

        // Start execution
        self.state_machine
            .transition(ExecutionState::Executing, "Schedule ready")
            .map_err(|e| ExecutionError::StateTransition(e))?;

        info!(
            "Starting execution for {} {} {} over {} seconds",
            order.side, order.total_quantity, order.symbol, order.time_horizon
        );

        // Main execution loop
        let mut slice_count = 0u32;
        let mut _total_volume = 0.0f64;
        let max_slices = 1000u32; // Safety limit

        while !self.state_machine.is_terminal() && slice_count < max_slices {
            // Extract needed data from parent order first to avoid borrow issues
            let (remaining_qty, time_expired, symbol, parent_side, _parent_urgency, parent_clone) = {
                let parent = self.parent_orders.get(&order_id).unwrap();
                (
                    parent.remaining_quantity(),
                    parent.is_time_expired(),
                    parent.symbol.clone(),
                    parent.side,
                    parent.urgency,
                    parent.clone(),
                )
            };

            // Check if order is complete
            if remaining_qty <= 0.0 {
                self.state_machine
                    .transition(ExecutionState::Completing, "Order filled")
                    .map_err(|e| ExecutionError::StateTransition(e))?;
                break;
            }

            // Check if time expired
            if time_expired {
                self.state_machine
                    .transition(ExecutionState::Completing, "Time expired")
                    .map_err(|e| ExecutionError::StateTransition(e))?;
                break;
            }

            // Get market data
            let orderbook = self.get_orderbook(&symbol).await?;

            // Get LLM decision if enabled
            let llm_decision = if self.config.use_llm {
                self.get_llm_decision(&parent_clone, &orderbook).await.ok()
            } else {
                None
            };

            if let Some(ref decision) = llm_decision {
                self.llm_decisions.push(decision.clone());

                // Handle LLM decision
                match decision.action {
                    crate::execution::ExecutionAction::Pause => {
                        self.state_machine
                            .transition(ExecutionState::Paused, "LLM recommended pause")
                            .ok();
                        continue;
                    }
                    crate::execution::ExecutionAction::Cancel => {
                        self.state_machine
                            .transition(ExecutionState::Cancelled, "LLM recommended cancel")
                            .map_err(|e| ExecutionError::StateTransition(e))?;
                        break;
                    }
                    crate::execution::ExecutionAction::Wait => {
                        // Wait before next slice
                        tokio::time::sleep(tokio::time::Duration::from_millis(
                            self.config.min_slice_interval_ms,
                        ))
                        .await;
                        continue;
                    }
                    _ => {}
                }
            }

            // Calculate slice parameters
            let slice = strategy.next_slice(&parent_clone, &orderbook)?;

            if slice.quantity > 0.0 {
                // Create child order
                let child = ChildOrder::new(
                    order_id.clone(),
                    symbol.clone(),
                    parent_side,
                    slice.quantity,
                    slice.limit_price,
                )
                .with_slice_index(slice_count);

                // Execute the child order (simulated)
                let fill_price = self.simulate_fill(&child, &orderbook)?;

                // Record the fill
                if let Some(parent_mut) = self.parent_orders.get_mut(&order_id) {
                    parent_mut.record_fill(child.quantity, fill_price);
                }

                // Store child order
                if let Some(children) = self.child_orders.get_mut(&order_id) {
                    let mut filled_child = child;
                    filled_child.record_fill(filled_child.quantity, fill_price);
                    children.push(filled_child);
                }

                _total_volume += slice.quantity;
                slice_count += 1;

                if self.config.verbose {
                    debug!(
                        "Slice {}: {} @ {:.2}",
                        slice_count, slice.quantity, fill_price
                    );
                }
            }

            // Wait before next slice
            let interval = if let Some(ref decision) = llm_decision {
                if decision.action == crate::execution::ExecutionAction::Accelerate {
                    self.config.min_slice_interval_ms
                } else if decision.action == crate::execution::ExecutionAction::Decelerate {
                    self.config.max_slice_interval_ms
                } else {
                    strategy.slice_interval_ms()
                }
            } else {
                strategy.slice_interval_ms()
            };

            tokio::time::sleep(tokio::time::Duration::from_millis(interval)).await;
        }

        // Complete execution
        self.state_machine
            .transition(ExecutionState::Completed, "Execution finished")
            .ok();

        // Build result
        let parent = self.parent_orders.get(&order_id).unwrap();
        let end_time = Utc::now();
        let duration = (end_time - parent.started_at.unwrap()).num_seconds() as u64;

        // Calculate metrics
        let avg_price = parent.average_price.unwrap_or(arrival_price);
        let is_bps = ((avg_price - arrival_price) / arrival_price * 10000.0) * parent.side.sign();

        // Simulated market VWAP (in real implementation, would be calculated from trades)
        let market_vwap = arrival_price * (1.0 + (rand::random::<f64>() - 0.5) * 0.001);
        let vwap_slippage = ((avg_price - market_vwap) / market_vwap * 10000.0) * parent.side.sign();

        let result = ExecutionResult {
            order_id: order_id.clone(),
            symbol: parent.symbol.clone(),
            side: parent.side,
            total_quantity: parent.total_quantity,
            filled_quantity: parent.filled_quantity,
            child_order_count: slice_count,
            average_price: avg_price,
            arrival_price,
            market_vwap,
            implementation_shortfall: is_bps,
            vwap_slippage,
            participation_rate: 0.0, // Would need market volume data
            start_time: parent.started_at.unwrap(),
            end_time,
            duration_secs: duration,
            status: parent.status,
            state_history: self.state_machine.history().to_vec(),
            llm_decisions: self.llm_decisions.clone(),
        };

        info!(
            "Execution complete: filled {}/{} @ {:.2} (IS: {:.2} bps)",
            result.filled_quantity, result.total_quantity, result.average_price, result.implementation_shortfall
        );

        Ok(result)
    }

    /// Get current price for a symbol
    async fn get_current_price(&self, symbol: &str) -> Result<f64, ExecutionError> {
        // In real implementation, would fetch from exchange
        // For now, return a simulated price
        if symbol.contains("BTC") {
            Ok(50000.0 + (rand::random::<f64>() - 0.5) * 100.0)
        } else if symbol.contains("ETH") {
            Ok(3000.0 + (rand::random::<f64>() - 0.5) * 10.0)
        } else {
            Ok(100.0 + (rand::random::<f64>() - 0.5) * 1.0)
        }
    }

    /// Get order book for a symbol
    async fn get_orderbook(&mut self, symbol: &str) -> Result<OrderBook, ExecutionError> {
        // Check cache first
        if let Some(book) = self.market_data_cache.get(symbol) {
            return Ok(book.clone());
        }

        // In real implementation, would fetch from exchange
        // For now, create a simulated order book
        let mid_price = self.get_current_price(symbol).await?;
        let mut book = OrderBook::new(symbol.to_string());

        // Add bid levels
        for i in 0..20 {
            let price = mid_price * (1.0 - 0.0001 * (i + 1) as f64);
            let qty = 1.0 + rand::random::<f64>() * 5.0;
            book.update_bid(price, qty);
        }

        // Add ask levels
        for i in 0..20 {
            let price = mid_price * (1.0 + 0.0001 * (i + 1) as f64);
            let qty = 1.0 + rand::random::<f64>() * 5.0;
            book.update_ask(price, qty);
        }

        self.market_data_cache.insert(symbol.to_string(), book.clone());
        Ok(book)
    }

    /// Get LLM decision for execution
    async fn get_llm_decision(
        &self,
        order: &ParentOrder,
        orderbook: &OrderBook,
    ) -> Result<LlmDecision, ExecutionError> {
        let adapter = self.llm_adapter.as_ref().ok_or_else(|| {
            ExecutionError::Config("LLM adapter not configured".into())
        })?;

        let market_state = crate::execution::MarketState::from_orderbook(orderbook);
        let context = crate::execution::build_execution_context(
            order,
            order.max_participation,
            0.0, // Would calculate actual participation
            order.average_price.unwrap_or(0.0),
            orderbook.mid_price().unwrap_or(0.0),
        );

        // Try LLM first, fall back to heuristic
        match adapter.get_decision(&market_state, &context).await {
            Ok(decision) => Ok(decision),
            Err(e) => {
                warn!("LLM decision failed, using heuristic: {}", e);
                Ok(adapter.get_heuristic_decision(&market_state, &context))
            }
        }
    }

    /// Simulate a fill (for backtesting/demo)
    fn simulate_fill(&self, order: &ChildOrder, orderbook: &OrderBook) -> Result<f64, ExecutionError> {
        let (avg_price, _impact) = match order.side {
            Side::Buy => orderbook.buy_impact(order.quantity),
            Side::Sell => orderbook.sell_impact(order.quantity),
        }
        .ok_or_else(|| ExecutionError::MarketData("Insufficient liquidity".into()))?;

        Ok(avg_price)
    }

    /// Get a parent order by ID
    pub fn get_order(&self, order_id: &OrderId) -> Option<&ParentOrder> {
        self.parent_orders.get(order_id)
    }

    /// Get child orders for a parent order
    pub fn get_child_orders(&self, order_id: &OrderId) -> Option<&Vec<ChildOrder>> {
        self.child_orders.get(order_id)
    }

    /// Get the current execution state
    pub fn state(&self) -> ExecutionState {
        self.state_machine.current_state()
    }

    /// Pause execution
    pub fn pause(&mut self) -> Result<(), ExecutionError> {
        self.state_machine
            .transition(ExecutionState::Paused, "User requested pause")
            .map_err(|e| ExecutionError::StateTransition(e))
    }

    /// Resume execution
    pub fn resume(&mut self) -> Result<(), ExecutionError> {
        self.state_machine
            .transition(ExecutionState::Executing, "User requested resume")
            .map_err(|e| ExecutionError::StateTransition(e))
    }

    /// Cancel execution
    pub fn cancel(&mut self, order_id: &OrderId) -> Result<(), ExecutionError> {
        if let Some(order) = self.parent_orders.get_mut(order_id) {
            order.cancel();
        }
        self.state_machine
            .transition(ExecutionState::Cancelled, "User requested cancel")
            .map_err(|e| ExecutionError::StateTransition(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::TwapStrategy;

    #[tokio::test]
    async fn test_execution_engine_basic() {
        let config = ExecutionConfig {
            min_slice_interval_ms: 10, // Fast for testing
            max_slice_interval_ms: 20,
            ..Default::default()
        };

        let mut engine = ExecutionEngine::new(config);

        let order = ParentOrder::new(
            "BTCUSDT".to_string(),
            Side::Buy,
            0.1,
            5, // 5 seconds
        );

        let strategy = Box::new(TwapStrategy::new(1)); // 1 second slices

        let result = engine.execute(order, strategy).await.unwrap();

        assert!(result.filled_quantity > 0.0);
        assert!(result.child_order_count > 0);
        assert_eq!(result.status, ParentOrderStatus::Completed);
    }

    #[test]
    fn test_execution_config_presets() {
        let aggressive = ExecutionConfig::aggressive();
        assert!(aggressive.max_participation_rate > 0.3);

        let passive = ExecutionConfig::passive();
        assert!(passive.max_participation_rate < 0.15);

        let llm = ExecutionConfig::with_llm();
        assert!(llm.use_llm);
    }

    #[test]
    fn test_validate_order() {
        let engine = ExecutionEngine::new(ExecutionConfig::default());

        // Valid order
        let valid = ParentOrder::new("BTCUSDT".to_string(), Side::Buy, 1.0, 3600);
        assert!(engine.validate_order(&valid).is_ok());

        // Invalid: zero quantity
        let invalid_qty = ParentOrder::new("BTCUSDT".to_string(), Side::Buy, 0.0, 3600);
        assert!(engine.validate_order(&invalid_qty).is_err());

        // Invalid: zero time
        let invalid_time = ParentOrder::new("BTCUSDT".to_string(), Side::Buy, 1.0, 0);
        assert!(engine.validate_order(&invalid_time).is_err());
    }
}
