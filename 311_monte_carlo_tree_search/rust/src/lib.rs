use anyhow::Result;
use rand::Rng;
use serde::Deserialize;
use std::fmt;

// =============================================================================
// Trading State and Actions
// =============================================================================

/// Possible trading actions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingAction {
    Buy,
    Sell,
    Hold,
}

impl TradingAction {
    /// Returns all possible actions
    pub fn all() -> Vec<TradingAction> {
        vec![TradingAction::Buy, TradingAction::Sell, TradingAction::Hold]
    }
}

impl fmt::Display for TradingAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TradingAction::Buy => write!(f, "Buy"),
            TradingAction::Sell => write!(f, "Sell"),
            TradingAction::Hold => write!(f, "Hold"),
        }
    }
}

/// Represents the current trading state
#[derive(Debug, Clone)]
pub struct TradingState {
    /// Historical price data
    pub prices: Vec<f64>,
    /// Current position: positive = long, negative = short, 0 = flat
    pub position: f64,
    /// Available cash balance
    pub balance: f64,
    /// Current step in the price series
    pub step: usize,
    /// Maximum number of steps to simulate
    pub max_steps: usize,
    /// Trade size as fraction of balance
    pub trade_fraction: f64,
}

impl TradingState {
    /// Create a new trading state from price data
    pub fn new(prices: Vec<f64>, initial_balance: f64) -> Self {
        let max_steps = prices.len().saturating_sub(1);
        TradingState {
            prices,
            position: 0.0,
            balance: initial_balance,
            step: 0,
            max_steps,
            trade_fraction: 0.1,
        }
    }

    /// Check if the state is terminal
    pub fn is_terminal(&self) -> bool {
        self.step >= self.max_steps || self.step >= self.prices.len() - 1
    }

    /// Get the current price
    pub fn current_price(&self) -> f64 {
        self.prices[self.step.min(self.prices.len() - 1)]
    }

    /// Get available actions from the current state
    pub fn available_actions(&self) -> Vec<TradingAction> {
        if self.is_terminal() {
            return vec![];
        }
        let mut actions = vec![TradingAction::Hold];
        if self.balance > 0.0 {
            actions.push(TradingAction::Buy);
        }
        if self.position > 0.0 {
            actions.push(TradingAction::Sell);
        }
        actions
    }

    /// Apply an action and return the new state
    pub fn apply_action(&self, action: TradingAction) -> TradingState {
        let mut new_state = self.clone();
        let price = self.current_price();

        match action {
            TradingAction::Buy => {
                if new_state.balance > 0.0 && price > 0.0 {
                    let spend = new_state.balance * new_state.trade_fraction;
                    let units = spend / price;
                    new_state.position += units;
                    new_state.balance -= spend;
                }
            }
            TradingAction::Sell => {
                if new_state.position > 0.0 {
                    let units = new_state.position * new_state.trade_fraction;
                    let revenue = units * price;
                    new_state.position -= units;
                    new_state.balance += revenue;
                }
            }
            TradingAction::Hold => {}
        }

        new_state.step += 1;
        new_state
    }

    /// Calculate portfolio value (balance + position value)
    pub fn portfolio_value(&self) -> f64 {
        self.balance + self.position * self.current_price()
    }

    /// Calculate reward as normalized PnL
    pub fn reward(&self, initial_value: f64) -> f64 {
        if initial_value == 0.0 {
            return 0.0;
        }
        (self.portfolio_value() - initial_value) / initial_value
    }
}

// =============================================================================
// MCTS Node
// =============================================================================

/// A node in the Monte Carlo search tree
#[derive(Debug)]
pub struct MCTSNode {
    /// The trading state at this node
    pub state: TradingState,
    /// The action that led to this node (None for root)
    pub action: Option<TradingAction>,
    /// Parent node index in the arena
    pub parent: Option<usize>,
    /// Child node indices in the arena
    pub children: Vec<usize>,
    /// Number of times this node has been visited
    pub visits: u32,
    /// Total accumulated value from simulations through this node
    pub total_value: f64,
    /// Actions that haven't been tried yet
    pub untried_actions: Vec<TradingAction>,
}

impl MCTSNode {
    /// Create a new MCTS node
    pub fn new(state: TradingState, action: Option<TradingAction>, parent: Option<usize>) -> Self {
        let untried_actions = state.available_actions();
        MCTSNode {
            state,
            action,
            parent,
            children: Vec::new(),
            visits: 0,
            total_value: 0.0,
            untried_actions,
        }
    }

    /// Calculate the average value of this node
    pub fn average_value(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_value / self.visits as f64
        }
    }

    /// Check if this node is fully expanded
    pub fn is_fully_expanded(&self) -> bool {
        self.untried_actions.is_empty()
    }

    /// Check if this node is a leaf
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

// =============================================================================
// MCTS Engine
// =============================================================================

/// Monte Carlo Tree Search engine for trading
pub struct MCTS {
    /// Exploration constant (c in UCT formula)
    pub exploration_constant: f64,
    /// Number of iterations to run
    pub iterations: u32,
    /// Node arena (all nodes stored here, referenced by index)
    pub nodes: Vec<MCTSNode>,
    /// Maximum rollout depth
    pub max_rollout_depth: usize,
}

impl MCTS {
    /// Create a new MCTS instance
    pub fn new(exploration_constant: f64, iterations: u32) -> Self {
        MCTS {
            exploration_constant,
            iterations,
            nodes: Vec::new(),
            max_rollout_depth: 20,
        }
    }

    /// Run MCTS from the given state and return the best action
    pub fn search(&mut self, state: &TradingState) -> Option<TradingAction> {
        self.nodes.clear();
        let root = MCTSNode::new(state.clone(), None, None);
        self.nodes.push(root);

        for _ in 0..self.iterations {
            // Selection
            let selected = self.select(0);

            // Expansion
            let expanded = self.expand(selected);

            // Simulation
            let reward = self.simulate(expanded);

            // Backpropagation
            self.backpropagate(expanded, reward);
        }

        self.best_child_action(0)
    }

    /// Select a leaf node using UCT
    fn select(&self, node_idx: usize) -> usize {
        let mut current = node_idx;
        loop {
            let node = &self.nodes[current];
            if node.state.is_terminal() {
                return current;
            }
            if !node.is_fully_expanded() {
                return current;
            }
            if node.children.is_empty() {
                return current;
            }
            current = self.uct_select_child(current);
        }
    }

    /// Select the best child using the UCT formula
    fn uct_select_child(&self, node_idx: usize) -> usize {
        let parent = &self.nodes[node_idx];
        let parent_visits = parent.visits as f64;
        let ln_parent = parent_visits.ln();

        let mut best_idx = parent.children[0];
        let mut best_score = f64::NEG_INFINITY;

        for &child_idx in &parent.children {
            let child = &self.nodes[child_idx];
            let exploitation = child.average_value();
            let exploration = self.exploration_constant
                * (ln_parent / child.visits as f64).sqrt();
            let uct_score = exploitation + exploration;

            if uct_score > best_score {
                best_score = uct_score;
                best_idx = child_idx;
            }
        }

        best_idx
    }

    /// Expand a node by adding a child for an untried action
    fn expand(&mut self, node_idx: usize) -> usize {
        if self.nodes[node_idx].state.is_terminal() {
            return node_idx;
        }

        if self.nodes[node_idx].untried_actions.is_empty() {
            return node_idx;
        }

        let mut rng = rand::thread_rng();
        let action_idx = rng.gen_range(0..self.nodes[node_idx].untried_actions.len());
        let action = self.nodes[node_idx].untried_actions.remove(action_idx);

        let new_state = self.nodes[node_idx].state.apply_action(action);
        let child = MCTSNode::new(new_state, Some(action), Some(node_idx));
        let child_idx = self.nodes.len();
        self.nodes.push(child);
        self.nodes[node_idx].children.push(child_idx);

        child_idx
    }

    /// Run a random simulation from the given node
    fn simulate(&self, node_idx: usize) -> f64 {
        let mut state = self.nodes[node_idx].state.clone();
        let initial_value = state.portfolio_value();
        let mut rng = rand::thread_rng();
        let mut depth = 0;

        while !state.is_terminal() && depth < self.max_rollout_depth {
            let actions = state.available_actions();
            if actions.is_empty() {
                break;
            }
            let action = actions[rng.gen_range(0..actions.len())];
            state = state.apply_action(action);
            depth += 1;
        }

        state.reward(initial_value)
    }

    /// Backpropagate the simulation result up the tree
    fn backpropagate(&mut self, node_idx: usize, reward: f64) {
        let mut current = Some(node_idx);
        while let Some(idx) = current {
            self.nodes[idx].visits += 1;
            self.nodes[idx].total_value += reward;
            current = self.nodes[idx].parent;
        }
    }

    /// Get the best action from the root based on visit count
    fn best_child_action(&self, node_idx: usize) -> Option<TradingAction> {
        let node = &self.nodes[node_idx];
        if node.children.is_empty() {
            return None;
        }

        let best_child_idx = node
            .children
            .iter()
            .max_by_key(|&&idx| self.nodes[idx].visits)
            .copied()?;

        self.nodes[best_child_idx].action
    }

    /// Get statistics for all root children (for reporting)
    pub fn root_children_stats(&self) -> Vec<(TradingAction, u32, f64)> {
        if self.nodes.is_empty() {
            return vec![];
        }
        let root = &self.nodes[0];
        root.children
            .iter()
            .filter_map(|&idx| {
                let child = &self.nodes[idx];
                child
                    .action
                    .map(|a| (a, child.visits, child.average_value()))
            })
            .collect()
    }
}

// =============================================================================
// Bybit API Integration
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub symbol: Option<String>,
    pub category: Option<String>,
    pub list: Vec<Vec<String>>,
}

/// Fetch kline data from Bybit API
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: u32,
) -> Result<Vec<f64>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let response: BybitResponse = client
        .get(&url)
        .header("User-Agent", "monte-carlo-tree-search/0.1.0")
        .send()?
        .json()?;

    if response.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", response.ret_msg);
    }

    // Kline format: [timestamp, open, high, low, close, volume, turnover]
    // Extract closing prices, reverse to chronological order
    let mut prices: Vec<f64> = response
        .result
        .list
        .iter()
        .filter_map(|kline| {
            if kline.len() >= 5 {
                kline[4].parse::<f64>().ok()
            } else {
                None
            }
        })
        .collect();

    prices.reverse(); // Bybit returns newest first
    Ok(prices)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_prices() -> Vec<f64> {
        vec![
            100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 110.0,
        ]
    }

    #[test]
    fn test_trading_state_creation() {
        let state = TradingState::new(sample_prices(), 10000.0);
        assert_eq!(state.balance, 10000.0);
        assert_eq!(state.position, 0.0);
        assert_eq!(state.step, 0);
        assert_eq!(state.max_steps, 9);
        assert!(!state.is_terminal());
    }

    #[test]
    fn test_buy_action() {
        let state = TradingState::new(sample_prices(), 10000.0);
        let new_state = state.apply_action(TradingAction::Buy);

        // Should spend 10% of balance (1000) to buy at price 100
        assert!(new_state.balance < state.balance);
        assert!(new_state.position > 0.0);
        assert_eq!(new_state.step, 1);

        let expected_units = 1000.0 / 100.0; // 10 units
        assert!((new_state.position - expected_units).abs() < 1e-10);
        assert!((new_state.balance - 9000.0).abs() < 1e-10);
    }

    #[test]
    fn test_sell_action() {
        let mut state = TradingState::new(sample_prices(), 10000.0);
        state.position = 10.0;
        state.balance = 9000.0;

        let new_state = state.apply_action(TradingAction::Sell);
        // Sell 10% of position (1 unit) at price 100
        assert!(new_state.position < state.position);
        assert!(new_state.balance > state.balance);
        assert!((new_state.position - 9.0).abs() < 1e-10);
        assert!((new_state.balance - 9100.0).abs() < 1e-10);
    }

    #[test]
    fn test_hold_action() {
        let state = TradingState::new(sample_prices(), 10000.0);
        let new_state = state.apply_action(TradingAction::Hold);
        assert_eq!(new_state.balance, state.balance);
        assert_eq!(new_state.position, state.position);
        assert_eq!(new_state.step, state.step + 1);
    }

    #[test]
    fn test_terminal_state() {
        let mut state = TradingState::new(sample_prices(), 10000.0);
        state.step = 9;
        assert!(state.is_terminal());
        assert!(state.available_actions().is_empty());
    }

    #[test]
    fn test_portfolio_value() {
        let mut state = TradingState::new(sample_prices(), 10000.0);
        state.position = 5.0;
        state.balance = 9500.0;
        // Portfolio = 9500 + 5 * 100 = 10000
        assert!((state.portfolio_value() - 10000.0).abs() < 1e-10);
    }

    #[test]
    fn test_mcts_search_returns_action() {
        let state = TradingState::new(sample_prices(), 10000.0);
        let mut mcts = MCTS::new(1.414, 500);
        let action = mcts.search(&state);
        assert!(action.is_some());
    }

    #[test]
    fn test_mcts_node_creation() {
        let state = TradingState::new(sample_prices(), 10000.0);
        let node = MCTSNode::new(state, None, None);
        assert_eq!(node.visits, 0);
        assert_eq!(node.total_value, 0.0);
        assert!(node.children.is_empty());
        assert!(node.parent.is_none());
        assert!(!node.untried_actions.is_empty());
    }

    #[test]
    fn test_mcts_multiple_iterations() {
        let state = TradingState::new(sample_prices(), 10000.0);
        let mut mcts = MCTS::new(1.414, 1000);
        let _action = mcts.search(&state);

        // Root should have been visited many times
        assert!(mcts.nodes[0].visits > 0);

        // Should have expanded children
        assert!(!mcts.nodes[0].children.is_empty());

        // Stats should be available
        let stats = mcts.root_children_stats();
        assert!(!stats.is_empty());

        // Total visits of children should approximately equal root visits - 1
        let total_child_visits: u32 = stats.iter().map(|(_, v, _)| v).sum();
        assert!(total_child_visits > 0);
    }

    #[test]
    fn test_uct_exploration() {
        let state = TradingState::new(sample_prices(), 10000.0);

        // Run with high exploration to encourage visiting all children
        let mut mcts = MCTS::new(10.0, 500);
        let _action = mcts.search(&state);

        let stats = mcts.root_children_stats();
        // With high exploration, all children should be visited
        for (_, visits, _) in &stats {
            assert!(*visits > 0, "All children should be visited with high exploration");
        }
    }
}
