use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use anyhow::Result;

// ============================================================================
// Goal Types
// ============================================================================

/// Types of trading goals the agent can pursue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradingGoal {
    /// Target cumulative return (e.g., 0.05 = 5%)
    TargetReturn(f64),
    /// Maximum allowed drawdown (e.g., -0.10 = -10%)
    MaxDrawdown(f64),
    /// Target annualized Sharpe ratio (e.g., 1.5)
    TargetSharpe(f64),
}

impl TradingGoal {
    /// Convert goal to a numeric vector for network input
    pub fn to_vector(&self) -> Vec<f64> {
        match self {
            TradingGoal::TargetReturn(r) => vec![1.0, *r, 0.0],
            TradingGoal::MaxDrawdown(d) => vec![0.0, *d, 0.0],
            TradingGoal::TargetSharpe(s) => vec![0.0, 0.0, *s],
        }
    }

    /// Extract the primary goal value
    pub fn value(&self) -> f64 {
        match self {
            TradingGoal::TargetReturn(v) => *v,
            TradingGoal::MaxDrawdown(v) => *v,
            TradingGoal::TargetSharpe(v) => *v,
        }
    }
}

// ============================================================================
// Trading Actions
// ============================================================================

/// Discrete trading actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingAction {
    Hold = 0,
    Buy = 1,
    Sell = 2,
}

impl TradingAction {
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => TradingAction::Hold,
            1 => TradingAction::Buy,
            2 => TradingAction::Sell,
            _ => TradingAction::Hold,
        }
    }
}

pub const NUM_ACTIONS: usize = 3;

// ============================================================================
// Goal-Augmented State
// ============================================================================

/// Goal-conditioned state: market observation concatenated with goal vector
#[derive(Debug, Clone)]
pub struct GoalConditionedState {
    /// Raw market features (prices, volumes, indicators)
    pub market_state: Vec<f64>,
    /// Goal specification vector
    pub goal: Vec<f64>,
    /// Current achieved goal metrics
    pub achieved_goal: Vec<f64>,
}

impl GoalConditionedState {
    pub fn new(market_state: Vec<f64>, goal: Vec<f64>, achieved_goal: Vec<f64>) -> Self {
        Self {
            market_state,
            goal,
            achieved_goal,
        }
    }

    /// Concatenate state and goal into a single input vector for the network
    pub fn to_augmented_vector(&self) -> Vec<f64> {
        let mut v = self.market_state.clone();
        v.extend_from_slice(&self.goal);
        v
    }

    /// Dimension of the augmented state
    pub fn dim(&self) -> usize {
        self.market_state.len() + self.goal.len()
    }
}

// ============================================================================
// Experience / Transition
// ============================================================================

/// A single transition in the replay buffer
#[derive(Debug, Clone)]
pub struct Transition {
    pub state: GoalConditionedState,
    pub action: usize,
    pub reward: f64,
    pub next_state: GoalConditionedState,
    pub done: bool,
}

// ============================================================================
// HER Replay Buffer
// ============================================================================

/// Strategy for relabeling goals in Hindsight Experience Replay
#[derive(Debug, Clone, Copy)]
pub enum HERStrategy {
    /// Use the final achieved goal of the episode
    Final,
    /// Sample a future achieved goal from the episode
    Future,
    /// Sample any achieved goal from the episode
    Episode,
}

/// Replay buffer with Hindsight Experience Replay support
pub struct HERReplayBuffer {
    pub transitions: Vec<Transition>,
    pub capacity: usize,
    pub strategy: HERStrategy,
    pub her_ratio: f64,
    /// Temporary episode storage for HER relabeling
    pub current_episode: Vec<Transition>,
}

impl HERReplayBuffer {
    pub fn new(capacity: usize, strategy: HERStrategy, her_ratio: f64) -> Self {
        Self {
            transitions: Vec::with_capacity(capacity),
            capacity,
            strategy,
            her_ratio,
            current_episode: Vec::new(),
        }
    }

    /// Add a transition to the current episode
    pub fn add_transition(&mut self, transition: Transition) {
        self.current_episode.push(transition);
    }

    /// Finalize the episode: store original transitions and HER-relabeled ones
    pub fn end_episode(&mut self) {
        let episode = std::mem::take(&mut self.current_episode);
        let mut rng = rand::thread_rng();

        // Store original transitions
        for t in &episode {
            self.push(t.clone());
        }

        // Generate HER-relabeled transitions
        let num_her = (episode.len() as f64 * self.her_ratio) as usize;
        for i in 0..episode.len().min(num_her) {
            let new_goal = match self.strategy {
                HERStrategy::Final => {
                    // Use the achieved goal from the final state
                    episode.last().unwrap().next_state.achieved_goal.clone()
                }
                HERStrategy::Future => {
                    // Sample a future state's achieved goal
                    if i + 1 < episode.len() {
                        let future_idx = rng.gen_range(i + 1..episode.len());
                        episode[future_idx].next_state.achieved_goal.clone()
                    } else {
                        episode.last().unwrap().next_state.achieved_goal.clone()
                    }
                }
                HERStrategy::Episode => {
                    // Sample any state's achieved goal from the episode
                    let idx = rng.gen_range(0..episode.len());
                    episode[idx].next_state.achieved_goal.clone()
                }
            };

            // Relabel the transition with the new goal
            let original = &episode[i];
            let relabeled_state = GoalConditionedState::new(
                original.state.market_state.clone(),
                new_goal.clone(),
                original.state.achieved_goal.clone(),
            );
            let relabeled_next_state = GoalConditionedState::new(
                original.next_state.market_state.clone(),
                new_goal.clone(),
                original.next_state.achieved_goal.clone(),
            );

            // Recompute reward based on new goal
            let reward = compute_goal_reward(
                &original.next_state.achieved_goal,
                &new_goal,
                0.01,
            );

            let relabeled = Transition {
                state: relabeled_state,
                action: original.action,
                reward,
                next_state: relabeled_next_state,
                done: original.done,
            };
            self.push(relabeled);
        }
    }

    fn push(&mut self, transition: Transition) {
        if self.transitions.len() >= self.capacity {
            self.transitions.remove(0);
        }
        self.transitions.push(transition);
    }

    /// Sample a minibatch of transitions
    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        let mut rng = rand::thread_rng();
        let len = self.transitions.len();
        if len == 0 {
            return Vec::new();
        }
        (0..batch_size)
            .map(|_| &self.transitions[rng.gen_range(0..len)])
            .collect()
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }
}

// ============================================================================
// Reward Computation
// ============================================================================

/// Compute goal-conditioned reward: dense shaping + sparse bonus
pub fn compute_goal_reward(achieved: &[f64], desired: &[f64], tolerance: f64) -> f64 {
    let distance: f64 = achieved
        .iter()
        .zip(desired.iter())
        .map(|(a, d)| (a - d).powi(2))
        .sum::<f64>()
        .sqrt();

    let shaping = -distance;
    let sparse_bonus = if distance < tolerance { 1.0 } else { 0.0 };
    shaping + sparse_bonus
}

// ============================================================================
// Universal Value Function Approximator
// ============================================================================

/// Simple feed-forward network for Q(s, g, a)
/// Uses a two-layer architecture with ReLU activations
pub struct UniversalValueFunction {
    /// Weights: input_dim -> hidden_dim
    pub w1: Array2<f64>,
    /// Bias for first layer
    pub b1: Array1<f64>,
    /// Weights: hidden_dim -> num_actions
    pub w2: Array2<f64>,
    /// Bias for second layer
    pub b2: Array1<f64>,
    /// Learning rate
    pub lr: f64,
    /// Input dimension (state_dim + goal_dim)
    pub input_dim: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
}

impl UniversalValueFunction {
    pub fn new(input_dim: usize, hidden_dim: usize, lr: f64) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / input_dim as f64).sqrt();
        let scale2 = (2.0 / hidden_dim as f64).sqrt();

        let w1 = Array2::from_shape_fn((input_dim, hidden_dim), |_| {
            rng.gen_range(-scale1..scale1)
        });
        let b1 = Array1::zeros(hidden_dim);
        let w2 = Array2::from_shape_fn((hidden_dim, NUM_ACTIONS), |_| {
            rng.gen_range(-scale2..scale2)
        });
        let b2 = Array1::zeros(NUM_ACTIONS);

        Self {
            w1, b1, w2, b2, lr, input_dim, hidden_dim,
        }
    }

    /// Forward pass: returns Q-values for all actions
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let x = Array1::from_vec(input.to_vec());

        // Hidden layer with ReLU
        let h = x.dot(&self.w1) + &self.b1;
        let h_relu = h.mapv(|v| v.max(0.0));

        // Output layer
        let out = h_relu.dot(&self.w2) + &self.b2;
        out.to_vec()
    }

    /// Get Q-value for a specific action
    pub fn q_value(&self, input: &[f64], action: usize) -> f64 {
        self.forward(input)[action]
    }

    /// Get the best action (argmax Q)
    pub fn best_action(&self, input: &[f64]) -> usize {
        let q_values = self.forward(input);
        q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Update weights using a single transition (online SGD)
    pub fn update(
        &mut self,
        input: &[f64],
        action: usize,
        target: f64,
    ) {
        let x = Array1::from_vec(input.to_vec());

        // Forward pass
        let h = x.dot(&self.w1) + &self.b1;
        let h_relu = h.mapv(|v| v.max(0.0));
        let out = h_relu.dot(&self.w2) + &self.b2;

        // Compute error for the selected action
        let prediction = out[action];
        let error = target - prediction;

        // Backprop through output layer
        let mut d_out = Array1::zeros(NUM_ACTIONS);
        d_out[action] = error;

        // Gradient for w2: h_relu^T * d_out
        let h_relu_col = h_relu.clone().insert_axis(ndarray::Axis(1));
        let d_out_row = d_out.clone().insert_axis(ndarray::Axis(0));
        let dw2 = h_relu_col.dot(&d_out_row);

        // Backprop through ReLU
        let d_hidden = d_out.dot(&self.w2.t());
        let d_hidden_relu = &d_hidden * &h.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });

        // Gradient for w1: x^T * d_hidden_relu
        let x_col = x.insert_axis(ndarray::Axis(1));
        let dh_row = d_hidden_relu.clone().insert_axis(ndarray::Axis(0));
        let dw1 = x_col.dot(&dh_row);

        // Apply gradients
        self.w2 = &self.w2 + &(dw2 * self.lr);
        self.b2 = &self.b2 + &(d_out * self.lr);
        self.w1 = &self.w1 + &(dw1 * self.lr);
        self.b1 = &self.b1 + &(d_hidden_relu * self.lr);
    }

    /// Copy weights from another UVF (for target network)
    pub fn copy_from(&mut self, other: &UniversalValueFunction) {
        self.w1.assign(&other.w1);
        self.b1.assign(&other.b1);
        self.w2.assign(&other.w2);
        self.b2.assign(&other.b2);
    }
}

// ============================================================================
// Goal-Conditioned Policy
// ============================================================================

/// Epsilon-greedy goal-conditioned policy
pub struct GoalConditionedPolicy {
    pub q_network: UniversalValueFunction,
    pub target_network: UniversalValueFunction,
    pub epsilon: f64,
    pub epsilon_min: f64,
    pub epsilon_decay: f64,
    pub gamma: f64,
}

impl GoalConditionedPolicy {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        lr: f64,
        epsilon: f64,
        gamma: f64,
    ) -> Self {
        let q_network = UniversalValueFunction::new(input_dim, hidden_dim, lr);
        let target_network = UniversalValueFunction::new(input_dim, hidden_dim, lr);

        Self {
            q_network,
            target_network,
            epsilon,
            epsilon_min: 0.01,
            epsilon_decay: 0.995,
            gamma,
        }
    }

    /// Select an action using epsilon-greedy policy
    pub fn select_action(&self, state: &GoalConditionedState) -> usize {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..NUM_ACTIONS)
        } else {
            let input = state.to_augmented_vector();
            self.q_network.best_action(&input)
        }
    }

    /// Train on a batch of transitions
    pub fn train_batch(&mut self, batch: &[&Transition]) {
        for transition in batch {
            let input = transition.state.to_augmented_vector();
            let next_input = transition.next_state.to_augmented_vector();

            let target = if transition.done {
                transition.reward
            } else {
                let next_q = self.target_network.forward(&next_input);
                let max_next_q = next_q
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                transition.reward + self.gamma * max_next_q
            };

            self.q_network.update(&input, transition.action, target);
        }
    }

    /// Update target network
    pub fn update_target_network(&mut self) {
        self.target_network.copy_from(&self.q_network);
    }

    /// Decay epsilon
    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
    }
}

// ============================================================================
// Trading Environment
// ============================================================================

/// Market candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketCandle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Trading environment that supports goal conditioning
pub struct TradingEnvironment {
    pub candles: Vec<MarketCandle>,
    pub current_step: usize,
    pub initial_balance: f64,
    pub balance: f64,
    pub position: f64,
    pub entry_price: f64,
    pub peak_balance: f64,
    pub returns: Vec<f64>,
    pub goal: TradingGoal,
    pub state_dim: usize,
    pub goal_dim: usize,
    pub window_size: usize,
}

impl TradingEnvironment {
    pub fn new(candles: Vec<MarketCandle>, goal: TradingGoal, window_size: usize) -> Self {
        let initial_balance = 10000.0;
        Self {
            candles,
            current_step: window_size,
            initial_balance,
            balance: initial_balance,
            position: 0.0,
            entry_price: 0.0,
            peak_balance: initial_balance,
            returns: Vec::new(),
            goal,
            state_dim: window_size * 5, // OHLCV features per candle
            goal_dim: 3,
            window_size,
        }
    }

    /// Reset the environment
    pub fn reset(&mut self) -> GoalConditionedState {
        self.current_step = self.window_size;
        self.balance = self.initial_balance;
        self.position = 0.0;
        self.entry_price = 0.0;
        self.peak_balance = self.initial_balance;
        self.returns.clear();
        self.get_state()
    }

    /// Get current goal-conditioned state
    pub fn get_state(&self) -> GoalConditionedState {
        let market_state = self.get_market_features();
        let goal_vec = self.goal.to_vector();
        let achieved = self.get_achieved_goal();
        GoalConditionedState::new(market_state, goal_vec, achieved)
    }

    /// Extract market features from the observation window
    fn get_market_features(&self) -> Vec<f64> {
        let mut features = Vec::with_capacity(self.state_dim);
        let start = if self.current_step >= self.window_size {
            self.current_step - self.window_size
        } else {
            0
        };
        let end = self.current_step.min(self.candles.len());

        for i in start..end {
            let c = &self.candles[i];
            // Normalize by the current close price
            let norm = if c.close > 0.0 { c.close } else { 1.0 };
            features.push(c.open / norm - 1.0);
            features.push(c.high / norm - 1.0);
            features.push(c.low / norm - 1.0);
            features.push(c.close / norm - 1.0);
            features.push(c.volume / 1e9); // Scale volume
        }

        // Pad if needed
        while features.len() < self.state_dim {
            features.push(0.0);
        }
        features.truncate(self.state_dim);
        features
    }

    /// Compute the currently achieved goal metrics
    pub fn get_achieved_goal(&self) -> Vec<f64> {
        let cumulative_return = (self.balance - self.initial_balance) / self.initial_balance;
        let drawdown = (self.balance - self.peak_balance) / self.peak_balance;
        let sharpe = self.compute_sharpe();
        vec![cumulative_return, drawdown, sharpe]
    }

    fn compute_sharpe(&self) -> f64 {
        if self.returns.len() < 2 {
            return 0.0;
        }
        let mean: f64 = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
        let variance: f64 = self.returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / (self.returns.len() - 1) as f64;
        let std = variance.sqrt();
        if std < 1e-10 {
            return 0.0;
        }
        (mean / std) * (252.0_f64).sqrt()
    }

    /// Execute one step in the environment
    pub fn step(&mut self, action: usize) -> (GoalConditionedState, f64, bool) {
        let prev_balance = self.balance;

        if self.current_step >= self.candles.len() - 1 {
            let state = self.get_state();
            let achieved = self.get_achieved_goal();
            let goal_vec = self.goal.to_vector();
            let reward = compute_goal_reward(&achieved, &goal_vec, 0.01);
            return (state, reward, true);
        }

        let current_price = self.candles[self.current_step].close;
        let next_price = self.candles[self.current_step + 1].close;

        match TradingAction::from_index(action) {
            TradingAction::Buy => {
                if self.position == 0.0 {
                    self.position = self.balance / current_price;
                    self.entry_price = current_price;
                    self.balance = 0.0;
                }
            }
            TradingAction::Sell => {
                if self.position > 0.0 {
                    self.balance = self.position * current_price;
                    self.position = 0.0;
                    self.entry_price = 0.0;
                }
            }
            TradingAction::Hold => {}
        }

        // Update balance if holding a position
        if self.position > 0.0 {
            let total_value = self.position * next_price;
            let daily_return = (next_price - current_price) / current_price;
            self.returns.push(daily_return);
            self.balance = 0.0;
            // Track effective balance
            let effective_balance = total_value;
            if effective_balance > self.peak_balance {
                self.peak_balance = effective_balance;
            }
        } else {
            self.returns.push(0.0);
            if self.balance > self.peak_balance {
                self.peak_balance = self.balance;
            }
        }

        self.current_step += 1;

        let effective_balance = if self.position > 0.0 {
            self.position * self.candles[self.current_step].close
        } else {
            self.balance
        };

        let step_return = if prev_balance > 0.0 {
            (effective_balance - prev_balance) / prev_balance
        } else {
            0.0
        };

        let done = self.current_step >= self.candles.len() - 1;
        let state = self.get_state();
        let achieved = self.get_achieved_goal();
        let goal_vec = self.goal.to_vector();

        // Goal-conditioned reward: distance shaping + sparse bonus
        let reward = compute_goal_reward(&achieved, &goal_vec, 0.01) + step_return;

        (state, reward, done)
    }

    /// Set a new goal for the environment
    pub fn set_goal(&mut self, goal: TradingGoal) {
        self.goal = goal;
    }

    /// Get total portfolio value
    pub fn portfolio_value(&self) -> f64 {
        if self.position > 0.0 && self.current_step < self.candles.len() {
            self.position * self.candles[self.current_step].close
        } else {
            self.balance
        }
    }
}

// ============================================================================
// GCRL Training Agent
// ============================================================================

/// Complete Goal-Conditioned RL training agent
pub struct GCRLAgent {
    pub policy: GoalConditionedPolicy,
    pub replay_buffer: HERReplayBuffer,
    pub batch_size: usize,
    pub target_update_freq: usize,
    pub train_steps: usize,
}

impl GCRLAgent {
    pub fn new(
        state_dim: usize,
        goal_dim: usize,
        hidden_dim: usize,
        lr: f64,
        buffer_capacity: usize,
        batch_size: usize,
        her_strategy: HERStrategy,
    ) -> Self {
        let input_dim = state_dim + goal_dim;
        let policy = GoalConditionedPolicy::new(input_dim, hidden_dim, lr, 1.0, 0.99);
        let replay_buffer = HERReplayBuffer::new(buffer_capacity, her_strategy, 0.8);

        Self {
            policy,
            replay_buffer,
            batch_size,
            target_update_freq: 10,
            train_steps: 0,
        }
    }

    /// Run one training episode
    pub fn train_episode(&mut self, env: &mut TradingEnvironment) -> f64 {
        let mut state = env.reset();
        let mut total_reward = 0.0;

        loop {
            let action = self.policy.select_action(&state);
            let (next_state, reward, done) = env.step(action);

            let transition = Transition {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            };

            self.replay_buffer.add_transition(transition);
            total_reward += reward;
            state = next_state;

            if done {
                break;
            }
        }

        // End episode (triggers HER relabeling)
        self.replay_buffer.end_episode();

        // Train on replay buffer
        if self.replay_buffer.len() >= self.batch_size {
            let batch = self.replay_buffer.sample(self.batch_size);
            self.policy.train_batch(&batch);
        }

        self.train_steps += 1;

        // Update target network periodically
        if self.train_steps % self.target_update_freq == 0 {
            self.policy.update_target_network();
        }

        // Decay exploration
        self.policy.decay_epsilon();

        total_reward
    }

    /// Evaluate the agent with a specific goal (no exploration)
    pub fn evaluate(&self, env: &mut TradingEnvironment, goal: TradingGoal) -> (f64, Vec<f64>) {
        env.set_goal(goal);
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let old_epsilon = self.policy.epsilon;

        loop {
            let input = state.to_augmented_vector();
            let action = self.policy.q_network.best_action(&input);
            let (next_state, reward, done) = env.step(action);
            total_reward += reward;
            state = next_state;

            if done {
                break;
            }
        }

        let achieved = env.get_achieved_goal();
        let _ = old_epsilon; // epsilon not modified since we use q_network directly
        (total_reward, achieved)
    }
}

// ============================================================================
// Bybit API Client
// ============================================================================

/// Response structures for Bybit V5 API
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

/// Bybit market data client
pub struct BybitClient {
    pub base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data from Bybit V5 API
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<MarketCandle>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: BybitResponse = reqwest::blocking::get(&url)?.json()?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let mut candles: Vec<MarketCandle> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 6 {
                    Some(MarketCandle {
                        timestamp: item[0].parse().unwrap_or(0),
                        open: item[1].parse().unwrap_or(0.0),
                        high: item[2].parse().unwrap_or(0.0),
                        low: item[3].parse().unwrap_or(0.0),
                        close: item[4].parse().unwrap_or(0.0),
                        volume: item[5].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first; reverse to chronological order
        candles.reverse();
        Ok(candles)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate synthetic candle data for testing
pub fn generate_synthetic_candles(num_candles: usize, start_price: f64) -> Vec<MarketCandle> {
    let mut rng = rand::thread_rng();
    let mut candles = Vec::with_capacity(num_candles);
    let mut price = start_price;

    for i in 0..num_candles {
        let change = rng.gen_range(-0.02..0.02);
        let open = price;
        let close = price * (1.0 + change);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..10000.0);

        candles.push(MarketCandle {
            timestamp: 1700000000 + i as u64 * 3600,
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    candles
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_goal_to_vector() {
        let goal = TradingGoal::TargetReturn(0.05);
        let v = goal.to_vector();
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 1.0);
        assert!((v[1] - 0.05).abs() < 1e-10);
        assert_eq!(v[2], 0.0);

        let goal2 = TradingGoal::MaxDrawdown(-0.10);
        let v2 = goal2.to_vector();
        assert_eq!(v2[0], 0.0);
        assert!((v2[1] - (-0.10)).abs() < 1e-10);

        let goal3 = TradingGoal::TargetSharpe(1.5);
        let v3 = goal3.to_vector();
        assert!((v3[2] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_goal_conditioned_state_augmentation() {
        let market = vec![1.0, 2.0, 3.0];
        let goal = vec![0.05, 0.0, 0.0];
        let achieved = vec![0.02, -0.01, 0.5];
        let state = GoalConditionedState::new(market, goal, achieved);

        let augmented = state.to_augmented_vector();
        assert_eq!(augmented.len(), 6);
        assert_eq!(augmented[0], 1.0);
        assert_eq!(augmented[3], 0.05);
        assert_eq!(state.dim(), 6);
    }

    #[test]
    fn test_compute_goal_reward() {
        // Achieved matches desired exactly
        let achieved = vec![0.05];
        let desired = vec![0.05];
        let reward = compute_goal_reward(&achieved, &desired, 0.01);
        assert!(reward > 0.0); // Should get sparse bonus

        // Achieved far from desired
        let achieved2 = vec![0.0];
        let desired2 = vec![1.0];
        let reward2 = compute_goal_reward(&achieved2, &desired2, 0.01);
        assert!(reward2 < 0.0); // Should be negative (distance penalty)

        // Close but not exact
        let achieved3 = vec![0.049];
        let desired3 = vec![0.05];
        let reward3 = compute_goal_reward(&achieved3, &desired3, 0.01);
        assert!(reward3 > 0.0); // Within tolerance
    }

    #[test]
    fn test_uvf_forward_and_best_action() {
        let uvf = UniversalValueFunction::new(6, 16, 0.01);
        let input = vec![1.0, 0.5, -0.3, 0.05, 0.0, 0.0];
        let q_values = uvf.forward(&input);
        assert_eq!(q_values.len(), NUM_ACTIONS);

        let best = uvf.best_action(&input);
        assert!(best < NUM_ACTIONS);
    }

    #[test]
    fn test_her_replay_buffer() {
        let mut buffer = HERReplayBuffer::new(1000, HERStrategy::Final, 1.0);

        // Create a small episode
        for i in 0..5 {
            let state = GoalConditionedState::new(
                vec![i as f64; 3],
                vec![0.1, 0.0, 0.0],
                vec![i as f64 * 0.01, 0.0, 0.0],
            );
            let next_state = GoalConditionedState::new(
                vec![(i + 1) as f64; 3],
                vec![0.1, 0.0, 0.0],
                vec![(i + 1) as f64 * 0.01, 0.0, 0.0],
            );
            buffer.add_transition(Transition {
                state,
                action: 1,
                reward: 0.0,
                next_state,
                done: i == 4,
            });
        }

        assert_eq!(buffer.len(), 0); // Not yet finalized
        buffer.end_episode();
        assert!(buffer.len() > 5); // Should have original + HER transitions
    }

    #[test]
    fn test_trading_environment_step() {
        let candles = generate_synthetic_candles(50, 100.0);
        let goal = TradingGoal::TargetReturn(0.05);
        let mut env = TradingEnvironment::new(candles, goal, 5);

        let state = env.reset();
        assert_eq!(state.market_state.len(), 25); // 5 candles * 5 features
        assert_eq!(state.goal.len(), 3);

        // Take a buy action
        let (_next_state, _reward, done) = env.step(1); // Buy
        assert!(!done);

        // Take a hold action
        let (_, _, _) = env.step(0); // Hold

        // Take a sell action
        let (_, _, _) = env.step(2); // Sell

        let value = env.portfolio_value();
        assert!(value > 0.0);
    }

    #[test]
    fn test_gcrl_agent_training() {
        let candles = generate_synthetic_candles(100, 100.0);
        let goal = TradingGoal::TargetReturn(0.05);
        let window_size = 5;
        let state_dim = window_size * 5;
        let goal_dim = 3;

        let mut agent = GCRLAgent::new(
            state_dim,
            goal_dim,
            32,
            0.001,
            10000,
            16,
            HERStrategy::Future,
        );

        let mut env = TradingEnvironment::new(candles, goal, window_size);

        // Run a few training episodes
        let mut rewards = Vec::new();
        for _ in 0..3 {
            let r = agent.train_episode(&mut env);
            rewards.push(r);
        }

        assert_eq!(rewards.len(), 3);
        assert!(agent.replay_buffer.len() > 0);
        assert!(agent.policy.epsilon < 1.0); // Epsilon should have decayed
    }

    #[test]
    fn test_action_from_index() {
        assert_eq!(TradingAction::from_index(0), TradingAction::Hold);
        assert_eq!(TradingAction::from_index(1), TradingAction::Buy);
        assert_eq!(TradingAction::from_index(2), TradingAction::Sell);
        assert_eq!(TradingAction::from_index(99), TradingAction::Hold);
    }

    #[test]
    fn test_synthetic_candles() {
        let candles = generate_synthetic_candles(100, 50000.0);
        assert_eq!(candles.len(), 100);
        for c in &candles {
            assert!(c.high >= c.open.max(c.close));
            assert!(c.low <= c.open.min(c.close));
            assert!(c.volume > 0.0);
        }
    }
}
