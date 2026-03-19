//! # MuZero Trading
//!
//! A MuZero implementation for algorithmic trading.
//! MuZero learns to plan without knowing the environment rules by learning
//! three networks: representation, dynamics, and prediction.

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of possible trading actions: StrongBuy, Buy, Hold, Sell, StrongSell
pub const NUM_ACTIONS: usize = 5;

/// Default hidden state dimension
pub const HIDDEN_DIM: usize = 32;

/// Default observation dimension (OHLCV * window + position + pnl)
pub const OBS_DIM: usize = 40;

/// MCTS exploration constant (c_puct)
pub const C_PUCT: f64 = 1.25;

/// Discount factor
pub const GAMMA: f64 = 0.99;

// ---------------------------------------------------------------------------
// Trading Actions
// ---------------------------------------------------------------------------

/// Discrete trading actions available to the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TradingAction {
    StrongBuy = 0,
    Buy = 1,
    Hold = 2,
    Sell = 3,
    StrongSell = 4,
}

impl TradingAction {
    /// Convert an index to a TradingAction.
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => TradingAction::StrongBuy,
            1 => TradingAction::Buy,
            2 => TradingAction::Hold,
            3 => TradingAction::Sell,
            4 => TradingAction::StrongSell,
            _ => TradingAction::Hold,
        }
    }

    /// Get all possible actions.
    pub fn all() -> Vec<TradingAction> {
        vec![
            TradingAction::StrongBuy,
            TradingAction::Buy,
            TradingAction::Hold,
            TradingAction::Sell,
            TradingAction::StrongSell,
        ]
    }

    /// Convert action to one-hot encoding.
    pub fn to_one_hot(&self) -> Array1<f64> {
        let mut v = Array1::zeros(NUM_ACTIONS);
        v[*self as usize] = 1.0;
        v
    }
}

// ---------------------------------------------------------------------------
// Simple Dense Layer
// ---------------------------------------------------------------------------

/// A simple fully-connected layer: y = activation(W * x + b)
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
    pub use_relu: bool,
}

impl DenseLayer {
    /// Create a new dense layer with random Xavier initialization.
    pub fn new(input_dim: usize, output_dim: usize, use_relu: bool) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let weights = Array2::from_shape_fn((output_dim, input_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let bias = Array1::zeros(output_dim);
        DenseLayer {
            weights,
            bias,
            use_relu,
        }
    }

    /// Forward pass through the layer.
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let z = self.weights.dot(input) + &self.bias;
        if self.use_relu {
            z.mapv(|x| x.max(0.0))
        } else {
            z
        }
    }
}

// ---------------------------------------------------------------------------
// Representation Network: observation -> hidden state
// ---------------------------------------------------------------------------

/// Representation network h_theta: maps raw observations to a hidden state.
#[derive(Debug, Clone)]
pub struct RepresentationNet {
    pub layer1: DenseLayer,
    pub layer2: DenseLayer,
}

impl RepresentationNet {
    /// Create a new representation network.
    pub fn new(obs_dim: usize, hidden_dim: usize) -> Self {
        RepresentationNet {
            layer1: DenseLayer::new(obs_dim, 64, true),
            layer2: DenseLayer::new(64, hidden_dim, true),
        }
    }

    /// Forward pass: observation -> hidden state.
    pub fn forward(&self, observation: &Array1<f64>) -> Array1<f64> {
        let h1 = self.layer1.forward(observation);
        self.layer2.forward(&h1)
    }
}

// ---------------------------------------------------------------------------
// Dynamics Network: (hidden_state, action) -> (next_hidden_state, reward)
// ---------------------------------------------------------------------------

/// Dynamics network g_theta: predicts next hidden state and reward
/// given current hidden state and action.
#[derive(Debug, Clone)]
pub struct DynamicsNet {
    pub layer1: DenseLayer,
    pub layer2: DenseLayer,
    pub reward_head: DenseLayer,
}

impl DynamicsNet {
    /// Create a new dynamics network.
    pub fn new(hidden_dim: usize) -> Self {
        let input_dim = hidden_dim + NUM_ACTIONS; // hidden state + one-hot action
        DynamicsNet {
            layer1: DenseLayer::new(input_dim, 64, true),
            layer2: DenseLayer::new(64, hidden_dim, true),
            reward_head: DenseLayer::new(64, 1, false),
        }
    }

    /// Forward pass: (hidden_state, action) -> (next_hidden_state, reward).
    pub fn forward(&self, hidden_state: &Array1<f64>, action: TradingAction) -> (Array1<f64>, f64) {
        let action_one_hot = action.to_one_hot();
        let mut input = Array1::zeros(hidden_state.len() + NUM_ACTIONS);
        input
            .slice_mut(ndarray::s![..hidden_state.len()])
            .assign(hidden_state);
        input
            .slice_mut(ndarray::s![hidden_state.len()..])
            .assign(&action_one_hot);

        let h1 = self.layer1.forward(&input);
        let next_state = self.layer2.forward(&h1);
        let reward = self.reward_head.forward(&h1)[0];
        (next_state, reward)
    }
}

// ---------------------------------------------------------------------------
// Prediction Network: hidden_state -> (policy, value)
// ---------------------------------------------------------------------------

/// Prediction network f_theta: maps hidden state to policy and value.
#[derive(Debug, Clone)]
pub struct PredictionNet {
    pub shared: DenseLayer,
    pub policy_head: DenseLayer,
    pub value_head: DenseLayer,
}

impl PredictionNet {
    /// Create a new prediction network.
    pub fn new(hidden_dim: usize) -> Self {
        PredictionNet {
            shared: DenseLayer::new(hidden_dim, 64, true),
            policy_head: DenseLayer::new(64, NUM_ACTIONS, false),
            value_head: DenseLayer::new(64, 1, false),
        }
    }

    /// Forward pass: hidden_state -> (policy logits, value).
    pub fn forward(&self, hidden_state: &Array1<f64>) -> (Array1<f64>, f64) {
        let h = self.shared.forward(hidden_state);
        let policy_logits = self.policy_head.forward(&h);
        let value = self.value_head.forward(&h)[0].tanh(); // bounded [-1, 1]
        (softmax(&policy_logits), value)
    }
}

/// Softmax function for converting logits to probabilities.
pub fn softmax(logits: &Array1<f64>) -> Array1<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals = logits.mapv(|x| (x - max_val).exp());
    let sum: f64 = exp_vals.sum();
    if sum > 0.0 {
        exp_vals / sum
    } else {
        Array1::from_elem(logits.len(), 1.0 / logits.len() as f64)
    }
}

// ---------------------------------------------------------------------------
// MuZero Model (all three networks combined)
// ---------------------------------------------------------------------------

/// The complete MuZero model comprising representation, dynamics, and prediction networks.
#[derive(Debug, Clone)]
pub struct MuZeroModel {
    pub representation: RepresentationNet,
    pub dynamics: DynamicsNet,
    pub prediction: PredictionNet,
    pub hidden_dim: usize,
}

impl MuZeroModel {
    /// Create a new MuZero model.
    pub fn new(obs_dim: usize, hidden_dim: usize) -> Self {
        MuZeroModel {
            representation: RepresentationNet::new(obs_dim, hidden_dim),
            dynamics: DynamicsNet::new(hidden_dim),
            prediction: PredictionNet::new(hidden_dim),
            hidden_dim,
        }
    }

    /// Get initial hidden state from observation.
    pub fn initial_inference(&self, observation: &Array1<f64>) -> (Array1<f64>, Array1<f64>, f64) {
        let hidden = self.representation.forward(observation);
        let (policy, value) = self.prediction.forward(&hidden);
        (hidden, policy, value)
    }

    /// Recurrent inference: step the learned model forward.
    pub fn recurrent_inference(
        &self,
        hidden_state: &Array1<f64>,
        action: TradingAction,
    ) -> (Array1<f64>, f64, Array1<f64>, f64) {
        let (next_hidden, reward) = self.dynamics.forward(hidden_state, action);
        let (policy, value) = self.prediction.forward(&next_hidden);
        (next_hidden, reward, policy, value)
    }
}

// ---------------------------------------------------------------------------
// MCTS Node and Search
// ---------------------------------------------------------------------------

/// A node in the MCTS tree.
#[derive(Debug, Clone)]
pub struct MCTSNode {
    pub hidden_state: Array1<f64>,
    pub reward: f64,
    pub prior: f64,
    pub visit_count: u32,
    pub value_sum: f64,
    pub children: HashMap<usize, MCTSNode>,
}

impl MCTSNode {
    /// Create a new MCTS node.
    pub fn new(hidden_state: Array1<f64>, reward: f64, prior: f64) -> Self {
        MCTSNode {
            hidden_state,
            reward,
            prior,
            visit_count: 0,
            value_sum: 0.0,
            children: HashMap::new(),
        }
    }

    /// Get the mean value Q(s, a) for this node.
    pub fn value(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f64
        }
    }

    /// Check if this node has been expanded.
    pub fn is_expanded(&self) -> bool {
        !self.children.is_empty()
    }
}

/// Select the best action using the PUCT formula.
pub fn select_action_puct(node: &MCTSNode) -> usize {
    let total_visits = node.visit_count as f64;
    let sqrt_total = total_visits.sqrt();

    let mut best_score = f64::NEG_INFINITY;
    let mut best_action = 0;

    for action_idx in 0..NUM_ACTIONS {
        let score = if let Some(child) = node.children.get(&action_idx) {
            let q = child.value();
            let u = C_PUCT * child.prior * sqrt_total / (1.0 + child.visit_count as f64);
            q + u
        } else {
            // Unvisited child gets maximum exploration bonus
            C_PUCT * sqrt_total
        };

        if score > best_score {
            best_score = score;
            best_action = action_idx;
        }
    }

    best_action
}

/// Expand a node using the MuZero model.
pub fn expand_node(node: &mut MCTSNode, model: &MuZeroModel) {
    for action_idx in 0..NUM_ACTIONS {
        let action = TradingAction::from_index(action_idx);
        let (next_hidden, reward, policy, _value) =
            model.recurrent_inference(&node.hidden_state, action);
        let prior = policy[action_idx];
        let child = MCTSNode::new(next_hidden, reward, prior);
        node.children.insert(action_idx, child);
    }
}

/// Run MCTS from the given root node for a specified number of simulations.
pub fn run_mcts(model: &MuZeroModel, observation: &Array1<f64>, num_simulations: usize) -> MCTSNode {
    let (hidden, policy, _value) = model.initial_inference(observation);
    let mut root = MCTSNode::new(hidden, 0.0, 1.0);

    // Expand root with initial policy priors
    for action_idx in 0..NUM_ACTIONS {
        let action = TradingAction::from_index(action_idx);
        let (next_hidden, reward, child_policy, _) =
            model.recurrent_inference(&root.hidden_state, action);
        let prior = policy[action_idx];
        let mut child = MCTSNode::new(next_hidden, reward, prior);
        // Pre-set child priors from their own prediction
        for a in 0..NUM_ACTIONS {
            let ca = TradingAction::from_index(a);
            let (ch, cr, _cp, _) = model.recurrent_inference(&child.hidden_state, ca);
            let grandchild = MCTSNode::new(ch, cr, child_policy[a]);
            child.children.insert(a, grandchild);
        }
        root.children.insert(action_idx, child);
    }
    root.visit_count = 1;

    for _ in 0..num_simulations {
        // Selection: traverse tree to leaf
        let mut path: Vec<usize> = Vec::new();
        let current_action = select_action_puct(&root);
        path.push(current_action);

        // For simplicity we do a single-level MCTS expansion
        // (in production, you'd recurse deeper)
        if let Some(child) = root.children.get(&current_action) {
            if !child.is_expanded() {
                // Expand
                let child_mut = root.children.get_mut(&current_action).unwrap();
                expand_node(child_mut, model);
            }

            // Evaluate leaf
            let child = root.children.get(&current_action).unwrap();
            let (_, value) = model.prediction.forward(&child.hidden_state);
            let backed_up_value = child.reward + GAMMA * value;

            // Backup
            let child_mut = root.children.get_mut(&current_action).unwrap();
            child_mut.visit_count += 1;
            child_mut.value_sum += backed_up_value;
            root.visit_count += 1;
        }
    }

    root
}

/// Select the best action from MCTS root based on visit counts.
pub fn select_action_from_mcts(root: &MCTSNode, temperature: f64) -> (usize, Array1<f64>) {
    let mut visit_counts = Array1::zeros(NUM_ACTIONS);
    for (action_idx, child) in &root.children {
        visit_counts[*action_idx] = child.visit_count as f64;
    }

    let policy = if temperature < 1e-6 {
        // Greedy selection
        let mut p = Array1::zeros(NUM_ACTIONS);
        let best = visit_counts
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(2);
        p[best] = 1.0;
        p
    } else {
        // Temperature-based selection
        let powered = visit_counts.mapv(|x| x.powf(1.0 / temperature));
        let sum: f64 = powered.sum();
        if sum > 0.0 {
            powered / sum
        } else {
            Array1::from_elem(NUM_ACTIONS, 1.0 / NUM_ACTIONS as f64)
        }
    };

    // Sample from policy
    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen();
    let mut cumsum = 0.0;
    let mut selected = 0;
    for (i, p) in policy.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            selected = i;
            break;
        }
    }

    (selected, policy)
}

// ---------------------------------------------------------------------------
// Training: Replay Buffer and Loss Computation
// ---------------------------------------------------------------------------

/// A single transition stored in the replay buffer.
#[derive(Debug, Clone)]
pub struct Transition {
    pub observation: Array1<f64>,
    pub action: usize,
    pub reward: f64,
    pub mcts_policy: Array1<f64>,
    pub value_target: f64,
}

/// Replay buffer for storing and sampling training data.
#[derive(Debug, Clone)]
pub struct ReplayBuffer {
    pub transitions: Vec<Transition>,
    pub capacity: usize,
}

impl ReplayBuffer {
    /// Create a new replay buffer with given capacity.
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            transitions: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Add a transition to the buffer.
    pub fn push(&mut self, transition: Transition) {
        if self.transitions.len() >= self.capacity {
            self.transitions.remove(0);
        }
        self.transitions.push(transition);
    }

    /// Sample a batch of transitions.
    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        let mut rng = rand::thread_rng();
        let len = self.transitions.len();
        if len == 0 {
            return vec![];
        }
        (0..batch_size)
            .map(|_| {
                let idx = rng.gen_range(0..len);
                &self.transitions[idx]
            })
            .collect()
    }

    /// Get the number of transitions in the buffer.
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }
}

/// Compute the training loss for a batch of transitions.
/// Returns (policy_loss, value_loss, reward_loss, total_loss).
pub fn compute_loss(
    model: &MuZeroModel,
    batch: &[&Transition],
) -> (f64, f64, f64, f64) {
    if batch.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let mut policy_loss = 0.0;
    let mut value_loss = 0.0;
    let mut reward_loss = 0.0;

    for transition in batch {
        // Initial inference
        let (hidden, pred_policy, pred_value) =
            model.initial_inference(&transition.observation);

        // Policy loss: cross-entropy
        for i in 0..NUM_ACTIONS {
            if pred_policy[i] > 1e-8 {
                policy_loss -= transition.mcts_policy[i] * pred_policy[i].ln();
            }
        }

        // Value loss: MSE
        value_loss += (transition.value_target - pred_value).powi(2);

        // Reward loss: step the dynamics model and compare
        let action = TradingAction::from_index(transition.action);
        let (_, pred_reward) = model.dynamics.forward(&hidden, action);
        reward_loss += (transition.reward - pred_reward).powi(2);
    }

    let n = batch.len() as f64;
    policy_loss /= n;
    value_loss /= n;
    reward_loss /= n;
    let total_loss = policy_loss + value_loss + reward_loss;

    (policy_loss, value_loss, reward_loss, total_loss)
}

/// Apply simple stochastic gradient descent to update model parameters.
/// This is a simplified training step that perturbs weights in the direction
/// that reduces loss (finite-difference approximation).
pub fn train_step(
    model: &mut MuZeroModel,
    batch: &[&Transition],
    learning_rate: f64,
) -> f64 {
    let (_, _, _, base_loss) = compute_loss(model, batch);

    // Perturb representation layer1 weights
    let epsilon = 1e-4;
    let (rows, cols) = model.representation.layer1.weights.dim();
    let mut rng = rand::thread_rng();

    // Stochastic coordinate descent: update a random subset of weights
    let num_updates = 20.min(rows * cols);
    for _ in 0..num_updates {
        let r = rng.gen_range(0..rows);
        let c = rng.gen_range(0..cols);

        model.representation.layer1.weights[[r, c]] += epsilon;
        let (_, _, _, loss_plus) = compute_loss(model, batch);
        model.representation.layer1.weights[[r, c]] -= epsilon;

        let grad = (loss_plus - base_loss) / epsilon;
        model.representation.layer1.weights[[r, c]] -= learning_rate * grad;
    }

    // Similarly update prediction policy head
    let (rows, cols) = model.prediction.policy_head.weights.dim();
    let num_updates = 20.min(rows * cols);
    for _ in 0..num_updates {
        let r = rng.gen_range(0..rows);
        let c = rng.gen_range(0..cols);

        model.prediction.policy_head.weights[[r, c]] += epsilon;
        let (_, _, _, loss_plus) = compute_loss(model, batch);
        model.prediction.policy_head.weights[[r, c]] -= epsilon;

        let grad = (loss_plus - base_loss) / epsilon;
        model.prediction.policy_head.weights[[r, c]] -= learning_rate * grad;
    }

    base_loss
}

// ---------------------------------------------------------------------------
// Trading Environment (for self-play / evaluation)
// ---------------------------------------------------------------------------

/// Simple trading environment for evaluating MuZero agent.
#[derive(Debug, Clone)]
pub struct TradingEnv {
    pub prices: Vec<f64>,
    pub current_step: usize,
    pub position: f64,       // -1.0 (short) to 1.0 (long)
    pub entry_price: f64,
    pub total_pnl: f64,
    pub window_size: usize,
    pub obs_dim: usize,
}

impl TradingEnv {
    /// Create a new trading environment from price data.
    pub fn new(prices: Vec<f64>, window_size: usize) -> Self {
        let obs_dim = window_size * 5 + 2; // 5 features per candle + position + pnl
        TradingEnv {
            prices,
            current_step: window_size,
            position: 0.0,
            entry_price: 0.0,
            total_pnl: 0.0,
            window_size,
            obs_dim,
        }
    }

    /// Get the current observation vector.
    pub fn get_observation(&self) -> Array1<f64> {
        let mut obs = Array1::zeros(self.obs_dim);
        let start = self.current_step.saturating_sub(self.window_size);

        for (i, step) in (start..self.current_step).enumerate() {
            if step < self.prices.len() && step > 0 {
                // Log return
                let log_ret = (self.prices[step] / self.prices[step - 1]).ln();
                obs[i * 5] = log_ret;
                // Normalized price relative to window mean
                let mean: f64 = self.prices[start..self.current_step]
                    .iter()
                    .sum::<f64>()
                    / self.window_size as f64;
                obs[i * 5 + 1] = (self.prices[step] - mean) / mean.max(1e-8);
                // Simple volatility proxy
                obs[i * 5 + 2] = log_ret.abs();
                // Price momentum (simple)
                if step >= 3 {
                    obs[i * 5 + 3] = (self.prices[step] / self.prices[step - 3] - 1.0) * 10.0;
                }
                // Volume placeholder (normalized)
                obs[i * 5 + 4] = 0.5;
            }
        }

        // Position and PnL
        let pos_idx = self.window_size * 5;
        obs[pos_idx] = self.position;
        obs[pos_idx + 1] = self.total_pnl.tanh();

        obs
    }

    /// Execute a trading action and return (reward, done).
    pub fn step(&mut self, action: TradingAction) -> (f64, bool) {
        if self.current_step >= self.prices.len() - 1 {
            return (0.0, true);
        }

        let current_price = self.prices[self.current_step];
        let next_price = self.prices[self.current_step + 1];
        let price_change = (next_price - current_price) / current_price;

        // Map action to target position
        let target_position = match action {
            TradingAction::StrongBuy => 1.0,
            TradingAction::Buy => 0.5,
            TradingAction::Hold => self.position,
            TradingAction::Sell => -0.5,
            TradingAction::StrongSell => -1.0,
        };

        // Transaction cost for changing position
        let position_change = (target_position - self.position).abs();
        let transaction_cost = position_change * 0.001; // 0.1% per unit change

        // P&L from holding position during price change
        let pnl = self.position * price_change - transaction_cost;
        self.total_pnl += pnl;

        // Update position
        if (target_position - self.position).abs() > 0.01 {
            self.position = target_position;
            self.entry_price = current_price;
        }

        self.current_step += 1;
        let done = self.current_step >= self.prices.len() - 1;

        // Risk-adjusted reward (penalize large drawdowns)
        let reward = pnl - 0.5 * pnl.min(0.0).powi(2);

        (reward, done)
    }

    /// Reset the environment.
    pub fn reset(&mut self) {
        self.current_step = self.window_size;
        self.position = 0.0;
        self.entry_price = 0.0;
        self.total_pnl = 0.0;
    }

    /// Check if the episode is done.
    pub fn is_done(&self) -> bool {
        self.current_step >= self.prices.len() - 1
    }
}

// ---------------------------------------------------------------------------
// Self-play: generate training data
// ---------------------------------------------------------------------------

/// Run one episode of self-play using MCTS to generate training data.
pub fn self_play_episode(
    model: &MuZeroModel,
    env: &mut TradingEnv,
    num_simulations: usize,
    temperature: f64,
) -> Vec<Transition> {
    env.reset();
    let mut transitions = Vec::new();

    while !env.is_done() {
        let obs = env.get_observation();
        let root = run_mcts(model, &obs, num_simulations);
        let (action_idx, mcts_policy) = select_action_from_mcts(&root, temperature);
        let action = TradingAction::from_index(action_idx);

        let (reward, _done) = env.step(action);

        transitions.push(Transition {
            observation: obs,
            action: action_idx,
            reward,
            mcts_policy,
            value_target: 0.0, // will be filled by bootstrapping
        });
    }

    // Bootstrap value targets using n-step returns
    let n = transitions.len();
    for i in (0..n).rev() {
        let future_value = if i + 1 < n {
            transitions[i + 1].value_target
        } else {
            0.0
        };
        transitions[i].value_target = transitions[i].reward + GAMMA * future_value;
    }

    transitions
}

// ---------------------------------------------------------------------------
// Reanalysis
// ---------------------------------------------------------------------------

/// Reanalyze old trajectories with the current model to improve targets.
pub fn reanalyze(
    model: &MuZeroModel,
    buffer: &mut ReplayBuffer,
    num_simulations: usize,
) {
    for transition in buffer.transitions.iter_mut() {
        let root = run_mcts(model, &transition.observation, num_simulations);
        let (_, new_policy) = select_action_from_mcts(&root, 1.0);
        transition.mcts_policy = new_policy;

        // Update value target with current model's estimate
        let (_, _, value) = model.initial_inference(&transition.observation);
        transition.value_target = transition.reward + GAMMA * value;
    }
}

// ---------------------------------------------------------------------------
// Value scaling (for large reward ranges)
// ---------------------------------------------------------------------------

/// Invertible value scaling: h(x) = sign(x) * (sqrt(|x| + 1) - 1) + eps * x
pub fn scale_value(x: f64, epsilon: f64) -> f64 {
    x.signum() * ((x.abs() + 1.0).sqrt() - 1.0) + epsilon * x
}

/// Inverse of value scaling.
pub fn inverse_scale_value(x: f64, epsilon: f64) -> f64 {
    let _sign = x.signum();
    let _x_abs = x.abs();
    // Solve quadratic: (1+eps)*y = sign*(sqrt(y+1)-1) + eps*y ... simplified
    // Approximate inverse via Newton's method
    let mut y = x;
    for _ in 0..10 {
        let f = scale_value(y, epsilon) - x;
        let df = 1.0 / (2.0 * (y.abs() + 1.0).sqrt()) + epsilon;
        y -= f / df;
    }
    y
}

// ---------------------------------------------------------------------------
// Simple DQN baseline for comparison
// ---------------------------------------------------------------------------

/// A simple DQN agent for comparison with MuZero.
#[derive(Debug, Clone)]
pub struct SimpleDQN {
    pub layer1: DenseLayer,
    pub layer2: DenseLayer,
    pub output: DenseLayer,
}

impl SimpleDQN {
    /// Create a new DQN.
    pub fn new(obs_dim: usize) -> Self {
        SimpleDQN {
            layer1: DenseLayer::new(obs_dim, 64, true),
            layer2: DenseLayer::new(64, 32, true),
            output: DenseLayer::new(32, NUM_ACTIONS, false),
        }
    }

    /// Get Q-values for an observation.
    pub fn q_values(&self, obs: &Array1<f64>) -> Array1<f64> {
        let h1 = self.layer1.forward(obs);
        let h2 = self.layer2.forward(&h1);
        self.output.forward(&h2)
    }

    /// Select action using epsilon-greedy.
    pub fn select_action(&self, obs: &Array1<f64>, epsilon: f64) -> usize {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < epsilon {
            rng.gen_range(0..NUM_ACTIONS)
        } else {
            let q = self.q_values(obs);
            q.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(2)
        }
    }
}

// ---------------------------------------------------------------------------
// Bybit API
// ---------------------------------------------------------------------------

/// Kline (candlestick) data from Bybit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub open_time: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API response structure.
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
    pub list: Vec<Vec<String>>,
}

/// Fetch kline data from Bybit API.
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<Kline>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::Client::new();
    let resp: BybitResponse = client.get(&url).send().await?.json().await?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let mut klines: Vec<Kline> = resp
        .result
        .list
        .iter()
        .filter_map(|item| {
            if item.len() >= 6 {
                Some(Kline {
                    open_time: item[0].parse().unwrap_or(0),
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

    // Bybit returns newest first, reverse to chronological order
    klines.reverse();
    Ok(klines)
}

/// Fetch kline data synchronously (blocking).
pub fn fetch_bybit_klines_blocking(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<Kline>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let mut klines: Vec<Kline> = resp
        .result
        .list
        .iter()
        .filter_map(|item| {
            if item.len() >= 6 {
                Some(Kline {
                    open_time: item[0].parse().unwrap_or(0),
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

    klines.reverse();
    Ok(klines)
}

/// Convert klines to a vector of close prices.
pub fn klines_to_prices(klines: &[Kline]) -> Vec<f64> {
    klines.iter().map(|k| k.close).collect()
}

/// Generate synthetic price data for testing.
pub fn generate_synthetic_prices(length: usize, seed_price: f64) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut prices = Vec::with_capacity(length);
    let mut price = seed_price;

    for _ in 0..length {
        prices.push(price);
        let ret = rng.gen_range(-0.02..0.02);
        price *= 1.0 + ret;
        price = price.max(seed_price * 0.5);
    }

    prices
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_action_roundtrip() {
        for i in 0..NUM_ACTIONS {
            let action = TradingAction::from_index(i);
            assert_eq!(action as usize, i);
        }
        assert_eq!(TradingAction::all().len(), NUM_ACTIONS);
    }

    #[test]
    fn test_one_hot_encoding() {
        let action = TradingAction::Buy;
        let one_hot = action.to_one_hot();
        assert_eq!(one_hot.len(), NUM_ACTIONS);
        assert_eq!(one_hot[1], 1.0);
        assert_eq!(one_hot[0], 0.0);
        assert_eq!(one_hot.sum(), 1.0);
    }

    #[test]
    fn test_dense_layer_shapes() {
        let layer = DenseLayer::new(10, 5, true);
        let input = Array1::zeros(10);
        let output = layer.forward(&input);
        assert_eq!(output.len(), 5);
        // With ReLU, all zeros input -> non-negative outputs
        for &val in output.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_representation_net() {
        let net = RepresentationNet::new(OBS_DIM, HIDDEN_DIM);
        let obs = Array1::from_elem(OBS_DIM, 0.5);
        let hidden = net.forward(&obs);
        assert_eq!(hidden.len(), HIDDEN_DIM);
    }

    #[test]
    fn test_dynamics_net() {
        let net = DynamicsNet::new(HIDDEN_DIM);
        let hidden = Array1::from_elem(HIDDEN_DIM, 0.1);
        let (next_hidden, reward) = net.forward(&hidden, TradingAction::Buy);
        assert_eq!(next_hidden.len(), HIDDEN_DIM);
        assert!(reward.is_finite());
    }

    #[test]
    fn test_prediction_net() {
        let net = PredictionNet::new(HIDDEN_DIM);
        let hidden = Array1::from_elem(HIDDEN_DIM, 0.1);
        let (policy, value) = net.forward(&hidden);
        assert_eq!(policy.len(), NUM_ACTIONS);
        // Policy should sum to ~1.0
        assert!((policy.sum() - 1.0).abs() < 1e-6);
        // Value should be in [-1, 1] (tanh)
        assert!(value >= -1.0 && value <= 1.0);
    }

    #[test]
    fn test_muzero_model_inference() {
        let model = MuZeroModel::new(OBS_DIM, HIDDEN_DIM);
        let obs = Array1::from_elem(OBS_DIM, 0.3);

        let (hidden, policy, _value) = model.initial_inference(&obs);
        assert_eq!(hidden.len(), HIDDEN_DIM);
        assert_eq!(policy.len(), NUM_ACTIONS);
        assert!((policy.sum() - 1.0).abs() < 1e-6);

        let (next_hidden, reward, next_policy, next_value) =
            model.recurrent_inference(&hidden, TradingAction::Hold);
        assert_eq!(next_hidden.len(), HIDDEN_DIM);
        assert!(reward.is_finite());
        assert_eq!(next_policy.len(), NUM_ACTIONS);
        assert!(next_value >= -1.0 && next_value <= 1.0);
    }

    #[test]
    fn test_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let probs = softmax(&logits);
        assert!((probs.sum() - 1.0).abs() < 1e-6);
        // Probabilities should be increasing
        for i in 1..probs.len() {
            assert!(probs[i] > probs[i - 1]);
        }
    }

    #[test]
    fn test_mcts_runs() {
        let model = MuZeroModel::new(OBS_DIM, HIDDEN_DIM);
        let obs = Array1::from_elem(OBS_DIM, 0.2);
        let root = run_mcts(&model, &obs, 10);

        // Root should have been visited
        assert!(root.visit_count > 0);
        // Should have children for each action
        assert_eq!(root.children.len(), NUM_ACTIONS);
    }

    #[test]
    fn test_trading_env() {
        let prices = generate_synthetic_prices(100, 50000.0);
        let mut env = TradingEnv::new(prices, 7);
        let obs = env.get_observation();
        assert_eq!(obs.len(), env.obs_dim);

        let (reward, done) = env.step(TradingAction::Buy);
        assert!(reward.is_finite());
        assert!(!done);
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100);
        assert!(buffer.is_empty());

        buffer.push(Transition {
            observation: Array1::zeros(OBS_DIM),
            action: 2,
            reward: 0.1,
            mcts_policy: Array1::from_elem(NUM_ACTIONS, 0.2),
            value_target: 0.5,
        });

        assert_eq!(buffer.len(), 1);
        let sample = buffer.sample(1);
        assert_eq!(sample.len(), 1);
    }

    #[test]
    fn test_value_scaling() {
        let eps = 0.001;
        let values = vec![0.0, 1.0, -1.0, 100.0, -100.0, 0.5];
        for v in values {
            let scaled = scale_value(v, eps);
            let unscaled = inverse_scale_value(scaled, eps);
            assert!(
                (v - unscaled).abs() < 0.1,
                "Value scaling roundtrip failed for {}: got {}",
                v,
                unscaled
            );
        }
    }

    #[test]
    fn test_self_play_episode() {
        let model = MuZeroModel::new(37, HIDDEN_DIM); // 7*5+2 = 37
        let prices = generate_synthetic_prices(30, 50000.0);
        let mut env = TradingEnv::new(prices, 7);

        let transitions = self_play_episode(&model, &mut env, 5, 1.0);
        assert!(!transitions.is_empty());

        // Check that value targets were computed
        for t in &transitions {
            assert!(t.value_target.is_finite());
        }
    }

    #[test]
    fn test_compute_loss() {
        let model = MuZeroModel::new(OBS_DIM, HIDDEN_DIM);
        let transitions = vec![Transition {
            observation: Array1::from_elem(OBS_DIM, 0.1),
            action: 0,
            reward: 0.05,
            mcts_policy: Array1::from_elem(NUM_ACTIONS, 0.2),
            value_target: 0.3,
        }];

        let refs: Vec<&Transition> = transitions.iter().collect();
        let (pl, vl, rl, total) = compute_loss(&model, &refs);
        assert!(total.is_finite());
        assert!(total >= 0.0);
        assert!((total - pl - vl - rl).abs() < 1e-10);
    }

    #[test]
    fn test_generate_synthetic_prices() {
        let prices = generate_synthetic_prices(200, 50000.0);
        assert_eq!(prices.len(), 200);
        assert_eq!(prices[0], 50000.0);
        for p in &prices {
            assert!(*p > 0.0);
        }
    }

    #[test]
    fn test_dqn_baseline() {
        let dqn = SimpleDQN::new(OBS_DIM);
        let obs = Array1::from_elem(OBS_DIM, 0.3);
        let q = dqn.q_values(&obs);
        assert_eq!(q.len(), NUM_ACTIONS);

        let action = dqn.select_action(&obs, 0.0);
        assert!(action < NUM_ACTIONS);

        // With epsilon=1.0, should still return valid action
        let random_action = dqn.select_action(&obs, 1.0);
        assert!(random_action < NUM_ACTIONS);
    }
}
