//! # Planning RL Trading
//!
//! Model-based reinforcement learning with explicit planning for algorithmic trading.
//! Implements world models, Dyna-Q, MPC planning, and model uncertainty estimation.

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Trading types
// ---------------------------------------------------------------------------

/// Discrete trading actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Buy,
    Hold,
    Sell,
}

impl Action {
    pub fn all() -> &'static [Action] {
        &[Action::Buy, Action::Hold, Action::Sell]
    }

    pub fn index(&self) -> usize {
        match self {
            Action::Buy => 0,
            Action::Hold => 1,
            Action::Sell => 2,
        }
    }

    pub fn from_index(i: usize) -> Self {
        match i % 3 {
            0 => Action::Buy,
            1 => Action::Hold,
            _ => Action::Sell,
        }
    }
}

/// A single transition recorded from the environment.
#[derive(Debug, Clone)]
pub struct Transition {
    pub state: Vec<f64>,
    pub action: Action,
    pub reward: f64,
    pub next_state: Vec<f64>,
}

// ---------------------------------------------------------------------------
// World Model
// ---------------------------------------------------------------------------

/// A learned transition + reward model.
///
/// Uses a simple linear model: next_state = W_s * state + W_a * action_onehot + bias
/// Reward similarly: reward = w_r * state + w_ra * action_onehot + bias_r
pub struct WorldModel {
    state_dim: usize,
    num_actions: usize,
    // Transition weights
    w_transition: Array2<f64>,
    bias_transition: Array1<f64>,
    // Reward weights
    w_reward: Array1<f64>,
    bias_reward: f64,
    learning_rate: f64,
}

impl WorldModel {
    /// Create a new world model with random weights.
    pub fn new(state_dim: usize, learning_rate: f64) -> Self {
        let num_actions = 3;
        let input_dim = state_dim + num_actions;
        let mut rng = rand::thread_rng();

        let w_transition = Array2::from_shape_fn((state_dim, input_dim), |_| {
            rng.gen_range(-0.1..0.1)
        });
        let bias_transition = Array1::zeros(state_dim);

        let w_reward = Array1::from_shape_fn(input_dim, |_| rng.gen_range(-0.1..0.1));
        let bias_reward = 0.0;

        Self {
            state_dim,
            num_actions,
            w_transition,
            bias_transition,
            w_reward,
            bias_reward,
            learning_rate,
        }
    }

    fn action_onehot(&self, action: Action) -> Vec<f64> {
        let mut v = vec![0.0; self.num_actions];
        v[action.index()] = 1.0;
        v
    }

    fn build_input(&self, state: &[f64], action: Action) -> Array1<f64> {
        let mut input = state.to_vec();
        input.extend(self.action_onehot(action));
        Array1::from_vec(input)
    }

    /// Predict next state given current state and action.
    pub fn predict_next_state(&self, state: &[f64], action: Action) -> Vec<f64> {
        let input = self.build_input(state, action);
        let pred = self.w_transition.dot(&input) + &self.bias_transition;
        pred.to_vec()
    }

    /// Predict reward given current state and action.
    pub fn predict_reward(&self, state: &[f64], action: Action) -> f64 {
        let input = self.build_input(state, action);
        self.w_reward.dot(&input) + self.bias_reward
    }

    /// Train on a single transition using gradient descent.
    pub fn train_step(&mut self, transition: &Transition) {
        let input = self.build_input(&transition.state, transition.action);
        let target_state = Array1::from_vec(transition.next_state.clone());

        // --- Transition model update ---
        let pred_state = self.w_transition.dot(&input) + &self.bias_transition;
        let error_state = &target_state - &pred_state;

        for i in 0..self.state_dim {
            for j in 0..input.len() {
                self.w_transition[[i, j]] += self.learning_rate * error_state[i] * input[j];
            }
            self.bias_transition[i] += self.learning_rate * error_state[i];
        }

        // --- Reward model update ---
        let pred_reward = self.w_reward.dot(&input) + self.bias_reward;
        let error_reward = transition.reward - pred_reward;

        for j in 0..input.len() {
            self.w_reward[j] += self.learning_rate * error_reward * input[j];
        }
        self.bias_reward += self.learning_rate * error_reward;
    }

    /// Train on a batch of transitions for multiple epochs.
    pub fn train(&mut self, transitions: &[Transition], epochs: usize) {
        for _ in 0..epochs {
            for t in transitions {
                self.train_step(t);
            }
        }
    }

    /// Compute prediction error on a batch of transitions (MSE).
    pub fn evaluate(&self, transitions: &[Transition]) -> (f64, f64) {
        if transitions.is_empty() {
            return (0.0, 0.0);
        }
        let mut state_error = 0.0;
        let mut reward_error = 0.0;
        for t in transitions {
            let pred_state = self.predict_next_state(&t.state, t.action);
            for (p, a) in pred_state.iter().zip(t.next_state.iter()) {
                state_error += (p - a).powi(2);
            }
            let pred_r = self.predict_reward(&t.state, t.action);
            reward_error += (pred_r - t.reward).powi(2);
        }
        let n = transitions.len() as f64;
        (state_error / n, reward_error / n)
    }
}

// ---------------------------------------------------------------------------
// Model Ensemble for Uncertainty Estimation
// ---------------------------------------------------------------------------

/// Ensemble of world models for uncertainty estimation.
pub struct ModelEnsemble {
    pub models: Vec<WorldModel>,
}

impl ModelEnsemble {
    /// Create an ensemble of `n` world models.
    pub fn new(n: usize, state_dim: usize, learning_rate: f64) -> Self {
        let models = (0..n)
            .map(|_| WorldModel::new(state_dim, learning_rate))
            .collect();
        Self { models }
    }

    /// Train all models on the given transitions (each with different random init).
    pub fn train(&mut self, transitions: &[Transition], epochs: usize) {
        for model in &mut self.models {
            model.train(transitions, epochs);
        }
    }

    /// Predict next state as ensemble mean.
    pub fn predict_next_state(&self, state: &[f64], action: Action) -> Vec<f64> {
        let predictions: Vec<Vec<f64>> = self
            .models
            .iter()
            .map(|m| m.predict_next_state(state, action))
            .collect();
        let dim = predictions[0].len();
        let n = predictions.len() as f64;
        (0..dim)
            .map(|i| predictions.iter().map(|p| p[i]).sum::<f64>() / n)
            .collect()
    }

    /// Predict reward as ensemble mean.
    pub fn predict_reward(&self, state: &[f64], action: Action) -> f64 {
        let n = self.models.len() as f64;
        self.models
            .iter()
            .map(|m| m.predict_reward(state, action))
            .sum::<f64>()
            / n
    }

    /// Estimate prediction uncertainty (variance across ensemble members).
    pub fn uncertainty(&self, state: &[f64], action: Action) -> f64 {
        let predictions: Vec<Vec<f64>> = self
            .models
            .iter()
            .map(|m| m.predict_next_state(state, action))
            .collect();
        let mean = self.predict_next_state(state, action);
        let dim = mean.len();
        let n = predictions.len() as f64;

        let mut total_var = 0.0;
        for i in 0..dim {
            let var: f64 = predictions.iter().map(|p| (p[i] - mean[i]).powi(2)).sum::<f64>() / n;
            total_var += var;
        }
        total_var / dim as f64
    }
}

// ---------------------------------------------------------------------------
// Dyna-Q Agent
// ---------------------------------------------------------------------------

/// Dyna-Q agent: model-free Q-learning augmented with model-based rollouts.
pub struct DynaQAgent {
    /// Q-values: state_key -> [Q(buy), Q(hold), Q(sell)]
    q_table: HashMap<String, [f64; 3]>,
    /// Replay buffer of real transitions.
    replay_buffer: Vec<Transition>,
    /// World model for simulated rollouts.
    world_model: WorldModel,
    /// Learning rate for Q updates.
    alpha: f64,
    /// Discount factor.
    gamma: f64,
    /// Exploration rate.
    epsilon: f64,
    /// Number of simulated rollouts per real step.
    planning_steps: usize,
    /// Discretization resolution for state keys.
    state_resolution: f64,
}

impl DynaQAgent {
    pub fn new(
        state_dim: usize,
        alpha: f64,
        gamma: f64,
        epsilon: f64,
        planning_steps: usize,
    ) -> Self {
        Self {
            q_table: HashMap::new(),
            replay_buffer: Vec::new(),
            world_model: WorldModel::new(state_dim, 0.001),
            alpha,
            gamma,
            epsilon,
            planning_steps,
            state_resolution: 100.0,
        }
    }

    fn state_key(&self, state: &[f64]) -> String {
        state
            .iter()
            .map(|v| ((v * self.state_resolution).round() as i64).to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    fn get_q(&self, state: &[f64]) -> [f64; 3] {
        let key = self.state_key(state);
        *self.q_table.get(&key).unwrap_or(&[0.0; 3])
    }

    fn max_q(&self, state: &[f64]) -> f64 {
        self.get_q(state)
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Select action using epsilon-greedy policy.
    pub fn select_action(&self, state: &[f64]) -> Action {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.epsilon {
            Action::from_index(rng.gen_range(0..3))
        } else {
            let q = self.get_q(state);
            let best = q
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            Action::from_index(best)
        }
    }

    /// Update Q-value for a single transition.
    fn update_q(&mut self, state: &[f64], action: Action, reward: f64, next_state: &[f64]) {
        let key = self.state_key(state);
        let a_idx = action.index();
        let max_next_q = self.max_q(next_state);
        let alpha = self.alpha;
        let gamma = self.gamma;
        let q_values = self.q_table.entry(key).or_insert([0.0; 3]);
        let target = reward + gamma * max_next_q;
        q_values[a_idx] += alpha * (target - q_values[a_idx]);
    }

    /// Process a real transition: update Q, store in buffer, train model, do planning.
    pub fn step(&mut self, transition: Transition) {
        // 1. Direct RL update
        self.update_q(
            &transition.state,
            transition.action,
            transition.reward,
            &transition.next_state,
        );

        // 2. Store in replay buffer
        self.replay_buffer.push(transition.clone());

        // 3. Train world model
        self.world_model.train_step(&transition);

        // 4. Planning: simulated rollouts
        let mut rng = rand::thread_rng();
        let buffer_len = self.replay_buffer.len();
        for _ in 0..self.planning_steps {
            let idx = rng.gen_range(0..buffer_len);
            let sample_state = self.replay_buffer[idx].state.clone();
            let action = Action::from_index(rng.gen_range(0..3));

            let sim_next = self.world_model.predict_next_state(&sample_state, action);
            let sim_reward = self.world_model.predict_reward(&sample_state, action);

            self.update_q(&sample_state, action, sim_reward, &sim_next);
        }
    }

    /// Get the current Q-table size (number of visited states).
    pub fn q_table_size(&self) -> usize {
        self.q_table.len()
    }

    /// Get the replay buffer size.
    pub fn buffer_size(&self) -> usize {
        self.replay_buffer.len()
    }
}

// ---------------------------------------------------------------------------
// MPC Planner
// ---------------------------------------------------------------------------

/// Planning method for MPC.
#[derive(Debug, Clone, Copy)]
pub enum PlanningMethod {
    RandomShooting,
    CEM { elite_frac: f64, iterations: usize },
}

/// Model Predictive Control planner.
pub struct MPCPlanner {
    /// World model (or ensemble) used for rollouts.
    ensemble: ModelEnsemble,
    /// Planning horizon (number of steps to look ahead).
    horizon: usize,
    /// Number of candidate action sequences to evaluate.
    num_candidates: usize,
    /// Discount factor.
    gamma: f64,
    /// Planning method.
    method: PlanningMethod,
    /// Uncertainty threshold -- if exceeded, prefer Hold.
    uncertainty_threshold: f64,
}

impl MPCPlanner {
    pub fn new(
        ensemble: ModelEnsemble,
        horizon: usize,
        num_candidates: usize,
        gamma: f64,
        method: PlanningMethod,
        uncertainty_threshold: f64,
    ) -> Self {
        Self {
            ensemble,
            horizon,
            num_candidates,
            gamma,
            method,
            uncertainty_threshold,
        }
    }

    /// Train the underlying ensemble on transitions.
    pub fn train(&mut self, transitions: &[Transition], epochs: usize) {
        self.ensemble.train(transitions, epochs);
    }

    /// Plan the best action from the given state.
    pub fn plan(&self, state: &[f64]) -> (Action, f64) {
        // Check uncertainty -- if too high, be conservative
        let unc = self
            .ensemble
            .uncertainty(state, Action::Hold);
        if unc > self.uncertainty_threshold {
            return (Action::Hold, 0.0);
        }

        match self.method {
            PlanningMethod::RandomShooting => self.random_shooting(state),
            PlanningMethod::CEM {
                elite_frac,
                iterations,
            } => self.cem(state, elite_frac, iterations),
        }
    }

    fn random_shooting(&self, state: &[f64]) -> (Action, f64) {
        let mut rng = rand::thread_rng();
        let mut best_reward = f64::NEG_INFINITY;
        let mut best_first_action = Action::Hold;

        for _ in 0..self.num_candidates {
            let actions: Vec<Action> = (0..self.horizon)
                .map(|_| Action::from_index(rng.gen_range(0..3)))
                .collect();

            let total_reward = self.evaluate_sequence(state, &actions);
            if total_reward > best_reward {
                best_reward = total_reward;
                best_first_action = actions[0];
            }
        }
        (best_first_action, best_reward)
    }

    fn cem(&self, state: &[f64], elite_frac: f64, iterations: usize) -> (Action, f64) {
        let mut rng = rand::thread_rng();
        // For discrete actions, CEM maintains probability distribution over actions at each step.
        // action_probs[h] = [p(buy), p(hold), p(sell)] at step h
        let mut action_probs: Vec<[f64; 3]> = vec![[1.0 / 3.0; 3]; self.horizon];
        let elite_count = ((self.num_candidates as f64) * elite_frac).max(1.0) as usize;

        let mut best_reward = f64::NEG_INFINITY;
        let mut best_first_action = Action::Hold;

        for _ in 0..iterations {
            // Sample candidates
            let mut candidates: Vec<(Vec<Action>, f64)> = Vec::new();
            for _ in 0..self.num_candidates {
                let actions: Vec<Action> = (0..self.horizon)
                    .map(|h| {
                        let r: f64 = rng.gen();
                        let probs = action_probs[h];
                        if r < probs[0] {
                            Action::Buy
                        } else if r < probs[0] + probs[1] {
                            Action::Hold
                        } else {
                            Action::Sell
                        }
                    })
                    .collect();
                let reward = self.evaluate_sequence(state, &actions);
                candidates.push((actions, reward));
            }

            // Sort by reward descending and take elite
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let elite = &candidates[..elite_count.min(candidates.len())];

            if elite[0].1 > best_reward {
                best_reward = elite[0].1;
                best_first_action = elite[0].0[0];
            }

            // Refit action probabilities from elite set
            for h in 0..self.horizon {
                let mut counts = [0.0f64; 3];
                for (actions, _) in elite {
                    counts[actions[h].index()] += 1.0;
                }
                let total: f64 = counts.iter().sum();
                for c in &mut counts {
                    *c = (*c + 0.1) / (total + 0.3); // Laplace smoothing
                }
                action_probs[h] = counts;
            }
        }

        (best_first_action, best_reward)
    }

    /// Evaluate a sequence of actions by rolling out the ensemble world model.
    fn evaluate_sequence(&self, state: &[f64], actions: &[Action]) -> f64 {
        let mut current_state = state.to_vec();
        let mut total_reward = 0.0;
        let mut discount = 1.0;

        for &action in actions {
            let reward = self.ensemble.predict_reward(&current_state, action);
            total_reward += discount * reward;
            discount *= self.gamma;
            current_state = self.ensemble.predict_next_state(&current_state, action);
        }

        total_reward
    }

    /// Get the ensemble uncertainty for a given state and action.
    pub fn get_uncertainty(&self, state: &[f64], action: Action) -> f64 {
        self.ensemble.uncertainty(state, action)
    }
}

// ---------------------------------------------------------------------------
// Bybit API Client
// ---------------------------------------------------------------------------

/// Kline (candlestick) data from Bybit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Client for fetching market data from Bybit.
pub struct BybitClient {
    base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline data for a given symbol and interval.
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: retCode={}", resp.ret_code);
        }

        let mut klines: Vec<Kline> = resp
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Kline {
                        timestamp: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first; reverse to chronological order
        klines.reverse();
        Ok(klines)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Feature Engineering
// ---------------------------------------------------------------------------

/// Build trading state features from kline data.
pub fn build_state(klines: &[Kline], index: usize, lookback: usize) -> Option<Vec<f64>> {
    if index < lookback || index >= klines.len() {
        return None;
    }

    let mut features = Vec::new();

    // Price return
    let ret = (klines[index].close - klines[index - 1].close) / klines[index - 1].close;
    features.push(ret);

    // Log return
    features.push((klines[index].close / klines[index - 1].close).ln());

    // Moving average ratio (close / SMA)
    let sma: f64 = klines[index - lookback..index]
        .iter()
        .map(|k| k.close)
        .sum::<f64>()
        / lookback as f64;
    features.push(klines[index].close / sma - 1.0);

    // Volume ratio (current / average)
    let avg_vol: f64 = klines[index - lookback..index]
        .iter()
        .map(|k| k.volume)
        .sum::<f64>()
        / lookback as f64;
    if avg_vol > 0.0 {
        features.push(klines[index].volume / avg_vol - 1.0);
    } else {
        features.push(0.0);
    }

    // Volatility (rolling std of returns)
    let returns: Vec<f64> = (index - lookback + 1..=index)
        .map(|i| (klines[i].close - klines[i - 1].close) / klines[i - 1].close)
        .collect();
    let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / returns.len() as f64;
    features.push(variance.sqrt());

    Some(features)
}

/// Build transitions from kline data for training.
pub fn build_transitions(klines: &[Kline], lookback: usize) -> Vec<Transition> {
    let mut transitions = Vec::new();

    for i in lookback..klines.len() - 1 {
        let state = match build_state(klines, i, lookback) {
            Some(s) => s,
            None => continue,
        };
        let next_state = match build_state(klines, i + 1, lookback) {
            Some(s) => s,
            None => continue,
        };

        // Compute reward for each action
        let price_change = (klines[i + 1].close - klines[i].close) / klines[i].close;

        for &action in Action::all() {
            let reward = match action {
                Action::Buy => price_change - 0.0004,  // transaction cost
                Action::Hold => 0.0,
                Action::Sell => -price_change - 0.0004,
            };

            transitions.push(Transition {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
            });
        }
    }

    transitions
}

/// Generate synthetic kline data for testing.
pub fn generate_synthetic_klines(n: usize, initial_price: f64) -> Vec<Kline> {
    let mut rng = rand::thread_rng();
    let mut klines = Vec::with_capacity(n);
    let mut price = initial_price;

    for i in 0..n {
        let change = rng.gen_range(-0.02..0.02);
        let open = price;
        let close = price * (1.0 + change);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..1000.0);

        klines.push(Kline {
            timestamp: 1_700_000_000 + (i as u64) * 3600,
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    klines
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_model_predict_shape() {
        let model = WorldModel::new(5, 0.01);
        let state = vec![0.1, -0.05, 0.02, 0.5, 0.01];
        let next = model.predict_next_state(&state, Action::Buy);
        assert_eq!(next.len(), 5);
        let reward = model.predict_reward(&state, Action::Hold);
        assert!(reward.is_finite());
    }

    #[test]
    fn test_world_model_training_reduces_error() {
        let mut model = WorldModel::new(3, 0.01);
        let transitions = vec![
            Transition {
                state: vec![1.0, 0.0, 0.5],
                action: Action::Buy,
                reward: 0.1,
                next_state: vec![1.1, 0.05, 0.6],
            },
            Transition {
                state: vec![1.1, 0.05, 0.6],
                action: Action::Hold,
                reward: 0.0,
                next_state: vec![1.1, 0.02, 0.55],
            },
        ];

        let (err_before, _) = model.evaluate(&transitions);
        model.train(&transitions, 100);
        let (err_after, _) = model.evaluate(&transitions);

        assert!(
            err_after < err_before,
            "Training should reduce error: before={}, after={}",
            err_before,
            err_after
        );
    }

    #[test]
    fn test_ensemble_uncertainty() {
        let ensemble = ModelEnsemble::new(5, 3, 0.01);
        let state = vec![0.1, -0.05, 0.02];
        let unc = ensemble.uncertainty(&state, Action::Buy);
        // With random init, ensemble members disagree -> uncertainty > 0
        assert!(unc >= 0.0, "Uncertainty should be non-negative");
    }

    #[test]
    fn test_dynaq_agent_learns() {
        let mut agent = DynaQAgent::new(3, 0.1, 0.99, 0.1, 5);

        // Feed transitions that reward buying when state[0] > 0
        for _ in 0..50 {
            let t = Transition {
                state: vec![0.5, 0.1, 0.2],
                action: Action::Buy,
                reward: 1.0,
                next_state: vec![0.6, 0.15, 0.25],
            };
            agent.step(t);
        }

        // Agent should have learned something
        assert!(agent.q_table_size() > 0);
        assert_eq!(agent.buffer_size(), 50);

        // The Q-value for Buy in that state should be positive
        let q = agent.get_q(&[0.5, 0.1, 0.2]);
        assert!(q[Action::Buy.index()] > 0.0);
    }

    #[test]
    fn test_mpc_random_shooting() {
        let ensemble = ModelEnsemble::new(3, 3, 0.01);
        let planner = MPCPlanner::new(
            ensemble,
            5,
            50,
            0.99,
            PlanningMethod::RandomShooting,
            1e6, // high threshold so it doesn't fall back to Hold
        );

        let state = vec![0.1, -0.05, 0.02];
        let (action, reward) = planner.plan(&state);
        // Just check it returns a valid action
        assert!(
            action == Action::Buy || action == Action::Hold || action == Action::Sell
        );
        assert!(reward.is_finite());
    }

    #[test]
    fn test_mpc_cem() {
        let ensemble = ModelEnsemble::new(3, 3, 0.01);
        let planner = MPCPlanner::new(
            ensemble,
            5,
            50,
            0.99,
            PlanningMethod::CEM {
                elite_frac: 0.2,
                iterations: 3,
            },
            1e6,
        );

        let state = vec![0.1, -0.05, 0.02];
        let (action, reward) = planner.plan(&state);
        assert!(
            action == Action::Buy || action == Action::Hold || action == Action::Sell
        );
        assert!(reward.is_finite());
    }

    #[test]
    fn test_build_state_features() {
        let klines = generate_synthetic_klines(30, 50000.0);
        let state = build_state(&klines, 15, 10);
        assert!(state.is_some());
        let features = state.unwrap();
        assert_eq!(features.len(), 5);
        for f in &features {
            assert!(f.is_finite(), "Feature should be finite: {}", f);
        }
    }

    #[test]
    fn test_build_transitions() {
        let klines = generate_synthetic_klines(30, 50000.0);
        let transitions = build_transitions(&klines, 10);
        // We should get (30 - 10 - 1) * 3 = 57 transitions (3 actions per timestep)
        assert!(!transitions.is_empty());
        for t in &transitions {
            assert_eq!(t.state.len(), 5);
            assert_eq!(t.next_state.len(), 5);
            assert!(t.reward.is_finite());
        }
    }

    #[test]
    fn test_action_roundtrip() {
        for &a in Action::all() {
            assert_eq!(Action::from_index(a.index()), a);
        }
    }

    #[test]
    fn test_uncertainty_triggers_hold() {
        // Create ensemble with high variance (different random seeds)
        let ensemble = ModelEnsemble::new(5, 3, 0.01);
        let planner = MPCPlanner::new(
            ensemble,
            5,
            50,
            0.99,
            PlanningMethod::RandomShooting,
            0.0, // zero threshold -> always uncertain
        );

        let state = vec![0.1, -0.05, 0.02];
        let (action, _) = planner.plan(&state);
        // With zero threshold, should always return Hold
        assert_eq!(action, Action::Hold);
    }
}
