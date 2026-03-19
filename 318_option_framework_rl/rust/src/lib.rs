//! # Option Framework RL for Trading
//!
//! Implementation of the Options Framework for hierarchical reinforcement learning
//! applied to algorithmic trading. Based on the semi-MDP formulation where temporal
//! abstractions (options) enable multi-timescale decision making.
//!
//! Key concepts:
//! - **Option**: A triple (I, π, β) — initiation set, intra-option policy, termination condition
//! - **Semi-MDP**: Extension of MDP allowing variable-duration actions (options)
//! - **Option-Critic**: Architecture that learns both options and a policy over options
//! - **Trading Options**: TrendFollow, MeanRevert, Hold as macro-level strategies

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Market state representation
// ---------------------------------------------------------------------------

/// Discretized market state used by the option framework.
/// Combines price momentum, volatility regime, and volume signal into a single index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MarketState {
    /// Momentum bucket: 0=strong_down, 1=down, 2=neutral, 3=up, 4=strong_up
    pub momentum: usize,
    /// Volatility bucket: 0=low, 1=medium, 2=high
    pub volatility: usize,
    /// Volume bucket: 0=low, 1=normal, 2=high
    pub volume: usize,
}

impl MarketState {
    pub const MOMENTUM_BINS: usize = 5;
    pub const VOLATILITY_BINS: usize = 3;
    pub const VOLUME_BINS: usize = 3;
    pub const TOTAL_STATES: usize =
        Self::MOMENTUM_BINS * Self::VOLATILITY_BINS * Self::VOLUME_BINS; // 45

    /// Flatten state to a single index for table lookups.
    pub fn to_index(&self) -> usize {
        self.momentum * Self::VOLATILITY_BINS * Self::VOLUME_BINS
            + self.volatility * Self::VOLUME_BINS
            + self.volume
    }

    /// Build a `MarketState` from raw feature values.
    pub fn from_features(returns: f64, vol: f64, vol_ratio: f64) -> Self {
        let momentum = if returns < -0.02 {
            0
        } else if returns < -0.005 {
            1
        } else if returns < 0.005 {
            2
        } else if returns < 0.02 {
            3
        } else {
            4
        };

        let volatility = if vol < 0.01 {
            0
        } else if vol < 0.03 {
            1
        } else {
            2
        };

        let volume = if vol_ratio < 0.8 {
            0
        } else if vol_ratio < 1.2 {
            1
        } else {
            2
        };

        MarketState {
            momentum,
            volatility,
            volume,
        }
    }
}

// ---------------------------------------------------------------------------
// Primitive actions
// ---------------------------------------------------------------------------

/// Low-level (primitive) trading actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveAction {
    Buy,
    Sell,
    Hold,
}

impl PrimitiveAction {
    pub const COUNT: usize = 3;

    pub fn to_index(self) -> usize {
        match self {
            PrimitiveAction::Buy => 0,
            PrimitiveAction::Sell => 1,
            PrimitiveAction::Hold => 2,
        }
    }

    pub fn from_index(i: usize) -> Self {
        match i {
            0 => PrimitiveAction::Buy,
            1 => PrimitiveAction::Sell,
            _ => PrimitiveAction::Hold,
        }
    }
}

// ---------------------------------------------------------------------------
// Trading options (temporal abstractions)
// ---------------------------------------------------------------------------

/// Identifiers for the high-level trading options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TradingOptionKind {
    TrendFollow,
    MeanRevert,
    Hold,
}

impl TradingOptionKind {
    pub const COUNT: usize = 3;

    pub fn to_index(self) -> usize {
        match self {
            TradingOptionKind::TrendFollow => 0,
            TradingOptionKind::MeanRevert => 1,
            TradingOptionKind::Hold => 2,
        }
    }

    pub fn from_index(i: usize) -> Self {
        match i {
            0 => TradingOptionKind::TrendFollow,
            1 => TradingOptionKind::MeanRevert,
            _ => TradingOptionKind::Hold,
        }
    }

    pub fn all() -> [TradingOptionKind; 3] {
        [
            TradingOptionKind::TrendFollow,
            TradingOptionKind::MeanRevert,
            TradingOptionKind::Hold,
        ]
    }
}

// ---------------------------------------------------------------------------
// Option struct — (I, π, β)
// ---------------------------------------------------------------------------

/// A single **option** in the Options Framework sense.
///
/// An option is defined by:
/// - `initiation_set`: which states the option can be started in
/// - `intra_option_policy`: π_o(a|s) — action probabilities inside the option
/// - `termination_prob`: β_o(s) — probability of terminating in each state
#[derive(Debug, Clone)]
pub struct TradingOption {
    pub kind: TradingOptionKind,
    /// For each state index, whether the option can be initiated.
    pub initiation_set: Vec<bool>,
    /// π_o(a|s): for each state, a distribution over primitive actions.
    /// Shape conceptually [num_states][num_actions].
    pub intra_option_policy: Vec<[f64; PrimitiveAction::COUNT]>,
    /// β_o(s): termination probability per state.
    pub termination_prob: Vec<f64>,
}

impl TradingOption {
    /// Create a new option with uniform policy and given termination probability.
    pub fn new(kind: TradingOptionKind, num_states: usize, default_term_prob: f64) -> Self {
        let uniform = [1.0 / PrimitiveAction::COUNT as f64; PrimitiveAction::COUNT];
        TradingOption {
            kind,
            initiation_set: vec![true; num_states],
            intra_option_policy: vec![uniform; num_states],
            termination_prob: vec![default_term_prob; num_states],
        }
    }

    /// Create a TrendFollow option: biased toward Buy in uptrends, Sell in downtrends.
    pub fn trend_follow(num_states: usize) -> Self {
        let mut opt = Self::new(TradingOptionKind::TrendFollow, num_states, 0.2);
        // Set intra-option policy based on momentum component of state
        for s in 0..num_states {
            let momentum = s / (MarketState::VOLATILITY_BINS * MarketState::VOLUME_BINS);
            opt.intra_option_policy[s] = match momentum {
                0 | 1 => [0.1, 0.7, 0.2], // down → sell
                2 => [0.2, 0.2, 0.6],      // neutral → hold
                _ => [0.7, 0.1, 0.2],      // up → buy
            };
            // Terminate more often in neutral markets (trend unclear)
            opt.termination_prob[s] = if momentum == 2 { 0.5 } else { 0.15 };
        }
        // Initiation: prefer trending markets
        for s in 0..num_states {
            let momentum = s / (MarketState::VOLATILITY_BINS * MarketState::VOLUME_BINS);
            opt.initiation_set[s] = momentum != 2; // don't initiate in neutral
        }
        opt
    }

    /// Create a MeanRevert option: buy dips, sell rallies.
    pub fn mean_revert(num_states: usize) -> Self {
        let mut opt = Self::new(TradingOptionKind::MeanRevert, num_states, 0.3);
        for s in 0..num_states {
            let momentum = s / (MarketState::VOLATILITY_BINS * MarketState::VOLUME_BINS);
            opt.intra_option_policy[s] = match momentum {
                0 | 1 => [0.7, 0.1, 0.2], // down → buy (revert)
                2 => [0.2, 0.2, 0.6],      // neutral → hold
                _ => [0.1, 0.7, 0.2],      // up → sell (revert)
            };
            // Terminate when market returns to neutral
            opt.termination_prob[s] = if momentum == 2 { 0.6 } else { 0.2 };
        }
        // Initiation: prefer extreme states
        for s in 0..num_states {
            let momentum = s / (MarketState::VOLATILITY_BINS * MarketState::VOLUME_BINS);
            opt.initiation_set[s] = momentum == 0 || momentum == 4;
        }
        opt
    }

    /// Create a Hold option: mostly hold, low termination (long duration).
    pub fn hold(num_states: usize) -> Self {
        let mut opt = Self::new(TradingOptionKind::Hold, num_states, 0.1);
        for s in 0..num_states {
            opt.intra_option_policy[s] = [0.05, 0.05, 0.9]; // strongly prefer hold
        }
        opt
    }

    /// Check if this option can be initiated in the given state.
    pub fn can_initiate(&self, state_idx: usize) -> bool {
        self.initiation_set.get(state_idx).copied().unwrap_or(false)
    }

    /// Sample a primitive action from the intra-option policy.
    pub fn sample_action(&self, state_idx: usize, rng: &mut impl Rng) -> PrimitiveAction {
        let probs = &self.intra_option_policy[state_idx];
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return PrimitiveAction::from_index(i);
            }
        }
        PrimitiveAction::Hold
    }

    /// Sample whether the option terminates in the given state.
    pub fn should_terminate(&self, state_idx: usize, rng: &mut impl Rng) -> bool {
        let beta = self.termination_prob.get(state_idx).copied().unwrap_or(1.0);
        rng.gen::<f64>() < beta
    }
}

// ---------------------------------------------------------------------------
// Semi-MDP trading environment
// ---------------------------------------------------------------------------

/// A simplified trading environment that operates as a semi-MDP.
/// State transitions happen at variable timescales depending on option duration.
#[derive(Debug)]
pub struct SemiMdpEnv {
    /// Historical price series
    pub prices: Vec<f64>,
    /// Historical volume series
    pub volumes: Vec<f64>,
    /// Current time index
    pub time: usize,
    /// Current position: -1.0 (short), 0.0 (flat), 1.0 (long)
    pub position: f64,
    /// Accumulated PnL
    pub pnl: f64,
    /// Lookback window for feature computation
    pub lookback: usize,
}

impl SemiMdpEnv {
    pub fn new(prices: Vec<f64>, volumes: Vec<f64>, lookback: usize) -> Self {
        SemiMdpEnv {
            prices,
            volumes,
            time: lookback,
            position: 0.0,
            pnl: 0.0,
            lookback,
        }
    }

    /// Reset environment to initial state.
    pub fn reset(&mut self) -> MarketState {
        self.time = self.lookback;
        self.position = 0.0;
        self.pnl = 0.0;
        self.observe()
    }

    /// Compute current market state from price/volume history.
    pub fn observe(&self) -> MarketState {
        if self.time < self.lookback || self.time >= self.prices.len() {
            return MarketState {
                momentum: 2,
                volatility: 1,
                volume: 1,
            };
        }

        let window = &self.prices[self.time - self.lookback..self.time];
        let returns = (window.last().unwrap() / window.first().unwrap()) - 1.0;

        // Simple volatility: std of log returns
        let log_rets: Vec<f64> = window
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
        let mean_ret = log_rets.iter().sum::<f64>() / log_rets.len() as f64;
        let vol = (log_rets.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
            / log_rets.len() as f64)
            .sqrt();

        // Volume ratio vs average
        let vol_window = &self.volumes[self.time - self.lookback..self.time];
        let avg_vol = vol_window.iter().sum::<f64>() / vol_window.len() as f64;
        let vol_ratio = if avg_vol > 0.0 {
            self.volumes[self.time - 1] / avg_vol
        } else {
            1.0
        };

        MarketState::from_features(returns, vol, vol_ratio)
    }

    /// Execute a primitive action and advance one step.
    /// Returns (next_state, reward, done).
    pub fn step(&mut self, action: PrimitiveAction) -> (MarketState, f64, bool) {
        if self.time >= self.prices.len() - 1 {
            return (self.observe(), 0.0, true);
        }

        let price_now = self.prices[self.time];
        let price_next = self.prices[self.time + 1];
        let price_return = (price_next - price_now) / price_now;

        // Update position
        let new_position = match action {
            PrimitiveAction::Buy => 1.0,
            PrimitiveAction::Sell => -1.0,
            PrimitiveAction::Hold => self.position,
        };

        // Transaction cost for changing position
        let transaction_cost = if (new_position - self.position).abs() > 0.5 {
            0.001 * (new_position - self.position).abs() // 10bp per unit change
        } else {
            0.0
        };

        // PnL from current position
        let step_pnl = self.position * price_return - transaction_cost;
        self.pnl += step_pnl;
        self.position = new_position;
        self.time += 1;

        let done = self.time >= self.prices.len() - 1;
        let next_state = self.observe();

        (next_state, step_pnl, done)
    }

    /// Execute a full option until termination, accumulating rewards.
    /// Returns (final_state, cumulative_reward, steps_taken, done).
    pub fn execute_option(
        &mut self,
        option: &TradingOption,
        gamma: f64,
        rng: &mut impl Rng,
    ) -> (MarketState, f64, usize, bool) {
        let mut cumulative_reward = 0.0;
        let mut discount = 1.0;
        let mut steps = 0;
        loop {
            let state = self.observe();
            let state_idx = state.to_index();

            // Sample action from intra-option policy
            let action = option.sample_action(state_idx, rng);

            // Take step
            let (next_state, reward, step_done) = self.step(action);
            cumulative_reward += discount * reward;
            discount *= gamma;
            steps += 1;

            if step_done {
                break;
            }

            // Check termination
            let next_idx = next_state.to_index();
            if option.should_terminate(next_idx, rng) {
                break;
            }

            // Safety: limit option duration
            if steps >= 50 {
                break;
            }
        }

        let done = self.is_done();
        (self.observe(), cumulative_reward, steps, done)
    }

    pub fn is_done(&self) -> bool {
        self.time >= self.prices.len() - 1
    }
}

// ---------------------------------------------------------------------------
// Intra-Option Q-Learning
// ---------------------------------------------------------------------------

/// Intra-option Q-learning with option-level value function.
///
/// Maintains Q(s, o) — the value of executing option o in state s,
/// and updates it using the semi-MDP temporal difference.
#[derive(Debug, Clone)]
pub struct IntraOptionQLearning {
    /// Q(s, o): value of option o in state s.
    /// Shape: [num_states][num_options]
    pub q_values: Vec<[f64; TradingOptionKind::COUNT]>,
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub num_states: usize,
}

impl IntraOptionQLearning {
    pub fn new(num_states: usize, alpha: f64, gamma: f64, epsilon: f64) -> Self {
        IntraOptionQLearning {
            q_values: vec![[0.0; TradingOptionKind::COUNT]; num_states],
            alpha,
            gamma,
            epsilon,
            num_states,
        }
    }

    /// Select an option using epsilon-greedy over Q(s, o).
    /// Only considers options whose initiation set includes the current state.
    pub fn select_option(
        &self,
        state_idx: usize,
        options: &[TradingOption],
        rng: &mut impl Rng,
    ) -> usize {
        // Get available options
        let available: Vec<usize> = options
            .iter()
            .enumerate()
            .filter(|(_, o)| o.can_initiate(state_idx))
            .map(|(i, _)| i)
            .collect();

        if available.is_empty() {
            return 2; // default to Hold
        }

        if rng.gen::<f64>() < self.epsilon {
            // Random exploration
            available[rng.gen_range(0..available.len())]
        } else {
            // Greedy
            *available
                .iter()
                .max_by(|&&a, &&b| {
                    self.q_values[state_idx][a]
                        .partial_cmp(&self.q_values[state_idx][b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap()
        }
    }

    /// Update Q(s, o) after option o executed for k steps with cumulative reward R.
    pub fn update(
        &mut self,
        state_idx: usize,
        option_idx: usize,
        next_state_idx: usize,
        cumulative_reward: f64,
        steps: usize,
    ) {
        let gamma_k = self.gamma.powi(steps as i32);
        let max_next_q = self.q_values[next_state_idx]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let td_target = cumulative_reward + gamma_k * max_next_q;
        let td_error = td_target - self.q_values[state_idx][option_idx];
        self.q_values[state_idx][option_idx] += self.alpha * td_error;
    }

    /// Get the best option index for a given state.
    pub fn best_option(&self, state_idx: usize) -> usize {
        self.q_values[state_idx]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(2)
    }
}

// ---------------------------------------------------------------------------
// Option-Critic Architecture (simplified tabular version)
// ---------------------------------------------------------------------------

/// Simplified tabular Option-Critic that jointly learns:
/// - Policy over options: Q_Ω(s, o)
/// - Intra-option policies: π_o(a|s) via policy gradient
/// - Termination functions: β_o(s) via termination gradient
#[derive(Debug, Clone)]
pub struct OptionCritic {
    /// Q_Ω(s, o): option-value function
    pub q_omega: Vec<[f64; TradingOptionKind::COUNT]>,
    /// Q_U(s, o, a): intra-option action-value function
    pub q_u: Vec<[[f64; PrimitiveAction::COUNT]; TradingOptionKind::COUNT]>,
    /// Termination parameters (logits) per (state, option)
    pub termination_logits: Vec<[f64; TradingOptionKind::COUNT]>,
    pub alpha_critic: f64,
    pub alpha_intra: f64,
    pub alpha_term: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub num_states: usize,
}

impl OptionCritic {
    pub fn new(
        num_states: usize,
        alpha_critic: f64,
        alpha_intra: f64,
        alpha_term: f64,
        gamma: f64,
        epsilon: f64,
    ) -> Self {
        OptionCritic {
            q_omega: vec![[0.0; TradingOptionKind::COUNT]; num_states],
            q_u: vec![[[0.0; PrimitiveAction::COUNT]; TradingOptionKind::COUNT]; num_states],
            termination_logits: vec![[0.0; TradingOptionKind::COUNT]; num_states],
            alpha_critic,
            alpha_intra,
            alpha_term,
            gamma,
            epsilon,
            num_states,
        }
    }

    /// Sigmoid function for termination probability.
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Get termination probability β_o(s).
    pub fn termination_prob(&self, state_idx: usize, option_idx: usize) -> f64 {
        Self::sigmoid(self.termination_logits[state_idx][option_idx])
    }

    /// Select option using epsilon-greedy over Q_Ω.
    pub fn select_option(&self, state_idx: usize, rng: &mut impl Rng) -> usize {
        if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..TradingOptionKind::COUNT)
        } else {
            self.q_omega[state_idx]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }

    /// Select primitive action from intra-option policy (softmax over Q_U).
    pub fn select_action(
        &self,
        state_idx: usize,
        option_idx: usize,
        rng: &mut impl Rng,
    ) -> PrimitiveAction {
        let q_vals = &self.q_u[state_idx][option_idx];
        // Softmax with temperature
        let temperature = 0.5;
        let max_q = q_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = q_vals.iter().map(|q| ((q - max_q) / temperature).exp()).collect();
        let sum_exp: f64 = exp_vals.iter().sum();
        let probs: Vec<f64> = exp_vals.iter().map(|e| e / sum_exp).collect();

        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return PrimitiveAction::from_index(i);
            }
        }
        PrimitiveAction::Hold
    }

    /// Perform one-step update of all components.
    ///
    /// Given transition (s, o, a, r, s'), update:
    /// 1. Q_U(s, o, a) — intra-option critic
    /// 2. Q_Ω(s, o) — option-level critic
    /// 3. β_o(s') — termination function
    pub fn update(
        &mut self,
        state_idx: usize,
        option_idx: usize,
        action_idx: usize,
        reward: f64,
        next_state_idx: usize,
        done: bool,
    ) {
        if done {
            // Terminal update
            let td_error_u =
                reward - self.q_u[state_idx][option_idx][action_idx];
            self.q_u[state_idx][option_idx][action_idx] +=
                self.alpha_intra * td_error_u;

            let td_error_omega = reward - self.q_omega[state_idx][option_idx];
            self.q_omega[state_idx][option_idx] += self.alpha_critic * td_error_omega;
            return;
        }

        // Compute U(o, s') = (1 - β_o(s')) * Q_Ω(s', o) + β_o(s') * max_o' Q_Ω(s', o')
        let beta = self.termination_prob(next_state_idx, option_idx);
        let max_q_next = self.q_omega[next_state_idx]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let u_value = (1.0 - beta) * self.q_omega[next_state_idx][option_idx]
            + beta * max_q_next;

        // Update Q_U (intra-option critic)
        let td_target_u = reward + self.gamma * u_value;
        let td_error_u = td_target_u - self.q_u[state_idx][option_idx][action_idx];
        self.q_u[state_idx][option_idx][action_idx] += self.alpha_intra * td_error_u;

        // Update Q_Ω (option-level critic)
        let td_target_omega = reward + self.gamma * u_value;
        let td_error_omega = td_target_omega - self.q_omega[state_idx][option_idx];
        self.q_omega[state_idx][option_idx] += self.alpha_critic * td_error_omega;

        // Update termination: gradient of advantage A_Ω(s', o) = Q_Ω(s', o) - V_Ω(s')
        let advantage =
            self.q_omega[next_state_idx][option_idx] - max_q_next;
        // Gradient: ∂β/∂logit = β(1-β), direction: -advantage (terminate if option is bad)
        let beta_grad = beta * (1.0 - beta);
        self.termination_logits[next_state_idx][option_idx] -=
            self.alpha_term * advantage * beta_grad;
    }

    /// Run one full episode of option-critic training.
    pub fn train_episode(
        &mut self,
        env: &mut SemiMdpEnv,
        rng: &mut impl Rng,
    ) -> f64 {
        let state = env.reset();
        let mut state_idx = state.to_index();
        let mut current_option = self.select_option(state_idx, rng);
        let mut total_reward = 0.0;

        loop {
            let action = self.select_action(state_idx, current_option, rng);
            let (next_state, reward, done) = env.step(action);
            let next_state_idx = next_state.to_index();

            self.update(
                state_idx,
                current_option,
                action.to_index(),
                reward,
                next_state_idx,
                done,
            );

            total_reward += reward;

            if done {
                break;
            }

            // Check termination and possibly switch option
            let should_terminate =
                rng.gen::<f64>() < self.termination_prob(next_state_idx, current_option);
            if should_terminate {
                current_option = self.select_option(next_state_idx, rng);
            }

            state_idx = next_state_idx;
        }

        total_reward
    }
}

// ---------------------------------------------------------------------------
// Training statistics
// ---------------------------------------------------------------------------

/// Tracks performance during training.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingStats {
    pub episode_rewards: Vec<f64>,
    pub option_selections: HashMap<String, usize>,
}

impl TrainingStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_reward(&mut self, reward: f64) {
        self.episode_rewards.push(reward);
    }

    pub fn record_option(&mut self, kind: TradingOptionKind) {
        let key = format!("{:?}", kind);
        *self.option_selections.entry(key).or_insert(0) += 1;
    }

    pub fn mean_reward(&self, last_n: usize) -> f64 {
        let n = last_n.min(self.episode_rewards.len());
        if n == 0 {
            return 0.0;
        }
        let start = self.episode_rewards.len() - n;
        self.episode_rewards[start..].iter().sum::<f64>() / n as f64
    }
}

// ---------------------------------------------------------------------------
// Bybit API integration
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

/// Response envelope from Bybit v5 API.
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

/// Fetch kline data from Bybit API (blocking).
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<Kline>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let mut klines: Vec<Kline> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() < 6 {
                return None;
            }
            Some(Kline {
                open_time: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();

    // Bybit returns newest first; reverse to chronological order
    klines.reverse();
    Ok(klines)
}

/// Convert klines to price and volume vectors.
pub fn klines_to_series(klines: &[Kline]) -> (Vec<f64>, Vec<f64>) {
    let prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();
    (prices, volumes)
}

/// Generate synthetic price data for testing/offline use.
pub fn generate_synthetic_prices(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut price = 50000.0_f64;
    let mut prices = Vec::with_capacity(n);
    let mut volumes = Vec::with_capacity(n);

    // Generate with regime changes
    for i in 0..n {
        let regime = (i / 100) % 3; // cycle every 300 steps
        let drift = match regime {
            0 => 0.001,  // uptrend
            1 => -0.001, // downtrend
            _ => 0.0,    // sideways
        };
        let noise: f64 = rng.gen_range(-0.02..0.02);
        price *= 1.0 + drift + noise;
        prices.push(price);
        volumes.push(rng.gen_range(100.0..1000.0));
    }

    (prices, volumes)
}

// ---------------------------------------------------------------------------
// Utility: compute Sharpe ratio
// ---------------------------------------------------------------------------

/// Compute annualized Sharpe ratio from a series of returns.
pub fn sharpe_ratio(returns: &[f64], periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std = var.sqrt();
    if std < 1e-10 {
        return 0.0;
    }
    (mean / std) * periods_per_year.sqrt()
}

/// Compute maximum drawdown from cumulative PnL series.
pub fn max_drawdown(equity_curve: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0;
    for &val in equity_curve {
        if val > peak {
            peak = val;
        }
        let dd = (peak - val) / peak.abs().max(1e-10);
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_market_state_from_features() {
        let s = MarketState::from_features(-0.03, 0.005, 1.0);
        assert_eq!(s.momentum, 0); // strong down
        assert_eq!(s.volatility, 0); // low vol
        assert_eq!(s.volume, 1); // normal volume

        let s2 = MarketState::from_features(0.03, 0.05, 1.5);
        assert_eq!(s2.momentum, 4); // strong up
        assert_eq!(s2.volatility, 2); // high vol
        assert_eq!(s2.volume, 2); // high volume
    }

    #[test]
    fn test_state_index_roundtrip() {
        for m in 0..MarketState::MOMENTUM_BINS {
            for v in 0..MarketState::VOLATILITY_BINS {
                for vol in 0..MarketState::VOLUME_BINS {
                    let s = MarketState {
                        momentum: m,
                        volatility: v,
                        volume: vol,
                    };
                    let idx = s.to_index();
                    assert!(idx < MarketState::TOTAL_STATES);
                }
            }
        }
        // Check total unique indices
        let mut indices: Vec<usize> = Vec::new();
        for m in 0..MarketState::MOMENTUM_BINS {
            for v in 0..MarketState::VOLATILITY_BINS {
                for vol in 0..MarketState::VOLUME_BINS {
                    let s = MarketState {
                        momentum: m,
                        volatility: v,
                        volume: vol,
                    };
                    indices.push(s.to_index());
                }
            }
        }
        indices.sort();
        indices.dedup();
        assert_eq!(indices.len(), MarketState::TOTAL_STATES);
    }

    #[test]
    fn test_trading_option_policies() {
        let n = MarketState::TOTAL_STATES;
        let tf = TradingOption::trend_follow(n);
        let mr = TradingOption::mean_revert(n);
        let hold = TradingOption::hold(n);

        // TrendFollow in strong up state should favor Buy
        let up_state = MarketState {
            momentum: 4,
            volatility: 1,
            volume: 1,
        };
        let idx = up_state.to_index();
        assert!(tf.intra_option_policy[idx][0] > 0.5); // Buy prob high

        // MeanRevert in strong up state should favor Sell
        assert!(mr.intra_option_policy[idx][1] > 0.5); // Sell prob high

        // Hold should always strongly prefer Hold action
        assert!(hold.intra_option_policy[idx][2] > 0.8);
    }

    #[test]
    fn test_semi_mdp_env_step() {
        let (prices, volumes) = generate_synthetic_prices(200, 42);
        let mut env = SemiMdpEnv::new(prices, volumes, 20);
        let initial = env.reset();
        assert!(initial.to_index() < MarketState::TOTAL_STATES);

        // Take some steps
        let (s1, r1, done1) = env.step(PrimitiveAction::Buy);
        assert!(!done1);
        assert!(s1.to_index() < MarketState::TOTAL_STATES);

        let (s2, _r2, done2) = env.step(PrimitiveAction::Hold);
        assert!(!done2);
        let _ = s2;
        let _ = r1;
    }

    #[test]
    fn test_option_execution() {
        let (prices, volumes) = generate_synthetic_prices(200, 42);
        let mut env = SemiMdpEnv::new(prices, volumes, 20);
        env.reset();

        let option = TradingOption::trend_follow(MarketState::TOTAL_STATES);
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);

        let (_next_state, cum_reward, steps, _done) =
            env.execute_option(&option, 0.99, &mut rng);

        assert!(steps >= 1);
        assert!(steps <= 50);
        // cum_reward can be positive or negative
        let _ = cum_reward;
    }

    #[test]
    fn test_intra_option_q_learning() {
        let n = MarketState::TOTAL_STATES;
        let mut learner = IntraOptionQLearning::new(n, 0.1, 0.99, 0.3);
        let options = vec![
            TradingOption::trend_follow(n),
            TradingOption::mean_revert(n),
            TradingOption::hold(n),
        ];

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Simulate some updates
        let state_idx = 10;
        let opt_idx = learner.select_option(state_idx, &options, &mut rng);
        assert!(opt_idx < TradingOptionKind::COUNT);

        // Update with positive reward
        learner.update(state_idx, opt_idx, 15, 0.05, 3);
        assert!(learner.q_values[state_idx][opt_idx] > 0.0);

        // Update same option with negative reward
        learner.update(state_idx, opt_idx, 15, -0.1, 2);
        // Q-value should have decreased
    }

    #[test]
    fn test_option_critic_training() {
        let (prices, volumes) = generate_synthetic_prices(300, 42);
        let mut env = SemiMdpEnv::new(prices, volumes, 20);
        let mut critic = OptionCritic::new(
            MarketState::TOTAL_STATES,
            0.1,  // alpha_critic
            0.1,  // alpha_intra
            0.01, // alpha_term
            0.99, // gamma
            0.3,  // epsilon
        );

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Train for a few episodes
        let mut rewards = Vec::new();
        for _ in 0..5 {
            let r = critic.train_episode(&mut env, &mut rng);
            rewards.push(r);
        }

        assert_eq!(rewards.len(), 5);
        // Verify Q-values are no longer all zero
        let has_nonzero = critic
            .q_omega
            .iter()
            .any(|row| row.iter().any(|&v| v.abs() > 1e-10));
        assert!(has_nonzero);
    }

    #[test]
    fn test_sharpe_and_drawdown() {
        let returns = vec![0.01, 0.02, -0.005, 0.015, -0.01, 0.03, 0.005];
        let sr = sharpe_ratio(&returns, 252.0);
        assert!(sr > 0.0); // positive on average

        let equity = vec![100.0, 102.0, 105.0, 100.0, 103.0, 108.0];
        let dd = max_drawdown(&equity);
        assert!(dd > 0.0);
        assert!(dd < 1.0);
        // Max drawdown should be around (105 - 100) / 105 ≈ 0.0476
        assert!((dd - 0.0476).abs() < 0.01);
    }

    #[test]
    fn test_synthetic_price_generation() {
        let (prices, volumes) = generate_synthetic_prices(500, 99);
        assert_eq!(prices.len(), 500);
        assert_eq!(volumes.len(), 500);
        assert!(prices.iter().all(|&p| p > 0.0));
        assert!(volumes.iter().all(|&v| v > 0.0));
    }
}
