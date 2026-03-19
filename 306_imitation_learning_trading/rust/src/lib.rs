//! # Imitation Learning for Trading
//!
//! A comparative framework implementing Behavioral Cloning (BC) and
//! Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL) for
//! learning trading policies from expert demonstrations.

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Discrete trading action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradingAction {
    Buy,
    Hold,
    Sell,
}

impl TradingAction {
    /// Convert to numeric index (0, 1, 2).
    pub fn to_index(self) -> usize {
        match self {
            TradingAction::Buy => 0,
            TradingAction::Hold => 1,
            TradingAction::Sell => 2,
        }
    }

    /// Convert from numeric index.
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => TradingAction::Buy,
            1 => TradingAction::Hold,
            _ => TradingAction::Sell,
        }
    }

    pub const NUM_ACTIONS: usize = 3;
}

/// Market state features used as input to policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingState {
    /// Normalised price return over the lookback window.
    pub price_return: f64,
    /// Rolling volatility estimate.
    pub volatility: f64,
    /// RSI-like momentum indicator in [0, 1].
    pub momentum: f64,
    /// Volume ratio (current / average).
    pub volume_ratio: f64,
}

impl TradingState {
    /// Convert to a feature vector.
    pub fn to_features(&self) -> Vec<f64> {
        vec![self.price_return, self.volatility, self.momentum, self.volume_ratio]
    }

    /// Number of features.
    pub const NUM_FEATURES: usize = 4;
}

/// A single expert demonstration (state, action) pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Demonstration {
    pub state: TradingState,
    pub action: TradingAction,
}

/// Evaluation metrics for comparing IL methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    /// Fraction of actions matching the expert.
    pub accuracy: f64,
    /// Cumulative PnL from following the learned policy.
    pub total_pnl: f64,
    /// Sharpe ratio estimate (annualised).
    pub sharpe_ratio: f64,
    /// Maximum drawdown fraction.
    pub max_drawdown: f64,
}

// ---------------------------------------------------------------------------
// Trait: ImitationLearner
// ---------------------------------------------------------------------------

/// Unified interface for all imitation learning methods.
pub trait ImitationLearner {
    /// Train the learner on expert demonstrations.
    fn train(&mut self, demonstrations: &[Demonstration]) -> Result<()>;

    /// Predict an action for the given state.
    fn predict(&self, state: &TradingState) -> TradingAction;

    /// Evaluate the learned policy against held-out demonstrations.
    fn evaluate(&self, test_data: &[Demonstration]) -> EvaluationMetrics {
        if test_data.is_empty() {
            return EvaluationMetrics {
                accuracy: 0.0,
                total_pnl: 0.0,
                sharpe_ratio: 0.0,
                max_drawdown: 0.0,
            };
        }

        let mut correct = 0usize;
        let mut returns: Vec<f64> = Vec::new();
        let mut equity = 1.0_f64;
        let mut peak = 1.0_f64;
        let mut max_dd = 0.0_f64;

        for demo in test_data {
            let predicted = self.predict(&demo.state);
            if predicted == demo.action {
                correct += 1;
            }

            // Simulate PnL: long on Buy, flat on Hold, short on Sell.
            let position = match predicted {
                TradingAction::Buy => 1.0,
                TradingAction::Hold => 0.0,
                TradingAction::Sell => -1.0,
            };
            let step_return = position * demo.state.price_return;
            returns.push(step_return);
            equity *= 1.0 + step_return;
            if equity > peak {
                peak = equity;
            }
            let dd = (peak - equity) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        let accuracy = correct as f64 / test_data.len() as f64;
        let total_pnl = equity - 1.0;
        let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
        let var = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
            / returns.len() as f64;
        let std = var.sqrt();
        let sharpe = if std > 1e-12 {
            (mean_ret / std) * (252.0_f64).sqrt()
        } else {
            0.0
        };

        EvaluationMetrics {
            accuracy,
            total_pnl,
            sharpe_ratio: sharpe,
            max_drawdown: max_dd,
        }
    }
}

// ---------------------------------------------------------------------------
// Behavioral Cloning
// ---------------------------------------------------------------------------

/// Behavioral Cloning: supervised classification from state -> action.
///
/// Internally uses a simple softmax linear model trained via gradient descent.
pub struct BehavioralCloner {
    /// Weight matrix: (NUM_ACTIONS x NUM_FEATURES).
    weights: Array2<f64>,
    /// Bias vector: (NUM_ACTIONS,).
    bias: Array1<f64>,
    /// Learning rate.
    lr: f64,
    /// Number of training epochs.
    epochs: usize,
}

impl BehavioralCloner {
    pub fn new(lr: f64, epochs: usize) -> Self {
        Self {
            weights: Array2::zeros((TradingAction::NUM_ACTIONS, TradingState::NUM_FEATURES)),
            bias: Array1::zeros(TradingAction::NUM_ACTIONS),
            lr,
            epochs,
        }
    }

    /// Compute softmax probabilities for a given feature vector.
    fn softmax(&self, features: &[f64]) -> Vec<f64> {
        let feats = Array1::from_vec(features.to_vec());
        let logits: Vec<f64> = (0..TradingAction::NUM_ACTIONS)
            .map(|a| {
                self.weights.row(a).dot(&feats) + self.bias[a]
            })
            .collect();

        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|l| (l - max_logit).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }
}

impl ImitationLearner for BehavioralCloner {
    fn train(&mut self, demonstrations: &[Demonstration]) -> Result<()> {
        let mut rng = rand::thread_rng();

        // Initialise weights with small random values.
        for w in self.weights.iter_mut() {
            *w = rng.gen_range(-0.01..0.01);
        }

        for _epoch in 0..self.epochs {
            for demo in demonstrations {
                let features = demo.state.to_features();
                let probs = self.softmax(&features);
                let target = demo.action.to_index();

                // Gradient of cross-entropy loss w.r.t. logits.
                let feats = Array1::from_vec(features);
                for a in 0..TradingAction::NUM_ACTIONS {
                    let indicator = if a == target { 1.0 } else { 0.0 };
                    let grad = probs[a] - indicator;
                    for j in 0..TradingState::NUM_FEATURES {
                        self.weights[[a, j]] -= self.lr * grad * feats[j];
                    }
                    self.bias[a] -= self.lr * grad;
                }
            }
        }

        Ok(())
    }

    fn predict(&self, state: &TradingState) -> TradingAction {
        let features = state.to_features();
        let probs = self.softmax(&features);
        let best = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(1);
        TradingAction::from_index(best)
    }
}

// ---------------------------------------------------------------------------
// Maximum Entropy IRL
// ---------------------------------------------------------------------------

/// Maximum Entropy Inverse Reinforcement Learning.
///
/// Recovers a linear reward function R(s,a) = psi^T phi(s,a) such that the
/// induced softmax policy best explains the expert demonstrations.
pub struct MaxEntropyIRL {
    /// Reward weight vector: one weight per (feature x action) combination.
    reward_weights: Vec<f64>,
    /// Learning rate for reward weight updates.
    lr: f64,
    /// Number of iterations for reward learning.
    iterations: usize,
}

impl MaxEntropyIRL {
    pub fn new(lr: f64, iterations: usize) -> Self {
        let dim = TradingState::NUM_FEATURES * TradingAction::NUM_ACTIONS;
        Self {
            reward_weights: vec![0.0; dim],
            lr,
            iterations,
        }
    }

    /// Compute joint feature vector phi(s, a).
    fn joint_features(state: &TradingState, action: TradingAction) -> Vec<f64> {
        let feats = state.to_features();
        let mut phi = vec![0.0; TradingState::NUM_FEATURES * TradingAction::NUM_ACTIONS];
        let offset = action.to_index() * TradingState::NUM_FEATURES;
        for (i, &f) in feats.iter().enumerate() {
            phi[offset + i] = f;
        }
        phi
    }

    /// Compute reward for a (state, action) pair.
    pub fn reward(&self, state: &TradingState, action: TradingAction) -> f64 {
        let phi = Self::joint_features(state, action);
        self.reward_weights
            .iter()
            .zip(phi.iter())
            .map(|(w, f)| w * f)
            .sum()
    }

    /// Softmax policy induced by the current reward weights.
    fn policy_probs(&self, state: &TradingState) -> Vec<f64> {
        let rewards: Vec<f64> = (0..TradingAction::NUM_ACTIONS)
            .map(|a| self.reward(state, TradingAction::from_index(a)))
            .collect();

        let max_r = rewards.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = rewards.iter().map(|r| (r - max_r).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    /// Get the learned reward weights (for interpretability).
    pub fn get_reward_weights(&self) -> &[f64] {
        &self.reward_weights
    }
}

impl ImitationLearner for MaxEntropyIRL {
    fn train(&mut self, demonstrations: &[Demonstration]) -> Result<()> {
        let mut rng = rand::thread_rng();

        // Small random initialisation.
        for w in self.reward_weights.iter_mut() {
            *w = rng.gen_range(-0.01..0.01);
        }

        for _iter in 0..self.iterations {
            // Accumulate gradient: expert features - expected features under policy.
            let dim = self.reward_weights.len();
            let mut grad = vec![0.0; dim];

            for demo in demonstrations {
                // Expert feature counts.
                let expert_phi = Self::joint_features(&demo.state, demo.action);

                // Expected feature counts under current policy.
                let probs = self.policy_probs(&demo.state);
                let mut expected_phi = vec![0.0; dim];
                for a in 0..TradingAction::NUM_ACTIONS {
                    let phi_a = Self::joint_features(&demo.state, TradingAction::from_index(a));
                    for (j, &f) in phi_a.iter().enumerate() {
                        expected_phi[j] += probs[a] * f;
                    }
                }

                // Gradient = expert - expected (ascent on log-likelihood).
                for j in 0..dim {
                    grad[j] += expert_phi[j] - expected_phi[j];
                }
            }

            // Update weights.
            let n = demonstrations.len() as f64;
            for j in 0..dim {
                self.reward_weights[j] += self.lr * grad[j] / n;
            }
        }

        Ok(())
    }

    fn predict(&self, state: &TradingState) -> TradingAction {
        let probs = self.policy_probs(state);
        let best = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(1);
        TradingAction::from_index(best)
    }
}

// ---------------------------------------------------------------------------
// Expert policy simulation
// ---------------------------------------------------------------------------

/// Type of expert trading strategy.
#[derive(Debug, Clone, Copy)]
pub enum ExpertType {
    /// Buys on positive momentum, sells on negative.
    Momentum,
    /// Buys on dips, sells on rallies.
    MeanReversion,
    /// Combines momentum and mean-reversion signals.
    Hybrid,
}

/// Simulates an expert policy for generating demonstrations.
pub struct ExpertPolicy {
    pub expert_type: ExpertType,
    /// Threshold for triggering trades.
    pub threshold: f64,
}

impl ExpertPolicy {
    pub fn new(expert_type: ExpertType, threshold: f64) -> Self {
        Self {
            expert_type,
            threshold,
        }
    }

    /// Determine the expert action for a given state.
    pub fn act(&self, state: &TradingState) -> TradingAction {
        match self.expert_type {
            ExpertType::Momentum => {
                if state.momentum > 0.5 + self.threshold {
                    TradingAction::Buy
                } else if state.momentum < 0.5 - self.threshold {
                    TradingAction::Sell
                } else {
                    TradingAction::Hold
                }
            }
            ExpertType::MeanReversion => {
                // Buy when price is low (negative return) and volatility is high.
                if state.price_return < -self.threshold && state.volatility > 0.01 {
                    TradingAction::Buy
                } else if state.price_return > self.threshold && state.volatility > 0.01 {
                    TradingAction::Sell
                } else {
                    TradingAction::Hold
                }
            }
            ExpertType::Hybrid => {
                let momentum_signal = state.momentum - 0.5;
                let reversion_signal = -state.price_return;
                let combined = 0.6 * momentum_signal + 0.4 * reversion_signal;

                if combined > self.threshold {
                    TradingAction::Buy
                } else if combined < -self.threshold {
                    TradingAction::Sell
                } else {
                    TradingAction::Hold
                }
            }
        }
    }

    /// Generate demonstrations from synthetic or provided market data.
    pub fn generate_demonstrations(&self, states: &[TradingState]) -> Vec<Demonstration> {
        states
            .iter()
            .map(|s| Demonstration {
                state: s.clone(),
                action: self.act(s),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Bybit API client
// ---------------------------------------------------------------------------

/// OHLCV candle from Bybit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API response structures.
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

/// Fetch OHLCV klines from Bybit public API.
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::Client::new();
    let resp: BybitResponse = client.get(&url).send().await?.json().await?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: retCode={}", resp.ret_code);
    }

    let mut candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() < 6 {
                return None;
            }
            Some(Candle {
                timestamp: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();

    // Bybit returns newest first; reverse to chronological order.
    candles.reverse();
    Ok(candles)
}

// ---------------------------------------------------------------------------
// Feature engineering from candles
// ---------------------------------------------------------------------------

/// Convert raw candles into TradingState feature vectors.
pub fn candles_to_states(candles: &[Candle], lookback: usize) -> Vec<TradingState> {
    if candles.len() < lookback + 1 {
        return Vec::new();
    }

    let mut states = Vec::new();
    let avg_volume: f64 = candles.iter().map(|c| c.volume).sum::<f64>() / candles.len() as f64;

    for i in lookback..candles.len() {
        let price_return = (candles[i].close - candles[i - 1].close) / candles[i - 1].close;

        // Rolling volatility.
        let returns: Vec<f64> = (i - lookback + 1..=i)
            .map(|j| (candles[j].close - candles[j - 1].close) / candles[j - 1].close)
            .collect();
        let mean_r = returns.iter().sum::<f64>() / returns.len() as f64;
        let var = returns.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>() / returns.len() as f64;
        let volatility = var.sqrt();

        // Simple RSI-like momentum in [0, 1].
        let gains: f64 = returns.iter().filter(|r| **r > 0.0).sum();
        let losses: f64 = returns.iter().filter(|r| **r < 0.0).map(|r| r.abs()).sum();
        let momentum = if gains + losses > 1e-12 {
            gains / (gains + losses)
        } else {
            0.5
        };

        let volume_ratio = if avg_volume > 1e-12 {
            candles[i].volume / avg_volume
        } else {
            1.0
        };

        states.push(TradingState {
            price_return,
            volatility,
            momentum,
            volume_ratio,
        });
    }

    states
}

/// Generate synthetic market states for testing when API is unavailable.
pub fn generate_synthetic_states(n: usize, seed: u64) -> Vec<TradingState> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    (0..n)
        .map(|_| TradingState {
            price_return: rng.gen_range(-0.05..0.05),
            volatility: rng.gen_range(0.005..0.04),
            momentum: rng.gen_range(0.0..1.0),
            volume_ratio: rng.gen_range(0.5..2.0),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Comparative evaluation
// ---------------------------------------------------------------------------

/// Run a comparative evaluation of BC vs IRL on the same demonstrations.
pub fn compare_methods(
    train_data: &[Demonstration],
    test_data: &[Demonstration],
) -> Result<(EvaluationMetrics, EvaluationMetrics)> {
    // Train Behavioral Cloning.
    let mut bc = BehavioralCloner::new(0.01, 100);
    bc.train(train_data)?;
    let bc_metrics = bc.evaluate(test_data);

    // Train MaxEnt IRL.
    let mut irl = MaxEntropyIRL::new(0.05, 200);
    irl.train(train_data)?;
    let irl_metrics = irl.evaluate(test_data);

    Ok((bc_metrics, irl_metrics))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_states() -> Vec<TradingState> {
        generate_synthetic_states(200, 42)
    }

    fn sample_demos(expert_type: ExpertType) -> Vec<Demonstration> {
        let expert = ExpertPolicy::new(expert_type, 0.05);
        let states = sample_states();
        expert.generate_demonstrations(&states)
    }

    #[test]
    fn test_bc_trains_and_predicts() {
        let demos = sample_demos(ExpertType::Momentum);
        let mut bc = BehavioralCloner::new(0.01, 50);
        bc.train(&demos).unwrap();

        let state = TradingState {
            price_return: 0.02,
            volatility: 0.01,
            momentum: 0.9,
            volume_ratio: 1.2,
        };
        let action = bc.predict(&state);
        // High momentum -> expect Buy.
        assert_eq!(action, TradingAction::Buy);
    }

    #[test]
    fn test_irl_trains_and_predicts() {
        let demos = sample_demos(ExpertType::Momentum);
        let mut irl = MaxEntropyIRL::new(0.05, 100);
        irl.train(&demos).unwrap();

        let state = TradingState {
            price_return: -0.03,
            volatility: 0.02,
            momentum: 0.1,
            volume_ratio: 0.8,
        };
        let action = irl.predict(&state);
        // Low momentum -> expect Sell.
        assert_eq!(action, TradingAction::Sell);
    }

    #[test]
    fn test_irl_reward_weights_nonzero() {
        let demos = sample_demos(ExpertType::MeanReversion);
        let mut irl = MaxEntropyIRL::new(0.05, 100);
        irl.train(&demos).unwrap();

        let weights = irl.get_reward_weights();
        let sum_abs: f64 = weights.iter().map(|w| w.abs()).sum();
        assert!(sum_abs > 0.01, "Reward weights should be non-trivial after training");
    }

    #[test]
    fn test_expert_policy_consistency() {
        let expert = ExpertPolicy::new(ExpertType::Momentum, 0.1);

        let buy_state = TradingState {
            price_return: 0.02,
            volatility: 0.01,
            momentum: 0.8,
            volume_ratio: 1.0,
        };
        assert_eq!(expert.act(&buy_state), TradingAction::Buy);

        let sell_state = TradingState {
            price_return: -0.02,
            volatility: 0.01,
            momentum: 0.2,
            volume_ratio: 1.0,
        };
        assert_eq!(expert.act(&sell_state), TradingAction::Sell);

        let hold_state = TradingState {
            price_return: 0.0,
            volatility: 0.01,
            momentum: 0.5,
            volume_ratio: 1.0,
        };
        assert_eq!(expert.act(&hold_state), TradingAction::Hold);
    }

    #[test]
    fn test_evaluation_metrics() {
        let demos = sample_demos(ExpertType::Momentum);
        let (train, test) = demos.split_at(150);

        let mut bc = BehavioralCloner::new(0.01, 100);
        bc.train(train).unwrap();
        let metrics = bc.evaluate(test);

        assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
        assert!(metrics.max_drawdown >= 0.0 && metrics.max_drawdown <= 1.0);
    }

    #[test]
    fn test_compare_methods() {
        let demos = sample_demos(ExpertType::MeanReversion);
        let (train, test) = demos.split_at(150);

        let (bc_m, irl_m) = compare_methods(train, test).unwrap();

        // Both should produce valid metrics.
        assert!(bc_m.accuracy >= 0.0);
        assert!(irl_m.accuracy >= 0.0);
    }

    #[test]
    fn test_candles_to_states() {
        let candles: Vec<Candle> = (0..30)
            .map(|i| Candle {
                timestamp: 1000 + i * 60,
                open: 100.0 + i as f64 * 0.1,
                high: 100.5 + i as f64 * 0.1,
                low: 99.5 + i as f64 * 0.1,
                close: 100.0 + i as f64 * 0.15,
                volume: 1000.0 + i as f64 * 10.0,
            })
            .collect();

        let states = candles_to_states(&candles, 5);
        assert_eq!(states.len(), 25); // 30 - 5 = 25
        assert!(states[0].volatility >= 0.0);
        assert!(states[0].momentum >= 0.0 && states[0].momentum <= 1.0);
    }

    #[test]
    fn test_trading_action_roundtrip() {
        for action in [TradingAction::Buy, TradingAction::Hold, TradingAction::Sell] {
            assert_eq!(TradingAction::from_index(action.to_index()), action);
        }
    }
}
