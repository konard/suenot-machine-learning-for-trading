use ndarray::Array2;
use rand::Rng;
use serde::Deserialize;


// ============================================================
// Multi-Objective Reward
// ============================================================

/// Number of objectives: return, negative drawdown, negative volatility.
pub const NUM_OBJECTIVES: usize = 3;

/// Computes the multi-objective reward vector from price data and a trading action.
pub struct MultiObjectiveReward {
    pub window_size: usize,
    price_history: Vec<f64>,
    peak: f64,
}

impl MultiObjectiveReward {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            price_history: Vec::new(),
            peak: f64::NEG_INFINITY,
        }
    }

    pub fn reset(&mut self) {
        self.price_history.clear();
        self.peak = f64::NEG_INFINITY;
    }

    /// Compute the reward vector [return, -drawdown, -volatility].
    /// `position` is -1 (short), 0 (neutral), or 1 (long).
    pub fn compute(&mut self, prev_price: f64, curr_price: f64, position: i32) -> [f64; NUM_OBJECTIVES] {
        let log_return = (curr_price / prev_price).ln();
        let position_return = position as f64 * log_return;

        self.price_history.push(curr_price);

        // Update peak for drawdown
        if curr_price > self.peak {
            self.peak = curr_price;
        }
        let drawdown = if self.peak > 0.0 {
            (self.peak - curr_price) / self.peak
        } else {
            0.0
        };

        // Compute rolling volatility
        let volatility = self.compute_volatility();

        [position_return, -drawdown, -volatility]
    }

    fn compute_volatility(&self) -> f64 {
        let n = self.price_history.len();
        if n < 2 {
            return 0.0;
        }
        let start = if n > self.window_size { n - self.window_size } else { 0 };
        let window = &self.price_history[start..];

        let returns: Vec<f64> = window
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        variance.sqrt()
    }
}

// ============================================================
// State Discretization
// ============================================================

/// Discretize a return into 5 buckets: strong_down, down, flat, up, strong_up
pub fn discretize_return(log_return: f64) -> usize {
    if log_return < -0.02 {
        0 // strong down
    } else if log_return < -0.005 {
        1 // down
    } else if log_return < 0.005 {
        2 // flat
    } else if log_return < 0.02 {
        3 // up
    } else {
        4 // strong up
    }
}

/// Map position (-1, 0, 1) to index (0, 1, 2)
pub fn position_to_index(position: i32) -> usize {
    (position + 1) as usize
}

/// Map index (0, 1, 2) back to position (-1, 0, 1)
pub fn index_to_position(index: usize) -> i32 {
    index as i32 - 1
}

/// Total number of states: 5 return buckets * 3 positions = 15
pub const NUM_STATES: usize = 15;
/// Number of actions: sell/short, hold, buy/long
pub const NUM_ACTIONS: usize = 3;

/// Convert (return_bucket, position_index) to a single state index.
pub fn to_state(return_bucket: usize, position_index: usize) -> usize {
    return_bucket * 3 + position_index
}

// ============================================================
// Scalarized Q-Learning
// ============================================================

/// A tabular Q-learning agent with scalarized multi-objective reward.
pub struct ScalarizedQLearning {
    pub q_table: Array2<f64>,
    pub weights: [f64; NUM_OBJECTIVES],
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub epsilon: f64,
}

impl ScalarizedQLearning {
    pub fn new(weights: [f64; NUM_OBJECTIVES], learning_rate: f64, discount_factor: f64, epsilon: f64) -> Self {
        Self {
            q_table: Array2::zeros((NUM_STATES, NUM_ACTIONS)),
            weights,
            learning_rate,
            discount_factor,
            epsilon,
        }
    }

    /// Scalarize a multi-objective reward vector using the weight vector.
    pub fn scalarize(&self, reward: &[f64; NUM_OBJECTIVES]) -> f64 {
        self.weights
            .iter()
            .zip(reward.iter())
            .map(|(w, r)| w * r)
            .sum()
    }

    /// Select an action using epsilon-greedy policy.
    pub fn select_action(&self, state: usize) -> usize {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..NUM_ACTIONS)
        } else {
            self.greedy_action(state)
        }
    }

    /// Select the greedy (best) action for a given state.
    pub fn greedy_action(&self, state: usize) -> usize {
        let row = self.q_table.row(state);
        let mut best_action = 0;
        let mut best_value = row[0];
        for a in 1..NUM_ACTIONS {
            if row[a] > best_value {
                best_value = row[a];
                best_action = a;
            }
        }
        best_action
    }

    /// Perform a Q-learning update.
    pub fn update(&mut self, state: usize, action: usize, reward: &[f64; NUM_OBJECTIVES], next_state: usize) {
        let scalar_reward = self.scalarize(reward);
        let next_max = self.q_table.row(next_state).iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let current = self.q_table[[state, action]];
        let td_target = scalar_reward + self.discount_factor * next_max;
        self.q_table[[state, action]] = current + self.learning_rate * (td_target - current);
    }

    /// Train on a sequence of prices.
    pub fn train_on_prices(&mut self, prices: &[f64], episodes: usize) {
        let mut rng = rand::thread_rng();
        for _ in 0..episodes {
            let mut reward_calc = MultiObjectiveReward::new(20);
            let mut position: i32 = 0;

            for i in 1..prices.len() {
                let log_ret = (prices[i] / prices[i - 1]).ln();
                let return_bucket = discretize_return(log_ret);
                let pos_idx = position_to_index(position);
                let state = to_state(return_bucket, pos_idx);

                let action = self.select_action(state);
                let new_position = index_to_position(action);

                let reward = reward_calc.compute(prices[i - 1], prices[i], new_position);

                let next_log_ret = if i + 1 < prices.len() {
                    (prices[i + 1] / prices[i]).ln()
                } else {
                    0.0
                };
                let next_return_bucket = discretize_return(next_log_ret);
                let next_pos_idx = position_to_index(new_position);
                let next_state = to_state(next_return_bucket, next_pos_idx);

                self.update(state, action, &reward, next_state);
                position = new_position;
            }

            // Decay epsilon
            self.epsilon *= 0.995;
            if self.epsilon < 0.01 {
                self.epsilon = 0.01;
            }

            // Randomize start position occasionally
            if rng.gen::<f64>() < 0.3 {
                // reset is implicit via new episode
            }
        }
    }
}

// ============================================================
// Policy Evaluator
// ============================================================

/// Evaluates a trained Q-learning agent on all objectives independently.
pub struct PolicyEvaluator;

impl PolicyEvaluator {
    /// Evaluate a policy on price data and return the average reward vector.
    pub fn evaluate(agent: &ScalarizedQLearning, prices: &[f64]) -> [f64; NUM_OBJECTIVES] {
        let mut reward_calc = MultiObjectiveReward::new(20);
        let mut total_reward = [0.0f64; NUM_OBJECTIVES];
        let mut position: i32 = 0;
        let mut count = 0;

        for i in 1..prices.len() {
            let log_ret = (prices[i] / prices[i - 1]).ln();
            let return_bucket = discretize_return(log_ret);
            let pos_idx = position_to_index(position);
            let state = to_state(return_bucket, pos_idx);

            let action = agent.greedy_action(state);
            let new_position = index_to_position(action);

            let reward = reward_calc.compute(prices[i - 1], prices[i], new_position);

            for j in 0..NUM_OBJECTIVES {
                total_reward[j] += reward[j];
            }
            count += 1;
            position = new_position;
        }

        if count > 0 {
            for j in 0..NUM_OBJECTIVES {
                total_reward[j] /= count as f64;
            }
        }
        total_reward
    }
}

// ============================================================
// Pareto Front
// ============================================================

/// A solution on the Pareto front: objective values and associated weight vector.
#[derive(Debug, Clone)]
pub struct ParetoSolution {
    pub objectives: [f64; NUM_OBJECTIVES],
    pub weights: [f64; NUM_OBJECTIVES],
}

/// Maintains a set of non-dominated solutions forming the Pareto front.
#[derive(Debug, Clone)]
pub struct ParetoFront {
    pub solutions: Vec<ParetoSolution>,
}

impl ParetoFront {
    pub fn new() -> Self {
        Self {
            solutions: Vec::new(),
        }
    }

    /// Check if `a` Pareto-dominates `b`.
    pub fn dominates(a: &[f64; NUM_OBJECTIVES], b: &[f64; NUM_OBJECTIVES]) -> bool {
        let mut at_least_one_better = false;
        for i in 0..NUM_OBJECTIVES {
            if a[i] < b[i] {
                return false;
            }
            if a[i] > b[i] {
                at_least_one_better = true;
            }
        }
        at_least_one_better
    }

    /// Insert a new solution, removing any it dominates.
    pub fn insert(&mut self, solution: ParetoSolution) {
        // Check if the new solution is dominated by any existing one
        for existing in &self.solutions {
            if Self::dominates(&existing.objectives, &solution.objectives) {
                return; // New solution is dominated, don't add
            }
        }

        // Remove solutions dominated by the new one
        self.solutions
            .retain(|existing| !Self::dominates(&solution.objectives, &existing.objectives));

        self.solutions.push(solution);
    }

    /// Compute the hypervolume indicator with respect to a reference point.
    /// For simplicity, this is a 2D hypervolume using the first two objectives.
    pub fn hypervolume_2d(&self, reference: [f64; 2]) -> f64 {
        if self.solutions.is_empty() {
            return 0.0;
        }

        // Sort by first objective descending
        let mut points: Vec<[f64; 2]> = self
            .solutions
            .iter()
            .map(|s| [s.objectives[0], s.objectives[1]])
            .collect();
        points.sort_by(|a, b| b[0].partial_cmp(&a[0]).unwrap());

        let mut volume = 0.0;
        let mut prev_y = reference[1];

        for point in &points {
            if point[0] > reference[0] && point[1] > reference[1] {
                let width = point[0] - reference[0];
                let height = point[1] - prev_y;
                if height > 0.0 {
                    volume += width * height;
                    prev_y = point[1];
                }
            }
        }
        volume
    }

    /// Select the best policy for a given preference (weight vector).
    pub fn select_policy(&self, preference: &[f64; NUM_OBJECTIVES]) -> Option<&ParetoSolution> {
        self.solutions.iter().max_by(|a, b| {
            let score_a: f64 = a
                .objectives
                .iter()
                .zip(preference.iter())
                .map(|(o, w)| o * w)
                .sum();
            let score_b: f64 = b
                .objectives
                .iter()
                .zip(preference.iter())
                .map(|(o, w)| o * w)
                .sum();
            score_a.partial_cmp(&score_b).unwrap()
        })
    }

    /// Return the number of solutions on the front.
    pub fn len(&self) -> usize {
        self.solutions.len()
    }

    /// Check if the front is empty.
    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }
}

impl Default for ParetoFront {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Bybit API Client
// ============================================================

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

/// Client for fetching OHLCV data from Bybit V5 public API.
pub struct BybitClient {
    base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data for a given symbol and interval.
    /// Returns a vector of closing prices sorted chronologically.
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<f64>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: BybitResponse = reqwest::blocking::get(&url)?.json()?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let mut prices: Vec<f64> = response
            .result
            .list
            .iter()
            .filter_map(|candle| {
                if candle.len() >= 5 {
                    candle[4].parse::<f64>().ok() // Close price
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, reverse for chronological order
        prices.reverse();

        Ok(prices)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Weight Generation
// ============================================================

/// Generate a set of weight vectors uniformly distributed on the simplex.
pub fn generate_weight_vectors(num_weights: usize) -> Vec<[f64; NUM_OBJECTIVES]> {
    let mut weights = Vec::new();
    let steps = ((num_weights as f64).cbrt().ceil()) as usize;

    for i in 0..=steps {
        for j in 0..=(steps - i) {
            let k = steps - i - j;
            let w = [
                i as f64 / steps as f64,
                j as f64 / steps as f64,
                k as f64 / steps as f64,
            ];
            weights.push(w);
        }
    }

    weights
}

/// Chebyshev scalarization: returns min_i w_i * (r_i - z_i)
pub fn chebyshev_scalarize(
    weights: &[f64; NUM_OBJECTIVES],
    reward: &[f64; NUM_OBJECTIVES],
    reference: &[f64; NUM_OBJECTIVES],
) -> f64 {
    let mut min_val = f64::INFINITY;
    for i in 0..NUM_OBJECTIVES {
        let val = weights[i] * (reward[i] - reference[i]);
        if val < min_val {
            min_val = val;
        }
    }
    min_val
}

// ============================================================
// Training Orchestrator
// ============================================================

/// Configuration for multi-objective RL training.
pub struct MORLConfig {
    pub num_weight_samples: usize,
    pub episodes_per_weight: usize,
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub initial_epsilon: f64,
}

impl Default for MORLConfig {
    fn default() -> Self {
        Self {
            num_weight_samples: 10,
            episodes_per_weight: 50,
            learning_rate: 0.1,
            discount_factor: 0.99,
            initial_epsilon: 0.3,
        }
    }
}

/// Train multiple agents with different weight vectors and build the Pareto front.
pub fn train_pareto_front(
    prices: &[f64],
    config: &MORLConfig,
) -> (ParetoFront, Vec<(ScalarizedQLearning, [f64; NUM_OBJECTIVES])>) {
    let weight_vectors = generate_weight_vectors(config.num_weight_samples);
    let mut pareto_front = ParetoFront::new();
    let mut agents_with_objectives = Vec::new();

    for weights in &weight_vectors {
        let mut agent = ScalarizedQLearning::new(
            *weights,
            config.learning_rate,
            config.discount_factor,
            config.initial_epsilon,
        );

        agent.train_on_prices(prices, config.episodes_per_weight);

        let objectives = PolicyEvaluator::evaluate(&agent, prices);

        pareto_front.insert(ParetoSolution {
            objectives,
            weights: *weights,
        });

        agents_with_objectives.push((agent, objectives));
    }

    (pareto_front, agents_with_objectives)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_prices() -> Vec<f64> {
        vec![
            100.0, 101.0, 102.5, 101.0, 99.5, 100.5, 103.0, 104.0, 102.0, 101.0,
            103.0, 105.0, 104.5, 106.0, 107.0, 105.0, 104.0, 106.0, 108.0, 110.0,
        ]
    }

    #[test]
    fn test_multi_objective_reward_computation() {
        let mut reward = MultiObjectiveReward::new(10);
        let r = reward.compute(100.0, 105.0, 1);
        // Positive return for long position with price increase
        assert!(r[0] > 0.0, "Return should be positive for long with price increase");
        // Drawdown should be 0 (first price is new peak)
        assert!(r[1] <= 0.0 || (r[1]).abs() < 1e-10, "Drawdown component should be <= 0");
    }

    #[test]
    fn test_discretize_return() {
        assert_eq!(discretize_return(-0.05), 0); // strong down
        assert_eq!(discretize_return(-0.01), 1); // down
        assert_eq!(discretize_return(0.0), 2);   // flat
        assert_eq!(discretize_return(0.01), 3);  // up
        assert_eq!(discretize_return(0.05), 4);  // strong up
    }

    #[test]
    fn test_pareto_dominance() {
        let a = [1.0, 2.0, 3.0];
        let b = [0.5, 1.5, 2.5];
        assert!(ParetoFront::dominates(&a, &b));
        assert!(!ParetoFront::dominates(&b, &a));

        // Non-dominating pair
        let c = [2.0, 1.0, 3.0];
        assert!(!ParetoFront::dominates(&a, &c));
        assert!(!ParetoFront::dominates(&c, &a));
    }

    #[test]
    fn test_pareto_front_insertion() {
        let mut front = ParetoFront::new();

        front.insert(ParetoSolution {
            objectives: [1.0, 2.0, 3.0],
            weights: [0.33, 0.33, 0.34],
        });
        assert_eq!(front.len(), 1);

        // This is dominated, should not be added
        front.insert(ParetoSolution {
            objectives: [0.5, 1.0, 2.0],
            weights: [0.5, 0.3, 0.2],
        });
        assert_eq!(front.len(), 1);

        // This is non-dominated (better in obj 0, worse in obj 1)
        front.insert(ParetoSolution {
            objectives: [2.0, 1.0, 2.5],
            weights: [0.6, 0.2, 0.2],
        });
        assert_eq!(front.len(), 2);
    }

    #[test]
    fn test_scalarized_q_learning() {
        let weights = [0.5, 0.3, 0.2];
        let mut agent = ScalarizedQLearning::new(weights, 0.1, 0.99, 0.3);

        let reward = [0.01, -0.005, -0.002];
        let scalar = agent.scalarize(&reward);
        let expected = 0.5 * 0.01 + 0.3 * (-0.005) + 0.2 * (-0.002);
        assert!((scalar - expected).abs() < 1e-10);

        // Train on sample prices
        let prices = sample_prices();
        agent.train_on_prices(&prices, 10);

        // Q-table should have been updated (not all zeros)
        let sum: f64 = agent.q_table.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0, "Q-table should be updated after training");
    }

    #[test]
    fn test_policy_evaluator() {
        let prices = sample_prices();
        let weights = [0.5, 0.3, 0.2];
        let mut agent = ScalarizedQLearning::new(weights, 0.1, 0.99, 0.3);
        agent.train_on_prices(&prices, 20);

        let objectives = PolicyEvaluator::evaluate(&agent, &prices);
        // Objectives should be finite numbers
        for obj in &objectives {
            assert!(obj.is_finite(), "Objectives should be finite");
        }
    }

    #[test]
    fn test_chebyshev_scalarization() {
        let weights = [0.5, 0.3, 0.2];
        let reward = [1.0, 2.0, 3.0];
        let reference = [0.0, 0.0, 0.0];
        let result = chebyshev_scalarize(&weights, &reward, &reference);
        // min(0.5*1.0, 0.3*2.0, 0.2*3.0) = min(0.5, 0.6, 0.6) = 0.5
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_weight_generation() {
        let weights = generate_weight_vectors(10);
        assert!(!weights.is_empty());
        // All weights should sum to ~1.0
        for w in &weights {
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "Weights should sum to 1.0, got {}", sum);
        }
    }

    #[test]
    fn test_pareto_front_select_policy() {
        let mut front = ParetoFront::new();
        front.insert(ParetoSolution {
            objectives: [1.0, 0.5, 0.3],
            weights: [0.6, 0.2, 0.2],
        });
        front.insert(ParetoSolution {
            objectives: [0.3, 1.0, 0.8],
            weights: [0.2, 0.5, 0.3],
        });

        // Prefer return
        let preference = [1.0, 0.0, 0.0];
        let selected = front.select_policy(&preference).unwrap();
        assert!((selected.objectives[0] - 1.0).abs() < 1e-10);

        // Prefer drawdown (second objective)
        let preference = [0.0, 1.0, 0.0];
        let selected = front.select_policy(&preference).unwrap();
        assert!((selected.objectives[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_train_pareto_front() {
        let prices = sample_prices();
        let config = MORLConfig {
            num_weight_samples: 3,
            episodes_per_weight: 5,
            learning_rate: 0.1,
            discount_factor: 0.99,
            initial_epsilon: 0.3,
        };

        let (front, agents) = train_pareto_front(&prices, &config);
        assert!(!front.is_empty(), "Pareto front should not be empty");
        assert!(!agents.is_empty(), "Should have trained agents");
    }
}
