//! Hyperparameter Optimization Library for Trading Models
//!
//! Implements grid search, random search, Bayesian optimization with GP surrogate,
//! Successive Halving, Hyperband, and walk-forward cross-validation for trading.

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Hyperparameter space definitions
// ---------------------------------------------------------------------------

/// Type of hyperparameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HyperparameterType {
    /// Continuous parameter with (min, max, log_scale).
    Continuous { min: f64, max: f64, log_scale: bool },
    /// Discrete integer parameter with possible values.
    Discrete { values: Vec<i64> },
    /// Categorical parameter with named options.
    Categorical { options: Vec<String> },
}

/// A single hyperparameter definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterDef {
    pub name: String,
    pub param_type: HyperparameterType,
}

/// A search space is a collection of hyperparameter definitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpace {
    pub params: Vec<HyperparameterDef>,
}

/// A concrete hyperparameter value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HyperparameterValue {
    Continuous(f64),
    Discrete(i64),
    Categorical(String),
}

/// A configuration is a map from parameter name to value.
pub type Config = HashMap<String, HyperparameterValue>;

impl SearchSpace {
    pub fn new() -> Self {
        Self { params: Vec::new() }
    }

    pub fn add_continuous(&mut self, name: &str, min: f64, max: f64, log_scale: bool) {
        self.params.push(HyperparameterDef {
            name: name.to_string(),
            param_type: HyperparameterType::Continuous {
                min,
                max,
                log_scale,
            },
        });
    }

    pub fn add_discrete(&mut self, name: &str, values: Vec<i64>) {
        self.params.push(HyperparameterDef {
            name: name.to_string(),
            param_type: HyperparameterType::Discrete { values },
        });
    }

    pub fn add_categorical(&mut self, name: &str, options: Vec<&str>) {
        self.params.push(HyperparameterDef {
            name: name.to_string(),
            param_type: HyperparameterType::Categorical {
                options: options.into_iter().map(|s| s.to_string()).collect(),
            },
        });
    }

    /// Sample a random configuration from the search space.
    pub fn sample_random(&self, rng: &mut impl Rng) -> Config {
        let mut config = Config::new();
        for param in &self.params {
            let value = match &param.param_type {
                HyperparameterType::Continuous {
                    min,
                    max,
                    log_scale,
                } => {
                    if *log_scale {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        let v = rng.gen_range(log_min..=log_max);
                        HyperparameterValue::Continuous(v.exp())
                    } else {
                        HyperparameterValue::Continuous(rng.gen_range(*min..=*max))
                    }
                }
                HyperparameterType::Discrete { values } => {
                    let idx = rng.gen_range(0..values.len());
                    HyperparameterValue::Discrete(values[idx])
                }
                HyperparameterType::Categorical { options } => {
                    let idx = rng.gen_range(0..options.len());
                    HyperparameterValue::Categorical(options[idx].clone())
                }
            };
            config.insert(param.name.clone(), value);
        }
        config
    }

    /// Generate all grid configurations. Continuous params are discretized into `n_points` values.
    pub fn generate_grid(&self, n_points: usize) -> Vec<Config> {
        let mut axes: Vec<Vec<(String, HyperparameterValue)>> = Vec::new();

        for param in &self.params {
            let values: Vec<(String, HyperparameterValue)> = match &param.param_type {
                HyperparameterType::Continuous {
                    min,
                    max,
                    log_scale,
                } => {
                    (0..n_points)
                        .map(|i| {
                            let t = if n_points > 1 {
                                i as f64 / (n_points - 1) as f64
                            } else {
                                0.5
                            };
                            let v = if *log_scale {
                                let log_min = min.ln();
                                let log_max = max.ln();
                                (log_min + t * (log_max - log_min)).exp()
                            } else {
                                min + t * (max - min)
                            };
                            (param.name.clone(), HyperparameterValue::Continuous(v))
                        })
                        .collect()
                }
                HyperparameterType::Discrete { values } => values
                    .iter()
                    .map(|v| (param.name.clone(), HyperparameterValue::Discrete(*v)))
                    .collect(),
                HyperparameterType::Categorical { options } => options
                    .iter()
                    .map(|o| (param.name.clone(), HyperparameterValue::Categorical(o.clone())))
                    .collect(),
            };
            axes.push(values);
        }

        // Cartesian product
        let mut configs: Vec<Config> = vec![Config::new()];
        for axis in &axes {
            let mut new_configs = Vec::new();
            for config in &configs {
                for (name, value) in axis {
                    let mut new_config = config.clone();
                    new_config.insert(name.clone(), value.clone());
                    new_configs.push(new_config);
                }
            }
            configs = new_configs;
        }
        configs
    }

    /// Encode a config as a vector of f64 for the GP surrogate.
    pub fn encode_config(&self, config: &Config) -> Vec<f64> {
        let mut encoded = Vec::new();
        for param in &self.params {
            if let Some(val) = config.get(&param.name) {
                match (&param.param_type, val) {
                    (
                        HyperparameterType::Continuous {
                            min,
                            max,
                            log_scale,
                        },
                        HyperparameterValue::Continuous(v),
                    ) => {
                        // Normalize to [0, 1]
                        if *log_scale {
                            let log_min = min.ln();
                            let log_max = max.ln();
                            encoded.push((v.ln() - log_min) / (log_max - log_min));
                        } else {
                            encoded.push((v - min) / (max - min));
                        }
                    }
                    (
                        HyperparameterType::Discrete { values },
                        HyperparameterValue::Discrete(v),
                    ) => {
                        let idx = values.iter().position(|x| x == v).unwrap_or(0);
                        let norm = if values.len() > 1 {
                            idx as f64 / (values.len() - 1) as f64
                        } else {
                            0.5
                        };
                        encoded.push(norm);
                    }
                    (
                        HyperparameterType::Categorical { options },
                        HyperparameterValue::Categorical(v),
                    ) => {
                        let idx = options.iter().position(|x| x == v).unwrap_or(0);
                        let norm = if options.len() > 1 {
                            idx as f64 / (options.len() - 1) as f64
                        } else {
                            0.5
                        };
                        encoded.push(norm);
                    }
                    _ => encoded.push(0.0),
                }
            } else {
                encoded.push(0.0);
            }
        }
        encoded
    }
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Trial history
// ---------------------------------------------------------------------------

/// A single trial result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    pub config: Config,
    pub score: f64,
}

/// History of all evaluated trials.
#[derive(Debug, Clone, Default)]
pub struct TrialHistory {
    pub trials: Vec<Trial>,
}

impl TrialHistory {
    pub fn new() -> Self {
        Self { trials: Vec::new() }
    }

    pub fn add(&mut self, config: Config, score: f64) {
        self.trials.push(Trial { config, score });
    }

    /// Return the best trial (highest score).
    pub fn best(&self) -> Option<&Trial> {
        self.trials
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
    }

    pub fn len(&self) -> usize {
        self.trials.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trials.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Gaussian Process with RBF kernel
// ---------------------------------------------------------------------------

/// Gaussian Process surrogate model with RBF kernel.
pub struct GaussianProcess {
    length_scale: f64,
    signal_variance: f64,
    noise_variance: f64,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<f64>>,
    alpha: Option<Array1<f64>>, // K_inv * y
    k_inv: Option<Array2<f64>>,
}

impl GaussianProcess {
    pub fn new(length_scale: f64, signal_variance: f64, noise_variance: f64) -> Self {
        Self {
            length_scale,
            signal_variance,
            noise_variance,
            x_train: None,
            y_train: None,
            alpha: None,
            k_inv: None,
        }
    }

    /// RBF kernel between two points.
    fn rbf_kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let sq_dist: f64 = x1
            .iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        self.signal_variance * (-sq_dist / (2.0 * self.length_scale.powi(2))).exp()
    }

    /// Fit the GP to training data.
    pub fn fit(&mut self, x: Array2<f64>, y: Array1<f64>) {
        let n = x.nrows();
        let mut k = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                k[[i, j]] = self.rbf_kernel(
                    x.row(i).as_slice().unwrap(),
                    x.row(j).as_slice().unwrap(),
                );
                if i == j {
                    k[[i, j]] += self.noise_variance;
                }
            }
        }

        // Simple matrix inversion via Gauss-Jordan for small matrices
        let k_inv = Self::invert_matrix(&k);
        let alpha = k_inv.dot(&y);

        self.x_train = Some(x);
        self.y_train = Some(y);
        self.alpha = Some(alpha);
        self.k_inv = Some(k_inv);
    }

    /// Predict mean and variance at a new point.
    pub fn predict(&self, x: &[f64]) -> (f64, f64) {
        let x_train = match &self.x_train {
            Some(xt) => xt,
            None => return (0.0, self.signal_variance),
        };
        let alpha = self.alpha.as_ref().unwrap();
        let k_inv = self.k_inv.as_ref().unwrap();

        let n = x_train.nrows();
        let mut k_star = Array1::<f64>::zeros(n);
        for i in 0..n {
            k_star[i] = self.rbf_kernel(x, x_train.row(i).as_slice().unwrap());
        }

        let mean = k_star.dot(alpha);
        let k_ss = self.rbf_kernel(x, x);
        let v = k_inv.dot(&k_star);
        let variance = (k_ss - k_star.dot(&v)).max(1e-10);

        (mean, variance)
    }

    /// Simple matrix inversion using Gauss-Jordan elimination.
    fn invert_matrix(m: &Array2<f64>) -> Array2<f64> {
        let n = m.nrows();
        let mut augmented = Array2::<f64>::zeros((n, 2 * n));

        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = m[[i, j]];
            }
            augmented[[i, n + i]] = 1.0;
        }

        for i in 0..n {
            let mut max_row = i;
            let mut max_val = augmented[[i, i]].abs();
            for k in (i + 1)..n {
                if augmented[[k, i]].abs() > max_val {
                    max_val = augmented[[k, i]].abs();
                    max_row = k;
                }
            }

            if max_row != i {
                for j in 0..(2 * n) {
                    let tmp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[max_row, j]];
                    augmented[[max_row, j]] = tmp;
                }
            }

            let pivot = augmented[[i, i]];
            if pivot.abs() < 1e-12 {
                // Near-singular: add jitter
                augmented[[i, i]] += 1e-6;
                let pivot = augmented[[i, i]];
                for j in 0..(2 * n) {
                    augmented[[i, j]] /= pivot;
                }
            } else {
                for j in 0..(2 * n) {
                    augmented[[i, j]] /= pivot;
                }
            }

            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]];
                    for j in 0..(2 * n) {
                        augmented[[k, j]] -= factor * augmented[[i, j]];
                    }
                }
            }
        }

        let mut result = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                result[[i, j]] = augmented[[i, n + j]];
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Acquisition functions
// ---------------------------------------------------------------------------

/// Standard normal PDF.
fn standard_normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard normal CDF (approximation).
fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz and Stegun).
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

/// Expected Improvement acquisition function.
pub fn expected_improvement(mean: f64, variance: f64, best_score: f64) -> f64 {
    let sigma = variance.sqrt();
    if sigma < 1e-10 {
        return (mean - best_score).max(0.0);
    }
    let z = (mean - best_score) / sigma;
    let ei = (mean - best_score) * standard_normal_cdf(z) + sigma * standard_normal_pdf(z);
    ei.max(0.0)
}

/// Upper Confidence Bound acquisition function.
pub fn upper_confidence_bound(mean: f64, variance: f64, kappa: f64) -> f64 {
    mean + kappa * variance.sqrt()
}

/// Probability of Improvement acquisition function.
pub fn probability_of_improvement(mean: f64, variance: f64, best_score: f64) -> f64 {
    let sigma = variance.sqrt();
    if sigma < 1e-10 {
        return if mean > best_score { 1.0 } else { 0.0 };
    }
    standard_normal_cdf((mean - best_score) / sigma)
}

// ---------------------------------------------------------------------------
// Optimizers
// ---------------------------------------------------------------------------

/// Grid search optimizer.
pub fn grid_search(
    space: &SearchSpace,
    objective: &dyn Fn(&Config) -> f64,
    n_points_per_axis: usize,
) -> TrialHistory {
    let mut history = TrialHistory::new();
    let configs = space.generate_grid(n_points_per_axis);

    for config in configs {
        let score = objective(&config);
        history.add(config, score);
    }

    history
}

/// Random search optimizer.
pub fn random_search(
    space: &SearchSpace,
    objective: &dyn Fn(&Config) -> f64,
    n_trials: usize,
    seed: u64,
) -> TrialHistory {
    let mut history = TrialHistory::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    for _ in 0..n_trials {
        let config = space.sample_random(&mut rng);
        let score = objective(&config);
        history.add(config, score);
    }

    history
}

/// Bayesian optimization with GP surrogate and Expected Improvement.
pub fn bayesian_optimization(
    space: &SearchSpace,
    objective: &dyn Fn(&Config) -> f64,
    n_trials: usize,
    n_initial: usize,
    seed: u64,
) -> TrialHistory {
    use rand::SeedableRng;
    let mut history = TrialHistory::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Phase 1: random initial evaluations
    let init_count = n_initial.min(n_trials);
    for _ in 0..init_count {
        let config = space.sample_random(&mut rng);
        let score = objective(&config);
        history.add(config, score);
    }

    // Phase 2: Bayesian optimization loop
    let mut gp = GaussianProcess::new(0.5, 1.0, 0.01);
    let n_candidates = 200; // number of random candidates to evaluate acquisition on

    for _ in init_count..n_trials {
        // Build training data for GP
        let dim = space.params.len();
        let n = history.len();
        let mut x_data = Array2::<f64>::zeros((n, dim));
        let mut y_data = Array1::<f64>::zeros(n);

        for (i, trial) in history.trials.iter().enumerate() {
            let encoded = space.encode_config(&trial.config);
            for (j, &v) in encoded.iter().enumerate() {
                x_data[[i, j]] = v;
            }
            y_data[i] = trial.score;
        }

        // Normalize y for numerical stability
        let y_mean = y_data.mean().unwrap_or(0.0);
        let y_std = {
            let var = y_data.mapv(|v| (v - y_mean).powi(2)).mean().unwrap_or(1.0);
            var.sqrt().max(1e-8)
        };
        let y_norm = y_data.mapv(|v| (v - y_mean) / y_std);

        gp.fit(x_data, y_norm);

        let best_score_norm = history
            .best()
            .map(|t| (t.score - y_mean) / y_std)
            .unwrap_or(0.0);

        // Find next config by maximizing EI over random candidates
        let mut best_ei = f64::NEG_INFINITY;
        let mut best_config = space.sample_random(&mut rng);

        for _ in 0..n_candidates {
            let candidate = space.sample_random(&mut rng);
            let encoded = space.encode_config(&candidate);
            let (mean, var) = gp.predict(&encoded);
            let ei = expected_improvement(mean, var, best_score_norm);
            if ei > best_ei {
                best_ei = ei;
                best_config = candidate;
            }
        }

        let score = objective(&best_config);
        history.add(best_config, score);
    }

    history
}

/// Successive Halving algorithm.
pub fn successive_halving(
    space: &SearchSpace,
    objective: &dyn Fn(&Config, usize) -> f64, // objective takes budget
    n_configs: usize,
    max_budget: usize,
    eta: usize,
    seed: u64,
) -> TrialHistory {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut history = TrialHistory::new();

    // Generate initial configurations
    let mut configs: Vec<Config> = (0..n_configs)
        .map(|_| space.sample_random(&mut rng))
        .collect();

    let s_max = (n_configs as f64).log(eta as f64).floor() as usize;

    let mut n_remaining = n_configs;
    for i in 0..=s_max {
        let budget = max_budget * eta.pow(i as u32) / eta.pow(s_max as u32);
        let budget = budget.max(1);

        // Evaluate all remaining configs with current budget
        let mut scores: Vec<(usize, f64)> = configs
            .iter()
            .enumerate()
            .map(|(idx, config)| {
                let score = objective(config, budget);
                (idx, score)
            })
            .collect();

        // Record trials
        for (idx, score) in &scores {
            history.add(configs[*idx].clone(), *score);
        }

        // Keep top 1/eta fraction
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        n_remaining = (n_remaining / eta).max(1);
        let surviving_indices: Vec<usize> =
            scores.iter().take(n_remaining).map(|(idx, _)| *idx).collect();
        configs = surviving_indices
            .into_iter()
            .map(|idx| configs[idx].clone())
            .collect();

        if configs.len() <= 1 {
            break;
        }
    }

    history
}

/// Hyperband optimizer.
pub fn hyperband(
    space: &SearchSpace,
    objective: &dyn Fn(&Config, usize) -> f64,
    max_budget: usize,
    eta: usize,
    seed: u64,
) -> TrialHistory {
    let mut history = TrialHistory::new();
    let s_max = (max_budget as f64).log(eta as f64).floor() as usize;

    for s in (0..=s_max).rev() {
        let n = ((s_max + 1) as f64 / (s + 1) as f64 * (eta.pow(s as u32) as f64)).ceil()
            as usize;
        let _r = max_budget / eta.pow(s as u32);
        let _r = _r.max(1);

        let bracket_seed = seed.wrapping_add(s as u64);
        let bracket_history = successive_halving(space, objective, n, max_budget, eta, bracket_seed);

        for trial in bracket_history.trials {
            history.add(trial.config, trial.score);
        }
    }

    history
}

// ---------------------------------------------------------------------------
// Walk-forward cross-validation
// ---------------------------------------------------------------------------

/// Candle (OHLCV) data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Walk-forward cross-validation objective for trading.
/// Returns average score across folds.
pub fn walk_forward_objective(
    candles: &[Candle],
    config: &Config,
    n_folds: usize,
    strategy_fn: &dyn Fn(&[Candle], &[Candle], &Config) -> f64,
) -> f64 {
    if candles.len() < n_folds + 1 {
        return f64::NEG_INFINITY;
    }

    let fold_size = candles.len() / (n_folds + 1);
    let mut scores = Vec::new();

    for i in 0..n_folds {
        let train_end = (i + 1) * fold_size;
        let val_end = (i + 2) * fold_size;

        if val_end > candles.len() {
            break;
        }

        let train = &candles[..train_end];
        let val = &candles[train_end..val_end];

        let score = strategy_fn(train, val, config);
        scores.push(score);
    }

    if scores.is_empty() {
        return f64::NEG_INFINITY;
    }

    scores.iter().sum::<f64>() / scores.len() as f64
}

/// Compute Sharpe ratio from a vector of returns.
pub fn sharpe_ratio(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    if std < 1e-10 {
        return 0.0;
    }
    // Annualize assuming daily returns
    mean / std * (252.0_f64).sqrt()
}

/// Compute maximum drawdown from a sequence of portfolio values.
pub fn max_drawdown(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut peak = values[0];
    let mut max_dd = 0.0;
    for &v in values {
        if v > peak {
            peak = v;
        }
        let dd = (peak - v) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

/// Simple momentum strategy for backtesting.
/// Returns a scalarized multi-objective score: sharpe - weight * max_drawdown.
pub fn momentum_strategy(train: &[Candle], val: &[Candle], config: &Config) -> f64 {
    let lookback = match config.get("lookback") {
        Some(HyperparameterValue::Discrete(v)) => *v as usize,
        _ => 20,
    };
    let threshold = match config.get("threshold") {
        Some(HyperparameterValue::Continuous(v)) => *v,
        _ => 0.0,
    };
    let position_size = match config.get("position_size") {
        Some(HyperparameterValue::Continuous(v)) => *v,
        _ => 1.0,
    };
    let stop_loss = match config.get("stop_loss") {
        Some(HyperparameterValue::Continuous(v)) => *v,
        _ => 0.05,
    };

    // Compute momentum signal on validation data
    let all_candles: Vec<&Candle> = train.iter().chain(val.iter()).collect();
    let val_start = train.len();

    let mut returns = Vec::new();
    let mut portfolio_values = vec![1.0];
    let mut current_value = 1.0;

    for i in val_start..all_candles.len() {
        if i < lookback {
            continue;
        }

        // Momentum: return over lookback period
        let past_close = all_candles[i - lookback].close;
        let current_close = all_candles[i].close;
        let momentum = (current_close - past_close) / past_close;

        // Generate signal
        let signal = if momentum > threshold {
            position_size
        } else if momentum < -threshold {
            -position_size
        } else {
            0.0
        };

        // Daily return
        if i > val_start && i > lookback {
            let prev_close = all_candles[i - 1].close;
            let daily_return = (current_close - prev_close) / prev_close;
            let strategy_return = signal * daily_return;

            // Apply stop loss
            let capped_return = if strategy_return < -stop_loss {
                -stop_loss
            } else {
                strategy_return
            };

            returns.push(capped_return);
            current_value *= 1.0 + capped_return;
            portfolio_values.push(current_value);
        }
    }

    if returns.is_empty() {
        return f64::NEG_INFINITY;
    }

    let sr = sharpe_ratio(&returns);
    let mdd = max_drawdown(&portfolio_values);

    // Multi-objective: Sharpe - 2 * MaxDrawdown
    sr - 2.0 * mdd
}

// ---------------------------------------------------------------------------
// Bybit API client
// ---------------------------------------------------------------------------

/// Bybit API response structures.
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

/// Bybit client for fetching market data.
pub struct BybitClient {
    base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline/candlestick data from Bybit.
    /// `symbol`: e.g., "BTCUSDT"
    /// `interval`: e.g., "D" for daily, "60" for 1h, "15" for 15m
    /// `limit`: number of candles (max 1000)
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let client = reqwest::blocking::Client::new();
        let resp: BybitResponse = client
            .get(&url)
            .header("User-Agent", "hyperparameter-optimization-rust/0.1.0")
            .send()?
            .json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: {} (code {})", resp.ret_msg, resp.ret_code);
        }

        let mut candles: Vec<Candle> = resp
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 6 {
                    Some(Candle {
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

// ---------------------------------------------------------------------------
// Utility: extract f64 from config
// ---------------------------------------------------------------------------

pub fn config_get_f64(config: &Config, key: &str) -> Option<f64> {
    match config.get(key) {
        Some(HyperparameterValue::Continuous(v)) => Some(*v),
        Some(HyperparameterValue::Discrete(v)) => Some(*v as f64),
        _ => None,
    }
}

pub fn config_get_i64(config: &Config, key: &str) -> Option<i64> {
    match config.get(key) {
        Some(HyperparameterValue::Discrete(v)) => Some(*v),
        Some(HyperparameterValue::Continuous(v)) => Some(*v as i64),
        _ => None,
    }
}

pub fn config_get_str<'a>(config: &'a Config, key: &str) -> Option<&'a str> {
    match config.get(key) {
        Some(HyperparameterValue::Categorical(v)) => Some(v.as_str()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_search_space_random_sample() {
        let mut space = SearchSpace::new();
        space.add_continuous("lr", 0.001, 0.1, true);
        space.add_discrete("hidden", vec![32, 64, 128]);
        space.add_categorical("activation", vec!["relu", "tanh"]);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let config = space.sample_random(&mut rng);

        assert!(config.contains_key("lr"));
        assert!(config.contains_key("hidden"));
        assert!(config.contains_key("activation"));

        if let Some(HyperparameterValue::Continuous(lr)) = config.get("lr") {
            assert!(*lr >= 0.001 && *lr <= 0.1);
        } else {
            panic!("lr should be continuous");
        }
    }

    #[test]
    fn test_grid_generation() {
        let mut space = SearchSpace::new();
        space.add_continuous("x", 0.0, 1.0, false);
        space.add_discrete("y", vec![1, 2, 3]);

        let configs = space.generate_grid(3);
        // 3 continuous points * 3 discrete values = 9
        assert_eq!(configs.len(), 9);
    }

    #[test]
    fn test_gaussian_process() {
        let mut gp = GaussianProcess::new(1.0, 1.0, 0.01);

        let x = Array2::from_shape_vec((3, 1), vec![0.0, 0.5, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        gp.fit(x, y);

        let (mean, var) = gp.predict(&[0.5]);
        assert!((mean - 1.0).abs() < 0.5, "GP mean at 0.5 should be near 1.0, got {}", mean);
        assert!(var < 0.5, "GP variance at training point should be small, got {}", var);

        // Prediction far from training data should have higher variance
        let (_, var_far) = gp.predict(&[2.0]);
        assert!(var_far > var, "Variance far from data should be larger");
    }

    #[test]
    fn test_expected_improvement() {
        let ei = expected_improvement(1.5, 0.25, 1.0);
        assert!(ei > 0.0, "EI should be positive when mean > best");

        let ei_zero = expected_improvement(0.5, 0.0001, 1.0);
        assert!(ei_zero < 0.01, "EI should be near zero when mean < best and low variance");
    }

    #[test]
    fn test_ucb() {
        let ucb = upper_confidence_bound(1.0, 0.25, 2.0);
        assert!((ucb - 2.0).abs() < 1e-10, "UCB = 1.0 + 2.0 * 0.5 = 2.0");
    }

    #[test]
    fn test_probability_of_improvement() {
        let pi = probability_of_improvement(2.0, 0.01, 1.0);
        assert!(pi > 0.99, "PI should be near 1 when mean >> best");

        let pi_low = probability_of_improvement(0.0, 0.01, 1.0);
        assert!(pi_low < 0.01, "PI should be near 0 when mean << best");
    }

    #[test]
    fn test_grid_search_optimizer() {
        let mut space = SearchSpace::new();
        space.add_continuous("x", -2.0, 2.0, false);

        // Maximize negative quadratic: -(x - 0.5)^2
        let objective = |config: &Config| -> f64 {
            let x = config_get_f64(config, "x").unwrap();
            -(x - 0.5).powi(2)
        };

        let history = grid_search(&space, &objective, 21);
        let best = history.best().unwrap();
        let best_x = config_get_f64(&best.config, "x").unwrap();

        assert!(
            (best_x - 0.5).abs() < 0.3,
            "Grid search should find x near 0.5, got {}",
            best_x
        );
    }

    #[test]
    fn test_random_search_optimizer() {
        let mut space = SearchSpace::new();
        space.add_continuous("x", -2.0, 2.0, false);

        let objective = |config: &Config| -> f64 {
            let x = config_get_f64(config, "x").unwrap();
            -(x - 0.5).powi(2)
        };

        let history = random_search(&space, &objective, 100, 42);
        let best = history.best().unwrap();
        let best_x = config_get_f64(&best.config, "x").unwrap();

        assert!(
            (best_x - 0.5).abs() < 0.5,
            "Random search should find x near 0.5, got {}",
            best_x
        );
    }

    #[test]
    fn test_bayesian_optimization_optimizer() {
        let mut space = SearchSpace::new();
        space.add_continuous("x", -2.0, 2.0, false);

        let objective = |config: &Config| -> f64 {
            let x = config_get_f64(config, "x").unwrap();
            -(x - 0.5).powi(2)
        };

        let history = bayesian_optimization(&space, &objective, 30, 5, 42);
        let best = history.best().unwrap();
        let best_x = config_get_f64(&best.config, "x").unwrap();

        assert!(
            (best_x - 0.5).abs() < 0.8,
            "Bayesian optimization should find x near 0.5, got {}",
            best_x
        );
    }

    #[test]
    fn test_successive_halving() {
        let mut space = SearchSpace::new();
        space.add_continuous("x", -2.0, 2.0, false);

        let objective = |config: &Config, _budget: usize| -> f64 {
            let x = config_get_f64(config, "x").unwrap();
            -(x - 0.5).powi(2)
        };

        let history = successive_halving(&space, &objective, 27, 81, 3, 42);
        let best = history.best().unwrap();

        assert!(best.score > -1.0, "SHA should find a reasonable config");
    }

    #[test]
    fn test_sharpe_ratio_calculation() {
        let returns = vec![0.01, 0.02, -0.005, 0.015, 0.01];
        let sr = sharpe_ratio(&returns);
        assert!(sr > 0.0, "Sharpe ratio should be positive for positive-mean returns");
    }

    #[test]
    fn test_max_drawdown_calculation() {
        let values = vec![100.0, 110.0, 105.0, 95.0, 100.0, 108.0];
        let mdd = max_drawdown(&values);
        // Peak was 110, trough was 95 -> drawdown = 15/110 ~ 0.1364
        assert!(
            (mdd - 15.0 / 110.0).abs() < 1e-6,
            "Max drawdown should be ~13.64%, got {}",
            mdd
        );
    }

    #[test]
    fn test_walk_forward_objective() {
        // Create synthetic candle data
        let candles: Vec<Candle> = (0..200)
            .map(|i| {
                let price = 100.0 + (i as f64 * 0.1).sin() * 10.0;
                Candle {
                    timestamp: i as u64 * 86400000,
                    open: price,
                    high: price * 1.02,
                    low: price * 0.98,
                    close: price + (i as f64 * 0.05).cos(),
                    volume: 1000.0,
                }
            })
            .collect();

        let mut config = Config::new();
        config.insert("lookback".to_string(), HyperparameterValue::Discrete(10));
        config.insert(
            "threshold".to_string(),
            HyperparameterValue::Continuous(0.01),
        );
        config.insert(
            "position_size".to_string(),
            HyperparameterValue::Continuous(1.0),
        );
        config.insert(
            "stop_loss".to_string(),
            HyperparameterValue::Continuous(0.05),
        );

        let score = walk_forward_objective(&candles, &config, 3, &momentum_strategy);
        // Just verify it returns a finite number
        assert!(score.is_finite(), "Walk-forward score should be finite");
    }

    #[test]
    fn test_config_encode() {
        let mut space = SearchSpace::new();
        space.add_continuous("x", 0.0, 10.0, false);
        space.add_discrete("n", vec![1, 2, 3]);
        space.add_categorical("c", vec!["a", "b"]);

        let mut config = Config::new();
        config.insert("x".to_string(), HyperparameterValue::Continuous(5.0));
        config.insert("n".to_string(), HyperparameterValue::Discrete(2));
        config.insert("c".to_string(), HyperparameterValue::Categorical("b".to_string()));

        let encoded = space.encode_config(&config);
        assert_eq!(encoded.len(), 3);
        assert!((encoded[0] - 0.5).abs() < 1e-10, "x=5 in [0,10] -> 0.5");
        assert!((encoded[1] - 0.5).abs() < 1e-10, "n=2 is index 1 of 3 -> 0.5");
        assert!((encoded[2] - 1.0).abs() < 1e-10, "c=b is index 1 of 2 -> 1.0");
    }

    #[test]
    fn test_trial_history() {
        let mut history = TrialHistory::new();
        assert!(history.is_empty());

        let mut c1 = Config::new();
        c1.insert("x".to_string(), HyperparameterValue::Continuous(1.0));
        history.add(c1, 0.5);

        let mut c2 = Config::new();
        c2.insert("x".to_string(), HyperparameterValue::Continuous(2.0));
        history.add(c2, 0.8);

        assert_eq!(history.len(), 2);
        let best = history.best().unwrap();
        assert!((best.score - 0.8).abs() < 1e-10);
    }
}

// Re-export for use by rand in external code
pub use rand::SeedableRng;
