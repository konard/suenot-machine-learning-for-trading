use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian Process with RBF kernel
// ─────────────────────────────────────────────────────────────────────────────

/// RBF (Squared Exponential) kernel parameters.
#[derive(Debug, Clone)]
pub struct RbfKernel {
    pub length_scale: f64,
    pub signal_variance: f64,
}

impl Default for RbfKernel {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            signal_variance: 1.0,
        }
    }
}

impl RbfKernel {
    /// Compute k(x, x').
    pub fn compute(&self, x: &[f64], xp: &[f64]) -> f64 {
        let sq_dist: f64 = x.iter().zip(xp.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        self.signal_variance * (-sq_dist / (2.0 * self.length_scale.powi(2))).exp()
    }
}

/// A Gaussian Process regression model.
#[derive(Debug, Clone)]
pub struct GaussianProcess {
    pub kernel: RbfKernel,
    pub noise_variance: f64,
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    // Cached: (K + σ²I)⁻¹ y
    alpha: Vec<f64>,
    // Cached: (K + σ²I)⁻¹  stored row-major
    k_inv: Vec<Vec<f64>>,
}

impl GaussianProcess {
    pub fn new(kernel: RbfKernel, noise_variance: f64) -> Self {
        Self {
            kernel,
            noise_variance,
            x_train: Vec::new(),
            y_train: Vec::new(),
            alpha: Vec::new(),
            k_inv: Vec::new(),
        }
    }

    /// Fit the GP to observed data. Computes and caches the inverse kernel matrix.
    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> Result<()> {
        let n = x.len();
        if n == 0 {
            return Err(anyhow!("Cannot fit GP with zero observations"));
        }
        if n != y.len() {
            return Err(anyhow!("x and y must have the same length"));
        }

        // Build kernel matrix K + σ²I
        let mut k = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                k[[i, j]] = self.kernel.compute(&x[i], &x[j]);
            }
            k[[i, i]] += self.noise_variance;
        }

        // Invert using Cholesky-like approach (simple for small n)
        let k_inv = invert_matrix(&k)?;

        // alpha = K_inv * y
        let y_arr = Array1::from(y.clone());
        let alpha_arr = k_inv.dot(&y_arr);

        self.x_train = x;
        self.y_train = y;
        self.alpha = alpha_arr.to_vec();
        self.k_inv = (0..n).map(|i| (0..n).map(|j| k_inv[[i, j]]).collect()).collect();

        Ok(())
    }

    /// Predict mean and variance at a new point.
    pub fn predict(&self, x_star: &[f64]) -> (f64, f64) {
        if self.x_train.is_empty() {
            return (0.0, self.kernel.signal_variance);
        }

        let n = self.x_train.len();

        // k_star = [k(x*, x_1), ..., k(x*, x_n)]
        let k_star: Vec<f64> = (0..n)
            .map(|i| self.kernel.compute(x_star, &self.x_train[i]))
            .collect();

        // mean = k_star^T * alpha
        let mean: f64 = k_star.iter().zip(self.alpha.iter()).map(|(a, b)| a * b).sum();

        // variance = k(x*, x*) - k_star^T * K_inv * k_star
        let k_ss = self.kernel.compute(x_star, x_star);
        let mut v = 0.0;
        for i in 0..n {
            for j in 0..n {
                v += k_star[i] * self.k_inv[i][j] * k_star[j];
            }
        }
        let variance = (k_ss - v).max(1e-10);

        (mean, variance)
    }

    /// Return number of training points.
    pub fn n_samples(&self) -> usize {
        self.x_train.len()
    }
}

/// Simple matrix inversion via Gauss-Jordan elimination.
fn invert_matrix(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(anyhow!("Matrix must be square"));
    }

    // Augment with identity
    let mut aug = Array2::<f64>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return Err(anyhow!("Matrix is singular or near-singular"));
        }

        // Swap rows
        if max_row != col {
            for j in 0..(2 * n) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Scale pivot row
        let pivot = aug[[col, col]];
        for j in 0..(2 * n) {
            aug[[col, j]] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[[row, col]];
            for j in 0..(2 * n) {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    // Extract inverse
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    Ok(inv)
}

// ─────────────────────────────────────────────────────────────────────────────
// Acquisition Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Available acquisition functions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AcquisitionFunction {
    ExpectedImprovement,
    UpperConfidenceBound { kappa: f64 },
    ProbabilityOfImprovement,
}

/// Standard normal PDF.
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard normal CDF (approximation).
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz & Stegun 7.1.26).
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Evaluate the acquisition function at a point given GP predictions.
pub fn evaluate_acquisition(
    acq: AcquisitionFunction,
    mean: f64,
    variance: f64,
    f_best: f64,
) -> f64 {
    let sigma = variance.sqrt();
    if sigma < 1e-10 {
        return 0.0;
    }

    match acq {
        AcquisitionFunction::ExpectedImprovement => {
            let z = (mean - f_best) / sigma;
            (mean - f_best) * norm_cdf(z) + sigma * norm_pdf(z)
        }
        AcquisitionFunction::UpperConfidenceBound { kappa } => mean + kappa * sigma,
        AcquisitionFunction::ProbabilityOfImprovement => {
            let z = (mean - f_best) / sigma;
            norm_cdf(z)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bayesian Optimizer
// ─────────────────────────────────────────────────────────────────────────────

/// Search space bounds: (lower, upper) for each dimension.
pub type Bounds = Vec<(f64, f64)>;

/// Result of a Bayesian optimization run.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub best_params: Vec<f64>,
    pub best_value: f64,
    pub all_params: Vec<Vec<f64>>,
    pub all_values: Vec<f64>,
    pub n_iterations: usize,
}

/// Bayesian Optimizer using GP surrogate and configurable acquisition function.
pub struct BayesianOptimizer {
    pub bounds: Bounds,
    pub acquisition: AcquisitionFunction,
    pub n_initial: usize,
    pub n_candidates: usize,
    pub gp: GaussianProcess,
}

impl BayesianOptimizer {
    pub fn new(bounds: Bounds, acquisition: AcquisitionFunction) -> Self {
        let kernel = RbfKernel {
            length_scale: 1.0,
            signal_variance: 1.0,
        };
        Self {
            bounds,
            acquisition,
            n_initial: 5,
            n_candidates: 1000,
            gp: GaussianProcess::new(kernel, 1e-6),
        }
    }

    /// Set number of initial random evaluations.
    pub fn with_n_initial(mut self, n: usize) -> Self {
        self.n_initial = n;
        self
    }

    /// Set number of random candidates for acquisition optimization.
    pub fn with_n_candidates(mut self, n: usize) -> Self {
        self.n_candidates = n;
        self
    }

    /// Set the GP kernel length scale.
    pub fn with_length_scale(mut self, ls: f64) -> Self {
        self.gp.kernel.length_scale = ls;
        self
    }

    /// Sample a random point within bounds.
    fn random_point(&self, rng: &mut impl Rng) -> Vec<f64> {
        self.bounds
            .iter()
            .map(|(lo, hi)| rng.gen_range(*lo..=*hi))
            .collect()
    }

    /// Normalize parameters to [0, 1] for the GP.
    fn normalize(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(self.bounds.iter())
            .map(|(v, (lo, hi))| {
                if (hi - lo).abs() < 1e-12 {
                    0.5
                } else {
                    (v - lo) / (hi - lo)
                }
            })
            .collect()
    }

    /// Run the optimization loop.
    ///
    /// `objective` is the function to maximize. It receives raw (unnormalized) parameters.
    /// `n_iterations` is the total evaluation budget (including initial random evaluations).
    pub fn optimize<F>(&mut self, objective: F, n_iterations: usize) -> OptimizationResult
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng = rand::thread_rng();
        let mut all_params: Vec<Vec<f64>> = Vec::new();
        let mut all_values: Vec<f64> = Vec::new();

        // Phase 1: Random initialization
        let n_init = self.n_initial.min(n_iterations);
        for _ in 0..n_init {
            let x = self.random_point(&mut rng);
            let y = objective(&x);
            all_params.push(x);
            all_values.push(y);
        }

        // Phase 2: BO iterations
        for _ in n_init..n_iterations {
            // Normalize training data for GP
            let x_norm: Vec<Vec<f64>> = all_params.iter().map(|x| self.normalize(x)).collect();
            let y_vals = all_values.clone();

            // Fit GP
            if self.gp.fit(x_norm, y_vals).is_err() {
                // Fallback to random if GP fails
                let x = self.random_point(&mut rng);
                let y = objective(&x);
                all_params.push(x);
                all_values.push(y);
                continue;
            }

            // Find f_best
            let f_best = all_values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            // Optimize acquisition function via random sampling
            let mut best_acq = f64::NEG_INFINITY;
            let mut best_x = self.random_point(&mut rng);

            for _ in 0..self.n_candidates {
                let x_raw = self.random_point(&mut rng);
                let x_norm = self.normalize(&x_raw);
                let (mean, var) = self.gp.predict(&x_norm);
                let acq_val = evaluate_acquisition(self.acquisition, mean, var, f_best);
                if acq_val > best_acq {
                    best_acq = acq_val;
                    best_x = x_raw;
                }
            }

            let y = objective(&best_x);
            all_params.push(best_x);
            all_values.push(y);
        }

        // Find best
        let (best_idx, &best_value) = all_values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let best_params = all_params[best_idx].clone();

        OptimizationResult {
            best_params,
            best_value,
            all_params,
            all_values,
            n_iterations,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trading Backtester (MA Crossover)
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a backtest.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub sharpe_ratio: f64,
    pub total_return: f64,
    pub max_drawdown: f64,
    pub n_trades: usize,
}

/// A simple MA crossover backtester.
pub struct TradingBacktester {
    pub risk_free_rate: f64,
}

impl Default for TradingBacktester {
    fn default() -> Self {
        Self {
            risk_free_rate: 0.0,
        }
    }
}

impl TradingBacktester {
    pub fn new(risk_free_rate: f64) -> Self {
        Self { risk_free_rate }
    }

    /// Compute a simple moving average.
    fn sma(prices: &[f64], window: usize) -> Vec<f64> {
        if window == 0 || prices.len() < window {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(prices.len() - window + 1);
        let mut sum: f64 = prices[..window].iter().sum();
        result.push(sum / window as f64);
        for i in window..prices.len() {
            sum += prices[i] - prices[i - window];
            result.push(sum / window as f64);
        }
        result
    }

    /// Run MA crossover backtest.
    ///
    /// Returns the Sharpe ratio as the objective to maximize.
    /// If fast_window >= slow_window, returns a large negative penalty.
    pub fn run(&self, fast_window: usize, slow_window: usize, prices: &[f64]) -> BacktestResult {
        let penalty = BacktestResult {
            sharpe_ratio: -10.0,
            total_return: -1.0,
            max_drawdown: 1.0,
            n_trades: 0,
        };

        if fast_window >= slow_window || fast_window == 0 || slow_window == 0 {
            return penalty;
        }
        if prices.len() < slow_window + 1 {
            return penalty;
        }

        let fast_ma = Self::sma(prices, fast_window);
        let slow_ma = Self::sma(prices, slow_window);

        // Align: slow_ma starts at index (slow_window - 1),
        // fast_ma starts at index (fast_window - 1).
        // We need both to be available.
        let offset = slow_window - fast_window;
        if offset >= fast_ma.len() {
            return penalty;
        }

        let fast_aligned = &fast_ma[offset..];
        let len = fast_aligned.len().min(slow_ma.len());
        if len < 2 {
            return penalty;
        }

        // Generate signals: 1.0 = long, -1.0 = short/flat
        let mut position = 0.0_f64;
        let mut returns = Vec::with_capacity(len - 1);
        let mut n_trades = 0_usize;

        // Price series aligned with the MA signals
        let price_start = slow_window - 1;

        for i in 0..(len - 1) {
            let new_position = if fast_aligned[i] > slow_ma[i] {
                1.0
            } else {
                -1.0
            };
            if new_position != position {
                n_trades += 1;
                position = new_position;
            }

            let price_idx = price_start + i;
            if price_idx + 1 < prices.len() {
                let ret = (prices[price_idx + 1] - prices[price_idx]) / prices[price_idx];
                returns.push(position * ret);
            }
        }

        if returns.is_empty() {
            return penalty;
        }

        // Compute metrics
        let total_return: f64 = returns.iter().fold(1.0, |acc, r| acc * (1.0 + r)) - 1.0;

        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
            / returns.len() as f64;
        let std_dev = variance.sqrt();

        let sharpe_ratio = if std_dev > 1e-10 {
            (mean_return - self.risk_free_rate / 252.0) / std_dev * (252.0_f64).sqrt()
        } else {
            0.0
        };

        // Max drawdown
        let mut peak = 1.0_f64;
        let mut max_dd = 0.0_f64;
        let mut equity = 1.0_f64;
        for r in &returns {
            equity *= 1.0 + r;
            if equity > peak {
                peak = equity;
            }
            let dd = (peak - equity) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        BacktestResult {
            sharpe_ratio,
            total_return,
            max_drawdown: max_dd,
            n_trades,
        }
    }

    /// Run backtest and return only the Sharpe ratio (for use as BO objective).
    pub fn sharpe_objective(
        &self,
        fast_window: usize,
        slow_window: usize,
        prices: &[f64],
    ) -> f64 {
        self.run(fast_window, slow_window, prices).sharpe_ratio
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bybit API Client
// ─────────────────────────────────────────────────────────────────────────────

/// OHLCV candle data.
#[derive(Debug, Clone)]
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

/// Client for fetching market data from Bybit.
pub struct BybitClient {
    base_url: String,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }
}

impl BybitClient {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fetch historical klines (candles) from Bybit.
    ///
    /// - `symbol`: e.g. "BTCUSDT"
    /// - `interval`: e.g. "60" for 1h, "D" for daily
    /// - `limit`: number of candles (max 1000)
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: BybitResponse = reqwest::blocking::get(&url)?.json()?;

        if response.ret_code != 0 {
            return Err(anyhow!(
                "Bybit API error: {} (code {})",
                response.ret_msg,
                response.ret_code
            ));
        }

        let mut candles: Vec<Candle> = response
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

        // Bybit returns newest first; reverse to chronological order
        candles.reverse();
        Ok(candles)
    }

    /// Fetch close prices for a symbol.
    pub fn fetch_close_prices(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<f64>> {
        let candles = self.fetch_klines(symbol, interval, limit)?;
        Ok(candles.iter().map(|c| c.close).collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hyperparameter Search Space
// ─────────────────────────────────────────────────────────────────────────────

/// Definition of a hyperparameter with name and bounds.
#[derive(Debug, Clone)]
pub struct HyperParameter {
    pub name: String,
    pub lower: f64,
    pub upper: f64,
    pub is_integer: bool,
}

impl HyperParameter {
    pub fn new(name: &str, lower: f64, upper: f64, is_integer: bool) -> Self {
        Self {
            name: name.to_string(),
            lower,
            upper,
            is_integer,
        }
    }

    /// Round to integer if this is an integer parameter.
    pub fn cast(&self, value: f64) -> f64 {
        if self.is_integer {
            value.round()
        } else {
            value
        }
    }
}

/// Search space for multiple hyperparameters.
#[derive(Debug, Clone)]
pub struct SearchSpace {
    pub params: Vec<HyperParameter>,
}

impl SearchSpace {
    pub fn new() -> Self {
        Self { params: Vec::new() }
    }

    pub fn add(mut self, param: HyperParameter) -> Self {
        self.params.push(param);
        self
    }

    /// Get bounds as Vec<(f64, f64)> for the optimizer.
    pub fn bounds(&self) -> Bounds {
        self.params.iter().map(|p| (p.lower, p.upper)).collect()
    }

    /// Cast raw parameter values according to type constraints.
    pub fn cast_params(&self, raw: &[f64]) -> Vec<f64> {
        raw.iter()
            .zip(self.params.iter())
            .map(|(v, p)| p.cast(*v))
            .collect()
    }
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbf_kernel_same_point() {
        let kernel = RbfKernel::default();
        let x = vec![1.0, 2.0, 3.0];
        let k = kernel.compute(&x, &x);
        assert!((k - 1.0).abs() < 1e-10, "k(x,x) should be signal_variance");
    }

    #[test]
    fn test_rbf_kernel_different_points() {
        let kernel = RbfKernel {
            length_scale: 1.0,
            signal_variance: 1.0,
        };
        let x1 = vec![0.0];
        let x2 = vec![1.0];
        let k = kernel.compute(&x1, &x2);
        let expected = (-0.5_f64).exp();
        assert!((k - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gp_fit_and_predict() {
        let kernel = RbfKernel {
            length_scale: 0.5,
            signal_variance: 1.0,
        };
        let mut gp = GaussianProcess::new(kernel, 1e-6);
        let x = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y = vec![0.0, 1.0, 0.0];
        gp.fit(x, y).unwrap();

        // At x=1.0 should predict close to 1.0
        let (mean, var) = gp.predict(&[1.0]);
        assert!(
            (mean - 1.0).abs() < 0.1,
            "Mean at observed point should be close to observation: got {}",
            mean
        );
        assert!(var < 0.1, "Variance at observed point should be small: got {}", var);

        // At x=0.5 variance should be larger than at observed point
        let (_, var_mid) = gp.predict(&[0.5]);
        assert!(var_mid > var, "Variance between points should be larger");
    }

    #[test]
    fn test_acquisition_expected_improvement() {
        // When mean > f_best with low variance, EI should be positive
        let ei = evaluate_acquisition(AcquisitionFunction::ExpectedImprovement, 1.5, 0.01, 1.0);
        assert!(ei > 0.0, "EI should be positive when mean > f_best");

        // When sigma is ~0, EI should be ~0
        let ei_zero = evaluate_acquisition(
            AcquisitionFunction::ExpectedImprovement,
            0.5,
            1e-20,
            1.0,
        );
        assert!(
            ei_zero.abs() < 1e-5,
            "EI should be ~0 when variance is ~0"
        );
    }

    #[test]
    fn test_acquisition_ucb() {
        let ucb = evaluate_acquisition(
            AcquisitionFunction::UpperConfidenceBound { kappa: 2.0 },
            1.0,
            0.25,
            0.5,
        );
        // UCB = 1.0 + 2.0 * sqrt(0.25) = 1.0 + 1.0 = 2.0
        assert!(
            (ucb - 2.0).abs() < 1e-10,
            "UCB should be mean + kappa * sigma: got {}",
            ucb
        );
    }

    #[test]
    fn test_backtester_penalty_for_invalid_params() {
        let bt = TradingBacktester::default();
        let prices: Vec<f64> = (0..200).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();

        // fast >= slow should give penalty
        let result = bt.run(50, 20, &prices);
        assert!(result.sharpe_ratio < -5.0, "Should return penalty for fast >= slow");

        // zero windows should give penalty
        let result = bt.run(0, 50, &prices);
        assert!(result.sharpe_ratio < -5.0, "Should return penalty for zero window");
    }

    #[test]
    fn test_backtester_valid_run() {
        let bt = TradingBacktester::default();
        // Create trending price data
        let prices: Vec<f64> = (0..500)
            .map(|i| 100.0 + i as f64 * 0.5 + (i as f64 * 0.1).sin() * 5.0)
            .collect();

        let result = bt.run(10, 50, &prices);
        assert!(
            result.sharpe_ratio > -10.0,
            "Sharpe ratio should not be penalty value"
        );
        assert!(result.n_trades > 0, "Should have some trades");
        assert!(
            result.max_drawdown >= 0.0 && result.max_drawdown <= 1.0,
            "Max drawdown should be in [0, 1]"
        );
    }

    #[test]
    fn test_bayesian_optimizer_simple() {
        // Optimize a simple 1D function: f(x) = -(x - 3)^2 + 10
        let bounds = vec![(0.0, 6.0)];
        let mut optimizer = BayesianOptimizer::new(bounds, AcquisitionFunction::ExpectedImprovement);
        optimizer.n_initial = 3;
        optimizer.n_candidates = 200;

        let result = optimizer.optimize(|x| -(x[0] - 3.0).powi(2) + 10.0, 20);

        assert!(
            (result.best_params[0] - 3.0).abs() < 1.5,
            "Should find near x=3: got {}",
            result.best_params[0]
        );
        assert!(
            result.best_value > 8.0,
            "Best value should be near 10: got {}",
            result.best_value
        );
    }

    #[test]
    fn test_search_space() {
        let space = SearchSpace::new()
            .add(HyperParameter::new("fast_window", 5.0, 50.0, true))
            .add(HyperParameter::new("slow_window", 20.0, 200.0, true));

        let bounds = space.bounds();
        assert_eq!(bounds.len(), 2);
        assert_eq!(bounds[0], (5.0, 50.0));
        assert_eq!(bounds[1], (20.0, 200.0));

        let casted = space.cast_params(&[10.7, 55.3]);
        assert_eq!(casted[0], 11.0);
        assert_eq!(casted[1], 55.0);
    }

    #[test]
    fn test_norm_cdf_values() {
        // CDF(0) = 0.5
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-6);
        // CDF(-inf) ~ 0, CDF(inf) ~ 1
        assert!(norm_cdf(-5.0) < 1e-4);
        assert!(norm_cdf(5.0) > 1.0 - 1e-4);
    }

    #[test]
    fn test_matrix_inversion() {
        // 2x2 identity should invert to itself
        let eye = Array2::eye(2);
        let inv = invert_matrix(&eye).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[[i, j]] - expected).abs() < 1e-10);
            }
        }

        // Known 2x2 matrix
        let mut m = Array2::<f64>::zeros((2, 2));
        m[[0, 0]] = 4.0;
        m[[0, 1]] = 7.0;
        m[[1, 0]] = 2.0;
        m[[1, 1]] = 6.0;
        let inv = invert_matrix(&m).unwrap();
        // det = 4*6 - 7*2 = 10
        assert!((inv[[0, 0]] - 0.6).abs() < 1e-10);
        assert!((inv[[0, 1]] - (-0.7)).abs() < 1e-10);
        assert!((inv[[1, 0]] - (-0.2)).abs() < 1e-10);
        assert!((inv[[1, 1]] - 0.4).abs() < 1e-10);
    }
}
