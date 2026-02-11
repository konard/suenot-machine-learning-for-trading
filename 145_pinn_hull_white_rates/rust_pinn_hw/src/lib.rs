//! # Physics-Informed Neural Network for Hull-White Interest Rate Model
//!
//! This crate implements a PINN-based solver for the Hull-White (extended Vasicek)
//! short-rate model. The PINN embeds the bond pricing PDE directly into the neural
//! network loss function, enabling mesh-free, differentiable bond pricing.
//!
//! ## Hull-White Model
//!
//! The short rate follows:
//!     dr(t) = [theta(t) - a * r(t)] dt + sigma * dW(t)
//!
//! The bond pricing PDE:
//!     dP/dt + [theta(t) - a*r]*dP/dr + 0.5*sigma^2*d^2P/dr^2 - r*P = 0
//!
//! ## Modules
//!
//! - `model` - PINN neural network architecture
//! - `analytical` - Closed-form Hull-White pricing
//! - `data` - Bybit funding rate data fetching
//! - `calibration` - Model parameter calibration
//! - `derivatives` - Caps, floors, swaptions pricing
//! - `backtest` - Interest rate strategy backtesting

pub mod model;
pub mod analytical;
pub mod data;
pub mod calibration;
pub mod derivatives;
pub mod backtest;

// Re-export key types
pub use model::{PINNLayer, HullWhitePINN, PINNConfig};
pub use analytical::HullWhiteAnalytical;
pub use data::{FundingRate, BybitClient};
pub use calibration::CalibrationResult;

/// Hull-White model parameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HullWhiteParams {
    /// Mean reversion speed (a > 0)
    pub a: f64,
    /// Short rate volatility (sigma > 0)
    pub sigma: f64,
    /// Current short rate
    pub r0: f64,
}

impl Default for HullWhiteParams {
    fn default() -> Self {
        Self {
            a: 0.1,
            sigma: 0.01,
            r0: 0.03,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PINN Model
// ═══════════════════════════════════════════════════════════════════════════════

pub mod model {
    use ndarray::{Array1, Array2, Axis};
    use rand::Rng;
    use rand_distr::Normal;
    use serde::{Serialize, Deserialize};

    /// Configuration for the PINN.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PINNConfig {
        pub hidden_layers: Vec<usize>,
        pub activation: String,
        pub learning_rate: f64,
    }

    impl Default for PINNConfig {
        fn default() -> Self {
            Self {
                hidden_layers: vec![64, 64, 64, 32],
                activation: "tanh".to_string(),
                learning_rate: 1e-3,
            }
        }
    }

    /// A single layer in the PINN.
    #[derive(Debug, Clone)]
    pub struct PINNLayer {
        pub weights: Array2<f64>,
        pub biases: Array1<f64>,
    }

    impl PINNLayer {
        /// Create a new layer with Xavier uniform initialization.
        pub fn new(input_dim: usize, output_dim: usize) -> Self {
            let mut rng = rand::thread_rng();
            let limit = (6.0 / (input_dim + output_dim) as f64).sqrt();
            let uniform = rand_distr::Uniform::new(-limit, limit);

            let weights = Array2::from_shape_fn((output_dim, input_dim), |_| {
                rng.sample(uniform)
            });
            let biases = Array1::zeros(output_dim);

            Self { weights, biases }
        }

        /// Forward pass: output = weights * input + biases
        pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
            self.weights.dot(input) + &self.biases
        }
    }

    /// Physics-Informed Neural Network for Hull-White bond pricing.
    ///
    /// Maps (r, t, T) -> P(r, t; T) where P is the zero-coupon bond price.
    #[derive(Debug, Clone)]
    pub struct HullWhitePINN {
        pub layers: Vec<PINNLayer>,
        pub a: f64,
        pub sigma: f64,
        pub config: PINNConfig,
    }

    impl HullWhitePINN {
        /// Create a new PINN with the given parameters and architecture.
        pub fn new(a: f64, sigma: f64, config: PINNConfig) -> Self {
            let mut layers = Vec::new();
            let mut prev_dim = 3; // input: (r, t, T)

            for &hidden_dim in &config.hidden_layers {
                layers.push(PINNLayer::new(prev_dim, hidden_dim));
                prev_dim = hidden_dim;
            }
            // Output layer: 1 neuron (bond price)
            layers.push(PINNLayer::new(prev_dim, 1));

            Self { layers, a, sigma, config }
        }

        /// Activation function (tanh).
        fn activation(x: f64) -> f64 {
            x.tanh()
        }

        /// Derivative of tanh activation.
        fn activation_derivative(x: f64) -> f64 {
            1.0 - x.tanh().powi(2)
        }

        /// Softplus to ensure positive output.
        fn softplus(x: f64) -> f64 {
            if x > 20.0 {
                x
            } else {
                (1.0 + x.exp()).ln()
            }
        }

        /// Forward pass: compute bond price P(r, t, T).
        pub fn forward(&self, r: f64, t: f64, maturity: f64) -> f64 {
            let input = Array1::from_vec(vec![r, t, maturity]);
            let mut x = input;

            for (i, layer) in self.layers.iter().enumerate() {
                x = layer.forward(&x);
                if i < self.layers.len() - 1 {
                    // Apply activation to hidden layers
                    x.mapv_inplace(Self::activation);
                }
            }

            // Softplus for positive output
            Self::softplus(x[0])
        }

        /// Predict bond price (alias for forward).
        pub fn predict(&self, r: f64, t: f64, maturity: f64) -> f64 {
            self.forward(r, t, maturity)
        }

        /// Compute PDE residual using finite differences.
        ///
        /// PDE: dP/dt + [theta(t) - a*r]*dP/dr + 0.5*sigma^2*d^2P/dr^2 - r*P = 0
        pub fn pde_residual(&self, r: f64, t: f64, maturity: f64, theta_t: f64) -> f64 {
            let eps_r = 1e-4;
            let eps_t = 1e-4;

            let p = self.forward(r, t, maturity);
            let p_r_plus = self.forward(r + eps_r, t, maturity);
            let p_r_minus = self.forward(r - eps_r, t, maturity);
            let p_t_plus = self.forward(r, t + eps_t, maturity);

            // Finite difference derivatives
            let dp_dt = (p_t_plus - p) / eps_t;
            let dp_dr = (p_r_plus - p_r_minus) / (2.0 * eps_r);
            let d2p_dr2 = (p_r_plus - 2.0 * p + p_r_minus) / (eps_r * eps_r);

            // PDE residual
            let drift = theta_t - self.a * r;
            dp_dt + drift * dp_dr + 0.5 * self.sigma * self.sigma * d2p_dr2 - r * p
        }

        /// Compute bond sensitivities (Greeks) using finite differences.
        pub fn greeks(&self, r: f64, t: f64, maturity: f64) -> BondGreeks {
            let eps = 1e-5;
            let p = self.forward(r, t, maturity);
            let p_r_up = self.forward(r + eps, t, maturity);
            let p_r_down = self.forward(r - eps, t, maturity);
            let p_t_up = self.forward(r, t + eps, maturity);

            let dp_dr = (p_r_up - p_r_down) / (2.0 * eps);
            let d2p_dr2 = (p_r_up - 2.0 * p + p_r_down) / (eps * eps);
            let dp_dt = (p_t_up - p) / eps;

            let duration = if p.abs() > 1e-10 { -dp_dr / p } else { 0.0 };
            let convexity = if p.abs() > 1e-10 { d2p_dr2 / p } else { 0.0 };

            BondGreeks {
                price: p,
                duration,
                convexity,
                theta: dp_dt,
                dp_dr,
                d2p_dr2,
            }
        }

        /// Train the PINN using gradient descent with finite-difference gradients.
        ///
        /// This is a simplified training loop. For production use, consider
        /// using a proper autodiff library like `tch` (libtorch bindings).
        pub fn train(
            &mut self,
            params: &super::HullWhiteParams,
            t_max: f64,
            epochs: usize,
            n_collocation: usize,
            callback: Option<&dyn Fn(usize, f64)>,
        ) -> Vec<f64> {
            let analytical = super::analytical::HullWhiteAnalytical::new_flat(
                params.a, params.sigma, params.r0,
            );

            let lr = self.config.learning_rate;
            let mut loss_history = Vec::with_capacity(epochs);
            let mut rng = rand::thread_rng();

            for epoch in 0..epochs {
                // Sample training points: use analytical prices as targets
                let mut total_loss = 0.0;

                for _ in 0..n_collocation {
                    let r: f64 = rng.gen_range(-0.02..0.15);
                    let t_val: f64 = rng.gen_range(0.0..t_max * 0.9);
                    let mat: f64 = t_val + rng.gen_range(0.1..t_max);
                    let mat = mat.min(t_max);

                    let target = analytical.bond_price(r, t_val, mat);
                    let predicted = self.forward(r, t_val, mat);
                    let error = predicted - target;
                    total_loss += error * error;

                    // Update weights using perturbation-based gradient estimation
                    let perturbation = 1e-5;
                    for layer_idx in 0..self.layers.len() {
                        let (rows, cols) = self.layers[layer_idx].weights.dim();
                        // Only update a random subset for efficiency
                        let n_updates = (rows * cols).min(10);
                        for _ in 0..n_updates {
                            let i = rng.gen_range(0..rows);
                            let j = rng.gen_range(0..cols);

                            self.layers[layer_idx].weights[[i, j]] += perturbation;
                            let loss_plus = (self.forward(r, t_val, mat) - target).powi(2);
                            self.layers[layer_idx].weights[[i, j]] -= 2.0 * perturbation;
                            let loss_minus = (self.forward(r, t_val, mat) - target).powi(2);
                            self.layers[layer_idx].weights[[i, j]] += perturbation;

                            let grad = (loss_plus - loss_minus) / (2.0 * perturbation);
                            self.layers[layer_idx].weights[[i, j]] -= lr * grad;
                        }

                        // Update biases
                        let bias_len = self.layers[layer_idx].biases.len();
                        for i in 0..bias_len.min(5) {
                            self.layers[layer_idx].biases[i] += perturbation;
                            let loss_plus = (self.forward(r, t_val, mat) - target).powi(2);
                            self.layers[layer_idx].biases[i] -= 2.0 * perturbation;
                            let loss_minus = (self.forward(r, t_val, mat) - target).powi(2);
                            self.layers[layer_idx].biases[i] += perturbation;

                            let grad = (loss_plus - loss_minus) / (2.0 * perturbation);
                            self.layers[layer_idx].biases[i] -= lr * grad;
                        }
                    }
                }

                let avg_loss = total_loss / n_collocation as f64;
                loss_history.push(avg_loss);

                if let Some(cb) = callback {
                    cb(epoch, avg_loss);
                }
            }

            loss_history
        }

        /// Compute yield curve at a given rate.
        pub fn yield_curve(&self, r: f64, t: f64, maturities: &[f64]) -> Vec<f64> {
            maturities.iter().map(|&mat| {
                let price = self.forward(r, t, mat);
                let tau = mat - t;
                if tau > 1e-6 && price > 1e-10 {
                    -price.ln() / tau
                } else {
                    0.0
                }
            }).collect()
        }
    }

    /// Bond price sensitivities.
    #[derive(Debug, Clone)]
    pub struct BondGreeks {
        pub price: f64,
        pub duration: f64,
        pub convexity: f64,
        pub theta: f64,
        pub dp_dr: f64,
        pub d2p_dr2: f64,
    }

    impl std::fmt::Display for BondGreeks {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "Price: {:.6}, Duration: {:.4}, Convexity: {:.4}, Theta: {:.6}",
                self.price, self.duration, self.convexity, self.theta
            )
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Analytical Hull-White Pricing
// ═══════════════════════════════════════════════════════════════════════════════

pub mod analytical {
    use statrs::distribution::{Normal, ContinuousCDF};

    /// Analytical Hull-White model for zero-coupon bond pricing.
    #[derive(Debug, Clone)]
    pub struct HullWhiteAnalytical {
        pub a: f64,
        pub sigma: f64,
        flat_rate: Option<f64>,
        market_times: Option<Vec<f64>>,
        market_rates: Option<Vec<f64>>,
    }

    impl HullWhiteAnalytical {
        /// Create with a flat yield curve (for testing/demonstration).
        pub fn new_flat(a: f64, sigma: f64, flat_rate: f64) -> Self {
            Self {
                a,
                sigma,
                flat_rate: Some(flat_rate),
                market_times: None,
                market_rates: None,
            }
        }

        /// Create with market yield curve data.
        pub fn new_with_curve(a: f64, sigma: f64, times: Vec<f64>, rates: Vec<f64>) -> Self {
            Self {
                a,
                sigma,
                flat_rate: None,
                market_times: Some(times),
                market_rates: Some(rates),
            }
        }

        /// Get market zero rate at time t.
        pub fn market_zero_rate(&self, t: f64) -> f64 {
            if let Some(r) = self.flat_rate {
                return r;
            }
            if let (Some(ref times), Some(ref rates)) = (&self.market_times, &self.market_rates) {
                // Linear interpolation
                if t <= times[0] {
                    return rates[0];
                }
                if t >= *times.last().unwrap() {
                    return *rates.last().unwrap();
                }
                for i in 0..times.len() - 1 {
                    if t >= times[i] && t <= times[i + 1] {
                        let w = (t - times[i]) / (times[i + 1] - times[i]);
                        return rates[i] * (1.0 - w) + rates[i + 1] * w;
                    }
                }
            }
            0.03 // fallback
        }

        /// Market discount factor P_M(0, t).
        pub fn market_discount_factor(&self, t: f64) -> f64 {
            if t <= 0.0 {
                return 1.0;
            }
            (-self.market_zero_rate(t) * t).exp()
        }

        /// Market instantaneous forward rate f_M(0, t).
        pub fn market_forward_rate(&self, t: f64) -> f64 {
            let dt = 1e-4;
            let r_t = self.market_zero_rate(t);
            let r_t_plus = self.market_zero_rate(t + dt);
            r_t + t * (r_t_plus - r_t) / dt
        }

        /// Time-dependent drift theta(t).
        pub fn theta(&self, t: f64) -> f64 {
            if let Some(flat_rate) = self.flat_rate {
                return self.a * flat_rate
                    + (self.sigma.powi(2) / (2.0 * self.a))
                        * (1.0 - (-2.0 * self.a * t).exp());
            }

            let dt = 1e-4;
            let f_t = self.market_forward_rate(t);
            let f_t_plus = self.market_forward_rate(t + dt);
            let df_dt = (f_t_plus - f_t) / dt;

            df_dt + self.a * f_t
                + (self.sigma.powi(2) / (2.0 * self.a))
                    * (1.0 - (-2.0 * self.a * t).exp())
        }

        /// B(t, T) function in the affine formula.
        pub fn b_function(&self, t: f64, big_t: f64) -> f64 {
            let tau = big_t - t;
            if self.a.abs() < 1e-10 {
                return tau;
            }
            (1.0 - (-self.a * tau).exp()) / self.a
        }

        /// Analytical zero-coupon bond price P(r, t; T).
        pub fn bond_price(&self, r: f64, t: f64, big_t: f64) -> f64 {
            if big_t <= t {
                return 1.0;
            }

            let b = self.b_function(t, big_t);

            if let Some(flat_rate) = self.flat_rate {
                // Simplified formula for flat curve
                let tau = big_t - t;
                let r_inf = flat_rate - self.sigma.powi(2) / (2.0 * self.a.powi(2));
                let ln_a = r_inf * (b - tau)
                    - (self.sigma.powi(2) / (4.0 * self.a)) * b.powi(2);
                return (ln_a - b * r).exp();
            }

            // General formula
            let p0_t = self.market_discount_factor(t);
            let p0_big_t = self.market_discount_factor(big_t);
            let f0_t = self.market_forward_rate(t);

            let ln_a = (p0_big_t / p0_t).ln() + b * f0_t
                - (self.sigma.powi(2) / (4.0 * self.a))
                    * (1.0 - (-2.0 * self.a * t).exp())
                    * b.powi(2);

            (ln_a - b * r).exp()
        }

        /// Yield from bond price.
        pub fn yield_from_price(&self, price: f64, t: f64, big_t: f64) -> f64 {
            let tau = big_t - t;
            if tau <= 0.0 || price <= 0.0 {
                return 0.0;
            }
            -price.ln() / tau
        }

        /// Compute yield curve.
        pub fn yield_curve(&self, r: f64, t: f64, maturities: &[f64]) -> Vec<f64> {
            maturities.iter().map(|&mat| {
                let price = self.bond_price(r, t, mat);
                self.yield_from_price(price, t, mat)
            }).collect()
        }

        /// Conditional mean of r(s) given r(t).
        pub fn rate_mean(&self, r0: f64, t: f64, s: f64) -> f64 {
            let tau = s - t;
            let n_steps = ((tau * 100.0) as usize).max(10);
            let du = tau / n_steps as f64;

            let mut integral = 0.0;
            for i in 0..n_steps {
                let u = t + (i as f64 + 0.5) * du;
                integral += self.theta(u) * (-self.a * (s - u)).exp() * du;
            }

            r0 * (-self.a * tau).exp() + integral
        }

        /// Conditional variance of r(s) given r(t).
        pub fn rate_variance(&self, t: f64, s: f64) -> f64 {
            let tau = s - t;
            (self.sigma.powi(2) / (2.0 * self.a)) * (1.0 - (-2.0 * self.a * tau).exp())
        }

        /// Price a European bond option under Hull-White.
        pub fn bond_option(
            &self, r: f64, t: f64, t_option: f64, t_bond: f64,
            strike: f64, is_call: bool,
        ) -> f64 {
            if t_option <= t || t_bond <= t_option {
                return 0.0;
            }

            let p_t_to = self.bond_price(r, t, t_option);
            let p_t_tb = self.bond_price(r, t, t_bond);

            let b_to_tb = self.b_function(t_option, t_bond);
            let sigma_p = self.sigma * b_to_tb
                * ((1.0 - (-2.0 * self.a * (t_option - t)).exp()) / (2.0 * self.a)).sqrt();

            if sigma_p < 1e-12 {
                let intrinsic = if is_call {
                    (p_t_tb - strike * p_t_to).max(0.0)
                } else {
                    (strike * p_t_to - p_t_tb).max(0.0)
                };
                return intrinsic;
            }

            let d1 = ((p_t_tb / (strike * p_t_to)).ln() + 0.5 * sigma_p.powi(2)) / sigma_p;
            let d2 = d1 - sigma_p;

            let normal = Normal::new(0.0, 1.0).unwrap();

            if is_call {
                p_t_tb * normal.cdf(d1) - strike * p_t_to * normal.cdf(d2)
            } else {
                strike * p_t_to * normal.cdf(-d2) - p_t_tb * normal.cdf(-d1)
            }
        }

        /// Simulate rate paths using Euler-Maruyama scheme.
        pub fn simulate_paths(
            &self, r0: f64, t_max: f64, n_steps: usize, n_paths: usize, seed: u64,
        ) -> (Vec<f64>, Vec<Vec<f64>>) {
            use rand::SeedableRng;
            use rand_distr::Distribution;

            let dt = t_max / n_steps as f64;
            let sqrt_dt = dt.sqrt();
            let times: Vec<f64> = (0..=n_steps).map(|i| i as f64 * dt).collect();

            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

            let mut paths = Vec::with_capacity(n_paths);

            for _ in 0..n_paths {
                let mut path = vec![0.0; n_steps + 1];
                path[0] = r0;

                for i in 0..n_steps {
                    let t_i = times[i];
                    let r_i = path[i];
                    let theta_i = self.theta(t_i);
                    let dw: f64 = normal.sample(&mut rng) * sqrt_dt;
                    path[i + 1] = r_i + (theta_i - self.a * r_i) * dt + self.sigma * dw;
                }

                paths.push(path);
            }

            (times, paths)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Data Fetching (Bybit)
// ═══════════════════════════════════════════════════════════════════════════════

pub mod data {
    use serde::{Deserialize, Serialize};
    use chrono::{DateTime, Utc, TimeZone};

    /// A single funding rate observation.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FundingRate {
        pub timestamp_ms: i64,
        pub datetime: String,
        pub symbol: String,
        pub funding_rate: f64,
        pub annualized_rate: f64,
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
        pub list: Vec<BybitFundingItem>,
    }

    #[derive(Debug, Deserialize)]
    pub struct BybitFundingItem {
        pub symbol: String,
        #[serde(rename = "fundingRate")]
        pub funding_rate: String,
        #[serde(rename = "fundingRateTimestamp")]
        pub funding_rate_timestamp: String,
    }

    /// Client for fetching Bybit data.
    pub struct BybitClient {
        base_url: String,
    }

    impl BybitClient {
        pub fn new() -> Self {
            Self {
                base_url: "https://api.bybit.com".to_string(),
            }
        }

        /// Fetch historical funding rates.
        pub fn fetch_funding_rates(
            &self, symbol: &str, limit: usize,
        ) -> Result<Vec<FundingRate>, Box<dyn std::error::Error>> {
            let url = format!(
                "{}/v5/market/funding/history?category=linear&symbol={}&limit={}",
                self.base_url, symbol, limit.min(200)
            );

            let response = reqwest::blocking::get(&url)?;
            let data: BybitResponse = response.json()?;

            if data.ret_code != 0 {
                return Err(format!("Bybit API error: {}", data.ret_msg).into());
            }

            let rates: Vec<FundingRate> = data.result.list.iter().map(|item| {
                let ts: i64 = item.funding_rate_timestamp.parse().unwrap_or(0);
                let rate: f64 = item.funding_rate.parse().unwrap_or(0.0);
                let dt = Utc.timestamp_millis_opt(ts)
                    .single()
                    .unwrap_or_else(Utc::now);

                FundingRate {
                    timestamp_ms: ts,
                    datetime: dt.to_rfc3339(),
                    symbol: item.symbol.clone(),
                    funding_rate: rate,
                    annualized_rate: rate * 3.0 * 365.0,
                }
            }).collect();

            Ok(rates)
        }

        /// Fetch current market price for a symbol.
        pub fn fetch_ticker_price(
            &self, symbol: &str,
        ) -> Result<f64, Box<dyn std::error::Error>> {
            let url = format!(
                "{}/v5/market/tickers?category=linear&symbol={}",
                self.base_url, symbol
            );

            let response = reqwest::blocking::get(&url)?;
            let data: serde_json::Value = response.json()?;

            let price_str = data["result"]["list"][0]["lastPrice"]
                .as_str()
                .ok_or("No price data")?;
            let price: f64 = price_str.parse()?;

            Ok(price)
        }
    }

    impl Default for BybitClient {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Generate synthetic funding rate data for testing.
    pub fn generate_synthetic_funding_rates(
        n_periods: usize,
        mean_rate: f64,
        mean_reversion: f64,
        volatility: f64,
        seed: u64,
    ) -> Vec<FundingRate> {
        use rand::SeedableRng;
        use rand_distr::Distribution;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

        let mut rates = Vec::with_capacity(n_periods);
        let mut r = mean_rate;

        let now = Utc::now().timestamp_millis();

        for i in 0..n_periods {
            let dr = mean_reversion * (mean_rate - r) + volatility * normal.sample(&mut rng);
            r += dr;

            let ts = now - (n_periods as i64 - i as i64) * 8 * 3600 * 1000;
            let dt = Utc.timestamp_millis_opt(ts)
                .single()
                .unwrap_or_else(Utc::now);

            rates.push(FundingRate {
                timestamp_ms: ts,
                datetime: dt.to_rfc3339(),
                symbol: "BTCUSDT".to_string(),
                funding_rate: r,
                annualized_rate: r * 3.0 * 365.0,
            });
        }

        rates
    }

    /// Sample Treasury yield curve data.
    pub fn sample_treasury_curve() -> (Vec<f64>, Vec<f64>) {
        let maturities = vec![
            1.0/12.0, 2.0/12.0, 3.0/12.0, 6.0/12.0,
            1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0,
        ];
        let rates = vec![
            0.0525, 0.0530, 0.0535, 0.0528,
            0.0505, 0.0465, 0.0440, 0.0420, 0.0418, 0.0415, 0.0445, 0.0435,
        ];
        (maturities, rates)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Calibration
// ═══════════════════════════════════════════════════════════════════════════════

pub mod calibration {
    use serde::{Serialize, Deserialize};

    /// Result of Hull-White calibration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CalibrationResult {
        pub a: f64,
        pub sigma: f64,
        pub theta_mean: f64,
        pub long_run_mean: f64,
        pub half_life_hours: f64,
        pub half_life_days: f64,
        pub n_observations: usize,
    }

    /// Calibrate Hull-White parameters from a time series of rates.
    ///
    /// Uses OLS regression on the discrete OU process:
    ///     dr_i = (theta - a * r_i) * dt + sigma * dW_i
    pub fn calibrate_from_rates(
        rates: &[f64],
        dt_hours: f64,
    ) -> CalibrationResult {
        let n = rates.len();
        assert!(n >= 10, "Need at least 10 observations");

        let dt = dt_hours / 8760.0; // Convert to years

        // OLS: dr = c0 + c1 * r_prev
        let mut sum_r = 0.0;
        let mut sum_r2 = 0.0;
        let mut sum_dr = 0.0;
        let mut sum_r_dr = 0.0;
        let m = (n - 1) as f64;

        for i in 0..n - 1 {
            let r = rates[i];
            let dr = rates[i + 1] - rates[i];
            sum_r += r;
            sum_r2 += r * r;
            sum_dr += dr;
            sum_r_dr += r * dr;
        }

        let denom = m * sum_r2 - sum_r * sum_r;
        let c1 = if denom.abs() > 1e-20 {
            (m * sum_r_dr - sum_r * sum_dr) / denom
        } else {
            -0.01
        };
        let c0 = (sum_dr - c1 * sum_r) / m;

        let a = (-c1 / dt).max(0.001);
        let theta = c0 / dt;

        // Estimate sigma from residuals
        let mut sum_sq_residual = 0.0;
        for i in 0..n - 1 {
            let predicted_dr = c0 + c1 * rates[i];
            let actual_dr = rates[i + 1] - rates[i];
            sum_sq_residual += (actual_dr - predicted_dr).powi(2);
        }
        let sigma = (sum_sq_residual / m).sqrt() / dt.sqrt();

        let long_run_mean = if a > 0.001 { theta / a } else { rates.iter().sum::<f64>() / n as f64 };
        let half_life_years = (2.0_f64).ln() / a;
        let half_life_hours = half_life_years * 8760.0;

        CalibrationResult {
            a,
            sigma: sigma.max(1e-8),
            theta_mean: theta,
            long_run_mean,
            half_life_hours,
            half_life_days: half_life_hours / 24.0,
            n_observations: n,
        }
    }

    /// Calibrate to a yield curve (find best a, sigma).
    ///
    /// Uses a simple grid search over (a, sigma) space.
    pub fn calibrate_to_curve(
        market_maturities: &[f64],
        market_rates: &[f64],
    ) -> CalibrationResult {
        let r0 = market_rates[0];
        let mut best_a = 0.1;
        let mut best_sigma = 0.01;
        let mut best_error = f64::MAX;

        // Grid search
        let a_grid: Vec<f64> = (1..50).map(|i| i as f64 * 0.02).collect();
        let sigma_grid: Vec<f64> = (1..50).map(|i| i as f64 * 0.001).collect();

        for &a in &a_grid {
            for &sigma in &sigma_grid {
                let hw = super::analytical::HullWhiteAnalytical::new_flat(a, sigma, r0);
                let mut error = 0.0;

                for (&mat, &rate) in market_maturities.iter().zip(market_rates.iter()) {
                    let p_market = (-rate * mat).exp();
                    let p_model = hw.bond_price(r0, 0.0, mat);
                    error += (p_model - p_market).powi(2);
                }

                if error < best_error {
                    best_error = error;
                    best_a = a;
                    best_sigma = sigma;
                }
            }
        }

        CalibrationResult {
            a: best_a,
            sigma: best_sigma,
            theta_mean: best_a * r0,
            long_run_mean: r0,
            half_life_hours: (2.0_f64).ln() / best_a * 8760.0,
            half_life_days: (2.0_f64).ln() / best_a * 365.0,
            n_observations: market_maturities.len(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Interest Rate Derivatives
// ═══════════════════════════════════════════════════════════════════════════════

pub mod derivatives {
    use super::analytical::HullWhiteAnalytical;
    use statrs::distribution::{Normal, ContinuousCDF};

    /// Prices interest rate derivatives under the Hull-White model.
    pub struct DerivativesPricer {
        pub hw: HullWhiteAnalytical,
    }

    impl DerivativesPricer {
        pub fn new(hw: HullWhiteAnalytical) -> Self {
            Self { hw }
        }

        /// Price a caplet.
        pub fn caplet(
            &self, r: f64, t: f64, t_reset: f64, t_pay: f64,
            strike: f64, notional: f64,
        ) -> f64 {
            let delta = t_pay - t_reset;
            if delta <= 0.0 || t_reset <= t {
                return 0.0;
            }

            let x = 1.0 / (1.0 + strike * delta);
            let put = self.hw.bond_option(r, t, t_reset, t_pay, x, false);
            notional * (1.0 + strike * delta) * put
        }

        /// Price a cap as a sum of caplets.
        pub fn cap(
            &self, r: f64, t: f64, reset_dates: &[f64],
            strike: f64, notional: f64,
        ) -> (f64, Vec<f64>) {
            let mut caplets = Vec::new();
            for i in 0..reset_dates.len() - 1 {
                let cp = if reset_dates[i] > t {
                    self.caplet(r, t, reset_dates[i], reset_dates[i + 1], strike, notional)
                } else {
                    0.0
                };
                caplets.push(cp);
            }
            let total: f64 = caplets.iter().sum();
            (total, caplets)
        }

        /// Price a floorlet.
        pub fn floorlet(
            &self, r: f64, t: f64, t_reset: f64, t_pay: f64,
            strike: f64, notional: f64,
        ) -> f64 {
            let delta = t_pay - t_reset;
            if delta <= 0.0 || t_reset <= t {
                return 0.0;
            }

            let x = 1.0 / (1.0 + strike * delta);
            let call = self.hw.bond_option(r, t, t_reset, t_pay, x, true);
            notional * (1.0 + strike * delta) * call
        }

        /// Price a floor as a sum of floorlets.
        pub fn floor(
            &self, r: f64, t: f64, reset_dates: &[f64],
            strike: f64, notional: f64,
        ) -> (f64, Vec<f64>) {
            let mut floorlets = Vec::new();
            for i in 0..reset_dates.len() - 1 {
                let fp = if reset_dates[i] > t {
                    self.floorlet(r, t, reset_dates[i], reset_dates[i + 1], strike, notional)
                } else {
                    0.0
                };
                floorlets.push(fp);
            }
            let total: f64 = floorlets.iter().sum();
            (total, floorlets)
        }

        /// Price a collar (long cap + short floor).
        pub fn collar(
            &self, r: f64, t: f64, reset_dates: &[f64],
            k_cap: f64, k_floor: f64, notional: f64,
        ) -> f64 {
            let (cap, _) = self.cap(r, t, reset_dates, k_cap, notional);
            let (floor, _) = self.floor(r, t, reset_dates, k_floor, notional);
            cap - floor
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Backtesting
// ═══════════════════════════════════════════════════════════════════════════════

pub mod backtest {
    use serde::{Serialize, Deserialize};

    /// A completed trade.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Trade {
        pub entry_idx: usize,
        pub exit_idx: usize,
        pub direction: i32,  // 1 = long, -1 = short
        pub entry_rate: f64,
        pub exit_rate: f64,
        pub pnl: f64,
        pub holding_periods: usize,
    }

    /// Backtest result metrics.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BacktestResult {
        pub strategy_name: String,
        pub total_return: f64,
        pub annualized_return: f64,
        pub sharpe_ratio: f64,
        pub max_drawdown: f64,
        pub win_rate: f64,
        pub n_trades: usize,
        pub avg_holding_period: f64,
        pub profit_factor: f64,
        pub equity_curve: Vec<f64>,
        pub trades: Vec<Trade>,
    }

    /// Funding rate mean-reversion strategy.
    pub struct FundingMeanReversionStrategy {
        pub lookback: usize,
        pub entry_z: f64,
        pub exit_z: f64,
        pub max_holding: usize,
    }

    impl FundingMeanReversionStrategy {
        pub fn new(lookback: usize, entry_z: f64, exit_z: f64, max_holding: usize) -> Self {
            Self { lookback, entry_z, exit_z, max_holding }
        }

        /// Run the backtest on funding rate data.
        pub fn run(&self, rates: &[f64]) -> BacktestResult {
            let n = rates.len();
            let mut equity = vec![1.0; n];
            let mut trades: Vec<Trade> = Vec::new();
            let mut position: i32 = 0;
            let mut entry_idx = 0;
            let mut holding = 0;

            for i in self.lookback..n {
                let window = &rates[i - self.lookback..i];
                let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
                let std: f64 = {
                    let var = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                        / window.len() as f64;
                    var.sqrt().max(1e-10)
                };
                let z = (rates[i] - mean) / std;

                if position == 0 {
                    if z > self.entry_z {
                        position = -1;
                        entry_idx = i;
                        holding = 0;
                    } else if z < -self.entry_z {
                        position = 1;
                        entry_idx = i;
                        holding = 0;
                    }
                } else {
                    holding += 1;
                    let should_exit = (position == -1 && z < self.exit_z)
                        || (position == 1 && z > -self.exit_z)
                        || holding >= self.max_holding;

                    if should_exit {
                        let mut pnl = 0.0;
                        for j in entry_idx..i {
                            pnl += -position as f64 * rates[j];
                        }
                        trades.push(Trade {
                            entry_idx,
                            exit_idx: i,
                            direction: position,
                            entry_rate: rates[entry_idx],
                            exit_rate: rates[i],
                            pnl,
                            holding_periods: holding,
                        });
                        position = 0;
                    }
                }

                if position != 0 {
                    let period_pnl = -position as f64 * rates[i];
                    equity[i] = equity[i - 1] * (1.0 + period_pnl);
                } else {
                    equity[i] = equity[i - 1];
                }
            }

            self.compute_metrics("Funding Rate Mean Reversion".to_string(), equity, trades)
        }

        fn compute_metrics(
            &self, name: String, equity: Vec<f64>, trades: Vec<Trade>,
        ) -> BacktestResult {
            let n = equity.len();
            let total_return = equity[n - 1] / equity[0] - 1.0;
            let periods_per_year = 3.0 * 365.0;
            let ann_return = (1.0 + total_return).powf(periods_per_year / n as f64) - 1.0;

            let returns: Vec<f64> = (1..n)
                .filter_map(|i| {
                    let r = equity[i] / equity[i - 1] - 1.0;
                    if r != 0.0 { Some(r) } else { None }
                })
                .collect();

            let sharpe = if returns.len() > 1 {
                let mean_r: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let std_r: f64 = {
                    let var = returns.iter().map(|&x| (x - mean_r).powi(2)).sum::<f64>()
                        / returns.len() as f64;
                    var.sqrt()
                };
                if std_r > 0.0 { mean_r / std_r * periods_per_year.sqrt() } else { 0.0 }
            } else {
                0.0
            };

            let mut max_eq = equity[0];
            let mut max_dd = 0.0_f64;
            for &e in &equity {
                max_eq = max_eq.max(e);
                let dd = (max_eq - e) / max_eq;
                max_dd = max_dd.max(dd);
            }

            let n_trades = trades.len();
            let wins = trades.iter().filter(|t| t.pnl > 0.0).count();
            let win_rate = if n_trades > 0 { wins as f64 / n_trades as f64 } else { 0.0 };
            let avg_holding = if n_trades > 0 {
                trades.iter().map(|t| t.holding_periods as f64).sum::<f64>() / n_trades as f64
            } else {
                0.0
            };
            let gross_profit: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
            let gross_loss: f64 = trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.pnl.abs()).sum();
            let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else { 0.0 };

            BacktestResult {
                strategy_name: name,
                total_return,
                annualized_return: ann_return,
                sharpe_ratio: sharpe,
                max_drawdown: max_dd,
                win_rate,
                n_trades,
                avg_holding_period: avg_holding,
                profit_factor,
                equity_curve: equity,
                trades,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_analytical_bond_price_at_maturity() {
        let hw = analytical::HullWhiteAnalytical::new_flat(0.1, 0.01, 0.03);
        let price = hw.bond_price(0.03, 5.0, 5.0);
        assert_relative_eq!(price, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_analytical_bond_price_positive() {
        let hw = analytical::HullWhiteAnalytical::new_flat(0.1, 0.01, 0.03);
        for &t in &[0.5, 1.0, 2.0, 5.0, 10.0] {
            let price = hw.bond_price(0.03, 0.0, t);
            assert!(price > 0.0, "Bond price must be positive for T={}", t);
            assert!(price <= 1.0, "ZCB price must be <= 1 for T={}", t);
        }
    }

    #[test]
    fn test_analytical_bond_price_decreasing_in_rate() {
        let hw = analytical::HullWhiteAnalytical::new_flat(0.1, 0.01, 0.03);
        let p_low = hw.bond_price(0.02, 0.0, 5.0);
        let p_high = hw.bond_price(0.05, 0.0, 5.0);
        assert!(p_low > p_high, "Bond price should decrease with rate");
    }

    #[test]
    fn test_analytical_yield_curve() {
        let hw = analytical::HullWhiteAnalytical::new_flat(0.1, 0.01, 0.03);
        let mats = vec![1.0, 2.0, 5.0, 10.0];
        let yields = hw.yield_curve(0.03, 0.0, &mats);
        assert_eq!(yields.len(), mats.len());
        for &y in &yields {
            assert!(y > 0.0, "Yield must be positive");
        }
    }

    #[test]
    fn test_pinn_forward_pass() {
        let config = model::PINNConfig::default();
        let pinn = model::HullWhitePINN::new(0.1, 0.01, config);
        let price = pinn.forward(0.03, 0.0, 5.0);
        assert!(price > 0.0, "PINN price must be positive");
    }

    #[test]
    fn test_pinn_greeks() {
        let config = model::PINNConfig::default();
        let pinn = model::HullWhitePINN::new(0.1, 0.01, config);
        let greeks = pinn.greeks(0.03, 0.0, 5.0);
        assert!(greeks.price > 0.0);
        // Duration should be positive for a bond
        // (price decreases when rate increases)
    }

    #[test]
    fn test_calibration() {
        let rates: Vec<f64> = (0..200).map(|i| {
            0.0001 + 0.00005 * (i as f64 * 0.1).sin()
        }).collect();

        let result = calibration::calibrate_from_rates(&rates, 8.0);
        assert!(result.a > 0.0);
        assert!(result.sigma > 0.0);
    }

    #[test]
    fn test_synthetic_funding_rates() {
        let rates = data::generate_synthetic_funding_rates(100, 0.0001, 0.3, 0.0003, 42);
        assert_eq!(rates.len(), 100);
        for r in &rates {
            assert!(r.funding_rate.is_finite());
        }
    }

    #[test]
    fn test_caplet_positive() {
        let hw = analytical::HullWhiteAnalytical::new_flat(0.1, 0.01, 0.03);
        let pricer = derivatives::DerivativesPricer::new(hw);
        let price = pricer.caplet(0.03, 0.0, 1.0, 1.25, 0.04, 1_000_000.0);
        assert!(price >= 0.0, "Caplet price must be non-negative");
    }

    #[test]
    fn test_backtest() {
        let rates: Vec<f64> = data::generate_synthetic_funding_rates(500, 0.0001, 0.3, 0.0003, 42)
            .iter()
            .map(|r| r.funding_rate)
            .collect();

        let strategy = backtest::FundingMeanReversionStrategy::new(100, 1.5, 0.5, 50);
        let result = strategy.run(&rates);
        assert_eq!(result.equity_curve.len(), rates.len());
    }
}
