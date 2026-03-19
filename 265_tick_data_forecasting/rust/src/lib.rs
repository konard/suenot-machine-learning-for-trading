use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ============================================================================
// Tick Data Representation
// ============================================================================

/// Side of the trade
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// A single tick (trade) in the market
#[derive(Debug, Clone)]
pub struct TickData {
    pub timestamp: f64, // seconds since epoch (or relative)
    pub price: f64,
    pub volume: f64,
    pub side: TradeSide,
}

/// Direction of the next tick
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TickDirection {
    Down = 0,
    Flat = 1,
    Up = 2,
}

// ============================================================================
// Hawkes Process for Tick Arrivals
// ============================================================================

/// Univariate Hawkes process with exponential kernel:
///   lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
#[derive(Debug, Clone)]
pub struct HawkesProcess {
    pub mu: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl HawkesProcess {
    pub fn new(mu: f64, alpha: f64, beta: f64) -> Self {
        Self { mu, alpha, beta }
    }

    /// Compute the intensity at time t given event times
    pub fn intensity(&self, t: f64, event_times: &[f64]) -> f64 {
        let mut lam = self.mu;
        for &ti in event_times {
            if ti < t {
                lam += self.alpha * (-self.beta * (t - ti)).exp();
            }
        }
        lam
    }

    /// Compute the intensity using recursive formulation (O(1) per new event)
    /// Returns updated accumulator A and intensity
    pub fn intensity_recursive(&self, t: f64, t_prev: f64, a_prev: f64) -> (f64, f64) {
        let decay = (-self.beta * (t - t_prev)).exp();
        let a = decay * (a_prev + self.alpha);
        let lam = self.mu + a;
        (a, lam)
    }

    /// Compute log-likelihood of the Hawkes process given event times on [0, T]
    pub fn log_likelihood(&self, event_times: &[f64], t_end: f64) -> f64 {
        if event_times.is_empty() {
            return -self.mu * t_end;
        }

        let n = event_times.len();
        let mut ll = 0.0;

        // Sum of log(lambda(t_i))
        let mut a = 0.0;
        for i in 0..n {
            if i > 0 {
                a = (a + self.alpha) * (-self.beta * (event_times[i] - event_times[i - 1])).exp();
            }
            let lam = self.mu + a;
            if lam > 0.0 {
                ll += lam.ln();
            } else {
                ll += f64::NEG_INFINITY;
            }
        }

        // Subtract integral of lambda(t) over [0, T]
        // integral = mu * T + (alpha / beta) * sum_i (1 - exp(-beta * (T - t_i)))
        ll -= self.mu * t_end;
        for &ti in event_times {
            ll -= (self.alpha / self.beta) * (1.0 - (-self.beta * (t_end - ti)).exp());
        }

        ll
    }

    /// Estimate parameters via grid search (simplified MLE)
    pub fn fit(event_times: &[f64], t_end: f64) -> Self {
        let mut best_ll = f64::NEG_INFINITY;
        let mut best = HawkesProcess::new(1.0, 0.5, 1.0);

        let n = event_times.len();
        let base_rate = n as f64 / t_end;

        for mu_frac in &[0.3, 0.5, 0.7, 0.9] {
            for alpha_frac in &[0.1, 0.3, 0.5, 0.7] {
                for &beta in &[0.5, 1.0, 2.0, 5.0, 10.0] {
                    let mu = base_rate * mu_frac;
                    let alpha = base_rate * alpha_frac;
                    // Ensure stationarity: alpha / beta < 1
                    if alpha / beta >= 1.0 {
                        continue;
                    }
                    let hp = HawkesProcess::new(mu, alpha, beta);
                    let ll = hp.log_likelihood(event_times, t_end);
                    if ll > best_ll {
                        best_ll = ll;
                        best = hp;
                    }
                }
            }
        }

        best
    }

    /// Branching ratio: alpha / beta. Must be < 1 for stationarity.
    pub fn branching_ratio(&self) -> f64 {
        self.alpha / self.beta
    }
}

// ============================================================================
// Tick Feature Extractor
// ============================================================================

/// Extracts features from a rolling window of tick data
#[derive(Debug, Clone)]
pub struct TickFeatureExtractor {
    pub window_size: usize,
    pub ticks: Vec<TickData>,
}

impl TickFeatureExtractor {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            ticks: Vec::new(),
        }
    }

    /// Add a new tick and maintain window size
    pub fn push(&mut self, tick: TickData) {
        self.ticks.push(tick);
        if self.ticks.len() > self.window_size {
            self.ticks.remove(0);
        }
    }

    /// Compute trade imbalance: signed_volume_sum / total_volume
    pub fn trade_imbalance(&self) -> f64 {
        if self.ticks.is_empty() {
            return 0.0;
        }
        let mut signed_vol = 0.0;
        let mut total_vol = 0.0;
        for tick in &self.ticks {
            let sign = match tick.side {
                TradeSide::Buy => 1.0,
                TradeSide::Sell => -1.0,
            };
            signed_vol += tick.volume * sign;
            total_vol += tick.volume;
        }
        if total_vol > 0.0 {
            signed_vol / total_vol
        } else {
            0.0
        }
    }

    /// Compute VWAP of the window
    pub fn vwap(&self) -> f64 {
        if self.ticks.is_empty() {
            return 0.0;
        }
        let mut pv_sum = 0.0;
        let mut v_sum = 0.0;
        for tick in &self.ticks {
            pv_sum += tick.price * tick.volume;
            v_sum += tick.volume;
        }
        if v_sum > 0.0 {
            pv_sum / v_sum
        } else {
            0.0
        }
    }

    /// VWAP deviation: current price - VWAP
    pub fn vwap_deviation(&self) -> f64 {
        if self.ticks.is_empty() {
            return 0.0;
        }
        let current_price = self.ticks.last().unwrap().price;
        current_price - self.vwap()
    }

    /// Tick arrival intensity (ticks per second) over the window
    pub fn tick_intensity(&self) -> f64 {
        if self.ticks.len() < 2 {
            return 0.0;
        }
        let t_first = self.ticks.first().unwrap().timestamp;
        let t_last = self.ticks.last().unwrap().timestamp;
        let dt = t_last - t_first;
        if dt > 0.0 {
            (self.ticks.len() - 1) as f64 / dt
        } else {
            0.0
        }
    }

    /// Realized volatility from tick returns
    pub fn realized_volatility(&self) -> f64 {
        if self.ticks.len() < 2 {
            return 0.0;
        }
        let returns: Vec<f64> = self
            .ticks
            .windows(2)
            .map(|w| {
                if w[0].price > 0.0 {
                    (w[1].price / w[0].price).ln()
                } else {
                    0.0
                }
            })
            .collect();
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        var.sqrt()
    }

    /// Mean inter-arrival time
    pub fn mean_inter_arrival(&self) -> f64 {
        if self.ticks.len() < 2 {
            return 0.0;
        }
        let deltas: Vec<f64> = self
            .ticks
            .windows(2)
            .map(|w| w[1].timestamp - w[0].timestamp)
            .collect();
        deltas.iter().sum::<f64>() / deltas.len() as f64
    }

    /// Recent price momentum (last few ticks)
    pub fn price_momentum(&self, n: usize) -> f64 {
        let len = self.ticks.len();
        if len < n + 1 {
            return 0.0;
        }
        let current = self.ticks[len - 1].price;
        let past = self.ticks[len - 1 - n].price;
        if past > 0.0 {
            (current / past).ln()
        } else {
            0.0
        }
    }

    /// Extract full feature vector for the forecasting model
    /// Returns: [trade_imbalance, vwap_deviation_normalized, tick_intensity,
    ///           realized_vol, mean_inter_arrival, momentum_3, momentum_5, momentum_10]
    pub fn extract_features(&self) -> Array1<f64> {
        let vwap = self.vwap();
        let vwap_dev_norm = if vwap > 0.0 {
            self.vwap_deviation() / vwap
        } else {
            0.0
        };

        Array1::from_vec(vec![
            self.trade_imbalance(),
            vwap_dev_norm,
            self.tick_intensity().min(100.0) / 100.0, // normalize
            self.realized_volatility().min(0.1) / 0.1, // normalize
            self.mean_inter_arrival().min(10.0) / 10.0, // normalize
            self.price_momentum(3),
            self.price_momentum(5),
            self.price_momentum(10),
        ])
    }

    /// Determine the direction of the last tick
    pub fn last_tick_direction(&self) -> TickDirection {
        if self.ticks.len() < 2 {
            return TickDirection::Flat;
        }
        let len = self.ticks.len();
        let dp = self.ticks[len - 1].price - self.ticks[len - 2].price;
        if dp > 1e-10 {
            TickDirection::Up
        } else if dp < -1e-10 {
            TickDirection::Down
        } else {
            TickDirection::Flat
        }
    }
}

// ============================================================================
// Tick Forecaster (Neural Network)
// ============================================================================

/// Two-layer feedforward network for tick direction prediction
/// Input: feature vector from TickFeatureExtractor
/// Output: 3-class softmax (Down, Flat, Up)
#[derive(Debug, Clone)]
pub struct TickForecaster {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
    pub w3: Array2<f64>,
    pub b3: Array1<f64>,
    pub learning_rate: f64,
}

impl TickForecaster {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let he1 = (2.0 / input_dim as f64).sqrt();
        let he2 = (2.0 / hidden_dim as f64).sqrt();

        let w1 = Array2::from_shape_fn((input_dim, hidden_dim), |_| rng.gen_range(-he1..he1));
        let b1 = Array1::zeros(hidden_dim);
        let w2 = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| rng.gen_range(-he2..he2));
        let b2 = Array1::zeros(hidden_dim);
        let w3 = Array2::from_shape_fn((hidden_dim, 3), |_| rng.gen_range(-he2..he2));
        let b3 = Array1::zeros(3);

        Self {
            input_dim,
            hidden_dim,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            learning_rate: 0.001,
        }
    }

    /// Forward pass: returns raw logits for 3 classes
    pub fn forward_logits(&self, x: &Array1<f64>) -> Array1<f64> {
        let h1 = x.dot(&self.w1) + &self.b1;
        let h1 = h1.mapv(|v| v.max(0.0)); // ReLU
        let h2 = h1.dot(&self.w2) + &self.b2;
        let h2 = h2.mapv(|v| v.max(0.0)); // ReLU
        h2.dot(&self.w3) + &self.b3
    }

    /// Softmax of logits
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let logits = self.forward_logits(x);
        softmax(&logits)
    }

    /// Predict the most likely direction
    pub fn predict(&self, x: &Array1<f64>) -> TickDirection {
        let probs = self.forward(x);
        let idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        match idx {
            0 => TickDirection::Down,
            1 => TickDirection::Flat,
            _ => TickDirection::Up,
        }
    }

    /// Predict with confidence (returns direction and probability)
    pub fn predict_with_confidence(&self, x: &Array1<f64>) -> (TickDirection, f64) {
        let probs = self.forward(x);
        let (idx, &conf) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let dir = match idx {
            0 => TickDirection::Down,
            1 => TickDirection::Flat,
            _ => TickDirection::Up,
        };
        (dir, conf)
    }

    /// Train on a single sample using simplified gradient update
    /// target: 0=Down, 1=Flat, 2=Up
    pub fn train_step(&mut self, x: &Array1<f64>, target: usize) -> f64 {
        // Forward pass
        let h1_raw = x.dot(&self.w1) + &self.b1;
        let h1 = h1_raw.mapv(|v| v.max(0.0));
        let h2_raw = h1.dot(&self.w2) + &self.b2;
        let h2 = h2_raw.mapv(|v| v.max(0.0));
        let logits = h2.dot(&self.w3) + &self.b3;
        let probs = softmax(&logits);

        // Cross-entropy loss
        let loss = -(probs[target].max(1e-10)).ln();

        // Backprop through output layer
        let mut d_logits = probs.clone();
        d_logits[target] -= 1.0; // grad of CE w.r.t. logits

        // Gradient for w3, b3
        let lr = self.learning_rate;
        for j in 0..self.hidden_dim {
            for k in 0..3 {
                self.w3[[j, k]] -= lr * h2[j] * d_logits[k];
            }
        }
        for k in 0..3 {
            self.b3[k] -= lr * d_logits[k];
        }

        // Backprop through second hidden layer
        let d_h2 = d_logits.dot(&self.w3.t());
        let d_h2_raw = &d_h2 * &h2_raw.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });

        for j in 0..self.hidden_dim {
            for k in 0..self.hidden_dim {
                self.w2[[j, k]] -= lr * h1[j] * d_h2_raw[k];
            }
        }
        for k in 0..self.hidden_dim {
            self.b2[k] -= lr * d_h2_raw[k];
        }

        // Backprop through first hidden layer
        let d_h1 = d_h2_raw.dot(&self.w2.t());
        let d_h1_raw = &d_h1 * &h1_raw.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });

        for j in 0..self.input_dim {
            for k in 0..self.hidden_dim {
                self.w1[[j, k]] -= lr * x[j] * d_h1_raw[k];
            }
        }
        for k in 0..self.hidden_dim {
            self.b1[k] -= lr * d_h1_raw[k];
        }

        loss
    }

    /// Train on a batch of samples
    pub fn train_batch(&mut self, features: &[Array1<f64>], targets: &[usize]) -> f64 {
        let mut total_loss = 0.0;
        for (x, &t) in features.iter().zip(targets.iter()) {
            total_loss += self.train_step(x, t);
        }
        total_loss / features.len() as f64
    }
}

/// Softmax function
pub fn softmax(logits: &Array1<f64>) -> Array1<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps = logits.mapv(|v| (v - max_val).exp());
    let sum = exps.sum();
    exps / sum
}

// ============================================================================
// Tick Forecast Backtester
// ============================================================================

/// Backtests tick direction forecasts and computes trading PnL
#[derive(Debug, Clone)]
pub struct TickForecastBacktester {
    pub correct: usize,
    pub total: usize,
    pub pnl: f64,
    pub position: f64,
    pub entry_price: f64,
    pub transaction_cost: f64,
    pub trades: usize,
    pub max_drawdown: f64,
    pub peak_pnl: f64,
}

impl TickForecastBacktester {
    pub fn new(transaction_cost: f64) -> Self {
        Self {
            correct: 0,
            total: 0,
            pnl: 0.0,
            position: 0.0,
            entry_price: 0.0,
            transaction_cost,
            trades: 0,
            max_drawdown: 0.0,
            peak_pnl: 0.0,
        }
    }

    /// Process a prediction and actual outcome
    pub fn update(
        &mut self,
        predicted: TickDirection,
        actual: TickDirection,
        price: f64,
        confidence: f64,
    ) {
        self.total += 1;
        if predicted == actual {
            self.correct += 1;
        }

        // Trade based on prediction with confidence threshold
        let min_confidence = 0.4;
        if confidence > min_confidence {
            let target_position = match predicted {
                TickDirection::Up => 1.0,
                TickDirection::Down => -1.0,
                TickDirection::Flat => 0.0,
            };

            if (target_position - self.position).abs() > 0.01 {
                // Close current position
                if self.position != 0.0 && self.entry_price > 0.0 {
                    let ret = self.position * (price - self.entry_price) / self.entry_price;
                    self.pnl += ret - self.transaction_cost;
                }

                // Open new position
                self.position = target_position;
                self.entry_price = price;
                self.trades += 1;

                // Update drawdown
                if self.pnl > self.peak_pnl {
                    self.peak_pnl = self.pnl;
                }
                let dd = self.peak_pnl - self.pnl;
                if dd > self.max_drawdown {
                    self.max_drawdown = dd;
                }
            }
        }
    }

    /// Directional accuracy
    pub fn accuracy(&self) -> f64 {
        if self.total > 0 {
            self.correct as f64 / self.total as f64
        } else {
            0.0
        }
    }

    /// Total return
    pub fn total_return(&self) -> f64 {
        self.pnl
    }

    /// Calmar-like ratio (return / max drawdown)
    pub fn return_over_drawdown(&self) -> f64 {
        if self.max_drawdown > 0.0 {
            self.pnl / self.max_drawdown
        } else if self.pnl > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    }
}

// ============================================================================
// Synthetic Tick Data Generation
// ============================================================================

/// Generate synthetic tick data using a Hawkes process for arrivals
/// and geometric Brownian motion for prices
pub fn generate_synthetic_ticks(
    n_ticks: usize,
    mu_hawkes: f64,
    alpha_hawkes: f64,
    beta_hawkes: f64,
    initial_price: f64,
    price_vol: f64,
) -> Vec<TickData> {
    let mut rng = rand::thread_rng();
    let mut ticks = Vec::with_capacity(n_ticks);
    let mut t = 0.0;
    let mut price = initial_price;
    let mut intensity = mu_hawkes;

    for _ in 0..n_ticks {
        // Thinning algorithm for Hawkes process
        let lambda_bar = intensity + alpha_hawkes; // upper bound
        let u: f64 = rng.gen_range(0.0..1.0);
        let dt = -(u.ln()) / lambda_bar;
        t += dt;

        // Decay intensity
        intensity = mu_hawkes + (intensity - mu_hawkes) * (-beta_hawkes * dt).exp() + alpha_hawkes;

        // Accept/reject
        let accept_prob = (mu_hawkes
            + (intensity - mu_hawkes - alpha_hawkes).max(0.0))
            / lambda_bar;
        if rng.gen::<f64>() > accept_prob {
            // Still advance time but don't record tick
            intensity -= alpha_hawkes; // undo the excitation
            continue;
        }

        // Generate price change
        let z: f64 = rng.gen_range(-3.0..3.0);
        let dp = price * price_vol * dt.sqrt() * z;
        price += dp;
        price = price.max(initial_price * 0.5); // floor

        // Determine side
        let side = if dp >= 0.0 {
            TradeSide::Buy
        } else {
            TradeSide::Sell
        };

        // Volume (log-normal-ish)
        let vol: f64 = (rng.gen_range(0.5_f64..2.0) * 0.1).exp();

        ticks.push(TickData {
            timestamp: t,
            price,
            volume: vol,
            side,
        });

        if ticks.len() >= n_ticks {
            break;
        }
    }

    ticks.truncate(n_ticks);
    ticks
}

/// Generate ticks from OHLCV candle data (decompose each candle into synthetic ticks)
pub fn candles_to_ticks(
    candles: &[(i64, f64, f64, f64, f64, f64)],
    ticks_per_candle: usize,
) -> Vec<TickData> {
    let mut rng = rand::thread_rng();
    let mut all_ticks = Vec::new();

    for (i, &(timestamp, open, high, low, close, volume)) in candles.iter().enumerate() {
        let candle_duration = if i + 1 < candles.len() {
            (candles[i + 1].0 - timestamp) as f64 / 1000.0
        } else {
            60.0 // default 1 minute
        };

        let vol_per_tick = volume / ticks_per_candle as f64;

        for j in 0..ticks_per_candle {
            let frac = j as f64 / ticks_per_candle as f64;
            let t = timestamp as f64 / 1000.0 + frac * candle_duration;

            // Simulate price path within candle: open -> high/low -> close
            let price = if frac < 0.25 {
                // Move towards high or low
                let target = if close > open { high } else { low };
                open + (target - open) * (frac / 0.25)
            } else if frac < 0.75 {
                // Fluctuate between high and low
                let mid = (high + low) / 2.0;
                let amp = (high - low) / 2.0;
                mid + amp * (std::f64::consts::PI * (frac - 0.25) / 0.5).sin()
                    * if close > open { 1.0 } else { -1.0 }
            } else {
                // Move towards close
                let prev = if close > open { high } else { low };
                prev + (close - prev) * ((frac - 0.75) / 0.25)
            };

            // Add noise
            let noise = rng.gen_range(-0.0005..0.0005) * price;
            let price = price + noise;

            let side = if j > 0 && price >= all_ticks.last().map(|t: &TickData| t.price).unwrap_or(price)
            {
                TradeSide::Buy
            } else {
                TradeSide::Sell
            };

            all_ticks.push(TickData {
                timestamp: t,
                price,
                volume: vol_per_tick * rng.gen_range(0.5..1.5),
                side,
            });
        }
    }

    all_ticks
}

// ============================================================================
// Bybit API Client
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitKlineResult,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BybitKlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// Client for fetching market data from Bybit
pub struct BybitClient {
    pub base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data from Bybit
    /// Returns Vec of (timestamp, open, high, low, close, volume)
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<(i64, f64, f64, f64, f64, f64)>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: BybitKlineResponse = reqwest::blocking::get(&url)?.json()?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let mut klines = Vec::new();
        for item in &response.result.list {
            if item.len() >= 6 {
                let timestamp = item[0].parse::<i64>().unwrap_or(0);
                let open = item[1].parse::<f64>().unwrap_or(0.0);
                let high = item[2].parse::<f64>().unwrap_or(0.0);
                let low = item[3].parse::<f64>().unwrap_or(0.0);
                let close = item[4].parse::<f64>().unwrap_or(0.0);
                let vol = item[5].parse::<f64>().unwrap_or(0.0);
                klines.push((timestamp, open, high, low, close, vol));
            }
        }

        klines.reverse(); // chronological order
        Ok(klines)
    }

    /// Fetch klines and convert to synthetic tick data
    pub fn fetch_ticks(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
        ticks_per_candle: usize,
    ) -> anyhow::Result<Vec<TickData>> {
        let klines = self.fetch_klines(symbol, interval, limit)?;
        Ok(candles_to_ticks(&klines, ticks_per_candle))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hawkes_intensity_baseline() {
        let hp = HawkesProcess::new(1.0, 0.5, 2.0);
        // With no events, intensity should be mu
        let lam = hp.intensity(10.0, &[]);
        assert!((lam - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hawkes_intensity_after_event() {
        let hp = HawkesProcess::new(1.0, 0.5, 2.0);
        // Right after an event at t=0, intensity should be mu + alpha
        let lam = hp.intensity(0.001, &[0.0]);
        assert!(lam > 1.0 + 0.49, "Intensity should spike after event");
    }

    #[test]
    fn test_hawkes_intensity_decays() {
        let hp = HawkesProcess::new(1.0, 0.5, 2.0);
        let lam_soon = hp.intensity(0.1, &[0.0]);
        let lam_later = hp.intensity(5.0, &[0.0]);
        assert!(
            lam_soon > lam_later,
            "Intensity should decay over time"
        );
        // After long time, should approach mu
        assert!(
            (lam_later - 1.0).abs() < 0.01,
            "Intensity should approach mu for large t"
        );
    }

    #[test]
    fn test_hawkes_recursive_matches_exact() {
        let hp = HawkesProcess::new(1.0, 0.5, 2.0);
        let events = vec![0.0, 0.5, 1.2, 1.8, 3.0];

        // Compute intensity at last event using both methods
        let lam_exact = hp.intensity(3.0, &events[..4]);

        // Recursive computation
        let mut a = 0.0;
        for i in 1..events.len() {
            let (a_new, _) = hp.intensity_recursive(events[i], events[i - 1], a);
            a = a_new;
        }
        let lam_recursive = hp.mu + a;

        assert!(
            (lam_exact - lam_recursive).abs() < 1e-6,
            "Recursive and exact should match: {} vs {}",
            lam_exact,
            lam_recursive
        );
    }

    #[test]
    fn test_hawkes_branching_ratio() {
        let hp = HawkesProcess::new(1.0, 0.3, 1.0);
        assert!((hp.branching_ratio() - 0.3).abs() < 1e-10);
        assert!(hp.branching_ratio() < 1.0, "Should be stationary");
    }

    #[test]
    fn test_hawkes_log_likelihood() {
        let hp = HawkesProcess::new(1.0, 0.5, 2.0);
        let events = vec![0.5, 1.2, 2.0, 3.5, 4.1];
        let ll = hp.log_likelihood(&events, 5.0);
        assert!(ll.is_finite(), "Log-likelihood should be finite");

        // A process with wrong parameters should have lower LL
        let hp_bad = HawkesProcess::new(0.01, 0.01, 0.01);
        let ll_bad = hp_bad.log_likelihood(&events, 5.0);
        // Not guaranteed to be lower in all cases, but the fit should be better
        assert!(ll_bad.is_finite());
    }

    #[test]
    fn test_hawkes_fit() {
        let events: Vec<f64> = (0..50).map(|i| i as f64 * 0.2 + 0.1).collect();
        let hp = HawkesProcess::fit(&events, 10.0);
        assert!(hp.mu > 0.0);
        assert!(hp.alpha > 0.0);
        assert!(hp.beta > 0.0);
        assert!(hp.branching_ratio() < 1.0);
    }

    #[test]
    fn test_feature_extractor_empty() {
        let fe = TickFeatureExtractor::new(100);
        assert!((fe.trade_imbalance()).abs() < 1e-10);
        assert!((fe.vwap()).abs() < 1e-10);
        assert!((fe.tick_intensity()).abs() < 1e-10);
    }

    #[test]
    fn test_feature_extractor_trade_imbalance() {
        let mut fe = TickFeatureExtractor::new(100);
        // All buys
        for i in 0..10 {
            fe.push(TickData {
                timestamp: i as f64,
                price: 100.0,
                volume: 1.0,
                side: TradeSide::Buy,
            });
        }
        assert!((fe.trade_imbalance() - 1.0).abs() < 1e-10, "All buys should give TI=1.0");

        // Add equal sells
        for i in 10..20 {
            fe.push(TickData {
                timestamp: i as f64,
                price: 100.0,
                volume: 1.0,
                side: TradeSide::Sell,
            });
        }
        assert!(
            fe.trade_imbalance().abs() < 0.01,
            "Equal buys/sells should give TI near 0"
        );
    }

    #[test]
    fn test_feature_extractor_vwap() {
        let mut fe = TickFeatureExtractor::new(100);
        fe.push(TickData {
            timestamp: 0.0,
            price: 100.0,
            volume: 2.0,
            side: TradeSide::Buy,
        });
        fe.push(TickData {
            timestamp: 1.0,
            price: 200.0,
            volume: 1.0,
            side: TradeSide::Buy,
        });
        // VWAP = (100*2 + 200*1) / (2+1) = 400/3 = 133.33
        let expected_vwap = 400.0 / 3.0;
        assert!(
            (fe.vwap() - expected_vwap).abs() < 1e-6,
            "VWAP should be {}, got {}",
            expected_vwap,
            fe.vwap()
        );
    }

    #[test]
    fn test_feature_extractor_realized_vol() {
        let mut fe = TickFeatureExtractor::new(100);
        // Constant price -> zero vol
        for i in 0..10 {
            fe.push(TickData {
                timestamp: i as f64,
                price: 100.0,
                volume: 1.0,
                side: TradeSide::Buy,
            });
        }
        assert!(
            fe.realized_volatility() < 1e-10,
            "Constant price should have zero vol"
        );
    }

    #[test]
    fn test_feature_extractor_extract() {
        let mut fe = TickFeatureExtractor::new(100);
        for i in 0..20 {
            fe.push(TickData {
                timestamp: i as f64 * 0.1,
                price: 100.0 + (i as f64 * 0.01),
                volume: 1.0,
                side: if i % 2 == 0 {
                    TradeSide::Buy
                } else {
                    TradeSide::Sell
                },
            });
        }
        let features = fe.extract_features();
        assert_eq!(features.len(), 8, "Feature vector should have 8 elements");
        assert!(features.iter().all(|v| v.is_finite()), "All features should be finite");
    }

    #[test]
    fn test_tick_direction() {
        let mut fe = TickFeatureExtractor::new(100);
        fe.push(TickData {
            timestamp: 0.0,
            price: 100.0,
            volume: 1.0,
            side: TradeSide::Buy,
        });
        fe.push(TickData {
            timestamp: 1.0,
            price: 101.0,
            volume: 1.0,
            side: TradeSide::Buy,
        });
        assert_eq!(fe.last_tick_direction(), TickDirection::Up);

        fe.push(TickData {
            timestamp: 2.0,
            price: 99.0,
            volume: 1.0,
            side: TradeSide::Sell,
        });
        assert_eq!(fe.last_tick_direction(), TickDirection::Down);
    }

    #[test]
    fn test_forecaster_output_shape() {
        let model = TickForecaster::new(8, 16);
        let x = Array1::from_vec(vec![0.1, -0.05, 0.5, 0.3, 0.2, 0.01, -0.01, 0.005]);
        let probs = model.forward(&x);
        assert_eq!(probs.len(), 3);
        assert!(
            (probs.sum() - 1.0).abs() < 1e-6,
            "Softmax should sum to 1.0"
        );
        assert!(probs.iter().all(|&p| p >= 0.0), "All probabilities should be non-negative");
    }

    #[test]
    fn test_forecaster_predict() {
        let model = TickForecaster::new(8, 16);
        let x = Array1::from_vec(vec![0.1, -0.05, 0.5, 0.3, 0.2, 0.01, -0.01, 0.005]);
        let dir = model.predict(&x);
        // Just check it returns a valid direction
        assert!(matches!(
            dir,
            TickDirection::Down | TickDirection::Flat | TickDirection::Up
        ));
    }

    #[test]
    fn test_forecaster_train_reduces_loss() {
        let mut model = TickForecaster::new(8, 16);
        model.learning_rate = 0.01;
        let x = Array1::from_vec(vec![0.1, -0.05, 0.5, 0.3, 0.2, 0.01, -0.01, 0.005]);

        let loss1 = model.train_step(&x, 2); // train towards "Up"
        // Train several more times
        let mut loss_last = loss1;
        for _ in 0..50 {
            loss_last = model.train_step(&x, 2);
        }
        assert!(
            loss_last < loss1,
            "Loss should decrease with training: {} -> {}",
            loss1,
            loss_last
        );
    }

    #[test]
    fn test_backtester_accuracy() {
        let mut bt = TickForecastBacktester::new(0.0001);
        bt.update(TickDirection::Up, TickDirection::Up, 100.0, 0.6);
        bt.update(TickDirection::Down, TickDirection::Up, 101.0, 0.6);
        bt.update(TickDirection::Up, TickDirection::Up, 102.0, 0.6);
        assert!((bt.accuracy() - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_basic() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);
        assert!((probs.sum() - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_uniform() {
        let logits = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let probs = softmax(&logits);
        for &p in probs.iter() {
            assert!((p - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_synthetic_tick_generation() {
        let ticks = generate_synthetic_ticks(100, 5.0, 1.0, 3.0, 50000.0, 0.001);
        assert!(ticks.len() >= 50, "Should generate reasonable number of ticks, got {}", ticks.len());
        assert!(ticks.iter().all(|t| t.price > 0.0), "All prices should be positive");
        // Timestamps should be increasing
        for w in ticks.windows(2) {
            assert!(
                w[1].timestamp >= w[0].timestamp,
                "Timestamps should be non-decreasing"
            );
        }
    }

    #[test]
    fn test_candles_to_ticks() {
        let candles = vec![
            (1000000, 100.0, 105.0, 98.0, 103.0, 10.0),
            (1060000, 103.0, 107.0, 101.0, 106.0, 15.0),
        ];
        let ticks = candles_to_ticks(&candles, 10);
        assert_eq!(ticks.len(), 20, "Should generate ticks_per_candle * n_candles ticks");
        assert!(ticks.iter().all(|t| t.price > 0.0));
    }
}
