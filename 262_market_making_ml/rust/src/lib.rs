use anyhow::Result;
use ndarray::Array1;
use rand::Rng;
use serde::Deserialize;

// ─── Bybit API types ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub list: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct BybitOrderBookResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    pub result: OrderBookResult,
}

#[derive(Debug, Deserialize)]
pub struct OrderBookResult {
    #[serde(rename = "a")]
    pub asks: Vec<Vec<String>>,
    #[serde(rename = "b")]
    pub bids: Vec<Vec<String>>,
}

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

/// A single level in the order book.
#[derive(Debug, Clone)]
pub struct OrderBookLevel {
    pub price: f64,
    pub size: f64,
}

/// Order book snapshot with bids and asks.
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
}

impl OrderBook {
    /// Best bid price, or None if empty.
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Best ask price, or None if empty.
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Mid-price: average of best bid and ask.
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some((b + a) / 2.0),
            _ => None,
        }
    }

    /// Bid-ask spread.
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some(a - b),
            _ => None,
        }
    }

    /// Volume-weighted bid-ask imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol).
    /// Uses the top `depth` levels. Returns value in [-1, 1].
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_vol: f64 = self.bids.iter().take(depth).map(|l| l.size).sum();
        let ask_vol: f64 = self.asks.iter().take(depth).map(|l| l.size).sum();
        let total = bid_vol + ask_vol;
        if total == 0.0 {
            0.0
        } else {
            (bid_vol - ask_vol) / total
        }
    }

    /// Weighted mid-price using best level volumes.
    pub fn weighted_mid(&self) -> Option<f64> {
        match (self.bids.first(), self.asks.first()) {
            (Some(bid), Some(ask)) => {
                let total = bid.size + ask.size;
                if total == 0.0 {
                    return self.mid_price();
                }
                Some((ask.size * bid.price + bid.size * ask.price) / total)
            }
            _ => None,
        }
    }
}

// ─── Bybit Client ──────────────────────────────────────────────────────────

/// HTTP client for Bybit V5 API (klines and order book).
pub struct BybitClient {
    base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data for a symbol.
    /// `symbol`: e.g. "BTCUSDT"
    /// `interval`: e.g. "1" for 1-minute, "60" for 1-hour candles
    /// `limit`: number of candles (max 1000)
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

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

        // Bybit returns newest first; reverse to chronological order
        candles.reverse();
        Ok(candles)
    }

    /// Fetch order book snapshot for a symbol.
    /// `symbol`: e.g. "BTCUSDT"
    /// `depth`: number of levels (max 200)
    pub fn fetch_orderbook(&self, symbol: &str, depth: usize) -> Result<OrderBook> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, depth
        );

        let resp: BybitOrderBookResponse = reqwest::blocking::get(&url)?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: retCode={}", resp.ret_code);
        }

        let bids: Vec<OrderBookLevel> = resp
            .result
            .bids
            .iter()
            .filter_map(|row| {
                if row.len() < 2 {
                    return None;
                }
                Some(OrderBookLevel {
                    price: row[0].parse().ok()?,
                    size: row[1].parse().ok()?,
                })
            })
            .collect();

        let asks: Vec<OrderBookLevel> = resp
            .result
            .asks
            .iter()
            .filter_map(|row| {
                if row.len() < 2 {
                    return None;
                }
                Some(OrderBookLevel {
                    price: row[0].parse().ok()?,
                    size: row[1].parse().ok()?,
                })
            })
            .collect();

        Ok(OrderBook { bids, asks })
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Volatility Estimation ─────────────────────────────────────────────────

/// Computes realized volatility from a series of prices (standard deviation of returns).
pub fn realized_volatility(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }
    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

/// Garman-Klass volatility estimator from OHLC candles.
/// More efficient than close-to-close volatility.
pub fn garman_klass_volatility(candles: &[Candle]) -> f64 {
    if candles.is_empty() {
        return 0.0;
    }
    let sum: f64 = candles
        .iter()
        .map(|c| {
            let hl = (c.high / c.low).ln();
            let co = (c.close / c.open).ln();
            0.5 * hl * hl - (2.0 * 2.0_f64.ln() - 1.0) * co * co
        })
        .sum();
    (sum / candles.len() as f64).abs().sqrt()
}

// ─── Feature Engineering ────────────────────────────────────────────────────

/// Market making features extracted from order book and trade data.
#[derive(Debug, Clone)]
pub struct MarketFeatures {
    /// Current inventory level
    pub inventory: f64,
    /// Estimated volatility
    pub volatility: f64,
    /// Order flow imbalance
    pub ofi: f64,
    /// Volume-weighted price impact (VPIN proxy)
    pub vpin: f64,
    /// Current bid-ask spread (relative to mid)
    pub spread: f64,
    /// Order book imbalance at top levels
    pub imbalance: f64,
    /// Remaining time fraction (0 to 1)
    pub time_remaining: f64,
}

impl MarketFeatures {
    /// Convert features to a fixed-size array for ML models.
    pub fn to_array(&self) -> Array1<f64> {
        Array1::from(vec![
            self.inventory,
            self.volatility,
            self.ofi,
            self.vpin,
            self.spread,
            self.imbalance,
            self.time_remaining,
        ])
    }
}

/// Compute VPIN (Volume-Synchronized Probability of Informed Trading) proxy
/// from a series of candles. Uses volume-bucketed absolute price changes.
pub fn compute_vpin(candles: &[Candle], bucket_size: f64) -> f64 {
    if candles.is_empty() || bucket_size <= 0.0 {
        return 0.0;
    }

    let mut buy_volume = 0.0;
    let mut sell_volume = 0.0;
    let mut total_volume = 0.0;

    for c in candles {
        let signed = if c.close >= c.open {
            c.volume
        } else {
            -c.volume
        };
        if signed > 0.0 {
            buy_volume += signed;
        } else {
            sell_volume += signed.abs();
        }
        total_volume += c.volume;
    }

    if total_volume == 0.0 {
        return 0.0;
    }

    // VPIN ≈ |buy_volume - sell_volume| / total_volume
    (buy_volume - sell_volume).abs() / total_volume
}

/// Compute order flow imbalance (OFI) from consecutive order book snapshots.
/// Positive OFI means more buying pressure.
pub fn compute_ofi(prev: &OrderBook, curr: &OrderBook) -> f64 {
    let prev_bid_vol: f64 = prev.bids.iter().take(5).map(|l| l.size).sum();
    let curr_bid_vol: f64 = curr.bids.iter().take(5).map(|l| l.size).sum();
    let prev_ask_vol: f64 = prev.asks.iter().take(5).map(|l| l.size).sum();
    let curr_ask_vol: f64 = curr.asks.iter().take(5).map(|l| l.size).sum();

    let delta_bid = curr_bid_vol - prev_bid_vol;
    let delta_ask = curr_ask_vol - prev_ask_vol;

    // OFI = increase in bid volume - increase in ask volume
    delta_bid - delta_ask
}

// ─── Avellaneda-Stoikov Model ──────────────────────────────────────────────

/// Configuration for the Avellaneda-Stoikov optimal quoting model.
#[derive(Debug, Clone)]
pub struct AvellanedaStoikovConfig {
    /// Risk aversion parameter (gamma). Higher = more conservative.
    pub gamma: f64,
    /// Fill rate sensitivity (kappa). Controls how fill prob decays with distance.
    pub kappa: f64,
    /// Total time horizon in seconds.
    pub time_horizon: f64,
}

impl Default for AvellanedaStoikovConfig {
    fn default() -> Self {
        Self {
            gamma: 0.1,
            kappa: 1.5,
            time_horizon: 3600.0,
        }
    }
}

/// The Avellaneda-Stoikov optimal market making model.
/// Computes bid and ask offsets from mid-price based on inventory, volatility,
/// risk aversion, and time remaining.
pub struct AvellanedaStoikov {
    pub config: AvellanedaStoikovConfig,
}

impl AvellanedaStoikov {
    pub fn new(config: AvellanedaStoikovConfig) -> Self {
        Self { config }
    }

    /// Compute the reservation price (indifference price) given current state.
    /// r = S - q * gamma * sigma^2 * (T - t)
    pub fn reservation_price(
        &self,
        mid_price: f64,
        inventory: f64,
        volatility: f64,
        time_remaining: f64,
    ) -> f64 {
        mid_price
            - inventory
                * self.config.gamma
                * volatility.powi(2)
                * time_remaining
    }

    /// Compute the optimal spread around the reservation price.
    /// delta = (2/gamma) * ln(1 + gamma/kappa) + gamma * sigma^2 * (T-t)
    pub fn optimal_spread(&self, volatility: f64, time_remaining: f64) -> f64 {
        let g = self.config.gamma;
        let k = self.config.kappa;
        (2.0 / g) * (1.0 + g / k).ln() + g * volatility.powi(2) * time_remaining
    }

    /// Compute optimal bid and ask quotes.
    /// Returns (bid_price, ask_price).
    pub fn compute_quotes(
        &self,
        mid_price: f64,
        inventory: f64,
        volatility: f64,
        time_remaining: f64,
    ) -> (f64, f64) {
        let reservation = self.reservation_price(mid_price, inventory, volatility, time_remaining);
        let spread = self.optimal_spread(volatility, time_remaining);
        let half_spread = spread / 2.0;

        let bid = reservation - half_spread;
        let ask = reservation + half_spread;

        (bid, ask)
    }

    /// Fill probability for a limit order at distance delta from mid.
    /// P(fill) = A * exp(-kappa * delta)
    pub fn fill_probability(&self, delta: f64) -> f64 {
        (-self.config.kappa * delta).exp()
    }
}

// ─── Inventory Manager ─────────────────────────────────────────────────────

/// Tracks market maker inventory and computes risk penalties.
#[derive(Debug, Clone)]
pub struct InventoryManager {
    /// Current inventory (positive = long, negative = short)
    pub inventory: f64,
    /// Maximum allowed absolute inventory
    pub max_inventory: f64,
    /// Risk aversion for quadratic penalty
    pub gamma: f64,
    /// Cumulative realized PnL
    pub realized_pnl: f64,
    /// Trade history: (price, quantity) where positive = buy
    trades: Vec<(f64, f64)>,
}

impl InventoryManager {
    pub fn new(max_inventory: f64, gamma: f64) -> Self {
        Self {
            inventory: 0.0,
            max_inventory,
            gamma,
            realized_pnl: 0.0,
            trades: Vec::new(),
        }
    }

    /// Record a fill. Returns true if the trade is within limits.
    pub fn record_fill(&mut self, price: f64, quantity: f64) -> bool {
        let new_inventory = self.inventory + quantity;
        if new_inventory.abs() > self.max_inventory {
            return false;
        }

        // Track realized PnL when reducing position
        if (self.inventory > 0.0 && quantity < 0.0) || (self.inventory < 0.0 && quantity > 0.0) {
            // Closing (or partially closing) a position
            let closing_qty = quantity.abs().min(self.inventory.abs());
            if !self.trades.is_empty() {
                let avg_entry = self.average_entry_price();
                let pnl = if self.inventory > 0.0 {
                    (price - avg_entry) * closing_qty
                } else {
                    (avg_entry - price) * closing_qty
                };
                self.realized_pnl += pnl;
            }
        }

        self.inventory = new_inventory;
        self.trades.push((price, quantity));
        true
    }

    /// Average entry price of current position.
    pub fn average_entry_price(&self) -> f64 {
        let mut total_cost = 0.0;
        let mut total_qty = 0.0;
        for &(price, qty) in &self.trades {
            total_cost += price * qty.abs();
            total_qty += qty.abs();
        }
        if total_qty == 0.0 {
            0.0
        } else {
            total_cost / total_qty
        }
    }

    /// Quadratic inventory risk penalty: gamma * q^2 * sigma^2 * dt.
    pub fn inventory_penalty(&self, volatility: f64, dt: f64) -> f64 {
        self.gamma * self.inventory.powi(2) * volatility.powi(2) * dt
    }

    /// Unrealized PnL at a given mark price.
    pub fn unrealized_pnl(&self, mark_price: f64) -> f64 {
        if self.trades.is_empty() || self.inventory == 0.0 {
            return 0.0;
        }
        let avg = self.average_entry_price();
        (mark_price - avg) * self.inventory
    }

    /// Total PnL (realized + unrealized).
    pub fn total_pnl(&self, mark_price: f64) -> f64 {
        self.realized_pnl + self.unrealized_pnl(mark_price)
    }

    /// Whether the inventory is within allowed limits for a given trade.
    pub fn can_trade(&self, quantity: f64) -> bool {
        (self.inventory + quantity).abs() <= self.max_inventory
    }
}

// ─── Adverse Selection Detector ────────────────────────────────────────────

/// Logistic regression model for detecting adverse selection.
/// Predicts P(adverse move) from market microstructure features.
pub struct AdverseSelectionDetector {
    /// Feature weights [vpin, ofi, spread, volume_ratio, imbalance]
    weights: Array1<f64>,
    /// Bias term
    bias: f64,
    /// Threshold for flagging adverse conditions
    pub threshold: f64,
}

impl AdverseSelectionDetector {
    /// Create a new detector with given weights and threshold.
    pub fn new(weights: Array1<f64>, bias: f64, threshold: f64) -> Self {
        Self {
            weights,
            bias,
            threshold,
        }
    }

    /// Create a detector with default (pre-calibrated) weights.
    /// These weights are illustrative; in production, train on labeled data.
    pub fn default_detector() -> Self {
        // Weights: [vpin, ofi, spread, volume_ratio, imbalance]
        let weights = Array1::from(vec![2.0, -0.5, 1.5, 0.8, -1.0]);
        Self::new(weights, -1.0, 0.6)
    }

    /// Sigmoid activation.
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Predict the probability of adverse selection.
    /// `features`: [vpin, ofi, spread, volume_ratio, imbalance]
    pub fn predict(&self, features: &Array1<f64>) -> f64 {
        let z = self.weights.dot(features) + self.bias;
        Self::sigmoid(z)
    }

    /// Check if current conditions are toxic (above threshold).
    pub fn is_toxic(&self, features: &Array1<f64>) -> bool {
        self.predict(features) > self.threshold
    }

    /// Train the detector using gradient descent on labeled examples.
    /// `data`: Vec of (features, label) where label is 0.0 or 1.0.
    /// Returns final average loss.
    pub fn train(
        &mut self,
        data: &[(Array1<f64>, f64)],
        learning_rate: f64,
        epochs: usize,
    ) -> f64 {
        let n = data.len() as f64;
        let mut loss = 0.0;

        for _ in 0..epochs {
            let mut grad_w: Array1<f64> = Array1::zeros(self.weights.len());
            let mut grad_b = 0.0;
            loss = 0.0;

            for (features, label) in data {
                let pred = self.predict(features);
                let error = pred - label;

                // Binary cross-entropy gradient
                grad_w = grad_w + &(features * error);
                grad_b += error;

                // Binary cross-entropy loss
                let eps = 1e-15;
                let p = pred.clamp(eps, 1.0 - eps);
                loss += -(label * p.ln() + (1.0 - label) * (1.0 - p).ln());
            }

            self.weights = &self.weights - &(grad_w * (learning_rate / n));
            self.bias -= grad_b * learning_rate / n;
            loss /= n;
        }

        loss
    }
}

// ─── Reinforcement Learning Market Maker ───────────────────────────────────

/// Discretized state for the tabular Q-learning market maker.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct RLState {
    /// Inventory bin (0..num_inventory_bins)
    pub inventory_bin: usize,
    /// Volatility bin (0..num_volatility_bins)
    pub volatility_bin: usize,
    /// Flow imbalance bin (0..num_imbalance_bins)
    pub imbalance_bin: usize,
}

/// Action: discrete bid/ask offset levels.
#[derive(Debug, Clone, Copy)]
pub struct QuoteAction {
    /// Bid offset index (0 = tight, higher = wider)
    pub bid_offset: usize,
    /// Ask offset index (0 = tight, higher = wider)
    pub ask_offset: usize,
}

/// Configuration for the RL market maker.
#[derive(Debug, Clone)]
pub struct MarketMakingRLConfig {
    pub num_inventory_bins: usize,
    pub num_volatility_bins: usize,
    pub num_imbalance_bins: usize,
    pub num_spread_levels: usize,
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub epsilon: f64,
    pub inventory_penalty: f64,
}

impl Default for MarketMakingRLConfig {
    fn default() -> Self {
        Self {
            num_inventory_bins: 11,   // -5 to +5
            num_volatility_bins: 5,
            num_imbalance_bins: 5,
            num_spread_levels: 5,
            learning_rate: 0.1,
            discount_factor: 0.95,
            epsilon: 0.1,
            inventory_penalty: 0.01,
        }
    }
}

/// Tabular Q-learning agent for market making.
/// State: (inventory_bin, volatility_bin, imbalance_bin)
/// Action: (bid_offset, ask_offset)
pub struct MarketMakingRL {
    pub config: MarketMakingRLConfig,
    /// Q-table: [inv_bins][vol_bins][imb_bins][bid_levels][ask_levels]
    q_table: Vec<f64>,
    /// Total number of state-action pairs
    total_entries: usize,
}

impl MarketMakingRL {
    pub fn new(config: MarketMakingRLConfig) -> Self {
        let total = config.num_inventory_bins
            * config.num_volatility_bins
            * config.num_imbalance_bins
            * config.num_spread_levels
            * config.num_spread_levels;

        Self {
            q_table: vec![0.0; total],
            total_entries: total,
            config,
        }
    }

    /// Flatten a (state, action) to a 1D index.
    fn index(&self, state: &RLState, action: &QuoteAction) -> usize {
        let c = &self.config;
        let mut idx = state.inventory_bin;
        idx = idx * c.num_volatility_bins + state.volatility_bin;
        idx = idx * c.num_imbalance_bins + state.imbalance_bin;
        idx = idx * c.num_spread_levels + action.bid_offset;
        idx = idx * c.num_spread_levels + action.ask_offset;
        idx.min(self.total_entries - 1)
    }

    /// Get Q-value for a state-action pair.
    pub fn get_q(&self, state: &RLState, action: &QuoteAction) -> f64 {
        self.q_table[self.index(state, action)]
    }

    /// Set Q-value for a state-action pair.
    fn set_q(&mut self, state: &RLState, action: &QuoteAction, value: f64) {
        let idx = self.index(state, action);
        self.q_table[idx] = value;
    }

    /// Find the best action for a given state (greedy).
    pub fn best_action(&self, state: &RLState) -> QuoteAction {
        let mut best_q = f64::NEG_INFINITY;
        let mut best_action = QuoteAction {
            bid_offset: 0,
            ask_offset: 0,
        };

        for bid in 0..self.config.num_spread_levels {
            for ask in 0..self.config.num_spread_levels {
                let action = QuoteAction {
                    bid_offset: bid,
                    ask_offset: ask,
                };
                let q = self.get_q(state, &action);
                if q > best_q {
                    best_q = q;
                    best_action = action;
                }
            }
        }

        best_action
    }

    /// Epsilon-greedy action selection.
    pub fn select_action(&self, state: &RLState, rng: &mut impl Rng) -> QuoteAction {
        if rng.gen::<f64>() < self.config.epsilon {
            // Random action
            QuoteAction {
                bid_offset: rng.gen_range(0..self.config.num_spread_levels),
                ask_offset: rng.gen_range(0..self.config.num_spread_levels),
            }
        } else {
            self.best_action(state)
        }
    }

    /// Q-learning update step.
    pub fn update(
        &mut self,
        state: &RLState,
        action: &QuoteAction,
        reward: f64,
        next_state: &RLState,
    ) {
        let current_q = self.get_q(state, action);
        let next_best = self.best_action(next_state);
        let next_q = self.get_q(next_state, &next_best);

        let new_q = current_q
            + self.config.learning_rate
                * (reward + self.config.discount_factor * next_q - current_q);

        self.set_q(state, action, new_q);
    }

    /// Discretize continuous inventory to a bin index.
    pub fn discretize_inventory(&self, inventory: f64) -> usize {
        let half = (self.config.num_inventory_bins as f64 - 1.0) / 2.0;
        let bin = (inventory + half).round() as i64;
        bin.clamp(0, self.config.num_inventory_bins as i64 - 1) as usize
    }

    /// Discretize continuous volatility to a bin index.
    pub fn discretize_volatility(&self, volatility: f64, max_vol: f64) -> usize {
        let ratio = (volatility / max_vol).clamp(0.0, 0.999);
        let bin = (ratio * self.config.num_volatility_bins as f64) as usize;
        bin.min(self.config.num_volatility_bins - 1)
    }

    /// Discretize continuous imbalance [-1, 1] to a bin index.
    pub fn discretize_imbalance(&self, imbalance: f64) -> usize {
        let shifted = (imbalance + 1.0) / 2.0; // map [-1,1] to [0,1]
        let bin = (shifted * self.config.num_imbalance_bins as f64) as usize;
        bin.min(self.config.num_imbalance_bins - 1)
    }

    /// Convert features to an RLState.
    pub fn featurize(&self, features: &MarketFeatures, max_vol: f64) -> RLState {
        RLState {
            inventory_bin: self.discretize_inventory(features.inventory),
            volatility_bin: self.discretize_volatility(features.volatility, max_vol),
            imbalance_bin: self.discretize_imbalance(features.imbalance),
        }
    }

    /// Compute the reward for a single step.
    /// reward = pnl - inventory_penalty * q^2
    pub fn compute_reward(&self, pnl: f64, inventory: f64) -> f64 {
        pnl - self.config.inventory_penalty * inventory.powi(2)
    }
}

// ─── Spread Predictor (Neural Network) ─────────────────────────────────────

/// Simple feed-forward neural network for predicting optimal spread.
/// Architecture: input -> hidden1 -> hidden2 -> output (softplus)
pub struct SpreadPredictor {
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    w3: Vec<f64>,
    b3: f64,
    input_size: usize,
    hidden1_size: usize,
    hidden2_size: usize,
}

impl SpreadPredictor {
    /// Create a new spread predictor with random initialization.
    pub fn new(input_size: usize, hidden1_size: usize, hidden2_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        let w1 = (0..hidden1_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.gen_range(-0.5..0.5) / (input_size as f64).sqrt())
                    .collect()
            })
            .collect();
        let b1 = vec![0.0; hidden1_size];

        let w2 = (0..hidden2_size)
            .map(|_| {
                (0..hidden1_size)
                    .map(|_| rng.gen_range(-0.5..0.5) / (hidden1_size as f64).sqrt())
                    .collect()
            })
            .collect();
        let b2 = vec![0.0; hidden2_size];

        let w3 = (0..hidden2_size)
            .map(|_| rng.gen_range(-0.5..0.5) / (hidden2_size as f64).sqrt())
            .collect();
        let b3 = 0.0;

        Self {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            input_size,
            hidden1_size,
            hidden2_size,
        }
    }

    fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    fn softplus(x: f64) -> f64 {
        (1.0 + x.exp()).ln()
    }

    /// Forward pass: predict optimal spread from market features.
    pub fn predict(&self, features: &[f64]) -> f64 {
        assert_eq!(features.len(), self.input_size);

        // Layer 1: ReLU
        let mut h1 = vec![0.0; self.hidden1_size];
        for i in 0..self.hidden1_size {
            let mut sum = self.b1[i];
            for j in 0..self.input_size {
                sum += self.w1[i][j] * features[j];
            }
            h1[i] = Self::relu(sum);
        }

        // Layer 2: ReLU
        let mut h2 = vec![0.0; self.hidden2_size];
        for i in 0..self.hidden2_size {
            let mut sum = self.b2[i];
            for j in 0..self.hidden1_size {
                sum += self.w2[i][j] * h1[j];
            }
            h2[i] = Self::relu(sum);
        }

        // Output layer: softplus (ensures positive spread)
        let mut out = self.b3;
        for i in 0..self.hidden2_size {
            out += self.w3[i] * h2[i];
        }

        Self::softplus(out)
    }

    /// Train on a batch of (features, target_spread) pairs using gradient descent.
    /// Returns final average MSE loss.
    pub fn train(
        &mut self,
        data: &[(Vec<f64>, f64)],
        learning_rate: f64,
        epochs: usize,
    ) -> f64 {
        let n = data.len() as f64;
        let mut loss = 0.0;

        for _ in 0..epochs {
            loss = 0.0;

            for (features, target) in data {
                // Forward pass with stored activations
                let mut h1 = vec![0.0; self.hidden1_size];
                let mut h1_pre = vec![0.0; self.hidden1_size];
                for i in 0..self.hidden1_size {
                    let mut sum = self.b1[i];
                    for j in 0..self.input_size {
                        sum += self.w1[i][j] * features[j];
                    }
                    h1_pre[i] = sum;
                    h1[i] = Self::relu(sum);
                }

                let mut h2 = vec![0.0; self.hidden2_size];
                let mut h2_pre = vec![0.0; self.hidden2_size];
                for i in 0..self.hidden2_size {
                    let mut sum = self.b2[i];
                    for j in 0..self.hidden1_size {
                        sum += self.w2[i][j] * h1[j];
                    }
                    h2_pre[i] = sum;
                    h2[i] = Self::relu(sum);
                }

                let mut out_pre = self.b3;
                for i in 0..self.hidden2_size {
                    out_pre += self.w3[i] * h2[i];
                }
                let pred = Self::softplus(out_pre);

                let error = pred - target;
                loss += error * error;

                // Backprop through softplus: d_softplus/dx = sigmoid(x)
                let d_out = error * (1.0 / (1.0 + (-out_pre).exp()));

                // Gradients for w3, b3
                for i in 0..self.hidden2_size {
                    self.w3[i] -= learning_rate * d_out * h2[i] / n;
                }
                self.b3 -= learning_rate * d_out / n;

                // Backprop through layer 2
                let mut d_h2 = vec![0.0; self.hidden2_size];
                for i in 0..self.hidden2_size {
                    d_h2[i] = d_out * self.w3[i] * if h2_pre[i] > 0.0 { 1.0 } else { 0.0 };
                }

                for i in 0..self.hidden2_size {
                    for j in 0..self.hidden1_size {
                        self.w2[i][j] -= learning_rate * d_h2[i] * h1[j] / n;
                    }
                    self.b2[i] -= learning_rate * d_h2[i] / n;
                }

                // Backprop through layer 1
                let mut d_h1 = vec![0.0; self.hidden1_size];
                for j in 0..self.hidden1_size {
                    let mut sum = 0.0;
                    for i in 0..self.hidden2_size {
                        sum += d_h2[i] * self.w2[i][j];
                    }
                    d_h1[j] = sum * if h1_pre[j] > 0.0 { 1.0 } else { 0.0 };
                }

                for i in 0..self.hidden1_size {
                    for j in 0..self.input_size {
                        self.w1[i][j] -= learning_rate * d_h1[i] * features[j] / n;
                    }
                    self.b1[i] -= learning_rate * d_h1[i] / n;
                }
            }

            loss /= n;
        }

        loss
    }
}

// ─── Market Making Simulation ──────────────────────────────────────────────

/// Result of a single simulation step.
#[derive(Debug, Clone)]
pub struct SimulationStep {
    pub timestamp: u64,
    pub mid_price: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub inventory: f64,
    pub pnl: f64,
    pub spread: f64,
    pub fills: i32,
}

/// Run a market making backtest on historical candle data.
/// Uses the Avellaneda-Stoikov model with adverse selection detection.
pub fn simulate_market_making(
    candles: &[Candle],
    as_config: AvellanedaStoikovConfig,
    max_inventory: f64,
) -> Vec<SimulationStep> {
    if candles.len() < 20 {
        return Vec::new();
    }

    let model = AvellanedaStoikov::new(as_config.clone());
    let mut inventory = InventoryManager::new(max_inventory, as_config.gamma);
    let detector = AdverseSelectionDetector::default_detector();

    let total_steps = candles.len();
    let mut results = Vec::with_capacity(total_steps);
    let mut rng = rand::thread_rng();

    // Pre-compute volatility for each window
    let vol_window = 20;

    for i in vol_window..total_steps {
        let prices: Vec<f64> = candles[i - vol_window..i]
            .iter()
            .map(|c| c.close)
            .collect();
        let vol = realized_volatility(&prices);
        let mid = candles[i].close;
        let time_remaining = (total_steps - i) as f64 / total_steps as f64;

        // Adverse selection check
        let vpin = compute_vpin(&candles[i.saturating_sub(10)..i], mid * 0.001);
        let imbalance = if candles[i].close > candles[i].open {
            0.5
        } else {
            -0.5
        };
        let as_features = Array1::from(vec![
            vpin,
            imbalance,
            vol * 100.0,
            candles[i].volume / 1000.0,
            imbalance,
        ]);

        let is_toxic = detector.is_toxic(&as_features);

        // Compute quotes
        let (mut bid, mut ask) = model.compute_quotes(
            mid,
            inventory.inventory,
            vol,
            time_remaining,
        );

        // Widen spread if toxic
        if is_toxic {
            let extra = (ask - bid) * 0.5;
            bid -= extra;
            ask += extra;
        }

        let spread = ask - bid;

        // Simulate fills based on price movement
        let mut fills = 0;
        let _price_move = candles[i].close - candles[i].open;

        // Bid fill: price dropped to our bid level
        if candles[i].low <= bid && inventory.can_trade(1.0) {
            let fill_prob = model.fill_probability((mid - bid) / mid);
            if rng.gen::<f64>() < fill_prob {
                inventory.record_fill(bid, 1.0);
                fills += 1;
            }
        }

        // Ask fill: price rose to our ask level
        if candles[i].high >= ask && inventory.can_trade(-1.0) {
            let fill_prob = model.fill_probability((ask - mid) / mid);
            if rng.gen::<f64>() < fill_prob {
                inventory.record_fill(ask, -1.0);
                fills += 1;
            }
        }

        let pnl = inventory.total_pnl(mid);

        results.push(SimulationStep {
            timestamp: candles[i].timestamp,
            mid_price: mid,
            bid_price: bid,
            ask_price: ask,
            inventory: inventory.inventory,
            pnl,
            spread,
            fills,
        });
    }

    results
}

/// Compute summary statistics from simulation results.
#[derive(Debug, Clone)]
pub struct SimulationSummary {
    pub total_steps: usize,
    pub total_fills: i32,
    pub final_pnl: f64,
    pub max_pnl: f64,
    pub min_pnl: f64,
    pub max_drawdown: f64,
    pub avg_spread: f64,
    pub avg_inventory: f64,
    pub max_inventory: f64,
    pub sharpe_ratio: f64,
}

pub fn compute_summary(results: &[SimulationStep]) -> SimulationSummary {
    if results.is_empty() {
        return SimulationSummary {
            total_steps: 0,
            total_fills: 0,
            final_pnl: 0.0,
            max_pnl: 0.0,
            min_pnl: 0.0,
            max_drawdown: 0.0,
            avg_spread: 0.0,
            avg_inventory: 0.0,
            max_inventory: 0.0,
            sharpe_ratio: 0.0,
        };
    }

    let total_fills: i32 = results.iter().map(|s| s.fills).sum();
    let final_pnl = results.last().map(|s| s.pnl).unwrap_or(0.0);

    let mut max_pnl = f64::NEG_INFINITY;
    let mut min_pnl = f64::INFINITY;
    let mut max_drawdown = 0.0;
    let mut peak = f64::NEG_INFINITY;

    let mut pnl_changes: Vec<f64> = Vec::new();
    let mut prev_pnl = 0.0;

    for step in results {
        if step.pnl > max_pnl {
            max_pnl = step.pnl;
        }
        if step.pnl < min_pnl {
            min_pnl = step.pnl;
        }
        if step.pnl > peak {
            peak = step.pnl;
        }
        let dd = peak - step.pnl;
        if dd > max_drawdown {
            max_drawdown = dd;
        }
        pnl_changes.push(step.pnl - prev_pnl);
        prev_pnl = step.pnl;
    }

    let avg_spread: f64 =
        results.iter().map(|s| s.spread).sum::<f64>() / results.len() as f64;
    let avg_inventory: f64 =
        results.iter().map(|s| s.inventory.abs()).sum::<f64>() / results.len() as f64;
    let max_inventory: f64 = results
        .iter()
        .map(|s| s.inventory.abs())
        .fold(0.0, f64::max);

    // Sharpe ratio of PnL changes
    let n = pnl_changes.len() as f64;
    let mean_ret = pnl_changes.iter().sum::<f64>() / n;
    let var = pnl_changes
        .iter()
        .map(|r| (r - mean_ret).powi(2))
        .sum::<f64>()
        / (n - 1.0).max(1.0);
    let std_ret = var.sqrt();
    let sharpe_ratio = if std_ret > 0.0 {
        mean_ret / std_ret * (252.0_f64).sqrt()
    } else {
        0.0
    };

    SimulationSummary {
        total_steps: results.len(),
        total_fills,
        final_pnl,
        max_pnl,
        min_pnl,
        max_drawdown,
        avg_spread,
        avg_inventory,
        max_inventory,
        sharpe_ratio,
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_candles() -> Vec<Candle> {
        (0..100)
            .map(|i| {
                let base = 50000.0 + (i as f64 * 0.1).sin() * 500.0;
                Candle {
                    timestamp: 1700000000 + i * 60000,
                    open: base,
                    high: base + 100.0,
                    low: base - 100.0,
                    close: base + 50.0 * if i % 2 == 0 { 1.0 } else { -1.0 },
                    volume: 10.0 + (i as f64 * 0.3).sin().abs() * 5.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_realized_volatility() {
        let prices = vec![100.0, 101.0, 99.5, 100.5, 102.0];
        let vol = realized_volatility(&prices);
        assert!(vol > 0.0, "Volatility should be positive");
        assert!(vol < 1.0, "Volatility should be reasonable");
    }

    #[test]
    fn test_garman_klass() {
        let candles = sample_candles();
        let vol = garman_klass_volatility(&candles[..20]);
        assert!(vol > 0.0);
    }

    #[test]
    fn test_orderbook_imbalance() {
        let ob = OrderBook {
            bids: vec![
                OrderBookLevel { price: 100.0, size: 10.0 },
                OrderBookLevel { price: 99.0, size: 5.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 101.0, size: 5.0 },
                OrderBookLevel { price: 102.0, size: 5.0 },
            ],
        };
        let imb = ob.imbalance(2);
        // bid_vol=15, ask_vol=10 => imb = 5/25 = 0.2
        assert!((imb - 0.2).abs() < 1e-10);
        assert!((ob.mid_price().unwrap() - 100.5).abs() < 1e-10);
        assert!((ob.spread().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_avellaneda_stoikov_quotes() {
        let config = AvellanedaStoikovConfig {
            gamma: 0.1,
            kappa: 1.5,
            time_horizon: 3600.0,
        };
        let model = AvellanedaStoikov::new(config);

        // Zero inventory => symmetric quotes
        let (bid, ask) = model.compute_quotes(100.0, 0.0, 0.01, 1.0);
        let mid = (bid + ask) / 2.0;
        assert!((mid - 100.0).abs() < 1e-6, "Mid should be close to 100 with zero inventory");
        assert!(ask > bid, "Ask must be above bid");

        // Long inventory => skew quotes down
        let (bid_long, ask_long) = model.compute_quotes(100.0, 5.0, 0.01, 1.0);
        let mid_long = (bid_long + ask_long) / 2.0;
        assert!(mid_long < 100.0, "Long inventory should push quotes lower");
    }

    #[test]
    fn test_inventory_manager() {
        let mut mgr = InventoryManager::new(10.0, 0.1);
        assert!(mgr.record_fill(100.0, 5.0));
        assert_eq!(mgr.inventory, 5.0);
        assert!(mgr.can_trade(5.0));
        assert!(!mgr.can_trade(6.0));
        assert!(mgr.record_fill(105.0, -5.0));
        assert_eq!(mgr.inventory, 0.0);
        assert!(mgr.realized_pnl > 0.0); // bought at 100, sold at 105
    }

    #[test]
    fn test_adverse_selection_detector() {
        let mut detector = AdverseSelectionDetector::default_detector();

        // High VPIN, wide spread -> likely toxic
        let toxic_features = Array1::from(vec![0.8, -0.5, 2.0, 1.5, -0.8]);
        let prob = detector.predict(&toxic_features);
        assert!(prob > 0.0 && prob < 1.0, "Probability must be in (0, 1)");

        // Train on some dummy data
        let data: Vec<(Array1<f64>, f64)> = (0..50)
            .map(|i| {
                let label = if i % 2 == 0 { 1.0 } else { 0.0 };
                let feat = Array1::from(vec![
                    label * 0.8 + 0.1,
                    -0.3 + label * 0.2,
                    0.5 + label * 1.0,
                    0.5,
                    -0.2 + label * 0.5,
                ]);
                (feat, label)
            })
            .collect();

        let loss = detector.train(&data, 0.1, 100);
        assert!(loss < 1.0, "Loss should decrease with training");
    }

    #[test]
    fn test_rl_agent() {
        let config = MarketMakingRLConfig::default();
        let mut agent = MarketMakingRL::new(config);
        let mut rng = rand::thread_rng();

        let state = RLState {
            inventory_bin: 5,
            volatility_bin: 2,
            imbalance_bin: 2,
        };
        let action = agent.select_action(&state, &mut rng);
        assert!(action.bid_offset < 5);
        assert!(action.ask_offset < 5);

        // Run a few updates
        let next_state = RLState {
            inventory_bin: 6,
            volatility_bin: 2,
            imbalance_bin: 3,
        };
        agent.update(&state, &action, 1.0, &next_state);
        let q = agent.get_q(&state, &action);
        assert!(q > 0.0, "Q-value should increase after positive reward");
    }

    #[test]
    fn test_spread_predictor() {
        let mut pred = SpreadPredictor::new(5, 16, 8);

        // Predict something
        let features = vec![0.5, 0.01, 0.2, 0.3, 0.1];
        let spread = pred.predict(&features);
        assert!(spread > 0.0, "Spread must be positive (softplus output)");

        // Train on dummy data
        let data: Vec<(Vec<f64>, f64)> = (0..100)
            .map(|_| {
                let f = vec![0.5, 0.02, 0.3, 0.5, 0.1];
                let target = 0.002; // 20 bps spread
                (f, target)
            })
            .collect();

        let loss = pred.train(&data, 0.01, 50);
        assert!(loss < 1.0);
    }

    #[test]
    fn test_simulation() {
        let candles = sample_candles();
        let config = AvellanedaStoikovConfig::default();
        let results = simulate_market_making(&candles, config, 10.0);
        assert!(!results.is_empty());

        let summary = compute_summary(&results);
        assert!(summary.total_steps > 0);
        assert!(summary.avg_spread > 0.0);
    }

    #[test]
    fn test_vpin() {
        let candles = sample_candles();
        let vpin = compute_vpin(&candles[..20], 100.0);
        assert!(vpin >= 0.0 && vpin <= 1.0);
    }

    #[test]
    fn test_discretization() {
        let config = MarketMakingRLConfig::default();
        let agent = MarketMakingRL::new(config);

        assert_eq!(agent.discretize_inventory(0.0), 5);  // center bin
        assert_eq!(agent.discretize_inventory(-5.0), 0); // lowest bin
        assert_eq!(agent.discretize_inventory(5.0), 10); // highest bin
        assert_eq!(agent.discretize_imbalance(-1.0), 0);
        assert_eq!(agent.discretize_imbalance(1.0), 4);
    }
}
