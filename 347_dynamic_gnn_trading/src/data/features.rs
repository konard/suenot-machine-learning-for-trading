//! Feature engineering for market data

use ndarray::Array1;
use std::collections::VecDeque;

use super::{Kline, OrderBookSnapshot, Trade};

/// Market features computed from raw data
#[derive(Debug, Clone)]
pub struct MarketFeatures {
    /// Log returns
    pub returns: f64,
    /// Volatility (rolling std of returns)
    pub volatility: f64,
    /// RSI indicator
    pub rsi: f64,
    /// MACD value
    pub macd: f64,
    /// MACD signal line
    pub macd_signal: f64,
    /// MACD histogram
    pub macd_hist: f64,
    /// Bollinger band position (-1 to 1)
    pub bb_position: f64,
    /// Order book imbalance
    pub ob_imbalance: f64,
    /// Bid-ask spread
    pub spread: f64,
    /// Trade flow imbalance
    pub trade_imbalance: f64,
    /// Volume relative to average
    pub relative_volume: f64,
    /// Price momentum
    pub momentum: f64,
    /// Timestamp
    pub timestamp: u64,
}

impl Default for MarketFeatures {
    fn default() -> Self {
        Self {
            returns: 0.0,
            volatility: 0.0,
            rsi: 50.0,
            macd: 0.0,
            macd_signal: 0.0,
            macd_hist: 0.0,
            bb_position: 0.0,
            ob_imbalance: 0.0,
            spread: 0.0,
            trade_imbalance: 0.0,
            relative_volume: 1.0,
            momentum: 0.0,
            timestamp: 0,
        }
    }
}

impl MarketFeatures {
    /// Convert to feature vector for ML
    pub fn to_vector(&self) -> Array1<f64> {
        Array1::from_vec(vec![
            self.returns.tanh(),           // Bound returns
            self.volatility.min(0.1),      // Cap volatility
            (self.rsi - 50.0) / 50.0,      // Normalize RSI to [-1, 1]
            self.macd.tanh(),              // Bound MACD
            self.macd_signal.tanh(),
            self.macd_hist.tanh(),
            self.bb_position,
            self.ob_imbalance,
            self.spread.min(0.01) / 0.01,  // Normalize spread
            self.trade_imbalance,
            (self.relative_volume.ln()).tanh(), // Log-normalize volume
            self.momentum.tanh(),
        ])
    }

    /// Get feature dimension
    pub fn dim() -> usize {
        12
    }
}

/// Feature engineering engine
#[derive(Debug)]
pub struct FeatureEngine {
    /// Price history for calculations
    prices: VecDeque<f64>,
    /// Volume history
    volumes: VecDeque<f64>,
    /// Returns history
    returns: VecDeque<f64>,
    /// EMA values for MACD
    ema_12: Option<f64>,
    ema_26: Option<f64>,
    ema_signal: Option<f64>,
    /// Gains/losses for RSI
    gains: VecDeque<f64>,
    losses: VecDeque<f64>,
    /// Configuration
    pub window_size: usize,
    pub rsi_period: usize,
    pub bb_period: usize,
}

impl FeatureEngine {
    /// Create a new feature engine
    pub fn new() -> Self {
        Self {
            prices: VecDeque::new(),
            volumes: VecDeque::new(),
            returns: VecDeque::new(),
            ema_12: None,
            ema_26: None,
            ema_signal: None,
            gains: VecDeque::new(),
            losses: VecDeque::new(),
            window_size: 100,
            rsi_period: 14,
            bb_period: 20,
        }
    }

    /// Create with custom window size
    pub fn with_window(window_size: usize) -> Self {
        Self {
            window_size,
            ..Self::new()
        }
    }

    /// Update with new price and volume
    pub fn update(&mut self, price: f64, volume: f64) {
        // Calculate return
        let ret = if let Some(&last_price) = self.prices.back() {
            if last_price > 0.0 {
                (price / last_price).ln()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Update histories
        self.prices.push_back(price);
        self.volumes.push_back(volume);
        self.returns.push_back(ret);

        // Update RSI components
        if ret > 0.0 {
            self.gains.push_back(ret);
            self.losses.push_back(0.0);
        } else {
            self.gains.push_back(0.0);
            self.losses.push_back(-ret);
        }

        // Maintain window sizes
        while self.prices.len() > self.window_size {
            self.prices.pop_front();
        }
        while self.volumes.len() > self.window_size {
            self.volumes.pop_front();
        }
        while self.returns.len() > self.window_size {
            self.returns.pop_front();
        }
        while self.gains.len() > self.rsi_period {
            self.gains.pop_front();
        }
        while self.losses.len() > self.rsi_period {
            self.losses.pop_front();
        }

        // Update EMAs for MACD
        self.update_ema(price);
    }

    /// Update from kline
    pub fn update_from_kline(&mut self, kline: &Kline) {
        self.update(kline.close, kline.volume);
    }

    /// Update EMAs
    fn update_ema(&mut self, price: f64) {
        let alpha_12 = 2.0 / 13.0;
        let alpha_26 = 2.0 / 27.0;
        let alpha_signal = 2.0 / 10.0;

        self.ema_12 = Some(match self.ema_12 {
            Some(ema) => alpha_12 * price + (1.0 - alpha_12) * ema,
            None => price,
        });

        self.ema_26 = Some(match self.ema_26 {
            Some(ema) => alpha_26 * price + (1.0 - alpha_26) * ema,
            None => price,
        });

        // Update signal line
        if let (Some(ema12), Some(ema26)) = (self.ema_12, self.ema_26) {
            let macd = ema12 - ema26;
            self.ema_signal = Some(match self.ema_signal {
                Some(signal) => alpha_signal * macd + (1.0 - alpha_signal) * signal,
                None => macd,
            });
        }
    }

    /// Compute all features
    pub fn compute_features(
        &self,
        orderbook: Option<&OrderBookSnapshot>,
        trades: Option<&[Trade]>,
    ) -> MarketFeatures {
        let returns = self.returns.back().copied().unwrap_or(0.0);
        let volatility = self.compute_volatility();
        let rsi = self.compute_rsi();
        let (macd, macd_signal, macd_hist) = self.compute_macd();
        let bb_position = self.compute_bb_position();
        let momentum = self.compute_momentum(10);
        let relative_volume = self.compute_relative_volume();

        let (ob_imbalance, spread) = if let Some(ob) = orderbook {
            (ob.imbalance, ob.spread_pct)
        } else {
            (0.0, 0.0)
        };

        let trade_imbalance = if let Some(t) = trades {
            self.compute_trade_imbalance(t)
        } else {
            0.0
        };

        MarketFeatures {
            returns,
            volatility,
            rsi,
            macd,
            macd_signal,
            macd_hist,
            bb_position,
            ob_imbalance,
            spread,
            trade_imbalance,
            relative_volume,
            momentum,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }

    /// Compute volatility (standard deviation of returns)
    fn compute_volatility(&self) -> f64 {
        if self.returns.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = self.returns.iter().copied().collect();
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / (returns.len() - 1) as f64;

        variance.sqrt()
    }

    /// Compute RSI
    fn compute_rsi(&self) -> f64 {
        if self.gains.is_empty() || self.losses.is_empty() {
            return 50.0;
        }

        let avg_gain: f64 = self.gains.iter().sum::<f64>() / self.gains.len() as f64;
        let avg_loss: f64 = self.losses.iter().sum::<f64>() / self.losses.len() as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    /// Compute MACD values
    fn compute_macd(&self) -> (f64, f64, f64) {
        match (self.ema_12, self.ema_26, self.ema_signal) {
            (Some(ema12), Some(ema26), Some(signal)) => {
                let macd = ema12 - ema26;
                let hist = macd - signal;
                (macd, signal, hist)
            }
            _ => (0.0, 0.0, 0.0),
        }
    }

    /// Compute Bollinger Band position (-1 at lower band, 1 at upper band)
    fn compute_bb_position(&self) -> f64 {
        if self.prices.len() < self.bb_period {
            return 0.0;
        }

        let prices: Vec<f64> = self.prices.iter().rev().take(self.bb_period).copied().collect();
        let current_price = *self.prices.back().unwrap_or(&0.0);

        let mean: f64 = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance: f64 = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>()
            / prices.len() as f64;
        let std = variance.sqrt();

        if std == 0.0 {
            return 0.0;
        }

        // Position relative to bands (2 std deviations)
        let band_width = 2.0 * std;
        ((current_price - mean) / band_width).clamp(-1.0, 1.0)
    }

    /// Compute price momentum over N periods
    fn compute_momentum(&self, periods: usize) -> f64 {
        if self.prices.len() < periods + 1 {
            return 0.0;
        }

        let current = *self.prices.back().unwrap_or(&0.0);
        let past = self.prices[self.prices.len() - periods - 1];

        if past > 0.0 {
            (current - past) / past
        } else {
            0.0
        }
    }

    /// Compute relative volume (current vs average)
    fn compute_relative_volume(&self) -> f64 {
        if self.volumes.is_empty() {
            return 1.0;
        }

        let current = *self.volumes.back().unwrap_or(&0.0);
        let average: f64 = self.volumes.iter().sum::<f64>() / self.volumes.len() as f64;

        if average > 0.0 {
            current / average
        } else {
            1.0
        }
    }

    /// Compute trade imbalance from recent trades
    fn compute_trade_imbalance(&self, trades: &[Trade]) -> f64 {
        if trades.is_empty() {
            return 0.0;
        }

        let mut buy_volume = 0.0;
        let mut sell_volume = 0.0;

        for trade in trades {
            if trade.is_buy() {
                buy_volume += trade.size;
            } else {
                sell_volume += trade.size;
            }
        }

        let total = buy_volume + sell_volume;
        if total > 0.0 {
            (buy_volume - sell_volume) / total
        } else {
            0.0
        }
    }

    /// Get current price
    pub fn current_price(&self) -> Option<f64> {
        self.prices.back().copied()
    }

    /// Get price history
    pub fn price_history(&self) -> Vec<f64> {
        self.prices.iter().copied().collect()
    }

    /// Get returns history
    pub fn returns_history(&self) -> Vec<f64> {
        self.returns.iter().copied().collect()
    }

    /// Check if engine has enough data
    pub fn is_ready(&self) -> bool {
        self.prices.len() >= self.bb_period
    }

    /// Reset the engine
    pub fn reset(&mut self) {
        self.prices.clear();
        self.volumes.clear();
        self.returns.clear();
        self.gains.clear();
        self.losses.clear();
        self.ema_12 = None;
        self.ema_26 = None;
        self.ema_signal = None;
    }
}

impl Default for FeatureEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_engine() {
        let mut engine = FeatureEngine::new();

        // Add some price data
        for i in 0..30 {
            let price = 100.0 + (i as f64) * 0.5 + (i as f64 * 0.1).sin();
            engine.update(price, 1000.0);
        }

        let features = engine.compute_features(None, None);

        // RSI should be above 50 (uptrend)
        assert!(features.rsi > 50.0);
        assert!(features.volatility > 0.0);
    }

    #[test]
    fn test_rsi_calculation() {
        let mut engine = FeatureEngine::new();

        // Simulate uptrend
        for i in 0..20 {
            engine.update(100.0 + i as f64, 1000.0);
        }

        let features = engine.compute_features(None, None);
        assert!(features.rsi > 70.0); // Should be overbought

        // Simulate downtrend
        engine.reset();
        for i in 0..20 {
            engine.update(100.0 - i as f64 * 0.5, 1000.0);
        }

        let features = engine.compute_features(None, None);
        assert!(features.rsi < 30.0); // Should be oversold
    }

    #[test]
    fn test_feature_vector() {
        let features = MarketFeatures::default();
        let vec = features.to_vector();
        assert_eq!(vec.len(), MarketFeatures::dim());
    }

    #[test]
    fn test_momentum() {
        let mut engine = FeatureEngine::new();

        for i in 0..15 {
            engine.update(100.0 + i as f64 * 2.0, 1000.0);
        }

        let features = engine.compute_features(None, None);
        assert!(features.momentum > 0.0); // Positive momentum in uptrend
    }
}
