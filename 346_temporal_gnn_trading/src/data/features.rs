//! Feature engineering for market data

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Market features extracted from raw data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketFeatures {
    /// Price features
    pub price: f64,
    pub price_return: f64,
    pub log_return: f64,

    /// Volume features
    pub volume: f64,
    pub volume_ma: f64,
    pub volume_ratio: f64,

    /// Volatility features
    pub volatility: f64,
    pub atr: f64,

    /// Order book features
    pub spread: f64,
    pub bid_depth: f64,
    pub ask_depth: f64,
    pub imbalance: f64,

    /// Momentum features
    pub rsi: f64,
    pub momentum: f64,

    /// Funding features (for perps)
    pub funding_rate: f64,
}

impl Default for MarketFeatures {
    fn default() -> Self {
        Self {
            price: 0.0,
            price_return: 0.0,
            log_return: 0.0,
            volume: 0.0,
            volume_ma: 0.0,
            volume_ratio: 1.0,
            volatility: 0.0,
            atr: 0.0,
            spread: 0.0,
            bid_depth: 0.0,
            ask_depth: 0.0,
            imbalance: 0.0,
            rsi: 50.0,
            momentum: 0.0,
            funding_rate: 0.0,
        }
    }
}

impl MarketFeatures {
    /// Convert to feature vector
    pub fn to_vector(&self) -> Array1<f64> {
        Array1::from_vec(vec![
            self.price.ln().max(0.0),
            self.price_return,
            self.log_return,
            self.volume.ln().max(0.0),
            self.volume_ratio.ln(),
            self.volatility,
            self.atr,
            self.spread.ln().max(-10.0),
            self.imbalance,
            (self.rsi - 50.0) / 50.0, // Normalize RSI to [-1, 1]
            self.momentum,
            self.funding_rate * 100.0, // Scale funding rate
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
    /// Price history
    price_history: VecDeque<f64>,
    /// Volume history
    volume_history: VecDeque<f64>,
    /// Return history for volatility calculation
    return_history: VecDeque<f64>,
    /// High history for ATR
    high_history: VecDeque<f64>,
    /// Low history for ATR
    low_history: VecDeque<f64>,
    /// Window size for calculations
    window_size: usize,
    /// Last price for return calculation
    last_price: Option<f64>,
}

impl FeatureEngine {
    /// Create new feature engine
    pub fn new(window_size: usize) -> Self {
        Self {
            price_history: VecDeque::with_capacity(window_size),
            volume_history: VecDeque::with_capacity(window_size),
            return_history: VecDeque::with_capacity(window_size),
            high_history: VecDeque::with_capacity(window_size),
            low_history: VecDeque::with_capacity(window_size),
            window_size,
            last_price: None,
        }
    }

    /// Update with new price data
    pub fn update(&mut self, price: f64, volume: f64, high: f64, low: f64) {
        // Calculate return
        let price_return = if let Some(last) = self.last_price {
            (price - last) / last
        } else {
            0.0
        };

        // Update histories
        if self.price_history.len() >= self.window_size {
            self.price_history.pop_front();
            self.volume_history.pop_front();
            self.return_history.pop_front();
            self.high_history.pop_front();
            self.low_history.pop_front();
        }

        self.price_history.push_back(price);
        self.volume_history.push_back(volume);
        self.return_history.push_back(price_return);
        self.high_history.push_back(high);
        self.low_history.push_back(low);

        self.last_price = Some(price);
    }

    /// Calculate current features
    pub fn calculate(&self) -> MarketFeatures {
        let price = self.last_price.unwrap_or(0.0);

        // Price return
        let price_return = self.return_history.back().copied().unwrap_or(0.0);
        let log_return = if price > 0.0 && self.price_history.len() > 1 {
            (price / self.price_history[self.price_history.len() - 2]).ln()
        } else {
            0.0
        };

        // Volume features
        let volume = self.volume_history.back().copied().unwrap_or(0.0);
        let volume_ma = if !self.volume_history.is_empty() {
            self.volume_history.iter().sum::<f64>() / self.volume_history.len() as f64
        } else {
            volume
        };
        let volume_ratio = if volume_ma > 0.0 { volume / volume_ma } else { 1.0 };

        // Volatility (standard deviation of returns)
        let volatility = self.calculate_volatility();

        // ATR (Average True Range)
        let atr = self.calculate_atr();

        // RSI
        let rsi = self.calculate_rsi();

        // Momentum (rate of change)
        let momentum = self.calculate_momentum();

        MarketFeatures {
            price,
            price_return,
            log_return,
            volume,
            volume_ma,
            volume_ratio,
            volatility,
            atr,
            spread: 0.0,
            bid_depth: 0.0,
            ask_depth: 0.0,
            imbalance: 0.0,
            rsi,
            momentum,
            funding_rate: 0.0,
        }
    }

    /// Calculate volatility (standard deviation of returns)
    fn calculate_volatility(&self) -> f64 {
        if self.return_history.len() < 2 {
            return 0.0;
        }

        let mean: f64 = self.return_history.iter().sum::<f64>() / self.return_history.len() as f64;
        let variance: f64 = self.return_history
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / self.return_history.len() as f64;

        variance.sqrt()
    }

    /// Calculate ATR
    fn calculate_atr(&self) -> f64 {
        if self.high_history.len() < 2 {
            return 0.0;
        }

        let mut true_ranges: Vec<f64> = Vec::new();

        for i in 1..self.high_history.len() {
            let high = self.high_history[i];
            let low = self.low_history[i];
            let prev_close = self.price_history[i - 1];

            let tr = (high - low)
                .max((high - prev_close).abs())
                .max((low - prev_close).abs());

            true_ranges.push(tr);
        }

        if true_ranges.is_empty() {
            0.0
        } else {
            true_ranges.iter().sum::<f64>() / true_ranges.len() as f64
        }
    }

    /// Calculate RSI
    fn calculate_rsi(&self) -> f64 {
        if self.return_history.len() < 2 {
            return 50.0;
        }

        let gains: Vec<f64> = self.return_history.iter().map(|&r| r.max(0.0)).collect();
        let losses: Vec<f64> = self.return_history.iter().map(|&r| (-r).max(0.0)).collect();

        let avg_gain: f64 = gains.iter().sum::<f64>() / gains.len() as f64;
        let avg_loss: f64 = losses.iter().sum::<f64>() / losses.len() as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    /// Calculate momentum (rate of change)
    fn calculate_momentum(&self) -> f64 {
        if self.price_history.len() < 2 {
            return 0.0;
        }

        let lookback = self.price_history.len().min(10);
        let old_price = self.price_history[self.price_history.len() - lookback];
        let current_price = self.price_history.back().unwrap();

        if old_price > 0.0 {
            (current_price - old_price) / old_price
        } else {
            0.0
        }
    }

    /// Reset the engine
    pub fn reset(&mut self) {
        self.price_history.clear();
        self.volume_history.clear();
        self.return_history.clear();
        self.high_history.clear();
        self.low_history.clear();
        self.last_price = None;
    }
}

impl Default for FeatureEngine {
    fn default() -> Self {
        Self::new(20)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_engine() {
        let mut engine = FeatureEngine::new(10);

        // Add some price data
        for i in 0..15 {
            let price = 100.0 + i as f64;
            let volume = 1000.0 + i as f64 * 10.0;
            engine.update(price, volume, price + 1.0, price - 1.0);
        }

        let features = engine.calculate();

        assert!(features.price > 0.0);
        assert!(features.volatility >= 0.0);
        assert!(features.rsi >= 0.0 && features.rsi <= 100.0);
    }

    #[test]
    fn test_market_features_vector() {
        let features = MarketFeatures {
            price: 50000.0,
            price_return: 0.01,
            log_return: 0.00995,
            volume: 1000.0,
            volume_ratio: 1.5,
            volatility: 0.02,
            atr: 500.0,
            spread: 10.0,
            imbalance: 0.3,
            rsi: 65.0,
            momentum: 0.05,
            funding_rate: 0.0001,
            ..Default::default()
        };

        let vector = features.to_vector();
        assert_eq!(vector.len(), MarketFeatures::dim());
    }
}
