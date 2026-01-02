//! Node feature extraction for Graph Transformer

use crate::api::types::{Kline, OrderBook, Ticker};
use crate::features::indicators::TechnicalIndicators;
use ndarray::Array1;

/// Node features structure
#[derive(Debug, Clone)]
pub struct NodeFeatures {
    /// Feature vector
    pub features: Array1<f64>,
    /// Feature names for interpretability
    pub feature_names: Vec<String>,
}

impl NodeFeatures {
    /// Create features from ticker data (snapshot)
    pub fn from_ticker(ticker: &Ticker) -> Self {
        let mut features = Vec::new();
        let mut names = Vec::new();

        // Price features
        features.push(ticker.price_change_24h);
        names.push("price_change_24h".to_string());

        // Volatility proxy (high-low range / price)
        let range = (ticker.high_24h - ticker.low_24h) / ticker.last_price;
        features.push(range);
        names.push("daily_range".to_string());

        // Volume features (log-normalized)
        let log_volume = (ticker.volume_24h + 1.0).ln();
        features.push(log_volume);
        names.push("log_volume".to_string());

        let log_turnover = (ticker.turnover_24h + 1.0).ln();
        features.push(log_turnover);
        names.push("log_turnover".to_string());

        // Spread features
        let spread = (ticker.ask_price - ticker.bid_price) / ticker.last_price;
        features.push(spread * 10000.0); // In bps
        names.push("spread_bps".to_string());

        // Mid-price position (between bid and ask)
        let mid = (ticker.bid_price + ticker.ask_price) / 2.0;
        let mid_deviation = (ticker.last_price - mid) / mid;
        features.push(mid_deviation);
        names.push("mid_deviation".to_string());

        // Position in daily range
        let range_position = if ticker.high_24h > ticker.low_24h {
            (ticker.last_price - ticker.low_24h) / (ticker.high_24h - ticker.low_24h)
        } else {
            0.5
        };
        features.push(range_position);
        names.push("range_position".to_string());

        Self {
            features: Array1::from_vec(features),
            feature_names: names,
        }
    }

    /// Create features from kline history
    pub fn from_klines(klines: &[Kline]) -> Self {
        if klines.is_empty() {
            return Self {
                features: Array1::zeros(0),
                feature_names: vec![],
            };
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let lows: Vec<f64> = klines.iter().map(|k| k.low).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        let mut features = Vec::new();
        let mut names = Vec::new();

        // Returns at different timeframes
        let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();

        if !returns.is_empty() {
            features.push(*returns.last().unwrap_or(&0.0));
            names.push("return_1".to_string());

            if returns.len() >= 5 {
                let ret_5: f64 = returns[returns.len()-5..].iter().sum();
                features.push(ret_5);
                names.push("return_5".to_string());
            }

            if returns.len() >= 15 {
                let ret_15: f64 = returns[returns.len()-15..].iter().sum();
                features.push(ret_15);
                names.push("return_15".to_string());
            }
        }

        // RSI
        let rsi = TechnicalIndicators::rsi(&closes, 14);
        if !rsi.is_empty() {
            features.push((*rsi.last().unwrap() - 50.0) / 50.0); // Normalize to [-1, 1]
            names.push("rsi_14".to_string());
        }

        // MACD
        let (macd, signal, _) = TechnicalIndicators::macd(&closes, 12, 26, 9);
        if !macd.is_empty() && !signal.is_empty() {
            let macd_signal = macd.last().unwrap() - signal.last().unwrap();
            features.push(macd_signal / closes.last().unwrap() * 100.0);
            names.push("macd_signal".to_string());
        }

        // Bollinger Band position
        let (upper, middle, lower) = TechnicalIndicators::bollinger_bands(&closes, 20, 2.0);
        if !upper.is_empty() {
            let last_close = *closes.last().unwrap();
            let bb_pos = (last_close - *lower.last().unwrap()) /
                        (*upper.last().unwrap() - *lower.last().unwrap()).max(0.0001);
            features.push(bb_pos);
            names.push("bb_position".to_string());
        }

        // Volatility
        let std_dev = TechnicalIndicators::ema(&returns.iter().map(|r| r.abs()).collect::<Vec<_>>(), 14);
        if !std_dev.is_empty() {
            features.push(*std_dev.last().unwrap() * 100.0);
            names.push("volatility".to_string());
        }

        // Volume features
        let vol_ma = TechnicalIndicators::ema(&volumes, 20);
        if !vol_ma.is_empty() {
            let vol_ratio = volumes.last().unwrap() / vol_ma.last().unwrap().max(1.0);
            features.push(vol_ratio);
            names.push("volume_ratio".to_string());
        }

        // Momentum
        let momentum = TechnicalIndicators::momentum(&closes, 10);
        if !momentum.is_empty() {
            features.push(*momentum.last().unwrap());
            names.push("momentum_10".to_string());
        }

        Self {
            features: Array1::from_vec(features),
            feature_names: names,
        }
    }

    /// Create features from order book
    pub fn from_orderbook(ob: &OrderBook) -> Self {
        let mut features = Vec::new();
        let mut names = Vec::new();

        // Spread
        if let Some(spread) = ob.spread_bps() {
            features.push(spread);
            names.push("spread_bps".to_string());
        }

        // Depth imbalance at different levels
        for levels in [1, 5, 10] {
            let imb = ob.depth_imbalance(levels);
            features.push(imb);
            names.push(format!("depth_imbalance_{}", levels));
        }

        // Total depth ratio
        let bid_depth = ob.total_bid_depth();
        let ask_depth = ob.total_ask_depth();
        let depth_ratio = if ask_depth > 0.0 {
            bid_depth / ask_depth
        } else {
            1.0
        };
        features.push((depth_ratio - 1.0).min(2.0).max(-2.0));
        names.push("depth_ratio".to_string());

        // Log total depth
        features.push((bid_depth + ask_depth + 1.0).ln());
        names.push("log_total_depth".to_string());

        Self {
            features: Array1::from_vec(features),
            feature_names: names,
        }
    }

    /// Combine multiple feature sets
    pub fn combine(feature_sets: &[NodeFeatures]) -> Self {
        let mut all_features = Vec::new();
        let mut all_names = Vec::new();

        for fs in feature_sets {
            all_features.extend(fs.features.iter().cloned());
            all_names.extend(fs.feature_names.clone());
        }

        Self {
            features: Array1::from_vec(all_features),
            feature_names: all_names,
        }
    }

    /// Get feature dimension
    pub fn dim(&self) -> usize {
        self.features.len()
    }

    /// Pad or truncate to target dimension
    pub fn to_dim(&self, target_dim: usize) -> Array1<f64> {
        let mut result = Array1::zeros(target_dim);
        for i in 0..target_dim.min(self.features.len()) {
            result[i] = self.features[i];
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ticker_features() {
        let ticker = Ticker {
            symbol: "BTCUSDT".to_string(),
            last_price: 45000.0,
            high_24h: 46000.0,
            low_24h: 44000.0,
            price_change_24h: 0.02,
            volume_24h: 1000.0,
            turnover_24h: 45000000.0,
            bid_price: 44990.0,
            ask_price: 45010.0,
        };

        let features = NodeFeatures::from_ticker(&ticker);
        assert!(features.dim() > 0);
    }

    #[test]
    fn test_kline_features() {
        let klines: Vec<Kline> = (1..=50)
            .map(|i| Kline {
                start_time: i * 60000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
                turnover: 100000.0,
            })
            .collect();

        let features = NodeFeatures::from_klines(&klines);
        assert!(features.dim() > 0);
    }
}
