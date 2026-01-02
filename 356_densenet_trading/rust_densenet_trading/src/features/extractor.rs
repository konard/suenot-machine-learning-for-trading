//! Feature Extractor
//!
//! Extracts comprehensive features from market data for DenseNet input.

use ndarray::Array2;

use crate::data::candle::CandleSeries;
use crate::data::orderbook::OrderBook;
use super::indicators::TechnicalIndicators;
use super::normalizer::FeatureNormalizer;

/// Feature extractor for cryptocurrency data
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// Whether to normalize features
    pub normalize: bool,

    /// Feature normalizer
    pub normalizer: Option<FeatureNormalizer>,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new() -> Self {
        Self {
            normalize: true,
            normalizer: None,
        }
    }

    /// Create without normalization
    pub fn without_normalization() -> Self {
        Self {
            normalize: false,
            normalizer: None,
        }
    }

    /// Extract features from candle series
    ///
    /// Returns: [num_candles, num_features] array
    pub fn extract_from_candles(&self, candles: &CandleSeries) -> Array2<f64> {
        let n = candles.len();
        if n == 0 {
            return Array2::zeros((0, 0));
        }

        let closes = candles.closes();
        let opens = candles.opens();
        let highs = candles.highs();
        let lows = candles.lows();
        let volumes = candles.volumes();

        // Calculate all features
        let mut features: Vec<Vec<f64>> = Vec::new();

        // 1. Price features
        features.push(TechnicalIndicators::log_returns(&closes));

        // 2. Range features
        let range: Vec<f64> = highs.iter().zip(lows.iter()).zip(closes.iter())
            .map(|((&h, &l), &c)| (h - l) / c)
            .collect();
        features.push(range);

        // 3. Body features
        let body: Vec<f64> = opens.iter().zip(closes.iter())
            .map(|(&o, &c)| (c - o) / o)
            .collect();
        features.push(body);

        // 4. Close position in range
        let close_pos: Vec<f64> = highs.iter().zip(lows.iter()).zip(closes.iter())
            .map(|((&h, &l), &c)| {
                let range = h - l;
                if range > 0.0 { (c - l) / range } else { 0.5 }
            })
            .collect();
        features.push(close_pos);

        // 5. Volume features
        let vol_ma = TechnicalIndicators::sma(&volumes, 20);
        let vol_ratio: Vec<f64> = volumes.iter().zip(vol_ma.iter())
            .map(|(&v, &ma)| if ma > 0.0 && !ma.is_nan() { v / ma } else { 1.0 })
            .collect();
        features.push(vol_ratio);

        // 6. Moving averages
        for period in [5, 10, 20, 50] {
            let sma = TechnicalIndicators::sma(&closes, period);
            let ma_ratio: Vec<f64> = closes.iter().zip(sma.iter())
                .map(|(&c, &ma)| if ma > 0.0 && !ma.is_nan() { c / ma - 1.0 } else { 0.0 })
                .collect();
            features.push(ma_ratio);
        }

        // 7. EMA ratios
        for period in [12, 26] {
            let ema = TechnicalIndicators::ema(&closes, period);
            let ema_ratio: Vec<f64> = closes.iter().zip(ema.iter())
                .map(|(&c, &e)| if e > 0.0 && !e.is_nan() { c / e - 1.0 } else { 0.0 })
                .collect();
            features.push(ema_ratio);
        }

        // 8. RSI
        let rsi = TechnicalIndicators::rsi(&closes, 14);
        let rsi_norm: Vec<f64> = rsi.iter().map(|&r| (r - 50.0) / 50.0).collect();
        features.push(rsi_norm);

        // 9. MACD features
        let (macd, signal, histogram) = TechnicalIndicators::macd(&closes, 12, 26, 9);
        let macd_norm: Vec<f64> = macd.iter().zip(closes.iter())
            .map(|(&m, &c)| if c > 0.0 { m / c * 100.0 } else { 0.0 })
            .collect();
        features.push(macd_norm);

        let hist_norm: Vec<f64> = histogram.iter().zip(closes.iter())
            .map(|(&h, &c)| if c > 0.0 { h / c * 100.0 } else { 0.0 })
            .collect();
        features.push(hist_norm);

        // 10. Bollinger Bands position
        let (bb_upper, bb_middle, bb_lower) = TechnicalIndicators::bollinger_bands(&closes, 20, 2.0);
        let bb_pos: Vec<f64> = closes.iter().enumerate()
            .map(|(i, &c)| {
                let range = bb_upper[i] - bb_lower[i];
                if range > 0.0 && !range.is_nan() {
                    (c - bb_lower[i]) / range
                } else {
                    0.5
                }
            })
            .collect();
        features.push(bb_pos);

        // 11. ATR ratio
        let atr = TechnicalIndicators::atr(&highs, &lows, &closes, 14);
        let atr_ratio: Vec<f64> = atr.iter().zip(closes.iter())
            .map(|(&a, &c)| if c > 0.0 && !a.is_nan() { a / c } else { 0.0 })
            .collect();
        features.push(atr_ratio);

        // 12. Stochastic oscillator
        let (stoch_k, stoch_d) = TechnicalIndicators::stochastic(&highs, &lows, &closes, 14, 3);
        let stoch_k_norm: Vec<f64> = stoch_k.iter().map(|&k| (k - 50.0) / 50.0).collect();
        let stoch_d_norm: Vec<f64> = stoch_d.iter().map(|&d| (d - 50.0) / 50.0).collect();
        features.push(stoch_k_norm);
        features.push(stoch_d_norm);

        // 13. Williams %R
        let williams = TechnicalIndicators::williams_r(&highs, &lows, &closes, 14);
        let williams_norm: Vec<f64> = williams.iter().map(|&w| (w + 50.0) / 50.0).collect();
        features.push(williams_norm);

        // 14. ROC (momentum)
        for period in [5, 10, 20] {
            let roc = TechnicalIndicators::roc(&closes, period);
            let roc_norm: Vec<f64> = roc.iter().map(|&r| r / 10.0).collect(); // Scale down
            features.push(roc_norm);
        }

        // 15. Realized volatility
        let vol = TechnicalIndicators::realized_volatility(&closes, 20);
        features.push(vol);

        // 16. OBV trend
        let obv = TechnicalIndicators::obv(&closes, &volumes);
        let obv_ma = TechnicalIndicators::sma(&obv, 20);
        let obv_trend: Vec<f64> = obv.iter().zip(obv_ma.iter())
            .map(|(&o, &ma)| {
                if ma != 0.0 && !ma.is_nan() && !o.is_nan() {
                    (o / ma - 1.0).max(-1.0).min(1.0)
                } else {
                    0.0
                }
            })
            .collect();
        features.push(obv_trend);

        // 17. CCI
        let cci = TechnicalIndicators::cci(&highs, &lows, &closes, 20);
        let cci_norm: Vec<f64> = cci.iter().map(|&c| (c / 200.0).max(-1.0).min(1.0)).collect();
        features.push(cci_norm);

        // Convert to 2D array
        let num_features = features.len();
        let mut result = Array2::zeros((n, num_features));

        for (f_idx, feature) in features.iter().enumerate() {
            for (i, &val) in feature.iter().enumerate() {
                result[[i, f_idx]] = if val.is_nan() { 0.0 } else { val };
            }
        }

        result
    }

    /// Extract order book features
    pub fn extract_orderbook_features(&self, orderbook: &OrderBook) -> Vec<f64> {
        let mut features = Vec::new();

        // Basic features
        features.push(orderbook.spread_percent().unwrap_or(0.0));
        features.push(orderbook.order_imbalance());

        // Depth imbalance at different levels
        for depth in [1, 5, 10, 20] {
            features.push(orderbook.order_imbalance_at_depth(depth));
        }

        // Volume features
        let bid_vol = orderbook.total_bid_volume();
        let ask_vol = orderbook.total_ask_volume();
        let total_vol = bid_vol + ask_vol;

        features.push(if total_vol > 0.0 { bid_vol / total_vol } else { 0.5 });

        // Weighted mid price deviation
        if let (Some(mid), Some(wmid)) = (orderbook.mid_price(), orderbook.weighted_mid_price()) {
            features.push((wmid - mid) / mid * 10000.0); // In bps
        } else {
            features.push(0.0);
        }

        features
    }

    /// Get feature names
    pub fn feature_names(&self) -> Vec<&'static str> {
        vec![
            "log_return",
            "range_ratio",
            "body_ratio",
            "close_position",
            "volume_ratio",
            "sma5_ratio",
            "sma10_ratio",
            "sma20_ratio",
            "sma50_ratio",
            "ema12_ratio",
            "ema26_ratio",
            "rsi_norm",
            "macd_norm",
            "macd_hist_norm",
            "bb_position",
            "atr_ratio",
            "stoch_k",
            "stoch_d",
            "williams_r",
            "roc5",
            "roc10",
            "roc20",
            "realized_vol",
            "obv_trend",
            "cci_norm",
        ]
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.feature_names().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::candle::Candle;

    fn create_sample_candles(n: usize) -> CandleSeries {
        let mut candles = CandleSeries::new();
        let mut price = 100.0;

        for i in 0..n {
            let change = ((i as f64 * 0.1).sin()) * 2.0;
            price += change;

            let open = price;
            let close = price + change * 0.5;
            let high = open.max(close) + 1.0;
            let low = open.min(close) - 1.0;
            let volume = 1000.0 + (i as f64 * 10.0).cos() * 500.0;

            candles.push(Candle::new(
                (i as u64) * 60000,
                open,
                high,
                low.max(0.1),
                close,
                volume.max(100.0),
                volume * price,
            ));
        }

        candles
    }

    #[test]
    fn test_feature_extraction() {
        let candles = create_sample_candles(100);
        let extractor = FeatureExtractor::new();
        let features = extractor.extract_from_candles(&candles);

        assert_eq!(features.dim().0, 100);
        assert_eq!(features.dim().1, extractor.num_features());
    }

    #[test]
    fn test_feature_names() {
        let extractor = FeatureExtractor::new();
        let names = extractor.feature_names();

        assert!(!names.is_empty());
        assert!(names.contains(&"rsi_norm"));
        assert!(names.contains(&"macd_norm"));
    }

    #[test]
    fn test_no_nans() {
        let candles = create_sample_candles(200);
        let extractor = FeatureExtractor::new();
        let features = extractor.extract_from_candles(&candles);

        // Check that NaNs are replaced with 0
        for val in features.iter() {
            assert!(!val.is_nan(), "Found NaN in features");
        }
    }
}
