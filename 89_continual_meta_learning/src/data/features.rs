//! Feature engineering for trading data.
//!
//! This module provides functions to compute technical indicators
//! and create feature vectors for machine learning models.

use crate::data::Kline;
use crate::MarketRegime;

/// Configuration for feature generation.
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Window size for moving averages.
    pub sma_window: usize,
    /// Short window for EMA.
    pub ema_short: usize,
    /// Long window for EMA.
    pub ema_long: usize,
    /// RSI period.
    pub rsi_period: usize,
    /// Volatility window.
    pub vol_window: usize,
    /// Momentum period.
    pub momentum_period: usize,
    /// Whether to normalize features.
    pub normalize: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            sma_window: 20,
            ema_short: 12,
            ema_long: 26,
            rsi_period: 14,
            vol_window: 20,
            momentum_period: 10,
            normalize: true,
        }
    }
}

/// Trading features computed from kline data.
#[derive(Debug, Clone)]
pub struct TradingFeatures {
    /// Configuration used.
    pub config: FeatureConfig,
    /// Returns (price change percentage).
    pub returns: Vec<f64>,
    /// Simple Moving Average.
    pub sma: Vec<f64>,
    /// Exponential Moving Average (short).
    pub ema_short: Vec<f64>,
    /// Exponential Moving Average (long).
    pub ema_long: Vec<f64>,
    /// Relative Strength Index.
    pub rsi: Vec<f64>,
    /// Volatility (rolling standard deviation of returns).
    pub volatility: Vec<f64>,
    /// Momentum.
    pub momentum: Vec<f64>,
    /// Price relative to SMA (price/SMA - 1).
    pub price_to_sma: Vec<f64>,
    /// MACD (EMA short - EMA long).
    pub macd: Vec<f64>,
    /// Detected market regimes.
    pub regimes: Vec<MarketRegime>,
    /// Close prices (for reference).
    pub close_prices: Vec<f64>,
}

impl TradingFeatures {
    /// Compute features from kline data.
    pub fn from_klines(klines: &[Kline], config: FeatureConfig) -> Self {
        let close_prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let returns = compute_returns(&close_prices);
        let sma = compute_sma(&close_prices, config.sma_window);
        let ema_short = compute_ema(&close_prices, config.ema_short);
        let ema_long = compute_ema(&close_prices, config.ema_long);
        let rsi = compute_rsi(&close_prices, config.rsi_period);
        let volatility = compute_volatility(&returns, config.vol_window);
        let momentum = compute_momentum(&close_prices, config.momentum_period);
        let price_to_sma = compute_price_to_sma(&close_prices, &sma);
        let macd = compute_macd(&ema_short, &ema_long);
        let regimes = detect_regimes(&returns, &volatility);

        Self {
            config,
            returns,
            sma,
            ema_short,
            ema_long,
            rsi,
            volatility,
            momentum,
            price_to_sma,
            macd,
            regimes,
            close_prices,
        }
    }

    /// Get feature vector at index.
    pub fn get_features(&self, idx: usize) -> Option<Vec<f64>> {
        if idx >= self.returns.len() {
            return None;
        }

        let features = vec![
            self.returns.get(idx).copied().unwrap_or(0.0),
            self.sma.get(idx).copied().unwrap_or(0.0),
            self.ema_short.get(idx).copied().unwrap_or(0.0),
            self.ema_long.get(idx).copied().unwrap_or(0.0),
            self.rsi.get(idx).copied().unwrap_or(50.0),
            self.volatility.get(idx).copied().unwrap_or(0.0),
            self.momentum.get(idx).copied().unwrap_or(0.0),
            self.price_to_sma.get(idx).copied().unwrap_or(0.0),
            self.macd.get(idx).copied().unwrap_or(0.0),
        ];

        if self.config.normalize {
            Some(normalize_features(&features))
        } else {
            Some(features)
        }
    }

    /// Get all feature vectors.
    pub fn get_all_features(&self) -> Vec<Vec<f64>> {
        (0..self.returns.len())
            .filter_map(|i| self.get_features(i))
            .collect()
    }

    /// Get features for a specific regime.
    pub fn get_regime_features(&self, regime: MarketRegime) -> Vec<(usize, Vec<f64>)> {
        self.regimes
            .iter()
            .enumerate()
            .filter(|(_, r)| **r == regime)
            .filter_map(|(i, _)| self.get_features(i).map(|f| (i, f)))
            .collect()
    }

    /// Get regime at index.
    pub fn get_regime(&self, idx: usize) -> Option<MarketRegime> {
        self.regimes.get(idx).copied()
    }

    /// Get number of samples.
    pub fn len(&self) -> usize {
        self.returns.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.returns.is_empty()
    }

    /// Get feature dimension.
    pub fn feature_dim(&self) -> usize {
        9 // Number of features
    }
}

/// Compute returns from prices.
pub fn compute_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![0.0; prices.len()];
    }

    let mut returns = vec![0.0]; // First return is 0
    for i in 1..prices.len() {
        if prices[i - 1] > 0.0 {
            returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
        } else {
            returns.push(0.0);
        }
    }
    returns
}

/// Compute Simple Moving Average.
pub fn compute_sma(prices: &[f64], window: usize) -> Vec<f64> {
    if prices.is_empty() || window == 0 {
        return vec![0.0; prices.len()];
    }

    let mut sma = Vec::with_capacity(prices.len());

    for i in 0..prices.len() {
        if i < window - 1 {
            sma.push(prices[..=i].iter().sum::<f64>() / (i + 1) as f64);
        } else {
            let sum: f64 = prices[i + 1 - window..=i].iter().sum();
            sma.push(sum / window as f64);
        }
    }

    sma
}

/// Compute Exponential Moving Average.
pub fn compute_ema(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.is_empty() || period == 0 {
        return vec![0.0; prices.len()];
    }

    let multiplier = 2.0 / (period + 1) as f64;
    let mut ema = Vec::with_capacity(prices.len());

    // Initialize with SMA
    let initial_sma = if prices.len() >= period {
        prices[..period].iter().sum::<f64>() / period as f64
    } else {
        prices.iter().sum::<f64>() / prices.len() as f64
    };

    ema.push(initial_sma);

    for i in 1..prices.len() {
        let prev_ema = ema[i - 1];
        let new_ema = (prices[i] - prev_ema) * multiplier + prev_ema;
        ema.push(new_ema);
    }

    ema
}

/// Compute Relative Strength Index.
pub fn compute_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < 2 || period == 0 {
        return vec![50.0; prices.len()];
    }

    let mut rsi = vec![50.0]; // Initial RSI is neutral
    let mut gains = Vec::new();
    let mut losses = Vec::new();

    for i in 1..prices.len() {
        let change = prices[i] - prices[i - 1];
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { -change } else { 0.0 };

        gains.push(gain);
        losses.push(loss);

        let avg_gain: f64;
        let avg_loss: f64;

        if i < period {
            avg_gain = gains.iter().sum::<f64>() / i as f64;
            avg_loss = losses.iter().sum::<f64>() / i as f64;
        } else {
            let start = i - period;
            avg_gain = gains[start..].iter().sum::<f64>() / period as f64;
            avg_loss = losses[start..].iter().sum::<f64>() / period as f64;
        }

        let rs = if avg_loss > 0.0 {
            avg_gain / avg_loss
        } else {
            100.0
        };

        let rsi_value = 100.0 - (100.0 / (1.0 + rs));
        rsi.push(rsi_value);
    }

    rsi
}

/// Compute rolling volatility (standard deviation of returns).
pub fn compute_volatility(returns: &[f64], window: usize) -> Vec<f64> {
    if returns.is_empty() || window == 0 {
        return vec![0.0; returns.len()];
    }

    let mut volatility = Vec::with_capacity(returns.len());

    for i in 0..returns.len() {
        let start = if i >= window { i + 1 - window } else { 0 };
        let window_returns = &returns[start..=i];
        let n = window_returns.len() as f64;

        if n < 2.0 {
            volatility.push(0.0);
            continue;
        }

        let mean = window_returns.iter().sum::<f64>() / n;
        let variance = window_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        volatility.push(variance.sqrt());
    }

    volatility
}

/// Compute momentum (rate of change).
pub fn compute_momentum(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.is_empty() || period == 0 {
        return vec![0.0; prices.len()];
    }

    let mut momentum = Vec::with_capacity(prices.len());

    for i in 0..prices.len() {
        if i < period {
            momentum.push(0.0);
        } else {
            let past_price = prices[i - period];
            if past_price > 0.0 {
                momentum.push((prices[i] - past_price) / past_price);
            } else {
                momentum.push(0.0);
            }
        }
    }

    momentum
}

/// Compute price relative to SMA.
pub fn compute_price_to_sma(prices: &[f64], sma: &[f64]) -> Vec<f64> {
    prices
        .iter()
        .zip(sma.iter())
        .map(|(p, s)| if *s > 0.0 { p / s - 1.0 } else { 0.0 })
        .collect()
}

/// Compute MACD (EMA short - EMA long).
pub fn compute_macd(ema_short: &[f64], ema_long: &[f64]) -> Vec<f64> {
    ema_short
        .iter()
        .zip(ema_long.iter())
        .map(|(s, l)| s - l)
        .collect()
}

/// Detect market regime based on returns and volatility.
pub fn detect_regimes(returns: &[f64], volatility: &[f64]) -> Vec<MarketRegime> {
    if returns.is_empty() {
        return Vec::new();
    }

    // Compute thresholds
    let returns_mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let vol_mean = volatility.iter().sum::<f64>() / volatility.len() as f64;

    // Compute rolling averages for regime detection (20-period)
    let window = 20;
    let mut regimes = Vec::with_capacity(returns.len());

    for i in 0..returns.len() {
        let start = if i >= window { i + 1 - window } else { 0 };
        let window_returns = &returns[start..=i];
        let window_vol = &volatility[start..=i];

        let avg_return = window_returns.iter().sum::<f64>() / window_returns.len() as f64;
        let avg_vol = window_vol.iter().sum::<f64>() / window_vol.len() as f64;

        let regime = if avg_vol > vol_mean * 1.5 {
            MarketRegime::HighVolatility
        } else if avg_vol < vol_mean * 0.5 {
            MarketRegime::LowVolatility
        } else if avg_return > returns_mean + 0.001 {
            MarketRegime::Bull
        } else if avg_return < returns_mean - 0.001 {
            MarketRegime::Bear
        } else {
            MarketRegime::Sideways
        };

        regimes.push(regime);
    }

    regimes
}

/// Normalize features to [-1, 1] range.
pub fn normalize_features(features: &[f64]) -> Vec<f64> {
    if features.is_empty() {
        return Vec::new();
    }

    let max_abs = features.iter().map(|f| f.abs()).fold(0.0, f64::max);

    if max_abs > 0.0 {
        features.iter().map(|f| f / max_abs).collect()
    } else {
        features.to_vec()
    }
}

/// Z-score normalization.
pub fn z_score_normalize(features: &[f64]) -> Vec<f64> {
    if features.is_empty() {
        return Vec::new();
    }

    let mean = features.iter().sum::<f64>() / features.len() as f64;
    let variance = features.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / features.len() as f64;
    let std = variance.sqrt();

    if std > 0.0 {
        features.iter().map(|f| (f - mean) / std).collect()
    } else {
        vec![0.0; features.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines() -> Vec<Kline> {
        let prices = vec![
            100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 110.0,
            112.0, 111.0, 113.0, 115.0, 114.0, 116.0, 118.0, 117.0, 120.0, 122.0,
        ];

        prices
            .iter()
            .enumerate()
            .map(|(i, &p)| Kline {
                start_time: (i as i64) * 3600000,
                open: p - 0.5,
                high: p + 1.0,
                low: p - 1.0,
                close: p,
                volume: 1000.0,
                turnover: p * 1000.0,
            })
            .collect()
    }

    #[test]
    fn test_compute_returns() {
        let prices = vec![100.0, 105.0, 102.0, 108.0];
        let returns = compute_returns(&prices);

        assert_eq!(returns.len(), 4);
        assert_eq!(returns[0], 0.0);
        assert!((returns[1] - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_compute_sma() {
        let prices = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let sma = compute_sma(&prices, 3);

        assert_eq!(sma.len(), 5);
        assert!((sma[2] - 20.0).abs() < 0.001); // (10+20+30)/3
        assert!((sma[4] - 40.0).abs() < 0.001); // (30+40+50)/3
    }

    #[test]
    fn test_compute_rsi() {
        let prices = vec![44.0, 44.5, 44.0, 43.5, 44.0, 44.5, 45.0, 45.5, 45.0, 44.5];
        let rsi = compute_rsi(&prices, 5);

        assert_eq!(rsi.len(), 10);
        // RSI should be between 0 and 100
        for r in &rsi {
            assert!(*r >= 0.0 && *r <= 100.0);
        }
    }

    #[test]
    fn test_trading_features() {
        let klines = create_test_klines();
        let config = FeatureConfig::default();
        let features = TradingFeatures::from_klines(&klines, config);

        assert_eq!(features.len(), 20);
        assert_eq!(features.feature_dim(), 9);

        let feat = features.get_features(10).unwrap();
        assert_eq!(feat.len(), 9);
    }

    #[test]
    fn test_detect_regimes() {
        let returns = vec![0.02, 0.03, 0.01, -0.01, -0.02, -0.03, 0.01, 0.02];
        let volatility = vec![0.01, 0.01, 0.01, 0.02, 0.03, 0.04, 0.02, 0.01];

        let regimes = detect_regimes(&returns, &volatility);

        assert_eq!(regimes.len(), 8);
    }

    #[test]
    fn test_normalize() {
        let features = vec![1.0, -2.0, 3.0, -4.0];
        let normalized = normalize_features(&features);

        assert_eq!(normalized.len(), 4);
        // Max absolute value is 4, so features should be divided by 4
        assert!((normalized[3] - (-1.0)).abs() < 0.001);
    }
}
