//! Feature engineering for market data

use ndarray::Array1;
use super::{Kline, OrderBook, Ticker};
use crate::graph::AssetFeatures;

/// Feature builder for market data
#[derive(Debug, Clone)]
pub struct FeatureBuilder {
    /// Number of historical klines to use
    lookback: usize,
}

impl FeatureBuilder {
    /// Create a new feature builder
    pub fn new(lookback: usize) -> Self {
        Self { lookback }
    }

    /// Build asset features from klines and ticker
    pub fn build_asset_features(
        &self,
        klines: &[Kline],
        ticker: &Ticker,
        order_book: Option<&OrderBook>,
    ) -> AssetFeatures {
        let volatility = self.compute_volatility(klines);
        let (return_1h, return_4h, return_24h) = self.compute_returns(klines);

        let (spread, imbalance) = match order_book {
            Some(ob) => (
                ob.spread_bps().unwrap_or(0.0) / 10000.0,
                ob.imbalance(10),
            ),
            None => (ticker.spread_bps() / 10000.0, 0.0),
        };

        AssetFeatures {
            price: ticker.last_price,
            volume_24h: ticker.volume_24h,
            volatility,
            market_cap: 0.0,
            funding_rate: 0.0,
            open_interest: 0.0,
            return_1h,
            return_4h,
            return_24h,
            spread,
            imbalance,
            timestamp: ticker.timestamp,
        }
    }

    /// Compute historical volatility from klines
    fn compute_volatility(&self, klines: &[Kline]) -> f64 {
        if klines.len() < 2 {
            return 0.02;  // Default
        }

        let returns: Vec<f64> = klines
            .windows(2)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect();

        if returns.is_empty() {
            return 0.02;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        variance.sqrt()
    }

    /// Compute returns at different time horizons
    fn compute_returns(&self, klines: &[Kline]) -> (f64, f64, f64) {
        if klines.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let current_price = klines.last().map(|k| k.close).unwrap_or(1.0);

        let return_1h = if klines.len() > 1 {
            let past_price = klines.get(klines.len().saturating_sub(2)).map(|k| k.close).unwrap_or(current_price);
            (current_price / past_price - 1.0)
        } else {
            0.0
        };

        let return_4h = if klines.len() > 4 {
            let past_price = klines.get(klines.len().saturating_sub(5)).map(|k| k.close).unwrap_or(current_price);
            (current_price / past_price - 1.0)
        } else {
            0.0
        };

        let return_24h = if klines.len() > 24 {
            let past_price = klines.get(klines.len().saturating_sub(25)).map(|k| k.close).unwrap_or(current_price);
            (current_price / past_price - 1.0)
        } else {
            0.0
        };

        (return_1h, return_4h, return_24h)
    }

    /// Compute technical indicators
    pub fn compute_indicators(&self, klines: &[Kline]) -> TechnicalIndicators {
        TechnicalIndicators {
            sma_20: self.sma(klines, 20),
            sma_50: self.sma(klines, 50),
            ema_12: self.ema(klines, 12),
            ema_26: self.ema(klines, 26),
            rsi_14: self.rsi(klines, 14),
            macd: self.macd(klines),
            atr_14: self.atr(klines, 14),
            bollinger_upper: 0.0,
            bollinger_lower: 0.0,
        }
    }

    /// Simple Moving Average
    fn sma(&self, klines: &[Kline], period: usize) -> f64 {
        if klines.len() < period {
            return 0.0;
        }
        let sum: f64 = klines.iter().rev().take(period).map(|k| k.close).sum();
        sum / period as f64
    }

    /// Exponential Moving Average
    fn ema(&self, klines: &[Kline], period: usize) -> f64 {
        if klines.is_empty() {
            return 0.0;
        }

        let alpha = 2.0 / (period as f64 + 1.0);
        let mut ema = klines[0].close;

        for kline in klines.iter().skip(1) {
            ema = alpha * kline.close + (1.0 - alpha) * ema;
        }

        ema
    }

    /// Relative Strength Index
    fn rsi(&self, klines: &[Kline], period: usize) -> f64 {
        if klines.len() < period + 1 {
            return 50.0;  // Neutral
        }

        let changes: Vec<f64> = klines
            .windows(2)
            .map(|w| w[1].close - w[0].close)
            .collect();

        let gains: Vec<f64> = changes.iter().map(|c| if *c > 0.0 { *c } else { 0.0 }).collect();
        let losses: Vec<f64> = changes.iter().map(|c| if *c < 0.0 { -c } else { 0.0 }).collect();

        let avg_gain: f64 = gains.iter().rev().take(period).sum::<f64>() / period as f64;
        let avg_loss: f64 = losses.iter().rev().take(period).sum::<f64>() / period as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    /// MACD
    fn macd(&self, klines: &[Kline]) -> f64 {
        self.ema(klines, 12) - self.ema(klines, 26)
    }

    /// Average True Range
    fn atr(&self, klines: &[Kline], period: usize) -> f64 {
        if klines.len() < 2 {
            return 0.0;
        }

        let trs: Vec<f64> = klines
            .windows(2)
            .map(|w| {
                let high_low = w[1].high - w[1].low;
                let high_close = (w[1].high - w[0].close).abs();
                let low_close = (w[1].low - w[0].close).abs();
                high_low.max(high_close).max(low_close)
            })
            .collect();

        if trs.is_empty() {
            return 0.0;
        }

        trs.iter().rev().take(period).sum::<f64>() / period.min(trs.len()) as f64
    }
}

impl Default for FeatureBuilder {
    fn default() -> Self {
        Self::new(100)
    }
}

/// Technical indicators container
#[derive(Debug, Clone)]
pub struct TechnicalIndicators {
    pub sma_20: f64,
    pub sma_50: f64,
    pub ema_12: f64,
    pub ema_26: f64,
    pub rsi_14: f64,
    pub macd: f64,
    pub atr_14: f64,
    pub bollinger_upper: f64,
    pub bollinger_lower: f64,
}

impl TechnicalIndicators {
    /// Convert to feature vector
    pub fn to_vector(&self) -> Array1<f64> {
        Array1::from(vec![
            self.sma_20,
            self.sma_50,
            self.ema_12,
            self.ema_26,
            self.rsi_14 / 100.0,  // Normalize to 0-1
            self.macd,
            self.atr_14,
            self.bollinger_upper,
            self.bollinger_lower,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize, base_price: f64) -> Vec<Kline> {
        (0..n)
            .map(|i| {
                let price = base_price * (1.0 + (i as f64 * 0.001));
                Kline {
                    timestamp: i as u64 * 3600000,
                    open: price,
                    high: price * 1.01,
                    low: price * 0.99,
                    close: price,
                    volume: 1000.0,
                    turnover: price * 1000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_volatility() {
        let builder = FeatureBuilder::new(100);
        let klines = create_test_klines(50, 100.0);
        let vol = builder.compute_volatility(&klines);
        assert!(vol > 0.0);
        assert!(vol < 1.0);
    }

    #[test]
    fn test_sma() {
        let builder = FeatureBuilder::new(100);
        let klines = create_test_klines(50, 100.0);
        let sma = builder.sma(&klines, 20);
        assert!(sma > 0.0);
    }

    #[test]
    fn test_rsi() {
        let builder = FeatureBuilder::new(100);
        let klines = create_test_klines(50, 100.0);
        let rsi = builder.rsi(&klines, 14);
        assert!(rsi >= 0.0 && rsi <= 100.0);
    }

    #[test]
    fn test_indicators() {
        let builder = FeatureBuilder::new(100);
        let klines = create_test_klines(50, 100.0);
        let indicators = builder.compute_indicators(&klines);
        assert!(indicators.sma_20 > 0.0);
        assert!(indicators.rsi_14 >= 0.0);
    }
}
