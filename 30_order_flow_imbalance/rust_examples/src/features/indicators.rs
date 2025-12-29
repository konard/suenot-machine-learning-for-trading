//! # Technical Indicators
//!
//! Additional technical indicators for feature engineering.

use std::collections::VecDeque;

/// Technical indicators calculator
#[derive(Debug)]
pub struct TechnicalIndicators {
    /// Price history
    prices: VecDeque<f64>,
    /// Volume history
    volumes: VecDeque<f64>,
    /// Maximum history size
    max_size: usize,
}

impl TechnicalIndicators {
    /// Create new calculator
    pub fn new(max_size: usize) -> Self {
        Self {
            prices: VecDeque::with_capacity(max_size),
            volumes: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    /// Add price and volume
    pub fn update(&mut self, price: f64, volume: f64) {
        self.prices.push_back(price);
        self.volumes.push_back(volume);

        if self.prices.len() > self.max_size {
            self.prices.pop_front();
            self.volumes.pop_front();
        }
    }

    /// Simple Moving Average
    pub fn sma(&self, period: usize) -> Option<f64> {
        if self.prices.len() < period {
            return None;
        }

        let sum: f64 = self.prices.iter().rev().take(period).sum();
        Some(sum / period as f64)
    }

    /// Exponential Moving Average
    pub fn ema(&self, period: usize) -> Option<f64> {
        if self.prices.len() < period {
            return None;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let prices: Vec<f64> = self.prices.iter().rev().take(period).cloned().collect();

        let mut ema = prices.last()?;
        for price in prices.iter().rev().skip(1) {
            ema = (price - ema) * multiplier + ema;
        }

        Some(*ema)
    }

    /// Relative Strength Index
    pub fn rsi(&self, period: usize) -> Option<f64> {
        if self.prices.len() < period + 1 {
            return None;
        }

        let prices: Vec<f64> = self.prices.iter().rev().take(period + 1).cloned().collect();

        let mut gains = 0.0;
        let mut losses = 0.0;
        let mut count = 0;

        for window in prices.windows(2) {
            let change = window[0] - window[1]; // Note: reversed order
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
            count += 1;
        }

        if count == 0 {
            return Some(50.0);
        }

        let avg_gain = gains / count as f64;
        let avg_loss = losses / count as f64;

        if avg_loss == 0.0 {
            return Some(100.0);
        }

        let rs = avg_gain / avg_loss;
        Some(100.0 - 100.0 / (1.0 + rs))
    }

    /// Bollinger Bands
    pub fn bollinger_bands(&self, period: usize, std_dev: f64) -> Option<(f64, f64, f64)> {
        let sma = self.sma(period)?;
        let prices: Vec<f64> = self.prices.iter().rev().take(period).cloned().collect();

        let variance: f64 = prices.iter().map(|p| (p - sma).powi(2)).sum::<f64>() / period as f64;
        let std = variance.sqrt();

        let upper = sma + std_dev * std;
        let lower = sma - std_dev * std;

        Some((upper, sma, lower))
    }

    /// Average True Range (simplified without high/low)
    pub fn atr(&self, period: usize) -> Option<f64> {
        if self.prices.len() < period + 1 {
            return None;
        }

        let prices: Vec<f64> = self.prices.iter().rev().take(period + 1).cloned().collect();

        let true_ranges: Vec<f64> = prices
            .windows(2)
            .map(|w| (w[0] - w[1]).abs())
            .collect();

        Some(true_ranges.iter().sum::<f64>() / true_ranges.len() as f64)
    }

    /// Volume Weighted Average Price
    pub fn vwap(&self, period: usize) -> Option<f64> {
        if self.prices.len() < period || self.volumes.len() < period {
            return None;
        }

        let prices: Vec<f64> = self.prices.iter().rev().take(period).cloned().collect();
        let volumes: Vec<f64> = self.volumes.iter().rev().take(period).cloned().collect();

        let total_pv: f64 = prices.iter().zip(volumes.iter()).map(|(p, v)| p * v).sum();
        let total_v: f64 = volumes.iter().sum();

        if total_v > 0.0 {
            Some(total_pv / total_v)
        } else {
            None
        }
    }

    /// Price momentum (return over period)
    pub fn momentum(&self, period: usize) -> Option<f64> {
        if self.prices.len() < period {
            return None;
        }

        let current = self.prices.back()?;
        let past = self.prices.get(self.prices.len() - period)?;

        Some((current - past) / past * 100.0)
    }

    /// Rate of Change
    pub fn roc(&self, period: usize) -> Option<f64> {
        self.momentum(period)
    }

    /// Standard deviation of returns
    pub fn volatility(&self, period: usize) -> Option<f64> {
        if self.prices.len() < period + 1 {
            return None;
        }

        let prices: Vec<f64> = self.prices.iter().rev().take(period + 1).cloned().collect();

        let returns: Vec<f64> = prices.windows(2).map(|w| (w[0] - w[1]) / w[1]).collect();

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;

        Some(variance.sqrt())
    }

    /// Latest price
    pub fn current_price(&self) -> Option<f64> {
        self.prices.back().cloned()
    }

    /// Number of data points
    pub fn len(&self) -> usize {
        self.prices.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.prices.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let mut ind = TechnicalIndicators::new(100);

        for i in 1..=10 {
            ind.update(i as f64, 100.0);
        }

        let sma = ind.sma(5).unwrap();
        // Last 5: 6, 7, 8, 9, 10 -> avg = 8
        assert!((sma - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_rsi() {
        let mut ind = TechnicalIndicators::new(100);

        // Uptrend
        for i in 1..=20 {
            ind.update(100.0 + i as f64, 100.0);
        }

        let rsi = ind.rsi(14).unwrap();
        // Should be high (above 50) in uptrend
        assert!(rsi > 50.0);
    }

    #[test]
    fn test_bollinger() {
        let mut ind = TechnicalIndicators::new(100);

        for _ in 0..20 {
            ind.update(100.0, 100.0);
        }

        let (upper, middle, lower) = ind.bollinger_bands(20, 2.0).unwrap();
        // With constant prices, bands should be tight
        assert!((upper - middle).abs() < 0.001);
        assert!((lower - middle).abs() < 0.001);
    }
}
