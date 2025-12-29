//! Market data management and technical indicators.

use crate::data::Candle;
use ndarray::Array1;
use std::collections::VecDeque;

/// Market data container with technical indicators
#[derive(Debug, Clone)]
pub struct MarketData {
    /// Raw candle data
    candles: Vec<Candle>,
    /// Calculated returns
    returns: Vec<f64>,
    /// Simple Moving Average (short period)
    sma_short: Vec<f64>,
    /// Simple Moving Average (long period)
    sma_long: Vec<f64>,
    /// Relative Strength Index
    rsi: Vec<f64>,
    /// MACD line
    macd: Vec<f64>,
    /// MACD signal line
    macd_signal: Vec<f64>,
    /// Bollinger Bands (upper, middle, lower)
    bollinger: Vec<(f64, f64, f64)>,
    /// Average True Range (volatility)
    atr: Vec<f64>,
}

impl MarketData {
    /// Create MarketData from a vector of candles
    pub fn from_candles(candles: Vec<Candle>) -> Self {
        let mut data = Self {
            candles,
            returns: Vec::new(),
            sma_short: Vec::new(),
            sma_long: Vec::new(),
            rsi: Vec::new(),
            macd: Vec::new(),
            macd_signal: Vec::new(),
            bollinger: Vec::new(),
            atr: Vec::new(),
        };
        data.calculate_indicators();
        data
    }

    /// Get the number of data points
    pub fn len(&self) -> usize {
        self.candles.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.candles.is_empty()
    }

    /// Get candle at index
    pub fn get_candle(&self, index: usize) -> Option<&Candle> {
        self.candles.get(index)
    }

    /// Get closing prices
    pub fn close_prices(&self) -> Vec<f64> {
        self.candles.iter().map(|c| c.close).collect()
    }

    /// Get returns
    pub fn returns(&self) -> &[f64] {
        &self.returns
    }

    /// Calculate all technical indicators
    fn calculate_indicators(&mut self) {
        self.calculate_returns();
        self.calculate_sma(10, 50);
        self.calculate_rsi(14);
        self.calculate_macd(12, 26, 9);
        self.calculate_bollinger(20, 2.0);
        self.calculate_atr(14);
    }

    /// Calculate returns
    fn calculate_returns(&mut self) {
        self.returns = vec![0.0]; // First return is 0
        for i in 1..self.candles.len() {
            let prev_close = self.candles[i - 1].close;
            let curr_close = self.candles[i].close;
            if prev_close > 0.0 {
                self.returns.push((curr_close - prev_close) / prev_close);
            } else {
                self.returns.push(0.0);
            }
        }
    }

    /// Calculate Simple Moving Average
    fn calculate_sma(&mut self, short_period: usize, long_period: usize) {
        let closes = self.close_prices();

        self.sma_short = Self::sma(&closes, short_period);
        self.sma_long = Self::sma(&closes, long_period);
    }

    /// Helper to calculate SMA
    fn sma(data: &[f64], period: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(data.len());
        let mut window: VecDeque<f64> = VecDeque::with_capacity(period);
        let mut sum = 0.0;

        for &value in data {
            window.push_back(value);
            sum += value;

            if window.len() > period {
                sum -= window.pop_front().unwrap();
            }

            if window.len() >= period {
                result.push(sum / period as f64);
            } else {
                result.push(f64::NAN);
            }
        }

        result
    }

    /// Calculate Exponential Moving Average
    fn ema(data: &[f64], period: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(data.len());
        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema_prev = data[0];

        for (i, &value) in data.iter().enumerate() {
            if i == 0 {
                result.push(value);
            } else {
                let ema_curr = (value - ema_prev) * multiplier + ema_prev;
                result.push(ema_curr);
                ema_prev = ema_curr;
            }
        }

        result
    }

    /// Calculate RSI (Relative Strength Index)
    fn calculate_rsi(&mut self, period: usize) {
        let closes = self.close_prices();
        let mut gains = Vec::with_capacity(closes.len());
        let mut losses = Vec::with_capacity(closes.len());

        gains.push(0.0);
        losses.push(0.0);

        for i in 1..closes.len() {
            let change = closes[i] - closes[i - 1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        let avg_gains = Self::ema(&gains, period);
        let avg_losses = Self::ema(&losses, period);

        self.rsi = avg_gains
            .iter()
            .zip(avg_losses.iter())
            .map(|(&g, &l)| {
                if l == 0.0 {
                    100.0
                } else {
                    100.0 - (100.0 / (1.0 + g / l))
                }
            })
            .collect();
    }

    /// Calculate MACD
    fn calculate_macd(&mut self, fast: usize, slow: usize, signal: usize) {
        let closes = self.close_prices();
        let ema_fast = Self::ema(&closes, fast);
        let ema_slow = Self::ema(&closes, slow);

        self.macd = ema_fast
            .iter()
            .zip(ema_slow.iter())
            .map(|(&f, &s)| f - s)
            .collect();

        self.macd_signal = Self::ema(&self.macd, signal);
    }

    /// Calculate Bollinger Bands
    fn calculate_bollinger(&mut self, period: usize, std_dev: f64) {
        let closes = self.close_prices();
        let sma = Self::sma(&closes, period);

        self.bollinger = Vec::with_capacity(closes.len());

        for i in 0..closes.len() {
            if i < period - 1 {
                self.bollinger.push((f64::NAN, f64::NAN, f64::NAN));
            } else {
                let window: Vec<f64> = closes[i + 1 - period..=i].to_vec();
                let mean = sma[i];
                let variance: f64 =
                    window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
                let std = variance.sqrt();

                let upper = mean + std_dev * std;
                let lower = mean - std_dev * std;
                self.bollinger.push((upper, mean, lower));
            }
        }
    }

    /// Calculate Average True Range
    fn calculate_atr(&mut self, period: usize) {
        let mut tr = Vec::with_capacity(self.candles.len());

        for i in 0..self.candles.len() {
            if i == 0 {
                tr.push(self.candles[i].high - self.candles[i].low);
            } else {
                let high = self.candles[i].high;
                let low = self.candles[i].low;
                let prev_close = self.candles[i - 1].close;

                let tr1 = high - low;
                let tr2 = (high - prev_close).abs();
                let tr3 = (low - prev_close).abs();

                tr.push(tr1.max(tr2).max(tr3));
            }
        }

        self.atr = Self::ema(&tr, period);
    }

    /// Get state vector for a given index
    /// Returns: [returns, sma_ratio, rsi_norm, macd_norm, bb_position, atr_norm, volume_norm]
    pub fn get_state(&self, index: usize) -> Option<Array1<f64>> {
        if index >= self.len() {
            return None;
        }

        let candle = &self.candles[index];
        let close = candle.close;

        // Normalize returns (clip to [-0.1, 0.1] and scale to [-1, 1])
        let ret = (self.returns[index] * 10.0).clamp(-1.0, 1.0);

        // SMA ratio (price relative to SMA)
        let sma_ratio = if self.sma_short[index].is_nan() || self.sma_long[index].is_nan() {
            0.0
        } else {
            ((close / self.sma_short[index]) - 1.0).clamp(-0.1, 0.1) * 10.0
        };

        // RSI normalized to [-1, 1]
        let rsi_norm = (self.rsi[index] / 50.0) - 1.0;

        // MACD normalized
        let macd_norm = if index > 0 && close > 0.0 {
            ((self.macd[index] - self.macd_signal[index]) / close * 100.0).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Bollinger Band position (-1 = at lower, 0 = at middle, 1 = at upper)
        let bb_position = if self.bollinger[index].0.is_nan() {
            0.0
        } else {
            let (upper, middle, lower) = self.bollinger[index];
            let range = upper - lower;
            if range > 0.0 {
                ((close - middle) / (range / 2.0)).clamp(-1.0, 1.0)
            } else {
                0.0
            }
        };

        // ATR normalized by close price
        let atr_norm = if close > 0.0 {
            (self.atr[index] / close * 100.0).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Volume change (compared to previous)
        let volume_norm = if index > 0 {
            let prev_vol = self.candles[index - 1].volume;
            if prev_vol > 0.0 {
                ((candle.volume / prev_vol) - 1.0).clamp(-1.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        Some(Array1::from_vec(vec![
            ret,
            sma_ratio,
            rsi_norm,
            macd_norm,
            bb_position,
            atr_norm,
            volume_norm,
        ]))
    }

    /// Get the number of features in the state vector
    pub fn state_size() -> usize {
        7
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        let mut candles = Vec::with_capacity(n);
        let mut price = 100.0;

        for i in 0..n {
            let change = (i as f64).sin() * 5.0;
            price += change;
            candles.push(Candle::new(
                Utc::now(),
                "TEST".to_string(),
                price - 1.0,
                price + 2.0,
                price - 2.0,
                price,
                1000.0 + (i as f64) * 10.0,
                price * 1000.0,
            ));
        }

        candles
    }

    #[test]
    fn test_market_data_creation() {
        let candles = create_test_candles(100);
        let data = MarketData::from_candles(candles);

        assert_eq!(data.len(), 100);
        assert_eq!(data.returns().len(), 100);
    }

    #[test]
    fn test_get_state() {
        let candles = create_test_candles(100);
        let data = MarketData::from_candles(candles);

        let state = data.get_state(50);
        assert!(state.is_some());

        let state = state.unwrap();
        assert_eq!(state.len(), MarketData::state_size());

        // All values should be normalized (mostly between -1 and 1)
        for &val in state.iter() {
            assert!(val.is_finite());
        }
    }
}
