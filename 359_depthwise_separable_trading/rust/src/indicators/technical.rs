//! Technical Analysis Indicators
//!
//! Comprehensive collection of technical indicators for trading.

use ndarray::Array1;

use super::{ema, sma, std_dev};

/// Collection of technical indicators
#[derive(Debug, Clone)]
pub struct TechnicalIndicators {
    /// RSI period
    pub rsi_period: usize,
    /// MACD fast period
    pub macd_fast: usize,
    /// MACD slow period
    pub macd_slow: usize,
    /// MACD signal period
    pub macd_signal: usize,
    /// Bollinger Bands period
    pub bb_period: usize,
    /// Bollinger Bands standard deviations
    pub bb_std: f64,
    /// ATR period
    pub atr_period: usize,
}

impl Default for TechnicalIndicators {
    fn default() -> Self {
        Self {
            rsi_period: 14,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal: 9,
            bb_period: 20,
            bb_std: 2.0,
            atr_period: 14,
        }
    }
}

impl TechnicalIndicators {
    /// Create with default parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate RSI (Relative Strength Index)
    pub fn rsi(&self, close: &Array1<f64>) -> Array1<f64> {
        let n = close.len();
        let mut result = Array1::from_elem(n, 50.0); // Default to neutral

        if n <= self.rsi_period {
            return result;
        }

        // Calculate price changes
        let mut gains = Array1::zeros(n);
        let mut losses = Array1::zeros(n);

        for i in 1..n {
            let change = close[i] - close[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        // Calculate average gains and losses using EMA
        let avg_gain = ema(&gains, self.rsi_period);
        let avg_loss = ema(&losses, self.rsi_period);

        // Calculate RSI
        for i in self.rsi_period..n {
            if avg_loss[i] == 0.0 {
                result[i] = 100.0;
            } else if avg_gain[i] == 0.0 {
                result[i] = 0.0;
            } else {
                let rs = avg_gain[i] / avg_loss[i];
                result[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }

        result
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    ///
    /// Returns (macd_line, signal_line, histogram)
    pub fn macd(&self, close: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let n = close.len();

        let fast_ema = ema(close, self.macd_fast);
        let slow_ema = ema(close, self.macd_slow);

        // MACD Line = Fast EMA - Slow EMA
        let macd_line = &fast_ema - &slow_ema;

        // Signal Line = EMA of MACD Line
        let signal_line = ema(&macd_line, self.macd_signal);

        // Histogram = MACD Line - Signal Line
        let histogram = &macd_line - &signal_line;

        (macd_line, signal_line, histogram)
    }

    /// Calculate Bollinger Bands
    ///
    /// Returns (upper_band, middle_band, lower_band)
    pub fn bollinger_bands(
        &self,
        close: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let middle = sma(close, self.bb_period);
        let std = std_dev(close, self.bb_period);

        let upper = &middle + &(&std * self.bb_std);
        let lower = &middle - &(&std * self.bb_std);

        (upper, middle, lower)
    }

    /// Calculate ATR (Average True Range)
    pub fn atr(
        &self,
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
    ) -> Array1<f64> {
        let n = high.len();
        let mut tr = Array1::zeros(n);

        // Calculate True Range
        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
        }
        tr[0] = high[0] - low[0];

        // ATR is EMA of True Range
        ema(&tr, self.atr_period)
    }

    /// Calculate Stochastic Oscillator
    ///
    /// Returns (%K, %D)
    pub fn stochastic(
        &self,
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
        k_period: usize,
        d_period: usize,
    ) -> (Array1<f64>, Array1<f64>) {
        let n = close.len();
        let mut k = Array1::zeros(n);

        for i in k_period - 1..n {
            let period_high = high
                .slice(ndarray::s![i + 1 - k_period..=i])
                .fold(f64::MIN, |a, &b| a.max(b));
            let period_low = low
                .slice(ndarray::s![i + 1 - k_period..=i])
                .fold(f64::MAX, |a, &b| a.min(b));

            if period_high != period_low {
                k[i] = (close[i] - period_low) / (period_high - period_low) * 100.0;
            } else {
                k[i] = 50.0;
            }
        }

        let d = sma(&k, d_period);

        (k, d)
    }

    /// Calculate ADX (Average Directional Index)
    pub fn adx(
        &self,
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
        period: usize,
    ) -> Array1<f64> {
        let n = high.len();
        let mut plus_dm = Array1::zeros(n);
        let mut minus_dm = Array1::zeros(n);

        // Calculate directional movement
        for i in 1..n {
            let up_move = high[i] - high[i - 1];
            let down_move = low[i - 1] - low[i];

            if up_move > down_move && up_move > 0.0 {
                plus_dm[i] = up_move;
            }
            if down_move > up_move && down_move > 0.0 {
                minus_dm[i] = down_move;
            }
        }

        let atr = self.atr(high, low, close);
        let smooth_plus_dm = ema(&plus_dm, period);
        let smooth_minus_dm = ema(&minus_dm, period);

        let mut plus_di = Array1::zeros(n);
        let mut minus_di = Array1::zeros(n);
        let mut dx = Array1::zeros(n);

        for i in 0..n {
            if atr[i] > 0.0 {
                plus_di[i] = smooth_plus_dm[i] / atr[i] * 100.0;
                minus_di[i] = smooth_minus_dm[i] / atr[i] * 100.0;
            }

            let di_sum = plus_di[i] + minus_di[i];
            if di_sum > 0.0 {
                dx[i] = (plus_di[i] - minus_di[i]).abs() / di_sum * 100.0;
            }
        }

        ema(&dx, period)
    }

    /// Calculate OBV (On-Balance Volume)
    pub fn obv(&self, close: &Array1<f64>, volume: &Array1<f64>) -> Array1<f64> {
        let n = close.len();
        let mut obv = Array1::zeros(n);

        if n > 0 {
            obv[0] = volume[0];
        }

        for i in 1..n {
            if close[i] > close[i - 1] {
                obv[i] = obv[i - 1] + volume[i];
            } else if close[i] < close[i - 1] {
                obv[i] = obv[i - 1] - volume[i];
            } else {
                obv[i] = obv[i - 1];
            }
        }

        obv
    }

    /// Calculate VWAP (Volume Weighted Average Price)
    pub fn vwap(
        &self,
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
        volume: &Array1<f64>,
    ) -> Array1<f64> {
        let n = close.len();
        let mut vwap = Array1::zeros(n);

        let mut cum_tp_vol = 0.0;
        let mut cum_vol = 0.0;

        for i in 0..n {
            let tp = (high[i] + low[i] + close[i]) / 3.0;
            cum_tp_vol += tp * volume[i];
            cum_vol += volume[i];

            if cum_vol > 0.0 {
                vwap[i] = cum_tp_vol / cum_vol;
            }
        }

        vwap
    }

    /// Calculate all indicators and return as feature matrix
    pub fn calculate_all(
        &self,
        open: &Array1<f64>,
        high: &Array1<f64>,
        low: &Array1<f64>,
        close: &Array1<f64>,
        volume: &Array1<f64>,
    ) -> Vec<(&'static str, Array1<f64>)> {
        let rsi = self.rsi(close);
        let (macd, signal, histogram) = self.macd(close);
        let (bb_upper, bb_middle, bb_lower) = self.bollinger_bands(close);
        let atr = self.atr(high, low, close);
        let (stoch_k, stoch_d) = self.stochastic(high, low, close, 14, 3);
        let adx = self.adx(high, low, close, 14);
        let obv = self.obv(close, volume);
        let vwap = self.vwap(high, low, close, volume);

        // Moving averages
        let sma_20 = sma(close, 20);
        let sma_50 = sma(close, 50);
        let ema_12 = ema(close, 12);
        let ema_26 = ema(close, 26);

        vec![
            ("rsi", rsi),
            ("macd", macd),
            ("macd_signal", signal),
            ("macd_histogram", histogram),
            ("bb_upper", bb_upper),
            ("bb_middle", bb_middle),
            ("bb_lower", bb_lower),
            ("atr", atr),
            ("stoch_k", stoch_k),
            ("stoch_d", stoch_d),
            ("adx", adx),
            ("obv", obv),
            ("vwap", vwap),
            ("sma_20", sma_20),
            ("sma_50", sma_50),
            ("ema_12", ema_12),
            ("ema_26", ema_26),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        let close = Array1::from_vec(vec![
            100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0,
            110.0, 108.0, 109.0, 111.0, 113.0, 112.0, 114.0, 116.0, 115.0, 117.0,
            118.0, 116.0, 117.0, 119.0, 121.0, 120.0, 122.0, 124.0, 123.0, 125.0,
        ]);
        let high = &close + 2.0;
        let low = &close - 2.0;
        let volume = Array1::from_elem(30, 1000.0);

        (high, low, close, volume)
    }

    #[test]
    fn test_rsi() {
        let (_, _, close, _) = create_test_data();
        let indicators = TechnicalIndicators::new();
        let rsi = indicators.rsi(&close);

        // RSI should be between 0 and 100
        for val in rsi.iter() {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_macd() {
        let (_, _, close, _) = create_test_data();
        let indicators = TechnicalIndicators::new();
        let (macd, signal, histogram) = indicators.macd(&close);

        assert_eq!(macd.len(), close.len());
        assert_eq!(signal.len(), close.len());
        assert_eq!(histogram.len(), close.len());
    }

    #[test]
    fn test_bollinger_bands() {
        let (_, _, close, _) = create_test_data();
        let indicators = TechnicalIndicators::new();
        let (upper, middle, lower) = indicators.bollinger_bands(&close);

        // Upper should be > middle > lower
        for i in indicators.bb_period..close.len() {
            assert!(upper[i] >= middle[i]);
            assert!(middle[i] >= lower[i]);
        }
    }

    #[test]
    fn test_atr() {
        let (high, low, close, _) = create_test_data();
        let indicators = TechnicalIndicators::new();
        let atr = indicators.atr(&high, &low, &close);

        // ATR should be positive
        for i in indicators.atr_period..close.len() {
            assert!(atr[i] >= 0.0);
        }
    }

    #[test]
    fn test_stochastic() {
        let (high, low, close, _) = create_test_data();
        let indicators = TechnicalIndicators::new();
        let (k, d) = indicators.stochastic(&high, &low, &close, 14, 3);

        // Stochastic should be between 0 and 100
        for val in k.iter().skip(14) {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
        for val in d.iter().skip(17) {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }
}
