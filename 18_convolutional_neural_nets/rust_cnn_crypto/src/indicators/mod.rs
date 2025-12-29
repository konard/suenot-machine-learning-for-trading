//! # Технические индикаторы
//!
//! Реализация популярных технических индикаторов для анализа криптовалют:
//! RSI, MACD, Bollinger Bands, ATR, OBV и другие.

use serde::{Deserialize, Serialize};

/// Конфигурация индикаторов
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorConfig {
    /// Период RSI
    pub rsi_period: usize,
    /// Быстрый период MACD
    pub macd_fast: usize,
    /// Медленный период MACD
    pub macd_slow: usize,
    /// Сигнальный период MACD
    pub macd_signal: usize,
    /// Период Bollinger Bands
    pub bb_period: usize,
    /// Множитель стандартного отклонения для BB
    pub bb_std: f64,
    /// Период ATR
    pub atr_period: usize,
    /// Быстрая EMA
    pub ema_fast: usize,
    /// Медленная EMA
    pub ema_slow: usize,
}

impl Default for IndicatorConfig {
    fn default() -> Self {
        Self {
            rsi_period: 14,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal: 9,
            bb_period: 20,
            bb_std: 2.0,
            atr_period: 14,
            ema_fast: 12,
            ema_slow: 26,
        }
    }
}

/// Калькулятор технических индикаторов
pub struct TechnicalIndicators {
    config: IndicatorConfig,
}

impl TechnicalIndicators {
    /// Создание с конфигурацией
    pub fn new(config: &IndicatorConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Создание с конфигурацией по умолчанию
    pub fn default_config() -> Self {
        Self::new(&IndicatorConfig::default())
    }

    // ==================== Moving Averages ====================

    /// Simple Moving Average (SMA)
    pub fn sma(&self, data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period || period == 0 {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; period - 1];
        let mut sum: f64 = data[..period].iter().sum();
        result.push(sum / period as f64);

        for i in period..data.len() {
            sum = sum - data[i - period] + data[i];
            result.push(sum / period as f64);
        }

        result
    }

    /// Exponential Moving Average (EMA)
    pub fn ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        if data.is_empty() || period == 0 {
            return vec![f64::NAN; data.len()];
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut result = vec![f64::NAN; data.len()];

        // Первое значение EMA = SMA
        if data.len() >= period {
            let sma: f64 = data[..period].iter().sum::<f64>() / period as f64;
            result[period - 1] = sma;

            for i in period..data.len() {
                result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
            }
        }

        result
    }

    // ==================== Momentum Indicators ====================

    /// Relative Strength Index (RSI)
    pub fn rsi(&self, close: &[f64]) -> Vec<f64> {
        let period = self.config.rsi_period;
        let n = close.len();

        if n < period + 1 {
            return vec![50.0; n]; // Нейтральное значение
        }

        // Вычисляем изменения цены
        let mut gains = vec![0.0; n];
        let mut losses = vec![0.0; n];

        for i in 1..n {
            let change = close[i] - close[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        // Первое среднее
        let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

        let mut result = vec![50.0; period];

        // Вычисляем RSI
        for i in period..n {
            if i > period {
                avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
                avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
            }

            let rsi = if avg_loss == 0.0 {
                100.0
            } else {
                let rs = avg_gain / avg_loss;
                100.0 - (100.0 / (1.0 + rs))
            };

            result.push(rsi);
        }

        result
    }

    /// MACD (Moving Average Convergence Divergence)
    /// Возвращает (macd_line, signal_line, histogram)
    pub fn macd(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let fast_ema = self.ema(close, self.config.macd_fast);
        let slow_ema = self.ema(close, self.config.macd_slow);

        // MACD Line = Fast EMA - Slow EMA
        let macd_line: Vec<f64> = fast_ema
            .iter()
            .zip(slow_ema.iter())
            .map(|(f, s)| {
                if f.is_nan() || s.is_nan() {
                    0.0
                } else {
                    f - s
                }
            })
            .collect();

        // Signal Line = EMA of MACD Line
        let signal_line = self.ema(&macd_line, self.config.macd_signal);

        // Histogram = MACD Line - Signal Line
        let histogram: Vec<f64> = macd_line
            .iter()
            .zip(signal_line.iter())
            .map(|(m, s)| if s.is_nan() { 0.0 } else { m - s })
            .collect();

        (macd_line, signal_line, histogram)
    }

    // ==================== Volatility Indicators ====================

    /// Bollinger Bands
    /// Возвращает (upper_band, middle_band, lower_band)
    pub fn bollinger_bands(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let period = self.config.bb_period;
        let std_mult = self.config.bb_std;
        let n = close.len();

        let middle = self.sma(close, period);
        let mut upper = vec![f64::NAN; n];
        let mut lower = vec![f64::NAN; n];

        for i in (period - 1)..n {
            let slice = &close[i + 1 - period..=i];
            let mean = middle[i];

            // Стандартное отклонение
            let variance: f64 =
                slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();

            upper[i] = mean + std_mult * std;
            lower[i] = mean - std_mult * std;
        }

        (upper, middle, lower)
    }

    /// Average True Range (ATR)
    pub fn atr(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let period = self.config.atr_period;
        let n = high.len();

        if n < 2 {
            return vec![0.0; n];
        }

        // True Range
        let mut tr = vec![high[0] - low[0]];
        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr.push(hl.max(hc).max(lc));
        }

        // ATR = EMA of TR
        self.ema(&tr, period)
            .into_iter()
            .map(|x| if x.is_nan() { 0.0 } else { x })
            .collect()
    }

    // ==================== Volume Indicators ====================

    /// On Balance Volume (OBV)
    pub fn obv(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n == 0 {
            return Vec::new();
        }

        let mut obv = vec![volume[0]];

        for i in 1..n {
            let change = close[i] - close[i - 1];
            let prev_obv = obv[i - 1];

            let current_obv = if change > 0.0 {
                prev_obv + volume[i]
            } else if change < 0.0 {
                prev_obv - volume[i]
            } else {
                prev_obv
            };

            obv.push(current_obv);
        }

        obv
    }

    /// Volume Weighted Average Price (VWAP)
    pub fn vwap(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut cumulative_volume = 0.0;
        let mut cumulative_pv = 0.0;
        let mut vwap = Vec::with_capacity(n);

        for i in 0..n {
            let typical_price = (high[i] + low[i] + close[i]) / 3.0;
            cumulative_pv += typical_price * volume[i];
            cumulative_volume += volume[i];

            if cumulative_volume > 0.0 {
                vwap.push(cumulative_pv / cumulative_volume);
            } else {
                vwap.push(close[i]);
            }
        }

        vwap
    }

    // ==================== Trend Indicators ====================

    /// Average Directional Index (ADX)
    pub fn adx(&self, high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
        let n = high.len();
        if n < period + 1 {
            return vec![25.0; n]; // Нейтральное значение
        }

        // Вычисляем +DM и -DM
        let mut plus_dm = vec![0.0];
        let mut minus_dm = vec![0.0];

        for i in 1..n {
            let up_move = high[i] - high[i - 1];
            let down_move = low[i - 1] - low[i];

            if up_move > down_move && up_move > 0.0 {
                plus_dm.push(up_move);
            } else {
                plus_dm.push(0.0);
            }

            if down_move > up_move && down_move > 0.0 {
                minus_dm.push(down_move);
            } else {
                minus_dm.push(0.0);
            }
        }

        let atr = self.atr(high, low, close);
        let smoothed_plus_dm = self.ema(&plus_dm, period);
        let smoothed_minus_dm = self.ema(&minus_dm, period);

        // Вычисляем +DI и -DI
        let mut dx = vec![0.0; n];
        for i in 0..n {
            if atr[i] > 0.0 && !smoothed_plus_dm[i].is_nan() && !smoothed_minus_dm[i].is_nan() {
                let plus_di = (smoothed_plus_dm[i] / atr[i]) * 100.0;
                let minus_di = (smoothed_minus_dm[i] / atr[i]) * 100.0;

                let di_sum = plus_di + minus_di;
                if di_sum > 0.0 {
                    dx[i] = ((plus_di - minus_di).abs() / di_sum) * 100.0;
                }
            }
        }

        // ADX = EMA of DX
        self.ema(&dx, period)
            .into_iter()
            .map(|x| if x.is_nan() { 25.0 } else { x })
            .collect()
    }

    // ==================== Utility Functions ====================

    /// Вычисление доходности
    pub fn returns(&self, close: &[f64]) -> Vec<f64> {
        if close.is_empty() {
            return Vec::new();
        }

        let mut returns = vec![0.0];
        for i in 1..close.len() {
            if close[i - 1] != 0.0 {
                returns.push((close[i] - close[i - 1]) / close[i - 1]);
            } else {
                returns.push(0.0);
            }
        }
        returns
    }

    /// Вычисление логарифмической доходности
    pub fn log_returns(&self, close: &[f64]) -> Vec<f64> {
        if close.is_empty() {
            return Vec::new();
        }

        let mut returns = vec![0.0];
        for i in 1..close.len() {
            if close[i - 1] > 0.0 && close[i] > 0.0 {
                returns.push((close[i] / close[i - 1]).ln());
            } else {
                returns.push(0.0);
            }
        }
        returns
    }

    /// Скользящее стандартное отклонение
    pub fn rolling_std(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; period - 1];

        for i in (period - 1)..n {
            let slice = &data[i + 1 - period..=i];
            let mean: f64 = slice.iter().sum::<f64>() / period as f64;
            let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / period as f64;
            result.push(variance.sqrt());
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let indicators = TechnicalIndicators::default_config();
        let sma = indicators.sma(&data, 3);

        assert!(sma[0].is_nan());
        assert!(sma[1].is_nan());
        assert!((sma[2] - 2.0).abs() < 0.001);
        assert!((sma[3] - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_rsi() {
        let close: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64).sin() * 10.0).collect();
        let indicators = TechnicalIndicators::default_config();
        let rsi = indicators.rsi(&close);

        assert_eq!(rsi.len(), close.len());
        // RSI должен быть в диапазоне [0, 100]
        for r in &rsi {
            assert!(*r >= 0.0 && *r <= 100.0);
        }
    }

    #[test]
    fn test_bollinger_bands() {
        let close: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let indicators = TechnicalIndicators::default_config();
        let (upper, middle, lower) = indicators.bollinger_bands(&close);

        assert_eq!(upper.len(), close.len());

        // Проверяем порядок: lower < middle < upper
        for i in 19..close.len() {
            assert!(lower[i] < middle[i]);
            assert!(middle[i] < upper[i]);
        }
    }
}
