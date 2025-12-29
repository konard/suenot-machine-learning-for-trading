//! Feature Engineering для TFT
//!
//! Модуль для расчета технических индикаторов и создания признаков.

use crate::api::Kline;
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

/// Технические индикаторы
#[derive(Debug, Clone)]
pub struct TechnicalIndicators {
    /// Simple Moving Average
    pub sma: Vec<f64>,
    /// Exponential Moving Average
    pub ema: Vec<f64>,
    /// Relative Strength Index
    pub rsi: Vec<f64>,
    /// MACD line
    pub macd: Vec<f64>,
    /// MACD signal line
    pub macd_signal: Vec<f64>,
    /// MACD histogram
    pub macd_histogram: Vec<f64>,
    /// Bollinger Bands upper
    pub bb_upper: Vec<f64>,
    /// Bollinger Bands middle
    pub bb_middle: Vec<f64>,
    /// Bollinger Bands lower
    pub bb_lower: Vec<f64>,
    /// Average True Range
    pub atr: Vec<f64>,
    /// Volume Moving Average
    pub volume_sma: Vec<f64>,
    /// Returns (процентное изменение)
    pub returns: Vec<f64>,
    /// Log returns
    pub log_returns: Vec<f64>,
    /// Realized volatility
    pub volatility: Vec<f64>,
}

/// Набор признаков для TFT
#[derive(Debug, Clone)]
pub struct Features {
    /// Названия признаков
    pub names: Vec<String>,
    /// Матрица признаков (samples x features)
    pub values: Array2<f64>,
    /// Статические признаки (не меняются во времени)
    pub static_features: HashMap<String, f64>,
    /// Временные метки
    pub timestamps: Vec<i64>,
}

impl Features {
    /// Создает пустой набор признаков
    pub fn new() -> Self {
        Self {
            names: Vec::new(),
            values: Array2::zeros((0, 0)),
            static_features: HashMap::new(),
            timestamps: Vec::new(),
        }
    }

    /// Возвращает количество временных шагов
    pub fn len(&self) -> usize {
        self.values.nrows()
    }

    /// Проверяет, пустой ли набор
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Возвращает количество признаков
    pub fn num_features(&self) -> usize {
        self.values.ncols()
    }

    /// Нормализует признаки (z-score)
    pub fn normalize(&mut self) {
        for col_idx in 0..self.values.ncols() {
            let column = self.values.column(col_idx);
            let mean = column.mean().unwrap_or(0.0);
            let std = column.std(0.0);

            if std > 1e-10 {
                for row_idx in 0..self.values.nrows() {
                    self.values[[row_idx, col_idx]] =
                        (self.values[[row_idx, col_idx]] - mean) / std;
                }
            }
        }
    }

    /// Возвращает срез данных
    pub fn slice(&self, start: usize, end: usize) -> Features {
        let end = end.min(self.len());
        let start = start.min(end);

        Features {
            names: self.names.clone(),
            values: self.values.slice(ndarray::s![start..end, ..]).to_owned(),
            static_features: self.static_features.clone(),
            timestamps: self.timestamps[start..end].to_vec(),
        }
    }
}

impl Default for Features {
    fn default() -> Self {
        Self::new()
    }
}

/// Экстрактор признаков из свечей
pub struct FeatureExtractor {
    /// Период для SMA
    pub sma_period: usize,
    /// Период для RSI
    pub rsi_period: usize,
    /// Период для Bollinger Bands
    pub bb_period: usize,
    /// Стандартные отклонения для BB
    pub bb_std: f64,
    /// Период для ATR
    pub atr_period: usize,
    /// Период для волатильности
    pub volatility_period: usize,
    /// Быстрый период для MACD
    pub macd_fast: usize,
    /// Медленный период для MACD
    pub macd_slow: usize,
    /// Сигнальный период для MACD
    pub macd_signal: usize,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self {
            sma_period: 20,
            rsi_period: 14,
            bb_period: 20,
            bb_std: 2.0,
            atr_period: 14,
            volatility_period: 20,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal: 9,
        }
    }
}

impl FeatureExtractor {
    /// Создает новый экстрактор с настройками по умолчанию
    pub fn new() -> Self {
        Self::default()
    }

    /// Создает экстрактор с кастомными периодами
    pub fn with_periods(sma: usize, rsi: usize, bb: usize, atr: usize) -> Self {
        Self {
            sma_period: sma,
            rsi_period: rsi,
            bb_period: bb,
            atr_period: atr,
            ..Default::default()
        }
    }

    /// Извлекает все признаки из свечей
    pub fn extract(&self, klines: &[Kline]) -> Features {
        if klines.is_empty() {
            return Features::new();
        }

        let n = klines.len();
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let lows: Vec<f64> = klines.iter().map(|k| k.low).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();
        let timestamps: Vec<i64> = klines.iter().map(|k| k.open_time).collect();

        // Рассчитываем индикаторы
        let indicators = self.calculate_indicators(klines);

        // Временные признаки
        let (hour_sin, hour_cos) = self.time_features_hour(&timestamps);
        let (dow_sin, dow_cos) = self.time_features_dow(&timestamps);

        // Собираем все признаки в матрицу
        let mut feature_names = vec![
            "close".to_string(),
            "high".to_string(),
            "low".to_string(),
            "volume".to_string(),
            "returns".to_string(),
            "log_returns".to_string(),
            "volatility".to_string(),
            "sma".to_string(),
            "ema".to_string(),
            "rsi".to_string(),
            "macd".to_string(),
            "macd_signal".to_string(),
            "macd_histogram".to_string(),
            "bb_upper".to_string(),
            "bb_middle".to_string(),
            "bb_lower".to_string(),
            "bb_position".to_string(),
            "atr".to_string(),
            "volume_sma".to_string(),
            "volume_ratio".to_string(),
            "hour_sin".to_string(),
            "hour_cos".to_string(),
            "dow_sin".to_string(),
            "dow_cos".to_string(),
        ];

        let num_features = feature_names.len();
        let mut values = Array2::zeros((n, num_features));

        // Заполняем матрицу
        for i in 0..n {
            values[[i, 0]] = closes[i];
            values[[i, 1]] = highs[i];
            values[[i, 2]] = lows[i];
            values[[i, 3]] = volumes[i];
            values[[i, 4]] = indicators.returns[i];
            values[[i, 5]] = indicators.log_returns[i];
            values[[i, 6]] = indicators.volatility[i];
            values[[i, 7]] = indicators.sma[i];
            values[[i, 8]] = indicators.ema[i];
            values[[i, 9]] = indicators.rsi[i];
            values[[i, 10]] = indicators.macd[i];
            values[[i, 11]] = indicators.macd_signal[i];
            values[[i, 12]] = indicators.macd_histogram[i];
            values[[i, 13]] = indicators.bb_upper[i];
            values[[i, 14]] = indicators.bb_middle[i];
            values[[i, 15]] = indicators.bb_lower[i];

            // BB position: где цена относительно полос (-1 до 1)
            let bb_range = indicators.bb_upper[i] - indicators.bb_lower[i];
            let bb_position = if bb_range > 0.0 {
                2.0 * (closes[i] - indicators.bb_lower[i]) / bb_range - 1.0
            } else {
                0.0
            };
            values[[i, 16]] = bb_position.clamp(-2.0, 2.0);

            values[[i, 17]] = indicators.atr[i];
            values[[i, 18]] = indicators.volume_sma[i];

            // Volume ratio: текущий объем / средний
            let volume_ratio = if indicators.volume_sma[i] > 0.0 {
                volumes[i] / indicators.volume_sma[i]
            } else {
                1.0
            };
            values[[i, 19]] = volume_ratio.clamp(0.0, 10.0);

            values[[i, 20]] = hour_sin[i];
            values[[i, 21]] = hour_cos[i];
            values[[i, 22]] = dow_sin[i];
            values[[i, 23]] = dow_cos[i];
        }

        Features {
            names: feature_names,
            values,
            static_features: HashMap::new(),
            timestamps,
        }
    }

    /// Рассчитывает технические индикаторы
    pub fn calculate_indicators(&self, klines: &[Kline]) -> TechnicalIndicators {
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let lows: Vec<f64> = klines.iter().map(|k| k.low).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        TechnicalIndicators {
            sma: self.sma(&closes, self.sma_period),
            ema: self.ema(&closes, self.sma_period),
            rsi: self.rsi(&closes, self.rsi_period),
            macd: self.macd(&closes),
            macd_signal: self.macd_signal(&closes),
            macd_histogram: self.macd_histogram(&closes),
            bb_upper: self.bollinger_upper(&closes),
            bb_middle: self.sma(&closes, self.bb_period),
            bb_lower: self.bollinger_lower(&closes),
            atr: self.atr(&highs, &lows, &closes),
            volume_sma: self.sma(&volumes, self.sma_period),
            returns: self.returns(&closes),
            log_returns: self.log_returns(&closes),
            volatility: self.volatility(&closes),
        }
    }

    /// Simple Moving Average
    pub fn sma(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.0; n];

        for i in 0..n {
            if i < period - 1 {
                // Используем доступные данные
                let sum: f64 = data[0..=i].iter().sum();
                result[i] = sum / (i + 1) as f64;
            } else {
                let sum: f64 = data[i + 1 - period..=i].iter().sum();
                result[i] = sum / period as f64;
            }
        }

        result
    }

    /// Exponential Moving Average
    pub fn ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.0; n];

        if n == 0 {
            return result;
        }

        let alpha = 2.0 / (period + 1) as f64;
        result[0] = data[0];

        for i in 1..n {
            result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Relative Strength Index
    pub fn rsi(&self, closes: &[f64], period: usize) -> Vec<f64> {
        let n = closes.len();
        let mut result = vec![50.0; n]; // Default to neutral

        if n < 2 {
            return result;
        }

        let mut gains = vec![0.0; n];
        let mut losses = vec![0.0; n];

        for i in 1..n {
            let change = closes[i] - closes[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        let avg_gains = self.ema(&gains, period);
        let avg_losses = self.ema(&losses, period);

        for i in 0..n {
            if avg_losses[i] > 1e-10 {
                let rs = avg_gains[i] / avg_losses[i];
                result[i] = 100.0 - (100.0 / (1.0 + rs));
            } else if avg_gains[i] > 1e-10 {
                result[i] = 100.0;
            }
        }

        result
    }

    /// MACD line
    fn macd(&self, closes: &[f64]) -> Vec<f64> {
        let fast_ema = self.ema(closes, self.macd_fast);
        let slow_ema = self.ema(closes, self.macd_slow);

        fast_ema
            .iter()
            .zip(slow_ema.iter())
            .map(|(f, s)| f - s)
            .collect()
    }

    /// MACD signal line
    fn macd_signal(&self, closes: &[f64]) -> Vec<f64> {
        let macd = self.macd(closes);
        self.ema(&macd, self.macd_signal)
    }

    /// MACD histogram
    fn macd_histogram(&self, closes: &[f64]) -> Vec<f64> {
        let macd = self.macd(closes);
        let signal = self.macd_signal(closes);

        macd.iter()
            .zip(signal.iter())
            .map(|(m, s)| m - s)
            .collect()
    }

    /// Bollinger Bands upper
    fn bollinger_upper(&self, closes: &[f64]) -> Vec<f64> {
        let middle = self.sma(closes, self.bb_period);
        let std = self.rolling_std(closes, self.bb_period);

        middle
            .iter()
            .zip(std.iter())
            .map(|(m, s)| m + self.bb_std * s)
            .collect()
    }

    /// Bollinger Bands lower
    fn bollinger_lower(&self, closes: &[f64]) -> Vec<f64> {
        let middle = self.sma(closes, self.bb_period);
        let std = self.rolling_std(closes, self.bb_period);

        middle
            .iter()
            .zip(std.iter())
            .map(|(m, s)| m - self.bb_std * s)
            .collect()
    }

    /// Rolling standard deviation
    fn rolling_std(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.0; n];

        for i in 0..n {
            let window_start = if i >= period - 1 { i + 1 - period } else { 0 };
            let window = &data[window_start..=i];

            let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let variance: f64 =
                window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
            result[i] = variance.sqrt();
        }

        result
    }

    /// Average True Range
    fn atr(&self, highs: &[f64], lows: &[f64], closes: &[f64]) -> Vec<f64> {
        let n = highs.len();
        let mut tr = vec![0.0; n];

        // True Range
        tr[0] = highs[0] - lows[0];
        for i in 1..n {
            let hl = highs[i] - lows[i];
            let hc = (highs[i] - closes[i - 1]).abs();
            let lc = (lows[i] - closes[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
        }

        // EMA of True Range
        self.ema(&tr, self.atr_period)
    }

    /// Returns (процентное изменение)
    fn returns(&self, closes: &[f64]) -> Vec<f64> {
        let n = closes.len();
        let mut result = vec![0.0; n];

        for i in 1..n {
            if closes[i - 1] > 0.0 {
                result[i] = (closes[i] - closes[i - 1]) / closes[i - 1];
            }
        }

        result
    }

    /// Log returns
    fn log_returns(&self, closes: &[f64]) -> Vec<f64> {
        let n = closes.len();
        let mut result = vec![0.0; n];

        for i in 1..n {
            if closes[i - 1] > 0.0 && closes[i] > 0.0 {
                result[i] = (closes[i] / closes[i - 1]).ln();
            }
        }

        result
    }

    /// Realized volatility (rolling std of returns)
    fn volatility(&self, closes: &[f64]) -> Vec<f64> {
        let returns = self.log_returns(closes);
        self.rolling_std(&returns, self.volatility_period)
    }

    /// Временные признаки: час дня (sin/cos)
    fn time_features_hour(&self, timestamps: &[i64]) -> (Vec<f64>, Vec<f64>) {
        use chrono::{TimeZone, Utc};

        let mut sin_vals = Vec::with_capacity(timestamps.len());
        let mut cos_vals = Vec::with_capacity(timestamps.len());

        for &ts in timestamps {
            let dt = Utc.timestamp_millis_opt(ts).unwrap();
            let hour = dt.format("%H").to_string().parse::<f64>().unwrap_or(0.0);

            let angle = 2.0 * std::f64::consts::PI * hour / 24.0;
            sin_vals.push(angle.sin());
            cos_vals.push(angle.cos());
        }

        (sin_vals, cos_vals)
    }

    /// Временные признаки: день недели (sin/cos)
    fn time_features_dow(&self, timestamps: &[i64]) -> (Vec<f64>, Vec<f64>) {
        use chrono::{Datelike, TimeZone, Utc};

        let mut sin_vals = Vec::with_capacity(timestamps.len());
        let mut cos_vals = Vec::with_capacity(timestamps.len());

        for &ts in timestamps {
            let dt = Utc.timestamp_millis_opt(ts).unwrap();
            let dow = dt.weekday().num_days_from_monday() as f64;

            let angle = 2.0 * std::f64::consts::PI * dow / 7.0;
            sin_vals.push(angle.sin());
            cos_vals.push(angle.cos());
        }

        (sin_vals, cos_vals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let extractor = FeatureExtractor::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = extractor.sma(&data, 3);

        assert!((sma[2] - 2.0).abs() < 1e-10);
        assert!((sma[3] - 3.0).abs() < 1e-10);
        assert!((sma[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rsi() {
        let extractor = FeatureExtractor::new();
        let data = vec![44.0, 44.5, 43.5, 44.5, 44.0, 43.0, 44.5, 44.0, 44.5, 43.5];
        let rsi = extractor.rsi(&data, 5);

        // RSI должен быть в диапазоне 0-100
        for val in rsi {
            assert!(val >= 0.0 && val <= 100.0);
        }
    }

    #[test]
    fn test_returns() {
        let extractor = FeatureExtractor::new();
        let data = vec![100.0, 110.0, 99.0];
        let returns = extractor.returns(&data);

        assert!((returns[0] - 0.0).abs() < 1e-10);
        assert!((returns[1] - 0.1).abs() < 1e-10);
        assert!((returns[2] - (-0.1)).abs() < 1e-10);
    }
}
