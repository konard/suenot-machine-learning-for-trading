//! # Data Processor
//!
//! Модуль для предобработки данных и извлечения признаков.
//! Включает нормализацию, расчет технических индикаторов и создание датасетов.

use crate::bybit_client::Kline;
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Метод нормализации данных
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationMethod {
    /// Min-Max нормализация в диапазон [0, 1]
    MinMax,
    /// Z-score нормализация (среднее=0, std=1)
    ZScore,
    /// Нормализация по максимальному абсолютному значению
    MaxAbs,
    /// Без нормализации
    None,
}

/// Набор признаков для обучения
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Features {
    /// Матрица признаков (samples x features)
    pub data: Vec<Vec<f64>>,
    /// Названия признаков
    pub names: Vec<String>,
    /// Временные метки
    pub timestamps: Vec<i64>,
}

impl Features {
    /// Создает новый набор признаков
    pub fn new(data: Vec<Vec<f64>>, names: Vec<String>, timestamps: Vec<i64>) -> Self {
        Self {
            data,
            names,
            timestamps,
        }
    }

    /// Возвращает количество образцов
    pub fn nrows(&self) -> usize {
        self.data.len()
    }

    /// Возвращает количество признаков
    pub fn ncols(&self) -> usize {
        self.names.len()
    }

    /// Преобразует в ndarray матрицу
    pub fn to_array(&self) -> Array2<f64> {
        let rows = self.data.len();
        let cols = if rows > 0 { self.data[0].len() } else { 0 };

        let flat: Vec<f64> = self.data.iter().flatten().copied().collect();
        Array2::from_shape_vec((rows, cols), flat).unwrap_or_else(|_| Array2::zeros((0, 0)))
    }

    /// Создает из ndarray матрицы
    pub fn from_array(arr: &Array2<f64>, names: Vec<String>, timestamps: Vec<i64>) -> Self {
        let data: Vec<Vec<f64>> = arr.outer_iter().map(|row| row.to_vec()).collect();
        Self::new(data, names, timestamps)
    }
}

/// Параметры нормализации для последующего обратного преобразования
#[derive(Debug, Clone)]
pub struct NormalizationParams {
    pub method: NormalizationMethod,
    pub min_vals: Vec<f64>,
    pub max_vals: Vec<f64>,
    pub mean_vals: Vec<f64>,
    pub std_vals: Vec<f64>,
}

/// Процессор данных для подготовки к обучению
pub struct DataProcessor {
    normalization: NormalizationMethod,
    lookback_period: usize,
    norm_params: Option<NormalizationParams>,
}

impl Default for DataProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl DataProcessor {
    /// Создает новый процессор с настройками по умолчанию
    pub fn new() -> Self {
        Self {
            normalization: NormalizationMethod::MinMax,
            lookback_period: 20,
            norm_params: None,
        }
    }

    /// Устанавливает метод нормализации
    pub fn with_normalization(mut self, method: NormalizationMethod) -> Self {
        self.normalization = method;
        self
    }

    /// Устанавливает период для расчета скользящих индикаторов
    pub fn with_lookback(mut self, period: usize) -> Self {
        self.lookback_period = period;
        self
    }

    /// Извлекает признаки из OHLCV данных
    pub fn extract_features(&self, klines: &[Kline]) -> Features {
        let n = klines.len();
        if n < self.lookback_period + 1 {
            return Features::new(vec![], vec![], vec![]);
        }

        let mut feature_matrix: Vec<Vec<f64>> = Vec::new();
        let mut timestamps: Vec<i64> = Vec::new();

        // Начинаем с lookback_period, чтобы иметь достаточно данных для индикаторов
        for i in self.lookback_period..n {
            let window = &klines[i - self.lookback_period..=i];
            let current = &klines[i];

            let mut features = Vec::new();

            // 1. Returns (доходности)
            features.push(current.price_change_percent());

            // 2. Log returns
            if klines[i - 1].close > 0.0 && current.close > 0.0 {
                features.push((current.close / klines[i - 1].close).ln());
            } else {
                features.push(0.0);
            }

            // 3. Volatility (стандартное отклонение доходностей)
            let returns: Vec<f64> = window
                .windows(2)
                .filter_map(|w| {
                    if w[0].close > 0.0 {
                        Some((w[1].close / w[0].close).ln())
                    } else {
                        None
                    }
                })
                .collect();
            features.push(std_dev(&returns));

            // 4. RSI (Relative Strength Index)
            features.push(self.calculate_rsi(window));

            // 5. SMA ratio (отношение цены к SMA)
            let sma = self.calculate_sma(window);
            if sma > 0.0 {
                features.push(current.close / sma - 1.0);
            } else {
                features.push(0.0);
            }

            // 6. EMA ratio
            let ema = self.calculate_ema(window, 0.1);
            if ema > 0.0 {
                features.push(current.close / ema - 1.0);
            } else {
                features.push(0.0);
            }

            // 7. Bollinger Bands position
            let (bb_upper, bb_lower) = self.calculate_bollinger_bands(window, 2.0);
            if bb_upper > bb_lower {
                features.push((current.close - bb_lower) / (bb_upper - bb_lower));
            } else {
                features.push(0.5);
            }

            // 8. ATR (Average True Range) normalized
            let atr = self.calculate_atr(window);
            if current.close > 0.0 {
                features.push(atr / current.close);
            } else {
                features.push(0.0);
            }

            // 9. Volume ratio (относительно среднего)
            let avg_volume: f64 = window.iter().map(|k| k.volume).sum::<f64>() / window.len() as f64;
            if avg_volume > 0.0 {
                features.push(current.volume / avg_volume);
            } else {
                features.push(1.0);
            }

            // 10. Price range ratio
            if current.close > 0.0 {
                features.push(current.range() / current.close);
            } else {
                features.push(0.0);
            }

            // 11. MACD
            let (macd, signal) = self.calculate_macd(window);
            features.push(macd - signal);

            // 12. Momentum (изменение за N периодов)
            if window.first().map_or(false, |k| k.close > 0.0) {
                let first_close = window.first().unwrap().close;
                features.push((current.close - first_close) / first_close);
            } else {
                features.push(0.0);
            }

            // 13. High-Low ratio
            let period_high = window.iter().map(|k| k.high).fold(f64::MIN, f64::max);
            let period_low = window.iter().map(|k| k.low).fold(f64::MAX, f64::min);
            if period_high > period_low {
                features.push((current.close - period_low) / (period_high - period_low));
            } else {
                features.push(0.5);
            }

            // 14. Candle body ratio
            if current.range() > 0.0 {
                features.push(current.price_change().abs() / current.range());
            } else {
                features.push(0.0);
            }

            // 15. Upper shadow ratio
            if current.range() > 0.0 {
                let upper_shadow = current.high - current.close.max(current.open);
                features.push(upper_shadow / current.range());
            } else {
                features.push(0.0);
            }

            // 16. Lower shadow ratio
            if current.range() > 0.0 {
                let lower_shadow = current.close.min(current.open) - current.low;
                features.push(lower_shadow / current.range());
            } else {
                features.push(0.0);
            }

            feature_matrix.push(features);
            timestamps.push(current.open_time);
        }

        let names = vec![
            "return".to_string(),
            "log_return".to_string(),
            "volatility".to_string(),
            "rsi".to_string(),
            "sma_ratio".to_string(),
            "ema_ratio".to_string(),
            "bb_position".to_string(),
            "atr_normalized".to_string(),
            "volume_ratio".to_string(),
            "price_range_ratio".to_string(),
            "macd_signal_diff".to_string(),
            "momentum".to_string(),
            "high_low_position".to_string(),
            "body_ratio".to_string(),
            "upper_shadow_ratio".to_string(),
            "lower_shadow_ratio".to_string(),
        ];

        Features::new(feature_matrix, names, timestamps)
    }

    /// Нормализует признаки
    pub fn normalize(&mut self, features: &Features) -> Features {
        let arr = features.to_array();
        let (normalized, params) = self.normalize_array(&arr);
        self.norm_params = Some(params);
        Features::from_array(&normalized, features.names.clone(), features.timestamps.clone())
    }

    /// Нормализует матрицу признаков
    fn normalize_array(&self, arr: &Array2<f64>) -> (Array2<f64>, NormalizationParams) {
        let n_cols = arr.ncols();
        let mut min_vals = vec![0.0; n_cols];
        let mut max_vals = vec![1.0; n_cols];
        let mut mean_vals = vec![0.0; n_cols];
        let mut std_vals = vec![1.0; n_cols];

        match self.normalization {
            NormalizationMethod::MinMax => {
                for (j, col) in arr.axis_iter(Axis(1)).enumerate() {
                    min_vals[j] = col.iter().copied().fold(f64::INFINITY, f64::min);
                    max_vals[j] = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                }
            }
            NormalizationMethod::ZScore => {
                for (j, col) in arr.axis_iter(Axis(1)).enumerate() {
                    let col_vec: Vec<f64> = col.to_vec();
                    mean_vals[j] = mean(&col_vec);
                    std_vals[j] = std_dev(&col_vec).max(1e-10);
                }
            }
            NormalizationMethod::MaxAbs => {
                for (j, col) in arr.axis_iter(Axis(1)).enumerate() {
                    max_vals[j] = col.iter().map(|x| x.abs()).fold(0.0, f64::max).max(1e-10);
                }
            }
            NormalizationMethod::None => {}
        }

        let params = NormalizationParams {
            method: self.normalization,
            min_vals,
            max_vals,
            mean_vals,
            std_vals,
        };

        let normalized = self.apply_normalization(arr, &params);
        (normalized, params)
    }

    /// Применяет нормализацию к матрице
    fn apply_normalization(&self, arr: &Array2<f64>, params: &NormalizationParams) -> Array2<f64> {
        let mut result = arr.clone();

        match params.method {
            NormalizationMethod::MinMax => {
                for (j, mut col) in result.axis_iter_mut(Axis(1)).enumerate() {
                    let range = params.max_vals[j] - params.min_vals[j];
                    if range > 1e-10 {
                        col.mapv_inplace(|x| (x - params.min_vals[j]) / range);
                    }
                }
            }
            NormalizationMethod::ZScore => {
                for (j, mut col) in result.axis_iter_mut(Axis(1)).enumerate() {
                    col.mapv_inplace(|x| (x - params.mean_vals[j]) / params.std_vals[j]);
                }
            }
            NormalizationMethod::MaxAbs => {
                for (j, mut col) in result.axis_iter_mut(Axis(1)).enumerate() {
                    col.mapv_inplace(|x| x / params.max_vals[j]);
                }
            }
            NormalizationMethod::None => {}
        }

        result
    }

    /// Обратная нормализация
    pub fn denormalize(&self, features: &Features) -> Features {
        if let Some(ref params) = self.norm_params {
            let arr = features.to_array();
            let denormalized = self.apply_denormalization(&arr, params);
            Features::from_array(
                &denormalized,
                features.names.clone(),
                features.timestamps.clone(),
            )
        } else {
            features.clone()
        }
    }

    /// Применяет обратную нормализацию
    fn apply_denormalization(
        &self,
        arr: &Array2<f64>,
        params: &NormalizationParams,
    ) -> Array2<f64> {
        let mut result = arr.clone();

        match params.method {
            NormalizationMethod::MinMax => {
                for (j, mut col) in result.axis_iter_mut(Axis(1)).enumerate() {
                    let range = params.max_vals[j] - params.min_vals[j];
                    col.mapv_inplace(|x| x * range + params.min_vals[j]);
                }
            }
            NormalizationMethod::ZScore => {
                for (j, mut col) in result.axis_iter_mut(Axis(1)).enumerate() {
                    col.mapv_inplace(|x| x * params.std_vals[j] + params.mean_vals[j]);
                }
            }
            NormalizationMethod::MaxAbs => {
                for (j, mut col) in result.axis_iter_mut(Axis(1)).enumerate() {
                    col.mapv_inplace(|x| x * params.max_vals[j]);
                }
            }
            NormalizationMethod::None => {}
        }

        result
    }

    /// Создает датасет для обучения с возможностью добавления шума
    pub fn create_training_set(&self, features: &Features, noise_std: f64) -> (Features, Features) {
        let arr = features.to_array();

        // Добавляем шум к входным данным (для denoising autoencoder)
        let noisy = if noise_std > 0.0 {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            arr.mapv(|x| x + rng.gen::<f64>() * noise_std * 2.0 - noise_std)
        } else {
            arr.clone()
        };

        let noisy_features =
            Features::from_array(&noisy, features.names.clone(), features.timestamps.clone());

        (noisy_features, features.clone())
    }

    // ============ Технические индикаторы ============

    /// Вычисляет RSI (Relative Strength Index)
    fn calculate_rsi(&self, window: &[Kline]) -> f64 {
        let gains: Vec<f64> = window
            .windows(2)
            .map(|w| {
                let change = w[1].close - w[0].close;
                if change > 0.0 {
                    change
                } else {
                    0.0
                }
            })
            .collect();

        let losses: Vec<f64> = window
            .windows(2)
            .map(|w| {
                let change = w[1].close - w[0].close;
                if change < 0.0 {
                    -change
                } else {
                    0.0
                }
            })
            .collect();

        let avg_gain = mean(&gains);
        let avg_loss = mean(&losses);

        if avg_loss < 1e-10 {
            100.0
        } else {
            100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        }
    }

    /// Вычисляет простую скользящую среднюю
    fn calculate_sma(&self, window: &[Kline]) -> f64 {
        let closes: Vec<f64> = window.iter().map(|k| k.close).collect();
        mean(&closes)
    }

    /// Вычисляет экспоненциальную скользящую среднюю
    fn calculate_ema(&self, window: &[Kline], alpha: f64) -> f64 {
        let mut ema = window.first().map_or(0.0, |k| k.close);
        for kline in window.iter().skip(1) {
            ema = alpha * kline.close + (1.0 - alpha) * ema;
        }
        ema
    }

    /// Вычисляет полосы Боллинджера
    fn calculate_bollinger_bands(&self, window: &[Kline], num_std: f64) -> (f64, f64) {
        let closes: Vec<f64> = window.iter().map(|k| k.close).collect();
        let sma = mean(&closes);
        let std = std_dev(&closes);

        (sma + num_std * std, sma - num_std * std)
    }

    /// Вычисляет ATR (Average True Range)
    fn calculate_atr(&self, window: &[Kline]) -> f64 {
        let true_ranges: Vec<f64> = window
            .windows(2)
            .map(|w| {
                let high_low = w[1].high - w[1].low;
                let high_close = (w[1].high - w[0].close).abs();
                let low_close = (w[1].low - w[0].close).abs();
                high_low.max(high_close).max(low_close)
            })
            .collect();

        mean(&true_ranges)
    }

    /// Вычисляет MACD и сигнальную линию
    fn calculate_macd(&self, window: &[Kline]) -> (f64, f64) {
        let ema12 = self.calculate_ema(window, 2.0 / 13.0);
        let ema26 = self.calculate_ema(window, 2.0 / 27.0);
        let macd = ema12 - ema26;

        // Сигнальная линия (упрощенная версия)
        let signal = macd * 0.9; // Упрощение для демонстрации

        (macd, signal)
    }
}

// ============ Вспомогательные функции ============

/// Вычисляет среднее значение
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Вычисляет стандартное отклонение
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| Kline {
                open_time: 1700000000000 + i as i64 * 3600000,
                open: 100.0 + (i as f64 * 0.1).sin() * 10.0,
                high: 105.0 + (i as f64 * 0.1).sin() * 10.0,
                low: 95.0 + (i as f64 * 0.1).sin() * 10.0,
                close: 102.0 + (i as f64 * 0.1).sin() * 10.0,
                volume: 1000.0 + i as f64 * 10.0,
                turnover: 100000.0 + i as f64 * 1000.0,
            })
            .collect()
    }

    #[test]
    fn test_extract_features() {
        let klines = create_test_klines(50);
        let processor = DataProcessor::new().with_lookback(20);
        let features = processor.extract_features(&klines);

        assert!(features.nrows() > 0);
        assert_eq!(features.ncols(), 16);
    }

    #[test]
    fn test_normalization() {
        let klines = create_test_klines(50);
        let mut processor = DataProcessor::new().with_normalization(NormalizationMethod::MinMax);
        let features = processor.extract_features(&klines);
        let normalized = processor.normalize(&features);

        let arr = normalized.to_array();
        for val in arr.iter() {
            assert!(*val >= -0.1 && *val <= 1.1); // С небольшим допуском
        }
    }

    #[test]
    fn test_mean_std() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&values), 3.0);
        assert!((std_dev(&values) - 1.5811).abs() < 0.01);
    }
}
