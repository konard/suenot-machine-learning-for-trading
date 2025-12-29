//! Обработка сырых данных в формат для CNN

use super::sample::{Label, Sample};
use crate::bybit::Kline;
use crate::indicators::{IndicatorConfig, TechnicalIndicators};
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Конфигурация процессора данных
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Размер окна (количество свечей)
    pub window_size: usize,
    /// Горизонт прогнозирования (свечи вперёд)
    pub prediction_horizon: usize,
    /// Порог для классификации (в процентах)
    pub classification_threshold: f64,
    /// Использовать логарифмические доходности
    pub use_log_returns: bool,
    /// Нормализация по окну
    pub normalize_window: bool,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            window_size: 60,
            prediction_horizon: 4,
            classification_threshold: 0.5,
            use_log_returns: true,
            normalize_window: true,
        }
    }
}

/// Процессор данных для CNN
pub struct DataProcessor {
    config: ProcessorConfig,
    indicator_config: IndicatorConfig,
    /// Статистики для нормализации (mean, std для каждого канала)
    normalization_stats: Option<Vec<(f64, f64)>>,
}

impl DataProcessor {
    /// Создание процессора с конфигурацией по умолчанию
    pub fn new() -> Self {
        Self {
            config: ProcessorConfig::default(),
            indicator_config: IndicatorConfig::default(),
            normalization_stats: None,
        }
    }

    /// Создание с кастомной конфигурацией
    pub fn with_config(config: ProcessorConfig) -> Self {
        Self {
            config,
            indicator_config: IndicatorConfig::default(),
            normalization_stats: None,
        }
    }

    /// Установка конфигурации индикаторов
    pub fn with_indicators(mut self, config: IndicatorConfig) -> Self {
        self.indicator_config = config;
        self
    }

    /// Преобразование свечей в матрицу признаков
    pub fn klines_to_features(&self, klines: &[Kline]) -> Array2<f64> {
        let n = klines.len();
        if n == 0 {
            return Array2::zeros((0, 0));
        }

        // Извлекаем базовые OHLCV данные
        let open: Vec<f64> = klines.iter().map(|k| k.open).collect();
        let high: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let low: Vec<f64> = klines.iter().map(|k| k.low).collect();
        let close: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volume: Vec<f64> = klines.iter().map(|k| k.volume).collect();

        // Вычисляем технические индикаторы
        let indicators = TechnicalIndicators::new(&self.indicator_config);
        let rsi = indicators.rsi(&close);
        let (macd, signal, hist) = indicators.macd(&close);
        let (bb_upper, bb_middle, bb_lower) = indicators.bollinger_bands(&close);
        let atr = indicators.atr(&high, &low, &close);
        let obv = indicators.obv(&close, &volume);
        let ema_fast = indicators.ema(&close, self.indicator_config.ema_fast);
        let ema_slow = indicators.ema(&close, self.indicator_config.ema_slow);

        // Вычисляем доходности
        let returns = self.compute_returns(&close);
        let log_returns = self.compute_log_returns(&close);

        // Нормализуем цены относительно SMA
        let price_normalized = self.normalize_prices(&close, &bb_middle);

        // Нормализуем объём
        let volume_normalized = self.normalize_volume(&volume);

        // Собираем все признаки в матрицу
        // Каналы: returns, log_returns, rsi, macd_hist, bb_position, atr, volume_norm, ema_diff
        let num_channels = 10;
        let mut features = Array2::zeros((num_channels, n));

        // Заполняем каналы
        for i in 0..n {
            features[[0, i]] = returns[i];
            features[[1, i]] = log_returns[i];
            features[[2, i]] = (rsi[i] - 50.0) / 50.0; // Нормализуем RSI к [-1, 1]
            features[[3, i]] = hist[i] / close[i] * 100.0; // MACD histogram нормализованный
            features[[4, i]] = price_normalized[i];
            features[[5, i]] = atr[i] / close[i] * 100.0; // ATR в процентах
            features[[6, i]] = volume_normalized[i];
            features[[7, i]] = (ema_fast[i] - ema_slow[i]) / close[i] * 100.0; // EMA crossover
            features[[8, i]] = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10); // BB позиция
            features[[9, i]] = macd[i] / close[i] * 100.0; // MACD нормализованный
        }

        features
    }

    /// Вычисление простых доходностей
    fn compute_returns(&self, close: &[f64]) -> Vec<f64> {
        let mut returns = vec![0.0];
        for i in 1..close.len() {
            if close[i - 1] != 0.0 {
                returns.push((close[i] - close[i - 1]) / close[i - 1] * 100.0);
            } else {
                returns.push(0.0);
            }
        }
        returns
    }

    /// Вычисление логарифмических доходностей
    fn compute_log_returns(&self, close: &[f64]) -> Vec<f64> {
        let mut returns = vec![0.0];
        for i in 1..close.len() {
            if close[i - 1] > 0.0 && close[i] > 0.0 {
                returns.push((close[i] / close[i - 1]).ln() * 100.0);
            } else {
                returns.push(0.0);
            }
        }
        returns
    }

    /// Нормализация цен относительно SMA
    fn normalize_prices(&self, close: &[f64], sma: &[f64]) -> Vec<f64> {
        close
            .iter()
            .zip(sma.iter())
            .map(|(c, s)| if *s != 0.0 { (c - s) / s * 100.0 } else { 0.0 })
            .collect()
    }

    /// Нормализация объёма (z-score по скользящему окну)
    fn normalize_volume(&self, volume: &[f64]) -> Vec<f64> {
        let window = 20.min(volume.len());
        let mut normalized = vec![0.0; volume.len()];

        for i in window..volume.len() {
            let window_slice = &volume[i - window..i];
            let mean: f64 = window_slice.iter().sum::<f64>() / window as f64;
            let variance: f64 = window_slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / window as f64;
            let std = variance.sqrt();

            if std > 0.0 {
                normalized[i] = (volume[i] - mean) / std;
            }
        }

        normalized
    }

    /// Создание образцов из данных свечей
    pub fn create_samples(&self, klines: &[Kline]) -> Vec<Sample> {
        let features = self.klines_to_features(klines);
        let (num_channels, total_len) = (features.shape()[0], features.shape()[1]);

        if total_len < self.config.window_size + self.config.prediction_horizon {
            return Vec::new();
        }

        let mut samples = Vec::new();
        let end_idx = total_len - self.config.prediction_horizon;

        for i in self.config.window_size..=end_idx {
            // Извлекаем окно признаков
            let window_start = i - self.config.window_size;
            let window_features = features.slice(ndarray::s![.., window_start..i]).to_owned();

            // Преобразуем в f32
            let window_f32: Array2<f32> = window_features.mapv(|x| x as f32);

            // Вычисляем целевую доходность
            let current_close = klines[i - 1].close;
            let future_close = klines[i - 1 + self.config.prediction_horizon].close;
            let future_return = (future_close - current_close) / current_close * 100.0;

            // Создаём метку
            let label = Label::from_return(future_return, self.config.classification_threshold);

            let sample = Sample::new(window_f32, klines[i - 1].timestamp)
                .with_label(label)
                .with_return(future_return);

            samples.push(sample);
        }

        samples
    }

    /// Нормализация признаков в образце
    pub fn normalize_sample(&self, sample: &mut Sample) {
        if !self.config.normalize_window {
            return;
        }

        // Z-score нормализация по каждому каналу
        for i in 0..sample.num_channels() {
            let row = sample.features.row(i);
            let mean: f32 = row.mean().unwrap_or(0.0);
            let std: f32 = row.std(0.0);

            if std > 1e-8 {
                for j in 0..sample.window_size() {
                    sample.features[[i, j]] = (sample.features[[i, j]] - mean) / std;
                }
            }
        }
    }

    /// Создание нормализованных образцов
    pub fn create_normalized_samples(&self, klines: &[Kline]) -> Vec<Sample> {
        let mut samples = self.create_samples(klines);
        for sample in &mut samples {
            self.normalize_sample(sample);
        }
        samples
    }

    /// Разделение на train/test
    pub fn train_test_split(
        samples: Vec<Sample>,
        test_ratio: f64,
    ) -> (Vec<Sample>, Vec<Sample>) {
        let split_idx = ((1.0 - test_ratio) * samples.len() as f64) as usize;
        let (train, test) = samples.split_at(split_idx);
        (train.to_vec(), test.to_vec())
    }

    /// Конфигурация процессора
    pub fn config(&self) -> &ProcessorConfig {
        &self.config
    }
}

impl Default for DataProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| Kline {
                timestamp: i as i64 * 60000,
                open: 100.0 + (i as f64) * 0.1,
                high: 101.0 + (i as f64) * 0.1,
                low: 99.0 + (i as f64) * 0.1,
                close: 100.5 + (i as f64) * 0.1,
                volume: 1000.0 + (i as f64) * 10.0,
                turnover: 100000.0,
            })
            .collect()
    }

    #[test]
    fn test_create_samples() {
        let klines = create_test_klines(100);
        let processor = DataProcessor::new();
        let samples = processor.create_samples(&klines);

        assert!(!samples.is_empty());
        assert_eq!(samples[0].num_channels(), 10);
        assert_eq!(samples[0].window_size(), 60);
    }

    #[test]
    fn test_train_test_split() {
        let klines = create_test_klines(200);
        let processor = DataProcessor::new();
        let samples = processor.create_samples(&klines);

        let (train, test) = DataProcessor::train_test_split(samples, 0.2);

        assert!(train.len() > test.len());
    }
}
