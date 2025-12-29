//! Dataset для TFT
//!
//! Модуль для создания обучающих выборок для Temporal Fusion Transformer.

use super::Features;
use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};

/// Один sample для TFT (encoder context + decoder target)
#[derive(Debug, Clone)]
pub struct TFTSample {
    /// Encoder input: прошлые значения признаков
    /// Shape: (encoder_length, num_features)
    pub encoder_input: Array2<f64>,

    /// Decoder input: известные будущие значения
    /// Shape: (prediction_length, num_known_features)
    pub decoder_input: Array2<f64>,

    /// Target: целевые значения для предсказания
    /// Shape: (prediction_length,)
    pub target: Array1<f64>,

    /// Статические признаки
    /// Shape: (num_static_features,)
    pub static_features: Array1<f64>,

    /// Временная метка начала encoder context
    pub timestamp_start: i64,

    /// Временная метка начала prediction
    pub timestamp_prediction: i64,
}

/// Dataset для обучения TFT
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Все samples
    pub samples: Vec<TFTSample>,

    /// Длина encoder context
    pub encoder_length: usize,

    /// Длина prediction horizon
    pub prediction_length: usize,

    /// Названия признаков encoder
    pub encoder_feature_names: Vec<String>,

    /// Названия признаков decoder (known future)
    pub decoder_feature_names: Vec<String>,

    /// Названия статических признаков
    pub static_feature_names: Vec<String>,

    /// Целевая переменная
    pub target_name: String,
}

impl Dataset {
    /// Создает пустой dataset
    pub fn new(
        encoder_length: usize,
        prediction_length: usize,
        target_name: &str,
    ) -> Self {
        Self {
            samples: Vec::new(),
            encoder_length,
            prediction_length,
            encoder_feature_names: Vec::new(),
            decoder_feature_names: Vec::new(),
            static_feature_names: Vec::new(),
            target_name: target_name.to_string(),
        }
    }

    /// Возвращает количество samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Проверяет, пустой ли dataset
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Разбивает dataset на train/validation/test
    pub fn train_val_test_split(
        &self,
        train_ratio: f64,
        val_ratio: f64,
    ) -> (Dataset, Dataset, Dataset) {
        let n = self.samples.len();
        let train_end = (n as f64 * train_ratio) as usize;
        let val_end = train_end + (n as f64 * val_ratio) as usize;

        let train = Dataset {
            samples: self.samples[..train_end].to_vec(),
            encoder_length: self.encoder_length,
            prediction_length: self.prediction_length,
            encoder_feature_names: self.encoder_feature_names.clone(),
            decoder_feature_names: self.decoder_feature_names.clone(),
            static_feature_names: self.static_feature_names.clone(),
            target_name: self.target_name.clone(),
        };

        let val = Dataset {
            samples: self.samples[train_end..val_end].to_vec(),
            encoder_length: self.encoder_length,
            prediction_length: self.prediction_length,
            encoder_feature_names: self.encoder_feature_names.clone(),
            decoder_feature_names: self.decoder_feature_names.clone(),
            static_feature_names: self.static_feature_names.clone(),
            target_name: self.target_name.clone(),
        };

        let test = Dataset {
            samples: self.samples[val_end..].to_vec(),
            encoder_length: self.encoder_length,
            prediction_length: self.prediction_length,
            encoder_feature_names: self.encoder_feature_names.clone(),
            decoder_feature_names: self.decoder_feature_names.clone(),
            static_feature_names: self.static_feature_names.clone(),
            target_name: self.target_name.clone(),
        };

        (train, val, test)
    }

    /// Возвращает batch samples
    pub fn get_batch(&self, start: usize, batch_size: usize) -> Vec<&TFTSample> {
        let end = (start + batch_size).min(self.samples.len());
        self.samples[start..end].iter().collect()
    }

    /// Возвращает итератор по batches
    pub fn batches(&self, batch_size: usize) -> BatchIterator {
        BatchIterator {
            dataset: self,
            batch_size,
            current: 0,
        }
    }
}

/// Итератор по batches
pub struct BatchIterator<'a> {
    dataset: &'a Dataset,
    batch_size: usize,
    current: usize,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = Vec<&'a TFTSample>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.dataset.samples.len() {
            return None;
        }

        let batch = self.dataset.get_batch(self.current, self.batch_size);
        self.current += self.batch_size;

        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }
}

/// Конфигурация для TimeSeriesDataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesDatasetConfig {
    /// Длина encoder context
    pub encoder_length: usize,

    /// Длина prediction horizon
    pub prediction_length: usize,

    /// Индекс целевой переменной в features
    pub target_idx: usize,

    /// Индексы признаков, известных в будущем
    pub known_future_indices: Vec<usize>,

    /// Индексы статических признаков
    pub static_indices: Vec<usize>,

    /// Шаг между samples
    pub step: usize,
}

impl Default for TimeSeriesDatasetConfig {
    fn default() -> Self {
        Self {
            encoder_length: 168, // 7 дней часовых данных
            prediction_length: 24, // 24 часа вперед
            target_idx: 4, // returns
            known_future_indices: vec![20, 21, 22, 23], // hour_sin/cos, dow_sin/cos
            static_indices: vec![],
            step: 1,
        }
    }
}

/// TimeSeriesDataset для работы с временными рядами
#[derive(Debug, Clone)]
pub struct TimeSeriesDataset {
    /// Конфигурация
    pub config: TimeSeriesDatasetConfig,

    /// Признаки
    pub features: Features,

    /// Статические признаки (общие для всего ряда)
    pub static_features: Array1<f64>,
}

impl TimeSeriesDataset {
    /// Создает dataset из features
    pub fn new(features: Features, config: TimeSeriesDatasetConfig) -> Self {
        // Извлекаем статические признаки (можно расширить)
        let static_features = Array1::zeros(config.static_indices.len());

        Self {
            config,
            features,
            static_features,
        }
    }

    /// Возвращает общее количество возможных samples
    pub fn num_samples(&self) -> usize {
        let total_length = self.config.encoder_length + self.config.prediction_length;
        if self.features.len() < total_length {
            return 0;
        }
        (self.features.len() - total_length) / self.config.step + 1
    }

    /// Получает sample по индексу
    pub fn get_sample(&self, idx: usize) -> Option<TFTSample> {
        let start = idx * self.config.step;
        let encoder_end = start + self.config.encoder_length;
        let prediction_end = encoder_end + self.config.prediction_length;

        if prediction_end > self.features.len() {
            return None;
        }

        // Encoder input: все признаки за прошлый период
        let encoder_input = self
            .features
            .values
            .slice(ndarray::s![start..encoder_end, ..])
            .to_owned();

        // Decoder input: только known future признаки
        let decoder_cols: Vec<usize> = self.config.known_future_indices.clone();
        let mut decoder_input =
            Array2::zeros((self.config.prediction_length, decoder_cols.len()));

        for (new_col, &orig_col) in decoder_cols.iter().enumerate() {
            for row in 0..self.config.prediction_length {
                decoder_input[[row, new_col]] =
                    self.features.values[[encoder_end + row, orig_col]];
            }
        }

        // Target: целевая переменная для prediction period
        let target_col = self.config.target_idx;
        let target = self
            .features
            .values
            .slice(ndarray::s![encoder_end..prediction_end, target_col])
            .to_owned();

        Some(TFTSample {
            encoder_input,
            decoder_input,
            target,
            static_features: self.static_features.clone(),
            timestamp_start: self.features.timestamps[start],
            timestamp_prediction: self.features.timestamps[encoder_end],
        })
    }

    /// Создает полный Dataset из TimeSeriesDataset
    pub fn to_dataset(&self) -> Dataset {
        let mut dataset = Dataset::new(
            self.config.encoder_length,
            self.config.prediction_length,
            &self.features.names[self.config.target_idx],
        );

        // Копируем имена признаков
        dataset.encoder_feature_names = self.features.names.clone();
        dataset.decoder_feature_names = self
            .config
            .known_future_indices
            .iter()
            .filter_map(|&i| self.features.names.get(i).cloned())
            .collect();

        // Создаем samples
        let num = self.num_samples();
        for i in 0..num {
            if let Some(sample) = self.get_sample(i) {
                dataset.samples.push(sample);
            }
        }

        dataset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_dataset_split() {
        let mut dataset = Dataset::new(10, 5, "returns");

        // Создаем фейковые samples
        for i in 0..100 {
            dataset.samples.push(TFTSample {
                encoder_input: Array2::zeros((10, 4)),
                decoder_input: Array2::zeros((5, 2)),
                target: Array1::zeros(5),
                static_features: Array1::zeros(1),
                timestamp_start: i as i64 * 1000,
                timestamp_prediction: (i + 10) as i64 * 1000,
            });
        }

        let (train, val, test) = dataset.train_val_test_split(0.7, 0.15);

        assert_eq!(train.len(), 70);
        assert_eq!(val.len(), 15);
        assert_eq!(test.len(), 15);
    }

    #[test]
    fn test_batch_iterator() {
        let mut dataset = Dataset::new(10, 5, "returns");

        for i in 0..25 {
            dataset.samples.push(TFTSample {
                encoder_input: Array2::zeros((10, 4)),
                decoder_input: Array2::zeros((5, 2)),
                target: Array1::zeros(5),
                static_features: Array1::zeros(1),
                timestamp_start: i as i64 * 1000,
                timestamp_prediction: (i + 10) as i64 * 1000,
            });
        }

        let batches: Vec<_> = dataset.batches(10).collect();

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 10);
        assert_eq!(batches[1].len(), 10);
        assert_eq!(batches[2].len(), 5);
    }
}
