//! Основной процессор данных для подготовки к обучению RNN

use super::features::FeatureExtractor;
use super::normalizer::{MinMaxNormalizer, Normalizer};
use crate::data::Candle;
use anyhow::{anyhow, Result};
use ndarray::{s, Array2, Array3};
use serde::{Deserialize, Serialize};

/// Процессор данных для подготовки последовательностей
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataProcessor {
    /// Количество шагов назад (lookback)
    pub sequence_length: usize,
    /// Количество шагов вперёд для предсказания
    pub forecast_horizon: usize,
    /// Нормализатор данных
    normalizer: MinMaxNormalizer,
    /// Индекс целевой переменной (обычно close = 3)
    pub target_index: usize,
    /// Обучен ли процессор
    is_fitted: bool,
}

impl DataProcessor {
    /// Создаёт новый процессор данных
    ///
    /// # Аргументы
    ///
    /// * `sequence_length` - Сколько шагов назад смотреть (lookback)
    /// * `forecast_horizon` - Сколько шагов вперёд предсказывать
    ///
    /// # Пример
    ///
    /// ```rust
    /// use crypto_rnn::preprocessing::DataProcessor;
    ///
    /// // Смотрим на 60 свечей назад, предсказываем 1 вперёд
    /// let processor = DataProcessor::new(60, 1);
    /// ```
    pub fn new(sequence_length: usize, forecast_horizon: usize) -> Self {
        Self {
            sequence_length,
            forecast_horizon,
            normalizer: MinMaxNormalizer::new(),
            target_index: 3, // close price by default
            is_fitted: false,
        }
    }

    /// Устанавливает индекс целевой переменной
    pub fn with_target_index(mut self, index: usize) -> Self {
        self.target_index = index;
        self
    }

    /// Подготавливает последовательности из свечных данных
    ///
    /// Возвращает:
    /// - X: 3D массив [samples, sequence_length, features]
    /// - y: 2D массив [samples, 1] (цены закрытия для предсказания)
    pub fn prepare_sequences(&mut self, candles: &[Candle]) -> Result<(Array3<f64>, Array2<f64>)> {
        if candles.len() < self.sequence_length + self.forecast_horizon {
            return Err(anyhow!(
                "Недостаточно данных: {} свечей, нужно минимум {}",
                candles.len(),
                self.sequence_length + self.forecast_horizon
            ));
        }

        // Извлекаем признаки
        let extractor = FeatureExtractor::default();
        let features = extractor.extract(candles);

        // Преобразуем в 2D массив
        let n_samples = features.len();
        let n_features = features[0].len();
        let mut data = Array2::zeros((n_samples, n_features));

        for (i, row) in features.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                data[[i, j]] = val;
            }
        }

        // Нормализуем данные
        let normalized = self.normalizer.fit_transform(&data);
        self.is_fitted = true;

        // Создаём последовательности
        let n_sequences = n_samples - self.sequence_length - self.forecast_horizon + 1;

        let mut x = Array3::zeros((n_sequences, self.sequence_length, n_features));
        let mut y = Array2::zeros((n_sequences, 1));

        for i in 0..n_sequences {
            // X: последовательность sequence_length шагов
            for t in 0..self.sequence_length {
                for f in 0..n_features {
                    x[[i, t, f]] = normalized[[i + t, f]];
                }
            }

            // y: цена закрытия через forecast_horizon шагов
            let target_idx = i + self.sequence_length + self.forecast_horizon - 1;
            y[[i, 0]] = normalized[[target_idx, self.target_index]];
        }

        Ok((x, y))
    }

    /// Подготавливает данные для классификации (вверх/вниз)
    ///
    /// Возвращает:
    /// - X: 3D массив [samples, sequence_length, features]
    /// - y: 2D массив [samples, 1] с метками (1.0 = вверх, 0.0 = вниз)
    pub fn prepare_classification(
        &mut self,
        candles: &[Candle],
    ) -> Result<(Array3<f64>, Array2<f64>)> {
        let (x, _) = self.prepare_sequences(candles)?;

        let n_sequences = x.shape()[0];
        let mut y = Array2::zeros((n_sequences, 1));

        for i in 0..n_sequences {
            let current_idx = i + self.sequence_length - 1;
            let future_idx = current_idx + self.forecast_horizon;

            if future_idx < candles.len() {
                let current_price = candles[current_idx].close;
                let future_price = candles[future_idx].close;

                // 1.0 если цена выросла, 0.0 если упала
                y[[i, 0]] = if future_price > current_price {
                    1.0
                } else {
                    0.0
                };
            }
        }

        Ok((x, y))
    }

    /// Разделяет данные на обучающую и тестовую выборки
    ///
    /// # Аргументы
    ///
    /// * `x` - Входные данные
    /// * `y` - Целевые значения
    /// * `train_ratio` - Доля обучающей выборки (0.0 - 1.0)
    pub fn train_test_split(
        &self,
        x: &Array3<f64>,
        y: &Array2<f64>,
        train_ratio: f64,
    ) -> (Array3<f64>, Array3<f64>, Array2<f64>, Array2<f64>) {
        let n_samples = x.shape()[0];
        let train_size = (n_samples as f64 * train_ratio) as usize;

        let x_train = x.slice(s![..train_size, .., ..]).to_owned();
        let x_test = x.slice(s![train_size.., .., ..]).to_owned();
        let y_train = y.slice(s![..train_size, ..]).to_owned();
        let y_test = y.slice(s![train_size.., ..]).to_owned();

        (x_train, x_test, y_train, y_test)
    }

    /// Подготавливает одну последовательность для предсказания
    pub fn prepare_single(&self, candles: &[Candle]) -> Result<Array3<f64>> {
        if !self.is_fitted {
            return Err(anyhow!("Процессор не обучен. Сначала вызовите prepare_sequences()"));
        }

        if candles.len() < self.sequence_length {
            return Err(anyhow!(
                "Недостаточно данных: {} свечей, нужно {}",
                candles.len(),
                self.sequence_length
            ));
        }

        let extractor = FeatureExtractor::default();
        let features = extractor.extract(candles);

        let n_features = features[0].len();
        let mut data = Array2::zeros((candles.len(), n_features));

        for (i, row) in features.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                data[[i, j]] = val;
            }
        }

        let normalized = self.normalizer.transform(&data);

        // Берём последние sequence_length элементов
        let start = candles.len() - self.sequence_length;
        let mut x = Array3::zeros((1, self.sequence_length, n_features));

        for t in 0..self.sequence_length {
            for f in 0..n_features {
                x[[0, t, f]] = normalized[[start + t, f]];
            }
        }

        Ok(x)
    }

    /// Обратное преобразование предсказанной цены
    pub fn inverse_transform_price(&self, normalized_price: f64) -> f64 {
        let min_vals = self.normalizer.min_vals().expect("Нормализатор не обучен");
        let max_vals = self.normalizer.max_vals().expect("Нормализатор не обучен");

        let min = min_vals[self.target_index];
        let max = max_vals[self.target_index];
        let range = max - min;

        normalized_price * range + min
    }

    /// Создаёт батчи данных для обучения
    pub fn create_batches(
        x: &Array3<f64>,
        y: &Array2<f64>,
        batch_size: usize,
    ) -> Vec<(Array3<f64>, Array2<f64>)> {
        let n_samples = x.shape()[0];
        let n_batches = (n_samples as f64 / batch_size as f64).ceil() as usize;

        let mut batches = Vec::with_capacity(n_batches);

        for i in 0..n_batches {
            let start = i * batch_size;
            let end = ((i + 1) * batch_size).min(n_samples);

            let x_batch = x.slice(s![start..end, .., ..]).to_owned();
            let y_batch = y.slice(s![start..end, ..]).to_owned();

            batches.push((x_batch, y_batch));
        }

        batches
    }

    /// Сохраняет процессор в файл
    pub fn save(&self, path: &str) -> Result<()> {
        let encoded = bincode::serialize(self)?;
        std::fs::write(path, encoded)?;
        Ok(())
    }

    /// Загружает процессор из файла
    pub fn load(path: &str) -> Result<Self> {
        let data = std::fs::read(path)?;
        let processor: Self = bincode::deserialize(&data)?;
        Ok(processor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| {
                let base_price = 100.0 + (i as f64 * 0.5);
                Candle::new(
                    i as i64 * 3600000,
                    base_price,
                    base_price + 2.0,
                    base_price - 1.0,
                    base_price + 1.0,
                    1000.0 + i as f64 * 10.0,
                    100000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_prepare_sequences() {
        let candles = create_test_candles(100);
        let mut processor = DataProcessor::new(10, 1);

        let (x, y) = processor.prepare_sequences(&candles).unwrap();

        assert_eq!(x.shape()[0], 90); // 100 - 10 - 1 + 1
        assert_eq!(x.shape()[1], 10); // sequence_length
        assert_eq!(y.shape()[0], 90);
    }

    #[test]
    fn test_train_test_split() {
        let candles = create_test_candles(100);
        let mut processor = DataProcessor::new(10, 1);

        let (x, y) = processor.prepare_sequences(&candles).unwrap();
        let (x_train, x_test, y_train, y_test) = processor.train_test_split(&x, &y, 0.8);

        assert_eq!(x_train.shape()[0], 72); // 80% of 90
        assert_eq!(x_test.shape()[0], 18); // 20% of 90
        assert_eq!(y_train.shape()[0], 72);
        assert_eq!(y_test.shape()[0], 18);
    }

    #[test]
    fn test_create_batches() {
        let candles = create_test_candles(100);
        let mut processor = DataProcessor::new(10, 1);

        let (x, y) = processor.prepare_sequences(&candles).unwrap();
        let batches = DataProcessor::create_batches(&x, &y, 32);

        assert_eq!(batches.len(), 3); // ceil(90 / 32) = 3
        assert_eq!(batches[0].0.shape()[0], 32);
        assert_eq!(batches[2].0.shape()[0], 26); // 90 - 64 = 26
    }
}
