//! Загрузчик данных для Informer

use crate::api::Kline;
use crate::data::{Features, TimeSeriesDataset};

/// Загрузчик и препроцессор данных
pub struct DataLoader {
    /// Окно для вычисления признаков
    lookback: usize,
    /// Окно для нормализации
    norm_window: usize,
}

impl DataLoader {
    /// Создаёт новый загрузчик
    pub fn new() -> Self {
        Self {
            lookback: 20,
            norm_window: 100,
        }
    }

    /// Создаёт загрузчик с кастомными параметрами
    pub fn with_params(lookback: usize, norm_window: usize) -> Self {
        Self {
            lookback,
            norm_window,
        }
    }

    /// Подготавливает dataset из свечей
    ///
    /// # Arguments
    ///
    /// * `klines` - Исторические свечи
    /// * `seq_len` - Длина входной последовательности
    /// * `pred_len` - Горизонт прогнозирования
    ///
    /// # Returns
    ///
    /// Dataset готовый для обучения
    pub fn prepare_dataset(
        &self,
        klines: &[Kline],
        seq_len: usize,
        pred_len: usize,
    ) -> Result<TimeSeriesDataset, String> {
        if klines.len() < self.lookback + seq_len + pred_len {
            return Err(format!(
                "Not enough klines: {} < {} + {} + {}",
                klines.len(), self.lookback, seq_len, pred_len
            ));
        }

        // Вычисляем признаки
        let features = Features::compute(klines, self.lookback);

        // Нормализуем
        let normalized = features.normalize(self.norm_window);

        // Преобразуем в матрицу
        let feature_matrix = normalized.to_matrix();

        // Целевые значения - log returns
        let targets = normalized.returns.clone();

        // Пропускаем начальные точки с неполными признаками
        let skip = self.lookback.max(self.norm_window);

        if feature_matrix.len() <= skip + seq_len + pred_len {
            return Err("Not enough data after preprocessing".to_string());
        }

        let valid_features: Vec<Vec<f64>> = feature_matrix[skip..].to_vec();
        let valid_targets: Vec<f64> = targets[skip..].to_vec();

        TimeSeriesDataset::new(&valid_features, &valid_targets, seq_len, pred_len)
    }

    /// Подготавливает данные для prediction (без targets)
    pub fn prepare_inference(
        &self,
        klines: &[Kline],
        seq_len: usize,
    ) -> Result<ndarray::Array3<f64>, String> {
        if klines.len() < seq_len + self.lookback {
            return Err(format!(
                "Not enough klines for inference: {} < {} + {}",
                klines.len(), seq_len, self.lookback
            ));
        }

        // Вычисляем признаки
        let features = Features::compute(klines, self.lookback);
        let normalized = features.normalize(self.norm_window);
        let feature_matrix = normalized.to_matrix();

        let skip = self.lookback.max(self.norm_window);

        if feature_matrix.len() < skip + seq_len {
            return Err("Not enough data after preprocessing".to_string());
        }

        // Берём последние seq_len точек
        let n_features = feature_matrix[0].len();
        let start = feature_matrix.len() - seq_len;

        let mut x = ndarray::Array3::zeros((1, seq_len, n_features));

        for t in 0..seq_len {
            for f in 0..n_features {
                x[[0, t, f]] = feature_matrix[start + t][f];
            }
        }

        Ok(x)
    }
}

impl Default for DataLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n).map(|i| {
            let base = 100.0 + (i as f64 * 0.01).sin() * 10.0;
            Kline {
                timestamp: i as u64 * 3600000,
                open: base,
                high: base + 1.0 + (i as f64 * 0.1).sin().abs(),
                low: base - 1.0 - (i as f64 * 0.1).cos().abs(),
                close: base + (i as f64 * 0.05).sin(),
                volume: 1000.0 + (i as f64 * 10.0).sin().abs() * 500.0,
                turnover: 100000.0,
            }
        }).collect()
    }

    #[test]
    fn test_prepare_dataset() {
        let klines = create_test_klines(500);
        let loader = DataLoader::new();

        let dataset = loader.prepare_dataset(&klines, 96, 24).unwrap();

        assert!(dataset.n_samples > 0);
        assert_eq!(dataset.seq_len, 96);
        assert_eq!(dataset.pred_len, 24);
        assert_eq!(dataset.n_features, 6);
    }

    #[test]
    fn test_prepare_dataset_not_enough_data() {
        let klines = create_test_klines(50);
        let loader = DataLoader::new();

        let result = loader.prepare_dataset(&klines, 96, 24);

        assert!(result.is_err());
    }

    #[test]
    fn test_prepare_inference() {
        let klines = create_test_klines(200);
        let loader = DataLoader::new();

        let x = loader.prepare_inference(&klines, 96).unwrap();

        assert_eq!(x.dim(), (1, 96, 6));
    }

    #[test]
    fn test_no_nan_in_features() {
        let klines = create_test_klines(500);
        let loader = DataLoader::new();

        let dataset = loader.prepare_dataset(&klines, 48, 12).unwrap();

        for val in dataset.x.iter() {
            assert!(!val.is_nan(), "Features contain NaN");
            assert!(!val.is_infinite(), "Features contain Inf");
        }

        for val in dataset.y.iter() {
            assert!(!val.is_nan(), "Targets contain NaN");
            assert!(!val.is_infinite(), "Targets contain Inf");
        }
    }
}
