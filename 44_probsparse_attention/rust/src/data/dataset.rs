//! Dataset для обучения модели Informer

use ndarray::{Array2, Array3};

/// Dataset для временных рядов
///
/// Хранит входные последовательности и целевые значения
#[derive(Debug, Clone)]
pub struct TimeSeriesDataset {
    /// Входные данные [n_samples, seq_len, n_features]
    pub x: Array3<f64>,
    /// Целевые значения [n_samples, pred_len]
    pub y: Array2<f64>,
    /// Количество образцов
    pub n_samples: usize,
    /// Длина входной последовательности
    pub seq_len: usize,
    /// Длина прогноза
    pub pred_len: usize,
    /// Количество признаков
    pub n_features: usize,
}

impl TimeSeriesDataset {
    /// Создаёт новый dataset из матрицы признаков
    ///
    /// # Arguments
    ///
    /// * `features` - Матрица признаков [n_points, n_features]
    /// * `targets` - Целевые значения [n_points]
    /// * `seq_len` - Длина входной последовательности
    /// * `pred_len` - Горизонт прогнозирования
    pub fn new(
        features: &[Vec<f64>],
        targets: &[f64],
        seq_len: usize,
        pred_len: usize,
    ) -> Result<Self, String> {
        let n_points = features.len();

        if n_points != targets.len() {
            return Err(format!(
                "Features and targets length mismatch: {} vs {}",
                n_points, targets.len()
            ));
        }

        if n_points < seq_len + pred_len {
            return Err(format!(
                "Not enough data points: {} < {} + {}",
                n_points, seq_len, pred_len
            ));
        }

        let n_features = features.first().map(|f| f.len()).unwrap_or(0);
        if n_features == 0 || features.iter().any(|row| row.len() != n_features) {
            return Err("All feature rows must have the same non-zero length".to_string());
        }
        let n_samples = n_points - seq_len - pred_len + 1;

        // Создаём массивы
        let mut x = Array3::zeros((n_samples, seq_len, n_features));
        let mut y = Array2::zeros((n_samples, pred_len));

        for i in 0..n_samples {
            // Входная последовательность: [i, i+seq_len)
            for t in 0..seq_len {
                for f in 0..n_features {
                    x[[i, t, f]] = features[i + t][f];
                }
            }

            // Целевые значения: [i+seq_len, i+seq_len+pred_len)
            for p in 0..pred_len {
                y[[i, p]] = targets[i + seq_len + p];
            }
        }

        Ok(Self {
            x,
            y,
            n_samples,
            seq_len,
            pred_len,
            n_features,
        })
    }

    /// Разделяет dataset на train/val/test
    ///
    /// Использует временное разделение (без перемешивания)
    pub fn split(
        &self,
        train_ratio: f64,
        val_ratio: f64,
    ) -> (TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset) {
        assert!(train_ratio >= 0.0 && train_ratio <= 1.0, "train_ratio must be in [0,1]");
        assert!(val_ratio >= 0.0 && val_ratio <= 1.0, "val_ratio must be in [0,1]");
        assert!(train_ratio + val_ratio <= 1.0, "train_ratio + val_ratio must be <= 1.0");
        let n_train = (self.n_samples as f64 * train_ratio) as usize;
        let n_val = (self.n_samples as f64 * val_ratio) as usize;

        let train = self.slice(0, n_train);
        let val = self.slice(n_train, n_train + n_val);
        let test = self.slice(n_train + n_val, self.n_samples);

        (train, val, test)
    }

    /// Возвращает срез dataset
    fn slice(&self, start: usize, end: usize) -> TimeSeriesDataset {
        let n_samples = end - start;

        let x = self.x.slice(ndarray::s![start..end, .., ..]).to_owned();
        let y = self.y.slice(ndarray::s![start..end, ..]).to_owned();

        TimeSeriesDataset {
            x,
            y,
            n_samples,
            seq_len: self.seq_len,
            pred_len: self.pred_len,
            n_features: self.n_features,
        }
    }

    /// Возвращает батч данных
    pub fn get_batch(&self, indices: &[usize]) -> (Array3<f64>, Array2<f64>) {
        let batch_size = indices.len();

        let mut x = Array3::zeros((batch_size, self.seq_len, self.n_features));
        let mut y = Array2::zeros((batch_size, self.pred_len));

        for (b, &i) in indices.iter().enumerate() {
            for t in 0..self.seq_len {
                for f in 0..self.n_features {
                    x[[b, t, f]] = self.x[[i, t, f]];
                }
            }
            for p in 0..self.pred_len {
                y[[b, p]] = self.y[[i, p]];
            }
        }

        (x, y)
    }

    /// Итератор по батчам
    pub fn batches(&self, batch_size: usize, shuffle: bool) -> BatchIterator {
        BatchIterator::new(self, batch_size, shuffle)
    }
}

/// Итератор по батчам
pub struct BatchIterator<'a> {
    dataset: &'a TimeSeriesDataset,
    indices: Vec<usize>,
    batch_size: usize,
    current: usize,
}

impl<'a> BatchIterator<'a> {
    fn new(dataset: &'a TimeSeriesDataset, batch_size: usize, shuffle: bool) -> Self {
        assert!(batch_size > 0, "batch_size must be > 0");
        let mut indices: Vec<usize> = (0..dataset.n_samples).collect();

        if shuffle {
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::thread_rng());
        }

        Self {
            dataset,
            indices,
            batch_size,
            current: 0,
        }
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = (Array3<f64>, Array2<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.indices.len() {
            return None;
        }

        let end = (self.current + self.batch_size).min(self.indices.len());
        let batch_indices: Vec<usize> = self.indices[self.current..end].to_vec();
        self.current = end;

        Some(self.dataset.get_batch(&batch_indices))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let features: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![i as f64 * 0.1, (i as f64 * 0.2).sin(), 1.0])
            .collect();

        let targets: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.1).sin())
            .collect();

        (features, targets)
    }

    #[test]
    fn test_dataset_creation() {
        let (features, targets) = create_test_data(100);
        let dataset = TimeSeriesDataset::new(&features, &targets, 24, 8).unwrap();

        assert_eq!(dataset.n_samples, 100 - 24 - 8 + 1);
        assert_eq!(dataset.seq_len, 24);
        assert_eq!(dataset.pred_len, 8);
        assert_eq!(dataset.n_features, 3);
    }

    #[test]
    fn test_dataset_not_enough_data() {
        let (features, targets) = create_test_data(10);
        let result = TimeSeriesDataset::new(&features, &targets, 24, 8);

        assert!(result.is_err());
    }

    #[test]
    fn test_dataset_split() {
        let (features, targets) = create_test_data(100);
        let dataset = TimeSeriesDataset::new(&features, &targets, 10, 5).unwrap();

        let (train, val, test) = dataset.split(0.7, 0.15);

        assert!(train.n_samples > 0);
        assert!(val.n_samples > 0);
        assert!(test.n_samples > 0);
        assert_eq!(
            train.n_samples + val.n_samples + test.n_samples,
            dataset.n_samples
        );
    }

    #[test]
    fn test_batch_iterator() {
        let (features, targets) = create_test_data(100);
        let dataset = TimeSeriesDataset::new(&features, &targets, 10, 5).unwrap();

        let batch_size = 8;
        let mut total_samples = 0;

        for (x, y) in dataset.batches(batch_size, false) {
            assert!(x.dim().0 <= batch_size);
            assert_eq!(x.dim().1, dataset.seq_len);
            assert_eq!(x.dim().2, dataset.n_features);
            assert!(y.dim().0 <= batch_size);
            assert_eq!(y.dim().1, dataset.pred_len);

            total_samples += x.dim().0;
        }

        assert_eq!(total_samples, dataset.n_samples);
    }

    #[test]
    fn test_get_batch() {
        let (features, targets) = create_test_data(50);
        let dataset = TimeSeriesDataset::new(&features, &targets, 10, 5).unwrap();

        let indices = vec![0, 5, 10];
        let (x, y) = dataset.get_batch(&indices);

        assert_eq!(x.dim(), (3, 10, 3));
        assert_eq!(y.dim(), (3, 5));
    }
}
