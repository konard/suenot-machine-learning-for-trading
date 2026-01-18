//! Структуры данных для мультивариантного датасета

use ndarray::{Array2, Array4};
use serde::{Deserialize, Serialize};

/// Матрица корреляций между активами
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    /// Матрица корреляций [n_assets, n_assets]
    pub matrix: Array2<f64>,
    /// Названия активов
    pub symbols: Vec<String>,
}

impl CorrelationMatrix {
    /// Получает корреляцию между двумя активами
    pub fn get(&self, symbol1: &str, symbol2: &str) -> Option<f64> {
        let idx1 = self.symbols.iter().position(|s| s == symbol1)?;
        let idx2 = self.symbols.iter().position(|s| s == symbol2)?;
        Some(self.matrix[[idx1, idx2]])
    }

    /// Возвращает наиболее коррелированные пары
    pub fn top_correlations(&self, n: usize) -> Vec<(String, String, f64)> {
        let mut correlations = Vec::new();

        for i in 0..self.symbols.len() {
            for j in (i + 1)..self.symbols.len() {
                correlations.push((
                    self.symbols[i].clone(),
                    self.symbols[j].clone(),
                    self.matrix[[i, j]],
                ));
            }
        }

        correlations.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
        correlations.into_iter().take(n).collect()
    }

    /// Выводит матрицу корреляций в красивом формате
    pub fn to_string(&self) -> String {
        let mut result = String::new();

        // Заголовок
        result.push_str("        ");
        for symbol in &self.symbols {
            result.push_str(&format!("{:>8}", &symbol[..symbol.len().min(8)]));
        }
        result.push('\n');

        // Данные
        for (i, symbol) in self.symbols.iter().enumerate() {
            result.push_str(&format!("{:<8}", &symbol[..symbol.len().min(8)]));
            for j in 0..self.symbols.len() {
                result.push_str(&format!("{:>8.3}", self.matrix[[i, j]]));
            }
            result.push('\n');
        }

        result
    }
}

/// Мультивариантный датасет для Stockformer
#[derive(Debug, Clone)]
pub struct MultiAssetDataset {
    /// Входные данные: [n_samples, encoder_length, n_assets, n_features]
    pub x: Array4<f64>,

    /// Целевые значения: [n_samples, n_assets] (логарифмические доходности)
    pub y: Array2<f64>,

    /// Названия активов
    pub symbols: Vec<String>,

    /// Названия признаков
    pub feature_names: Vec<String>,

    /// Временные метки (начало каждого периода)
    pub timestamps: Vec<u64>,

    /// Матрица корреляций
    pub correlation_matrix: CorrelationMatrix,

    /// Длина encoder context
    pub encoder_length: usize,

    /// Длина prediction horizon
    pub prediction_length: usize,
}

impl MultiAssetDataset {
    /// Количество примеров в датасете
    pub fn len(&self) -> usize {
        self.x.shape()[0]
    }

    /// Проверка на пустоту
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Количество активов
    pub fn n_assets(&self) -> usize {
        self.x.shape()[2]
    }

    /// Количество признаков
    pub fn n_features(&self) -> usize {
        self.x.shape()[3]
    }

    /// Получает один пример (x, y)
    pub fn get(&self, idx: usize) -> (Array2<f64>, Array2<f64>) {
        let x_sample = self.x.slice(ndarray::s![idx, .., .., ..]);
        let y_sample = self.y.slice(ndarray::s![idx, ..]);

        // Reshape x to [encoder_length * n_assets, n_features]
        let encoder_len = self.encoder_length;
        let n_assets = self.n_assets();
        let n_features = self.n_features();

        let mut x_2d = Array2::zeros((encoder_len, n_assets * n_features));
        for t in 0..encoder_len {
            for a in 0..n_assets {
                for f in 0..n_features {
                    x_2d[[t, a * n_features + f]] = x_sample[[t, a, f]];
                }
            }
        }

        let y_2d = y_sample.to_owned().insert_axis(ndarray::Axis(0));

        (x_2d, y_2d)
    }

    /// Создаёт срез датасета
    pub fn slice(&self, start: usize, end: usize) -> Self {
        let x = self.x.slice(ndarray::s![start..end, .., .., ..]).to_owned();
        let y = self.y.slice(ndarray::s![start..end, ..]).to_owned();

        Self {
            x,
            y,
            symbols: self.symbols.clone(),
            feature_names: self.feature_names.clone(),
            timestamps: if start < self.timestamps.len() && end <= self.timestamps.len() {
                self.timestamps[start..end].to_vec()
            } else {
                vec![]
            },
            correlation_matrix: self.correlation_matrix.clone(),
            encoder_length: self.encoder_length,
            prediction_length: self.prediction_length,
        }
    }

    /// Создаёт батч данных
    pub fn get_batch(&self, indices: &[usize]) -> (Array4<f64>, Array2<f64>) {
        let batch_size = indices.len();
        let encoder_len = self.encoder_length;
        let n_assets = self.n_assets();
        let n_features = self.n_features();

        let mut x_batch = Array4::zeros((batch_size, encoder_len, n_assets, n_features));
        let mut y_batch = Array2::zeros((batch_size, n_assets));

        for (i, &idx) in indices.iter().enumerate() {
            for t in 0..encoder_len {
                for a in 0..n_assets {
                    for f in 0..n_features {
                        x_batch[[i, t, a, f]] = self.x[[idx, t, a, f]];
                    }
                }
            }
            for a in 0..n_assets {
                y_batch[[i, a]] = self.y[[idx, a]];
            }
        }

        (x_batch, y_batch)
    }

    /// Итератор по батчам
    pub fn iter_batches(&self, batch_size: usize) -> impl Iterator<Item = (Array4<f64>, Array2<f64>)> + '_ {
        (0..self.len())
            .step_by(batch_size)
            .map(move |start| {
                let end = (start + batch_size).min(self.len());
                let indices: Vec<usize> = (start..end).collect();
                self.get_batch(&indices)
            })
    }

    /// Возвращает статистику датасета
    pub fn stats(&self) -> DatasetStats {
        let n_samples = self.len();
        let n_assets = self.n_assets();

        // Статистика целевых значений
        let mut y_means = vec![0.0; n_assets];
        let mut y_stds = vec![0.0; n_assets];

        for a in 0..n_assets {
            let values: Vec<f64> = (0..n_samples).map(|i| self.y[[i, a]]).collect();
            let mean = values.iter().sum::<f64>() / n_samples as f64;
            let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;

            y_means[a] = mean;
            y_stds[a] = variance.sqrt();
        }

        DatasetStats {
            n_samples,
            n_assets,
            n_features: self.n_features(),
            encoder_length: self.encoder_length,
            prediction_length: self.prediction_length,
            y_means,
            y_stds,
        }
    }
}

/// Статистика датасета
#[derive(Debug, Clone)]
pub struct DatasetStats {
    pub n_samples: usize,
    pub n_assets: usize,
    pub n_features: usize,
    pub encoder_length: usize,
    pub prediction_length: usize,
    pub y_means: Vec<f64>,
    pub y_stds: Vec<f64>,
}

/// Параметры разделения данных
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSplit {
    pub train_ratio: f64,
    pub val_ratio: f64,
    pub test_ratio: f64,
}

impl Default for DataSplit {
    fn default() -> Self {
        Self {
            train_ratio: 0.7,
            val_ratio: 0.15,
            test_ratio: 0.15,
        }
    }
}

impl DataSplit {
    pub fn validate(&self) -> Result<(), String> {
        let sum = self.train_ratio + self.val_ratio + self.test_ratio;
        if (sum - 1.0).abs() > 0.001 {
            return Err(format!(
                "Split ratios must sum to 1.0, got {}",
                sum
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    #[test]
    fn test_correlation_matrix() {
        let symbols = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mut matrix = Array2::zeros((3, 3));

        // Заполняем диагональ единицами
        for i in 0..3 {
            matrix[[i, i]] = 1.0;
        }
        // Добавляем корреляции
        matrix[[0, 1]] = 0.8;
        matrix[[1, 0]] = 0.8;
        matrix[[0, 2]] = -0.5;
        matrix[[2, 0]] = -0.5;
        matrix[[1, 2]] = 0.3;
        matrix[[2, 1]] = 0.3;

        let corr = CorrelationMatrix { matrix, symbols };

        assert_eq!(corr.get("A", "B"), Some(0.8));
        assert_eq!(corr.get("A", "C"), Some(-0.5));

        let top = corr.top_correlations(2);
        assert_eq!(top[0].2, 0.8); // A-B
        assert_eq!(top[1].2.abs(), 0.5); // A-C
    }

    #[test]
    fn test_dataset_slice() {
        let x = Array4::zeros((100, 24, 3, 5));
        let y = Array2::zeros((100, 3));

        let dataset = MultiAssetDataset {
            x,
            y,
            symbols: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            feature_names: vec!["f1".to_string(), "f2".to_string()],
            timestamps: (0..100).map(|i| i as u64 * 3600000).collect(),
            correlation_matrix: CorrelationMatrix {
                matrix: Array2::eye(3),
                symbols: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            },
            encoder_length: 24,
            prediction_length: 12,
        };

        let slice = dataset.slice(0, 50);
        assert_eq!(slice.len(), 50);
        assert_eq!(slice.n_assets(), 3);
    }
}
