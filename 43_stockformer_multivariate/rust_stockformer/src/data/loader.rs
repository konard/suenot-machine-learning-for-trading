//! Загрузчик мультивариантных данных для Stockformer

use crate::api::Kline;
use super::features::{Features, FeatureConfig, calculate_features};
use super::dataset::{MultiAssetDataset, CorrelationMatrix};
use anyhow::Result;
use ndarray::{Array2, Array3, Array4};
use std::collections::HashMap;
use tracing::{info, warn};

/// Загрузчик мультивариантных данных
pub struct MultiAssetLoader {
    /// Список символов активов
    pub symbols: Vec<String>,
    /// Конфигурация признаков
    pub feature_config: FeatureConfig,
}

impl MultiAssetLoader {
    /// Создаёт новый загрузчик
    pub fn new(symbols: Vec<String>) -> Self {
        Self {
            symbols,
            feature_config: FeatureConfig::default(),
        }
    }

    /// Создаёт загрузчик с кастомной конфигурацией признаков
    pub fn with_config(symbols: Vec<String>, config: FeatureConfig) -> Self {
        Self {
            symbols,
            feature_config: config,
        }
    }

    /// Количество активов
    pub fn n_assets(&self) -> usize {
        self.symbols.len()
    }

    /// Подготавливает датасет из данных klines
    ///
    /// # Arguments
    ///
    /// * `klines_map` - HashMap с klines для каждого символа
    /// * `encoder_length` - Длина входной последовательности
    /// * `prediction_length` - Горизонт прогнозирования
    ///
    /// # Returns
    ///
    /// Мультивариантный датасет для обучения Stockformer
    pub fn prepare_dataset(
        &self,
        klines_map: &HashMap<String, Vec<Kline>>,
        encoder_length: usize,
        prediction_length: usize,
    ) -> Result<MultiAssetDataset> {
        // Проверяем наличие данных для всех символов
        for symbol in &self.symbols {
            if !klines_map.contains_key(symbol) {
                anyhow::bail!("Missing data for symbol: {}", symbol);
            }
        }

        // Вычисляем признаки для каждого актива
        let mut features_map: HashMap<String, Features> = HashMap::new();
        for symbol in &self.symbols {
            let klines = &klines_map[symbol];
            let features = calculate_features(klines, &self.feature_config);
            features_map.insert(symbol.clone(), features);
        }

        // Находим общий временной диапазон
        let min_len = features_map
            .values()
            .map(|f| f.len())
            .min()
            .unwrap_or(0);

        if min_len < encoder_length + prediction_length {
            anyhow::bail!(
                "Not enough data: need {} samples, have {}",
                encoder_length + prediction_length,
                min_len
            );
        }

        // Выравниваем данные по времени и создаём 3D массив
        // Shape: [time_steps, n_assets, n_features]
        let n_features = features_map.values().next().map(|f| f.n_features()).unwrap_or(0);
        let n_assets = self.symbols.len();

        let mut data = Array3::zeros((min_len, n_assets, n_features));

        for (i, symbol) in self.symbols.iter().enumerate() {
            let features = &features_map[symbol];
            let values = features.values.slice(ndarray::s![..min_len, ..]);
            for t in 0..min_len {
                for f in 0..n_features {
                    data[[t, i, f]] = values[[t, f]];
                }
            }
        }

        // Получаем цены закрытия для целевых значений
        let mut closes = Array2::zeros((min_len, n_assets));
        for (i, symbol) in self.symbols.iter().enumerate() {
            let klines = &klines_map[symbol];
            for t in 0..min_len {
                closes[[t, i]] = klines[t].close;
            }
        }

        // Получаем временные метки
        let timestamps: Vec<u64> = features_map
            .values()
            .next()
            .map(|f| f.timestamps[..min_len].to_vec())
            .unwrap_or_default();

        // Вычисляем матрицу корреляций
        let correlation_matrix = self.compute_correlation_matrix(&closes);

        // Создаём последовательности X и y
        let n_samples = min_len - encoder_length - prediction_length + 1;

        // X: [n_samples, encoder_length, n_assets, n_features]
        let mut x = Array4::zeros((n_samples, encoder_length, n_assets, n_features));

        // y: [n_samples, n_assets] - целевые доходности
        let mut y = Array2::zeros((n_samples, n_assets));

        for i in 0..n_samples {
            // Входная последовательность
            for t in 0..encoder_length {
                for a in 0..n_assets {
                    for f in 0..n_features {
                        x[[i, t, a, f]] = data[[i + t, a, f]];
                    }
                }
            }

            // Целевые значения - логарифмическая доходность
            let start_idx = i + encoder_length - 1;
            let end_idx = i + encoder_length + prediction_length - 1;

            for a in 0..n_assets {
                let start_price = closes[[start_idx, a]];
                let end_price = closes[[end_idx, a]];
                if start_price > 0.0 {
                    y[[i, a]] = (end_price / start_price).ln();
                }
            }
        }

        // Получаем имена признаков
        let feature_names: Vec<String> = features_map
            .values()
            .next()
            .map(|f| f.names.clone())
            .unwrap_or_default();

        info!(
            "Prepared dataset: {} samples, {} assets, {} features",
            n_samples, n_assets, n_features
        );

        Ok(MultiAssetDataset {
            x,
            y,
            symbols: self.symbols.clone(),
            feature_names,
            timestamps,
            correlation_matrix,
            encoder_length,
            prediction_length,
        })
    }

    /// Вычисляет матрицу корреляций между активами
    fn compute_correlation_matrix(&self, closes: &Array2<f64>) -> CorrelationMatrix {
        use super::features::pearson_correlation;

        let n_assets = closes.ncols();
        let mut matrix = Array2::zeros((n_assets, n_assets));

        for i in 0..n_assets {
            let returns_i: Vec<f64> = (1..closes.nrows())
                .map(|t| {
                    if closes[[t - 1, i]] > 0.0 {
                        (closes[[t, i]] / closes[[t - 1, i]]).ln()
                    } else {
                        0.0
                    }
                })
                .collect();

            for j in 0..n_assets {
                if i == j {
                    matrix[[i, j]] = 1.0;
                } else if j > i {
                    let returns_j: Vec<f64> = (1..closes.nrows())
                        .map(|t| {
                            if closes[[t - 1, j]] > 0.0 {
                                (closes[[t, j]] / closes[[t - 1, j]]).ln()
                            } else {
                                0.0
                            }
                        })
                        .collect();

                    let corr = pearson_correlation(&returns_i, &returns_j);
                    matrix[[i, j]] = corr;
                    matrix[[j, i]] = corr;
                }
            }
        }

        CorrelationMatrix {
            matrix,
            symbols: self.symbols.clone(),
        }
    }

    /// Разделяет данные на train/validation/test
    pub fn split_data(
        &self,
        dataset: &MultiAssetDataset,
        train_ratio: f64,
        val_ratio: f64,
    ) -> (MultiAssetDataset, MultiAssetDataset, MultiAssetDataset) {
        let n_samples = dataset.x.shape()[0];
        let train_end = (n_samples as f64 * train_ratio) as usize;
        let val_end = (n_samples as f64 * (train_ratio + val_ratio)) as usize;

        let train = dataset.slice(0, train_end);
        let val = dataset.slice(train_end, val_end);
        let test = dataset.slice(val_end, n_samples);

        info!(
            "Split data: train={}, val={}, test={}",
            train.x.shape()[0],
            val.x.shape()[0],
            test.x.shape()[0]
        );

        (train, val, test)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| Kline {
                start_time: i as u64 * 3600000,
                open: 100.0 + (i as f64) * 0.5,
                high: 101.0 + (i as f64) * 0.5,
                low: 99.0 + (i as f64) * 0.5,
                close: 100.5 + (i as f64) * 0.5,
                volume: 1000.0 + (i as f64) * 10.0,
                turnover: 100000.0,
            })
            .collect()
    }

    #[test]
    fn test_prepare_dataset() {
        let symbols = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()];
        let loader = MultiAssetLoader::new(symbols.clone());

        let mut klines_map = HashMap::new();
        klines_map.insert("BTCUSDT".to_string(), create_test_klines(200));
        klines_map.insert("ETHUSDT".to_string(), create_test_klines(200));

        let dataset = loader.prepare_dataset(&klines_map, 24, 12).unwrap();

        assert_eq!(dataset.symbols.len(), 2);
        assert_eq!(dataset.x.shape()[1], 24); // encoder_length
        assert_eq!(dataset.x.shape()[2], 2);  // n_assets
    }

    #[test]
    fn test_correlation_matrix() {
        let symbols = vec!["A".to_string(), "B".to_string()];
        let loader = MultiAssetLoader::new(symbols);

        // Создаём идентичные серии цен
        let closes = Array2::from_shape_fn((100, 2), |(i, j)| 100.0 + i as f64);

        let corr = loader.compute_correlation_matrix(&closes);

        // Корреляция идентичных серий должна быть 1
        assert!((corr.matrix[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((corr.matrix[[0, 1]] - 1.0).abs() < 1e-5);
    }
}
