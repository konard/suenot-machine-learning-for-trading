//! # Risk Factors Analysis
//!
//! Модуль для анализа риск-факторов на основе латентного представления
//! автоэнкодера. Помогает интерпретировать скрытые факторы.

use crate::autoencoder::Autoencoder;
use crate::data_processor::Features;
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Риск-фактор, извлеченный автоэнкодером
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    /// Индекс фактора в латентном пространстве
    pub index: usize,
    /// Название фактора (если определено)
    pub name: String,
    /// Объясненная дисперсия (приблизительно)
    pub explained_variance: f64,
    /// Корреляции с исходными признаками
    pub feature_correlations: HashMap<String, f64>,
    /// Статистики фактора
    pub statistics: FactorStatistics,
}

/// Статистики для риск-фактора
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorStatistics {
    /// Среднее значение
    pub mean: f64,
    /// Стандартное отклонение
    pub std: f64,
    /// Минимум
    pub min: f64,
    /// Максимум
    pub max: f64,
    /// Асимметрия (skewness)
    pub skewness: f64,
    /// Эксцесс (kurtosis)
    pub kurtosis: f64,
}

impl FactorStatistics {
    /// Вычисляет статистики для вектора значений
    pub fn from_values(values: &[f64]) -> Self {
        let n = values.len() as f64;
        if n < 2.0 {
            return Self {
                mean: values.first().copied().unwrap_or(0.0),
                std: 0.0,
                min: values.first().copied().unwrap_or(0.0),
                max: values.first().copied().unwrap_or(0.0),
                skewness: 0.0,
                kurtosis: 0.0,
            };
        }

        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = variance.sqrt();

        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Skewness
        let skewness = if std > 1e-10 {
            let m3 = values.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>();
            m3 / n
        } else {
            0.0
        };

        // Kurtosis (excess kurtosis)
        let kurtosis = if std > 1e-10 {
            let m4 = values.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>();
            m4 / n - 3.0
        } else {
            0.0
        };

        Self {
            mean,
            std,
            min,
            max,
            skewness,
            kurtosis,
        }
    }
}

/// Анализатор риск-факторов
pub struct RiskFactorAnalyzer {
    /// Названия по умолчанию для факторов
    default_names: Vec<String>,
}

impl Default for RiskFactorAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl RiskFactorAnalyzer {
    /// Создает новый анализатор
    pub fn new() -> Self {
        Self {
            default_names: vec![
                "Market".to_string(),
                "Momentum".to_string(),
                "Volatility".to_string(),
                "Volume".to_string(),
                "Trend".to_string(),
                "Mean Reversion".to_string(),
                "Sentiment".to_string(),
                "Liquidity".to_string(),
            ],
        }
    }

    /// Анализирует латентное представление и извлекает риск-факторы
    pub fn analyze(
        &self,
        autoencoder: &mut Autoencoder,
        features: &Features,
    ) -> Vec<RiskFactor> {
        let data = features.to_array();
        let latent = autoencoder.transform(&data);

        let n_factors = latent.ncols();
        let mut risk_factors = Vec::with_capacity(n_factors);

        // Вычисляем общую дисперсию исходных данных
        let total_variance: f64 = data
            .axis_iter(Axis(1))
            .map(|col| {
                let col_vec: Vec<f64> = col.to_vec();
                variance(&col_vec)
            })
            .sum();

        for factor_idx in 0..n_factors {
            let factor_values: Vec<f64> = latent.column(factor_idx).to_vec();

            // Статистики
            let statistics = FactorStatistics::from_values(&factor_values);

            // Корреляции с исходными признаками
            let mut feature_correlations = HashMap::new();
            for (feat_idx, feat_name) in features.names.iter().enumerate() {
                let feat_values: Vec<f64> = data.column(feat_idx).to_vec();
                let corr = correlation(&factor_values, &feat_values);
                feature_correlations.insert(feat_name.clone(), corr);
            }

            // Приблизительная объясненная дисперсия
            let factor_variance = variance(&factor_values);
            let explained_variance = if total_variance > 0.0 {
                factor_variance / total_variance
            } else {
                0.0
            };

            // Название фактора
            let name = self.infer_factor_name(factor_idx, &feature_correlations);

            risk_factors.push(RiskFactor {
                index: factor_idx,
                name,
                explained_variance,
                feature_correlations,
                statistics,
            });
        }

        // Сортируем по объясненной дисперсии
        risk_factors.sort_by(|a, b| {
            b.explained_variance
                .partial_cmp(&a.explained_variance)
                .unwrap()
        });

        risk_factors
    }

    /// Пытается определить название фактора по корреляциям
    fn infer_factor_name(&self, index: usize, correlations: &HashMap<String, f64>) -> String {
        // Находим признак с максимальной абсолютной корреляцией
        let max_corr_feature = correlations
            .iter()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(name, _)| name.as_str())
            .unwrap_or("");

        // Определяем тип фактора по ключевым словам
        let factor_type = if max_corr_feature.contains("volatil") {
            "Volatility"
        } else if max_corr_feature.contains("volume") {
            "Volume/Liquidity"
        } else if max_corr_feature.contains("momentum") || max_corr_feature.contains("return") {
            "Momentum"
        } else if max_corr_feature.contains("rsi") || max_corr_feature.contains("bb_") {
            "Mean Reversion"
        } else if max_corr_feature.contains("sma") || max_corr_feature.contains("ema") {
            "Trend"
        } else if max_corr_feature.contains("macd") {
            "MACD Signal"
        } else {
            self.default_names
                .get(index)
                .map(String::as_str)
                .unwrap_or("Unknown")
        };

        format!("Factor {} ({})", index + 1, factor_type)
    }

    /// Вычисляет важность каждого признака для реконструкции
    pub fn feature_importance(
        &self,
        autoencoder: &mut Autoencoder,
        features: &Features,
    ) -> HashMap<String, f64> {
        let data = features.to_array();
        let reconstructed = autoencoder.inverse_transform(&autoencoder.transform(&data));

        let mut importance = HashMap::new();

        for (idx, name) in features.names.iter().enumerate() {
            let original: Vec<f64> = data.column(idx).to_vec();
            let recon: Vec<f64> = reconstructed.column(idx).to_vec();

            // Важность = 1 - normalized_mse
            let mse: f64 = original
                .iter()
                .zip(recon.iter())
                .map(|(o, r)| (o - r).powi(2))
                .sum::<f64>()
                / original.len() as f64;

            let original_var = variance(&original);
            let importance_score = if original_var > 0.0 {
                1.0 - (mse / original_var).min(1.0)
            } else {
                0.0
            };

            importance.insert(name.clone(), importance_score);
        }

        importance
    }

    /// Анализирует временную динамику факторов
    pub fn temporal_analysis(
        &self,
        latent_data: &Array2<f64>,
        window_size: usize,
    ) -> Vec<TemporalFactorStats> {
        let n_samples = latent_data.nrows();
        let n_factors = latent_data.ncols();

        if n_samples < window_size {
            return vec![];
        }

        let n_windows = n_samples - window_size + 1;
        let mut temporal_stats = Vec::with_capacity(n_factors);

        for factor_idx in 0..n_factors {
            let factor_values: Vec<f64> = latent_data.column(factor_idx).to_vec();

            let mut rolling_means = Vec::with_capacity(n_windows);
            let mut rolling_stds = Vec::with_capacity(n_windows);

            for i in 0..n_windows {
                let window = &factor_values[i..i + window_size];
                rolling_means.push(mean(window));
                rolling_stds.push(std_dev(window));
            }

            // Автокорреляция с лагом 1
            let autocorr = if factor_values.len() > 1 {
                let lagged: Vec<f64> = factor_values[..factor_values.len() - 1].to_vec();
                let current: Vec<f64> = factor_values[1..].to_vec();
                correlation(&lagged, &current)
            } else {
                0.0
            };

            // Тренд (линейная регрессия)
            let trend = linear_trend(&factor_values);

            temporal_stats.push(TemporalFactorStats {
                factor_index: factor_idx,
                rolling_means,
                rolling_stds,
                autocorrelation: autocorr,
                trend,
            });
        }

        temporal_stats
    }

    /// Кластеризует временные точки по латентному представлению
    pub fn cluster_regimes(
        &self,
        latent_data: &Array2<f64>,
        n_clusters: usize,
    ) -> Vec<usize> {
        // Простая k-means кластеризация
        let n_samples = latent_data.nrows();
        let n_features = latent_data.ncols();

        if n_samples < n_clusters {
            return (0..n_samples).map(|_| 0).collect();
        }

        // Инициализация центроидов (k-means++)
        let mut centroids = Vec::with_capacity(n_clusters);
        let mut rng = rand::thread_rng();

        // Первый центроид - случайная точка
        let first_idx = rand::Rng::gen_range(&mut rng, 0..n_samples);
        centroids.push(latent_data.row(first_idx).to_owned());

        // Остальные центроиды с вероятностью пропорциональной расстоянию
        for _ in 1..n_clusters {
            let mut distances: Vec<f64> = Vec::with_capacity(n_samples);
            for sample in latent_data.outer_iter() {
                let min_dist = centroids
                    .iter()
                    .map(|c| euclidean_distance(&sample.to_owned(), c))
                    .fold(f64::INFINITY, f64::min);
                distances.push(min_dist.powi(2));
            }

            let total: f64 = distances.iter().sum();
            let mut cumsum = 0.0;
            let threshold = rand::Rng::gen::<f64>(&mut rng) * total;

            for (i, &d) in distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    centroids.push(latent_data.row(i).to_owned());
                    break;
                }
            }
        }

        // K-means итерации
        let mut labels = vec![0; n_samples];
        let max_iterations = 100;

        for _ in 0..max_iterations {
            let old_labels = labels.clone();

            // Присваиваем метки
            for (i, sample) in latent_data.outer_iter().enumerate() {
                let mut min_dist = f64::INFINITY;
                for (j, centroid) in centroids.iter().enumerate() {
                    let dist = euclidean_distance(&sample.to_owned(), centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        labels[i] = j;
                    }
                }
            }

            // Обновляем центроиды
            for (j, centroid) in centroids.iter_mut().enumerate() {
                let cluster_points: Vec<_> = latent_data
                    .outer_iter()
                    .enumerate()
                    .filter(|(i, _)| labels[*i] == j)
                    .map(|(_, row)| row.to_owned())
                    .collect();

                if !cluster_points.is_empty() {
                    *centroid = Array1::zeros(n_features);
                    for point in &cluster_points {
                        *centroid = &*centroid + point;
                    }
                    *centroid = &*centroid / cluster_points.len() as f64;
                }
            }

            // Проверяем сходимость
            if labels == old_labels {
                break;
            }
        }

        labels
    }
}

/// Статистики временной динамики фактора
#[derive(Debug, Clone)]
pub struct TemporalFactorStats {
    /// Индекс фактора
    pub factor_index: usize,
    /// Скользящие средние
    pub rolling_means: Vec<f64>,
    /// Скользящие стандартные отклонения
    pub rolling_stds: Vec<f64>,
    /// Автокорреляция
    pub autocorrelation: f64,
    /// Тренд (наклон линейной регрессии)
    pub trend: f64,
}

// ============ Вспомогательные функции ============

/// Среднее значение
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Дисперсия
fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64
}

/// Стандартное отклонение
fn std_dev(values: &[f64]) -> f64 {
    variance(values).sqrt()
}

/// Корреляция Пирсона
fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let mean_x = mean(x);
    let mean_y = mean(y);

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        cov / denom
    }
}

/// Евклидово расстояние
fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    (a - b).mapv(|x| x.powi(2)).sum().sqrt()
}

/// Линейный тренд (наклон)
fn linear_trend(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let x: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();
    let mean_x = (n - 1.0) / 2.0;
    let mean_y = mean(values);

    let mut num = 0.0;
    let mut den = 0.0;

    for (i, y) in values.iter().enumerate() {
        let dx = i as f64 - mean_x;
        let dy = y - mean_y;
        num += dx * dy;
        den += dx * dx;
    }

    if den.abs() < 1e-10 {
        0.0
    } else {
        num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_statistics() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FactorStatistics::from_values(&values);

        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10); // Perfect correlation
    }

    #[test]
    fn test_linear_trend() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trend = linear_trend(&values);
        assert!((trend - 1.0).abs() < 1e-10); // Slope of 1
    }
}
