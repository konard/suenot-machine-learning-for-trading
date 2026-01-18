//! Feature engineering для Stockformer
//!
//! Вычисление технических индикаторов и признаков для каждого актива.

use crate::api::Kline;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Конфигурация вычисления признаков
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Период для скользящего среднего
    pub ma_period: usize,
    /// Период для волатильности
    pub volatility_period: usize,
    /// Период для RSI
    pub rsi_period: usize,
    /// Включить логарифмическую доходность
    pub include_log_return: bool,
    /// Включить изменение объёма
    pub include_volume_change: bool,
    /// Включить волатильность
    pub include_volatility: bool,
    /// Включить RSI
    pub include_rsi: bool,
    /// Включить нормализованный объём
    pub include_normalized_volume: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            ma_period: 20,
            volatility_period: 20,
            rsi_period: 14,
            include_log_return: true,
            include_volume_change: true,
            include_volatility: true,
            include_rsi: true,
            include_normalized_volume: true,
        }
    }
}

/// Структура для хранения вычисленных признаков
#[derive(Debug, Clone)]
pub struct Features {
    /// Названия признаков
    pub names: Vec<String>,
    /// Матрица признаков [time_steps, n_features]
    pub values: Array2<f64>,
    /// Временные метки
    pub timestamps: Vec<u64>,
}

impl Features {
    /// Количество временных шагов
    pub fn len(&self) -> usize {
        self.values.nrows()
    }

    /// Количество признаков
    pub fn n_features(&self) -> usize {
        self.values.ncols()
    }

    /// Проверка на пустоту
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Получает срез признаков по индексам времени
    pub fn slice(&self, start: usize, end: usize) -> Array2<f64> {
        self.values.slice(ndarray::s![start..end, ..]).to_owned()
    }
}

/// Вычисляет признаки для одного актива
pub fn calculate_features(klines: &[Kline], config: &FeatureConfig) -> Features {
    let n = klines.len();
    if n == 0 {
        return Features {
            names: vec![],
            values: Array2::zeros((0, 0)),
            timestamps: vec![],
        };
    }

    let mut feature_vecs: Vec<(&str, Vec<f64>)> = Vec::new();

    // Извлекаем базовые данные
    let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();
    let timestamps: Vec<u64> = klines.iter().map(|k| k.start_time).collect();

    // Логарифмическая доходность
    if config.include_log_return {
        let log_returns = log_returns(&closes);
        feature_vecs.push(("log_return", log_returns));
    }

    // Изменение объёма относительно MA
    if config.include_volume_change {
        let vol_change = volume_change(&volumes, config.ma_period);
        feature_vecs.push(("volume_change", vol_change));
    }

    // Волатильность (скользящее std доходностей)
    if config.include_volatility {
        let log_rets = log_returns(&closes);
        let vol = rolling_std(&log_rets, config.volatility_period);
        feature_vecs.push(("volatility", vol));
    }

    // RSI
    if config.include_rsi {
        let rsi = calculate_rsi(&closes, config.rsi_period);
        feature_vecs.push(("rsi", rsi));
    }

    // Нормализованный объём
    if config.include_normalized_volume {
        let norm_vol = normalize_series(&volumes);
        feature_vecs.push(("normalized_volume", norm_vol));
    }

    // Собираем в матрицу
    let names: Vec<String> = feature_vecs.iter().map(|(name, _)| name.to_string()).collect();
    let n_features = feature_vecs.len();

    let mut values = Array2::zeros((n, n_features));
    for (j, (_, vec)) in feature_vecs.iter().enumerate() {
        for (i, &val) in vec.iter().enumerate() {
            values[[i, j]] = val;
        }
    }

    Features {
        names,
        values,
        timestamps,
    }
}

/// Вычисляет логарифмическую доходность
fn log_returns(prices: &[f64]) -> Vec<f64> {
    let mut returns = vec![0.0];
    for i in 1..prices.len() {
        if prices[i - 1] > 0.0 {
            returns.push((prices[i] / prices[i - 1]).ln());
        } else {
            returns.push(0.0);
        }
    }
    returns
}

/// Вычисляет изменение объёма относительно скользящего среднего
fn volume_change(volumes: &[f64], period: usize) -> Vec<f64> {
    let ma = rolling_mean(volumes, period);
    volumes
        .iter()
        .zip(ma.iter())
        .map(|(&v, &m)| if m > 0.0 { v / m } else { 1.0 })
        .collect()
}

/// Скользящее среднее
fn rolling_mean(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];

    let mut sum = 0.0;
    for i in 0..n {
        sum += data[i];
        if i >= period {
            sum -= data[i - period];
            result[i] = sum / period as f64;
        } else {
            result[i] = sum / (i + 1) as f64;
        }
    }
    result
}

/// Скользящее стандартное отклонение
fn rolling_std(data: &[f64], period: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let start = if i >= period { i - period + 1 } else { 0 };
        let window: Vec<f64> = data[start..=i].to_vec();
        result[i] = std_dev(&window);
    }
    result
}

/// Стандартное отклонение
fn std_dev(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

/// Вычисляет RSI (Relative Strength Index)
fn calculate_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut rsi = vec![50.0; n]; // Нейтральное значение по умолчанию

    if n < 2 {
        return rsi;
    }

    // Вычисляем изменения цен
    let changes: Vec<f64> = prices
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect();

    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    // Инициализация - первый период
    for &change in changes.iter().take(period.min(changes.len())) {
        if change > 0.0 {
            avg_gain += change;
        } else {
            avg_loss += change.abs();
        }
    }

    let period_f64 = period as f64;
    avg_gain /= period_f64;
    avg_loss /= period_f64;

    // Вычисляем RSI
    for i in period..changes.len() {
        let change = changes[i];
        let (gain, loss) = if change > 0.0 {
            (change, 0.0)
        } else {
            (0.0, change.abs())
        };

        // Сглаженное среднее
        avg_gain = (avg_gain * (period_f64 - 1.0) + gain) / period_f64;
        avg_loss = (avg_loss * (period_f64 - 1.0) + loss) / period_f64;

        if avg_loss > 0.0 {
            let rs = avg_gain / avg_loss;
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs));
        } else if avg_gain > 0.0 {
            rsi[i + 1] = 100.0;
        } else {
            rsi[i + 1] = 50.0;
        }
    }

    // Нормализуем RSI к диапазону [0, 1]
    rsi.iter().map(|&r| r / 100.0).collect()
}

/// Нормализует серию (z-score)
fn normalize_series(data: &[f64]) -> Vec<f64> {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let std = std_dev(data);

    if std > 0.0 {
        data.iter().map(|&x| (x - mean) / std).collect()
    } else {
        vec![0.0; data.len()]
    }
}

/// Вычисляет корреляцию между двумя сериями
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom > 0.0 {
        cov / denom
    } else {
        0.0
    }
}

/// Вычисляет скользящую корреляцию между двумя сериями
pub fn rolling_correlation(x: &[f64], y: &[f64], window: usize) -> Vec<f64> {
    let n = x.len().min(y.len());
    let mut result = vec![0.0; n];

    for i in window..n {
        let x_window = &x[i - window..i];
        let y_window = &y[i - window..i];
        result[i] = pearson_correlation(x_window, y_window);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 105.0, 103.0, 108.0];
        let returns = log_returns(&prices);

        assert_eq!(returns.len(), 4);
        assert!((returns[0] - 0.0).abs() < 1e-10);
        assert!((returns[1] - (105.0_f64 / 100.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = rolling_mean(&data, 3);

        assert_eq!(ma.len(), 5);
        // MA(3) at index 2 = (1+2+3)/3 = 2.0
        assert!((ma[2] - 2.0).abs() < 1e-10);
        // MA(3) at index 4 = (3+4+5)/3 = 4.0
        assert!((ma[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let corr = pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10); // Perfect positive correlation

        let y_neg = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr_neg = pearson_correlation(&x, &y_neg);
        assert!((corr_neg - (-1.0)).abs() < 1e-10); // Perfect negative correlation
    }

    #[test]
    fn test_rsi() {
        // Создаём тренд вверх
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let rsi = calculate_rsi(&prices, 14);

        // RSI должен быть высоким для восходящего тренда
        assert!(rsi.last().unwrap() > &0.5);
    }
}
