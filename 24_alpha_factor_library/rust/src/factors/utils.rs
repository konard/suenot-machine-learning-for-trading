//! Вспомогательные функции для расчёта факторов

/// Рассчитать изменения (returns)
pub fn returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .map(|w| {
            if w[0] == 0.0 {
                0.0
            } else {
                (w[1] - w[0]) / w[0]
            }
        })
        .collect()
}

/// Логарифмические доходности
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }

    prices
        .windows(2)
        .map(|w| {
            if w[0] <= 0.0 || w[1] <= 0.0 {
                0.0
            } else {
                (w[1] / w[0]).ln()
            }
        })
        .collect()
}

/// Скользящее окно с функцией агрегации
pub fn rolling<F>(data: &[f64], window: usize, func: F) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    if data.len() < window || window == 0 {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; window - 1];

    for i in (window - 1)..data.len() {
        let slice = &data[(i + 1 - window)..=i];
        result.push(func(slice));
    }

    result
}

/// Среднее значение
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

/// Стандартное отклонение (sample)
pub fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return f64::NAN;
    }

    let m = mean(data);
    let variance: f64 = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}

/// Стандартное отклонение (population)
pub fn std_dev_pop(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }

    let m = mean(data);
    let variance: f64 = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

/// Максимальное значение
pub fn max(data: &[f64]) -> f64 {
    data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

/// Минимальное значение
pub fn min(data: &[f64]) -> f64 {
    data.iter().cloned().fold(f64::INFINITY, f64::min)
}

/// Сумма
pub fn sum(data: &[f64]) -> f64 {
    data.iter().sum()
}

/// Корреляция между двумя массивами
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }

    let n = x.len() as f64;
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

    if var_x == 0.0 || var_y == 0.0 {
        return f64::NAN;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Ковариация
pub fn covariance(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }

    let mean_x = mean(x);
    let mean_y = mean(y);

    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();

    cov / (x.len() - 1) as f64
}

/// Ранжирование данных (от 1 до n)
pub fn rank(data: &[f64]) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }

    let mut indexed: Vec<(usize, f64)> = data.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; data.len()];
    for (rank, (original_idx, _)) in indexed.iter().enumerate() {
        ranks[*original_idx] = (rank + 1) as f64;
    }

    ranks
}

/// Нормализация в диапазон [0, 1]
pub fn normalize(data: &[f64]) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }

    let min_val = min(data);
    let max_val = max(data);
    let range = max_val - min_val;

    if range == 0.0 {
        return vec![0.5; data.len()];
    }

    data.iter().map(|x| (x - min_val) / range).collect()
}

/// Z-score нормализация
pub fn zscore(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return vec![f64::NAN; data.len()];
    }

    let m = mean(data);
    let s = std_dev(data);

    if s == 0.0 {
        return vec![0.0; data.len()];
    }

    data.iter().map(|x| (x - m) / s).collect()
}

/// Разница (diff) с шагом n
pub fn diff(data: &[f64], n: usize) -> Vec<f64> {
    if data.len() <= n || n == 0 {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; n];
    for i in n..data.len() {
        result.push(data[i] - data[i - n]);
    }

    result
}

/// Сдвиг данных на n периодов
pub fn shift(data: &[f64], n: usize) -> Vec<f64> {
    if n >= data.len() {
        return vec![f64::NAN; data.len()];
    }

    let mut result = vec![f64::NAN; n];
    result.extend_from_slice(&data[..data.len() - n]);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_returns() {
        let prices = vec![100.0, 110.0, 105.0];
        let rets = returns(&prices);
        assert_eq!(rets.len(), 2);
        assert!((rets[0] - 0.1).abs() < 0.0001);
        assert!((rets[1] - (-0.0454545)).abs() < 0.0001);
    }

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&data), 3.0);
    }

    #[test]
    fn test_std_dev() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((std_dev(&data) - 2.138).abs() < 0.001);
    }

    #[test]
    fn test_rank() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let ranks = rank(&data);
        assert_eq!(ranks[0], 3.0); // 3.0 is 3rd
        assert_eq!(ranks[2], 4.0); // 4.0 is 4th
        assert_eq!(ranks[4], 5.0); // 5.0 is 5th
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.0001); // Perfect positive correlation
    }

    #[test]
    fn test_zscore() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let z = zscore(&data);
        assert!((z[2]).abs() < 0.0001); // Middle value should have z-score near 0
    }
}
