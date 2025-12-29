//! Базовые статистические функции

use statrs::statistics::{Data, Distribution, OrderStatistics};

/// Вычислить среднее значение
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Вычислить дисперсию
pub fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let sum_sq: f64 = data.iter().map(|x| (x - m).powi(2)).sum();
    sum_sq / (data.len() - 1) as f64
}

/// Вычислить стандартное отклонение
pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Вычислить ковариацию
pub fn covariance(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let mean_x = mean(x);
    let mean_y = mean(y);
    let sum: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();
    sum / (x.len() - 1) as f64
}

/// Вычислить корреляцию Пирсона
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let cov = covariance(x, y);
    let std_x = std_dev(x);
    let std_y = std_dev(y);

    if std_x == 0.0 || std_y == 0.0 {
        return 0.0;
    }

    cov / (std_x * std_y)
}

/// Скользящее среднее
pub fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || data.len() < window {
        return Vec::new();
    }

    data.windows(window)
        .map(|w| w.iter().sum::<f64>() / window as f64)
        .collect()
}

/// Экспоненциальное скользящее среднее (EMA)
pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() || period == 0 {
        return Vec::new();
    }

    let alpha = 2.0 / (period + 1) as f64;
    let mut result = Vec::with_capacity(data.len());

    // Первое значение = первый элемент данных
    result.push(data[0]);

    for i in 1..data.len() {
        let prev = result[i - 1];
        let current = alpha * data[i] + (1.0 - alpha) * prev;
        result.push(current);
    }

    result
}

/// Скользящее стандартное отклонение
pub fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    if window < 2 || data.len() < window {
        return Vec::new();
    }

    data.windows(window).map(|w| std_dev(w)).collect()
}

/// Описательная статистика
#[derive(Debug, Clone)]
pub struct DescriptiveStats {
    pub count: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub q25: f64,
    pub median: f64,
    pub q75: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

impl DescriptiveStats {
    pub fn new(data: &[f64]) -> Self {
        if data.is_empty() {
            return Self {
                count: 0,
                mean: f64::NAN,
                std: f64::NAN,
                min: f64::NAN,
                max: f64::NAN,
                q25: f64::NAN,
                median: f64::NAN,
                q75: f64::NAN,
                skewness: f64::NAN,
                kurtosis: f64::NAN,
            };
        }

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut data_obj = Data::new(sorted.clone());

        let m = mean(data);
        let s = std_dev(data);

        // Skewness и Kurtosis
        let n = data.len() as f64;
        let skew = if s > 0.0 {
            data.iter().map(|x| ((x - m) / s).powi(3)).sum::<f64>() / n
        } else {
            0.0
        };

        let kurt = if s > 0.0 {
            data.iter().map(|x| ((x - m) / s).powi(4)).sum::<f64>() / n - 3.0
        } else {
            0.0
        };

        Self {
            count: data.len(),
            mean: m,
            std: s,
            min: *sorted.first().unwrap(),
            max: *sorted.last().unwrap(),
            q25: data_obj.percentile(25),
            median: data_obj.median(),
            q75: data_obj.percentile(75),
            skewness: skew,
            kurtosis: kurt,
        }
    }

    pub fn display(&self) -> String {
        format!(
            "Count: {}\nMean: {:.4}\nStd: {:.4}\nMin: {:.4}\n25%: {:.4}\n50%: {:.4}\n75%: {:.4}\nMax: {:.4}\nSkewness: {:.4}\nKurtosis: {:.4}",
            self.count, self.mean, self.std, self.min,
            self.q25, self.median, self.q75, self.max,
            self.skewness, self.kurtosis
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&data) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((correlation(&x, &y) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = moving_average(&data, 3);
        assert_eq!(ma.len(), 3);
        assert!((ma[0] - 2.0).abs() < 1e-10);
    }
}
