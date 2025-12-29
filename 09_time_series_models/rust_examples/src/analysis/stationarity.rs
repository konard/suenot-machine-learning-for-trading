//! Тесты на стационарность временных рядов

use crate::analysis::statistics::{mean, std_dev};
use crate::types::TestResult;
use nalgebra::{DMatrix, DVector};

/// Расширенный тест Дики-Фуллера (ADF)
///
/// Проверяет наличие единичного корня во временном ряде.
/// H0: ряд имеет единичный корень (нестационарен)
/// H1: ряд стационарен
pub fn adf_test(data: &[f64], max_lag: Option<usize>) -> TestResult {
    let n = data.len();
    if n < 10 {
        return TestResult {
            test_name: "ADF Test".to_string(),
            statistic: f64::NAN,
            p_value: 1.0,
            critical_values: vec![],
            is_significant: false,
        };
    }

    // Вычисляем первую разность
    let diff: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();

    // Определяем количество лагов
    let lag = max_lag.unwrap_or_else(|| ((n as f64).powf(1.0 / 3.0) * 2.0) as usize);
    let lag = lag.min(n / 4).max(1);

    // Строим регрессию: Δy_t = α + β*y_{t-1} + Σγ_i*Δy_{t-i} + ε_t
    let effective_n = n - 1 - lag;
    if effective_n < lag + 3 {
        return TestResult {
            test_name: "ADF Test".to_string(),
            statistic: f64::NAN,
            p_value: 1.0,
            critical_values: vec![],
            is_significant: false,
        };
    }

    // Зависимая переменная: Δy_t (начиная с lag+1)
    let y: Vec<f64> = diff[lag..].to_vec();

    // Матрица регрессоров
    // [1, y_{t-1}, Δy_{t-1}, Δy_{t-2}, ..., Δy_{t-lag}]
    let num_regressors = 2 + lag;
    let mut x_data = Vec::with_capacity(effective_n * num_regressors);

    for t in lag..diff.len() {
        // Константа
        x_data.push(1.0);
        // y_{t-1} (уровень)
        x_data.push(data[t]);
        // Лаги разностей
        for i in 1..=lag {
            x_data.push(diff[t - i]);
        }
    }

    let x = DMatrix::from_row_slice(effective_n, num_regressors, &x_data);
    let y_vec = DVector::from_vec(y);

    // OLS: β = (X'X)^(-1) X'y
    let xtx = x.transpose() * &x;
    let xty = x.transpose() * &y_vec;

    let xtx_inv = match xtx.clone().try_inverse() {
        Some(inv) => inv,
        None => {
            return TestResult {
                test_name: "ADF Test".to_string(),
                statistic: f64::NAN,
                p_value: 1.0,
                critical_values: vec![],
                is_significant: false,
            };
        }
    };

    let beta = &xtx_inv * xty;

    // Вычисляем остатки и стандартную ошибку
    let y_hat = &x * &beta;
    let residuals = &y_vec - y_hat;
    let sse: f64 = residuals.iter().map(|r| r * r).sum();
    let mse = sse / (effective_n - num_regressors) as f64;

    // Стандартная ошибка коэффициента при y_{t-1}
    let se_beta = (mse * xtx_inv[(1, 1)]).sqrt();

    // t-статистика
    let t_stat = beta[1] / se_beta;

    // Критические значения для ADF теста (приблизительные, с константой)
    let critical_values = vec![
        ("1%".to_string(), -3.43),
        ("5%".to_string(), -2.86),
        ("10%".to_string(), -2.57),
    ];

    // Приблизительное p-value (упрощённая интерполяция)
    let p_value = adf_p_value(t_stat, n);

    TestResult {
        test_name: "ADF Test".to_string(),
        statistic: t_stat,
        p_value,
        critical_values,
        is_significant: p_value < 0.05,
    }
}

/// Приблизительное p-value для ADF теста
fn adf_p_value(t_stat: f64, n: usize) -> f64 {
    // Упрощённая аппроксимация на основе критических значений
    // Более точные значения требуют таблиц MacKinnon
    let cv_1 = -3.43 - 6.0 / n as f64;
    let cv_5 = -2.86 - 4.0 / n as f64;
    let cv_10 = -2.57 - 3.0 / n as f64;

    if t_stat < cv_1 {
        0.01 * (cv_1 - t_stat).exp().recip()
    } else if t_stat < cv_5 {
        0.01 + (0.05 - 0.01) * (t_stat - cv_1) / (cv_5 - cv_1)
    } else if t_stat < cv_10 {
        0.05 + (0.10 - 0.05) * (t_stat - cv_5) / (cv_10 - cv_5)
    } else {
        0.10 + 0.90 * (1.0 - (-0.5 * (t_stat - cv_10)).exp())
    }
}

/// Тест KPSS (Kwiatkowski-Phillips-Schmidt-Shin)
///
/// H0: ряд стационарен
/// H1: ряд имеет единичный корень
pub fn kpss_test(data: &[f64], trend: bool) -> TestResult {
    let n = data.len();
    if n < 10 {
        return TestResult {
            test_name: "KPSS Test".to_string(),
            statistic: f64::NAN,
            p_value: 0.0,
            critical_values: vec![],
            is_significant: false,
        };
    }

    // Детрендируем данные
    let detrended = if trend {
        // Линейный тренд: y = a + b*t + residual
        let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let t_mean = mean(&t);
        let y_mean = mean(data);

        let num: f64 = t
            .iter()
            .zip(data.iter())
            .map(|(ti, yi)| (ti - t_mean) * (yi - y_mean))
            .sum();
        let den: f64 = t.iter().map(|ti| (ti - t_mean).powi(2)).sum();

        let b = num / den;
        let a = y_mean - b * t_mean;

        data.iter()
            .enumerate()
            .map(|(i, y)| y - a - b * i as f64)
            .collect::<Vec<_>>()
    } else {
        // Только демеанинг
        let m = mean(data);
        data.iter().map(|y| y - m).collect::<Vec<_>>()
    };

    // Частичные суммы остатков
    let mut partial_sums = Vec::with_capacity(n);
    let mut cumsum = 0.0;
    for r in &detrended {
        cumsum += r;
        partial_sums.push(cumsum);
    }

    // Оценка долгосрочной дисперсии (Newey-West)
    let lag = (4.0 * (n as f64 / 100.0).powf(0.25)) as usize;

    let mut s2 = detrended.iter().map(|r| r * r).sum::<f64>() / n as f64;

    for l in 1..=lag {
        let weight = 1.0 - l as f64 / (lag + 1) as f64;
        let gamma: f64 = detrended[l..]
            .iter()
            .zip(detrended[..n - l].iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / n as f64;
        s2 += 2.0 * weight * gamma;
    }

    // KPSS статистика
    let eta: f64 = partial_sums.iter().map(|s| s * s).sum::<f64>() / (n * n) as f64 / s2;

    // Критические значения (для случая с константой)
    let critical_values = if trend {
        vec![
            ("1%".to_string(), 0.216),
            ("5%".to_string(), 0.146),
            ("10%".to_string(), 0.119),
        ]
    } else {
        vec![
            ("1%".to_string(), 0.739),
            ("5%".to_string(), 0.463),
            ("10%".to_string(), 0.347),
        ]
    };

    // Приблизительное p-value
    let cv_10 = if trend { 0.119 } else { 0.347 };
    let cv_5 = if trend { 0.146 } else { 0.463 };
    let cv_1 = if trend { 0.216 } else { 0.739 };

    let p_value = if eta < cv_10 {
        0.10 + 0.90 * (1.0 - eta / cv_10)
    } else if eta < cv_5 {
        0.05 + (0.10 - 0.05) * (cv_5 - eta) / (cv_5 - cv_10)
    } else if eta < cv_1 {
        0.01 + (0.05 - 0.01) * (cv_1 - eta) / (cv_1 - cv_5)
    } else {
        0.01 * (1.0 - (eta - cv_1) / cv_1).max(0.0)
    };

    TestResult {
        test_name: format!("KPSS Test ({})", if trend { "trend" } else { "constant" }),
        statistic: eta,
        p_value,
        critical_values,
        is_significant: eta > cv_5, // Отвергаем H0 о стационарности
    }
}

/// Проверка стационарности с помощью скользящих статистик
pub fn rolling_stationarity_check(data: &[f64], window: usize) -> RollingStationarityResult {
    if data.len() < window * 2 {
        return RollingStationarityResult {
            is_stable_mean: false,
            is_stable_variance: false,
            mean_variation: f64::NAN,
            variance_variation: f64::NAN,
            rolling_means: vec![],
            rolling_stds: vec![],
        };
    }

    let rolling_means: Vec<f64> = data.windows(window).map(|w| mean(w)).collect();

    let rolling_stds: Vec<f64> = data.windows(window).map(|w| std_dev(w)).collect();

    let mean_of_means = mean(&rolling_means);
    let std_of_means = std_dev(&rolling_means);
    let mean_variation = std_of_means / mean_of_means.abs().max(1e-10);

    let mean_of_stds = mean(&rolling_stds);
    let std_of_stds = std_dev(&rolling_stds);
    let variance_variation = std_of_stds / mean_of_stds.max(1e-10);

    RollingStationarityResult {
        is_stable_mean: mean_variation < 0.1,
        is_stable_variance: variance_variation < 0.3,
        mean_variation,
        variance_variation,
        rolling_means,
        rolling_stds,
    }
}

#[derive(Debug, Clone)]
pub struct RollingStationarityResult {
    pub is_stable_mean: bool,
    pub is_stable_variance: bool,
    pub mean_variation: f64,
    pub variance_variation: f64,
    pub rolling_means: Vec<f64>,
    pub rolling_stds: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adf_stationary() {
        // Стационарный ряд (белый шум)
        let data: Vec<f64> = (0..200)
            .map(|i| (i as f64 * 0.1).sin())
            .collect();
        let result = adf_test(&data, None);
        assert!(result.statistic < -2.0);
    }

    #[test]
    fn test_adf_random_walk() {
        // Случайное блуждание (нестационарный)
        let mut data = vec![0.0];
        for i in 1..200 {
            data.push(data[i - 1] + (i as f64 * 0.1).sin() * 0.1);
        }
        let result = adf_test(&data, None);
        // Ожидаем, что статистика будет ближе к 0
        assert!(result.statistic > -3.0);
    }
}
