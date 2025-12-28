//! Автокорреляционный анализ

use crate::analysis::statistics::{mean, variance};

/// Автокорреляционная функция (ACF)
///
/// Вычисляет автокорреляцию для лагов от 0 до max_lag
pub fn acf(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    if n < 2 {
        return vec![];
    }

    let max_lag = max_lag.min(n - 1);
    let m = mean(data);
    let var = variance(data);

    if var == 0.0 {
        return vec![1.0; max_lag + 1];
    }

    (0..=max_lag)
        .map(|lag| {
            if lag == 0 {
                return 1.0;
            }
            let sum: f64 = data[lag..]
                .iter()
                .zip(data[..n - lag].iter())
                .map(|(a, b)| (a - m) * (b - m))
                .sum();
            sum / (n as f64 * var)
        })
        .collect()
}

/// Частная автокорреляционная функция (PACF)
///
/// Использует алгоритм Дурбина-Левинсона
pub fn pacf(data: &[f64], max_lag: usize) -> Vec<f64> {
    let acf_values = acf(data, max_lag);
    if acf_values.is_empty() {
        return vec![];
    }

    let max_lag = max_lag.min(acf_values.len() - 1);
    let mut pacf_values = vec![0.0; max_lag + 1];
    pacf_values[0] = 1.0;

    if max_lag == 0 {
        return pacf_values;
    }

    // Алгоритм Дурбина-Левинсона
    let mut phi = vec![vec![0.0; max_lag + 1]; max_lag + 1];

    phi[1][1] = acf_values[1];
    pacf_values[1] = phi[1][1];

    for k in 2..=max_lag {
        // Числитель
        let mut num = acf_values[k];
        for j in 1..k {
            num -= phi[k - 1][j] * acf_values[k - j];
        }

        // Знаменатель
        let mut den = 1.0;
        for j in 1..k {
            den -= phi[k - 1][j] * acf_values[j];
        }

        if den.abs() < 1e-10 {
            break;
        }

        phi[k][k] = num / den;
        pacf_values[k] = phi[k][k];

        // Обновляем коэффициенты
        for j in 1..k {
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
        }
    }

    pacf_values
}

/// Границы доверительного интервала для ACF/PACF
///
/// Для нулевой гипотезы (белый шум): ±1.96/√n (95% CI)
pub fn confidence_bounds(n: usize, confidence: f64) -> f64 {
    let z = match confidence {
        c if c >= 0.99 => 2.576,
        c if c >= 0.95 => 1.96,
        c if c >= 0.90 => 1.645,
        _ => 1.96,
    };
    z / (n as f64).sqrt()
}

/// Тест Льюнг-Бокса
///
/// Проверяет, есть ли значимая автокорреляция в ряде
/// H0: все автокорреляции равны нулю
pub fn ljung_box_test(data: &[f64], lags: usize) -> LjungBoxResult {
    let n = data.len();
    let acf_values = acf(data, lags);

    if acf_values.len() < 2 {
        return LjungBoxResult {
            statistic: f64::NAN,
            p_value: 1.0,
            lags,
            is_significant: false,
        };
    }

    // Q = n(n+2) * Σ(ρ²_k / (n-k))
    let q: f64 = acf_values[1..]
        .iter()
        .enumerate()
        .map(|(k, rho)| {
            let k = k + 1;
            rho * rho / (n - k) as f64
        })
        .sum::<f64>()
        * n as f64
        * (n + 2) as f64;

    // p-value из распределения хи-квадрат
    let p_value = chi2_survival(q, lags);

    LjungBoxResult {
        statistic: q,
        p_value,
        lags,
        is_significant: p_value < 0.05,
    }
}

/// Результат теста Льюнг-Бокса
#[derive(Debug, Clone)]
pub struct LjungBoxResult {
    pub statistic: f64,
    pub p_value: f64,
    pub lags: usize,
    pub is_significant: bool,
}

/// Приблизительное p-value для chi-квадрат распределения
fn chi2_survival(x: f64, df: usize) -> f64 {
    use statrs::distribution::{ChiSquared, ContinuousCDF};

    if let Ok(chi2) = ChiSquared::new(df as f64) {
        1.0 - chi2.cdf(x)
    } else {
        1.0
    }
}

/// Визуализация ACF/PACF в текстовом формате
pub fn plot_acf_text(values: &[f64], name: &str, max_width: usize) -> String {
    let mut result = format!("\n{} (n={}):\n", name, values.len());
    result.push_str(&"-".repeat(max_width + 20));
    result.push('\n');

    let max_val = values
        .iter()
        .map(|x| x.abs())
        .fold(0.0, f64::max)
        .max(1.0);

    for (lag, &value) in values.iter().enumerate() {
        let bar_len = ((value.abs() / max_val) * max_width as f64) as usize;
        let bar = if value >= 0.0 {
            format!(
                "{:>4} | {:>6.3} |{}",
                lag,
                value,
                "#".repeat(bar_len)
            )
        } else {
            let spaces = max_width - bar_len;
            format!(
                "{:>4} | {:>6.3} |{}{}",
                lag,
                value,
                " ".repeat(spaces),
                "#".repeat(bar_len)
            )
        };
        result.push_str(&bar);
        result.push('\n');
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acf_white_noise() {
        // Для белого шума ACF должна быть близка к 0 для всех лагов > 0
        let data: Vec<f64> = (0..1000)
            .map(|i| ((i * 7919) % 1000) as f64 / 1000.0 - 0.5)
            .collect();

        let acf_vals = acf(&data, 10);
        assert!((acf_vals[0] - 1.0).abs() < 1e-10);

        // Проверяем, что остальные значения малы
        let bound = confidence_bounds(data.len(), 0.95);
        for &val in &acf_vals[1..] {
            assert!(val.abs() < bound * 3.0); // Допускаем некоторую погрешность
        }
    }

    #[test]
    fn test_pacf() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let pacf_vals = pacf(&data, 10);
        assert!((pacf_vals[0] - 1.0).abs() < 1e-10);
    }
}
