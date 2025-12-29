//! Тесты на коинтеграцию

use crate::analysis::stationarity::adf_test;
use crate::analysis::statistics::{mean, std_dev, correlation};
use crate::types::TestResult;
use nalgebra::{DMatrix, DVector};

/// Результат теста на коинтеграцию
#[derive(Debug, Clone)]
pub struct CointegrationResult {
    pub is_cointegrated: bool,
    pub test_statistic: f64,
    pub p_value: f64,
    pub hedge_ratio: f64,
    pub spread_mean: f64,
    pub spread_std: f64,
    pub half_life: Option<f64>,
}

/// Тест Энгла-Грейнджера на коинтеграцию
///
/// Двухшаговая процедура:
/// 1. Оценить коинтеграционную регрессию y = α + β*x + ε
/// 2. Проверить остатки на стационарность (ADF тест)
pub fn engle_granger_test(y: &[f64], x: &[f64]) -> Option<CointegrationResult> {
    let n = y.len();
    if n != x.len() || n < 20 {
        return None;
    }

    // Шаг 1: OLS регрессия y = α + β*x
    let (alpha, beta, residuals) = ols_regression(y, x)?;

    // Шаг 2: ADF тест на остатках
    let adf_result = adf_test(&residuals, None);

    // Вычисляем half-life спреда
    let half_life = compute_half_life(&residuals);

    let spread_mean = mean(&residuals);
    let spread_std = std_dev(&residuals);

    // Критические значения для теста Энгла-Грейнджера
    // (отличаются от стандартного ADF из-за оценки коинтеграционного вектора)
    let critical_value_5pct = -3.34; // Приблизительное значение для 2 переменных

    Some(CointegrationResult {
        is_cointegrated: adf_result.statistic < critical_value_5pct,
        test_statistic: adf_result.statistic,
        p_value: adf_result.p_value,
        hedge_ratio: beta,
        spread_mean,
        spread_std,
        half_life,
    })
}

/// OLS регрессия y = α + β*x
fn ols_regression(y: &[f64], x: &[f64]) -> Option<(f64, f64, Vec<f64>)> {
    let n = y.len();
    if n < 2 {
        return None;
    }

    // Матрица регрессоров [1, x]
    let x_matrix = DMatrix::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x[i] });
    let y_vec = DVector::from_vec(y.to_vec());

    let xtx = x_matrix.transpose() * &x_matrix;
    let xty = x_matrix.transpose() * &y_vec;

    let xtx_inv = xtx.try_inverse()?;
    let beta = &xtx_inv * xty;

    let alpha = beta[0];
    let slope = beta[1];

    // Остатки
    let y_hat = &x_matrix * &beta;
    let residuals: Vec<f64> = (&y_vec - y_hat).iter().cloned().collect();

    Some((alpha, slope, residuals))
}

/// Вычисление half-life для mean-reverting процесса
///
/// Предполагаем AR(1) процесс: Δspread = φ*(spread_{t-1} - μ) + ε
/// Half-life = -ln(2) / ln(1 + φ)
fn compute_half_life(spread: &[f64]) -> Option<f64> {
    let n = spread.len();
    if n < 10 {
        return None;
    }

    let mean_spread = mean(spread);

    // Регрессия Δspread_t на spread_{t-1}
    let y: Vec<f64> = spread.windows(2).map(|w| w[1] - w[0]).collect();
    let x: Vec<f64> = spread[..n - 1].iter().map(|s| s - mean_spread).collect();

    let (_, phi, _) = ols_regression(&y, &x)?;

    // φ должен быть отрицательным для mean-reversion
    if phi >= 0.0 || phi <= -1.0 {
        return None;
    }

    let half_life = -(2.0_f64.ln()) / (1.0 + phi).ln();

    if half_life > 0.0 && half_life < 1000.0 {
        Some(half_life)
    } else {
        None
    }
}

/// Процедура Йохансена (упрощённая версия)
///
/// Тестирует коинтеграцию нескольких временных рядов
pub fn johansen_test(series: &[Vec<f64>], lag: usize) -> Option<JohansenResult> {
    let k = series.len(); // Количество рядов
    if k < 2 {
        return None;
    }

    let n = series[0].len();
    for s in series {
        if s.len() != n {
            return None;
        }
    }

    if n < lag + 20 {
        return None;
    }

    // Первые разности
    let diffs: Vec<Vec<f64>> = series
        .iter()
        .map(|s| s.windows(2).map(|w| w[1] - w[0]).collect())
        .collect();

    let effective_n = n - lag - 1;

    // Строим матрицы для VAR модели
    // ΔY_t = Π*Y_{t-1} + Σ Γ_i*ΔY_{t-i} + ε_t

    // Y_{t-1} (уровни с лагом 1)
    let y_lagged: Vec<Vec<f64>> = series
        .iter()
        .map(|s| s[lag..n - 1].to_vec())
        .collect();

    // ΔY_t
    let dy: Vec<Vec<f64>> = diffs
        .iter()
        .map(|d| d[lag..].to_vec())
        .collect();

    // Simplified: используем корреляционный анализ
    // Проверяем, есть ли стационарная линейная комбинация

    let mut eigenvalues = Vec::new();
    let mut eigenvectors = Vec::new();

    // Для простоты проверяем только первую пару рядов
    if k >= 2 {
        let result = engle_granger_test(&series[0], &series[1]);
        if let Some(r) = result {
            eigenvalues.push(1.0 - r.p_value); // Псевдо-собственное значение
            eigenvectors.push(vec![1.0, -r.hedge_ratio]);
        }
    }

    Some(JohansenResult {
        num_series: k,
        eigenvalues,
        eigenvectors,
        trace_statistics: vec![],
        max_eigenvalue_statistics: vec![],
        cointegration_rank: if eigenvalues.first().map(|&e| e > 0.5).unwrap_or(false) {
            1
        } else {
            0
        },
    })
}

/// Результат теста Йохансена
#[derive(Debug, Clone)]
pub struct JohansenResult {
    pub num_series: usize,
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Vec<Vec<f64>>,
    pub trace_statistics: Vec<f64>,
    pub max_eigenvalue_statistics: Vec<f64>,
    pub cointegration_rank: usize,
}

/// Вычисление спреда для пары активов
pub fn compute_spread(y: &[f64], x: &[f64], hedge_ratio: f64) -> Vec<f64> {
    y.iter()
        .zip(x.iter())
        .map(|(&yi, &xi)| yi - hedge_ratio * xi)
        .collect()
}

/// Z-score спреда
pub fn spread_zscore(spread: &[f64], lookback: usize) -> Vec<f64> {
    if spread.len() < lookback {
        return vec![0.0; spread.len()];
    }

    let mut result = vec![0.0; lookback - 1];

    for i in (lookback - 1)..spread.len() {
        let window = &spread[i + 1 - lookback..=i];
        let m = mean(window);
        let s = std_dev(window);

        if s > 0.0 {
            result.push((spread[i] - m) / s);
        } else {
            result.push(0.0);
        }
    }

    result
}

/// Поиск коинтегрированных пар среди множества активов
pub fn find_cointegrated_pairs(
    prices: &[(&str, Vec<f64>)],
    min_correlation: f64,
    max_p_value: f64,
) -> Vec<CointegrationPair> {
    let mut pairs = Vec::new();

    for i in 0..prices.len() {
        for j in (i + 1)..prices.len() {
            let (name1, series1) = &prices[i];
            let (name2, series2) = &prices[j];

            // Быстрая проверка корреляции
            let corr = correlation(series1, series2).abs();
            if corr < min_correlation {
                continue;
            }

            // Тест на коинтеграцию
            if let Some(result) = engle_granger_test(series1, series2) {
                if result.p_value < max_p_value {
                    pairs.push(CointegrationPair {
                        asset1: name1.to_string(),
                        asset2: name2.to_string(),
                        correlation: corr,
                        hedge_ratio: result.hedge_ratio,
                        adf_statistic: result.test_statistic,
                        p_value: result.p_value,
                        half_life: result.half_life,
                    });
                }
            }
        }
    }

    // Сортируем по p-value
    pairs.sort_by(|a, b| a.p_value.partial_cmp(&b.p_value).unwrap());

    pairs
}

/// Коинтегрированная пара
#[derive(Debug, Clone)]
pub struct CointegrationPair {
    pub asset1: String,
    pub asset2: String,
    pub correlation: f64,
    pub hedge_ratio: f64,
    pub adf_statistic: f64,
    pub p_value: f64,
    pub half_life: Option<f64>,
}

impl CointegrationPair {
    pub fn display(&self) -> String {
        format!(
            "{}/{}: corr={:.3}, hedge={:.4}, ADF={:.3}, p={:.4}, hl={:?}",
            self.asset1,
            self.asset2,
            self.correlation,
            self.hedge_ratio,
            self.adf_statistic,
            self.p_value,
            self.half_life.map(|h| format!("{:.1}", h))
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cointegration() {
        // Создаём коинтегрированные ряды
        let n = 500;
        let mut x = vec![100.0];
        let mut noise = vec![0.0];

        for i in 1..n {
            // x — случайное блуждание
            let innovation = ((i * 7919) % 1000) as f64 / 5000.0 - 0.1;
            x.push(x[i - 1] + innovation);

            // Стационарный шум
            noise.push(((i * 1237) % 1000) as f64 / 5000.0 - 0.1);
        }

        // y = 2*x + noise (коинтегрирован с x)
        let y: Vec<f64> = x.iter().zip(noise.iter()).map(|(&xi, &ni)| 2.0 * xi + ni).collect();

        let result = engle_granger_test(&y, &x);
        assert!(result.is_some());

        let result = result.unwrap();
        // Hedge ratio должен быть близок к 2
        assert!((result.hedge_ratio - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_spread_zscore() {
        let spread = vec![0.0, 1.0, 2.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
        let zscore = spread_zscore(&spread, 5);
        assert_eq!(zscore.len(), spread.len());
    }
}
