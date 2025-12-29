//! ARIMA модель для прогнозирования временных рядов

use crate::analysis::statistics::mean;
use nalgebra::{DMatrix, DVector};

/// Параметры ARIMA модели
#[derive(Debug, Clone)]
pub struct ArimaParams {
    pub p: usize, // Порядок AR
    pub d: usize, // Порядок дифференцирования
    pub q: usize, // Порядок MA
}

impl ArimaParams {
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self { p, d, q }
    }
}

/// Обученная ARIMA модель
#[derive(Debug, Clone)]
pub struct ArimaModel {
    pub params: ArimaParams,
    pub ar_coeffs: Vec<f64>,    // Коэффициенты AR (φ)
    pub ma_coeffs: Vec<f64>,    // Коэффициенты MA (θ)
    pub constant: f64,           // Константа (c)
    pub residuals: Vec<f64>,     // Остатки модели
    pub sigma2: f64,             // Дисперсия остатков
    pub aic: f64,                // Информационный критерий Акаике
    pub bic: f64,                // Байесовский информационный критерий
}

impl ArimaModel {
    /// Создать и обучить ARIMA модель
    pub fn fit(data: &[f64], params: ArimaParams) -> Option<Self> {
        if data.len() < params.p + params.d + params.q + 10 {
            return None;
        }

        // 1. Дифференцирование
        let diff_data = difference(data, params.d);
        if diff_data.len() < params.p.max(params.q) + 5 {
            return None;
        }

        // 2. Оценка AR параметров методом Yule-Walker (если q=0) или условным MLE
        let (ar_coeffs, ma_coeffs, constant, residuals) = if params.q == 0 {
            // Чистая AR модель — используем OLS
            estimate_ar(&diff_data, params.p)?
        } else if params.p == 0 {
            // Чистая MA модель — используем итеративный метод
            estimate_ma(&diff_data, params.q)?
        } else {
            // ARMA модель — используем условный MLE
            estimate_arma(&diff_data, params.p, params.q)?
        };

        // 3. Вычисляем метрики качества
        let n = residuals.len() as f64;
        let k = (params.p + params.q + 1) as f64;
        let sigma2 = residuals.iter().map(|r| r * r).sum::<f64>() / n;
        let log_likelihood = -0.5 * n * (1.0 + (2.0 * std::f64::consts::PI * sigma2).ln());

        let aic = -2.0 * log_likelihood + 2.0 * k;
        let bic = -2.0 * log_likelihood + k * n.ln();

        Some(Self {
            params,
            ar_coeffs,
            ma_coeffs,
            constant,
            residuals,
            sigma2,
            aic,
            bic,
        })
    }

    /// Прогноз на h шагов вперёд
    pub fn forecast(&self, data: &[f64], h: usize) -> Vec<f64> {
        let diff_data = difference(data, self.params.d);
        let n = diff_data.len();
        let p = self.params.p;
        let q = self.params.q;

        let mut forecasts = Vec::with_capacity(h);
        let mut extended_data = diff_data.clone();
        let mut extended_residuals = self.residuals.clone();

        for _ in 0..h {
            let mut forecast = self.constant;

            // AR компонента
            for i in 0..p {
                let idx = extended_data.len() - 1 - i;
                if idx < extended_data.len() {
                    forecast += self.ar_coeffs[i] * extended_data[idx];
                }
            }

            // MA компонента
            for i in 0..q {
                let idx = extended_residuals.len() - 1 - i;
                if idx < extended_residuals.len() {
                    forecast += self.ma_coeffs[i] * extended_residuals[idx];
                }
            }

            extended_data.push(forecast);
            extended_residuals.push(0.0); // Прогнозные остатки = 0 (ожидание)
            forecasts.push(forecast);
        }

        // Обратное дифференцирование для получения прогнозов в исходном масштабе
        if self.params.d > 0 {
            let mut result = forecasts;
            for _ in 0..self.params.d {
                result = integrate(&result, *data.last().unwrap_or(&0.0));
            }
            result
        } else {
            forecasts
        }
    }

    /// Доверительный интервал для прогнозов
    pub fn forecast_interval(&self, data: &[f64], h: usize, confidence: f64) -> ForecastInterval {
        let forecasts = self.forecast(data, h);

        // Стандартные ошибки растут с горизонтом
        // Для AR(1): se_h = σ * sqrt(1 + φ² + φ⁴ + ... + φ^(2(h-1)))
        let z = match confidence {
            c if c >= 0.99 => 2.576,
            c if c >= 0.95 => 1.96,
            c if c >= 0.90 => 1.645,
            _ => 1.96,
        };

        let sigma = self.sigma2.sqrt();
        let mut se = vec![sigma; h];

        // Упрощённая оценка роста SE
        if !self.ar_coeffs.is_empty() {
            let phi_sum: f64 = self.ar_coeffs.iter().sum::<f64>().abs();
            for i in 1..h {
                se[i] = sigma * (1.0 + phi_sum.powi(i as i32 * 2)).sqrt();
            }
        }

        let lower: Vec<f64> = forecasts
            .iter()
            .zip(se.iter())
            .map(|(&f, &s)| f - z * s)
            .collect();

        let upper: Vec<f64> = forecasts
            .iter()
            .zip(se.iter())
            .map(|(&f, &s)| f + z * s)
            .collect();

        ForecastInterval {
            point: forecasts,
            lower,
            upper,
            confidence,
        }
    }

    /// Вывод информации о модели
    pub fn summary(&self) -> String {
        let mut s = format!(
            "ARIMA({},{},{}) Model Summary\n",
            self.params.p, self.params.d, self.params.q
        );
        s.push_str(&"=".repeat(40));
        s.push('\n');

        if !self.ar_coeffs.is_empty() {
            s.push_str("AR Coefficients:\n");
            for (i, &c) in self.ar_coeffs.iter().enumerate() {
                s.push_str(&format!("  φ{} = {:.6}\n", i + 1, c));
            }
        }

        if !self.ma_coeffs.is_empty() {
            s.push_str("MA Coefficients:\n");
            for (i, &c) in self.ma_coeffs.iter().enumerate() {
                s.push_str(&format!("  θ{} = {:.6}\n", i + 1, c));
            }
        }

        s.push_str(&format!("Constant: {:.6}\n", self.constant));
        s.push_str(&format!("Sigma²: {:.6}\n", self.sigma2));
        s.push_str(&format!("AIC: {:.2}\n", self.aic));
        s.push_str(&format!("BIC: {:.2}\n", self.bic));

        s
    }
}

/// Результат прогноза с доверительным интервалом
#[derive(Debug, Clone)]
pub struct ForecastInterval {
    pub point: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub confidence: f64,
}

/// Дифференцирование ряда d раз
pub fn difference(data: &[f64], d: usize) -> Vec<f64> {
    let mut result = data.to_vec();
    for _ in 0..d {
        if result.len() < 2 {
            return vec![];
        }
        result = result.windows(2).map(|w| w[1] - w[0]).collect();
    }
    result
}

/// Обратное дифференцирование (интегрирование)
fn integrate(diff: &[f64], start: f64) -> Vec<f64> {
    let mut result = Vec::with_capacity(diff.len());
    let mut cumsum = start;
    for &d in diff {
        cumsum += d;
        result.push(cumsum);
    }
    result
}

/// Оценка AR модели методом OLS
fn estimate_ar(data: &[f64], p: usize) -> Option<(Vec<f64>, Vec<f64>, f64, Vec<f64>)> {
    let n = data.len();
    if n < p + 2 {
        return None;
    }

    let effective_n = n - p;

    // Зависимая переменная
    let y: Vec<f64> = data[p..].to_vec();

    // Матрица регрессоров [1, y_{t-1}, y_{t-2}, ..., y_{t-p}]
    let mut x_data = Vec::with_capacity(effective_n * (p + 1));
    for t in p..n {
        x_data.push(1.0); // Константа
        for i in 1..=p {
            x_data.push(data[t - i]);
        }
    }

    let x = DMatrix::from_row_slice(effective_n, p + 1, &x_data);
    let y_vec = DVector::from_vec(y);

    // OLS: β = (X'X)^(-1) X'y
    let xtx = x.transpose() * &x;
    let xty = x.transpose() * &y_vec;

    let xtx_inv = xtx.try_inverse()?;
    let beta = &xtx_inv * xty;

    let constant = beta[0];
    let ar_coeffs: Vec<f64> = beta.iter().skip(1).cloned().collect();

    // Вычисляем остатки
    let y_hat = &x * &beta;
    let residuals: Vec<f64> = (&y_vec - y_hat).iter().cloned().collect();

    Some((ar_coeffs, vec![], constant, residuals))
}

/// Оценка MA модели итеративным методом
fn estimate_ma(data: &[f64], q: usize) -> Option<(Vec<f64>, Vec<f64>, f64, Vec<f64>)> {
    let n = data.len();
    let data_mean = mean(data);
    let centered: Vec<f64> = data.iter().map(|x| x - data_mean).collect();

    // Начальные значения MA коэффициентов
    let mut ma_coeffs = vec![0.0; q];

    // Итеративная оценка
    let max_iter = 100;
    let tol = 1e-6;

    for _ in 0..max_iter {
        // Вычисляем остатки
        let residuals = compute_ma_residuals(&centered, &ma_coeffs);

        // Обновляем коэффициенты (упрощённый подход)
        let mut new_coeffs = vec![0.0; q];
        for i in 0..q {
            let mut num = 0.0;
            let mut den = 0.0;
            for t in (i + 1)..n {
                if t - i - 1 < residuals.len() {
                    num += centered[t] * residuals[t - i - 1];
                    den += residuals[t - i - 1] * residuals[t - i - 1];
                }
            }
            if den > 0.0 {
                new_coeffs[i] = num / den;
            }
        }

        // Проверяем сходимость
        let diff: f64 = ma_coeffs
            .iter()
            .zip(new_coeffs.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        ma_coeffs = new_coeffs;

        if diff < tol {
            break;
        }
    }

    let residuals = compute_ma_residuals(&centered, &ma_coeffs);

    Some((vec![], ma_coeffs, data_mean, residuals))
}

/// Вычисление остатков MA модели
fn compute_ma_residuals(data: &[f64], ma_coeffs: &[f64]) -> Vec<f64> {
    let n = data.len();
    let q = ma_coeffs.len();
    let mut residuals = vec![0.0; n];

    for t in 0..n {
        let mut ma_part = 0.0;
        for i in 0..q {
            if t > i {
                ma_part += ma_coeffs[i] * residuals[t - i - 1];
            }
        }
        residuals[t] = data[t] - ma_part;
    }

    residuals
}

/// Оценка ARMA модели
fn estimate_arma(data: &[f64], p: usize, q: usize) -> Option<(Vec<f64>, Vec<f64>, f64, Vec<f64>)> {
    // Используем двухшаговую процедуру Hannan-Rissanen
    let n = data.len();
    let data_mean = mean(data);
    let centered: Vec<f64> = data.iter().map(|x| x - data_mean).collect();

    // Шаг 1: Оценка высокого порядка AR для получения приближённых остатков
    let ar_order = (p + q).max(10).min(n / 4);
    let (initial_ar, _, _, initial_residuals) = estimate_ar(&centered, ar_order)?;

    // Шаг 2: Регрессия с AR и MA компонентами
    let start = (p.max(q)).max(ar_order);
    let effective_n = n - start;

    if effective_n < p + q + 2 {
        return None;
    }

    // Строим матрицу регрессоров
    let num_params = p + q + 1;
    let mut x_data = Vec::with_capacity(effective_n * num_params);
    let mut y_data = Vec::with_capacity(effective_n);

    for t in start..n {
        y_data.push(centered[t]);

        // Константа
        x_data.push(1.0);

        // AR компоненты
        for i in 1..=p {
            x_data.push(centered[t - i]);
        }

        // MA компоненты (используем приближённые остатки)
        for i in 1..=q {
            if t - i < initial_residuals.len() {
                x_data.push(initial_residuals[t - i]);
            } else {
                x_data.push(0.0);
            }
        }
    }

    let x = DMatrix::from_row_slice(effective_n, num_params, &x_data);
    let y_vec = DVector::from_vec(y_data);

    let xtx = x.transpose() * &x;
    let xty = x.transpose() * &y_vec;

    let xtx_inv = xtx.try_inverse()?;
    let beta = &xtx_inv * xty;

    let constant = beta[0] + data_mean;
    let ar_coeffs: Vec<f64> = beta.iter().skip(1).take(p).cloned().collect();
    let ma_coeffs: Vec<f64> = beta.iter().skip(1 + p).take(q).cloned().collect();

    // Вычисляем финальные остатки
    let y_hat = &x * &beta;
    let residuals: Vec<f64> = (&y_vec - y_hat).iter().cloned().collect();

    Some((ar_coeffs, ma_coeffs, constant, residuals))
}

/// Автоматический выбор порядка ARIMA
pub fn auto_arima(
    data: &[f64],
    max_p: usize,
    max_d: usize,
    max_q: usize,
) -> Option<ArimaModel> {
    let mut best_model: Option<ArimaModel> = None;
    let mut best_aic = f64::INFINITY;

    for d in 0..=max_d {
        for p in 0..=max_p {
            for q in 0..=max_q {
                if p == 0 && q == 0 {
                    continue;
                }

                let params = ArimaParams::new(p, d, q);
                if let Some(model) = ArimaModel::fit(data, params) {
                    if model.aic < best_aic {
                        best_aic = model.aic;
                        best_model = Some(model);
                    }
                }
            }
        }
    }

    best_model
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_difference() {
        let data = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let diff1 = difference(&data, 1);
        assert_eq!(diff1, vec![2.0, 3.0, 4.0, 5.0]);

        let diff2 = difference(&data, 2);
        assert_eq!(diff2, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_ar_model() {
        // Генерируем AR(1) процесс
        let mut data = vec![0.0];
        let phi = 0.7;
        for i in 1..200 {
            let noise = ((i * 7919) % 1000) as f64 / 5000.0 - 0.1;
            data.push(phi * data[i - 1] + noise);
        }

        let params = ArimaParams::new(1, 0, 0);
        let model = ArimaModel::fit(&data, params);
        assert!(model.is_some());

        let model = model.unwrap();
        // Проверяем, что оценённый коэффициент близок к истинному
        assert!((model.ar_coeffs[0] - phi).abs() < 0.2);
    }
}
