//! GARCH модели для прогнозирования волатильности

use crate::analysis::statistics::{mean, variance};

/// Параметры GARCH модели
#[derive(Debug, Clone)]
pub struct GarchParams {
    pub p: usize, // Порядок GARCH (лаги условной дисперсии)
    pub q: usize, // Порядок ARCH (лаги квадратов остатков)
}

impl GarchParams {
    pub fn new(p: usize, q: usize) -> Self {
        Self { p, q }
    }

    /// Стандартный GARCH(1,1)
    pub fn garch11() -> Self {
        Self { p: 1, q: 1 }
    }
}

/// Обученная GARCH модель
#[derive(Debug, Clone)]
pub struct GarchModel {
    pub params: GarchParams,
    pub omega: f64,         // Константа (ω)
    pub alpha: Vec<f64>,    // Коэффициенты ARCH (α)
    pub beta: Vec<f64>,     // Коэффициенты GARCH (β)
    pub mu: f64,            // Среднее доходности
    pub conditional_var: Vec<f64>, // Условная дисперсия
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
}

impl GarchModel {
    /// Обучить GARCH модель на данных доходностей
    pub fn fit(returns: &[f64], params: GarchParams) -> Option<Self> {
        let n = returns.len();
        if n < params.p.max(params.q) + 20 {
            return None;
        }

        let mu = mean(returns);
        let residuals: Vec<f64> = returns.iter().map(|r| r - mu).collect();
        let unconditional_var = variance(&residuals);

        // Начальные значения параметров
        let omega_init = unconditional_var * 0.1;
        let alpha_init = vec![0.1; params.q];
        let beta_init = vec![0.8 / params.p as f64; params.p];

        // Оптимизация методом градиентного спуска
        let (omega, alpha, beta) = optimize_garch(
            &residuals,
            params.p,
            params.q,
            omega_init,
            alpha_init,
            beta_init,
            unconditional_var,
        )?;

        // Вычисляем условную дисперсию
        let conditional_var = compute_conditional_variance(
            &residuals,
            omega,
            &alpha,
            &beta,
            unconditional_var,
        );

        // Log-likelihood
        let log_likelihood = compute_log_likelihood(&residuals, &conditional_var);

        // Информационные критерии
        let k = (1 + params.p + params.q + 1) as f64; // omega + alpha + beta + mu
        let aic = -2.0 * log_likelihood + 2.0 * k;
        let bic = -2.0 * log_likelihood + k * (n as f64).ln();

        Some(Self {
            params,
            omega,
            alpha,
            beta,
            mu,
            conditional_var,
            log_likelihood,
            aic,
            bic,
        })
    }

    /// Прогноз волатильности на h шагов вперёд
    pub fn forecast_volatility(&self, returns: &[f64], h: usize) -> Vec<f64> {
        let mu = mean(returns);
        let residuals: Vec<f64> = returns.iter().map(|r| r - mu).collect();
        let unconditional_var = variance(&residuals);

        let current_var = compute_conditional_variance(
            &residuals,
            self.omega,
            &self.alpha,
            &self.beta,
            unconditional_var,
        );

        let mut forecasts = Vec::with_capacity(h);
        let mut sigma2 = *current_var.last().unwrap_or(&unconditional_var);
        let last_resid_sq = residuals.last().map(|r| r * r).unwrap_or(unconditional_var);

        // Долгосрочная дисперсия
        let alpha_sum: f64 = self.alpha.iter().sum();
        let beta_sum: f64 = self.beta.iter().sum();
        let persistence = alpha_sum + beta_sum;
        let long_run_var = if persistence < 1.0 {
            self.omega / (1.0 - persistence)
        } else {
            unconditional_var
        };

        for step in 0..h {
            if step == 0 {
                // Первый шаг прогноза
                sigma2 = self.omega
                    + self.alpha.iter().sum::<f64>() * last_resid_sq
                    + self.beta.iter().sum::<f64>() * sigma2;
            } else {
                // Последующие шаги: σ²_{t+h} → долгосрочная дисперсия
                sigma2 = long_run_var + persistence.powi(step as i32) * (sigma2 - long_run_var);
            }
            forecasts.push(sigma2.sqrt());
        }

        forecasts
    }

    /// Проверка стабильности модели
    pub fn is_stable(&self) -> bool {
        let alpha_sum: f64 = self.alpha.iter().sum();
        let beta_sum: f64 = self.beta.iter().sum();
        alpha_sum + beta_sum < 1.0
    }

    /// Persistence (устойчивость волатильности)
    pub fn persistence(&self) -> f64 {
        let alpha_sum: f64 = self.alpha.iter().sum();
        let beta_sum: f64 = self.beta.iter().sum();
        alpha_sum + beta_sum
    }

    /// Half-life (время полувозврата к долгосрочной волатильности)
    pub fn half_life(&self) -> Option<f64> {
        let persistence = self.persistence();
        if persistence >= 1.0 || persistence <= 0.0 {
            None
        } else {
            Some(-(2.0_f64.ln()) / persistence.ln())
        }
    }

    /// Вывод информации о модели
    pub fn summary(&self) -> String {
        let mut s = format!(
            "GARCH({},{}) Model Summary\n",
            self.params.p, self.params.q
        );
        s.push_str(&"=".repeat(40));
        s.push('\n');

        s.push_str(&format!("Mean (μ): {:.6}\n", self.mu));
        s.push_str(&format!("Omega (ω): {:.6}\n", self.omega));

        for (i, &a) in self.alpha.iter().enumerate() {
            s.push_str(&format!("Alpha[{}] (α): {:.6}\n", i + 1, a));
        }

        for (i, &b) in self.beta.iter().enumerate() {
            s.push_str(&format!("Beta[{}] (β): {:.6}\n", i + 1, b));
        }

        s.push_str(&format!("\nPersistence: {:.4}\n", self.persistence()));
        if let Some(hl) = self.half_life() {
            s.push_str(&format!("Half-life: {:.2} periods\n", hl));
        }
        s.push_str(&format!("Stable: {}\n", self.is_stable()));
        s.push_str(&format!("\nLog-likelihood: {:.2}\n", self.log_likelihood));
        s.push_str(&format!("AIC: {:.2}\n", self.aic));
        s.push_str(&format!("BIC: {:.2}\n", self.bic));

        s
    }
}

/// Вычисление условной дисперсии
fn compute_conditional_variance(
    residuals: &[f64],
    omega: f64,
    alpha: &[f64],
    beta: &[f64],
    unconditional_var: f64,
) -> Vec<f64> {
    let n = residuals.len();
    let p = beta.len();
    let q = alpha.len();

    let mut sigma2 = vec![unconditional_var; n];

    for t in 1..n {
        let mut var = omega;

        // ARCH компонента
        for i in 0..q {
            if t > i {
                var += alpha[i] * residuals[t - 1 - i].powi(2);
            } else {
                var += alpha[i] * unconditional_var;
            }
        }

        // GARCH компонента
        for i in 0..p {
            if t > i {
                var += beta[i] * sigma2[t - 1 - i];
            } else {
                var += beta[i] * unconditional_var;
            }
        }

        sigma2[t] = var.max(1e-10); // Предотвращаем отрицательную дисперсию
    }

    sigma2
}

/// Вычисление log-likelihood для GARCH
fn compute_log_likelihood(residuals: &[f64], sigma2: &[f64]) -> f64 {
    let n = residuals.len();
    let mut ll = 0.0;

    for t in 0..n {
        let s2 = sigma2[t];
        if s2 > 0.0 {
            ll -= 0.5 * (s2.ln() + residuals[t].powi(2) / s2);
        }
    }

    ll -= 0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln();
    ll
}

/// Оптимизация параметров GARCH
fn optimize_garch(
    residuals: &[f64],
    p: usize,
    q: usize,
    omega_init: f64,
    alpha_init: Vec<f64>,
    beta_init: Vec<f64>,
    unconditional_var: f64,
) -> Option<(f64, Vec<f64>, Vec<f64>)> {
    let mut omega = omega_init;
    let mut alpha = alpha_init;
    let mut beta = beta_init;

    let learning_rate = 0.001;
    let max_iter = 500;
    let tol = 1e-6;

    let mut prev_ll = f64::NEG_INFINITY;

    for _ in 0..max_iter {
        let sigma2 = compute_conditional_variance(residuals, omega, &alpha, &beta, unconditional_var);
        let ll = compute_log_likelihood(residuals, &sigma2);

        if (ll - prev_ll).abs() < tol {
            break;
        }
        prev_ll = ll;

        // Численные градиенты
        let eps = 1e-5;

        // Градиент по omega
        let sigma2_plus = compute_conditional_variance(
            residuals,
            omega + eps,
            &alpha,
            &beta,
            unconditional_var,
        );
        let ll_plus = compute_log_likelihood(residuals, &sigma2_plus);
        let grad_omega = (ll_plus - ll) / eps;

        omega += learning_rate * grad_omega;
        omega = omega.max(1e-8);

        // Градиенты по alpha
        for i in 0..q {
            let mut alpha_new = alpha.clone();
            alpha_new[i] += eps;
            let sigma2_plus = compute_conditional_variance(
                residuals,
                omega,
                &alpha_new,
                &beta,
                unconditional_var,
            );
            let ll_plus = compute_log_likelihood(residuals, &sigma2_plus);
            let grad = (ll_plus - ll) / eps;

            alpha[i] += learning_rate * grad;
            alpha[i] = alpha[i].max(0.0).min(0.99);
        }

        // Градиенты по beta
        for i in 0..p {
            let mut beta_new = beta.clone();
            beta_new[i] += eps;
            let sigma2_plus = compute_conditional_variance(
                residuals,
                omega,
                &alpha,
                &beta_new,
                unconditional_var,
            );
            let ll_plus = compute_log_likelihood(residuals, &sigma2_plus);
            let grad = (ll_plus - ll) / eps;

            beta[i] += learning_rate * grad;
            beta[i] = beta[i].max(0.0).min(0.99);
        }

        // Проверяем условие стабильности
        let alpha_sum: f64 = alpha.iter().sum();
        let beta_sum: f64 = beta.iter().sum();
        if alpha_sum + beta_sum >= 0.999 {
            let scale = 0.99 / (alpha_sum + beta_sum);
            for a in &mut alpha {
                *a *= scale;
            }
            for b in &mut beta {
                *b *= scale;
            }
        }
    }

    Some((omega, alpha, beta))
}

/// Тест на ARCH эффекты (Engle's ARCH test)
pub fn arch_test(residuals: &[f64], lags: usize) -> ArchTestResult {
    let n = residuals.len();
    if n < lags + 2 {
        return ArchTestResult {
            statistic: f64::NAN,
            p_value: 1.0,
            lags,
            is_significant: false,
        };
    }

    // Квадраты остатков
    let sq_resid: Vec<f64> = residuals.iter().map(|r| r * r).collect();

    // Регрессия: ε²_t = α₀ + Σ α_i ε²_{t-i}
    let effective_n = n - lags;
    let mut x_data = Vec::with_capacity(effective_n * (lags + 1));
    let mut y_data = Vec::with_capacity(effective_n);

    for t in lags..n {
        y_data.push(sq_resid[t]);
        x_data.push(1.0);
        for i in 1..=lags {
            x_data.push(sq_resid[t - i]);
        }
    }

    use nalgebra::{DMatrix, DVector};

    let x = DMatrix::from_row_slice(effective_n, lags + 1, &x_data);
    let y = DVector::from_vec(y_data.clone());

    let xtx = x.transpose() * &x;
    let xty = x.transpose() * &y;

    let xtx_inv = match xtx.try_inverse() {
        Some(inv) => inv,
        None => {
            return ArchTestResult {
                statistic: f64::NAN,
                p_value: 1.0,
                lags,
                is_significant: false,
            };
        }
    };

    let beta = &xtx_inv * xty;
    let y_hat = &x * &beta;

    // R² = 1 - SSR/SST
    let y_mean = mean(&y_data);
    let sst: f64 = y_data.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let ssr: f64 = (&y - y_hat).iter().map(|r| r * r).sum();
    let r2 = 1.0 - ssr / sst;

    // Test statistic: n * R² ~ χ²(lags)
    let stat = effective_n as f64 * r2;

    // p-value
    use statrs::distribution::{ChiSquared, ContinuousCDF};
    let p_value = if let Ok(chi2) = ChiSquared::new(lags as f64) {
        1.0 - chi2.cdf(stat)
    } else {
        1.0
    };

    ArchTestResult {
        statistic: stat,
        p_value,
        lags,
        is_significant: p_value < 0.05,
    }
}

/// Результат ARCH теста
#[derive(Debug, Clone)]
pub struct ArchTestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub lags: usize,
    pub is_significant: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_garch_fit() {
        // Генерируем данные с GARCH эффектами
        let n = 500;
        let omega = 0.0001;
        let alpha = 0.1;
        let beta = 0.85;

        let mut returns = Vec::with_capacity(n);
        let mut sigma2 = 0.0001;

        for i in 0..n {
            let z = ((i * 7919 + 1) % 2000) as f64 / 1000.0 - 1.0;
            let r = (sigma2).sqrt() * z;
            returns.push(r);
            sigma2 = omega + alpha * r * r + beta * sigma2;
        }

        let params = GarchParams::garch11();
        let model = GarchModel::fit(&returns, params);
        assert!(model.is_some());

        let model = model.unwrap();
        assert!(model.is_stable());
    }
}
