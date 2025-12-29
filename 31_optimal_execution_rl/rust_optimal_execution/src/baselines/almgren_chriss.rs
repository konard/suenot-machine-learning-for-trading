//! Almgren-Chriss Optimal Execution

use super::schedule::ExecutionSchedule;
use crate::impact::ImpactParams;

/// Almgren-Chriss Executor
///
/// Оптимальное аналитическое решение для минимизации
/// комбинации implementation shortfall и timing risk.
pub struct AlmgrenChrissExecutor {
    /// Общий объём для исполнения
    total_quantity: f64,
    /// Количество шагов
    num_steps: usize,
    /// Коэффициент неприятия риска (lambda)
    risk_aversion: f64,
    /// Волатильность
    volatility: f64,
    /// Параметр temporary impact (eta)
    eta: f64,
    /// Расчётное kappa
    kappa: f64,
    /// Оптимальная траектория (доли оставшегося)
    optimal_trajectory: Vec<f64>,
}

impl AlmgrenChrissExecutor {
    /// Создать новый Almgren-Chriss executor
    pub fn new(
        total_quantity: f64,
        num_steps: usize,
        risk_aversion: f64,
        volatility: f64,
        eta: f64,
    ) -> Self {
        let kappa = Self::compute_kappa(risk_aversion, volatility, eta);
        let optimal_trajectory = Self::compute_optimal_trajectory(num_steps, kappa);

        Self {
            total_quantity,
            num_steps,
            risk_aversion,
            volatility,
            eta,
            kappa,
            optimal_trajectory,
        }
    }

    /// Создать из параметров impact
    pub fn from_params(
        total_quantity: f64,
        num_steps: usize,
        risk_aversion: f64,
        params: &ImpactParams,
    ) -> Self {
        Self::new(
            total_quantity,
            num_steps,
            risk_aversion,
            params.volatility,
            params.eta,
        )
    }

    /// Вычислить kappa (параметр срочности)
    fn compute_kappa(risk_aversion: f64, volatility: f64, eta: f64) -> f64 {
        if eta > 0.0 {
            (risk_aversion * volatility.powi(2) / eta).sqrt()
        } else {
            1.0 // Fallback
        }
    }

    /// Вычислить оптимальную траекторию
    fn compute_optimal_trajectory(num_steps: usize, kappa: f64) -> Vec<f64> {
        if num_steps == 0 {
            return Vec::new();
        }

        let t_max = num_steps as f64;
        let sinh_kt = (kappa * t_max).sinh();

        (0..num_steps)
            .map(|t| {
                let remaining_time = t_max - t as f64;
                if sinh_kt.abs() > 1e-10 {
                    (kappa * remaining_time).sinh() / sinh_kt
                } else {
                    remaining_time / t_max
                }
            })
            .collect()
    }

    /// Получить оптимальный объём для шага
    pub fn get_quantity(&self, step: usize) -> f64 {
        if step >= self.num_steps {
            return 0.0;
        }

        // Текущая позиция (доля оставшегося)
        let current_pos = self.optimal_trajectory.get(step).copied().unwrap_or(0.0);

        // Следующая позиция
        let next_pos = if step + 1 < self.num_steps {
            self.optimal_trajectory.get(step + 1).copied().unwrap_or(0.0)
        } else {
            0.0
        };

        // Торгуемый объём = разница позиций
        (current_pos - next_pos) * self.total_quantity
    }

    /// Получить долю оставшегося для исполнения на шаге
    pub fn get_fraction(&self, step: usize) -> f64 {
        let current_pos = self.optimal_trajectory.get(step).copied().unwrap_or(0.0);
        let next_pos = if step + 1 < self.num_steps {
            self.optimal_trajectory.get(step + 1).copied().unwrap_or(0.0)
        } else {
            0.0
        };

        if current_pos > 0.0 {
            (current_pos - next_pos) / current_pos
        } else {
            1.0
        }
    }

    /// Сгенерировать полное расписание
    pub fn generate_schedule(&self) -> ExecutionSchedule {
        let quantities: Vec<f64> = (0..self.num_steps)
            .map(|t| self.get_quantity(t))
            .collect();

        ExecutionSchedule::new("Almgren-Chriss", self.total_quantity, quantities)
    }

    /// Рассчитать ожидаемую стоимость исполнения
    pub fn expected_cost(&self) -> f64 {
        let schedule = self.generate_schedule();

        // Temporary impact cost
        let temp_cost: f64 = schedule.quantities.iter()
            .map(|&q| self.eta * q.powi(2) / self.total_quantity)
            .sum();

        // Timing risk (variance)
        let timing_risk: f64 = schedule.quantities.iter()
            .enumerate()
            .map(|(t, _)| {
                let remaining = self.optimal_trajectory.get(t).copied().unwrap_or(0.0);
                self.volatility.powi(2) * (remaining * self.total_quantity).powi(2)
            })
            .sum();

        temp_cost + self.risk_aversion * timing_risk
    }

    /// Получить kappa
    pub fn kappa(&self) -> f64 {
        self.kappa
    }

    /// Интерпретация kappa
    pub fn urgency_interpretation(&self) -> &'static str {
        if self.kappa < 0.1 {
            "Низкая срочность: медленное исполнение, минимизация impact"
        } else if self.kappa < 1.0 {
            "Умеренная срочность: сбалансированное исполнение"
        } else {
            "Высокая срочность: быстрое исполнение, избегание риска"
        }
    }
}

/// Рассчитать оптимальный risk_aversion для целевого горизонта
pub fn calibrate_risk_aversion(
    target_fraction_at_midpoint: f64, // Какая доля должна быть исполнена к середине
    num_steps: usize,
    volatility: f64,
    eta: f64,
) -> f64 {
    // Бинарный поиск для нахождения lambda
    let mut low = 1e-10;
    let mut high = 1.0;

    for _ in 0..50 {
        let mid = (low + high) / 2.0;
        let executor = AlmgrenChrissExecutor::new(1.0, num_steps, mid, volatility, eta);

        let midpoint = num_steps / 2;
        let actual_fraction = 1.0 - executor.optimal_trajectory
            .get(midpoint)
            .copied()
            .unwrap_or(0.5);

        if actual_fraction < target_fraction_at_midpoint {
            low = mid;
        } else {
            high = mid;
        }
    }

    (low + high) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_almgren_chriss_basic() {
        let executor = AlmgrenChrissExecutor::new(
            1000.0,
            10,
            1e-6,
            0.02,
            0.0001,
        );

        let schedule = executor.generate_schedule();
        assert_eq!(schedule.num_steps(), 10);
        assert!(schedule.is_valid());
    }

    #[test]
    fn test_risk_aversion_effect() {
        // Низкое неприятие риска -> медленное исполнение (TWAP-like)
        let low_risk = AlmgrenChrissExecutor::new(1000.0, 10, 1e-10, 0.02, 0.0001);
        let low_schedule = low_risk.generate_schedule();

        // Высокое неприятие риска -> быстрое исполнение
        let high_risk = AlmgrenChrissExecutor::new(1000.0, 10, 1.0, 0.02, 0.0001);
        let high_schedule = high_risk.generate_schedule();

        // При высоком risk aversion первые шаги должны быть больше
        assert!(high_schedule.quantity_at(0) > low_schedule.quantity_at(0));
    }

    #[test]
    fn test_trajectory_properties() {
        let executor = AlmgrenChrissExecutor::new(1000.0, 10, 1e-6, 0.02, 0.0001);

        // Траектория должна начинаться с 1 и заканчиваться 0
        assert!((executor.optimal_trajectory[0] - 1.0).abs() < 0.001);

        // Траектория должна убывать
        for i in 1..executor.optimal_trajectory.len() {
            assert!(executor.optimal_trajectory[i] <= executor.optimal_trajectory[i - 1]);
        }
    }

    #[test]
    fn test_calibration() {
        let lambda = calibrate_risk_aversion(0.5, 10, 0.02, 0.0001);
        assert!(lambda > 0.0);

        // Проверяем, что калибровка работает
        let executor = AlmgrenChrissExecutor::new(1000.0, 10, lambda, 0.02, 0.0001);
        let midpoint_fraction = 1.0 - executor.optimal_trajectory[5];
        assert!((midpoint_fraction - 0.5).abs() < 0.1);
    }
}
