//! TWAP (Time-Weighted Average Price) Executor

use super::schedule::ExecutionSchedule;

/// TWAP Executor
///
/// Распределяет объём равномерно по времени.
pub struct TWAPExecutor {
    /// Общий объём для исполнения
    total_quantity: f64,
    /// Количество шагов
    num_steps: usize,
    /// Текущий шаг
    current_step: usize,
    /// Исполненный объём
    executed_quantity: f64,
}

impl TWAPExecutor {
    /// Создать новый TWAP executor
    pub fn new(total_quantity: f64, num_steps: usize) -> Self {
        Self {
            total_quantity,
            num_steps,
            current_step: 0,
            executed_quantity: 0.0,
        }
    }

    /// Сбросить состояние
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.executed_quantity = 0.0;
    }

    /// Получить объём для текущего шага
    pub fn get_quantity(&self) -> f64 {
        if self.current_step >= self.num_steps {
            return 0.0;
        }

        let remaining = self.total_quantity - self.executed_quantity;
        let remaining_steps = self.num_steps - self.current_step;

        if remaining_steps > 0 {
            remaining / remaining_steps as f64
        } else {
            remaining
        }
    }

    /// Получить долю для текущего шага
    pub fn get_fraction(&self) -> f64 {
        let remaining = self.total_quantity - self.executed_quantity;
        if remaining > 0.0 {
            self.get_quantity() / remaining
        } else {
            0.0
        }
    }

    /// Выполнить шаг
    pub fn step(&mut self, executed: f64) {
        self.executed_quantity += executed;
        self.current_step += 1;
    }

    /// Проверить завершение
    pub fn is_done(&self) -> bool {
        self.current_step >= self.num_steps
            || self.executed_quantity >= self.total_quantity
    }

    /// Сгенерировать полное расписание
    pub fn generate_schedule(&self) -> ExecutionSchedule {
        let quantity_per_step = self.total_quantity / self.num_steps as f64;
        let quantities = vec![quantity_per_step; self.num_steps];

        ExecutionSchedule::new("TWAP", self.total_quantity, quantities)
    }

    /// Текущий прогресс (0.0 - 1.0)
    pub fn progress(&self) -> f64 {
        if self.total_quantity > 0.0 {
            self.executed_quantity / self.total_quantity
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twap_basic() {
        let mut executor = TWAPExecutor::new(1000.0, 10);

        // Первый шаг: 1000 / 10 = 100
        assert!((executor.get_quantity() - 100.0).abs() < 0.001);

        executor.step(100.0);

        // Второй шаг: 900 / 9 = 100
        assert!((executor.get_quantity() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_twap_schedule() {
        let executor = TWAPExecutor::new(1000.0, 4);
        let schedule = executor.generate_schedule();

        assert_eq!(schedule.num_steps(), 4);
        assert!((schedule.quantity_at(0) - 250.0).abs() < 0.001);
        assert!(schedule.is_valid());
    }

    #[test]
    fn test_twap_progress() {
        let mut executor = TWAPExecutor::new(1000.0, 10);

        executor.step(100.0);
        assert!((executor.progress() - 0.1).abs() < 0.001);

        executor.step(400.0);
        assert!((executor.progress() - 0.5).abs() < 0.001);
    }
}
