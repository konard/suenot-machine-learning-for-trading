//! VWAP (Volume-Weighted Average Price) Executor

use super::schedule::ExecutionSchedule;

/// VWAP Executor
///
/// Распределяет объём пропорционально ожидаемому объёму торгов.
pub struct VWAPExecutor {
    /// Общий объём для исполнения
    total_quantity: f64,
    /// Профиль объёма (нормализованные доли)
    volume_profile: Vec<f64>,
    /// Текущий шаг
    current_step: usize,
    /// Исполненный объём
    executed_quantity: f64,
}

impl VWAPExecutor {
    /// Создать VWAP executor с профилем объёма
    pub fn new(total_quantity: f64, volume_profile: Vec<f64>) -> Self {
        // Нормализуем профиль
        let sum: f64 = volume_profile.iter().sum();
        let normalized: Vec<f64> = if sum > 0.0 {
            volume_profile.iter().map(|v| v / sum).collect()
        } else {
            let n = volume_profile.len();
            vec![1.0 / n as f64; n]
        };

        Self {
            total_quantity,
            volume_profile: normalized,
            current_step: 0,
            executed_quantity: 0.0,
        }
    }

    /// Создать с U-образным профилем (типичный внутридневной паттерн)
    pub fn with_u_shape(total_quantity: f64, num_steps: usize) -> Self {
        let profile = Self::generate_u_shape(num_steps);
        Self::new(total_quantity, profile)
    }

    /// Создать с равномерным профилем
    pub fn uniform(total_quantity: f64, num_steps: usize) -> Self {
        let profile = vec![1.0; num_steps];
        Self::new(total_quantity, profile)
    }

    /// Сгенерировать U-образный профиль объёма
    fn generate_u_shape(num_steps: usize) -> Vec<f64> {
        (0..num_steps)
            .map(|i| {
                let x = i as f64 / (num_steps - 1).max(1) as f64;
                // U-shape: высокий объём в начале и конце
                1.0 + 2.0 * (x - 0.5).powi(2)
            })
            .collect()
    }

    /// Сбросить состояние
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.executed_quantity = 0.0;
    }

    /// Обновить профиль объёма на основе наблюдаемого объёма
    pub fn update_profile(&mut self, observed_volumes: &[f64]) {
        if observed_volumes.is_empty() {
            return;
        }

        let sum: f64 = observed_volumes.iter().sum();
        if sum > 0.0 {
            self.volume_profile = observed_volumes.iter()
                .map(|v| v / sum)
                .collect();
        }
    }

    /// Получить объём для текущего шага
    pub fn get_quantity(&self) -> f64 {
        if self.current_step >= self.volume_profile.len() {
            return 0.0;
        }

        let remaining = self.total_quantity - self.executed_quantity;
        let remaining_fraction: f64 = self.volume_profile[self.current_step..].iter().sum();

        if remaining_fraction > 0.0 {
            remaining * self.volume_profile[self.current_step] / remaining_fraction
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
        self.current_step >= self.volume_profile.len()
            || self.executed_quantity >= self.total_quantity
    }

    /// Сгенерировать полное расписание
    pub fn generate_schedule(&self) -> ExecutionSchedule {
        let quantities: Vec<f64> = self.volume_profile.iter()
            .map(|&f| f * self.total_quantity)
            .collect();

        ExecutionSchedule::new("VWAP", self.total_quantity, quantities)
    }

    /// Количество шагов
    pub fn num_steps(&self) -> usize {
        self.volume_profile.len()
    }
}

/// Рассчитать VWAP из сделок
pub fn calculate_vwap(prices: &[f64], volumes: &[f64]) -> f64 {
    if prices.len() != volumes.len() || prices.is_empty() {
        return 0.0;
    }

    let total_value: f64 = prices.iter()
        .zip(volumes.iter())
        .map(|(p, v)| p * v)
        .sum();

    let total_volume: f64 = volumes.iter().sum();

    if total_volume > 0.0 {
        total_value / total_volume
    } else {
        prices.iter().sum::<f64>() / prices.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vwap_basic() {
        let profile = vec![2.0, 1.0, 1.0, 2.0]; // U-shape
        let executor = VWAPExecutor::new(1000.0, profile);

        let schedule = executor.generate_schedule();
        assert_eq!(schedule.num_steps(), 4);

        // Первый и последний шаг должны быть больше
        assert!(schedule.quantity_at(0) > schedule.quantity_at(1));
        assert!(schedule.quantity_at(3) > schedule.quantity_at(2));
    }

    #[test]
    fn test_vwap_u_shape() {
        let executor = VWAPExecutor::with_u_shape(1000.0, 10);

        let schedule = executor.generate_schedule();
        assert!(schedule.is_valid());

        // U-shape: концы больше середины
        assert!(schedule.quantity_at(0) > schedule.quantity_at(5));
        assert!(schedule.quantity_at(9) > schedule.quantity_at(5));
    }

    #[test]
    fn test_calculate_vwap() {
        let prices = vec![100.0, 101.0, 102.0];
        let volumes = vec![10.0, 20.0, 10.0];

        let vwap = calculate_vwap(&prices, &volumes);

        // VWAP = (100*10 + 101*20 + 102*10) / (10+20+10) = 3040 / 40 = 101
        assert!((vwap - 101.0).abs() < 0.001);
    }
}
