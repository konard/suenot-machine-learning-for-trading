//! Расписание исполнения

use serde::{Deserialize, Serialize};

/// Расписание исполнения ордера
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSchedule {
    /// Название стратегии
    pub name: String,
    /// Общий объём
    pub total_quantity: f64,
    /// Объёмы по шагам
    pub quantities: Vec<f64>,
    /// Ожидаемые доли объёма по шагам
    pub fractions: Vec<f64>,
}

impl ExecutionSchedule {
    /// Создать новое расписание
    pub fn new(name: impl Into<String>, total_quantity: f64, quantities: Vec<f64>) -> Self {
        let fractions: Vec<f64> = quantities.iter()
            .map(|q| if total_quantity > 0.0 { q / total_quantity } else { 0.0 })
            .collect();

        Self {
            name: name.into(),
            total_quantity,
            quantities,
            fractions,
        }
    }

    /// Создать из долей
    pub fn from_fractions(
        name: impl Into<String>,
        total_quantity: f64,
        fractions: Vec<f64>,
    ) -> Self {
        let quantities: Vec<f64> = fractions.iter()
            .map(|f| f * total_quantity)
            .collect();

        Self {
            name: name.into(),
            total_quantity,
            quantities,
            fractions,
        }
    }

    /// Количество шагов
    pub fn num_steps(&self) -> usize {
        self.quantities.len()
    }

    /// Получить объём для шага
    pub fn quantity_at(&self, step: usize) -> f64 {
        self.quantities.get(step).copied().unwrap_or(0.0)
    }

    /// Получить долю для шага
    pub fn fraction_at(&self, step: usize) -> f64 {
        self.fractions.get(step).copied().unwrap_or(0.0)
    }

    /// Кумулятивный исполненный объём
    pub fn cumulative_quantities(&self) -> Vec<f64> {
        let mut cumsum = 0.0;
        self.quantities.iter()
            .map(|&q| {
                cumsum += q;
                cumsum
            })
            .collect()
    }

    /// Нормализовать (убедиться, что сумма = total_quantity)
    pub fn normalize(&mut self) {
        let sum: f64 = self.quantities.iter().sum();
        if sum > 0.0 && (sum - self.total_quantity).abs() > 1e-10 {
            let factor = self.total_quantity / sum;
            for q in &mut self.quantities {
                *q *= factor;
            }
            for f in &mut self.fractions {
                *f *= factor;
            }
        }
    }

    /// Проверить корректность расписания
    pub fn is_valid(&self) -> bool {
        let sum: f64 = self.quantities.iter().sum();
        (sum - self.total_quantity).abs() < 1e-6
            && self.quantities.iter().all(|&q| q >= 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_creation() {
        let schedule = ExecutionSchedule::new(
            "Test",
            1000.0,
            vec![100.0, 200.0, 300.0, 400.0],
        );

        assert_eq!(schedule.num_steps(), 4);
        assert!((schedule.total_quantity - 1000.0).abs() < 0.001);
        assert!(schedule.is_valid());
    }

    #[test]
    fn test_from_fractions() {
        let schedule = ExecutionSchedule::from_fractions(
            "Test",
            1000.0,
            vec![0.25, 0.25, 0.25, 0.25],
        );

        assert!((schedule.quantity_at(0) - 250.0).abs() < 0.001);
        assert!(schedule.is_valid());
    }
}
