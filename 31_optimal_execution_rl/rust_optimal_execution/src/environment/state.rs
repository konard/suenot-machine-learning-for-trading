//! Состояния и действия среды исполнения

use serde::{Deserialize, Serialize};
use ndarray::Array1;

/// Действие агента - доля оставшегося объёма для исполнения
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionAction {
    /// Дискретное действие (индекс от 0 до num_actions-1)
    Discrete(usize),
    /// Непрерывное действие (доля от 0.0 до 1.0)
    Continuous(f64),
}

impl ExecutionAction {
    /// Преобразовать в долю (от 0.0 до 1.0)
    pub fn to_fraction(&self, num_actions: usize) -> f64 {
        match self {
            Self::Discrete(idx) => {
                if num_actions <= 1 {
                    1.0
                } else {
                    *idx as f64 / (num_actions - 1) as f64
                }
            }
            Self::Continuous(frac) => frac.clamp(0.0, 1.0),
        }
    }

    /// Создать из индекса
    pub fn from_index(idx: usize) -> Self {
        Self::Discrete(idx)
    }

    /// Создать из доли
    pub fn from_fraction(frac: f64) -> Self {
        Self::Continuous(frac.clamp(0.0, 1.0))
    }
}

impl Default for ExecutionAction {
    fn default() -> Self {
        Self::Discrete(0)
    }
}

/// Состояние среды исполнения
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionState {
    /// Доля оставшегося объёма (0.0 - 1.0)
    pub remaining_fraction: f64,
    /// Доля оставшегося времени (0.0 - 1.0)
    pub time_fraction: f64,
    /// Текущий спред (нормализованный)
    pub spread: f64,
    /// Волатильность (нормализованная)
    pub volatility: f64,
    /// Дисбаланс книги ордеров (-1.0 to 1.0)
    pub order_imbalance: f64,
    /// Momentum (изменение цены)
    pub momentum: f64,
    /// Объём торгов (нормализованный)
    pub volume: f64,
    /// Позиция относительно VWAP
    pub vwap_deviation: f64,
    /// Скорость исполнения (нормализованная)
    pub execution_rate: f64,
    /// Implementation shortfall до текущего момента
    pub current_shortfall: f64,
}

impl ExecutionState {
    /// Создать начальное состояние
    pub fn initial() -> Self {
        Self {
            remaining_fraction: 1.0,
            time_fraction: 1.0,
            spread: 0.0,
            volatility: 0.0,
            order_imbalance: 0.0,
            momentum: 0.0,
            volume: 0.0,
            vwap_deviation: 0.0,
            execution_rate: 0.0,
            current_shortfall: 0.0,
        }
    }

    /// Преобразовать в вектор признаков
    pub fn to_features(&self) -> Array1<f64> {
        Array1::from_vec(vec![
            self.remaining_fraction,
            self.time_fraction,
            self.spread,
            self.volatility,
            self.order_imbalance,
            self.momentum,
            self.volume,
            self.vwap_deviation,
            self.execution_rate,
            self.current_shortfall,
        ])
    }

    /// Создать из вектора признаков
    pub fn from_features(features: &Array1<f64>) -> Self {
        let f = features.as_slice().unwrap();
        Self {
            remaining_fraction: *f.get(0).unwrap_or(&1.0),
            time_fraction: *f.get(1).unwrap_or(&1.0),
            spread: *f.get(2).unwrap_or(&0.0),
            volatility: *f.get(3).unwrap_or(&0.0),
            order_imbalance: *f.get(4).unwrap_or(&0.0),
            momentum: *f.get(5).unwrap_or(&0.0),
            volume: *f.get(6).unwrap_or(&0.0),
            vwap_deviation: *f.get(7).unwrap_or(&0.0),
            execution_rate: *f.get(8).unwrap_or(&0.0),
            current_shortfall: *f.get(9).unwrap_or(&0.0),
        }
    }

    /// Размерность состояния
    pub fn state_dim() -> usize {
        10
    }

    /// Нормализовать состояние
    pub fn normalize(&mut self) {
        // Ограничиваем значения разумными диапазонами
        self.remaining_fraction = self.remaining_fraction.clamp(0.0, 1.0);
        self.time_fraction = self.time_fraction.clamp(0.0, 1.0);
        self.spread = self.spread.clamp(-1.0, 1.0);
        self.volatility = self.volatility.clamp(-3.0, 3.0);
        self.order_imbalance = self.order_imbalance.clamp(-1.0, 1.0);
        self.momentum = self.momentum.clamp(-3.0, 3.0);
        self.volume = self.volume.clamp(-3.0, 3.0);
        self.vwap_deviation = self.vwap_deviation.clamp(-3.0, 3.0);
        self.execution_rate = self.execution_rate.clamp(-3.0, 3.0);
        self.current_shortfall = self.current_shortfall.clamp(-1.0, 1.0);
    }
}

impl Default for ExecutionState {
    fn default() -> Self {
        Self::initial()
    }
}

/// Информация о шаге среды
#[derive(Debug, Clone)]
pub struct StepInfo {
    /// Исполненный объём на этом шаге
    pub executed_quantity: f64,
    /// Цена исполнения
    pub execution_price: f64,
    /// Стоимость исполнения (impact + комиссии)
    pub execution_cost: f64,
    /// Implementation shortfall на этом шаге
    pub step_shortfall: f64,
    /// Номер шага
    pub step: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_to_fraction() {
        let action = ExecutionAction::Discrete(5);
        let frac = action.to_fraction(11); // 0, 0.1, 0.2, ..., 1.0
        assert!((frac - 0.5).abs() < 0.001);

        let action = ExecutionAction::Continuous(0.75);
        let frac = action.to_fraction(11);
        assert!((frac - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_state_features() {
        let state = ExecutionState::initial();
        let features = state.to_features();
        assert_eq!(features.len(), ExecutionState::state_dim());

        let restored = ExecutionState::from_features(&features);
        assert!((restored.remaining_fraction - state.remaining_fraction).abs() < 0.001);
    }
}
