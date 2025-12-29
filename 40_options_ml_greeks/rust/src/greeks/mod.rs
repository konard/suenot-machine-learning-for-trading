//! # Модуль расчёта греков опционов
//!
//! Реализация модели Блэка-Шоулза для ценообразования опционов
//! и расчёта греков (Delta, Gamma, Theta, Vega, Rho).

mod black_scholes;

pub use black_scholes::BlackScholes;

use serde::{Deserialize, Serialize};

/// Тип опциона
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

impl std::fmt::Display for OptionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptionType::Call => write!(f, "Call"),
            OptionType::Put => write!(f, "Put"),
        }
    }
}

/// Греки опциона
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Greeks {
    /// Delta (Δ) - чувствительность к цене базового актива
    pub delta: f64,

    /// Gamma (Γ) - скорость изменения дельты
    pub gamma: f64,

    /// Theta (Θ) - временной распад (в день)
    pub theta: f64,

    /// Vega (ν) - чувствительность к волатильности (на 1%)
    pub vega: f64,

    /// Rho (ρ) - чувствительность к процентной ставке
    pub rho: f64,
}

impl Greeks {
    /// Создать новый набор греков
    pub fn new(delta: f64, gamma: f64, theta: f64, vega: f64, rho: f64) -> Self {
        Self {
            delta,
            gamma,
            theta,
            vega,
            rho,
        }
    }

    /// Пустые греки (все нули)
    pub fn zero() -> Self {
        Self {
            delta: 0.0,
            gamma: 0.0,
            theta: 0.0,
            vega: 0.0,
            rho: 0.0,
        }
    }

    /// Сложить греки (для портфеля)
    pub fn add(&self, other: &Greeks, multiplier: f64) -> Greeks {
        Greeks {
            delta: self.delta + other.delta * multiplier,
            gamma: self.gamma + other.gamma * multiplier,
            theta: self.theta + other.theta * multiplier,
            vega: self.vega + other.vega * multiplier,
            rho: self.rho + other.rho * multiplier,
        }
    }
}

impl std::fmt::Display for Greeks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Greeks {{ Δ: {:.4}, Γ: {:.6}, Θ: {:.4}, ν: {:.4}, ρ: {:.4} }}",
            self.delta, self.gamma, self.theta, self.vega, self.rho
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greeks_display() {
        let greeks = Greeks::new(0.5, 0.001, -0.05, 0.15, 0.02);
        let s = format!("{}", greeks);
        assert!(s.contains("0.5000"));
    }

    #[test]
    fn test_greeks_add() {
        let g1 = Greeks::new(0.5, 0.001, -0.05, 0.15, 0.02);
        let g2 = Greeks::new(0.3, 0.002, -0.03, 0.10, 0.01);
        let sum = g1.add(&g2, 2.0);

        assert!((sum.delta - 1.1).abs() < 0.0001);
        assert!((sum.gamma - 0.005).abs() < 0.0001);
    }
}
