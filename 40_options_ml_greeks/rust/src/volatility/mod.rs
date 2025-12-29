//! # Модуль расчёта и предсказания волатильности
//!
//! Включает:
//! - Расчёт реализованной волатильности (RV)
//! - Расчёт подразумеваемой волатильности (IV)
//! - Предсказание будущей волатильности
//! - Анализ премии за волатильностный риск (VRP)

mod realized;
mod implied;
mod predictor;
mod vrp;

pub use realized::RealizedVolatility;
pub use implied::ImpliedVolatility;
pub use predictor::VolatilityPredictor;
pub use vrp::VolatilityRiskPremium;

/// Количество дней в году для крипты (24/7 торговля)
pub const CRYPTO_DAYS_PER_YEAR: f64 = 365.0;

/// Количество дней в году для акций
pub const STOCK_DAYS_PER_YEAR: f64 = 252.0;

/// Конфигурация расчёта волатильности
#[derive(Debug, Clone)]
pub struct VolatilityConfig {
    /// Количество дней в году для аннуализации
    pub days_per_year: f64,
    /// Окно для расчёта RV по умолчанию
    pub default_window: usize,
    /// Использовать логарифмические доходности
    pub use_log_returns: bool,
}

impl Default for VolatilityConfig {
    fn default() -> Self {
        Self {
            days_per_year: CRYPTO_DAYS_PER_YEAR,
            default_window: 20,
            use_log_returns: true,
        }
    }
}

impl VolatilityConfig {
    /// Конфигурация для криптовалюты
    pub fn crypto() -> Self {
        Self {
            days_per_year: CRYPTO_DAYS_PER_YEAR,
            ..Default::default()
        }
    }

    /// Конфигурация для акций
    pub fn stock() -> Self {
        Self {
            days_per_year: STOCK_DAYS_PER_YEAR,
            ..Default::default()
        }
    }
}
