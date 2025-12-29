//! # Rust Optimal Execution
//!
//! Библиотека для оптимального исполнения ордеров с использованием
//! обучения с подкреплением на криптовалютной бирже Bybit.
//!
//! ## Модули
//!
//! - `api` - Клиент Bybit API для получения рыночных данных
//! - `impact` - Модели воздействия на рынок (market impact)
//! - `environment` - Среда для обучения с подкреплением
//! - `agent` - RL агенты (DQN, Q-learning)
//! - `baselines` - Базовые алгоритмы (TWAP, VWAP, Almgren-Chriss)
//! - `utils` - Вспомогательные функции и метрики

pub mod api;
pub mod impact;
pub mod environment;
pub mod agent;
pub mod baselines;
pub mod utils;

pub use api::{BybitClient, Candle, OrderBook, Interval};
pub use impact::{ImpactModel, LinearImpact, SquareRootImpact, TransientImpact};
pub use environment::{ExecutionEnv, ExecutionState, ExecutionAction, StepResult};
pub use agent::{Agent, DQNAgent, QLearningAgent, DQNConfig};
pub use baselines::{TWAPExecutor, VWAPExecutor, AlmgrenChrissExecutor, ExecutionSchedule};
pub use utils::{ExecutionMetrics, PerformanceStats};

/// Версия библиотеки
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Конфигурация по умолчанию для обучения
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Количество эпизодов обучения
    pub num_episodes: usize,
    /// Размер батча для обучения
    pub batch_size: usize,
    /// Скорость обучения
    pub learning_rate: f64,
    /// Коэффициент дисконтирования
    pub gamma: f64,
    /// Начальное значение epsilon
    pub epsilon_start: f64,
    /// Конечное значение epsilon
    pub epsilon_end: f64,
    /// Скорость затухания epsilon
    pub epsilon_decay: f64,
    /// Размер буфера воспроизведения
    pub buffer_size: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_episodes: 10000,
            batch_size: 64,
            learning_rate: 0.001,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            buffer_size: 100000,
        }
    }
}

/// Конфигурация среды исполнения
#[derive(Debug, Clone)]
pub struct EnvironmentConfig {
    /// Длина эпизода (количество шагов)
    pub episode_length: usize,
    /// Комиссия за сделку (в базисных пунктах)
    pub trading_cost_bps: f64,
    /// Начальный капитал
    pub initial_capital: f64,
    /// Максимальная просадка
    pub max_drawdown: f64,
    /// Коэффициент неприятия риска
    pub risk_aversion: f64,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            episode_length: 60,
            trading_cost_bps: 0.001,
            initial_capital: 10000.0,
            max_drawdown: 0.3,
            risk_aversion: 1e-6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_configs() {
        let training_config = TrainingConfig::default();
        assert_eq!(training_config.num_episodes, 10000);
        assert_eq!(training_config.batch_size, 64);

        let env_config = EnvironmentConfig::default();
        assert_eq!(env_config.episode_length, 60);
    }
}
