//! Модуль стратегии
//!
//! Этот модуль содержит:
//! - Генерацию торговых сигналов
//! - Расчёт весов портфеля

pub mod signals;
pub mod weights;

pub use signals::{generate_simple_signals, multi_timeframe_signal, SignalGenerator};
pub use weights::{
    correlation, ewma_volatility, rolling_volatility, WeightCalculator, WeightConfig,
};
