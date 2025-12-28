//! # Модуль анализа временных рядов
//!
//! Инструменты для диагностики и анализа свойств временных рядов.

mod statistics;
mod stationarity;
mod autocorrelation;
mod decomposition;

pub use statistics::*;
pub use stationarity::*;
pub use autocorrelation::*;
pub use decomposition::*;
