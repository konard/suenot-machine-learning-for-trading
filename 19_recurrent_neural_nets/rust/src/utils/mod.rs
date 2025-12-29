//! # Вспомогательные утилиты
//!
//! Функции для работы с данными, метриками и визуализацией.

mod metrics;
mod io;

pub use metrics::{mse, rmse, mae, mape, accuracy, r2_score};
pub use io::{save_candles_csv, load_candles_csv, save_predictions_csv};
