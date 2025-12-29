//! Технические индикаторы

mod sma;
mod ema;
mod rsi;
mod macd;
mod bollinger;

pub use sma::SMA;
pub use ema::EMA;
pub use rsi::RSI;
pub use macd::MACD;
pub use bollinger::BollingerBands;

/// Трейт для всех индикаторов
pub trait Indicator {
    /// Тип выходного значения
    type Output;

    /// Рассчитать индикатор для массива цен
    fn calculate(&self, prices: &[f64]) -> Self::Output;

    /// Минимальное количество данных для расчёта
    fn min_periods(&self) -> usize;
}

/// Вспомогательные функции для расчётов
pub mod utils {
    /// Среднее арифметическое
    pub fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Стандартное отклонение (популяционное)
    pub fn std_dev(data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let mean = mean(data);
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        variance.sqrt()
    }

    /// Стандартное отклонение (выборочное)
    pub fn sample_std_dev(data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let mean = mean(data);
        let variance =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        variance.sqrt()
    }

    /// Максимальное значение
    pub fn max(data: &[f64]) -> f64 {
        data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Минимальное значение
    pub fn min(data: &[f64]) -> f64 {
        data.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Изменение (разница между соседними элементами)
    pub fn diff(data: &[f64]) -> Vec<f64> {
        if data.len() < 2 {
            return vec![];
        }
        data.windows(2).map(|w| w[1] - w[0]).collect()
    }
}
