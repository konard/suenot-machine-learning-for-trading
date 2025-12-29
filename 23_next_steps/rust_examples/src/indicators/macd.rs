//! MACD (Moving Average Convergence Divergence)

use super::{ema::EMA, Indicator};

/// MACD результат
#[derive(Debug, Clone)]
pub struct MACDResult {
    /// MACD линия (fast EMA - slow EMA)
    pub macd_line: Vec<f64>,
    /// Сигнальная линия (EMA от MACD)
    pub signal_line: Vec<f64>,
    /// Гистограмма (MACD - Signal)
    pub histogram: Vec<f64>,
}

/// MACD (Moving Average Convergence Divergence)
///
/// Индикатор, показывающий взаимодействие двух EMA.
/// Стандартные параметры: 12, 26, 9
///
/// - MACD Line = EMA(12) - EMA(26)
/// - Signal Line = EMA(9) от MACD Line
/// - Histogram = MACD Line - Signal Line
#[derive(Debug, Clone)]
pub struct MACD {
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    fast_ema: EMA,
    slow_ema: EMA,
    signal_ema: EMA,
}

impl MACD {
    /// Создать MACD с указанными периодами
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        assert!(fast_period < slow_period, "Fast period must be less than slow period");

        Self {
            fast_period,
            slow_period,
            signal_period,
            fast_ema: EMA::new(fast_period),
            slow_ema: EMA::new(slow_period),
            signal_ema: EMA::new(signal_period),
        }
    }

    /// MACD со стандартными параметрами (12, 26, 9)
    pub fn standard() -> Self {
        Self::new(12, 26, 9)
    }

    /// Рассчитать текущие значения MACD
    pub fn current(&self, prices: &[f64]) -> Option<(f64, f64, f64)> {
        let result = self.calculate(prices);
        if result.macd_line.is_empty() {
            return None;
        }

        let macd = *result.macd_line.last()?;
        let signal = *result.signal_line.last()?;
        let histogram = *result.histogram.last()?;

        Some((macd, signal, histogram))
    }

    /// Проверить бычье пересечение (MACD пересекает Signal снизу вверх)
    pub fn is_bullish_crossover(prev_macd: f64, prev_signal: f64, curr_macd: f64, curr_signal: f64) -> bool {
        prev_macd <= prev_signal && curr_macd > curr_signal
    }

    /// Проверить медвежье пересечение (MACD пересекает Signal сверху вниз)
    pub fn is_bearish_crossover(prev_macd: f64, prev_signal: f64, curr_macd: f64, curr_signal: f64) -> bool {
        prev_macd >= prev_signal && curr_macd < curr_signal
    }
}

impl Indicator for MACD {
    type Output = MACDResult;

    fn calculate(&self, prices: &[f64]) -> Self::Output {
        let fast_ema = self.fast_ema.calculate(prices);
        let slow_ema = self.slow_ema.calculate(prices);

        if fast_ema.is_empty() || slow_ema.is_empty() {
            return MACDResult {
                macd_line: vec![],
                signal_line: vec![],
                histogram: vec![],
            };
        }

        // Выравниваем длины (slow EMA короче)
        let offset = fast_ema.len() - slow_ema.len();
        let macd_line: Vec<f64> = slow_ema
            .iter()
            .enumerate()
            .map(|(i, slow)| fast_ema[i + offset] - slow)
            .collect();

        // Сигнальная линия
        let signal_line = self.signal_ema.calculate(&macd_line);

        if signal_line.is_empty() {
            return MACDResult {
                macd_line,
                signal_line: vec![],
                histogram: vec![],
            };
        }

        // Выравниваем для гистограммы
        let macd_offset = macd_line.len() - signal_line.len();
        let histogram: Vec<f64> = signal_line
            .iter()
            .enumerate()
            .map(|(i, signal)| macd_line[i + macd_offset] - signal)
            .collect();

        // Возвращаем выровненные данные
        let final_macd_line = macd_line[macd_offset..].to_vec();

        MACDResult {
            macd_line: final_macd_line,
            signal_line,
            histogram,
        }
    }

    fn min_periods(&self) -> usize {
        self.slow_period + self.signal_period - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macd_calculation() {
        let macd = MACD::standard();

        // Нужно достаточно данных для расчёта
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.5)).collect();
        let result = macd.calculate(&prices);

        assert!(!result.macd_line.is_empty());
        assert!(!result.signal_line.is_empty());
        assert!(!result.histogram.is_empty());

        // Все массивы должны быть одной длины
        assert_eq!(result.macd_line.len(), result.signal_line.len());
        assert_eq!(result.signal_line.len(), result.histogram.len());
    }

    #[test]
    fn test_macd_crossover() {
        // Бычье пересечение: MACD идёт снизу вверх
        assert!(MACD::is_bullish_crossover(-1.0, 0.0, 1.0, 0.0));
        assert!(!MACD::is_bullish_crossover(1.0, 0.0, 2.0, 0.0));

        // Медвежье пересечение: MACD идёт сверху вниз
        assert!(MACD::is_bearish_crossover(1.0, 0.0, -1.0, 0.0));
        assert!(!MACD::is_bearish_crossover(-1.0, 0.0, -2.0, 0.0));
    }

    #[test]
    fn test_macd_histogram() {
        let macd = MACD::standard();
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.5)).collect();
        let result = macd.calculate(&prices);

        // Гистограмма = MACD - Signal
        for i in 0..result.histogram.len() {
            let expected = result.macd_line[i] - result.signal_line[i];
            assert!((result.histogram[i] - expected).abs() < 1e-10);
        }
    }
}
