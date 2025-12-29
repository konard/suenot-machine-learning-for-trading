//! Стратегия пересечения скользящих средних (SMA Crossover)

use crate::data::Kline;
use crate::indicators::{Indicator, SMA};
use super::{Signal, Strategy};

/// Стратегия пересечения SMA
///
/// Генерирует сигнал на покупку, когда быстрая SMA пересекает
/// медленную SMA снизу вверх, и сигнал на продажу при обратном пересечении.
///
/// # Пример
/// ```
/// use ml4t_bybit::strategies::{SmaCrossStrategy, Strategy, Signal};
/// use ml4t_bybit::data::Kline;
///
/// let strategy = SmaCrossStrategy::new(10, 20);
/// // ... получить klines ...
/// // let signal = strategy.generate_signal(&klines);
/// ```
#[derive(Debug, Clone)]
pub struct SmaCrossStrategy {
    name: String,
    fast_period: usize,
    slow_period: usize,
    fast_sma: SMA,
    slow_sma: SMA,
}

impl SmaCrossStrategy {
    /// Создать стратегию с указанными периодами
    pub fn new(fast_period: usize, slow_period: usize) -> Self {
        assert!(
            fast_period < slow_period,
            "Fast period must be less than slow period"
        );

        Self {
            name: format!("SMA Crossover ({}/{})", fast_period, slow_period),
            fast_period,
            slow_period,
            fast_sma: SMA::new(fast_period),
            slow_sma: SMA::new(slow_period),
        }
    }

    /// Классическая стратегия 10/20
    pub fn classic() -> Self {
        Self::new(10, 20)
    }

    /// Долгосрочная стратегия 50/200 (Golden Cross / Death Cross)
    pub fn golden_cross() -> Self {
        Self::new(50, 200)
    }

    /// Краткосрочная стратегия 5/10
    pub fn short_term() -> Self {
        Self::new(5, 10)
    }

    /// Получить периоды
    pub fn periods(&self) -> (usize, usize) {
        (self.fast_period, self.slow_period)
    }

    /// Рассчитать текущие значения SMA
    pub fn current_smas(&self, klines: &[Kline]) -> Option<(f64, f64)> {
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();

        let fast = self.fast_sma.current(&closes)?;
        let slow = self.slow_sma.current(&closes)?;

        Some((fast, slow))
    }

    /// Проверить наличие пересечения между двумя точками
    fn check_crossover(
        prev_fast: f64,
        prev_slow: f64,
        curr_fast: f64,
        curr_slow: f64,
    ) -> Signal {
        // Бычье пересечение: fast пересекает slow снизу вверх
        if prev_fast <= prev_slow && curr_fast > curr_slow {
            return Signal::Buy;
        }

        // Медвежье пересечение: fast пересекает slow сверху вниз
        if prev_fast >= prev_slow && curr_fast < curr_slow {
            return Signal::Sell;
        }

        Signal::Hold
    }
}

impl Strategy for SmaCrossStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn generate_signal(&self, klines: &[Kline]) -> Signal {
        if klines.len() < self.slow_period + 1 {
            return Signal::Hold;
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();

        let fast_values = self.fast_sma.calculate(&closes);
        let slow_values = self.slow_sma.calculate(&closes);

        if fast_values.len() < 2 || slow_values.len() < 2 {
            return Signal::Hold;
        }

        // Выравниваем длины
        let offset = fast_values.len() - slow_values.len();
        let fast_idx = fast_values.len() - 1;
        let slow_idx = slow_values.len() - 1;

        let prev_fast = fast_values[fast_idx - 1];
        let prev_slow = slow_values[slow_idx - 1];
        let curr_fast = fast_values[fast_idx];
        let curr_slow = slow_values[slow_idx];

        Self::check_crossover(prev_fast, prev_slow, curr_fast, curr_slow)
    }

    fn min_bars(&self) -> usize {
        self.slow_period + 1
    }
}

/// Стратегия пересечения EMA
#[derive(Debug, Clone)]
pub struct EmaCrossStrategy {
    name: String,
    fast_period: usize,
    slow_period: usize,
}

impl EmaCrossStrategy {
    pub fn new(fast_period: usize, slow_period: usize) -> Self {
        assert!(fast_period < slow_period);
        Self {
            name: format!("EMA Crossover ({}/{})", fast_period, slow_period),
            fast_period,
            slow_period,
        }
    }
}

impl Strategy for EmaCrossStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn generate_signal(&self, klines: &[Kline]) -> Signal {
        use crate::indicators::EMA;

        if klines.len() < self.slow_period + 1 {
            return Signal::Hold;
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();

        let fast_ema = EMA::new(self.fast_period);
        let slow_ema = EMA::new(self.slow_period);

        let fast_values = fast_ema.calculate(&closes);
        let slow_values = slow_ema.calculate(&closes);

        if fast_values.len() < 2 || slow_values.len() < 2 {
            return Signal::Hold;
        }

        let offset = fast_values.len() - slow_values.len();
        let fast_idx = fast_values.len() - 1;
        let slow_idx = slow_values.len() - 1;

        let prev_fast = fast_values[fast_idx - 1];
        let prev_slow = slow_values[slow_idx - 1];
        let curr_fast = fast_values[fast_idx];
        let curr_slow = slow_values[slow_idx];

        SmaCrossStrategy::check_crossover(prev_fast, prev_slow, curr_fast, curr_slow)
    }

    fn min_bars(&self) -> usize {
        self.slow_period + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_klines(prices: &[f64]) -> Vec<Kline> {
        prices
            .iter()
            .enumerate()
            .map(|(i, &price)| Kline {
                timestamp: i as u64 * 60000,
                open: price,
                high: price + 1.0,
                low: price - 1.0,
                close: price,
                volume: 100.0,
            })
            .collect()
    }

    #[test]
    fn test_sma_cross_bullish() {
        // Создаём данные, где быстрая SMA пересекает медленную снизу вверх
        let mut prices: Vec<f64> = (1..=30).map(|i| 100.0 - i as f64).collect();
        // Добавляем резкий рост
        prices.extend((1..=10).map(|i| 80.0 + i as f64 * 2.0));

        let klines = create_klines(&prices);
        let strategy = SmaCrossStrategy::new(5, 10);

        // Проверяем, что в какой-то точке появляется сигнал на покупку
        let signals = strategy.generate_signals(&klines);
        assert!(signals.iter().any(|s| *s == Signal::Buy));
    }

    #[test]
    fn test_sma_cross_bearish() {
        // Создаём данные, где быстрая SMA пересекает медленную сверху вниз
        let mut prices: Vec<f64> = (1..=30).map(|i| 100.0 + i as f64).collect();
        // Добавляем резкое падение
        prices.extend((1..=10).map(|i| 130.0 - i as f64 * 2.0));

        let klines = create_klines(&prices);
        let strategy = SmaCrossStrategy::new(5, 10);

        let signals = strategy.generate_signals(&klines);
        assert!(signals.iter().any(|s| *s == Signal::Sell));
    }

    #[test]
    fn test_insufficient_data() {
        let prices: Vec<f64> = (1..=10).map(|i| 100.0 + i as f64).collect();
        let klines = create_klines(&prices);
        let strategy = SmaCrossStrategy::new(5, 20);

        let signal = strategy.generate_signal(&klines);
        assert_eq!(signal, Signal::Hold);
    }
}
