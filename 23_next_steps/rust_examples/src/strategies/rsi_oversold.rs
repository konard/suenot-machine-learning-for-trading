//! Стратегия на основе RSI (перекупленность/перепроданность)

use crate::data::Kline;
use crate::indicators::{Indicator, RSI};
use super::{Signal, Strategy};

/// Стратегия на основе RSI
///
/// Генерирует сигнал на покупку при перепроданности (RSI < oversold)
/// и сигнал на продажу при перекупленности (RSI > overbought).
///
/// # Пример
/// ```
/// use ml4t_bybit::strategies::{RsiStrategy, Strategy};
///
/// // RSI(14) с уровнями 30/70
/// let strategy = RsiStrategy::new(14, 30.0, 70.0);
/// ```
#[derive(Debug, Clone)]
pub struct RsiStrategy {
    name: String,
    period: usize,
    oversold: f64,
    overbought: f64,
    rsi: RSI,
}

impl RsiStrategy {
    /// Создать стратегию с указанными параметрами
    pub fn new(period: usize, oversold: f64, overbought: f64) -> Self {
        assert!(oversold < overbought, "Oversold must be less than overbought");
        assert!(
            (0.0..=100.0).contains(&oversold) && (0.0..=100.0).contains(&overbought),
            "Thresholds must be between 0 and 100"
        );

        Self {
            name: format!("RSI({}) {}/{}", period, oversold, overbought),
            period,
            oversold,
            overbought,
            rsi: RSI::new(period),
        }
    }

    /// Стандартная стратегия RSI(14) с уровнями 30/70
    pub fn standard() -> Self {
        Self::new(14, 30.0, 70.0)
    }

    /// Агрессивная стратегия с уровнями 20/80
    pub fn aggressive() -> Self {
        Self::new(14, 20.0, 80.0)
    }

    /// Консервативная стратегия с уровнями 25/75
    pub fn conservative() -> Self {
        Self::new(14, 25.0, 75.0)
    }

    /// Краткосрочная стратегия RSI(7)
    pub fn short_term() -> Self {
        Self::new(7, 30.0, 70.0)
    }

    /// Получить текущее значение RSI
    pub fn current_rsi(&self, klines: &[Kline]) -> Option<f64> {
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        self.rsi.current(&closes)
    }

    /// Получить параметры
    pub fn params(&self) -> (usize, f64, f64) {
        (self.period, self.oversold, self.overbought)
    }
}

impl Strategy for RsiStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn generate_signal(&self, klines: &[Kline]) -> Signal {
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();

        if let Some(rsi) = self.rsi.current(&closes) {
            if rsi < self.oversold {
                return Signal::Buy;
            }
            if rsi > self.overbought {
                return Signal::Sell;
            }
        }

        Signal::Hold
    }

    fn min_bars(&self) -> usize {
        self.period + 1
    }
}

/// Стратегия RSI с дивергенцией
#[derive(Debug, Clone)]
pub struct RsiDivergenceStrategy {
    name: String,
    period: usize,
    lookback: usize,
    rsi: RSI,
}

impl RsiDivergenceStrategy {
    /// Создать стратегию дивергенции RSI
    pub fn new(period: usize, lookback: usize) -> Self {
        Self {
            name: format!("RSI Divergence({}, {})", period, lookback),
            period,
            lookback,
            rsi: RSI::new(period),
        }
    }

    /// Найти локальные минимумы/максимумы
    fn find_extremes(values: &[f64], lookback: usize) -> (Vec<usize>, Vec<usize>) {
        let mut mins = Vec::new();
        let mut maxs = Vec::new();

        for i in lookback..values.len() - lookback {
            let window = &values[i - lookback..=i + lookback];
            let current = values[i];

            if window.iter().all(|&v| v >= current) {
                mins.push(i);
            }
            if window.iter().all(|&v| v <= current) {
                maxs.push(i);
            }
        }

        (mins, maxs)
    }

    /// Проверить бычью дивергенцию (цена делает новый минимум, RSI — нет)
    fn check_bullish_divergence(
        prices: &[f64],
        rsi_values: &[f64],
        mins: &[usize],
    ) -> bool {
        if mins.len() < 2 {
            return false;
        }

        let last_min = mins[mins.len() - 1];
        let prev_min = mins[mins.len() - 2];

        // Цена делает новый минимум
        let price_lower = prices[last_min] < prices[prev_min];
        // RSI НЕ делает новый минимум
        let rsi_higher = rsi_values[last_min] > rsi_values[prev_min];

        price_lower && rsi_higher
    }

    /// Проверить медвежью дивергенцию (цена делает новый максимум, RSI — нет)
    fn check_bearish_divergence(
        prices: &[f64],
        rsi_values: &[f64],
        maxs: &[usize],
    ) -> bool {
        if maxs.len() < 2 {
            return false;
        }

        let last_max = maxs[maxs.len() - 1];
        let prev_max = maxs[maxs.len() - 2];

        // Цена делает новый максимум
        let price_higher = prices[last_max] > prices[prev_max];
        // RSI НЕ делает новый максимум
        let rsi_lower = rsi_values[last_max] < rsi_values[prev_max];

        price_higher && rsi_lower
    }
}

impl Strategy for RsiDivergenceStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn generate_signal(&self, klines: &[Kline]) -> Signal {
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let rsi_values = self.rsi.calculate(&closes);

        if rsi_values.len() < self.lookback * 2 + 1 {
            return Signal::Hold;
        }

        // Выравниваем длины
        let offset = closes.len() - rsi_values.len();
        let aligned_prices = &closes[offset..];

        let (mins, maxs) = Self::find_extremes(&rsi_values, self.lookback);

        if Self::check_bullish_divergence(aligned_prices, &rsi_values, &mins) {
            return Signal::Buy;
        }

        if Self::check_bearish_divergence(aligned_prices, &rsi_values, &maxs) {
            return Signal::Sell;
        }

        Signal::Hold
    }

    fn min_bars(&self) -> usize {
        self.period + self.lookback * 2 + 1
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
    fn test_rsi_oversold() {
        // Создаём сильное падение для перепроданности
        let prices: Vec<f64> = (0..30).map(|i| 100.0 - i as f64 * 2.0).collect();
        let klines = create_klines(&prices);

        let strategy = RsiStrategy::standard();
        let signal = strategy.generate_signal(&klines);

        // При сильном падении RSI должен быть низким -> сигнал Buy
        assert_eq!(signal, Signal::Buy);
    }

    #[test]
    fn test_rsi_overbought() {
        // Создаём сильный рост для перекупленности
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 2.0).collect();
        let klines = create_klines(&prices);

        let strategy = RsiStrategy::standard();
        let signal = strategy.generate_signal(&klines);

        // При сильном росте RSI должен быть высоким -> сигнал Sell
        assert_eq!(signal, Signal::Sell);
    }

    #[test]
    fn test_rsi_hold() {
        // Создаём боковое движение
        let prices: Vec<f64> = (0..30)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 2.0)
            .collect();
        let klines = create_klines(&prices);

        let strategy = RsiStrategy::standard();
        let signal = strategy.generate_signal(&klines);

        // При боковом движении должен быть Hold
        assert_eq!(signal, Signal::Hold);
    }

    #[test]
    fn test_current_rsi() {
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let klines = create_klines(&prices);

        let strategy = RsiStrategy::standard();
        let rsi = strategy.current_rsi(&klines);

        assert!(rsi.is_some());
        let rsi_value = rsi.unwrap();
        assert!(rsi_value >= 0.0 && rsi_value <= 100.0);
    }
}
