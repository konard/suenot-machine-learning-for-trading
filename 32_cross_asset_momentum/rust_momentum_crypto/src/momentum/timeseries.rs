//! Time-Series Momentum
//!
//! Этот модуль реализует time-series momentum (absolute momentum),
//! который сравнивает актив с самим собой в прошлом.

use crate::data::{PriceSeries, Signal};
use anyhow::Result;

/// Конфигурация для time-series momentum
#[derive(Debug, Clone)]
pub struct TimeSeriesMomentumConfig {
    /// Период для расчёта моментума (в свечах)
    pub lookback: usize,
    /// Количество свечей для пропуска (избегаем mean reversion)
    pub skip_period: usize,
    /// Порог для сигнала (по умолчанию 0)
    pub threshold: f64,
    /// Безрисковая ставка (годовая, для сравнения)
    pub risk_free_rate: f64,
}

impl Default for TimeSeriesMomentumConfig {
    fn default() -> Self {
        Self {
            lookback: 30,        // 30 дней для криптовалют
            skip_period: 1,      // Пропускаем последний день
            threshold: 0.0,      // Любой положительный return = long
            risk_free_rate: 0.0, // Для крипто обычно 0
        }
    }
}

impl TimeSeriesMomentumConfig {
    /// Создать конфигурацию для криптовалют
    pub fn crypto() -> Self {
        Self {
            lookback: 30,
            skip_period: 1,
            threshold: 0.0,
            risk_free_rate: 0.0,
        }
    }

    /// Создать конфигурацию для традиционных активов
    pub fn traditional() -> Self {
        Self {
            lookback: 252, // Год торговых дней
            skip_period: 21, // Пропускаем последний месяц
            threshold: 0.0,
            risk_free_rate: 0.04, // 4% годовых T-bills
        }
    }
}

/// Time-series momentum калькулятор
#[derive(Debug)]
pub struct TimeSeriesMomentum {
    config: TimeSeriesMomentumConfig,
}

impl TimeSeriesMomentum {
    /// Создать новый калькулятор
    pub fn new(config: TimeSeriesMomentumConfig) -> Self {
        Self { config }
    }

    /// Рассчитать momentum для временного ряда
    ///
    /// Возвращает вектор значений моментума для каждой точки,
    /// где достаточно исторических данных.
    pub fn calculate(&self, series: &PriceSeries) -> Result<Vec<f64>> {
        let closes = series.closes();
        let n = closes.len();
        let total_lookback = self.config.lookback + self.config.skip_period;

        if n < total_lookback + 1 {
            return Ok(Vec::new());
        }

        let mut momentum = Vec::with_capacity(n - total_lookback);

        for i in total_lookback..n {
            // Цена сейчас (с учётом skip_period)
            let current_price = closes[i - self.config.skip_period];
            // Цена lookback периодов назад
            let past_price = closes[i - total_lookback];

            // Процентное изменение
            let return_pct = (current_price - past_price) / past_price;

            momentum.push(return_pct);
        }

        Ok(momentum)
    }

    /// Рассчитать текущий (последний) momentum
    pub fn current_momentum(&self, series: &PriceSeries) -> Result<Option<f64>> {
        let momentum = self.calculate(series)?;
        Ok(momentum.last().copied())
    }

    /// Сгенерировать сигнал на основе моментума
    pub fn signal(&self, momentum: f64) -> Signal {
        // Учитываем безрисковую ставку (приведённую к периоду)
        let periods_per_year = 365.0 / self.config.lookback as f64;
        let period_rf = self.config.risk_free_rate / periods_per_year;

        let excess_return = momentum - period_rf;

        if excess_return > self.config.threshold {
            Signal::Long
        } else {
            Signal::Cash
        }
    }

    /// Сгенерировать сигнал для временного ряда
    pub fn generate_signal(&self, series: &PriceSeries) -> Result<Signal> {
        match self.current_momentum(series)? {
            Some(mom) => Ok(self.signal(mom)),
            None => Ok(Signal::Neutral), // Недостаточно данных
        }
    }

    /// Рассчитать моментум для нескольких периодов
    pub fn multi_period_momentum(
        series: &PriceSeries,
        periods: &[usize],
        weights: &[f64],
    ) -> Result<Option<f64>> {
        if periods.len() != weights.len() {
            anyhow::bail!("Periods and weights must have the same length");
        }

        let closes = series.closes();
        let n = closes.len();

        let max_period = *periods.iter().max().unwrap_or(&0);
        if n < max_period + 1 {
            return Ok(None);
        }

        let mut weighted_mom = 0.0;
        let mut total_weight = 0.0;

        for (&period, &weight) in periods.iter().zip(weights.iter()) {
            if n > period {
                let current = closes[n - 1];
                let past = closes[n - 1 - period];
                let mom = (current - past) / past;
                weighted_mom += mom * weight;
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            Ok(Some(weighted_mom / total_weight))
        } else {
            Ok(None)
        }
    }
}

/// Удобные функции для быстрого расчёта

/// Рассчитать простой моментум за период
pub fn simple_momentum(closes: &[f64], lookback: usize) -> Option<f64> {
    if closes.len() <= lookback {
        return None;
    }

    let current = closes[closes.len() - 1];
    let past = closes[closes.len() - 1 - lookback];

    Some((current - past) / past)
}

/// Рассчитать моментум с пропуском последних периодов
pub fn momentum_skip(closes: &[f64], lookback: usize, skip: usize) -> Option<f64> {
    if closes.len() <= lookback + skip {
        return None;
    }

    let current = closes[closes.len() - 1 - skip];
    let past = closes[closes.len() - 1 - lookback - skip];

    Some((current - past) / past)
}

/// Рассчитать скользящий моментум
pub fn rolling_momentum(closes: &[f64], lookback: usize) -> Vec<f64> {
    if closes.len() <= lookback {
        return Vec::new();
    }

    closes
        .windows(lookback + 1)
        .map(|w| (w[lookback] - w[0]) / w[0])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Candle;
    use chrono::Utc;

    fn create_test_series(prices: Vec<f64>) -> PriceSeries {
        let mut series = PriceSeries::new("TEST".to_string(), "D".to_string());
        for (i, price) in prices.iter().enumerate() {
            let candle = Candle::new(
                Utc::now() + chrono::Duration::days(i as i64),
                *price,
                *price,
                *price,
                *price,
                1000.0,
            );
            series.push(candle);
        }
        series
    }

    #[test]
    fn test_simple_momentum() {
        let closes = vec![100.0, 105.0, 110.0, 115.0, 120.0];
        let mom = simple_momentum(&closes, 4).unwrap();
        assert!((mom - 0.2).abs() < 1e-10); // 20% рост
    }

    #[test]
    fn test_momentum_signal() {
        let config = TimeSeriesMomentumConfig {
            lookback: 3,
            skip_period: 0,
            threshold: 0.0,
            risk_free_rate: 0.0,
        };
        let calc = TimeSeriesMomentum::new(config);

        // Положительный моментум
        assert_eq!(calc.signal(0.1), Signal::Long);
        // Отрицательный моментум
        assert_eq!(calc.signal(-0.1), Signal::Cash);
    }

    #[test]
    fn test_time_series_momentum_calculate() {
        let prices = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0];
        let series = create_test_series(prices);

        let config = TimeSeriesMomentumConfig {
            lookback: 3,
            skip_period: 0,
            threshold: 0.0,
            risk_free_rate: 0.0,
        };
        let calc = TimeSeriesMomentum::new(config);

        let momentum = calc.calculate(&series).unwrap();
        assert!(!momentum.is_empty());

        // Все значения должны быть положительными (восходящий тренд)
        for m in &momentum {
            assert!(*m > 0.0);
        }
    }

    #[test]
    fn test_rolling_momentum() {
        let closes = vec![100.0, 110.0, 121.0, 133.1, 146.41];
        let momentum = rolling_momentum(&closes, 1);

        assert_eq!(momentum.len(), 4);
        // Каждый период рост ~10%
        for m in &momentum {
            assert!((*m - 0.1).abs() < 0.01);
        }
    }
}
