//! Генерация торговых сигналов
//!
//! Этот модуль содержит функции для генерации торговых сигналов
//! на основе моментум стратегий.

use crate::data::{PriceSeries, Signal, Signals};
use crate::momentum::{DualMomentum, DualMomentumConfig};
use anyhow::Result;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Генератор сигналов
#[derive(Debug)]
pub struct SignalGenerator {
    /// Стратегия dual momentum
    dual_momentum: DualMomentum,
    /// Минимальный моментум для входа
    min_momentum_threshold: f64,
    /// Максимальное количество активов в портфеле
    max_positions: usize,
}

impl SignalGenerator {
    /// Создать новый генератор
    pub fn new(config: DualMomentumConfig) -> Self {
        let max_positions = config.top_n;
        Self {
            dual_momentum: DualMomentum::new(config),
            min_momentum_threshold: 0.0,
            max_positions,
        }
    }

    /// Установить минимальный порог моментума
    pub fn with_min_momentum(mut self, threshold: f64) -> Self {
        self.min_momentum_threshold = threshold;
        self
    }

    /// Генерировать сигналы
    pub fn generate(
        &self,
        price_data: &HashMap<String, PriceSeries>,
        timestamp: DateTime<Utc>,
    ) -> Result<Signals> {
        // Используем dual momentum для генерации сигналов
        let analysis = self.dual_momentum.analyze(price_data)?;

        let mut signals = Signals::new(timestamp);

        for result in &analysis {
            // Проверяем минимальный порог
            if result.ts_momentum < self.min_momentum_threshold {
                signals.set(&result.symbol, Signal::Cash);
                continue;
            }

            let signal = if result.selected {
                Signal::Long
            } else if !result.ts_passed {
                Signal::Cash
            } else {
                Signal::Neutral
            };

            signals.set(&result.symbol, signal);
        }

        Ok(signals)
    }

    /// Проверить, нужно ли ребалансировать
    pub fn should_rebalance(
        &self,
        current_signals: &Signals,
        new_signals: &Signals,
        drift_threshold: f64,
    ) -> bool {
        // Проверяем, изменились ли сигналы
        for (symbol, &new_signal) in &new_signals.signals {
            let current = current_signals.get(symbol);
            if current != new_signal {
                return true;
            }
        }

        false
    }
}

/// Простой генератор сигналов на основе моментума
pub fn generate_simple_signals(
    price_data: &HashMap<String, PriceSeries>,
    lookback: usize,
    top_n: usize,
    timestamp: DateTime<Utc>,
) -> Result<Signals> {
    let mut momentum_values: Vec<(String, f64)> = Vec::new();

    for (symbol, series) in price_data {
        let closes = series.closes();
        if closes.len() > lookback {
            let current = closes[closes.len() - 1];
            let past = closes[closes.len() - 1 - lookback];
            let momentum = (current - past) / past;
            momentum_values.push((symbol.clone(), momentum));
        }
    }

    // Сортируем по убыванию моментума
    momentum_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut signals = Signals::new(timestamp);

    for (i, (symbol, momentum)) in momentum_values.iter().enumerate() {
        let signal = if *momentum > 0.0 && i < top_n {
            Signal::Long
        } else if *momentum <= 0.0 {
            Signal::Cash
        } else {
            Signal::Neutral
        };

        signals.set(symbol, signal);
    }

    Ok(signals)
}

/// Генерировать сигнал на основе множественных периодов моментума
pub fn multi_timeframe_signal(
    closes: &[f64],
    periods: &[usize],
    weights: &[f64],
) -> Signal {
    if periods.len() != weights.len() {
        return Signal::Neutral;
    }

    let n = closes.len();
    let max_period = *periods.iter().max().unwrap_or(&0);

    if n <= max_period {
        return Signal::Neutral;
    }

    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    for (&period, &weight) in periods.iter().zip(weights.iter()) {
        if n > period {
            let current = closes[n - 1];
            let past = closes[n - 1 - period];
            let momentum = (current - past) / past;

            // Преобразуем в сигнал: положительный = 1, отрицательный = -1
            let signal_value = if momentum > 0.0 { 1.0 } else { -1.0 };
            weighted_sum += signal_value * weight;
            total_weight += weight;
        }
    }

    if total_weight == 0.0 {
        return Signal::Neutral;
    }

    let consensus = weighted_sum / total_weight;

    if consensus > 0.5 {
        Signal::Long
    } else if consensus < -0.5 {
        Signal::Short
    } else {
        Signal::Neutral
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Candle;

    fn create_test_series(symbol: &str, prices: Vec<f64>) -> PriceSeries {
        let mut series = PriceSeries::new(symbol.to_string(), "D".to_string());
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
    fn test_generate_simple_signals() {
        let mut price_data = HashMap::new();

        price_data.insert(
            "BTC".to_string(),
            create_test_series("BTC", vec![100.0, 110.0, 120.0, 130.0, 140.0]),
        );

        price_data.insert(
            "ETH".to_string(),
            create_test_series("ETH", vec![100.0, 105.0, 110.0, 115.0, 120.0]),
        );

        price_data.insert(
            "SOL".to_string(),
            create_test_series("SOL", vec![100.0, 95.0, 90.0, 85.0, 80.0]),
        );

        let signals = generate_simple_signals(&price_data, 3, 2, Utc::now()).unwrap();

        // BTC и ETH должны быть Long
        assert_eq!(signals.get("BTC"), Signal::Long);
        assert_eq!(signals.get("ETH"), Signal::Long);

        // SOL должен быть Cash (отрицательный моментум)
        assert_eq!(signals.get("SOL"), Signal::Cash);
    }

    #[test]
    fn test_multi_timeframe_signal() {
        // Восходящий тренд на всех таймфреймах
        let closes = vec![100.0, 102.0, 105.0, 108.0, 112.0, 116.0, 120.0];
        let periods = vec![3, 5];
        let weights = vec![0.5, 0.5];

        let signal = multi_timeframe_signal(&closes, &periods, &weights);
        assert_eq!(signal, Signal::Long);

        // Нисходящий тренд
        let falling = vec![120.0, 116.0, 112.0, 108.0, 105.0, 102.0, 100.0];
        let signal = multi_timeframe_signal(&falling, &periods, &weights);
        assert_eq!(signal, Signal::Short);
    }
}
