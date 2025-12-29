//! Dual Momentum
//!
//! Этот модуль реализует Dual Momentum стратегию,
//! которая комбинирует time-series и cross-sectional momentum.

use super::crosssection::CrossSectionalMomentum;
use super::timeseries::TimeSeriesMomentum;
use crate::data::{Portfolio, PriceSeries, Signal, Signals};
use anyhow::Result;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Конфигурация для Dual Momentum
#[derive(Debug, Clone)]
pub struct DualMomentumConfig {
    /// Период для time-series momentum
    pub ts_lookback: usize,
    /// Период для cross-sectional momentum
    pub cs_lookback: usize,
    /// Количество топ активов для покупки
    pub top_n: usize,
    /// Безрисковая ставка (годовая)
    pub risk_free_rate: f64,
    /// Пропуск периода для избежания mean reversion
    pub skip_period: usize,
    /// Использовать equal weight для выбранных активов
    pub equal_weight: bool,
}

impl Default for DualMomentumConfig {
    fn default() -> Self {
        Self {
            ts_lookback: 30,
            cs_lookback: 30,
            top_n: 3,
            risk_free_rate: 0.0,
            skip_period: 1,
            equal_weight: true,
        }
    }
}

impl DualMomentumConfig {
    /// Конфигурация для криптовалют
    pub fn crypto() -> Self {
        Self {
            ts_lookback: 30,
            cs_lookback: 30,
            top_n: 3,
            risk_free_rate: 0.0,
            skip_period: 1,
            equal_weight: true,
        }
    }

    /// Конфигурация как в оригинальной книге Antonacci
    pub fn antonacci() -> Self {
        Self {
            ts_lookback: 252, // 12 месяцев
            cs_lookback: 252,
            top_n: 1, // Выбор только лучшего актива
            risk_free_rate: 0.0,
            skip_period: 0,
            equal_weight: true,
        }
    }
}

/// Dual Momentum стратегия
#[derive(Debug)]
pub struct DualMomentum {
    config: DualMomentumConfig,
}

/// Результат анализа dual momentum для актива
#[derive(Debug, Clone)]
pub struct DualMomentumResult {
    /// Символ актива
    pub symbol: String,
    /// Time-series momentum
    pub ts_momentum: f64,
    /// Cross-sectional ранг
    pub cs_rank: usize,
    /// Прошёл time-series фильтр?
    pub ts_passed: bool,
    /// Выбран для покупки?
    pub selected: bool,
    /// Вес в портфеле
    pub weight: f64,
}

impl DualMomentum {
    /// Создать новую стратегию
    pub fn new(config: DualMomentumConfig) -> Self {
        Self { config }
    }

    /// Анализировать все активы
    pub fn analyze(
        &self,
        price_data: &HashMap<String, PriceSeries>,
    ) -> Result<Vec<DualMomentumResult>> {
        let mut results = Vec::new();

        // Шаг 1: Рассчитываем time-series momentum для каждого актива
        let mut ts_momentum: HashMap<String, f64> = HashMap::new();

        for (symbol, series) in price_data {
            let closes = series.closes();
            let total_lookback = self.config.ts_lookback + self.config.skip_period;

            if closes.len() > total_lookback {
                let current = closes[closes.len() - 1 - self.config.skip_period];
                let past = closes[closes.len() - 1 - total_lookback];
                let momentum = (current - past) / past;
                ts_momentum.insert(symbol.clone(), momentum);
            }
        }

        // Шаг 2: Фильтруем активы с положительным абсолютным моментумом
        let period_rf = self.config.risk_free_rate * (self.config.ts_lookback as f64 / 365.0);

        let passed_ts: Vec<(&String, f64)> = ts_momentum
            .iter()
            .filter(|(_, &mom)| mom > period_rf)
            .map(|(s, &m)| (s, m))
            .collect();

        // Шаг 3: Ранжируем по cross-sectional momentum
        // Используем тот же моментум, но только для тех, кто прошёл фильтр
        let mut sorted_passed: Vec<_> = passed_ts.clone();
        sorted_passed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Шаг 4: Выбираем топ N активов
        let selected: Vec<&String> = sorted_passed
            .iter()
            .take(self.config.top_n)
            .map(|(s, _)| *s)
            .collect();

        // Рассчитываем веса
        let weight = if self.config.equal_weight && !selected.is_empty() {
            1.0 / selected.len() as f64
        } else {
            0.0
        };

        // Формируем результаты для всех активов
        for (symbol, &momentum) in &ts_momentum {
            let ts_passed = momentum > period_rf;

            // Находим ранг среди прошедших фильтр
            let cs_rank = sorted_passed
                .iter()
                .position(|(s, _)| s == &symbol)
                .map(|p| p + 1)
                .unwrap_or(usize::MAX);

            let is_selected = selected.contains(&symbol);

            results.push(DualMomentumResult {
                symbol: symbol.clone(),
                ts_momentum: momentum,
                cs_rank,
                ts_passed,
                selected: is_selected,
                weight: if is_selected { weight } else { 0.0 },
            });
        }

        // Сортируем по моментуму
        results.sort_by(|a, b| {
            b.ts_momentum
                .partial_cmp(&a.ts_momentum)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Сгенерировать сигналы
    pub fn generate_signals(
        &self,
        price_data: &HashMap<String, PriceSeries>,
        timestamp: DateTime<Utc>,
    ) -> Result<Signals> {
        let analysis = self.analyze(price_data)?;
        let mut signals = Signals::new(timestamp);

        for result in analysis {
            let signal = if result.selected {
                Signal::Long
            } else if !result.ts_passed {
                Signal::Cash // Отрицательный моментум = кеш
            } else {
                Signal::Neutral // Прошёл фильтр, но не в топе
            };

            signals.set(&result.symbol, signal);
        }

        Ok(signals)
    }

    /// Сгенерировать портфель
    pub fn generate_portfolio(
        &self,
        price_data: &HashMap<String, PriceSeries>,
        timestamp: DateTime<Utc>,
    ) -> Result<Portfolio> {
        let analysis = self.analyze(price_data)?;
        let mut portfolio = Portfolio::new(timestamp);

        for result in analysis {
            if result.selected {
                portfolio.set_weight(&result.symbol, result.weight);
            }
        }

        Ok(portfolio)
    }

    /// Проверить, нужно ли уходить в кеш
    pub fn should_go_to_cash(&self, price_data: &HashMap<String, PriceSeries>) -> Result<bool> {
        let analysis = self.analyze(price_data)?;

        // Если ни один актив не прошёл time-series фильтр - уходим в кеш
        let any_passed = analysis.iter().any(|r| r.ts_passed);

        Ok(!any_passed)
    }

    /// Получить выбранные активы
    pub fn selected_assets(
        &self,
        price_data: &HashMap<String, PriceSeries>,
    ) -> Result<Vec<DualMomentumResult>> {
        let analysis = self.analyze(price_data)?;
        Ok(analysis.into_iter().filter(|r| r.selected).collect())
    }
}

/// Функция для быстрого расчёта dual momentum сигнала
pub fn quick_dual_momentum_signal(
    closes: &[f64],
    lookback: usize,
    threshold: f64,
) -> Signal {
    if closes.len() <= lookback {
        return Signal::Neutral;
    }

    let current = closes[closes.len() - 1];
    let past = closes[closes.len() - 1 - lookback];
    let momentum = (current - past) / past;

    if momentum > threshold {
        Signal::Long
    } else {
        Signal::Cash
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
    fn test_dual_momentum_analysis() {
        let mut price_data = HashMap::new();

        // BTC: сильный рост
        price_data.insert(
            "BTCUSDT".to_string(),
            create_test_series("BTCUSDT", vec![100.0, 110.0, 120.0, 130.0, 140.0]),
        );

        // ETH: умеренный рост
        price_data.insert(
            "ETHUSDT".to_string(),
            create_test_series("ETHUSDT", vec![100.0, 105.0, 110.0, 115.0, 120.0]),
        );

        // SOL: падение
        price_data.insert(
            "SOLUSDT".to_string(),
            create_test_series("SOLUSDT", vec![100.0, 95.0, 90.0, 85.0, 80.0]),
        );

        let config = DualMomentumConfig {
            ts_lookback: 3,
            cs_lookback: 3,
            top_n: 2,
            risk_free_rate: 0.0,
            skip_period: 0,
            equal_weight: true,
        };

        let strategy = DualMomentum::new(config);
        let analysis = strategy.analyze(&price_data).unwrap();

        // BTC и ETH должны быть выбраны
        let selected: Vec<_> = analysis.iter().filter(|a| a.selected).collect();
        assert_eq!(selected.len(), 2);

        // SOL не должен пройти time-series фильтр
        let sol = analysis.iter().find(|a| a.symbol == "SOLUSDT").unwrap();
        assert!(!sol.ts_passed);
        assert!(!sol.selected);
    }

    #[test]
    fn test_go_to_cash() {
        let mut price_data = HashMap::new();

        // Все активы падают
        price_data.insert(
            "BTCUSDT".to_string(),
            create_test_series("BTCUSDT", vec![100.0, 95.0, 90.0, 85.0, 80.0]),
        );

        price_data.insert(
            "ETHUSDT".to_string(),
            create_test_series("ETHUSDT", vec![100.0, 90.0, 80.0, 70.0, 60.0]),
        );

        let config = DualMomentumConfig {
            ts_lookback: 3,
            cs_lookback: 3,
            top_n: 2,
            risk_free_rate: 0.0,
            skip_period: 0,
            equal_weight: true,
        };

        let strategy = DualMomentum::new(config);

        // Должны уйти в кеш
        assert!(strategy.should_go_to_cash(&price_data).unwrap());
    }

    #[test]
    fn test_quick_signal() {
        let closes = vec![100.0, 110.0, 120.0, 130.0, 140.0];

        // Положительный моментум
        assert_eq!(quick_dual_momentum_signal(&closes, 3, 0.0), Signal::Long);

        let falling = vec![100.0, 95.0, 90.0, 85.0, 80.0];

        // Отрицательный моментум
        assert_eq!(quick_dual_momentum_signal(&falling, 3, 0.0), Signal::Cash);
    }
}
