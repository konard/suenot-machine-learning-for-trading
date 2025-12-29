//! Расчёт весов портфеля
//!
//! Этот модуль содержит функции для расчёта весов активов в портфеле
//! с учётом волатильности и risk parity.

use crate::data::{Portfolio, PriceSeries, Signals};
use anyhow::Result;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Конфигурация для расчёта весов
#[derive(Debug, Clone)]
pub struct WeightConfig {
    /// Целевая волатильность портфеля (годовая)
    pub target_volatility: f64,
    /// Период для расчёта волатильности
    pub volatility_lookback: usize,
    /// Максимальный вес одного актива
    pub max_weight: f64,
    /// Минимальный вес для включения в портфель
    pub min_weight: f64,
    /// Использовать risk parity
    pub use_risk_parity: bool,
}

impl Default for WeightConfig {
    fn default() -> Self {
        Self {
            target_volatility: 0.30, // 30% для крипто
            volatility_lookback: 30,
            max_weight: 0.40,
            min_weight: 0.05,
            use_risk_parity: true,
        }
    }
}

impl WeightConfig {
    /// Конфигурация для криптовалют
    pub fn crypto() -> Self {
        Self {
            target_volatility: 0.30,
            volatility_lookback: 30,
            max_weight: 0.40,
            min_weight: 0.05,
            use_risk_parity: true,
        }
    }

    /// Конфигурация для equal weight
    pub fn equal_weight() -> Self {
        Self {
            target_volatility: 1.0,
            volatility_lookback: 30,
            max_weight: 1.0,
            min_weight: 0.0,
            use_risk_parity: false,
        }
    }
}

/// Калькулятор весов портфеля
#[derive(Debug)]
pub struct WeightCalculator {
    config: WeightConfig,
}

impl WeightCalculator {
    /// Создать новый калькулятор
    pub fn new(config: WeightConfig) -> Self {
        Self { config }
    }

    /// Рассчитать волатильность актива
    pub fn calculate_volatility(&self, series: &PriceSeries) -> Option<f64> {
        let returns = series.returns();

        if returns.len() < self.config.volatility_lookback {
            return None;
        }

        // Берём последние N returns
        let recent_returns: Vec<f64> = returns
            .iter()
            .rev()
            .take(self.config.volatility_lookback)
            .copied()
            .collect();

        let n = recent_returns.len() as f64;
        let mean = recent_returns.iter().sum::<f64>() / n;
        let variance = recent_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let daily_vol = variance.sqrt();

        // Аннуализированная волатильность (365 дней для крипто)
        Some(daily_vol * (365.0_f64).sqrt())
    }

    /// Рассчитать inverse volatility weights
    pub fn inverse_volatility_weights(
        &self,
        price_data: &HashMap<String, PriceSeries>,
        selected_symbols: &[String],
    ) -> Result<HashMap<String, f64>> {
        let mut volatilities: HashMap<String, f64> = HashMap::new();

        for symbol in selected_symbols {
            if let Some(series) = price_data.get(symbol) {
                if let Some(vol) = self.calculate_volatility(series) {
                    if vol > 0.0 {
                        volatilities.insert(symbol.clone(), vol);
                    }
                }
            }
        }

        if volatilities.is_empty() {
            return Ok(HashMap::new());
        }

        // Inverse volatility weights
        let mut weights: HashMap<String, f64> = HashMap::new();
        let total_inv_vol: f64 = volatilities.values().map(|v| 1.0 / v).sum();

        for (symbol, vol) in &volatilities {
            let weight = (1.0 / vol) / total_inv_vol;
            weights.insert(symbol.clone(), weight);
        }

        Ok(weights)
    }

    /// Рассчитать веса с таргетированием волатильности
    pub fn volatility_targeted_weights(
        &self,
        price_data: &HashMap<String, PriceSeries>,
        selected_symbols: &[String],
    ) -> Result<HashMap<String, f64>> {
        let mut weights: HashMap<String, f64> = HashMap::new();

        for symbol in selected_symbols {
            if let Some(series) = price_data.get(symbol) {
                if let Some(vol) = self.calculate_volatility(series) {
                    if vol > 0.0 {
                        // Вес пропорционален target_vol / realized_vol
                        let raw_weight = self.config.target_volatility / vol;
                        let capped_weight = raw_weight.min(self.config.max_weight);
                        weights.insert(symbol.clone(), capped_weight);
                    }
                }
            }
        }

        // Нормализуем, если сумма > 1
        let total: f64 = weights.values().sum();
        if total > 1.0 {
            for weight in weights.values_mut() {
                *weight /= total;
            }
        }

        Ok(weights)
    }

    /// Рассчитать equal weights
    pub fn equal_weights(symbols: &[String]) -> HashMap<String, f64> {
        let weight = if symbols.is_empty() {
            0.0
        } else {
            1.0 / symbols.len() as f64
        };

        symbols.iter().map(|s| (s.clone(), weight)).collect()
    }

    /// Рассчитать веса на основе сигналов
    pub fn weights_from_signals(
        &self,
        price_data: &HashMap<String, PriceSeries>,
        signals: &Signals,
    ) -> Result<HashMap<String, f64>> {
        // Собираем символы с сигналом Long
        let long_symbols: Vec<String> = signals.long_symbols().into_iter().cloned().collect();

        if long_symbols.is_empty() {
            return Ok(HashMap::new());
        }

        if self.config.use_risk_parity {
            self.inverse_volatility_weights(price_data, &long_symbols)
        } else {
            Ok(Self::equal_weights(&long_symbols))
        }
    }

    /// Создать портфель из весов
    pub fn create_portfolio(
        &self,
        price_data: &HashMap<String, PriceSeries>,
        signals: &Signals,
        timestamp: DateTime<Utc>,
    ) -> Result<Portfolio> {
        let weights = self.weights_from_signals(price_data, signals)?;

        let mut portfolio = Portfolio::new(timestamp);
        for (symbol, weight) in weights {
            if weight >= self.config.min_weight {
                portfolio.set_weight(&symbol, weight);
            }
        }

        Ok(portfolio)
    }
}

/// Рассчитать простую скользящую волатильность
pub fn rolling_volatility(returns: &[f64], window: usize) -> Vec<f64> {
    if returns.len() < window {
        return Vec::new();
    }

    returns
        .windows(window)
        .map(|w| {
            let n = w.len() as f64;
            let mean = w.iter().sum::<f64>() / n;
            let variance = w.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
            variance.sqrt()
        })
        .collect()
}

/// Рассчитать экспоненциальную скользящую волатильность (EWMA)
pub fn ewma_volatility(returns: &[f64], lambda: f64) -> Vec<f64> {
    if returns.is_empty() {
        return Vec::new();
    }

    let mut variance = returns[0].powi(2);
    let mut volatilities = vec![variance.sqrt()];

    for &ret in returns.iter().skip(1) {
        variance = lambda * variance + (1.0 - lambda) * ret.powi(2);
        volatilities.push(variance.sqrt());
    }

    volatilities
}

/// Рассчитать корреляцию между двумя рядами
pub fn correlation(series1: &[f64], series2: &[f64]) -> Option<f64> {
    if series1.len() != series2.len() || series1.is_empty() {
        return None;
    }

    let n = series1.len() as f64;
    let mean1 = series1.iter().sum::<f64>() / n;
    let mean2 = series2.iter().sum::<f64>() / n;

    let cov: f64 = series1
        .iter()
        .zip(series2.iter())
        .map(|(a, b)| (a - mean1) * (b - mean2))
        .sum::<f64>()
        / n;

    let std1 = (series1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / n).sqrt();
    let std2 = (series2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / n).sqrt();

    if std1 == 0.0 || std2 == 0.0 {
        return None;
    }

    Some(cov / (std1 * std2))
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
    fn test_rolling_volatility() {
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.01, -0.01];
        let vol = rolling_volatility(&returns, 3);

        assert!(!vol.is_empty());
        assert!(vol.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn test_correlation() {
        // Идеальная положительная корреляция
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let corr = correlation(&series1, &series2).unwrap();
        assert!((corr - 1.0).abs() < 1e-10);

        // Отрицательная корреляция
        let series3 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr = correlation(&series1, &series3).unwrap();
        assert!((corr - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_equal_weights() {
        let symbols = vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()];
        let weights = WeightCalculator::equal_weights(&symbols);

        assert_eq!(weights.len(), 3);
        for (_, weight) in &weights {
            assert!((*weight - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_inverse_volatility_weights() {
        let config = WeightConfig::default();
        let calc = WeightCalculator::new(config);

        let mut price_data = HashMap::new();

        // Создаём серии с разной волатильностью
        // Низкая волатильность
        let stable_prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        price_data.insert("STABLE".to_string(), create_test_series("STABLE", stable_prices));

        // Высокая волатильность
        let volatile_prices: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.5 + (i % 3) as f64 * 5.0 - 5.0)
            .collect();
        price_data.insert("VOLATILE".to_string(), create_test_series("VOLATILE", volatile_prices));

        let selected = vec!["STABLE".to_string(), "VOLATILE".to_string()];
        let weights = calc.inverse_volatility_weights(&price_data, &selected).unwrap();

        // STABLE должен иметь больший вес (меньшая волатильность)
        if let (Some(&stable_w), Some(&volatile_w)) = (weights.get("STABLE"), weights.get("VOLATILE")) {
            assert!(stable_w > volatile_w);
        }
    }
}
