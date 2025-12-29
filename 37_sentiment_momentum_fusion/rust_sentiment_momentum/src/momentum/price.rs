//! Ценовой моментум
//!
//! Расчёт моментума на основе ценовых данных.

use super::{calculate_z_scores, MomentumResult};
use serde::{Deserialize, Serialize};

/// Ценовой моментум с различными метриками
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceMomentum {
    /// Rate of Change (ROC)
    pub roc: f64,
    /// Скользящее среднее
    pub sma: f64,
    /// Отклонение от SMA
    pub deviation_from_sma: f64,
    /// RSI-подобный индикатор
    pub rsi: f64,
    /// Период расчёта
    pub period: usize,
}

impl PriceMomentum {
    /// Расчёт ценового моментума
    pub fn calculate(prices: &[f64], period: usize) -> Option<Self> {
        if prices.len() < period + 1 {
            return None;
        }

        let current = *prices.last()?;
        let past = prices[prices.len() - period - 1];

        // Rate of Change
        let roc = if past != 0.0 {
            (current - past) / past
        } else {
            0.0
        };

        // SMA
        let recent_prices = &prices[prices.len() - period..];
        let sma = recent_prices.iter().sum::<f64>() / period as f64;

        // Отклонение от SMA
        let deviation_from_sma = if sma != 0.0 {
            (current - sma) / sma
        } else {
            0.0
        };

        // RSI-подобный расчёт
        let rsi = Self::calculate_rsi(&prices[prices.len() - period - 1..], period);

        Some(Self {
            roc,
            sma,
            deviation_from_sma,
            rsi,
            period,
        })
    }

    /// Расчёт RSI (Relative Strength Index)
    fn calculate_rsi(prices: &[f64], period: usize) -> f64 {
        if prices.len() < 2 {
            return 50.0;
        }

        let mut gains = 0.0;
        let mut losses = 0.0;
        let mut count = 0;

        for window in prices.windows(2) {
            let change = window[1] - window[0];
            if change > 0.0 {
                gains += change;
            } else {
                losses += change.abs();
            }
            count += 1;
        }

        if count == 0 {
            return 50.0;
        }

        let avg_gain = gains / count as f64;
        let avg_loss = losses / count as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    /// Является ли моментум бычьим
    pub fn is_bullish(&self) -> bool {
        self.roc > 0.0 && self.deviation_from_sma > 0.0
    }

    /// Является ли моментум медвежьим
    pub fn is_bearish(&self) -> bool {
        self.roc < 0.0 && self.deviation_from_sma < 0.0
    }

    /// Сила моментума (от 0 до 1)
    pub fn strength(&self) -> f64 {
        let roc_strength = self.roc.abs().min(0.2) / 0.2;
        let rsi_strength = ((self.rsi - 50.0).abs() / 50.0).min(1.0);

        (roc_strength + rsi_strength) / 2.0
    }
}

/// Расчёт моментума для временного ряда
pub fn calculate_price_momentum(prices: &[f64], period: usize) -> Vec<MomentumResult> {
    if prices.len() < period + 1 {
        return vec![];
    }

    let mut results = Vec::with_capacity(prices.len() - period);

    // Рассчитываем ROC для всех точек
    let rocs: Vec<f64> = (period..prices.len())
        .map(|i| {
            let past = prices[i - period];
            if past != 0.0 {
                (prices[i] - past) / past
            } else {
                0.0
            }
        })
        .collect();

    // Рассчитываем z-scores
    let z_scores = calculate_z_scores(&rocs, period.min(20));

    for (i, (&roc, &z_score)) in rocs.iter().zip(z_scores.iter()).enumerate() {
        results.push(MomentumResult::new(roc, period).with_z_score(z_score));
    }

    results
}

/// Расчёт кумулятивной доходности
pub fn cumulative_return(prices: &[f64], period: usize) -> Option<f64> {
    if prices.len() < period + 1 {
        return None;
    }

    let start = prices[prices.len() - period - 1];
    let end = *prices.last()?;

    if start != 0.0 {
        Some((end - start) / start)
    } else {
        None
    }
}

/// Расчёт волатильности
pub fn volatility(prices: &[f64], period: usize) -> Option<f64> {
    if prices.len() < period + 1 {
        return None;
    }

    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| {
            if w[0] != 0.0 {
                (w[1] - w[0]) / w[0]
            } else {
                0.0
            }
        })
        .collect();

    if returns.len() < period {
        return None;
    }

    let recent_returns = &returns[returns.len() - period..];
    let mean = recent_returns.iter().sum::<f64>() / period as f64;
    let variance = recent_returns
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        / period as f64;

    Some(variance.sqrt())
}

/// Расчёт экспоненциального скользящего среднего
pub fn ema(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.is_empty() {
        return vec![];
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema_values = Vec::with_capacity(prices.len());

    // Первое значение = SMA первых N точек
    if prices.len() >= period {
        let initial_sma: f64 = prices[..period].iter().sum::<f64>() / period as f64;
        ema_values.push(initial_sma);

        for &price in &prices[period..] {
            let prev_ema = *ema_values.last().unwrap();
            let new_ema = (price - prev_ema) * multiplier + prev_ema;
            ema_values.push(new_ema);
        }
    } else {
        // Недостаточно данных для SMA
        ema_values.push(prices[0]);
        for &price in &prices[1..] {
            let prev_ema = *ema_values.last().unwrap();
            let new_ema = (price - prev_ema) * multiplier + prev_ema;
            ema_values.push(new_ema);
        }
    }

    ema_values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_momentum() {
        let prices = vec![100.0, 102.0, 105.0, 103.0, 108.0, 110.0];
        let momentum = PriceMomentum::calculate(&prices, 3);

        assert!(momentum.is_some());
        let m = momentum.unwrap();
        assert!(m.roc > 0.0); // Цена выросла
    }

    #[test]
    fn test_calculate_price_momentum() {
        let prices = vec![100.0, 102.0, 104.0, 103.0, 105.0, 108.0, 107.0, 110.0];
        let results = calculate_price_momentum(&prices, 3);

        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.period == 3));
    }

    #[test]
    fn test_cumulative_return() {
        let prices = vec![100.0, 110.0, 120.0, 130.0, 140.0];
        let ret = cumulative_return(&prices, 4);

        assert!(ret.is_some());
        assert!((ret.unwrap() - 0.4).abs() < 1e-10); // 40% рост
    }

    #[test]
    fn test_volatility() {
        let prices = vec![100.0, 102.0, 98.0, 104.0, 99.0, 105.0];
        let vol = volatility(&prices, 4);

        assert!(vol.is_some());
        assert!(vol.unwrap() > 0.0);
    }

    #[test]
    fn test_ema() {
        let prices = vec![100.0, 102.0, 104.0, 103.0, 105.0, 108.0];
        let ema_values = ema(&prices, 3);

        assert_eq!(ema_values.len(), prices.len() - 2); // period - 1 меньше
        assert!(ema_values.last().unwrap() > &100.0); // EMA растёт с ценой
    }

    #[test]
    fn test_rsi_bounds() {
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0]; // Только рост
        let m = PriceMomentum::calculate(&prices, 3).unwrap();
        assert!(m.rsi >= 0.0 && m.rsi <= 100.0);

        let prices = vec![100.0, 99.0, 98.0, 97.0, 96.0, 95.0]; // Только падение
        let m = PriceMomentum::calculate(&prices, 3).unwrap();
        assert!(m.rsi >= 0.0 && m.rsi <= 100.0);
    }
}
