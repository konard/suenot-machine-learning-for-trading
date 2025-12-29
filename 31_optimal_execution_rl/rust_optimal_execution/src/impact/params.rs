//! Параметры моделей market impact

use serde::{Deserialize, Serialize};

/// Параметры модели market impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactParams {
    /// Среднедневной объём (Average Daily Volume)
    pub adv: f64,
    /// Коэффициент временного воздействия (eta)
    pub eta: f64,
    /// Коэффициент постоянного воздействия (gamma)
    pub gamma: f64,
    /// Волатильность (дневная)
    pub volatility: f64,
    /// Спред bid-ask
    pub spread: f64,
    /// Коэффициент затухания для transient impact
    pub decay_rate: f64,
}

impl ImpactParams {
    /// Создать параметры с заданными значениями
    pub fn new(adv: f64, volatility: f64) -> Self {
        // Эмпирические коэффициенты из исследований
        let eta = 0.01 * volatility; // Временный impact пропорционален волатильности
        let gamma = 0.1 * volatility; // Постоянный impact

        Self {
            adv,
            eta,
            gamma,
            volatility,
            spread: 0.0001, // 1 bps по умолчанию
            decay_rate: 0.5, // Период полураспада ~2 шага
        }
    }

    /// Создать параметры для криптовалютного рынка
    pub fn crypto_default() -> Self {
        Self {
            adv: 100_000_000.0, // $100M дневной объём
            eta: 0.0001,
            gamma: 0.001,
            volatility: 0.02, // 2% дневная волатильность
            spread: 0.0001, // 1 bps
            decay_rate: 0.3,
        }
    }

    /// Создать параметры для высоколиквидного рынка (BTC/ETH)
    pub fn high_liquidity() -> Self {
        Self {
            adv: 1_000_000_000.0, // $1B дневной объём
            eta: 0.00005,
            gamma: 0.0005,
            volatility: 0.015,
            spread: 0.00005, // 0.5 bps
            decay_rate: 0.2,
        }
    }

    /// Создать параметры для низколиквидного рынка (альткоины)
    pub fn low_liquidity() -> Self {
        Self {
            adv: 10_000_000.0, // $10M дневной объём
            eta: 0.001,
            gamma: 0.01,
            volatility: 0.05, // 5% дневная волатильность
            spread: 0.001, // 10 bps
            decay_rate: 0.5,
        }
    }

    /// Оценить параметры из исторических данных
    pub fn estimate_from_data(
        volumes: &[f64],
        prices: &[f64],
        _trades: &[(f64, f64)], // (size, price_change)
    ) -> Self {
        // Средний дневной объём
        let adv = if volumes.is_empty() {
            100_000_000.0
        } else {
            volumes.iter().sum::<f64>() / volumes.len() as f64
        };

        // Волатильность (стандартное отклонение доходностей)
        let volatility = if prices.len() < 2 {
            0.02
        } else {
            let returns: Vec<f64> = prices.windows(2)
                .map(|w| (w[1] / w[0]).ln())
                .collect();
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / returns.len() as f64;
            variance.sqrt()
        };

        Self::new(adv, volatility)
    }

    /// Получить нормализованный размер сделки
    pub fn normalized_size(&self, quantity: f64) -> f64 {
        quantity / self.adv
    }
}

impl Default for ImpactParams {
    fn default() -> Self {
        Self::crypto_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_creation() {
        let params = ImpactParams::new(1_000_000.0, 0.02);
        assert_eq!(params.adv, 1_000_000.0);
        assert!((params.volatility - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_normalized_size() {
        let params = ImpactParams::new(1_000_000.0, 0.02);
        assert!((params.normalized_size(10_000.0) - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_liquidity_presets() {
        let high = ImpactParams::high_liquidity();
        let low = ImpactParams::low_liquidity();

        assert!(high.adv > low.adv);
        assert!(high.eta < low.eta); // Меньший impact для более ликвидных рынков
    }
}
