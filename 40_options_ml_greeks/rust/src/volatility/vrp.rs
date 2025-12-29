//! # Анализ премии за волатильностный риск (VRP)
//!
//! VRP = IV - RV
//!
//! Положительный VRP означает, что опционы переоценены относительно
//! фактической волатильности, что создаёт возможность для продажи опционов.

use serde::{Deserialize, Serialize};

/// Статистика VRP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrpStatistics {
    /// Среднее значение VRP
    pub mean: f64,
    /// Стандартное отклонение
    pub std: f64,
    /// Текущий z-score
    pub current_zscore: f64,
    /// Доля времени, когда VRP > 0
    pub pct_positive: f64,
    /// Средний VRP когда положительный
    pub avg_when_positive: f64,
    /// Средний VRP когда отрицательный
    pub avg_when_negative: f64,
}

/// Торговый сигнал на основе VRP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrpSignal {
    /// Действие
    pub action: VrpAction,
    /// Размер edge (разница IV - predicted RV)
    pub edge: f64,
    /// Уверенность в сигнале (0-1)
    pub confidence: f64,
    /// Причина сигнала
    pub reason: String,
}

/// Тип действия
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VrpAction {
    /// Продать волатильность (IV переоценена)
    SellVolatility,
    /// Купить волатильность (IV недооценена)
    BuyVolatility,
    /// Нет сигнала
    NoTrade,
}

impl std::fmt::Display for VrpAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VrpAction::SellVolatility => write!(f, "SELL VOL"),
            VrpAction::BuyVolatility => write!(f, "BUY VOL"),
            VrpAction::NoTrade => write!(f, "NO TRADE"),
        }
    }
}

/// Анализатор премии за волатильностный риск
#[derive(Debug, Clone)]
pub struct VolatilityRiskPremium {
    /// Горизонт для сравнения (дни)
    lookback: usize,
    /// Минимальный порог для входа
    min_threshold: f64,
    /// Максимальный порог (для определения экстремальных ситуаций)
    max_threshold: f64,
    /// История VRP
    history: Vec<f64>,
}

impl VolatilityRiskPremium {
    /// Создать анализатор VRP
    pub fn new(lookback: usize, min_threshold: f64) -> Self {
        Self {
            lookback,
            min_threshold,
            max_threshold: min_threshold * 3.0,
            history: Vec::new(),
        }
    }

    /// Создать со стандартными параметрами
    pub fn default_crypto() -> Self {
        Self::new(7, 0.02) // 7 дней, 2% минимальный порог
    }

    /// Расчёт VRP
    ///
    /// # Arguments
    /// * `iv` - Текущая подразумеваемая волатильность
    /// * `rv` - Реализованная волатильность за прошедший период
    ///
    /// # Returns
    /// VRP = IV - RV
    pub fn calculate(&self, iv: f64, rv: f64) -> f64 {
        iv - rv
    }

    /// Добавить наблюдение в историю
    pub fn add_observation(&mut self, vrp: f64) {
        self.history.push(vrp);

        // Ограничиваем размер истории
        if self.history.len() > 252 {
            self.history.remove(0);
        }
    }

    /// Рассчитать статистику VRP
    pub fn statistics(&self) -> Option<VrpStatistics> {
        if self.history.len() < 20 {
            return None;
        }

        let n = self.history.len() as f64;

        // Среднее
        let mean = self.history.iter().sum::<f64>() / n;

        // Стандартное отклонение
        let variance = self
            .history
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / n;
        let std = variance.sqrt();

        // Z-score текущего значения
        let current = self.history.last().copied().unwrap_or(0.0);
        let current_zscore = if std > 0.0 {
            (current - mean) / std
        } else {
            0.0
        };

        // Доля положительных
        let positive: Vec<_> = self.history.iter().filter(|&&x| x > 0.0).collect();
        let pct_positive = positive.len() as f64 / n;

        // Среднее когда положительный/отрицательный
        let avg_when_positive = if positive.is_empty() {
            0.0
        } else {
            positive.iter().copied().sum::<f64>() / positive.len() as f64
        };

        let negative: Vec<_> = self.history.iter().filter(|&&x| x < 0.0).collect();
        let avg_when_negative = if negative.is_empty() {
            0.0
        } else {
            negative.iter().copied().sum::<f64>() / negative.len() as f64
        };

        Some(VrpStatistics {
            mean,
            std,
            current_zscore,
            pct_positive,
            avg_when_positive,
            avg_when_negative,
        })
    }

    /// Генерация торгового сигнала
    ///
    /// # Arguments
    /// * `current_iv` - Текущая IV
    /// * `predicted_rv` - Предсказанная RV
    /// * `stats` - Историческая статистика (опционально)
    pub fn trading_signal(
        &self,
        current_iv: f64,
        predicted_rv: f64,
        stats: Option<&VrpStatistics>,
    ) -> VrpSignal {
        let spread = current_iv - predicted_rv;
        let abs_spread = spread.abs();

        // Базовая логика
        if abs_spread < self.min_threshold {
            return VrpSignal {
                action: VrpAction::NoTrade,
                edge: 0.0,
                confidence: 0.0,
                reason: format!(
                    "Spread {:.2}% ниже порога {:.2}%",
                    spread * 100.0,
                    self.min_threshold * 100.0
                ),
            };
        }

        // Определяем уверенность на основе статистики
        let confidence = if let Some(s) = stats {
            // Чем более экстремальный z-score, тем больше уверенность
            let zscore_factor = (s.current_zscore.abs() / 2.0).min(1.0);

            // Учитываем историческую тенденцию
            let historical_factor = if spread > 0.0 {
                s.pct_positive // Если VRP обычно положительный, уверенность выше
            } else {
                1.0 - s.pct_positive
            };

            (zscore_factor * 0.6 + historical_factor * 0.4).min(1.0)
        } else {
            // Без статистики используем только размер спреда
            (abs_spread / self.max_threshold).min(1.0)
        };

        if spread > self.min_threshold {
            VrpSignal {
                action: VrpAction::SellVolatility,
                edge: spread,
                confidence,
                reason: format!(
                    "IV ({:.1}%) переоценена vs predicted RV ({:.1}%), edge = {:.2}%",
                    current_iv * 100.0,
                    predicted_rv * 100.0,
                    spread * 100.0
                ),
            }
        } else if spread < -self.min_threshold {
            VrpSignal {
                action: VrpAction::BuyVolatility,
                edge: -spread,
                confidence,
                reason: format!(
                    "IV ({:.1}%) недооценена vs predicted RV ({:.1}%), edge = {:.2}%",
                    current_iv * 100.0,
                    predicted_rv * 100.0,
                    -spread * 100.0
                ),
            }
        } else {
            VrpSignal {
                action: VrpAction::NoTrade,
                edge: 0.0,
                confidence: 0.0,
                reason: "Spread в пределах порога".to_string(),
            }
        }
    }

    /// Исторический VRP
    pub fn history(&self) -> &[f64] {
        &self.history
    }

    /// Текущий VRP
    pub fn current(&self) -> Option<f64> {
        self.history.last().copied()
    }

    /// Скользящее среднее VRP
    pub fn moving_average(&self, window: usize) -> Option<f64> {
        if self.history.len() < window {
            return None;
        }

        let sum: f64 = self.history[self.history.len() - window..].iter().sum();
        Some(sum / window as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vrp_calculation() {
        let vrp = VolatilityRiskPremium::default_crypto();

        // IV = 55%, RV = 45% → VRP = 10%
        let result = vrp.calculate(0.55, 0.45);
        assert!((result - 0.10).abs() < 0.001);
    }

    #[test]
    fn test_trading_signal_sell() {
        let vrp = VolatilityRiskPremium::default_crypto();

        // IV сильно выше predicted RV → продаём волатильность
        let signal = vrp.trading_signal(0.60, 0.45, None);

        assert_eq!(signal.action, VrpAction::SellVolatility);
        assert!(signal.edge > 0.10);
    }

    #[test]
    fn test_trading_signal_buy() {
        let vrp = VolatilityRiskPremium::default_crypto();

        // IV сильно ниже predicted RV → покупаем волатильность
        let signal = vrp.trading_signal(0.30, 0.50, None);

        assert_eq!(signal.action, VrpAction::BuyVolatility);
        assert!(signal.edge > 0.15);
    }

    #[test]
    fn test_trading_signal_no_trade() {
        let vrp = VolatilityRiskPremium::default_crypto();

        // Маленькая разница → нет сигнала
        let signal = vrp.trading_signal(0.30, 0.31, None);

        assert_eq!(signal.action, VrpAction::NoTrade);
    }

    #[test]
    fn test_statistics() {
        let mut vrp = VolatilityRiskPremium::default_crypto();

        // Добавляем историю (в основном положительную)
        for i in 0..50 {
            let value = if i % 5 == 0 { -0.02 } else { 0.05 };
            vrp.add_observation(value);
        }

        let stats = vrp.statistics().unwrap();

        assert!(stats.mean > 0.0); // В среднем положительный VRP
        assert!(stats.pct_positive > 0.7); // Чаще положительный
    }
}
