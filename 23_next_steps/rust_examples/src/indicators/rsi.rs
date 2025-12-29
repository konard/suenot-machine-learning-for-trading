//! Relative Strength Index (RSI) - Индекс относительной силы

use super::Indicator;

/// Relative Strength Index (RSI)
///
/// RSI измеряет скорость и изменение ценовых движений.
/// Значения от 0 до 100:
/// - RSI > 70 = перекупленность (overbought)
/// - RSI < 30 = перепроданность (oversold)
///
/// Формула:
/// RS = Average Gain / Average Loss
/// RSI = 100 - (100 / (1 + RS))
#[derive(Debug, Clone)]
pub struct RSI {
    period: usize,
}

impl RSI {
    /// Создать RSI с указанным периодом (стандартно 14)
    pub fn new(period: usize) -> Self {
        assert!(period > 0, "Period must be greater than 0");
        Self { period }
    }

    /// RSI с периодом 14 (стандарт)
    pub fn default_period() -> Self {
        Self::new(14)
    }

    /// Получить период
    pub fn period(&self) -> usize {
        self.period
    }

    /// Рассчитать текущее значение RSI
    pub fn current(&self, prices: &[f64]) -> Option<f64> {
        let rsi_values = self.calculate(prices);
        rsi_values.last().copied()
    }

    /// Проверить перекупленность
    pub fn is_overbought(rsi: f64, threshold: f64) -> bool {
        rsi > threshold
    }

    /// Проверить перепроданность
    pub fn is_oversold(rsi: f64, threshold: f64) -> bool {
        rsi < threshold
    }
}

impl Indicator for RSI {
    type Output = Vec<f64>;

    fn calculate(&self, prices: &[f64]) -> Self::Output {
        if prices.len() < self.period + 1 {
            return vec![];
        }

        let mut result = Vec::with_capacity(prices.len() - self.period);

        // Рассчитываем изменения цен
        let changes: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();

        // Первый RSI: используем простое среднее
        let first_gains: f64 = changes[..self.period]
            .iter()
            .filter(|&&c| c > 0.0)
            .sum();
        let first_losses: f64 = changes[..self.period]
            .iter()
            .filter(|&&c| c < 0.0)
            .map(|c| c.abs())
            .sum();

        let mut avg_gain = first_gains / self.period as f64;
        let mut avg_loss = first_losses / self.period as f64;

        // Первое значение RSI
        let rsi = if avg_loss == 0.0 {
            100.0
        } else {
            let rs = avg_gain / avg_loss;
            100.0 - (100.0 / (1.0 + rs))
        };
        result.push(rsi);

        // Последующие значения RSI (с использованием сглаженного среднего)
        for change in changes.iter().skip(self.period) {
            let (gain, loss) = if *change > 0.0 {
                (*change, 0.0)
            } else {
                (0.0, change.abs())
            };

            // Сглаженное среднее (Wilder's smoothing)
            avg_gain = (avg_gain * (self.period - 1) as f64 + gain) / self.period as f64;
            avg_loss = (avg_loss * (self.period - 1) as f64 + loss) / self.period as f64;

            let rsi = if avg_loss == 0.0 {
                100.0
            } else {
                let rs = avg_gain / avg_loss;
                100.0 - (100.0 / (1.0 + rs))
            };
            result.push(rsi);
        }

        result
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }
}

/// Stochastic RSI
#[derive(Debug, Clone)]
pub struct StochasticRSI {
    rsi_period: usize,
    stoch_period: usize,
}

impl StochasticRSI {
    pub fn new(rsi_period: usize, stoch_period: usize) -> Self {
        Self {
            rsi_period,
            stoch_period,
        }
    }
}

impl Indicator for StochasticRSI {
    type Output = Vec<f64>;

    fn calculate(&self, prices: &[f64]) -> Self::Output {
        let rsi = RSI::new(self.rsi_period);
        let rsi_values = rsi.calculate(prices);

        if rsi_values.len() < self.stoch_period {
            return vec![];
        }

        rsi_values
            .windows(self.stoch_period)
            .map(|window| {
                let min = window.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let current = window.last().unwrap();

                if (max - min).abs() < 1e-10 {
                    50.0
                } else {
                    (current - min) / (max - min) * 100.0
                }
            })
            .collect()
    }

    fn min_periods(&self) -> usize {
        self.rsi_period + self.stoch_period
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi_calculation() {
        let rsi = RSI::new(14);

        // Тестовые данные с явным трендом вверх
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let result = rsi.calculate(&prices);

        assert!(!result.is_empty());
        // При постоянном росте RSI должен быть близок к 100
        assert!(result.last().unwrap() > &90.0);
    }

    #[test]
    fn test_rsi_range() {
        let rsi = RSI::new(14);
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64).sin() * 10.0).collect();
        let result = rsi.calculate(&prices);

        for value in result {
            assert!(value >= 0.0 && value <= 100.0);
        }
    }

    #[test]
    fn test_overbought_oversold() {
        assert!(RSI::is_overbought(75.0, 70.0));
        assert!(!RSI::is_overbought(65.0, 70.0));
        assert!(RSI::is_oversold(25.0, 30.0));
        assert!(!RSI::is_oversold(35.0, 30.0));
    }
}
