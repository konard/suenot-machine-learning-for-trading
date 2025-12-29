//! Exponential Moving Average (EMA) - Экспоненциальное скользящее среднее

use super::Indicator;

/// Exponential Moving Average (EMA)
///
/// Экспоненциальное скользящее среднее придаёт больший вес последним значениям.
///
/// Формула: EMA = Price * k + EMA(prev) * (1 - k)
/// где k = 2 / (period + 1)
#[derive(Debug, Clone)]
pub struct EMA {
    period: usize,
    multiplier: f64,
}

impl EMA {
    /// Создать EMA с указанным периодом
    pub fn new(period: usize) -> Self {
        assert!(period > 0, "Period must be greater than 0");
        let multiplier = 2.0 / (period as f64 + 1.0);
        Self { period, multiplier }
    }

    /// Создать EMA с кастомным множителем
    pub fn with_multiplier(period: usize, multiplier: f64) -> Self {
        assert!(period > 0, "Period must be greater than 0");
        assert!(
            (0.0..=1.0).contains(&multiplier),
            "Multiplier must be between 0 and 1"
        );
        Self { period, multiplier }
    }

    /// Получить период
    pub fn period(&self) -> usize {
        self.period
    }

    /// Получить множитель
    pub fn multiplier(&self) -> f64 {
        self.multiplier
    }

    /// Обновить EMA при добавлении нового значения
    pub fn update(&self, prev_ema: f64, new_price: f64) -> f64 {
        new_price * self.multiplier + prev_ema * (1.0 - self.multiplier)
    }

    /// Рассчитать текущее значение EMA
    pub fn current(&self, prices: &[f64]) -> Option<f64> {
        let ema_values = self.calculate(prices);
        ema_values.last().copied()
    }
}

impl Indicator for EMA {
    type Output = Vec<f64>;

    fn calculate(&self, prices: &[f64]) -> Self::Output {
        if prices.len() < self.period {
            return vec![];
        }

        let mut result = Vec::with_capacity(prices.len() - self.period + 1);

        // Первое значение EMA = SMA за период
        let first_sma: f64 = prices[..self.period].iter().sum::<f64>() / self.period as f64;
        result.push(first_sma);

        // Последующие значения EMA
        let mut prev_ema = first_sma;
        for price in prices.iter().skip(self.period) {
            let ema = self.update(prev_ema, *price);
            result.push(ema);
            prev_ema = ema;
        }

        result
    }

    fn min_periods(&self) -> usize {
        self.period
    }
}

/// Рассчитать EMA для двух периодов одновременно
pub fn dual_ema(prices: &[f64], fast_period: usize, slow_period: usize) -> (Vec<f64>, Vec<f64>) {
    let fast_ema = EMA::new(fast_period);
    let slow_ema = EMA::new(slow_period);

    (fast_ema.calculate(prices), slow_ema.calculate(prices))
}

/// Triple EMA (TEMA) - тройное экспоненциальное среднее
#[derive(Debug, Clone)]
pub struct TEMA {
    period: usize,
    ema: EMA,
}

impl TEMA {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            ema: EMA::new(period),
        }
    }
}

impl Indicator for TEMA {
    type Output = Vec<f64>;

    fn calculate(&self, prices: &[f64]) -> Self::Output {
        let ema1 = self.ema.calculate(prices);
        if ema1.is_empty() {
            return vec![];
        }

        let ema2 = self.ema.calculate(&ema1);
        if ema2.is_empty() {
            return vec![];
        }

        let ema3 = self.ema.calculate(&ema2);
        if ema3.is_empty() {
            return vec![];
        }

        // TEMA = 3*EMA1 - 3*EMA2 + EMA3
        // Выравниваем длины
        let offset1 = ema1.len() - ema3.len();
        let offset2 = ema2.len() - ema3.len();

        ema3.iter()
            .enumerate()
            .map(|(i, &e3)| {
                let e1 = ema1[i + offset1];
                let e2 = ema2[i + offset2];
                3.0 * e1 - 3.0 * e2 + e3
            })
            .collect()
    }

    fn min_periods(&self) -> usize {
        self.period * 3 - 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_calculation() {
        let ema = EMA::new(3);
        let prices = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let result = ema.calculate(&prices);

        assert_eq!(result.len(), 3);
        // Первое значение = SMA(10, 11, 12) = 11
        assert!((result[0] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema_update() {
        let ema = EMA::new(3);
        let prev = 11.0;
        let new_price = 14.0;
        // k = 2/(3+1) = 0.5
        // EMA = 14 * 0.5 + 11 * 0.5 = 12.5
        let updated = ema.update(prev, new_price);

        assert!((updated - 12.5).abs() < 1e-10);
    }

    #[test]
    fn test_ema_multiplier() {
        let ema = EMA::new(9);
        // k = 2/(9+1) = 0.2
        assert!((ema.multiplier() - 0.2).abs() < 1e-10);
    }
}
