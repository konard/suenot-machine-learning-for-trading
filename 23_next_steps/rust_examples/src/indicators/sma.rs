//! Simple Moving Average (SMA) - Простое скользящее среднее

use super::Indicator;

/// Simple Moving Average (SMA)
///
/// Простое скользящее среднее - среднее арифметическое последних N цен.
///
/// # Пример
/// ```
/// use ml4t_bybit::indicators::SMA;
/// use ml4t_bybit::indicators::Indicator;
///
/// let sma = SMA::new(3);
/// let prices = vec![10.0, 11.0, 12.0, 13.0, 14.0];
/// let result = sma.calculate(&prices);
/// // result = [11.0, 12.0, 13.0]  (средние за последние 3 значения)
/// ```
#[derive(Debug, Clone)]
pub struct SMA {
    period: usize,
}

impl SMA {
    /// Создать SMA с указанным периодом
    pub fn new(period: usize) -> Self {
        assert!(period > 0, "Period must be greater than 0");
        Self { period }
    }

    /// Получить период
    pub fn period(&self) -> usize {
        self.period
    }

    /// Рассчитать текущее значение SMA для последних N значений
    pub fn current(&self, prices: &[f64]) -> Option<f64> {
        if prices.len() < self.period {
            return None;
        }
        let slice = &prices[prices.len() - self.period..];
        Some(slice.iter().sum::<f64>() / self.period as f64)
    }

    /// Обновить SMA при добавлении нового значения (оптимизированно)
    pub fn update(&self, old_sma: f64, old_value: f64, new_value: f64) -> f64 {
        old_sma + (new_value - old_value) / self.period as f64
    }
}

impl Indicator for SMA {
    type Output = Vec<f64>;

    fn calculate(&self, prices: &[f64]) -> Self::Output {
        if prices.len() < self.period {
            return vec![];
        }

        let mut result = Vec::with_capacity(prices.len() - self.period + 1);

        // Первое значение SMA
        let first_sum: f64 = prices[..self.period].iter().sum();
        let mut current_sum = first_sum;
        result.push(current_sum / self.period as f64);

        // Последующие значения (скользящая сумма)
        for i in self.period..prices.len() {
            current_sum = current_sum - prices[i - self.period] + prices[i];
            result.push(current_sum / self.period as f64);
        }

        result
    }

    fn min_periods(&self) -> usize {
        self.period
    }
}

/// Рассчитать SMA для двух периодов одновременно (для crossover стратегий)
pub fn dual_sma(prices: &[f64], fast_period: usize, slow_period: usize) -> (Vec<f64>, Vec<f64>) {
    let fast_sma = SMA::new(fast_period);
    let slow_sma = SMA::new(slow_period);

    (fast_sma.calculate(prices), slow_sma.calculate(prices))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma_calculation() {
        let sma = SMA::new(3);
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma.calculate(&prices);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
        assert!((result[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_sma_current() {
        let sma = SMA::new(3);
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let current = sma.current(&prices).unwrap();

        assert!((current - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_sma_update() {
        let sma = SMA::new(3);
        let old_sma = 3.0; // (2+3+4)/3
        let updated = sma.update(old_sma, 2.0, 5.0); // (3+4+5)/3 = 4

        assert!((updated - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_insufficient_data() {
        let sma = SMA::new(5);
        let prices = vec![1.0, 2.0, 3.0];
        let result = sma.calculate(&prices);

        assert!(result.is_empty());
    }
}
