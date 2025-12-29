//! # Расчёт реализованной волатильности
//!
//! Реализованная волатильность (RV) — это фактическая волатильность,
//! наблюдаемая в ценах за определённый период.

use super::VolatilityConfig;

/// Расчёт реализованной волатильности
#[derive(Debug, Clone)]
pub struct RealizedVolatility {
    config: VolatilityConfig,
}

impl RealizedVolatility {
    /// Создать калькулятор RV
    pub fn new(config: VolatilityConfig) -> Self {
        Self { config }
    }

    /// Создать калькулятор для крипты
    pub fn crypto() -> Self {
        Self::new(VolatilityConfig::crypto())
    }

    /// Расчёт логарифмических доходностей
    pub fn log_returns(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return vec![];
        }

        prices
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect()
    }

    /// Расчёт простых доходностей
    pub fn simple_returns(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return vec![];
        }

        prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Расчёт RV за период (годовая)
    ///
    /// # Arguments
    ///
    /// * `prices` - Вектор цен закрытия
    /// * `window` - Размер окна (None = все данные)
    ///
    /// # Returns
    ///
    /// Годовая реализованная волатильность
    pub fn calculate(&self, prices: &[f64], window: Option<usize>) -> Option<f64> {
        let returns = if self.config.use_log_returns {
            Self::log_returns(prices)
        } else {
            Self::simple_returns(prices)
        };

        if returns.is_empty() {
            return None;
        }

        let data = match window {
            Some(w) if w < returns.len() => &returns[returns.len() - w..],
            _ => &returns,
        };

        if data.is_empty() {
            return None;
        }

        // Среднее
        let mean = data.iter().sum::<f64>() / data.len() as f64;

        // Дисперсия
        let variance = data.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / data.len() as f64;

        // Годовая волатильность
        let daily_vol = variance.sqrt();
        let annual_vol = daily_vol * self.config.days_per_year.sqrt();

        Some(annual_vol)
    }

    /// Расчёт RV на скользящем окне
    ///
    /// # Returns
    ///
    /// Вектор RV для каждой точки (None для первых window-1 точек)
    pub fn rolling(&self, prices: &[f64], window: usize) -> Vec<Option<f64>> {
        if prices.len() < window + 1 {
            return vec![None; prices.len()];
        }

        let mut result = vec![None; window];

        for i in window..prices.len() {
            let slice = &prices[i - window..=i];
            result.push(self.calculate(slice, None));
        }

        result
    }

    /// Расчёт нескольких окон RV
    pub fn multi_window(&self, prices: &[f64], windows: &[usize]) -> Vec<Option<f64>> {
        windows
            .iter()
            .map(|&w| self.calculate(prices, Some(w)))
            .collect()
    }

    /// Расчёт волатильности волатильности (vol of vol)
    pub fn vol_of_vol(&self, prices: &[f64], rv_window: usize, vol_window: usize) -> Option<f64> {
        let rolling_rv = self.rolling(prices, rv_window);

        let valid_rv: Vec<f64> = rolling_rv.into_iter().flatten().collect();

        if valid_rv.len() < vol_window {
            return None;
        }

        let recent = &valid_rv[valid_rv.len() - vol_window..];
        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance = recent.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / recent.len() as f64;

        Some(variance.sqrt())
    }

    /// Расчёт Parkinson volatility (High-Low)
    /// Более эффективная оценка волатильности
    pub fn parkinson(highs: &[f64], lows: &[f64]) -> Option<f64> {
        if highs.len() != lows.len() || highs.is_empty() {
            return None;
        }

        let n = highs.len() as f64;
        let sum: f64 = highs
            .iter()
            .zip(lows.iter())
            .map(|(h, l)| (h / l).ln().powi(2))
            .sum();

        let factor = 1.0 / (4.0 * n * 2.0_f64.ln());
        let daily_var = factor * sum;
        let annual_vol = daily_var.sqrt() * 365.0_f64.sqrt();

        Some(annual_vol)
    }

    /// Расчёт Garman-Klass volatility (OHLC)
    /// Использует Open, High, Low, Close для лучшей оценки
    pub fn garman_klass(
        opens: &[f64],
        highs: &[f64],
        lows: &[f64],
        closes: &[f64],
    ) -> Option<f64> {
        if opens.len() != highs.len()
            || highs.len() != lows.len()
            || lows.len() != closes.len()
            || opens.is_empty()
        {
            return None;
        }

        let n = opens.len() as f64;
        let sum: f64 = opens
            .iter()
            .zip(highs.iter())
            .zip(lows.iter())
            .zip(closes.iter())
            .map(|(((o, h), l), c)| {
                let hl = (h / l).ln();
                let co = (c / o).ln();
                0.5 * hl.powi(2) - (2.0 * 2.0_f64.ln() - 1.0) * co.powi(2)
            })
            .sum();

        let daily_var = sum / n;
        let annual_vol = daily_var.sqrt() * 365.0_f64.sqrt();

        Some(annual_vol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 101.0, 102.0, 101.0];
        let returns = RealizedVolatility::log_returns(&prices);

        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.00995).abs() < 0.0001); // ln(101/100)
    }

    #[test]
    fn test_realized_volatility() {
        // Создаём данные с известной волатильностью
        let rv = RealizedVolatility::crypto();

        // 20 дней цен с небольшими изменениями
        let prices: Vec<f64> = (0..21).map(|i| 100.0 + (i as f64 * 0.5)).collect();

        let vol = rv.calculate(&prices, None).unwrap();

        // Волатильность должна быть низкой для линейного роста
        assert!(vol > 0.0 && vol < 0.5, "Volatility: {}", vol);
    }

    #[test]
    fn test_rolling_volatility() {
        let rv = RealizedVolatility::crypto();
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64).sin() * 5.0).collect();

        let rolling = rv.rolling(&prices, 10);

        assert_eq!(rolling.len(), prices.len());
        assert!(rolling[9].is_none()); // Первые 10 = None
        assert!(rolling[10].is_some()); // С 11-го есть значения
    }

    #[test]
    fn test_parkinson_volatility() {
        let highs = vec![102.0, 103.0, 104.0, 103.0, 105.0];
        let lows = vec![98.0, 99.0, 100.0, 99.0, 101.0];

        let vol = RealizedVolatility::parkinson(&highs, &lows).unwrap();
        assert!(vol > 0.0, "Parkinson vol should be positive");
    }
}
