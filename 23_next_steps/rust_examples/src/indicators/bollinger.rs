//! Bollinger Bands (Полосы Боллинджера)

use super::{sma::SMA, utils, Indicator};

/// Результат расчёта Bollinger Bands
#[derive(Debug, Clone)]
pub struct BollingerResult {
    /// Верхняя полоса
    pub upper: Vec<f64>,
    /// Средняя линия (SMA)
    pub middle: Vec<f64>,
    /// Нижняя полоса
    pub lower: Vec<f64>,
    /// Ширина полосы (bandwidth)
    pub bandwidth: Vec<f64>,
    /// Процентное положение цены в полосе (%B)
    pub percent_b: Vec<f64>,
}

/// Bollinger Bands
///
/// Индикатор волатильности, состоящий из:
/// - Средней линии (SMA)
/// - Верхней полосы (SMA + k * StdDev)
/// - Нижней полосы (SMA - k * StdDev)
///
/// Стандартные параметры: период 20, множитель 2
#[derive(Debug, Clone)]
pub struct BollingerBands {
    period: usize,
    num_std: f64,
    sma: SMA,
}

impl BollingerBands {
    /// Создать Bollinger Bands с указанными параметрами
    pub fn new(period: usize, num_std: f64) -> Self {
        Self {
            period,
            num_std,
            sma: SMA::new(period),
        }
    }

    /// Bollinger Bands со стандартными параметрами (20, 2.0)
    pub fn standard() -> Self {
        Self::new(20, 2.0)
    }

    /// Получить текущие значения полос
    pub fn current(&self, prices: &[f64]) -> Option<(f64, f64, f64)> {
        let result = self.calculate(prices);
        if result.upper.is_empty() {
            return None;
        }

        let upper = *result.upper.last()?;
        let middle = *result.middle.last()?;
        let lower = *result.lower.last()?;

        Some((upper, middle, lower))
    }

    /// Рассчитать %B (положение цены относительно полос)
    /// 0 = на нижней полосе, 1 = на верхней полосе
    pub fn percent_b(price: f64, upper: f64, lower: f64) -> f64 {
        let width = upper - lower;
        if width.abs() < 1e-10 {
            0.5
        } else {
            (price - lower) / width
        }
    }

    /// Рассчитать ширину полосы (bandwidth)
    pub fn bandwidth(upper: f64, middle: f64, lower: f64) -> f64 {
        if middle.abs() < 1e-10 {
            0.0
        } else {
            (upper - lower) / middle
        }
    }

    /// Проверить, пробита ли верхняя полоса
    pub fn is_upper_breakout(price: f64, upper: f64) -> bool {
        price > upper
    }

    /// Проверить, пробита ли нижняя полоса
    pub fn is_lower_breakout(price: f64, lower: f64) -> bool {
        price < lower
    }
}

impl Indicator for BollingerBands {
    type Output = BollingerResult;

    fn calculate(&self, prices: &[f64]) -> Self::Output {
        if prices.len() < self.period {
            return BollingerResult {
                upper: vec![],
                middle: vec![],
                lower: vec![],
                bandwidth: vec![],
                percent_b: vec![],
            };
        }

        let middle = self.sma.calculate(prices);
        let len = middle.len();

        let mut upper = Vec::with_capacity(len);
        let mut lower = Vec::with_capacity(len);
        let mut bandwidth = Vec::with_capacity(len);
        let mut percent_b = Vec::with_capacity(len);

        for i in 0..len {
            let window_start = i;
            let window_end = i + self.period;
            let window = &prices[window_start..window_end];

            let std_dev = utils::std_dev(window);
            let mid = middle[i];

            let up = mid + self.num_std * std_dev;
            let low = mid - self.num_std * std_dev;

            upper.push(up);
            lower.push(low);
            bandwidth.push(Self::bandwidth(up, mid, low));

            // %B для последней цены в окне
            let current_price = prices[window_end - 1];
            percent_b.push(Self::percent_b(current_price, up, low));
        }

        BollingerResult {
            upper,
            middle,
            lower,
            bandwidth,
            percent_b,
        }
    }

    fn min_periods(&self) -> usize {
        self.period
    }
}

/// Keltner Channels (альтернатива Bollinger Bands)
#[derive(Debug, Clone)]
pub struct KeltnerChannels {
    ema_period: usize,
    atr_period: usize,
    multiplier: f64,
}

impl KeltnerChannels {
    pub fn new(ema_period: usize, atr_period: usize, multiplier: f64) -> Self {
        Self {
            ema_period,
            atr_period,
            multiplier,
        }
    }

    pub fn standard() -> Self {
        Self::new(20, 10, 2.0)
    }

    /// Рассчитать ATR (Average True Range)
    fn calculate_atr(&self, highs: &[f64], lows: &[f64], closes: &[f64]) -> Vec<f64> {
        if highs.len() < 2 {
            return vec![];
        }

        let mut true_ranges = Vec::with_capacity(highs.len() - 1);

        // Первый TR
        true_ranges.push(highs[0] - lows[0]);

        // Последующие TR
        for i in 1..highs.len() {
            let tr = (highs[i] - lows[i])
                .max((highs[i] - closes[i - 1]).abs())
                .max((lows[i] - closes[i - 1]).abs());
            true_ranges.push(tr);
        }

        // ATR = EMA от TR
        use super::ema::EMA;
        let ema = EMA::new(self.atr_period);
        ema.calculate(&true_ranges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bollinger_bands() {
        let bb = BollingerBands::standard();
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64 * 0.1)).collect();
        let result = bb.calculate(&prices);

        assert!(!result.upper.is_empty());
        assert!(!result.middle.is_empty());
        assert!(!result.lower.is_empty());

        // Верхняя полоса должна быть выше средней
        for (u, m) in result.upper.iter().zip(result.middle.iter()) {
            assert!(u >= m);
        }

        // Нижняя полоса должна быть ниже средней
        for (l, m) in result.lower.iter().zip(result.middle.iter()) {
            assert!(l <= m);
        }
    }

    #[test]
    fn test_percent_b() {
        let upper = 110.0;
        let lower = 90.0;

        // Цена на средней линии
        let pb = BollingerBands::percent_b(100.0, upper, lower);
        assert!((pb - 0.5).abs() < 1e-10);

        // Цена на верхней полосе
        let pb = BollingerBands::percent_b(110.0, upper, lower);
        assert!((pb - 1.0).abs() < 1e-10);

        // Цена на нижней полосе
        let pb = BollingerBands::percent_b(90.0, upper, lower);
        assert!((pb - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_bandwidth() {
        let upper = 110.0;
        let middle = 100.0;
        let lower = 90.0;

        let bw = BollingerBands::bandwidth(upper, middle, lower);
        assert!((bw - 0.2).abs() < 1e-10); // (110-90)/100 = 0.2
    }

    #[test]
    fn test_breakouts() {
        assert!(BollingerBands::is_upper_breakout(111.0, 110.0));
        assert!(!BollingerBands::is_upper_breakout(109.0, 110.0));
        assert!(BollingerBands::is_lower_breakout(89.0, 90.0));
        assert!(!BollingerBands::is_lower_breakout(91.0, 90.0));
    }
}
