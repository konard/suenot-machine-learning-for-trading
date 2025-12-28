//! Модуль для работы с рыночными данными Bybit

use crate::models::Kline;

/// Таймфреймы для свечей
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interval {
    /// 1 минута
    M1,
    /// 3 минуты
    M3,
    /// 5 минут
    M5,
    /// 15 минут
    M15,
    /// 30 минут
    M30,
    /// 1 час
    H1,
    /// 2 часа
    H2,
    /// 4 часа
    H4,
    /// 6 часов
    H6,
    /// 12 часов
    H12,
    /// 1 день
    D1,
    /// 1 неделя
    W1,
    /// 1 месяц
    Month1,
}

impl Interval {
    /// Преобразовать в строку для API
    pub fn as_str(&self) -> &'static str {
        match self {
            Interval::M1 => "1",
            Interval::M3 => "3",
            Interval::M5 => "5",
            Interval::M15 => "15",
            Interval::M30 => "30",
            Interval::H1 => "60",
            Interval::H2 => "120",
            Interval::H4 => "240",
            Interval::H6 => "360",
            Interval::H12 => "720",
            Interval::D1 => "D",
            Interval::W1 => "W",
            Interval::Month1 => "M",
        }
    }

    /// Получить количество минут в интервале
    pub fn to_minutes(&self) -> u32 {
        match self {
            Interval::M1 => 1,
            Interval::M3 => 3,
            Interval::M5 => 5,
            Interval::M15 => 15,
            Interval::M30 => 30,
            Interval::H1 => 60,
            Interval::H2 => 120,
            Interval::H4 => 240,
            Interval::H6 => 360,
            Interval::H12 => 720,
            Interval::D1 => 1440,
            Interval::W1 => 10080,
            Interval::Month1 => 43200,
        }
    }
}

/// Расчёт технических индикаторов на основе свечей
pub struct TechnicalIndicators;

impl TechnicalIndicators {
    /// Рассчитать простую скользящую среднюю (SMA)
    pub fn sma(klines: &[Kline], period: usize) -> Vec<f64> {
        if klines.len() < period {
            return vec![];
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();

        closes
            .windows(period)
            .map(|window| window.iter().sum::<f64>() / period as f64)
            .collect()
    }

    /// Рассчитать экспоненциальную скользящую среднюю (EMA)
    pub fn ema(klines: &[Kline], period: usize) -> Vec<f64> {
        if klines.is_empty() || period == 0 {
            return vec![];
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let multiplier = 2.0 / (period as f64 + 1.0);

        let mut ema_values = Vec::with_capacity(closes.len());

        // Первое значение EMA = первая цена закрытия
        ema_values.push(closes[0]);

        for close in closes.iter().skip(1) {
            let prev_ema = *ema_values.last().unwrap();
            let new_ema = (close - prev_ema) * multiplier + prev_ema;
            ema_values.push(new_ema);
        }

        ema_values
    }

    /// Рассчитать RSI (Relative Strength Index)
    pub fn rsi(klines: &[Kline], period: usize) -> Vec<f64> {
        if klines.len() < period + 1 {
            return vec![];
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();

        // Рассчитываем изменения цены
        let changes: Vec<f64> = closes
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        let mut gains: Vec<f64> = Vec::new();
        let mut losses: Vec<f64> = Vec::new();

        for change in &changes {
            if *change > 0.0 {
                gains.push(*change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(change.abs());
            }
        }

        let mut rsi_values = Vec::new();

        // Первый средний прирост и убыток
        let mut avg_gain: f64 = gains.iter().take(period).sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses.iter().take(period).sum::<f64>() / period as f64;

        for i in period..changes.len() {
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

            let rs = if avg_loss == 0.0 {
                100.0
            } else {
                avg_gain / avg_loss
            };

            let rsi = 100.0 - (100.0 / (1.0 + rs));
            rsi_values.push(rsi);
        }

        rsi_values
    }

    /// Рассчитать волатильность (стандартное отклонение доходности)
    pub fn volatility(klines: &[Kline], period: usize) -> Vec<f64> {
        if klines.len() < period + 1 {
            return vec![];
        }

        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();

        // Рассчитываем логарифмическую доходность
        let returns: Vec<f64> = closes
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        returns
            .windows(period)
            .map(|window| {
                let mean = window.iter().sum::<f64>() / period as f64;
                let variance = window
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / period as f64;
                variance.sqrt()
            })
            .collect()
    }

    /// Определить тренд на основе SMA
    pub fn trend(klines: &[Kline], short_period: usize, long_period: usize) -> Option<Trend> {
        let short_sma = Self::sma(klines, short_period);
        let long_sma = Self::sma(klines, long_period);

        if short_sma.is_empty() || long_sma.is_empty() {
            return None;
        }

        let last_short = *short_sma.last().unwrap();
        let last_long = *long_sma.last().unwrap();

        let diff_percent = (last_short - last_long) / last_long * 100.0;

        Some(if diff_percent > 2.0 {
            Trend::StrongBullish
        } else if diff_percent > 0.5 {
            Trend::Bullish
        } else if diff_percent < -2.0 {
            Trend::StrongBearish
        } else if diff_percent < -0.5 {
            Trend::Bearish
        } else {
            Trend::Sideways
        })
    }
}

/// Направление тренда
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    /// Сильный бычий тренд
    StrongBullish,
    /// Бычий тренд
    Bullish,
    /// Боковой тренд
    Sideways,
    /// Медвежий тренд
    Bearish,
    /// Сильный медвежий тренд
    StrongBearish,
}

impl std::fmt::Display for Trend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Trend::StrongBullish => write!(f, "Strong Bullish 📈📈"),
            Trend::Bullish => write!(f, "Bullish 📈"),
            Trend::Sideways => write!(f, "Sideways ➡️"),
            Trend::Bearish => write!(f, "Bearish 📉"),
            Trend::StrongBearish => write!(f, "Strong Bearish 📉📉"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_klines(closes: Vec<f64>) -> Vec<Kline> {
        closes
            .into_iter()
            .map(|close| Kline {
                open_time: Utc::now(),
                open: close,
                high: close * 1.01,
                low: close * 0.99,
                close,
                volume: 1000.0,
                turnover: close * 1000.0,
            })
            .collect()
    }

    #[test]
    fn test_sma() {
        let klines = create_test_klines(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let sma = TechnicalIndicators::sma(&klines, 3);

        assert_eq!(sma.len(), 3);
        assert!((sma[0] - 20.0).abs() < 0.001); // (10+20+30)/3
        assert!((sma[1] - 30.0).abs() < 0.001); // (20+30+40)/3
        assert!((sma[2] - 40.0).abs() < 0.001); // (30+40+50)/3
    }

    #[test]
    fn test_interval_to_string() {
        assert_eq!(Interval::H1.as_str(), "60");
        assert_eq!(Interval::D1.as_str(), "D");
    }
}
