//! Модуль альфа-факторов и технических индикаторов
//!
//! Содержит реализации:
//! - Трендовых индикаторов (SMA, EMA, MACD, Bollinger Bands)
//! - Индикаторов моментума (RSI, ROC, Momentum)
//! - Индикаторов объёма (OBV, VWAP)
//! - Индикаторов волатильности (ATR, Historical Volatility)
//! - Формульных альфа-факторов из статьи WorldQuant 101 Alphas

pub mod trend;
pub mod momentum;
pub mod volume;
pub mod volatility;
pub mod alpha;
pub mod utils;

// Re-exports для удобства
pub use trend::*;
pub use momentum::*;
pub use volume::*;
pub use volatility::*;
pub use alpha::*;

/// Результат расчёта сигнала
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    /// Сильный сигнал на покупку
    StrongBuy,
    /// Сигнал на покупку
    Buy,
    /// Нейтрально (держать)
    Neutral,
    /// Сигнал на продажу
    Sell,
    /// Сильный сигнал на продажу
    StrongSell,
}

impl Signal {
    /// Преобразовать в числовое значение (-2, -1, 0, 1, 2)
    pub fn to_score(&self) -> i32 {
        match self {
            Signal::StrongBuy => 2,
            Signal::Buy => 1,
            Signal::Neutral => 0,
            Signal::Sell => -1,
            Signal::StrongSell => -2,
        }
    }

    /// Создать сигнал из числового значения
    pub fn from_score(score: f64) -> Self {
        if score >= 1.5 {
            Signal::StrongBuy
        } else if score >= 0.5 {
            Signal::Buy
        } else if score <= -1.5 {
            Signal::StrongSell
        } else if score <= -0.5 {
            Signal::Sell
        } else {
            Signal::Neutral
        }
    }
}

/// Расчёт нескольких факторов сразу
pub struct FactorCalculator {
    /// Цены закрытия
    closes: Vec<f64>,
    /// Максимумы
    highs: Vec<f64>,
    /// Минимумы
    lows: Vec<f64>,
    /// Объёмы
    volumes: Vec<f64>,
}

impl FactorCalculator {
    /// Создать калькулятор из OHLCV данных
    pub fn new(
        closes: Vec<f64>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        volumes: Vec<f64>,
    ) -> Self {
        Self {
            closes,
            highs,
            lows,
            volumes,
        }
    }

    /// Создать из списка свечей
    pub fn from_klines(klines: &[crate::Kline]) -> Self {
        use crate::data::kline::KlineVec;
        let klines_vec: Vec<_> = klines.to_vec();
        Self {
            closes: klines_vec.closes(),
            highs: klines_vec.highs(),
            lows: klines_vec.lows(),
            volumes: klines_vec.volumes(),
        }
    }

    /// Рассчитать все основные индикаторы
    pub fn calculate_all(&self) -> FactorSet {
        FactorSet {
            sma_20: sma(&self.closes, 20),
            sma_50: sma(&self.closes, 50),
            ema_12: ema(&self.closes, 12),
            ema_26: ema(&self.closes, 26),
            rsi_14: rsi(&self.closes, 14),
            macd: macd(&self.closes, 12, 26, 9),
            bollinger: bollinger_bands(&self.closes, 20, 2.0),
            atr_14: atr(&self.highs, &self.lows, &self.closes, 14),
            obv: obv(&self.closes, &self.volumes),
        }
    }

    /// Получить цены закрытия
    pub fn closes(&self) -> &[f64] {
        &self.closes
    }
}

/// Набор рассчитанных факторов
#[derive(Debug)]
pub struct FactorSet {
    pub sma_20: Vec<f64>,
    pub sma_50: Vec<f64>,
    pub ema_12: Vec<f64>,
    pub ema_26: Vec<f64>,
    pub rsi_14: Vec<f64>,
    pub macd: MACDResult,
    pub bollinger: BollingerResult,
    pub atr_14: Vec<f64>,
    pub obv: Vec<f64>,
}

impl FactorSet {
    /// Получить последние значения всех индикаторов
    pub fn last_values(&self) -> Option<FactorSnapshot> {
        Some(FactorSnapshot {
            sma_20: *self.sma_20.last()?,
            sma_50: *self.sma_50.last()?,
            ema_12: *self.ema_12.last()?,
            ema_26: *self.ema_26.last()?,
            rsi_14: *self.rsi_14.last()?,
            macd_line: *self.macd.macd_line.last()?,
            macd_signal: *self.macd.signal_line.last()?,
            macd_histogram: *self.macd.histogram.last()?,
            bb_upper: *self.bollinger.upper.last()?,
            bb_middle: *self.bollinger.middle.last()?,
            bb_lower: *self.bollinger.lower.last()?,
            atr_14: *self.atr_14.last()?,
            obv: *self.obv.last()?,
        })
    }
}

/// Снимок значений факторов в один момент времени
#[derive(Debug, Clone)]
pub struct FactorSnapshot {
    pub sma_20: f64,
    pub sma_50: f64,
    pub ema_12: f64,
    pub ema_26: f64,
    pub rsi_14: f64,
    pub macd_line: f64,
    pub macd_signal: f64,
    pub macd_histogram: f64,
    pub bb_upper: f64,
    pub bb_middle: f64,
    pub bb_lower: f64,
    pub atr_14: f64,
    pub obv: f64,
}

impl FactorSnapshot {
    /// Генерировать торговый сигнал на основе всех индикаторов
    pub fn generate_signal(&self, current_price: f64) -> Signal {
        let mut score: f64 = 0.0;

        // RSI сигнал
        if self.rsi_14 < 30.0 {
            score += 1.0; // Перепродан - покупать
        } else if self.rsi_14 > 70.0 {
            score -= 1.0; // Перекуплен - продавать
        }

        // MACD сигнал
        if self.macd_histogram > 0.0 {
            score += 0.5;
        } else {
            score -= 0.5;
        }

        // Bollinger Bands сигнал
        if current_price < self.bb_lower {
            score += 1.0; // Ниже нижней полосы - покупать
        } else if current_price > self.bb_upper {
            score -= 1.0; // Выше верхней полосы - продавать
        }

        // Тренд (SMA 20 vs SMA 50)
        if self.sma_20 > self.sma_50 {
            score += 0.5; // Бычий тренд
        } else {
            score -= 0.5; // Медвежий тренд
        }

        Signal::from_score(score)
    }
}
