//! Общие типы данных

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Тип торгового сигнала
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    /// Сильный сигнал на покупку
    StrongLong,
    /// Обычный сигнал на покупку
    Long,
    /// Нейтральный сигнал
    Neutral,
    /// Обычный сигнал на продажу
    Short,
    /// Сильный сигнал на продажу
    StrongShort,
    /// Сигнал на закрытие позиции (дивергенция)
    Exit,
}

impl SignalType {
    /// Преобразование сигнала в числовое значение
    pub fn to_value(&self) -> f64 {
        match self {
            SignalType::StrongLong => 1.0,
            SignalType::Long => 0.5,
            SignalType::Neutral => 0.0,
            SignalType::Short => -0.5,
            SignalType::StrongShort => -1.0,
            SignalType::Exit => 0.0,
        }
    }

    /// Создание сигнала из числового значения
    pub fn from_value(value: f64) -> Self {
        if value >= 0.75 {
            SignalType::StrongLong
        } else if value >= 0.25 {
            SignalType::Long
        } else if value <= -0.75 {
            SignalType::StrongShort
        } else if value <= -0.25 {
            SignalType::Short
        } else {
            SignalType::Neutral
        }
    }
}

/// Сила сигнала
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SignalStrength {
    /// Значение силы от -1.0 до 1.0
    pub value: f64,
    /// Уверенность в сигнале от 0.0 до 1.0
    pub confidence: f64,
}

impl SignalStrength {
    /// Создание новой силы сигнала
    pub fn new(value: f64, confidence: f64) -> Self {
        Self {
            value: value.clamp(-1.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Нулевой сигнал
    pub fn zero() -> Self {
        Self {
            value: 0.0,
            confidence: 0.0,
        }
    }
}

/// Торговый сигнал с полной информацией
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Временная метка сигнала
    pub timestamp: DateTime<Utc>,
    /// Символ торговой пары
    pub symbol: String,
    /// Тип сигнала
    pub signal_type: SignalType,
    /// Сила сигнала
    pub strength: SignalStrength,
    /// Компонент ценового моментума
    pub price_momentum: f64,
    /// Компонент sentiment моментума
    pub sentiment_momentum: f64,
    /// Наличие дивергенции
    pub has_divergence: bool,
    /// Тип дивергенции (если есть)
    pub divergence_type: Option<String>,
}

impl TradingSignal {
    /// Создание нового торгового сигнала
    pub fn new(
        timestamp: DateTime<Utc>,
        symbol: String,
        signal_type: SignalType,
        strength: SignalStrength,
        price_momentum: f64,
        sentiment_momentum: f64,
    ) -> Self {
        Self {
            timestamp,
            symbol,
            signal_type,
            strength,
            price_momentum,
            sentiment_momentum,
            has_divergence: false,
            divergence_type: None,
        }
    }

    /// Добавление информации о дивергенции
    pub fn with_divergence(mut self, divergence_type: &str) -> Self {
        self.has_divergence = true;
        self.divergence_type = Some(divergence_type.to_string());
        self
    }

    /// Проверка, является ли сигнал сигналом на покупку
    pub fn is_long(&self) -> bool {
        matches!(self.signal_type, SignalType::Long | SignalType::StrongLong)
    }

    /// Проверка, является ли сигнал сигналом на продажу
    pub fn is_short(&self) -> bool {
        matches!(
            self.signal_type,
            SignalType::Short | SignalType::StrongShort
        )
    }
}

/// Упрощённый сигнал
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Signal {
    /// Значение сигнала от -1.0 до 1.0
    pub value: f64,
    /// Тип сигнала
    pub signal_type: SignalType,
}

impl Signal {
    /// Создание нового сигнала
    pub fn new(value: f64) -> Self {
        Self {
            value: value.clamp(-1.0, 1.0),
            signal_type: SignalType::from_value(value),
        }
    }
}

/// Временной фрейм
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeFrame {
    /// 1 минута
    M1,
    /// 5 минут
    M5,
    /// 15 минут
    M15,
    /// 30 минут
    M30,
    /// 1 час
    H1,
    /// 4 часа
    H4,
    /// 1 день
    D1,
    /// 1 неделя
    W1,
}

impl TimeFrame {
    /// Преобразование в строку для Bybit API
    pub fn to_bybit_interval(&self) -> &'static str {
        match self {
            TimeFrame::M1 => "1",
            TimeFrame::M5 => "5",
            TimeFrame::M15 => "15",
            TimeFrame::M30 => "30",
            TimeFrame::H1 => "60",
            TimeFrame::H4 => "240",
            TimeFrame::D1 => "D",
            TimeFrame::W1 => "W",
        }
    }

    /// Количество миллисекунд во фрейме
    pub fn to_milliseconds(&self) -> i64 {
        match self {
            TimeFrame::M1 => 60_000,
            TimeFrame::M5 => 5 * 60_000,
            TimeFrame::M15 => 15 * 60_000,
            TimeFrame::M30 => 30 * 60_000,
            TimeFrame::H1 => 60 * 60_000,
            TimeFrame::H4 => 4 * 60 * 60_000,
            TimeFrame::D1 => 24 * 60 * 60_000,
            TimeFrame::W1 => 7 * 24 * 60 * 60_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_type_conversion() {
        assert_eq!(SignalType::from_value(0.9), SignalType::StrongLong);
        assert_eq!(SignalType::from_value(0.5), SignalType::Long);
        assert_eq!(SignalType::from_value(0.0), SignalType::Neutral);
        assert_eq!(SignalType::from_value(-0.5), SignalType::Short);
        assert_eq!(SignalType::from_value(-0.9), SignalType::StrongShort);
    }

    #[test]
    fn test_signal_strength_clamping() {
        let strength = SignalStrength::new(1.5, 2.0);
        assert_eq!(strength.value, 1.0);
        assert_eq!(strength.confidence, 1.0);

        let strength = SignalStrength::new(-1.5, -0.5);
        assert_eq!(strength.value, -1.0);
        assert_eq!(strength.confidence, 0.0);
    }

    #[test]
    fn test_timeframe_conversion() {
        assert_eq!(TimeFrame::H1.to_bybit_interval(), "60");
        assert_eq!(TimeFrame::D1.to_bybit_interval(), "D");
        assert_eq!(TimeFrame::H1.to_milliseconds(), 3_600_000);
    }
}
