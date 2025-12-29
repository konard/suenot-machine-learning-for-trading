//! Базовые типы и трейты для стратегий

use crate::data::Kline;

/// Торговый сигнал
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// Сигнал на покупку
    Buy,
    /// Сигнал на продажу
    Sell,
    /// Нет сигнала (удержание позиции)
    Hold,
}

impl Signal {
    /// Проверить, является ли сигнал активным (не Hold)
    pub fn is_active(&self) -> bool {
        !matches!(self, Signal::Hold)
    }

    /// Инвертировать сигнал
    pub fn invert(&self) -> Self {
        match self {
            Signal::Buy => Signal::Sell,
            Signal::Sell => Signal::Buy,
            Signal::Hold => Signal::Hold,
        }
    }
}

/// Текущая позиция
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Position {
    /// Нет позиции
    None,
    /// Длинная позиция (купили)
    Long(f64), // цена входа
    /// Короткая позиция (продали)
    Short(f64), // цена входа
}

impl Position {
    /// Есть ли открытая позиция
    pub fn is_open(&self) -> bool {
        !matches!(self, Position::None)
    }

    /// Длинная позиция?
    pub fn is_long(&self) -> bool {
        matches!(self, Position::Long(_))
    }

    /// Короткая позиция?
    pub fn is_short(&self) -> bool {
        matches!(self, Position::Short(_))
    }

    /// Получить цену входа
    pub fn entry_price(&self) -> Option<f64> {
        match self {
            Position::None => None,
            Position::Long(price) | Position::Short(price) => Some(*price),
        }
    }

    /// Рассчитать нереализованную прибыль/убыток
    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        match self {
            Position::None => 0.0,
            Position::Long(entry) => current_price - entry,
            Position::Short(entry) => entry - current_price,
        }
    }

    /// Рассчитать нереализованную прибыль/убыток в процентах
    pub fn unrealized_pnl_percent(&self, current_price: f64) -> f64 {
        match self {
            Position::None => 0.0,
            Position::Long(entry) => (current_price - entry) / entry * 100.0,
            Position::Short(entry) => (entry - current_price) / entry * 100.0,
        }
    }
}

/// Информация о сделке
#[derive(Debug, Clone)]
pub struct Trade {
    /// Время входа
    pub entry_time: u64,
    /// Цена входа
    pub entry_price: f64,
    /// Время выхода
    pub exit_time: u64,
    /// Цена выхода
    pub exit_price: f64,
    /// Направление (true = long, false = short)
    pub is_long: bool,
    /// Количество
    pub quantity: f64,
    /// Прибыль/убыток
    pub pnl: f64,
    /// Комиссия
    pub commission: f64,
}

impl Trade {
    /// Создать сделку
    pub fn new(
        entry_time: u64,
        entry_price: f64,
        exit_time: u64,
        exit_price: f64,
        is_long: bool,
        quantity: f64,
        commission_rate: f64,
    ) -> Self {
        let gross_pnl = if is_long {
            (exit_price - entry_price) * quantity
        } else {
            (entry_price - exit_price) * quantity
        };

        let commission = (entry_price + exit_price) * quantity * commission_rate;
        let pnl = gross_pnl - commission;

        Self {
            entry_time,
            entry_price,
            exit_time,
            exit_price,
            is_long,
            quantity,
            pnl,
            commission,
        }
    }

    /// Длительность сделки в миллисекундах
    pub fn duration(&self) -> u64 {
        self.exit_time.saturating_sub(self.entry_time)
    }

    /// Прибыль в процентах
    pub fn pnl_percent(&self) -> f64 {
        if self.is_long {
            (self.exit_price - self.entry_price) / self.entry_price * 100.0
        } else {
            (self.entry_price - self.exit_price) / self.entry_price * 100.0
        }
    }

    /// Сделка прибыльная?
    pub fn is_profitable(&self) -> bool {
        self.pnl > 0.0
    }
}

/// Трейт для торговых стратегий
pub trait Strategy {
    /// Название стратегии
    fn name(&self) -> &str;

    /// Сгенерировать сигнал на основе исторических данных
    fn generate_signal(&self, klines: &[Kline]) -> Signal;

    /// Минимальное количество свечей для работы стратегии
    fn min_bars(&self) -> usize;

    /// Генерировать сигналы для всех точек
    fn generate_signals(&self, klines: &[Kline]) -> Vec<Signal> {
        let min_bars = self.min_bars();
        if klines.len() < min_bars {
            return vec![];
        }

        (min_bars..=klines.len())
            .map(|i| self.generate_signal(&klines[..i]))
            .collect()
    }
}

/// Комбинированная стратегия (AND логика)
pub struct CombinedStrategy {
    name: String,
    strategies: Vec<Box<dyn Strategy>>,
}

impl CombinedStrategy {
    pub fn new(name: &str, strategies: Vec<Box<dyn Strategy>>) -> Self {
        Self {
            name: name.to_string(),
            strategies,
        }
    }
}

impl Strategy for CombinedStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn generate_signal(&self, klines: &[Kline]) -> Signal {
        let signals: Vec<Signal> = self
            .strategies
            .iter()
            .map(|s| s.generate_signal(klines))
            .collect();

        // Все стратегии должны согласиться
        if signals.iter().all(|s| *s == Signal::Buy) {
            Signal::Buy
        } else if signals.iter().all(|s| *s == Signal::Sell) {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn min_bars(&self) -> usize {
        self.strategies.iter().map(|s| s.min_bars()).max().unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal() {
        assert!(Signal::Buy.is_active());
        assert!(Signal::Sell.is_active());
        assert!(!Signal::Hold.is_active());

        assert_eq!(Signal::Buy.invert(), Signal::Sell);
        assert_eq!(Signal::Sell.invert(), Signal::Buy);
        assert_eq!(Signal::Hold.invert(), Signal::Hold);
    }

    #[test]
    fn test_position_pnl() {
        let long = Position::Long(100.0);
        assert_eq!(long.unrealized_pnl(110.0), 10.0);
        assert_eq!(long.unrealized_pnl(90.0), -10.0);

        let short = Position::Short(100.0);
        assert_eq!(short.unrealized_pnl(90.0), 10.0);
        assert_eq!(short.unrealized_pnl(110.0), -10.0);
    }

    #[test]
    fn test_trade() {
        let trade = Trade::new(
            1000,
            100.0,
            2000,
            110.0,
            true,  // long
            1.0,
            0.001, // 0.1% commission
        );

        assert!(trade.is_profitable());
        assert!((trade.pnl_percent() - 10.0).abs() < 1e-10);
        assert_eq!(trade.duration(), 1000);
    }
}
