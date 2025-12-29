//! # Модели данных
//!
//! Типы данных для работы с опционами, позициями и рыночными данными.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::greeks::{Greeks, OptionType};

/// Опционный контракт
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionContract {
    /// Символ базового актива
    pub symbol: String,
    /// Страйк-цена
    pub strike: f64,
    /// Дата экспирации
    pub expiry: DateTime<Utc>,
    /// Тип опциона
    pub option_type: OptionType,
    /// Текущая цена опциона
    pub price: f64,
    /// Подразумеваемая волатильность
    pub iv: f64,
    /// Открытый интерес
    pub open_interest: Option<f64>,
    /// Объём
    pub volume: Option<f64>,
    /// Bid цена
    pub bid: Option<f64>,
    /// Ask цена
    pub ask: Option<f64>,
}

impl OptionContract {
    /// Создать новый контракт
    pub fn new(
        symbol: impl Into<String>,
        strike: f64,
        expiry: DateTime<Utc>,
        option_type: OptionType,
        price: f64,
        iv: f64,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            strike,
            expiry,
            option_type,
            price,
            iv,
            open_interest: None,
            volume: None,
            bid: None,
            ask: None,
        }
    }

    /// Время до экспирации в днях
    pub fn days_to_expiry(&self) -> f64 {
        let now = Utc::now();
        let duration = self.expiry.signed_duration_since(now);
        duration.num_hours() as f64 / 24.0
    }

    /// Время до экспирации в годах
    pub fn time_to_expiry(&self) -> f64 {
        self.days_to_expiry() / 365.0
    }

    /// Спред bid-ask
    pub fn spread(&self) -> Option<f64> {
        match (self.bid, self.ask) {
            (Some(b), Some(a)) => Some(a - b),
            _ => None,
        }
    }

    /// Спред в процентах от mid
    pub fn spread_pct(&self) -> Option<f64> {
        match (self.bid, self.ask) {
            (Some(b), Some(a)) => {
                let mid = (a + b) / 2.0;
                if mid > 0.0 {
                    Some((a - b) / mid)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// Позиция в опционе
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionPosition {
    /// Контракт
    pub contract: OptionContract,
    /// Количество (положительное = long, отрицательное = short)
    pub quantity: f64,
    /// Цена входа
    pub entry_price: f64,
    /// Дата входа
    pub entry_date: DateTime<Utc>,
    /// Греки позиции
    pub greeks: Greeks,
}

impl OptionPosition {
    /// Создать новую позицию
    pub fn new(contract: OptionContract, quantity: f64, entry_price: f64) -> Self {
        Self {
            contract,
            quantity,
            entry_price,
            entry_date: Utc::now(),
            greeks: Greeks::zero(),
        }
    }

    /// Обновить греки
    pub fn with_greeks(mut self, greeks: Greeks) -> Self {
        self.greeks = Greeks {
            delta: greeks.delta * self.quantity,
            gamma: greeks.gamma * self.quantity,
            theta: greeks.theta * self.quantity,
            vega: greeks.vega * self.quantity,
            rho: greeks.rho * self.quantity,
        };
        self
    }

    /// Текущая P&L (в единицах базового актива)
    pub fn pnl(&self) -> f64 {
        (self.contract.price - self.entry_price) * self.quantity
    }

    /// P&L в процентах от entry
    pub fn pnl_pct(&self) -> f64 {
        if self.entry_price > 0.0 {
            (self.contract.price - self.entry_price) / self.entry_price
        } else {
            0.0
        }
    }

    /// Это long позиция?
    pub fn is_long(&self) -> bool {
        self.quantity > 0.0
    }

    /// Это short позиция?
    pub fn is_short(&self) -> bool {
        self.quantity < 0.0
    }
}

/// Портфель опционов
#[derive(Debug, Clone, Default)]
pub struct Portfolio {
    /// Позиции в опционах
    pub options: Vec<OptionPosition>,
    /// Позиция в базовом активе
    pub underlying_position: f64,
    /// Средняя цена базового актива
    pub underlying_avg_price: f64,
    /// Текущая цена базового актива
    pub underlying_spot: f64,
    /// Баланс наличных
    pub cash: f64,
}

impl Portfolio {
    /// Создать пустой портфель
    pub fn new(initial_cash: f64) -> Self {
        Self {
            options: Vec::new(),
            underlying_position: 0.0,
            underlying_avg_price: 0.0,
            underlying_spot: 0.0,
            cash: initial_cash,
        }
    }

    /// Добавить опционную позицию
    pub fn add_option(&mut self, position: OptionPosition) {
        // Проверяем, есть ли уже позиция в этом контракте
        let existing = self.options.iter_mut().find(|p| {
            p.contract.symbol == position.contract.symbol
                && p.contract.strike == position.contract.strike
                && p.contract.expiry == position.contract.expiry
                && p.contract.option_type == position.contract.option_type
        });

        if let Some(existing) = existing {
            // Обновляем существующую позицию
            let total_qty = existing.quantity + position.quantity;
            if total_qty.abs() < 0.0001 {
                // Позиция закрыта
                self.options.retain(|p| {
                    !(p.contract.symbol == position.contract.symbol
                        && p.contract.strike == position.contract.strike
                        && p.contract.expiry == position.contract.expiry
                        && p.contract.option_type == position.contract.option_type)
                });
            } else {
                existing.quantity = total_qty;
            }
        } else {
            self.options.push(position);
        }
    }

    /// Обновить позицию в базовом активе
    pub fn update_underlying(&mut self, delta_position: f64, price: f64) {
        let new_position = self.underlying_position + delta_position;

        if new_position.abs() > 0.0001 {
            // Пересчитываем среднюю цену
            if self.underlying_position.signum() == delta_position.signum() {
                // Добавляем к существующей позиции
                self.underlying_avg_price = (self.underlying_avg_price
                    * self.underlying_position
                    + price * delta_position)
                    / new_position;
            } else if delta_position.abs() > self.underlying_position.abs() {
                // Разворот позиции
                self.underlying_avg_price = price;
            }
            // Если просто уменьшаем позицию, avg_price остаётся
        } else {
            self.underlying_avg_price = 0.0;
        }

        self.underlying_position = new_position;
        self.underlying_spot = price;
    }

    /// Общая дельта портфеля
    pub fn total_delta(&self) -> f64 {
        let options_delta: f64 = self.options.iter().map(|p| p.greeks.delta).sum();
        options_delta + self.underlying_position
    }

    /// Общая гамма портфеля
    pub fn total_gamma(&self) -> f64 {
        self.options.iter().map(|p| p.greeks.gamma).sum()
    }

    /// Общая тета портфеля
    pub fn total_theta(&self) -> f64 {
        self.options.iter().map(|p| p.greeks.theta).sum()
    }

    /// Общая вега портфеля
    pub fn total_vega(&self) -> f64 {
        self.options.iter().map(|p| p.greeks.vega).sum()
    }

    /// Все греки портфеля
    pub fn total_greeks(&self) -> Greeks {
        let mut total = Greeks::zero();
        for p in &self.options {
            total = total.add(&p.greeks, 1.0);
        }
        // Дельта базового актива
        total.delta += self.underlying_position;
        total
    }

    /// Общая P&L по опционам
    pub fn options_pnl(&self) -> f64 {
        self.options.iter().map(|p| p.pnl()).sum()
    }

    /// P&L по базовому активу
    pub fn underlying_pnl(&self) -> f64 {
        if self.underlying_position.abs() > 0.0001 {
            (self.underlying_spot - self.underlying_avg_price) * self.underlying_position
        } else {
            0.0
        }
    }

    /// Общая P&L
    pub fn total_pnl(&self) -> f64 {
        self.options_pnl() + self.underlying_pnl()
    }
}

/// Свеча OHLCV
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Время открытия
    pub timestamp: DateTime<Utc>,
    /// Цена открытия
    pub open: f64,
    /// Максимум
    pub high: f64,
    /// Минимум
    pub low: f64,
    /// Цена закрытия
    pub close: f64,
    /// Объём
    pub volume: f64,
}

impl Candle {
    /// Создать новую свечу
    pub fn new(
        timestamp: DateTime<Utc>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Доходность (close-to-close)
    pub fn returns(&self, prev_close: f64) -> f64 {
        if prev_close > 0.0 {
            (self.close / prev_close).ln()
        } else {
            0.0
        }
    }

    /// Это бычья свеча?
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Размах (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Тело свечи
    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }
}

/// Сделка
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Время сделки
    pub timestamp: DateTime<Utc>,
    /// Символ
    pub symbol: String,
    /// Сторона (buy/sell)
    pub side: TradeSide,
    /// Количество
    pub quantity: f64,
    /// Цена
    pub price: f64,
    /// Комиссия
    pub fee: f64,
    /// Тип инструмента
    pub instrument_type: InstrumentType,
}

/// Сторона сделки
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Тип инструмента
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InstrumentType {
    Spot,
    Perpetual,
    Option,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_option_contract() {
        let expiry = Utc::now() + Duration::days(7);
        let contract = OptionContract::new("BTC", 42000.0, expiry, OptionType::Call, 800.0, 0.55);

        assert!(contract.days_to_expiry() > 6.9 && contract.days_to_expiry() < 7.1);
    }

    #[test]
    fn test_portfolio_delta() {
        let mut portfolio = Portfolio::new(10000.0);

        // Добавляем long call
        let expiry = Utc::now() + Duration::days(7);
        let contract = OptionContract::new("BTC", 42000.0, expiry, OptionType::Call, 800.0, 0.55);
        let mut position = OptionPosition::new(contract, 1.0, 800.0);
        position.greeks = Greeks::new(0.5, 0.001, -10.0, 20.0, 0.1);

        portfolio.add_option(position);

        assert!((portfolio.total_delta() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_portfolio_hedge() {
        let mut portfolio = Portfolio::new(10000.0);

        // Добавляем long call с дельтой 0.5
        let expiry = Utc::now() + Duration::days(7);
        let contract = OptionContract::new("BTC", 42000.0, expiry, OptionType::Call, 800.0, 0.55);
        let mut position = OptionPosition::new(contract, 1.0, 800.0);
        position.greeks = Greeks::new(0.5, 0.001, -10.0, 20.0, 0.1);
        portfolio.add_option(position);

        // Хеджируем продажей 0.5 базового актива
        portfolio.update_underlying(-0.5, 42000.0);

        assert!(portfolio.total_delta().abs() < 0.001);
    }
}
