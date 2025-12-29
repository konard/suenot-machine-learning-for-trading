//! Управление позициями

use super::signal::SignalDirection;
use serde::{Deserialize, Serialize};

/// Статус позиции
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionStatus {
    /// Открыта
    Open,
    /// Закрыта
    Closed,
    /// Ликвидирована (стоп-лосс)
    Liquidated,
}

/// Торговая позиция
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Уникальный ID
    pub id: u64,
    /// Символ
    pub symbol: String,
    /// Направление
    pub direction: SignalDirection,
    /// Цена входа
    pub entry_price: f64,
    /// Цена выхода
    pub exit_price: Option<f64>,
    /// Размер позиции (в базовой валюте)
    pub size: f64,
    /// Время открытия
    pub entry_time: i64,
    /// Время закрытия
    pub exit_time: Option<i64>,
    /// Статус
    pub status: PositionStatus,
    /// Стоп-лосс
    pub stop_loss: Option<f64>,
    /// Тейк-профит
    pub take_profit: Option<f64>,
    /// Комиссия
    pub commission: f64,
}

impl Position {
    /// Создание новой позиции
    pub fn new(
        id: u64,
        symbol: &str,
        direction: SignalDirection,
        entry_price: f64,
        size: f64,
        entry_time: i64,
    ) -> Self {
        Self {
            id,
            symbol: symbol.to_string(),
            direction,
            entry_price,
            exit_price: None,
            size,
            entry_time,
            exit_time: None,
            status: PositionStatus::Open,
            stop_loss: None,
            take_profit: None,
            commission: 0.0,
        }
    }

    /// Установка стоп-лосса и тейк-профита
    pub fn with_risk_management(
        mut self,
        stop_loss_pct: f64,
        take_profit_pct: f64,
    ) -> Self {
        match self.direction {
            SignalDirection::Long => {
                self.stop_loss = Some(self.entry_price * (1.0 - stop_loss_pct / 100.0));
                self.take_profit = Some(self.entry_price * (1.0 + take_profit_pct / 100.0));
            }
            SignalDirection::Short => {
                self.stop_loss = Some(self.entry_price * (1.0 + stop_loss_pct / 100.0));
                self.take_profit = Some(self.entry_price * (1.0 - take_profit_pct / 100.0));
            }
            SignalDirection::Neutral => {}
        }
        self
    }

    /// Установка комиссии
    pub fn with_commission(mut self, rate: f64) -> Self {
        self.commission = self.size * self.entry_price * rate;
        self
    }

    /// Проверка стоп-лосса
    pub fn check_stop_loss(&self, current_price: f64) -> bool {
        if let Some(sl) = self.stop_loss {
            match self.direction {
                SignalDirection::Long => current_price <= sl,
                SignalDirection::Short => current_price >= sl,
                _ => false,
            }
        } else {
            false
        }
    }

    /// Проверка тейк-профита
    pub fn check_take_profit(&self, current_price: f64) -> bool {
        if let Some(tp) = self.take_profit {
            match self.direction {
                SignalDirection::Long => current_price >= tp,
                SignalDirection::Short => current_price <= tp,
                _ => false,
            }
        } else {
            false
        }
    }

    /// Закрытие позиции
    pub fn close(&mut self, exit_price: f64, exit_time: i64, commission_rate: f64) {
        self.exit_price = Some(exit_price);
        self.exit_time = Some(exit_time);
        self.status = PositionStatus::Closed;
        self.commission += self.size * exit_price * commission_rate;
    }

    /// Ликвидация (стоп-лосс)
    pub fn liquidate(&mut self, exit_price: f64, exit_time: i64, commission_rate: f64) {
        self.exit_price = Some(exit_price);
        self.exit_time = Some(exit_time);
        self.status = PositionStatus::Liquidated;
        self.commission += self.size * exit_price * commission_rate;
    }

    /// Нереализованная прибыль/убыток
    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        let price_diff = match self.direction {
            SignalDirection::Long => current_price - self.entry_price,
            SignalDirection::Short => self.entry_price - current_price,
            SignalDirection::Neutral => 0.0,
        };
        price_diff * self.size - self.commission
    }

    /// Реализованная прибыль/убыток
    pub fn realized_pnl(&self) -> Option<f64> {
        self.exit_price.map(|exit| {
            let price_diff = match self.direction {
                SignalDirection::Long => exit - self.entry_price,
                SignalDirection::Short => self.entry_price - exit,
                SignalDirection::Neutral => 0.0,
            };
            price_diff * self.size - self.commission
        })
    }

    /// Доходность позиции в процентах
    pub fn return_pct(&self) -> Option<f64> {
        self.realized_pnl().map(|pnl| {
            let invested = self.entry_price * self.size;
            if invested > 0.0 {
                pnl / invested * 100.0
            } else {
                0.0
            }
        })
    }

    /// Длительность позиции в минутах
    pub fn duration_minutes(&self) -> Option<i64> {
        self.exit_time
            .map(|exit| (exit - self.entry_time) / 60000)
    }

    /// Открыта ли позиция
    pub fn is_open(&self) -> bool {
        self.status == PositionStatus::Open
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_pnl() {
        let mut pos = Position::new(1, "BTCUSDT", SignalDirection::Long, 50000.0, 0.1, 1000);

        // Unrealized PnL
        assert!((pos.unrealized_pnl(51000.0) - 100.0).abs() < 0.01);
        assert!((pos.unrealized_pnl(49000.0) - (-100.0)).abs() < 0.01);

        // Close position
        pos.close(52000.0, 2000, 0.0);
        let pnl = pos.realized_pnl().unwrap();
        assert!((pnl - 200.0).abs() < 0.01);
    }

    #[test]
    fn test_stop_loss() {
        let pos = Position::new(1, "BTCUSDT", SignalDirection::Long, 50000.0, 0.1, 1000)
            .with_risk_management(2.0, 4.0);

        assert!(pos.stop_loss.is_some());
        assert!(pos.check_stop_loss(48000.0));
        assert!(!pos.check_stop_loss(50000.0));
    }

    #[test]
    fn test_take_profit() {
        let pos = Position::new(1, "BTCUSDT", SignalDirection::Long, 50000.0, 0.1, 1000)
            .with_risk_management(2.0, 4.0);

        assert!(pos.take_profit.is_some());
        assert!(pos.check_take_profit(53000.0));
        assert!(!pos.check_take_profit(50000.0));
    }
}
