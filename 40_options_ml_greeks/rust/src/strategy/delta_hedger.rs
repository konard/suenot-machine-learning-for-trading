//! # Delta Hedging
//!
//! Поддержание дельта-нейтральной позиции для изоляции
//! ставки на волатильность от направления цены.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::models::Portfolio;

/// Запись о хедж-сделке
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HedgeTrade {
    /// Время сделки
    pub timestamp: DateTime<Utc>,
    /// Количество базового актива (+ = buy, - = sell)
    pub quantity: f64,
    /// Цена исполнения
    pub price: f64,
    /// Дельта до хеджа
    pub delta_before: f64,
    /// Дельта после хеджа
    pub delta_after: f64,
    /// Транзакционные издержки
    pub transaction_cost: f64,
}

/// Конфигурация хеджера
#[derive(Debug, Clone)]
pub struct HedgerConfig {
    /// Порог для ребалансировки (абсолютная дельта)
    pub hedge_threshold: f64,
    /// Транзакционные издержки (в долях от объёма)
    pub transaction_cost_rate: f64,
    /// Минимальный размер хедж-сделки
    pub min_hedge_size: f64,
    /// Максимальный размер хедж-сделки
    pub max_hedge_size: f64,
}

impl Default for HedgerConfig {
    fn default() -> Self {
        Self {
            hedge_threshold: 0.05,       // 5% от позиции
            transaction_cost_rate: 0.001, // 0.1% (10 bps)
            min_hedge_size: 0.001,        // Минимум 0.001 BTC
            max_hedge_size: 10.0,         // Максимум 10 BTC за раз
        }
    }
}

/// Дельта-хеджер
#[derive(Debug, Clone)]
pub struct DeltaHedger {
    /// Конфигурация
    config: HedgerConfig,
    /// История хедж-сделок
    history: Vec<HedgeTrade>,
    /// Общие транзакционные издержки
    total_costs: f64,
}

impl DeltaHedger {
    /// Создать новый хеджер
    pub fn new(config: HedgerConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            total_costs: 0.0,
        }
    }

    /// Создать хеджер с настройками по умолчанию
    pub fn default_crypto() -> Self {
        Self::new(HedgerConfig::default())
    }

    /// Проверить, нужен ли хедж
    pub fn needs_hedge(&self, current_delta: f64) -> bool {
        current_delta.abs() > self.config.hedge_threshold
    }

    /// Рассчитать необходимый хедж
    pub fn calculate_hedge(&self, current_delta: f64) -> f64 {
        if !self.needs_hedge(current_delta) {
            return 0.0;
        }

        // Хеджируем в противоположном направлении
        let raw_hedge = -current_delta;

        // Применяем ограничения
        let clamped = raw_hedge
            .abs()
            .min(self.config.max_hedge_size)
            .max(self.config.min_hedge_size);

        clamped * raw_hedge.signum()
    }

    /// Выполнить хедж-сделку
    pub fn execute_hedge(
        &mut self,
        portfolio: &mut Portfolio,
        spot_price: f64,
    ) -> Option<HedgeTrade> {
        let current_delta = portfolio.total_delta();

        if !self.needs_hedge(current_delta) {
            return None;
        }

        let hedge_quantity = self.calculate_hedge(current_delta);

        // Транзакционные издержки
        let notional = hedge_quantity.abs() * spot_price;
        let transaction_cost = notional * self.config.transaction_cost_rate;

        // Обновляем портфель
        portfolio.update_underlying(hedge_quantity, spot_price);
        portfolio.cash -= transaction_cost;

        let trade = HedgeTrade {
            timestamp: Utc::now(),
            quantity: hedge_quantity,
            price: spot_price,
            delta_before: current_delta,
            delta_after: portfolio.total_delta(),
            transaction_cost,
        };

        self.total_costs += transaction_cost;
        self.history.push(trade.clone());

        Some(trade)
    }

    /// Выполнить хедж до целевой дельты
    pub fn hedge_to_target(
        &mut self,
        portfolio: &mut Portfolio,
        target_delta: f64,
        spot_price: f64,
    ) -> Option<HedgeTrade> {
        let current_delta = portfolio.total_delta();
        let delta_diff = current_delta - target_delta;

        if delta_diff.abs() < self.config.min_hedge_size {
            return None;
        }

        let hedge_quantity = -delta_diff;
        let notional = hedge_quantity.abs() * spot_price;
        let transaction_cost = notional * self.config.transaction_cost_rate;

        portfolio.update_underlying(hedge_quantity, spot_price);
        portfolio.cash -= transaction_cost;

        let trade = HedgeTrade {
            timestamp: Utc::now(),
            quantity: hedge_quantity,
            price: spot_price,
            delta_before: current_delta,
            delta_after: portfolio.total_delta(),
            transaction_cost,
        };

        self.total_costs += transaction_cost;
        self.history.push(trade.clone());

        Some(trade)
    }

    /// История хедж-сделок
    pub fn history(&self) -> &[HedgeTrade] {
        &self.history
    }

    /// Общие транзакционные издержки
    pub fn total_costs(&self) -> f64 {
        self.total_costs
    }

    /// Количество хедж-сделок
    pub fn num_hedges(&self) -> usize {
        self.history.len()
    }

    /// Средний размер хедж-сделки
    pub fn avg_hedge_size(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }

        self.history.iter().map(|h| h.quantity.abs()).sum::<f64>() / self.history.len() as f64
    }

    /// P&L от хеджирования (исключая издержки)
    pub fn hedging_pnl(&self, current_price: f64) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }

        // Рассчитываем P&L от всех хедж-сделок
        let mut total_pnl = 0.0;
        let mut position = 0.0;
        let mut cost_basis = 0.0;

        for trade in &self.history {
            if position.signum() == trade.quantity.signum() || position.abs() < 0.0001 {
                // Добавляем к позиции
                cost_basis = (cost_basis * position + trade.price * trade.quantity)
                    / (position + trade.quantity);
                position += trade.quantity;
            } else {
                // Закрываем часть позиции
                let closed_qty = trade.quantity.abs().min(position.abs());
                total_pnl += (trade.price - cost_basis) * closed_qty * position.signum();
                position += trade.quantity;
            }
        }

        // Добавляем нереализованный P&L
        if position.abs() > 0.0001 {
            total_pnl += (current_price - cost_basis) * position;
        }

        total_pnl
    }

    /// Сброс истории
    pub fn reset(&mut self) {
        self.history.clear();
        self.total_costs = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::greeks::{Greeks, OptionType};
    use crate::models::{OptionContract, OptionPosition};
    use chrono::Duration;

    fn create_test_portfolio() -> Portfolio {
        let mut portfolio = Portfolio::new(100000.0);

        // Добавляем long call с дельтой 0.6
        let expiry = Utc::now() + Duration::days(7);
        let contract = OptionContract::new("BTC", 42000.0, expiry, OptionType::Call, 800.0, 0.55);
        let position = OptionPosition::new(contract, 1.0, 800.0)
            .with_greeks(Greeks::new(0.6, 0.001, -10.0, 20.0, 0.1));

        portfolio.add_option(position);
        portfolio.underlying_spot = 42000.0;

        portfolio
    }

    #[test]
    fn test_needs_hedge() {
        let hedger = DeltaHedger::default_crypto();

        assert!(!hedger.needs_hedge(0.03)); // Меньше порога
        assert!(hedger.needs_hedge(0.10));  // Больше порога
        assert!(hedger.needs_hedge(-0.10)); // Отрицательная дельта тоже
    }

    #[test]
    fn test_calculate_hedge() {
        let hedger = DeltaHedger::default_crypto();

        // Положительная дельта → продаём
        let hedge = hedger.calculate_hedge(0.5);
        assert!(hedge < 0.0);
        assert!((hedge.abs() - 0.5).abs() < 0.01);

        // Отрицательная дельта → покупаем
        let hedge = hedger.calculate_hedge(-0.5);
        assert!(hedge > 0.0);
    }

    #[test]
    fn test_execute_hedge() {
        let mut hedger = DeltaHedger::default_crypto();
        let mut portfolio = create_test_portfolio();

        // Начальная дельта = 0.6 (от call опциона)
        assert!((portfolio.total_delta() - 0.6).abs() < 0.01);

        // Выполняем хедж
        let trade = hedger.execute_hedge(&mut portfolio, 42000.0);
        assert!(trade.is_some());

        let trade = trade.unwrap();
        assert!(trade.quantity < 0.0); // Продали базовый актив
        assert!(trade.delta_after.abs() < trade.delta_before.abs()); // Дельта уменьшилась
    }

    #[test]
    fn test_hedge_to_target() {
        let mut hedger = DeltaHedger::default_crypto();
        let mut portfolio = create_test_portfolio();

        // Хеджируем до дельты = 0
        hedger.hedge_to_target(&mut portfolio, 0.0, 42000.0);

        assert!(portfolio.total_delta().abs() < 0.01);
    }
}
