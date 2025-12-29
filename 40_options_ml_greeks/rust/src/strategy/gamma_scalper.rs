//! # Gamma Scalping
//!
//! Стратегия гамма-скальпинга для long volatility позиций.
//!
//! Идея: Купить straddle и скальпировать дельту при движениях цены.
//! Profit = Gamma * (реализованное движение)^2 - Theta decay

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Запись о скальп-сделке
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalpTrade {
    /// Время сделки
    pub timestamp: DateTime<Utc>,
    /// Количество (+ = buy, - = sell)
    pub quantity: f64,
    /// Цена исполнения
    pub price: f64,
    /// Дельта до скальпа
    pub delta_before: f64,
    /// Реализованный P&L от этой сделки
    pub realized_pnl: f64,
}

/// Конфигурация гамма-скальпера
#[derive(Debug, Clone)]
pub struct GammaScalpConfig {
    /// Порог для ребалансировки (изменение дельты)
    pub rebalance_threshold: f64,
    /// Минимальный размер сделки
    pub min_trade_size: f64,
    /// Комиссия
    pub transaction_cost_rate: f64,
}

impl Default for GammaScalpConfig {
    fn default() -> Self {
        Self {
            rebalance_threshold: 0.10, // 10% изменение дельты
            min_trade_size: 0.001,     // Минимум 0.001 BTC
            transaction_cost_rate: 0.001, // 10 bps
        }
    }
}

/// Гамма-скальпер
#[derive(Debug, Clone)]
pub struct GammaScalper {
    /// Конфигурация
    config: GammaScalpConfig,
    /// История скальп-сделок
    history: Vec<ScalpTrade>,
    /// Предыдущая дельта
    previous_delta: f64,
    /// Накопленный P&L от скальпинга
    scalping_pnl: f64,
    /// Общие транзакционные издержки
    total_costs: f64,
}

impl GammaScalper {
    /// Создать новый скальпер
    pub fn new(config: GammaScalpConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            previous_delta: 0.0,
            scalping_pnl: 0.0,
            total_costs: 0.0,
        }
    }

    /// Создать скальпер по умолчанию
    pub fn default_crypto() -> Self {
        Self::new(GammaScalpConfig::default())
    }

    /// Инициализировать начальную дельту
    pub fn initialize(&mut self, initial_delta: f64) {
        self.previous_delta = initial_delta;
    }

    /// Проверить, нужен ли скальп
    pub fn should_scalp(&self, current_delta: f64) -> bool {
        let delta_change = (current_delta - self.previous_delta).abs();
        delta_change > self.config.rebalance_threshold
    }

    /// Выполнить скальп-сделку
    pub fn execute_scalp(
        &mut self,
        current_delta: f64,
        spot_price: f64,
    ) -> Option<ScalpTrade> {
        if !self.should_scalp(current_delta) {
            return None;
        }

        let delta_change = current_delta - self.previous_delta;

        // Продаём когда дельта выросла, покупаем когда упала
        let trade_quantity = -delta_change;

        if trade_quantity.abs() < self.config.min_trade_size {
            return None;
        }

        // Транзакционные издержки
        let notional = trade_quantity.abs() * spot_price;
        let transaction_cost = notional * self.config.transaction_cost_rate;

        // Рассчитываем P&L
        // Если продаём после роста дельты (цена выросла) → прибыль
        // Если покупаем после падения дельты (цена упала) → покупаем дёшево
        let implied_price_change = delta_change * spot_price; // Примерное изменение цены
        let scalp_pnl = if !self.history.is_empty() {
            // P&L = разница с предыдущей сделкой
            let last = self.history.last().unwrap();
            (spot_price - last.price) * (-trade_quantity)
        } else {
            0.0
        };

        let trade = ScalpTrade {
            timestamp: Utc::now(),
            quantity: trade_quantity,
            price: spot_price,
            delta_before: current_delta,
            realized_pnl: scalp_pnl - transaction_cost,
        };

        self.scalping_pnl += trade.realized_pnl;
        self.total_costs += transaction_cost;
        self.previous_delta = current_delta - delta_change; // Обнуляем изменение
        self.history.push(trade.clone());

        Some(trade)
    }

    /// Рассчитать теоретический P&L от гаммы
    ///
    /// P&L = 0.5 * Gamma * S^2 * (realized_vol^2 - implied_vol^2) * T
    ///
    /// Упрощённо: P&L = 0.5 * Gamma * sum(daily_moves^2)
    pub fn theoretical_gamma_pnl(gamma: f64, spot: f64, daily_moves: &[f64]) -> f64 {
        let sum_squared_moves: f64 = daily_moves.iter().map(|m| m.powi(2)).sum();
        0.5 * gamma * spot.powi(2) * sum_squared_moves
    }

    /// История скальп-сделок
    pub fn history(&self) -> &[ScalpTrade] {
        &self.history
    }

    /// Накопленный P&L от скальпинга
    pub fn scalping_pnl(&self) -> f64 {
        self.scalping_pnl
    }

    /// Общие транзакционные издержки
    pub fn total_costs(&self) -> f64 {
        self.total_costs
    }

    /// Чистый P&L (скальпинг - издержки)
    pub fn net_pnl(&self) -> f64 {
        self.scalping_pnl // Издержки уже учтены в realized_pnl
    }

    /// Количество скальп-сделок
    pub fn num_scalps(&self) -> usize {
        self.history.len()
    }

    /// Сброс истории
    pub fn reset(&mut self) {
        self.history.clear();
        self.previous_delta = 0.0;
        self.scalping_pnl = 0.0;
        self.total_costs = 0.0;
    }

    /// Статистика скальпинга
    pub fn statistics(&self) -> GammaScalpStats {
        if self.history.is_empty() {
            return GammaScalpStats::default();
        }

        let pnls: Vec<f64> = self.history.iter().map(|t| t.realized_pnl).collect();

        let profitable = pnls.iter().filter(|&&p| p > 0.0).count();
        let win_rate = profitable as f64 / pnls.len() as f64;

        let avg_pnl = pnls.iter().sum::<f64>() / pnls.len() as f64;
        let max_pnl = pnls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_pnl = pnls.iter().cloned().fold(f64::INFINITY, f64::min);

        // Std dev
        let variance = pnls.iter().map(|p| (p - avg_pnl).powi(2)).sum::<f64>() / pnls.len() as f64;
        let std_pnl = variance.sqrt();

        GammaScalpStats {
            num_scalps: self.history.len(),
            win_rate,
            total_pnl: self.scalping_pnl,
            avg_pnl,
            max_pnl,
            min_pnl,
            std_pnl,
            total_costs: self.total_costs,
        }
    }
}

/// Статистика гамма-скальпинга
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GammaScalpStats {
    /// Количество скальп-сделок
    pub num_scalps: usize,
    /// Процент прибыльных
    pub win_rate: f64,
    /// Общий P&L
    pub total_pnl: f64,
    /// Средний P&L
    pub avg_pnl: f64,
    /// Максимальный P&L
    pub max_pnl: f64,
    /// Минимальный P&L
    pub min_pnl: f64,
    /// Стандартное отклонение P&L
    pub std_pnl: f64,
    /// Общие издержки
    pub total_costs: f64,
}

impl std::fmt::Display for GammaScalpStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Gamma Scalp Stats:\n  Num scalps: {}\n  Win rate: {:.1}%\n  Total P&L: ${:.2}\n  Avg P&L: ${:.2}\n  Std P&L: ${:.2}\n  Costs: ${:.2}",
            self.num_scalps,
            self.win_rate * 100.0,
            self.total_pnl,
            self.avg_pnl,
            self.std_pnl,
            self.total_costs
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_scalp() {
        let mut scalper = GammaScalper::default_crypto();
        scalper.initialize(0.5);

        // Маленькое изменение → не скальпируем
        assert!(!scalper.should_scalp(0.55));

        // Большое изменение → скальпируем
        assert!(scalper.should_scalp(0.65));
    }

    #[test]
    fn test_execute_scalp() {
        let mut scalper = GammaScalper::default_crypto();
        scalper.initialize(0.5);

        // Дельта выросла → продаём
        let trade = scalper.execute_scalp(0.7, 42000.0);
        assert!(trade.is_some());

        let trade = trade.unwrap();
        assert!(trade.quantity < 0.0); // Продали
    }

    #[test]
    fn test_theoretical_pnl() {
        let gamma = 0.001;
        let spot = 42000.0;
        let daily_moves = vec![0.02, -0.015, 0.025, -0.01]; // Дневные движения в %

        let pnl = GammaScalper::theoretical_gamma_pnl(gamma, spot, &daily_moves);
        assert!(pnl > 0.0); // Гамма-скальпинг прибылен при движениях
    }

    #[test]
    fn test_statistics() {
        let mut scalper = GammaScalper::default_crypto();
        scalper.initialize(0.5);

        // Симулируем несколько скальпов
        scalper.execute_scalp(0.7, 42000.0);
        scalper.execute_scalp(0.5, 42500.0);
        scalper.execute_scalp(0.7, 42200.0);
        scalper.execute_scalp(0.4, 41800.0);

        let stats = scalper.statistics();
        assert_eq!(stats.num_scalps, 4);
    }
}
