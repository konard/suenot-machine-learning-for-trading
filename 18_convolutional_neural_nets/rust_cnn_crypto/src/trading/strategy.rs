//! Торговые стратегии

use super::{Position, Signal, SignalDirection};
use serde::{Deserialize, Serialize};

/// Конфигурация стратегии
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Минимальная сила сигнала для входа
    pub min_signal_strength: f64,
    /// Порог вероятности для классификации
    pub probability_threshold: f64,
    /// Размер позиции (доля от капитала)
    pub position_size: f64,
    /// Стоп-лосс в процентах
    pub stop_loss_pct: f64,
    /// Тейк-профит в процентах
    pub take_profit_pct: f64,
    /// Максимальное количество одновременных позиций
    pub max_positions: usize,
    /// Комиссия (в долях, например 0.001 = 0.1%)
    pub commission_rate: f64,
    /// Минимальное время между сделками (в мс)
    pub min_trade_interval: i64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            min_signal_strength: 0.6,
            probability_threshold: 0.5,
            position_size: 0.1,
            stop_loss_pct: 2.0,
            take_profit_pct: 4.0,
            max_positions: 1,
            commission_rate: 0.001,
            min_trade_interval: 900000, // 15 минут
        }
    }
}

/// Торговая стратегия на основе CNN сигналов
pub struct Strategy {
    config: StrategyConfig,
    positions: Vec<Position>,
    closed_positions: Vec<Position>,
    next_position_id: u64,
    last_trade_time: i64,
    capital: f64,
}

impl Strategy {
    /// Создание новой стратегии
    pub fn new(config: StrategyConfig, initial_capital: f64) -> Self {
        Self {
            config,
            positions: Vec::new(),
            closed_positions: Vec::new(),
            next_position_id: 1,
            last_trade_time: 0,
            capital: initial_capital,
        }
    }

    /// Обработка нового сигнала
    pub fn process_signal(&mut self, signal: &Signal, symbol: &str) -> Option<Position> {
        // Проверяем условия для новой сделки
        if !self.can_open_position(signal) {
            return None;
        }

        // Закрываем противоположные позиции
        self.close_opposite_positions(signal);

        // Открываем новую позицию
        let position_value = self.capital * self.config.position_size;
        let size = position_value / signal.price;

        let mut position = Position::new(
            self.next_position_id,
            symbol,
            signal.direction,
            signal.price,
            size,
            signal.timestamp,
        )
        .with_risk_management(self.config.stop_loss_pct, self.config.take_profit_pct)
        .with_commission(self.config.commission_rate);

        self.next_position_id += 1;
        self.last_trade_time = signal.timestamp;
        self.positions.push(position.clone());

        Some(position)
    }

    /// Обновление позиций на основе текущей цены
    pub fn update(&mut self, current_price: f64, current_time: i64) {
        let commission_rate = self.config.commission_rate;

        let mut positions_to_close = Vec::new();

        for (idx, pos) in self.positions.iter().enumerate() {
            if pos.check_stop_loss(current_price) {
                positions_to_close.push((idx, current_price, true));
            } else if pos.check_take_profit(current_price) {
                positions_to_close.push((idx, current_price, false));
            }
        }

        // Закрываем позиции (в обратном порядке для сохранения индексов)
        for (idx, price, is_stop_loss) in positions_to_close.into_iter().rev() {
            let mut pos = self.positions.remove(idx);
            if is_stop_loss {
                pos.liquidate(price, current_time, commission_rate);
            } else {
                pos.close(price, current_time, commission_rate);
            }
            if let Some(pnl) = pos.realized_pnl() {
                self.capital += pnl;
            }
            self.closed_positions.push(pos);
        }
    }

    /// Закрытие всех позиций
    pub fn close_all(&mut self, current_price: f64, current_time: i64) {
        let commission_rate = self.config.commission_rate;

        while let Some(mut pos) = self.positions.pop() {
            pos.close(current_price, current_time, commission_rate);
            if let Some(pnl) = pos.realized_pnl() {
                self.capital += pnl;
            }
            self.closed_positions.push(pos);
        }
    }

    /// Проверка возможности открытия позиции
    fn can_open_position(&self, signal: &Signal) -> bool {
        // Проверка направления
        if signal.direction == SignalDirection::Neutral {
            return false;
        }

        // Проверка силы сигнала
        if signal.strength < self.config.min_signal_strength {
            return false;
        }

        // Проверка максимального количества позиций
        if self.positions.len() >= self.config.max_positions {
            return false;
        }

        // Проверка минимального интервала
        if signal.timestamp - self.last_trade_time < self.config.min_trade_interval {
            return false;
        }

        // Проверка наличия такой же позиции
        for pos in &self.positions {
            if pos.direction == signal.direction {
                return false;
            }
        }

        true
    }

    /// Закрытие противоположных позиций
    fn close_opposite_positions(&mut self, signal: &Signal) {
        let opposite = match signal.direction {
            SignalDirection::Long => SignalDirection::Short,
            SignalDirection::Short => SignalDirection::Long,
            SignalDirection::Neutral => return,
        };

        let commission_rate = self.config.commission_rate;
        let mut to_close = Vec::new();

        for (idx, pos) in self.positions.iter().enumerate() {
            if pos.direction == opposite {
                to_close.push(idx);
            }
        }

        for idx in to_close.into_iter().rev() {
            let mut pos = self.positions.remove(idx);
            pos.close(signal.price, signal.timestamp, commission_rate);
            if let Some(pnl) = pos.realized_pnl() {
                self.capital += pnl;
            }
            self.closed_positions.push(pos);
        }
    }

    /// Текущий капитал
    pub fn capital(&self) -> f64 {
        self.capital
    }

    /// Текущий капитал с учётом нереализованной прибыли
    pub fn equity(&self, current_price: f64) -> f64 {
        let unrealized_pnl: f64 = self
            .positions
            .iter()
            .map(|p| p.unrealized_pnl(current_price))
            .sum();
        self.capital + unrealized_pnl
    }

    /// Открытые позиции
    pub fn open_positions(&self) -> &[Position] {
        &self.positions
    }

    /// Закрытые позиции
    pub fn closed_positions(&self) -> &[Position] {
        &self.closed_positions
    }

    /// Все позиции
    pub fn all_positions(&self) -> Vec<&Position> {
        self.positions
            .iter()
            .chain(self.closed_positions.iter())
            .collect()
    }

    /// Статистика стратегии
    pub fn statistics(&self, initial_capital: f64) -> StrategyStatistics {
        let total_trades = self.closed_positions.len();
        let winning_trades = self
            .closed_positions
            .iter()
            .filter(|p| p.realized_pnl().map(|pnl| pnl > 0.0).unwrap_or(false))
            .count();
        let losing_trades = self
            .closed_positions
            .iter()
            .filter(|p| p.realized_pnl().map(|pnl| pnl < 0.0).unwrap_or(false))
            .count();

        let total_pnl: f64 = self
            .closed_positions
            .iter()
            .filter_map(|p| p.realized_pnl())
            .sum();

        let gross_profit: f64 = self
            .closed_positions
            .iter()
            .filter_map(|p| p.realized_pnl())
            .filter(|&pnl| pnl > 0.0)
            .sum();

        let gross_loss: f64 = self
            .closed_positions
            .iter()
            .filter_map(|p| p.realized_pnl())
            .filter(|&pnl| pnl < 0.0)
            .map(|pnl| pnl.abs())
            .sum();

        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let return_pct = (self.capital - initial_capital) / initial_capital * 100.0;

        StrategyStatistics {
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            total_pnl,
            gross_profit,
            gross_loss,
            profit_factor,
            return_pct,
            final_capital: self.capital,
        }
    }
}

/// Статистика стратегии
#[derive(Debug, Clone)]
pub struct StrategyStatistics {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub gross_profit: f64,
    pub gross_loss: f64,
    pub profit_factor: f64,
    pub return_pct: f64,
    pub final_capital: f64,
}

impl std::fmt::Display for StrategyStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Strategy Statistics ===")?;
        writeln!(f, "Total Trades: {}", self.total_trades)?;
        writeln!(
            f,
            "Winning/Losing: {}/{}",
            self.winning_trades, self.losing_trades
        )?;
        writeln!(f, "Win Rate: {:.2}%", self.win_rate * 100.0)?;
        writeln!(f, "Total PnL: ${:.2}", self.total_pnl)?;
        writeln!(f, "Profit Factor: {:.2}", self.profit_factor)?;
        writeln!(f, "Return: {:.2}%", self.return_pct)?;
        writeln!(f, "Final Capital: ${:.2}", self.final_capital)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_creation() {
        let strategy = Strategy::new(StrategyConfig::default(), 10000.0);
        assert_eq!(strategy.capital(), 10000.0);
        assert!(strategy.open_positions().is_empty());
    }

    #[test]
    fn test_signal_processing() {
        let mut strategy = Strategy::new(StrategyConfig::default(), 10000.0);

        let signal = Signal::new(1000, [0.1, 0.1, 0.8], 50000.0, 0.5);
        let position = strategy.process_signal(&signal, "BTCUSDT");

        assert!(position.is_some());
        assert_eq!(strategy.open_positions().len(), 1);
    }
}
