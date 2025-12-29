//! Стакан заявок (Order Book)

use serde::{Deserialize, Serialize};

/// Уровень стакана заявок
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Цена
    pub price: f64,
    /// Количество
    pub quantity: f64,
}

impl OrderBookLevel {
    pub fn new(price: f64, quantity: f64) -> Self {
        Self { price, quantity }
    }

    /// Общая стоимость на уровне
    pub fn total(&self) -> f64 {
        self.price * self.quantity
    }
}

/// Стакан заявок
#[derive(Debug, Clone, Default)]
pub struct OrderBook {
    /// Символ
    pub symbol: String,
    /// Заявки на покупку (bids) - отсортированы по убыванию цены
    pub bids: Vec<OrderBookLevel>,
    /// Заявки на продажу (asks) - отсортированы по возрастанию цены
    pub asks: Vec<OrderBookLevel>,
    /// Время обновления
    pub timestamp: u64,
}

impl OrderBook {
    /// Создать новый стакан
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            bids: Vec::new(),
            asks: Vec::new(),
            timestamp: 0,
        }
    }

    /// Лучшая цена покупки (bid)
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Лучшая цена продажи (ask)
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Спред (разница между лучшей продажей и покупкой)
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Спред в процентах
    pub fn spread_percent(&self) -> Option<f64> {
        match (self.best_bid(), self.spread()) {
            (Some(bid), Some(spread)) if bid > 0.0 => Some((spread / bid) * 100.0),
            _ => None,
        }
    }

    /// Средняя цена (mid price)
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Взвешенная средняя цена
    pub fn weighted_mid_price(&self) -> Option<f64> {
        let (bid, ask) = (self.bids.first()?, self.asks.first()?);
        let total_qty = bid.quantity + ask.quantity;
        if total_qty > 0.0 {
            Some((bid.price * ask.quantity + ask.price * bid.quantity) / total_qty)
        } else {
            None
        }
    }

    /// Суммарный объём на стороне покупки до указанного уровня
    pub fn bid_depth(&self, levels: usize) -> f64 {
        self.bids
            .iter()
            .take(levels)
            .map(|l| l.quantity)
            .sum()
    }

    /// Суммарный объём на стороне продажи до указанного уровня
    pub fn ask_depth(&self, levels: usize) -> f64 {
        self.asks
            .iter()
            .take(levels)
            .map(|l| l.quantity)
            .sum()
    }

    /// Дисбаланс стакана (положительный = больше покупателей)
    pub fn imbalance(&self, levels: usize) -> f64 {
        let bid_depth = self.bid_depth(levels);
        let ask_depth = self.ask_depth(levels);
        let total = bid_depth + ask_depth;

        if total > 0.0 {
            (bid_depth - ask_depth) / total
        } else {
            0.0
        }
    }

    /// VWAP для стороны покупки
    pub fn bid_vwap(&self, levels: usize) -> Option<f64> {
        let levels: Vec<_> = self.bids.iter().take(levels).collect();
        if levels.is_empty() {
            return None;
        }

        let total_qty: f64 = levels.iter().map(|l| l.quantity).sum();
        if total_qty == 0.0 {
            return None;
        }

        let weighted_sum: f64 = levels.iter().map(|l| l.price * l.quantity).sum();
        Some(weighted_sum / total_qty)
    }

    /// VWAP для стороны продажи
    pub fn ask_vwap(&self, levels: usize) -> Option<f64> {
        let levels: Vec<_> = self.asks.iter().take(levels).collect();
        if levels.is_empty() {
            return None;
        }

        let total_qty: f64 = levels.iter().map(|l| l.quantity).sum();
        if total_qty == 0.0 {
            return None;
        }

        let weighted_sum: f64 = levels.iter().map(|l| l.price * l.quantity).sum();
        Some(weighted_sum / total_qty)
    }

    /// Обновить стакан из данных API
    pub fn update_from_raw(&mut self, bids: Vec<Vec<String>>, asks: Vec<Vec<String>>, timestamp: u64) {
        self.bids = bids
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 2 {
                    Some(OrderBookLevel {
                        price: row[0].parse().ok()?,
                        quantity: row[1].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        self.asks = asks
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 2 {
                    Some(OrderBookLevel {
                        price: row[0].parse().ok()?,
                        quantity: row[1].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        self.timestamp = timestamp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_orderbook() -> OrderBook {
        let mut ob = OrderBook::new("BTCUSDT");
        ob.bids = vec![
            OrderBookLevel::new(42000.0, 1.0),
            OrderBookLevel::new(41990.0, 2.0),
            OrderBookLevel::new(41980.0, 1.5),
        ];
        ob.asks = vec![
            OrderBookLevel::new(42010.0, 1.0),
            OrderBookLevel::new(42020.0, 2.0),
            OrderBookLevel::new(42030.0, 1.5),
        ];
        ob
    }

    #[test]
    fn test_best_prices() {
        let ob = create_test_orderbook();
        assert_eq!(ob.best_bid(), Some(42000.0));
        assert_eq!(ob.best_ask(), Some(42010.0));
    }

    #[test]
    fn test_spread() {
        let ob = create_test_orderbook();
        assert_eq!(ob.spread(), Some(10.0));
    }

    #[test]
    fn test_mid_price() {
        let ob = create_test_orderbook();
        assert_eq!(ob.mid_price(), Some(42005.0));
    }

    #[test]
    fn test_depth() {
        let ob = create_test_orderbook();
        assert_eq!(ob.bid_depth(2), 3.0);
        assert_eq!(ob.ask_depth(3), 4.5);
    }
}
