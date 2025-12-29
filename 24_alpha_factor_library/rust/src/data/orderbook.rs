//! Структура данных для стакана заявок (Order Book)

use serde::{Deserialize, Serialize};

/// Уровень в стакане заявок
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Цена
    pub price: f64,
    /// Объём на этой цене
    pub size: f64,
}

impl OrderBookLevel {
    pub fn new(price: f64, size: f64) -> Self {
        Self { price, size }
    }

    /// Общая стоимость на уровне
    pub fn value(&self) -> f64 {
        self.price * self.size
    }
}

/// Стакан заявок (Order Book)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Символ торговой пары
    pub symbol: String,
    /// Заявки на покупку (bids), отсортированы по убыванию цены
    pub bids: Vec<OrderBookLevel>,
    /// Заявки на продажу (asks), отсортированы по возрастанию цены
    pub asks: Vec<OrderBookLevel>,
    /// Временная метка
    pub timestamp: i64,
}

impl OrderBook {
    /// Создать новый стакан заявок
    pub fn new(symbol: impl Into<String>, bids: Vec<OrderBookLevel>, asks: Vec<OrderBookLevel>, timestamp: i64) -> Self {
        Self {
            symbol: symbol.into(),
            bids,
            asks,
            timestamp,
        }
    }

    /// Лучшая цена покупки (best bid)
    pub fn best_bid(&self) -> Option<&OrderBookLevel> {
        self.bids.first()
    }

    /// Лучшая цена продажи (best ask)
    pub fn best_ask(&self) -> Option<&OrderBookLevel> {
        self.asks.first()
    }

    /// Средняя цена (mid price)
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid.price + ask.price) / 2.0),
            _ => None,
        }
    }

    /// Спред в абсолютном выражении
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask.price - bid.price),
            _ => None,
        }
    }

    /// Спред в процентах
    pub fn spread_percent(&self) -> Option<f64> {
        match (self.mid_price(), self.spread()) {
            (Some(mid), Some(spread)) if mid > 0.0 => Some((spread / mid) * 100.0),
            _ => None,
        }
    }

    /// Общий объём на стороне покупки (до указанной глубины)
    pub fn total_bid_size(&self, depth: usize) -> f64 {
        self.bids.iter().take(depth).map(|l| l.size).sum()
    }

    /// Общий объём на стороне продажи (до указанной глубины)
    pub fn total_ask_size(&self, depth: usize) -> f64 {
        self.asks.iter().take(depth).map(|l| l.size).sum()
    }

    /// Общая стоимость на стороне покупки (до указанной глубины)
    pub fn total_bid_value(&self, depth: usize) -> f64 {
        self.bids.iter().take(depth).map(|l| l.value()).sum()
    }

    /// Общая стоимость на стороне продажи (до указанной глубины)
    pub fn total_ask_value(&self, depth: usize) -> f64 {
        self.asks.iter().take(depth).map(|l| l.value()).sum()
    }

    /// Дисбаланс стакана (положительный = больше покупателей)
    /// Формула: (bid_size - ask_size) / (bid_size + ask_size)
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_size = self.total_bid_size(depth);
        let ask_size = self.total_ask_size(depth);
        let total = bid_size + ask_size;

        if total == 0.0 {
            return 0.0;
        }

        (bid_size - ask_size) / total
    }

    /// Взвешенная средняя цена покупки (VWAP для bids)
    pub fn vwap_bid(&self, depth: usize) -> Option<f64> {
        let levels: Vec<_> = self.bids.iter().take(depth).collect();
        if levels.is_empty() {
            return None;
        }

        let total_value: f64 = levels.iter().map(|l| l.value()).sum();
        let total_size: f64 = levels.iter().map(|l| l.size).sum();

        if total_size == 0.0 {
            return None;
        }

        Some(total_value / total_size)
    }

    /// Взвешенная средняя цена продажи (VWAP для asks)
    pub fn vwap_ask(&self, depth: usize) -> Option<f64> {
        let levels: Vec<_> = self.asks.iter().take(depth).collect();
        if levels.is_empty() {
            return None;
        }

        let total_value: f64 = levels.iter().map(|l| l.value()).sum();
        let total_size: f64 = levels.iter().map(|l| l.size).sum();

        if total_size == 0.0 {
            return None;
        }

        Some(total_value / total_size)
    }

    /// Давление на покупку/продажу (Book Pressure)
    /// Сравнивает объёмы на близких уровнях к mid price
    pub fn book_pressure(&self, depth: usize) -> f64 {
        let bid_value = self.total_bid_value(depth);
        let ask_value = self.total_ask_value(depth);
        let total = bid_value + ask_value;

        if total == 0.0 {
            return 0.0;
        }

        (bid_value - ask_value) / total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_orderbook() -> OrderBook {
        let bids = vec![
            OrderBookLevel::new(100.0, 10.0),
            OrderBookLevel::new(99.0, 20.0),
            OrderBookLevel::new(98.0, 15.0),
        ];
        let asks = vec![
            OrderBookLevel::new(101.0, 8.0),
            OrderBookLevel::new(102.0, 12.0),
            OrderBookLevel::new(103.0, 18.0),
        ];
        OrderBook::new("BTCUSDT", bids, asks, 1700000000000)
    }

    #[test]
    fn test_best_prices() {
        let ob = sample_orderbook();
        assert_eq!(ob.best_bid().unwrap().price, 100.0);
        assert_eq!(ob.best_ask().unwrap().price, 101.0);
    }

    #[test]
    fn test_mid_price() {
        let ob = sample_orderbook();
        assert_eq!(ob.mid_price().unwrap(), 100.5);
    }

    #[test]
    fn test_spread() {
        let ob = sample_orderbook();
        assert_eq!(ob.spread().unwrap(), 1.0);
    }

    #[test]
    fn test_imbalance() {
        let ob = sample_orderbook();
        // depth=1: (10 - 8) / (10 + 8) = 2/18 ≈ 0.111
        assert!((ob.imbalance(1) - 0.111).abs() < 0.01);
    }

    #[test]
    fn test_total_sizes() {
        let ob = sample_orderbook();
        assert_eq!(ob.total_bid_size(3), 45.0); // 10 + 20 + 15
        assert_eq!(ob.total_ask_size(3), 38.0); // 8 + 12 + 18
    }
}
