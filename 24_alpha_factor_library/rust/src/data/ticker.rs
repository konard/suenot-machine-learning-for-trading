//! Структура данных для тикера (текущей цены)

use serde::{Deserialize, Serialize};

/// Тикер - текущая информация о торговой паре
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Символ торговой пары
    pub symbol: String,
    /// Последняя цена
    pub last_price: f64,
    /// Лучшая цена покупки (bid)
    pub bid_price: f64,
    /// Объём на лучшей цене покупки
    pub bid_size: f64,
    /// Лучшая цена продажи (ask)
    pub ask_price: f64,
    /// Объём на лучшей цене продажи
    pub ask_size: f64,
    /// Изменение цены за 24 часа
    pub price_change_24h: f64,
    /// Изменение цены за 24 часа в процентах
    pub price_change_percent_24h: f64,
    /// Максимальная цена за 24 часа
    pub high_24h: f64,
    /// Минимальная цена за 24 часа
    pub low_24h: f64,
    /// Объём за 24 часа в базовой валюте
    pub volume_24h: f64,
    /// Объём за 24 часа в котируемой валюте
    pub turnover_24h: f64,
    /// Временная метка
    pub timestamp: i64,
}

impl Ticker {
    /// Спред (разница между ask и bid)
    pub fn spread(&self) -> f64 {
        self.ask_price - self.bid_price
    }

    /// Спред в процентах от средней цены
    pub fn spread_percent(&self) -> f64 {
        if self.mid_price() == 0.0 {
            return 0.0;
        }
        (self.spread() / self.mid_price()) * 100.0
    }

    /// Средняя цена (mid price)
    pub fn mid_price(&self) -> f64 {
        (self.bid_price + self.ask_price) / 2.0
    }

    /// Дневной размах в процентах
    pub fn daily_range_percent(&self) -> f64 {
        if self.low_24h == 0.0 {
            return 0.0;
        }
        ((self.high_24h - self.low_24h) / self.low_24h) * 100.0
    }

    /// Дисбаланс bid/ask (положительный = больше покупателей)
    pub fn order_imbalance(&self) -> f64 {
        let total = self.bid_size + self.ask_size;
        if total == 0.0 {
            return 0.0;
        }
        (self.bid_size - self.ask_size) / total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ticker() -> Ticker {
        Ticker {
            symbol: "BTCUSDT".to_string(),
            last_price: 50000.0,
            bid_price: 49990.0,
            bid_size: 10.0,
            ask_price: 50010.0,
            ask_size: 8.0,
            price_change_24h: 1000.0,
            price_change_percent_24h: 2.04,
            high_24h: 51000.0,
            low_24h: 48000.0,
            volume_24h: 10000.0,
            turnover_24h: 500000000.0,
            timestamp: 1700000000000,
        }
    }

    #[test]
    fn test_spread() {
        let ticker = sample_ticker();
        assert_eq!(ticker.spread(), 20.0);
    }

    #[test]
    fn test_mid_price() {
        let ticker = sample_ticker();
        assert_eq!(ticker.mid_price(), 50000.0);
    }

    #[test]
    fn test_order_imbalance() {
        let ticker = sample_ticker();
        // (10 - 8) / (10 + 8) = 2/18 ≈ 0.111
        assert!((ticker.order_imbalance() - 0.111).abs() < 0.01);
    }
}
