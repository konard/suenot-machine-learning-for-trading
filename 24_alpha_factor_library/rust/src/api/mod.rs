//! Клиент API биржи Bybit
//!
//! Модуль предоставляет клиент для работы с публичным API Bybit:
//! - Получение OHLCV свечей (klines)
//! - Получение тикеров
//! - Получение стакана заявок
//!
//! ## Пример использования
//!
//! ```rust,no_run
//! use alpha_factors::BybitClient;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "1h", 100).await?;
//!     println!("Получено {} свечей", klines.len());
//!     Ok(())
//! }
//! ```

mod client;
mod error;
mod response;

pub use client::BybitClient;
pub use error::ApiError;
pub use response::*;

/// Базовый URL API Bybit (mainnet)
pub const BYBIT_API_URL: &str = "https://api.bybit.com";

/// Базовый URL API Bybit (testnet)
pub const BYBIT_TESTNET_URL: &str = "https://api-testnet.bybit.com";

/// Категории продуктов Bybit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    /// Спотовый рынок
    Spot,
    /// Бессрочные контракты (линейные, USDT)
    Linear,
    /// Бессрочные контракты (инверсные)
    Inverse,
    /// Опционы
    Option,
}

impl Category {
    pub fn as_str(&self) -> &'static str {
        match self {
            Category::Spot => "spot",
            Category::Linear => "linear",
            Category::Inverse => "inverse",
            Category::Option => "option",
        }
    }
}

impl std::fmt::Display for Category {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Интервалы свечей
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interval {
    /// 1 минута
    Min1,
    /// 3 минуты
    Min3,
    /// 5 минут
    Min5,
    /// 15 минут
    Min15,
    /// 30 минут
    Min30,
    /// 1 час
    Hour1,
    /// 2 часа
    Hour2,
    /// 4 часа
    Hour4,
    /// 6 часов
    Hour6,
    /// 12 часов
    Hour12,
    /// 1 день
    Day1,
    /// 1 неделя
    Week1,
    /// 1 месяц
    Month1,
}

impl Interval {
    pub fn as_str(&self) -> &'static str {
        match self {
            Interval::Min1 => "1",
            Interval::Min3 => "3",
            Interval::Min5 => "5",
            Interval::Min15 => "15",
            Interval::Min30 => "30",
            Interval::Hour1 => "60",
            Interval::Hour2 => "120",
            Interval::Hour4 => "240",
            Interval::Hour6 => "360",
            Interval::Hour12 => "720",
            Interval::Day1 => "D",
            Interval::Week1 => "W",
            Interval::Month1 => "M",
        }
    }

    /// Длительность интервала в миллисекундах
    pub fn duration_ms(&self) -> i64 {
        match self {
            Interval::Min1 => 60_000,
            Interval::Min3 => 180_000,
            Interval::Min5 => 300_000,
            Interval::Min15 => 900_000,
            Interval::Min30 => 1_800_000,
            Interval::Hour1 => 3_600_000,
            Interval::Hour2 => 7_200_000,
            Interval::Hour4 => 14_400_000,
            Interval::Hour6 => 21_600_000,
            Interval::Hour12 => 43_200_000,
            Interval::Day1 => 86_400_000,
            Interval::Week1 => 604_800_000,
            Interval::Month1 => 2_592_000_000, // ~30 days
        }
    }
}

impl std::fmt::Display for Interval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
