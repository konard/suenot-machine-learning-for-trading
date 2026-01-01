//! Bybit API client module
//!
//! This module provides a client for interacting with the Bybit V5 API.

pub mod client;
pub mod error;
pub mod types;

pub use client::BybitClient;
pub use error::ApiError;
pub use types::*;

/// Bybit API base URL
pub const BYBIT_API_URL: &str = "https://api.bybit.com";

/// Bybit Testnet API URL
pub const BYBIT_TESTNET_URL: &str = "https://api-testnet.bybit.com";

/// Trading category for API requests
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    /// Spot trading
    Spot,
    /// Linear perpetual (USDT-margined)
    Linear,
    /// Inverse perpetual (coin-margined)
    Inverse,
}

impl Category {
    /// Get the API string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Category::Spot => "spot",
            Category::Linear => "linear",
            Category::Inverse => "inverse",
        }
    }
}

/// Kline (candlestick) interval
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interval {
    /// 1 minute
    Min1,
    /// 3 minutes
    Min3,
    /// 5 minutes
    Min5,
    /// 15 minutes
    Min15,
    /// 30 minutes
    Min30,
    /// 1 hour
    Hour1,
    /// 2 hours
    Hour2,
    /// 4 hours
    Hour4,
    /// 6 hours
    Hour6,
    /// 12 hours
    Hour12,
    /// 1 day
    Day1,
    /// 1 week
    Week1,
    /// 1 month
    Month1,
}

impl Interval {
    /// Get the API string representation
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

    /// Get interval in minutes (for time calculations)
    pub fn minutes(&self) -> i64 {
        match self {
            Interval::Min1 => 1,
            Interval::Min3 => 3,
            Interval::Min5 => 5,
            Interval::Min15 => 15,
            Interval::Min30 => 30,
            Interval::Hour1 => 60,
            Interval::Hour2 => 120,
            Interval::Hour4 => 240,
            Interval::Hour6 => 360,
            Interval::Hour12 => 720,
            Interval::Day1 => 1440,
            Interval::Week1 => 10080,
            Interval::Month1 => 43200,
        }
    }
}
