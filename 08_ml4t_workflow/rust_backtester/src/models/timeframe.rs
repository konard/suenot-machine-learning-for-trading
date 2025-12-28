//! Timeframe definitions for candlestick data.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Candlestick timeframe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Timeframe {
    /// 1 minute
    M1,
    /// 3 minutes
    M3,
    /// 5 minutes
    M5,
    /// 15 minutes
    M15,
    /// 30 minutes
    M30,
    /// 1 hour
    H1,
    /// 2 hours
    H2,
    /// 4 hours
    H4,
    /// 6 hours
    H6,
    /// 12 hours
    H12,
    /// 1 day
    D1,
    /// 1 week
    W1,
    /// 1 month
    MN1,
}

impl Timeframe {
    /// Convert to Bybit API interval string.
    pub fn to_bybit_interval(&self) -> &'static str {
        match self {
            Timeframe::M1 => "1",
            Timeframe::M3 => "3",
            Timeframe::M5 => "5",
            Timeframe::M15 => "15",
            Timeframe::M30 => "30",
            Timeframe::H1 => "60",
            Timeframe::H2 => "120",
            Timeframe::H4 => "240",
            Timeframe::H6 => "360",
            Timeframe::H12 => "720",
            Timeframe::D1 => "D",
            Timeframe::W1 => "W",
            Timeframe::MN1 => "M",
        }
    }

    /// Get the duration in seconds.
    pub fn as_seconds(&self) -> i64 {
        match self {
            Timeframe::M1 => 60,
            Timeframe::M3 => 180,
            Timeframe::M5 => 300,
            Timeframe::M15 => 900,
            Timeframe::M30 => 1800,
            Timeframe::H1 => 3600,
            Timeframe::H2 => 7200,
            Timeframe::H4 => 14400,
            Timeframe::H6 => 21600,
            Timeframe::H12 => 43200,
            Timeframe::D1 => 86400,
            Timeframe::W1 => 604800,
            Timeframe::MN1 => 2592000, // 30 days approximation
        }
    }

    /// Get the duration in minutes.
    pub fn as_minutes(&self) -> i64 {
        self.as_seconds() / 60
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "1" | "1M" | "M1" => Some(Timeframe::M1),
            "3" | "3M" | "M3" => Some(Timeframe::M3),
            "5" | "5M" | "M5" => Some(Timeframe::M5),
            "15" | "15M" | "M15" => Some(Timeframe::M15),
            "30" | "30M" | "M30" => Some(Timeframe::M30),
            "60" | "1H" | "H1" => Some(Timeframe::H1),
            "120" | "2H" | "H2" => Some(Timeframe::H2),
            "240" | "4H" | "H4" => Some(Timeframe::H4),
            "360" | "6H" | "H6" => Some(Timeframe::H6),
            "720" | "12H" | "H12" => Some(Timeframe::H12),
            "D" | "1D" | "D1" => Some(Timeframe::D1),
            "W" | "1W" | "W1" => Some(Timeframe::W1),
            "M" | "1MN" | "MN1" => Some(Timeframe::MN1),
            _ => None,
        }
    }
}

impl fmt::Display for Timeframe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Timeframe::M1 => "1m",
            Timeframe::M3 => "3m",
            Timeframe::M5 => "5m",
            Timeframe::M15 => "15m",
            Timeframe::M30 => "30m",
            Timeframe::H1 => "1h",
            Timeframe::H2 => "2h",
            Timeframe::H4 => "4h",
            Timeframe::H6 => "6h",
            Timeframe::H12 => "12h",
            Timeframe::D1 => "1d",
            Timeframe::W1 => "1w",
            Timeframe::MN1 => "1M",
        };
        write!(f, "{}", s)
    }
}

impl Default for Timeframe {
    fn default() -> Self {
        Timeframe::H1
    }
}
