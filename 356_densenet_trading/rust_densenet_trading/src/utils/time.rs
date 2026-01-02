//! Time Utilities
//!
//! Time and date handling utilities.

use chrono::{DateTime, TimeZone, Utc, NaiveDateTime};

/// Convert Unix timestamp (milliseconds) to DateTime
pub fn timestamp_to_datetime(timestamp_ms: u64) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(timestamp_ms as i64)
        .single()
        .unwrap_or_else(Utc::now)
}

/// Convert DateTime to Unix timestamp (milliseconds)
pub fn datetime_to_timestamp(dt: DateTime<Utc>) -> u64 {
    dt.timestamp_millis() as u64
}

/// Get current timestamp in milliseconds
pub fn now_timestamp() -> u64 {
    Utc::now().timestamp_millis() as u64
}

/// Parse datetime string to timestamp
pub fn parse_datetime(s: &str) -> Option<u64> {
    // Try common formats
    let formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ];

    for fmt in formats {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Some(dt.and_utc().timestamp_millis() as u64);
        }
    }

    None
}

/// Format timestamp as string
pub fn format_timestamp(timestamp_ms: u64, format: &str) -> String {
    let dt = timestamp_to_datetime(timestamp_ms);
    dt.format(format).to_string()
}

/// Format timestamp as ISO 8601
pub fn format_iso(timestamp_ms: u64) -> String {
    format_timestamp(timestamp_ms, "%Y-%m-%dT%H:%M:%SZ")
}

/// Format duration in human-readable format
pub fn format_duration(milliseconds: u64) -> String {
    let seconds = milliseconds / 1000;
    let minutes = seconds / 60;
    let hours = minutes / 60;
    let days = hours / 24;

    if days > 0 {
        format!("{}d {}h", days, hours % 24)
    } else if hours > 0 {
        format!("{}h {}m", hours, minutes % 60)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds % 60)
    } else {
        format!("{}s", seconds)
    }
}

/// Convert interval string to milliseconds
pub fn interval_to_ms(interval: &str) -> Option<u64> {
    let (num_str, unit) = interval.split_at(interval.len() - 1);

    let num: u64 = if num_str.is_empty() {
        1
    } else {
        num_str.parse().ok()?
    };

    let ms = match unit.to_lowercase().as_str() {
        "m" => num * 60 * 1000,
        "h" => num * 60 * 60 * 1000,
        "d" => num * 24 * 60 * 60 * 1000,
        "w" => num * 7 * 24 * 60 * 60 * 1000,
        _ => return None,
    };

    Some(ms)
}

/// Calculate time ranges for backtesting
pub struct TimeRange {
    pub start: u64,
    pub end: u64,
}

impl TimeRange {
    pub fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }

    /// Create range for last N days
    pub fn last_days(days: u64) -> Self {
        let end = now_timestamp();
        let start = end - days * 24 * 60 * 60 * 1000;
        Self { start, end }
    }

    /// Create range for last N hours
    pub fn last_hours(hours: u64) -> Self {
        let end = now_timestamp();
        let start = end - hours * 60 * 60 * 1000;
        Self { start, end }
    }

    /// Duration in milliseconds
    pub fn duration(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    /// Duration in days
    pub fn days(&self) -> f64 {
        self.duration() as f64 / (24.0 * 60.0 * 60.0 * 1000.0)
    }

    /// Split into training and testing ranges
    pub fn train_test_split(&self, train_ratio: f64) -> (TimeRange, TimeRange) {
        let split_point = self.start + (self.duration() as f64 * train_ratio) as u64;

        (
            TimeRange::new(self.start, split_point),
            TimeRange::new(split_point, self.end),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_conversion() {
        let ts: u64 = 1704067200000; // 2024-01-01 00:00:00 UTC
        let dt = timestamp_to_datetime(ts);
        let back = datetime_to_timestamp(dt);

        assert_eq!(ts, back);
    }

    #[test]
    fn test_format_iso() {
        let ts: u64 = 1704067200000;
        let formatted = format_iso(ts);

        assert!(formatted.contains("2024-01-01"));
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(5000), "5s");
        assert_eq!(format_duration(65000), "1m 5s");
        assert_eq!(format_duration(3665000), "1h 1m");
        assert_eq!(format_duration(90000000), "1d 1h");
    }

    #[test]
    fn test_interval_to_ms() {
        assert_eq!(interval_to_ms("1m"), Some(60000));
        assert_eq!(interval_to_ms("5m"), Some(300000));
        assert_eq!(interval_to_ms("1h"), Some(3600000));
        assert_eq!(interval_to_ms("1d"), Some(86400000));
    }

    #[test]
    fn test_time_range() {
        let range = TimeRange::new(0, 86400000);
        assert!((range.days() - 1.0).abs() < 0.001);

        let (train, test) = range.train_test_split(0.8);
        assert!(train.end == test.start);
    }
}
