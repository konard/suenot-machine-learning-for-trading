//! Time utility functions

use chrono::{DateTime, TimeZone, Utc};

/// Convert Unix timestamp (milliseconds) to DateTime
pub fn timestamp_to_datetime(timestamp_ms: i64) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(timestamp_ms).unwrap()
}

/// Convert DateTime to Unix timestamp (milliseconds)
pub fn datetime_to_timestamp(dt: DateTime<Utc>) -> i64 {
    dt.timestamp_millis()
}

/// Get current timestamp in milliseconds
pub fn now_timestamp() -> i64 {
    Utc::now().timestamp_millis()
}

/// Format timestamp as readable string
pub fn format_timestamp(timestamp_ms: i64) -> String {
    let dt = timestamp_to_datetime(timestamp_ms);
    dt.format("%Y-%m-%d %H:%M:%S UTC").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_conversion() {
        let ts = 1704067200000i64; // 2024-01-01 00:00:00 UTC
        let dt = timestamp_to_datetime(ts);
        let back = datetime_to_timestamp(dt);
        assert_eq!(ts, back);
    }
}
