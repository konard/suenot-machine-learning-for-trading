//! Time Utilities

use chrono::{DateTime, TimeZone, Utc};

/// Convert timestamp (ms) to datetime
pub fn timestamp_to_datetime(timestamp_ms: u64) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(timestamp_ms as i64).unwrap()
}

/// Convert datetime to timestamp (ms)
pub fn datetime_to_timestamp(dt: DateTime<Utc>) -> u64 {
    dt.timestamp_millis() as u64
}
