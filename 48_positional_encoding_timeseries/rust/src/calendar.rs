//! Calendar and Market Session Encoding
//!
//! This module provides temporal encodings specific to financial markets,
//! including calendar features and trading session information.

use chrono::{DateTime, Datelike, NaiveDateTime, Timelike, Utc, Weekday};
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Calendar encoding for financial time series
///
/// Encodes temporal features that are important for market patterns:
/// - Hour of day (24h cycle)
/// - Day of week (weekly cycle)
/// - Day of month (monthly cycle)
/// - Month of year (yearly cycle)
/// - Quarter
/// - Is weekend
/// - Is month start/end
#[derive(Debug, Clone)]
pub struct CalendarEncoding {
    /// Whether to use cyclical (sin/cos) encoding
    cyclical: bool,
    /// Output dimension
    d_model: usize,
}

impl CalendarEncoding {
    /// Create a new calendar encoding with cyclical features
    pub fn new() -> Self {
        Self {
            cyclical: true,
            d_model: 20, // 2 per cycle (sin+cos) * 4 cycles + 4 binary features
        }
    }

    /// Create with linear (one-hot style) encoding
    pub fn linear() -> Self {
        Self {
            cyclical: false,
            d_model: 24 + 7 + 31 + 12 + 4, // hour + dow + dom + month + binary
        }
    }

    /// Get encoding dimension
    pub fn dim(&self) -> usize {
        self.d_model
    }

    /// Encode a single timestamp
    pub fn encode_timestamp(&self, timestamp: i64) -> Array1<f64> {
        let dt = DateTime::from_timestamp(timestamp, 0)
            .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap());

        if self.cyclical {
            self.encode_cyclical(&dt)
        } else {
            self.encode_linear(&dt)
        }
    }

    /// Encode multiple timestamps
    pub fn encode_timestamps(&self, timestamps: &[i64]) -> Array2<f64> {
        let mut result = Array2::zeros((timestamps.len(), self.d_model));
        for (i, &ts) in timestamps.iter().enumerate() {
            result.row_mut(i).assign(&self.encode_timestamp(ts));
        }
        result
    }

    fn encode_cyclical(&self, dt: &DateTime<Utc>) -> Array1<f64> {
        let mut features = Array1::zeros(self.d_model);
        let mut idx = 0;

        // Hour of day (24h cycle)
        let hour = dt.hour() as f64;
        features[idx] = (2.0 * PI * hour / 24.0).sin();
        features[idx + 1] = (2.0 * PI * hour / 24.0).cos();
        idx += 2;

        // Day of week (7-day cycle)
        let dow = dt.weekday().num_days_from_monday() as f64;
        features[idx] = (2.0 * PI * dow / 7.0).sin();
        features[idx + 1] = (2.0 * PI * dow / 7.0).cos();
        idx += 2;

        // Day of month (roughly 30-day cycle)
        let dom = dt.day() as f64;
        features[idx] = (2.0 * PI * dom / 31.0).sin();
        features[idx + 1] = (2.0 * PI * dom / 31.0).cos();
        idx += 2;

        // Month of year (12-month cycle)
        let month = dt.month() as f64;
        features[idx] = (2.0 * PI * month / 12.0).sin();
        features[idx + 1] = (2.0 * PI * month / 12.0).cos();
        idx += 2;

        // Minute of hour (60-minute cycle)
        let minute = dt.minute() as f64;
        features[idx] = (2.0 * PI * minute / 60.0).sin();
        features[idx + 1] = (2.0 * PI * minute / 60.0).cos();
        idx += 2;

        // Week of year (52-week cycle)
        let week = dt.iso_week().week() as f64;
        features[idx] = (2.0 * PI * week / 52.0).sin();
        features[idx + 1] = (2.0 * PI * week / 52.0).cos();
        idx += 2;

        // Binary features
        // Is weekend
        features[idx] = if dt.weekday() == Weekday::Sat || dt.weekday() == Weekday::Sun {
            1.0
        } else {
            0.0
        };
        idx += 1;

        // Is month start (first 3 days)
        features[idx] = if dt.day() <= 3 { 1.0 } else { 0.0 };
        idx += 1;

        // Is month end (last 3 days - approximation)
        features[idx] = if dt.day() >= 28 { 1.0 } else { 0.0 };
        idx += 1;

        // Quarter indicator
        features[idx] = ((dt.month() - 1) / 3) as f64 / 3.0;

        features
    }

    fn encode_linear(&self, dt: &DateTime<Utc>) -> Array1<f64> {
        let mut features = Array1::zeros(self.d_model);
        let mut idx = 0;

        // Hour of day (one-hot, 24 dims)
        features[idx + dt.hour() as usize] = 1.0;
        idx += 24;

        // Day of week (one-hot, 7 dims)
        features[idx + dt.weekday().num_days_from_monday() as usize] = 1.0;
        idx += 7;

        // Day of month (one-hot, 31 dims)
        features[idx + (dt.day() - 1) as usize] = 1.0;
        idx += 31;

        // Month (one-hot, 12 dims)
        features[idx + (dt.month() - 1) as usize] = 1.0;
        idx += 12;

        // Binary features
        features[idx] = if dt.weekday() == Weekday::Sat || dt.weekday() == Weekday::Sun {
            1.0
        } else {
            0.0
        };
        features[idx + 1] = if dt.day() <= 3 { 1.0 } else { 0.0 };
        features[idx + 2] = if dt.day() >= 28 { 1.0 } else { 0.0 };
        features[idx + 3] = ((dt.month() - 1) / 3) as f64 / 3.0;

        features
    }
}

impl Default for CalendarEncoding {
    fn default() -> Self {
        Self::new()
    }
}

/// Trading session encoding for different market types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketType {
    /// Traditional stock market (9:30 AM - 4:00 PM EST)
    Stock,
    /// Cryptocurrency market (24/7 with regional sessions)
    Crypto,
    /// Forex market (24/5 with regional sessions)
    Forex,
}

/// Market session encoding
///
/// Encodes trading session information for different market types.
/// Captures patterns like:
/// - Pre-market vs regular vs after-hours (stocks)
/// - Asian vs European vs American sessions (crypto/forex)
/// - Market open/close proximity
#[derive(Debug, Clone)]
pub struct MarketSessionEncoding {
    market_type: MarketType,
    d_model: usize,
}

impl MarketSessionEncoding {
    /// Create a new market session encoding
    pub fn new(market_type: MarketType) -> Self {
        let d_model = match market_type {
            MarketType::Stock => 8,
            MarketType::Crypto => 10,
            MarketType::Forex => 10,
        };
        Self { market_type, d_model }
    }

    /// Get encoding dimension
    pub fn dim(&self) -> usize {
        self.d_model
    }

    /// Encode a timestamp for the configured market
    pub fn encode_timestamp(&self, timestamp: i64) -> Array1<f64> {
        let dt = DateTime::from_timestamp(timestamp, 0)
            .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap());

        match self.market_type {
            MarketType::Stock => self.encode_stock_session(&dt),
            MarketType::Crypto => self.encode_crypto_session(&dt),
            MarketType::Forex => self.encode_forex_session(&dt),
        }
    }

    /// Encode multiple timestamps
    pub fn encode_timestamps(&self, timestamps: &[i64]) -> Array2<f64> {
        let mut result = Array2::zeros((timestamps.len(), self.d_model));
        for (i, &ts) in timestamps.iter().enumerate() {
            result.row_mut(i).assign(&self.encode_timestamp(ts));
        }
        result
    }

    fn encode_stock_session(&self, dt: &DateTime<Utc>) -> Array1<f64> {
        let mut features = Array1::zeros(self.d_model);

        // Convert to EST (UTC-5)
        let hour_est = (dt.hour() as i32 - 5).rem_euclid(24) as u32;
        let minute = dt.minute();
        let time_decimal = hour_est as f64 + minute as f64 / 60.0;

        // Session flags
        let is_premarket = time_decimal >= 4.0 && time_decimal < 9.5;
        let is_regular = time_decimal >= 9.5 && time_decimal < 16.0;
        let is_afterhours = time_decimal >= 16.0 && time_decimal < 20.0;
        let is_closed = !is_premarket && !is_regular && !is_afterhours;

        features[0] = if is_premarket { 1.0 } else { 0.0 };
        features[1] = if is_regular { 1.0 } else { 0.0 };
        features[2] = if is_afterhours { 1.0 } else { 0.0 };
        features[3] = if is_closed { 1.0 } else { 0.0 };

        // Time within regular session (normalized 0-1)
        if is_regular {
            features[4] = (time_decimal - 9.5) / 6.5;
        }

        // Open/close proximity (peaks at open and close)
        if is_regular {
            let dist_to_open = (time_decimal - 9.5).abs();
            let dist_to_close = (time_decimal - 16.0).abs();
            features[5] = (-dist_to_open).exp();
            features[6] = (-dist_to_close).exp();
        }

        // Weekend flag
        features[7] = if dt.weekday() == Weekday::Sat || dt.weekday() == Weekday::Sun {
            1.0
        } else {
            0.0
        };

        features
    }

    fn encode_crypto_session(&self, dt: &DateTime<Utc>) -> Array1<f64> {
        let mut features = Array1::zeros(self.d_model);

        let hour = dt.hour();

        // Regional sessions (rough boundaries)
        // Asia: 00:00-08:00 UTC
        // Europe: 08:00-16:00 UTC
        // Americas: 16:00-24:00 UTC
        let is_asia = hour < 8;
        let is_europe = hour >= 8 && hour < 16;
        let is_americas = hour >= 16;

        features[0] = if is_asia { 1.0 } else { 0.0 };
        features[1] = if is_europe { 1.0 } else { 0.0 };
        features[2] = if is_americas { 1.0 } else { 0.0 };

        // Session overlap periods (higher volatility)
        let asia_europe_overlap = hour >= 7 && hour <= 9;
        let europe_americas_overlap = hour >= 13 && hour <= 17;
        features[3] = if asia_europe_overlap { 1.0 } else { 0.0 };
        features[4] = if europe_americas_overlap { 1.0 } else { 0.0 };

        // Cyclical hour encoding
        features[5] = (2.0 * PI * hour as f64 / 24.0).sin();
        features[6] = (2.0 * PI * hour as f64 / 24.0).cos();

        // Weekend (typically lower volume)
        features[7] = if dt.weekday() == Weekday::Sat || dt.weekday() == Weekday::Sun {
            1.0
        } else {
            0.0
        };

        // Funding rate periods (every 8 hours for most exchanges)
        let hours_to_funding = hour % 8;
        features[8] = (hours_to_funding as f64 / 8.0).exp() - 1.0;

        // UTC midnight proximity
        features[9] = (-(hour as f64).min((24 - hour) as f64) / 4.0).exp();

        features
    }

    fn encode_forex_session(&self, dt: &DateTime<Utc>) -> Array1<f64> {
        let mut features = Array1::zeros(self.d_model);

        let hour = dt.hour();

        // Major forex sessions
        // Sydney: 21:00-06:00 UTC
        // Tokyo: 00:00-09:00 UTC
        // London: 08:00-17:00 UTC
        // New York: 13:00-22:00 UTC

        let is_sydney = hour >= 21 || hour < 6;
        let is_tokyo = hour < 9;
        let is_london = hour >= 8 && hour < 17;
        let is_newyork = hour >= 13 && hour < 22;

        features[0] = if is_sydney { 1.0 } else { 0.0 };
        features[1] = if is_tokyo { 1.0 } else { 0.0 };
        features[2] = if is_london { 1.0 } else { 0.0 };
        features[3] = if is_newyork { 1.0 } else { 0.0 };

        // London-NY overlap (highest volume)
        features[4] = if hour >= 13 && hour < 17 { 1.0 } else { 0.0 };

        // Tokyo-London overlap
        features[5] = if hour >= 8 && hour < 9 { 1.0 } else { 0.0 };

        // Number of active sessions
        let active_count = (is_sydney as u32)
            + (is_tokyo as u32)
            + (is_london as u32)
            + (is_newyork as u32);
        features[6] = active_count as f64 / 4.0;

        // Cyclical hour
        features[7] = (2.0 * PI * hour as f64 / 24.0).sin();
        features[8] = (2.0 * PI * hour as f64 / 24.0).cos();

        // Weekend (market closed)
        features[9] = if dt.weekday() == Weekday::Sat
            || (dt.weekday() == Weekday::Fri && hour >= 21)
            || (dt.weekday() == Weekday::Sun && hour < 21)
        {
            1.0
        } else {
            0.0
        };

        features
    }
}

/// Multi-scale temporal encoding
///
/// Combines multiple temporal features at different scales
/// for comprehensive time representation.
#[derive(Debug, Clone)]
pub struct MultiScaleTemporalEncoding {
    calendar: CalendarEncoding,
    session: MarketSessionEncoding,
}

impl MultiScaleTemporalEncoding {
    /// Create a new multi-scale encoding for a market type
    pub fn new(market_type: MarketType) -> Self {
        Self {
            calendar: CalendarEncoding::new(),
            session: MarketSessionEncoding::new(market_type),
        }
    }

    /// Get total encoding dimension
    pub fn dim(&self) -> usize {
        self.calendar.dim() + self.session.dim()
    }

    /// Encode a timestamp with all temporal features
    pub fn encode_timestamp(&self, timestamp: i64) -> Array1<f64> {
        let cal = self.calendar.encode_timestamp(timestamp);
        let sess = self.session.encode_timestamp(timestamp);

        // Concatenate features
        let mut result = Array1::zeros(self.dim());
        for (i, &v) in cal.iter().enumerate() {
            result[i] = v;
        }
        for (i, &v) in sess.iter().enumerate() {
            result[self.calendar.dim() + i] = v;
        }
        result
    }

    /// Encode multiple timestamps
    pub fn encode_timestamps(&self, timestamps: &[i64]) -> Array2<f64> {
        let mut result = Array2::zeros((timestamps.len(), self.dim()));
        for (i, &ts) in timestamps.iter().enumerate() {
            result.row_mut(i).assign(&self.encode_timestamp(ts));
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calendar_encoding_dim() {
        let enc = CalendarEncoding::new();
        let result = enc.encode_timestamp(1704067200); // 2024-01-01 00:00:00 UTC
        assert_eq!(result.len(), enc.dim());
    }

    #[test]
    fn test_calendar_encoding_weekend() {
        let enc = CalendarEncoding::new();

        // Saturday: 2024-01-06 12:00:00 UTC
        let saturday = enc.encode_timestamp(1704542400);
        assert_eq!(saturday[12], 1.0); // is_weekend should be true

        // Monday: 2024-01-08 12:00:00 UTC
        let monday = enc.encode_timestamp(1704715200);
        assert_eq!(monday[12], 0.0); // is_weekend should be false
    }

    #[test]
    fn test_market_session_crypto() {
        let enc = MarketSessionEncoding::new(MarketType::Crypto);

        // Test Asia session: 2024-01-01 04:00:00 UTC
        let asia = enc.encode_timestamp(1704081600);
        assert_eq!(asia[0], 1.0); // is_asia
        assert_eq!(asia[1], 0.0); // is_europe
        assert_eq!(asia[2], 0.0); // is_americas

        // Test Europe session: 2024-01-01 12:00:00 UTC
        let europe = enc.encode_timestamp(1704110400);
        assert_eq!(europe[0], 0.0);
        assert_eq!(europe[1], 1.0);
        assert_eq!(europe[2], 0.0);
    }

    #[test]
    fn test_multi_scale_encoding() {
        let enc = MultiScaleTemporalEncoding::new(MarketType::Crypto);
        let result = enc.encode_timestamp(1704067200);
        assert_eq!(result.len(), enc.dim());
    }

    #[test]
    fn test_batch_encoding() {
        let enc = CalendarEncoding::new();
        let timestamps = vec![1704067200, 1704153600, 1704240000];
        let result = enc.encode_timestamps(&timestamps);
        assert_eq!(result.shape(), &[3, enc.dim()]);
    }
}
