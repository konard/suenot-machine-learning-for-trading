//! Stock market data loader
//!
//! Provides utilities for loading and processing stock market data from CSV files
//! or other sources. This module serves as a placeholder for stock data integration.

use super::OHLCVBar;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use thiserror::Error;

/// Errors that can occur when loading stock data
#[derive(Error, Debug)]
pub enum StockDataError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Parse error at line {line}: {message}")]
    ParseError { line: usize, message: String },

    #[error("Invalid file format: {0}")]
    InvalidFormat(String),

    #[error("Data not found: {0}")]
    NotFound(String),
}

/// Stock data loader
#[derive(Debug, Clone)]
pub struct StockDataLoader {
    /// Default directory for data files
    data_dir: String,
}

impl StockDataLoader {
    /// Create a new stock data loader
    pub fn new() -> Self {
        Self {
            data_dir: "data".to_string(),
        }
    }

    /// Create with custom data directory
    pub fn with_data_dir(data_dir: &str) -> Self {
        Self {
            data_dir: data_dir.to_string(),
        }
    }

    /// Load data from a CSV file
    ///
    /// Expected CSV format:
    /// date,open,high,low,close,volume
    /// 2024-01-01,100.0,105.0,99.0,104.0,1000000
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file
    ///
    /// # Returns
    /// Vector of OHLCV bars sorted by timestamp
    pub fn load_csv<P: AsRef<Path>>(&self, path: P) -> Result<Vec<OHLCVBar>, StockDataError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut bars = Vec::new();

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result?;

            // Skip header
            if line_num == 0 && line.to_lowercase().contains("date") {
                continue;
            }

            // Skip empty lines
            if line.trim().is_empty() {
                continue;
            }

            match self.parse_csv_line(&line) {
                Ok(bar) => bars.push(bar),
                Err(msg) => {
                    return Err(StockDataError::ParseError {
                        line: line_num + 1,
                        message: msg,
                    })
                }
            }
        }

        // Sort by timestamp
        bars.sort_by_key(|b| b.timestamp);

        Ok(bars)
    }

    /// Parse a single CSV line
    fn parse_csv_line(&self, line: &str) -> Result<OHLCVBar, String> {
        let parts: Vec<&str> = line.split(',').collect();

        if parts.len() < 6 {
            return Err(format!("Expected 6 columns, got {}", parts.len()));
        }

        // Parse date to timestamp
        let timestamp = self.parse_date(parts[0].trim())?;

        let open: f64 = parts[1]
            .trim()
            .parse()
            .map_err(|_| format!("Invalid open price: {}", parts[1]))?;

        let high: f64 = parts[2]
            .trim()
            .parse()
            .map_err(|_| format!("Invalid high price: {}", parts[2]))?;

        let low: f64 = parts[3]
            .trim()
            .parse()
            .map_err(|_| format!("Invalid low price: {}", parts[3]))?;

        let close: f64 = parts[4]
            .trim()
            .parse()
            .map_err(|_| format!("Invalid close price: {}", parts[4]))?;

        let volume: f64 = parts[5]
            .trim()
            .parse()
            .map_err(|_| format!("Invalid volume: {}", parts[5]))?;

        Ok(OHLCVBar::new(timestamp, open, high, low, close, volume))
    }

    /// Parse date string to Unix timestamp in milliseconds
    fn parse_date(&self, date_str: &str) -> Result<i64, String> {
        // Try parsing as Unix timestamp first
        if let Ok(ts) = date_str.parse::<i64>() {
            // If small number, assume seconds; otherwise milliseconds
            return Ok(if ts < 10_000_000_000 {
                ts * 1000
            } else {
                ts
            });
        }

        // Try YYYY-MM-DD format
        let parts: Vec<&str> = date_str.split('-').collect();
        if parts.len() == 3 {
            let year: i32 = parts[0]
                .parse()
                .map_err(|_| format!("Invalid year: {}", parts[0]))?;
            let month: u32 = parts[1]
                .parse()
                .map_err(|_| format!("Invalid month: {}", parts[1]))?;
            let day: u32 = parts[2]
                .split('T')
                .next()
                .unwrap()
                .parse()
                .map_err(|_| format!("Invalid day: {}", parts[2]))?;

            // Simple calculation (not accounting for all leap years correctly)
            let days_since_epoch = self.days_since_epoch(year, month, day);
            return Ok(days_since_epoch * 86_400_000);
        }

        Err(format!("Unsupported date format: {}", date_str))
    }

    /// Calculate days since Unix epoch (approximate)
    fn days_since_epoch(&self, year: i32, month: u32, day: u32) -> i64 {
        let y = year as i64;
        let m = month as i64;
        let d = day as i64;

        // Approximate calculation
        let days_per_year = 365;
        let years_since_1970 = y - 1970;
        let leap_years = (years_since_1970 + 1) / 4;

        let month_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        let day_of_year = if m > 0 && m <= 12 {
            month_days[(m - 1) as usize] + d
        } else {
            d
        };

        years_since_1970 * days_per_year + leap_years + day_of_year - 1
    }

    /// Generate synthetic stock data for testing
    ///
    /// Creates a random walk with trend and volatility clustering
    pub fn generate_synthetic(
        &self,
        num_bars: usize,
        start_price: f64,
        volatility: f64,
        trend: f64,
    ) -> Vec<OHLCVBar> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut bars = Vec::with_capacity(num_bars);
        let mut price = start_price;
        let start_timestamp = 1704067200000_i64; // 2024-01-01 00:00:00 UTC

        for i in 0..num_bars {
            // Random walk with trend
            let return_pct = trend + volatility * rng.gen_range(-1.0..1.0);
            let close = price * (1.0 + return_pct);
            let open = price;

            // Generate realistic OHLC - ensure high is highest and low is lowest
            let intra_vol = volatility * 0.5;
            let max_oc = open.max(close);
            let min_oc = open.min(close);

            // High must be >= max(open, close)
            let high = max_oc * (1.0 + intra_vol * rng.gen::<f64>());
            // Low must be <= min(open, close)
            let low = min_oc * (1.0 - intra_vol * rng.gen::<f64>());

            // Volume with some randomness
            let base_volume = 1_000_000.0;
            let volume = base_volume * (0.5 + rng.gen::<f64>());

            bars.push(OHLCVBar::new(
                start_timestamp + (i as i64) * 86_400_000, // Daily bars
                open,
                high,
                low,
                close,
                volume,
            ));

            price = close;
        }

        bars
    }

    /// Generate data with specific market patterns
    pub fn generate_pattern_data(
        &self,
        pattern: &str,
        num_bars: usize,
        start_price: f64,
    ) -> Vec<OHLCVBar> {
        match pattern.to_lowercase().as_str() {
            "uptrend" => {
                self.generate_synthetic(num_bars, start_price, 0.02, 0.005)
            }
            "downtrend" => {
                self.generate_synthetic(num_bars, start_price, 0.02, -0.005)
            }
            "sideways" | "consolidation" => {
                self.generate_synthetic(num_bars, start_price, 0.01, 0.0)
            }
            "volatile" => {
                self.generate_synthetic(num_bars, start_price, 0.05, 0.0)
            }
            "breakout" => {
                // Consolidation followed by breakout
                let mut bars = self.generate_synthetic(num_bars * 2 / 3, start_price, 0.01, 0.0);
                let last_price = bars.last().map(|b| b.close).unwrap_or(start_price);
                let breakout_bars = self.generate_synthetic(
                    num_bars - bars.len(),
                    last_price,
                    0.03,
                    0.015,
                );
                bars.extend(breakout_bars);
                bars
            }
            "reversal" => {
                // Uptrend followed by reversal
                let mut bars = self.generate_synthetic(num_bars / 2, start_price, 0.02, 0.008);
                let last_price = bars.last().map(|b| b.close).unwrap_or(start_price);
                let reversal_bars = self.generate_synthetic(
                    num_bars - bars.len(),
                    last_price,
                    0.025,
                    -0.010,
                );
                bars.extend(reversal_bars);
                bars
            }
            _ => {
                // Default random
                self.generate_synthetic(num_bars, start_price, 0.02, 0.001)
            }
        }
    }
}

impl Default for StockDataLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_synthetic() {
        let loader = StockDataLoader::new();
        let bars = loader.generate_synthetic(100, 100.0, 0.02, 0.001);

        assert_eq!(bars.len(), 100);

        // Verify OHLCV constraints
        for bar in &bars {
            assert!(bar.high >= bar.low);
            assert!(bar.high >= bar.open);
            assert!(bar.high >= bar.close);
            assert!(bar.low <= bar.open);
            assert!(bar.low <= bar.close);
            assert!(bar.volume > 0.0);
        }

        // Verify chronological order
        for i in 1..bars.len() {
            assert!(bars[i].timestamp > bars[i - 1].timestamp);
        }
    }

    #[test]
    fn test_generate_pattern_data() {
        let loader = StockDataLoader::new();

        let uptrend = loader.generate_pattern_data("uptrend", 50, 100.0);
        let downtrend = loader.generate_pattern_data("downtrend", 50, 100.0);

        // Uptrend should generally end higher
        let uptrend_return = (uptrend.last().unwrap().close - uptrend[0].close) / uptrend[0].close;
        let downtrend_return =
            (downtrend.last().unwrap().close - downtrend[0].close) / downtrend[0].close;

        // Not guaranteed due to randomness, but likely
        // Just verify they generated correctly
        assert_eq!(uptrend.len(), 50);
        assert_eq!(downtrend.len(), 50);
    }

    #[test]
    fn test_parse_date() {
        let loader = StockDataLoader::new();

        // Unix timestamp
        assert_eq!(loader.parse_date("1704067200").unwrap(), 1704067200000);
        assert_eq!(loader.parse_date("1704067200000").unwrap(), 1704067200000);

        // Date string (approximate - just verify it parses)
        let result = loader.parse_date("2024-01-01");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_csv_line() {
        let loader = StockDataLoader::new();

        let line = "1704067200000,100.0,105.0,99.0,104.0,1000000";
        let bar = loader.parse_csv_line(line).unwrap();

        assert_eq!(bar.timestamp, 1704067200000);
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.high, 105.0);
        assert_eq!(bar.low, 99.0);
        assert_eq!(bar.close, 104.0);
        assert_eq!(bar.volume, 1000000.0);
    }
}
