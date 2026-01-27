//! Stock data loading and processing.
//!
//! Supports loading stock market data from CSV files for use as
//! source or target domain data in transfer learning.

use anyhow::Result;
use crate::data::features::OHLCVBar;

/// Stock data loader for CSV-formatted market data.
#[derive(Debug, Clone)]
pub struct StockDataLoader {
    /// Date format string (e.g., "%Y-%m-%d")
    date_format: String,
}

impl StockDataLoader {
    /// Create a new stock data loader.
    pub fn new() -> Self {
        Self {
            date_format: "%Y-%m-%d".to_string(),
        }
    }

    /// Set the date format for parsing CSV data.
    pub fn with_date_format(mut self, format: &str) -> Self {
        self.date_format = format.to_string();
        self
    }

    /// Load OHLCV data from a CSV file.
    ///
    /// Expected CSV columns: Date, Open, High, Low, Close, Volume
    pub fn load_csv(&self, path: &str) -> Result<Vec<OHLCVBar>> {
        let content = std::fs::read_to_string(path)?;
        self.parse_csv(&content)
    }

    /// Parse OHLCV data from CSV string content.
    pub fn parse_csv(&self, content: &str) -> Result<Vec<OHLCVBar>> {
        let mut bars = Vec::new();
        let mut lines = content.lines();

        // Skip header
        let _header = lines.next();

        for (line_num, line) in lines.enumerate() {
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() < 6 {
                tracing::warn!("Skipping line {}: insufficient fields", line_num + 2);
                continue;
            }

            let timestamp = chrono::NaiveDate::parse_from_str(fields[0].trim(), &self.date_format)
                .map(|d| {
                    d.and_hms_opt(0, 0, 0)
                        .unwrap_or_default()
                        .and_utc()
                        .timestamp()
                })
                .unwrap_or(line_num as i64);

            let open = fields[1].trim().parse::<f64>().unwrap_or(0.0);
            let high = fields[2].trim().parse::<f64>().unwrap_or(0.0);
            let low = fields[3].trim().parse::<f64>().unwrap_or(0.0);
            let close = fields[4].trim().parse::<f64>().unwrap_or(0.0);
            let volume = fields[5].trim().parse::<f64>().unwrap_or(0.0);

            bars.push(OHLCVBar::new(timestamp, open, high, low, close, volume));
        }

        Ok(bars)
    }

    /// Generate synthetic OHLCV data for testing.
    pub fn generate_synthetic(
        &self,
        n_bars: usize,
        start_price: f64,
        volatility: f64,
    ) -> Vec<OHLCVBar> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut bars = Vec::with_capacity(n_bars);
        let mut price = start_price;

        for i in 0..n_bars {
            let ret: f64 = rng.gen::<f64>() * 2.0 * volatility - volatility;
            price *= 1.0 + ret;
            price = price.max(0.01);

            let range = price * volatility * rng.gen::<f64>();
            let open = price * (1.0 + (rng.gen::<f64>() - 0.5) * volatility);
            let high = price.max(open) + range;
            let low = price.min(open) - range;
            let close = price;
            let volume = 1000.0 * (1.0 + rng.gen::<f64>() * 2.0);

            bars.push(OHLCVBar::new(
                i as i64 * 3600,
                open,
                high,
                low.max(0.01),
                close,
                volume,
            ));
        }

        bars
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
    fn test_stock_loader_creation() {
        let loader = StockDataLoader::new();
        assert_eq!(loader.date_format, "%Y-%m-%d");
    }

    #[test]
    fn test_parse_csv() {
        let loader = StockDataLoader::new();
        let csv = "Date,Open,High,Low,Close,Volume\n\
                   2024-01-01,100.0,105.0,95.0,103.0,1000\n\
                   2024-01-02,103.0,108.0,101.0,107.0,1500\n";
        let bars = loader.parse_csv(csv).unwrap();
        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].open, 100.0);
        assert_eq!(bars[0].close, 103.0);
        assert_eq!(bars[1].volume, 1500.0);
    }

    #[test]
    fn test_generate_synthetic() {
        let loader = StockDataLoader::new();
        let bars = loader.generate_synthetic(100, 100.0, 0.02);
        assert_eq!(bars.len(), 100);
        for bar in &bars {
            assert!(bar.high >= bar.low);
            assert!(bar.close > 0.0);
            assert!(bar.volume > 0.0);
        }
    }

    #[test]
    fn test_malformed_csv() {
        let loader = StockDataLoader::new();
        let csv = "Date,Open,High,Low,Close,Volume\n\
                   bad,data\n\
                   2024-01-01,100.0,105.0,95.0,103.0,1000\n";
        let bars = loader.parse_csv(csv).unwrap();
        assert_eq!(bars.len(), 1); // Skips malformed line
    }
}
