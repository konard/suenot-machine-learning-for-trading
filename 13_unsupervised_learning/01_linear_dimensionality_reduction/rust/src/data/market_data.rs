//! Market data structures and operations

use crate::api::{BybitClient, OHLCV, Timeframe};
use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Container for multi-asset market data
#[derive(Debug, Clone)]
pub struct MarketData {
    /// Symbol names
    pub symbols: Vec<String>,
    /// Timestamps (common across all symbols)
    pub timestamps: Vec<i64>,
    /// Close prices matrix (rows = timestamps, cols = symbols)
    pub close_prices: Array2<f64>,
    /// Volume matrix (rows = timestamps, cols = symbols)
    pub volumes: Array2<f64>,
}

impl MarketData {
    /// Create new MarketData from symbol data
    pub fn new(
        symbols: Vec<String>,
        timestamps: Vec<i64>,
        close_prices: Array2<f64>,
        volumes: Array2<f64>,
    ) -> Self {
        Self {
            symbols,
            timestamps,
            close_prices,
            volumes,
        }
    }

    /// Fetch market data for multiple symbols from Bybit
    pub fn fetch_from_bybit(
        client: &BybitClient,
        symbols: &[String],
        category: &str,
        timeframe: Timeframe,
        limit: u32,
    ) -> Result<Self> {
        let mut symbol_data: HashMap<String, Vec<OHLCV>> = HashMap::new();

        // Fetch data for each symbol
        for symbol in symbols {
            println!("Fetching data for {}...", symbol);
            let klines = client
                .get_klines(category, symbol, timeframe, Some(limit), None, None)
                .with_context(|| format!("Failed to fetch data for {}", symbol))?;

            if !klines.is_empty() {
                symbol_data.insert(symbol.clone(), klines);
            }
        }

        // Find common timestamps
        let all_timestamps: Vec<Vec<i64>> = symbol_data
            .values()
            .map(|klines| klines.iter().map(|k| k.timestamp).collect())
            .collect();

        if all_timestamps.is_empty() {
            anyhow::bail!("No data fetched for any symbol");
        }

        // Get intersection of timestamps
        let mut common_timestamps: Vec<i64> = all_timestamps[0].clone();
        for timestamps in all_timestamps.iter().skip(1) {
            let ts_set: std::collections::HashSet<i64> = timestamps.iter().copied().collect();
            common_timestamps.retain(|t| ts_set.contains(t));
        }
        common_timestamps.sort();

        if common_timestamps.is_empty() {
            anyhow::bail!("No common timestamps found across symbols");
        }

        // Build matrices
        let n_timestamps = common_timestamps.len();
        let n_symbols = symbol_data.len();
        let mut close_prices = Array2::zeros((n_timestamps, n_symbols));
        let mut volumes = Array2::zeros((n_timestamps, n_symbols));
        let mut valid_symbols = Vec::new();

        for (col_idx, symbol) in symbols.iter().enumerate() {
            if let Some(klines) = symbol_data.get(symbol) {
                valid_symbols.push(symbol.clone());

                // Create timestamp -> index map
                let ts_map: HashMap<i64, &OHLCV> =
                    klines.iter().map(|k| (k.timestamp, k)).collect();

                for (row_idx, &ts) in common_timestamps.iter().enumerate() {
                    if let Some(kline) = ts_map.get(&ts) {
                        close_prices[[row_idx, col_idx]] = kline.close;
                        volumes[[row_idx, col_idx]] = kline.volume;
                    }
                }
            }
        }

        // Trim to actual number of valid symbols
        let close_prices = close_prices.slice(ndarray::s![.., ..valid_symbols.len()]).to_owned();
        let volumes = volumes.slice(ndarray::s![.., ..valid_symbols.len()]).to_owned();

        Ok(Self {
            symbols: valid_symbols,
            timestamps: common_timestamps,
            close_prices,
            volumes,
        })
    }

    /// Number of symbols
    pub fn n_symbols(&self) -> usize {
        self.symbols.len()
    }

    /// Number of time periods
    pub fn n_periods(&self) -> usize {
        self.timestamps.len()
    }

    /// Get close prices for a specific symbol
    pub fn get_symbol_prices(&self, symbol: &str) -> Option<Array1<f64>> {
        let idx = self.symbols.iter().position(|s| s == symbol)?;
        Some(self.close_prices.column(idx).to_owned())
    }

    /// Get close prices for a specific time period
    pub fn get_period_prices(&self, idx: usize) -> Option<Array1<f64>> {
        if idx >= self.n_periods() {
            return None;
        }
        Some(self.close_prices.row(idx).to_owned())
    }

    /// Export to CSV
    pub fn to_csv(&self, path: &str) -> Result<()> {
        let mut writer = csv::Writer::from_path(path)?;

        // Write header
        let mut header = vec!["timestamp".to_string()];
        header.extend(self.symbols.iter().cloned());
        writer.write_record(&header)?;

        // Write data
        for (i, &ts) in self.timestamps.iter().enumerate() {
            let mut row = vec![ts.to_string()];
            for j in 0..self.n_symbols() {
                row.push(self.close_prices[[i, j]].to_string());
            }
            writer.write_record(&row)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load from CSV
    pub fn from_csv(path: &str) -> Result<Self> {
        let mut reader = csv::Reader::from_path(path)?;

        let headers: Vec<String> = reader.headers()?.iter().map(String::from).collect();
        let symbols: Vec<String> = headers.into_iter().skip(1).collect();

        let mut timestamps = Vec::new();
        let mut prices_data = Vec::new();

        for result in reader.records() {
            let record = result?;
            let ts: i64 = record[0].parse()?;
            timestamps.push(ts);

            let prices: Vec<f64> = record
                .iter()
                .skip(1)
                .map(|s| s.parse().unwrap_or(0.0))
                .collect();
            prices_data.push(prices);
        }

        let n_timestamps = timestamps.len();
        let n_symbols = symbols.len();
        let mut close_prices = Array2::zeros((n_timestamps, n_symbols));

        for (i, prices) in prices_data.iter().enumerate() {
            for (j, &price) in prices.iter().enumerate() {
                close_prices[[i, j]] = price;
            }
        }

        Ok(Self {
            symbols,
            timestamps,
            close_prices,
            volumes: Array2::zeros((n_timestamps, n_symbols)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_market_data_creation() {
        let symbols = vec!["BTC".to_string(), "ETH".to_string()];
        let timestamps = vec![1000, 2000, 3000];
        let prices = array![[100.0, 10.0], [110.0, 11.0], [105.0, 12.0]];
        let volumes = Array2::ones((3, 2));

        let data = MarketData::new(symbols.clone(), timestamps.clone(), prices, volumes);

        assert_eq!(data.n_symbols(), 2);
        assert_eq!(data.n_periods(), 3);
        assert_eq!(data.symbols, symbols);
    }

    #[test]
    fn test_get_symbol_prices() {
        let symbols = vec!["BTC".to_string(), "ETH".to_string()];
        let timestamps = vec![1000, 2000, 3000];
        let prices = array![[100.0, 10.0], [110.0, 11.0], [105.0, 12.0]];
        let volumes = Array2::ones((3, 2));

        let data = MarketData::new(symbols, timestamps, prices, volumes);

        let btc_prices = data.get_symbol_prices("BTC").unwrap();
        assert_eq!(btc_prices.len(), 3);
        assert_eq!(btc_prices[0], 100.0);
        assert_eq!(btc_prices[1], 110.0);
    }
}
