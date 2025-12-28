//! Data loading and saving utilities
//!
//! Provides functions to load and save market data to/from CSV files.

use super::types::Candle;
use anyhow::{Context, Result};
use csv::{Reader, Writer};
use std::fs::File;
use std::path::Path;

/// Data loader for CSV files
pub struct DataLoader;

impl DataLoader {
    /// Load candles from a CSV file
    pub fn load_candles<P: AsRef<Path>>(path: P) -> Result<Vec<Candle>> {
        let file = File::open(&path)
            .with_context(|| format!("Failed to open file: {:?}", path.as_ref()))?;

        let mut reader = Reader::from_reader(file);
        let mut candles = Vec::new();

        for result in reader.deserialize() {
            let candle: Candle = result.context("Failed to parse candle")?;
            candles.push(candle);
        }

        // Sort by timestamp
        candles.sort_by_key(|c| c.timestamp);

        Ok(candles)
    }

    /// Save candles to a CSV file
    pub fn save_candles<P: AsRef<Path>>(candles: &[Candle], path: P) -> Result<()> {
        let file = File::create(&path)
            .with_context(|| format!("Failed to create file: {:?}", path.as_ref()))?;

        let mut writer = Writer::from_writer(file);

        for candle in candles {
            writer.serialize(candle)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load candles from JSON file
    pub fn load_candles_json<P: AsRef<Path>>(path: P) -> Result<Vec<Candle>> {
        let file = File::open(&path)
            .with_context(|| format!("Failed to open file: {:?}", path.as_ref()))?;

        let candles: Vec<Candle> = serde_json::from_reader(file)?;
        Ok(candles)
    }

    /// Save candles to JSON file
    pub fn save_candles_json<P: AsRef<Path>>(candles: &[Candle], path: P) -> Result<()> {
        let file = File::create(&path)
            .with_context(|| format!("Failed to create file: {:?}", path.as_ref()))?;

        serde_json::to_writer_pretty(file, candles)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_save_and_load_candles() {
        let candles = vec![
            Candle {
                timestamp: 1000,
                open: 100.0,
                high: 110.0,
                low: 95.0,
                close: 105.0,
                volume: 1000.0,
                turnover: 100000.0,
            },
            Candle {
                timestamp: 2000,
                open: 105.0,
                high: 115.0,
                low: 100.0,
                close: 110.0,
                volume: 1200.0,
                turnover: 120000.0,
            },
        ];

        let dir = tempdir().unwrap();
        let path = dir.path().join("test_candles.csv");

        DataLoader::save_candles(&candles, &path).unwrap();
        let loaded = DataLoader::load_candles(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].timestamp, 1000);
        assert_eq!(loaded[1].close, 110.0);
    }
}
