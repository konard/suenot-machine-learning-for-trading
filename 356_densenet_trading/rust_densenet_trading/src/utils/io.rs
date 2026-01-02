//! I/O Utilities
//!
//! File input/output utilities for data persistence.

use anyhow::Result;
use csv::{Reader, Writer};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;

use crate::data::candle::Candle;

/// Save candles to CSV file
pub fn save_to_csv<P: AsRef<Path>>(candles: &[Candle], path: P) -> Result<()> {
    let mut writer = Writer::from_path(path)?;

    // Write header
    writer.write_record(&["timestamp", "open", "high", "low", "close", "volume", "turnover"])?;

    // Write data
    for candle in candles {
        writer.write_record(&[
            candle.timestamp.to_string(),
            candle.open.to_string(),
            candle.high.to_string(),
            candle.low.to_string(),
            candle.close.to_string(),
            candle.volume.to_string(),
            candle.turnover.to_string(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

/// Load candles from CSV file
pub fn load_from_csv<P: AsRef<Path>>(path: P) -> Result<Vec<Candle>> {
    let mut reader = Reader::from_path(path)?;
    let mut candles = Vec::new();

    for result in reader.records() {
        let record = result?;

        if record.len() >= 7 {
            let candle = Candle {
                timestamp: record[0].parse().unwrap_or(0),
                open: record[1].parse().unwrap_or(0.0),
                high: record[2].parse().unwrap_or(0.0),
                low: record[3].parse().unwrap_or(0.0),
                close: record[4].parse().unwrap_or(0.0),
                volume: record[5].parse().unwrap_or(0.0),
                turnover: record[6].parse().unwrap_or(0.0),
            };
            candles.push(candle);
        }
    }

    Ok(candles)
}

/// Save any serializable data to JSON
pub fn save_json<T: Serialize, P: AsRef<Path>>(data: &T, path: P) -> Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, data)?;
    Ok(())
}

/// Load data from JSON
pub fn load_json<T: for<'de> Deserialize<'de>, P: AsRef<Path>>(path: P) -> Result<T> {
    let file = File::open(path)?;
    let data = serde_json::from_reader(file)?;
    Ok(data)
}

/// Check if file exists
pub fn file_exists<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().exists()
}

/// Create directory if it doesn't exist
pub fn ensure_dir<P: AsRef<Path>>(path: P) -> Result<()> {
    if !path.as_ref().exists() {
        std::fs::create_dir_all(path)?;
    }
    Ok(())
}

/// Get file size in bytes
pub fn file_size<P: AsRef<Path>>(path: P) -> Result<u64> {
    let metadata = std::fs::metadata(path)?;
    Ok(metadata.len())
}

/// Format file size for display
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_csv_round_trip() {
        let candles = vec![
            Candle::new(1000, 100.0, 105.0, 98.0, 102.0, 1000.0, 100000.0),
            Candle::new(2000, 102.0, 108.0, 101.0, 106.0, 1500.0, 150000.0),
        ];

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        save_to_csv(&candles, path).unwrap();
        let loaded = load_from_csv(path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].timestamp, 1000);
        assert!((loaded[0].close - 102.0).abs() < 0.001);
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(500), "500 bytes");
        assert_eq!(format_size(1500), "1.46 KB");
        assert_eq!(format_size(1500000), "1.43 MB");
        assert_eq!(format_size(1500000000), "1.40 GB");
    }

    #[test]
    fn test_json_round_trip() {
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct TestData {
            value: i32,
            name: String,
        }

        let data = TestData {
            value: 42,
            name: "test".to_string(),
        };

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        save_json(&data, path).unwrap();
        let loaded: TestData = load_json(path).unwrap();

        assert_eq!(data, loaded);
    }
}
