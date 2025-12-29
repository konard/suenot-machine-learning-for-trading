//! Функции для сохранения и загрузки данных

use crate::api::Candle;
use anyhow::Result;
use std::path::Path;

/// Сохранить свечи в CSV
pub fn save_candles_csv(candles: &[Candle], path: impl AsRef<Path>) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;

    writer.write_record([
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
    ])?;

    for candle in candles {
        writer.write_record([
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

/// Загрузить свечи из CSV
pub fn load_candles_csv(path: impl AsRef<Path>) -> Result<Vec<Candle>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut candles = Vec::new();

    for result in reader.records() {
        let record = result?;
        if record.len() >= 7 {
            candles.push(Candle {
                timestamp: record[0].parse().unwrap_or(0),
                open: record[1].parse().unwrap_or(0.0),
                high: record[2].parse().unwrap_or(0.0),
                low: record[3].parse().unwrap_or(0.0),
                close: record[4].parse().unwrap_or(0.0),
                volume: record[5].parse().unwrap_or(0.0),
                turnover: record[6].parse().unwrap_or(0.0),
            });
        }
    }

    Ok(candles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;

    #[test]
    fn test_save_load_candles() {
        let candles = vec![
            Candle {
                timestamp: 1704067200000,
                open: 100.0,
                high: 110.0,
                low: 90.0,
                close: 105.0,
                volume: 1000.0,
                turnover: 101000.0,
            },
            Candle {
                timestamp: 1704067260000,
                open: 105.0,
                high: 115.0,
                low: 95.0,
                close: 110.0,
                volume: 1100.0,
                turnover: 115000.0,
            },
        ];

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        save_candles_csv(&candles, path).unwrap();
        let loaded = load_candles_csv(path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].timestamp, 1704067200000);
        assert!((loaded[0].close - 105.0).abs() < 0.001);
    }
}
