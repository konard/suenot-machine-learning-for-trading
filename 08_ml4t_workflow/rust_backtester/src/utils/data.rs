//! Data utilities for saving and loading market data.

use crate::models::Candle;
use anyhow::Result;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Save candles to a JSON file.
pub fn save_candles_json(candles: &[Candle], path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, candles)?;
    Ok(())
}

/// Load candles from a JSON file.
pub fn load_candles_json(path: &Path) -> Result<Vec<Candle>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let candles: Vec<Candle> = serde_json::from_reader(reader)?;
    Ok(candles)
}

/// Save candles to a CSV file.
pub fn save_candles_csv(candles: &[Candle], path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut content = String::from("timestamp,symbol,open,high,low,close,volume,turnover\n");

    for candle in candles {
        content.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            candle.timestamp.to_rfc3339(),
            candle.symbol,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
            candle.turnover
        ));
    }

    fs::write(path, content)?;
    Ok(())
}

/// Load candles from a CSV file.
pub fn load_candles_csv(path: &Path) -> Result<Vec<Candle>> {
    use chrono::DateTime;

    let content = fs::read_to_string(path)?;
    let mut candles = Vec::new();

    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue; // Skip header
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 8 {
            continue;
        }

        let timestamp = DateTime::parse_from_rfc3339(parts[0])?.with_timezone(&chrono::Utc);
        let candle = Candle {
            timestamp,
            symbol: parts[1].to_string(),
            open: parts[2].parse()?,
            high: parts[3].parse()?,
            low: parts[4].parse()?,
            close: parts[5].parse()?,
            volume: parts[6].parse()?,
            turnover: parts[7].parse()?,
        };
        candles.push(candle);
    }

    Ok(candles)
}

/// Extract closing prices from candles.
pub fn extract_closes(candles: &[Candle]) -> Vec<f64> {
    candles.iter().map(|c| c.close).collect()
}

/// Extract OHLC data as separate vectors.
pub fn extract_ohlc(candles: &[Candle]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let opens: Vec<f64> = candles.iter().map(|c| c.open).collect();
    let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    (opens, highs, lows, closes)
}

/// Resample candles to a higher timeframe.
pub fn resample_candles(candles: &[Candle], factor: usize) -> Vec<Candle> {
    if factor <= 1 || candles.is_empty() {
        return candles.to_vec();
    }

    candles
        .chunks(factor)
        .filter_map(|chunk| {
            if chunk.is_empty() {
                return None;
            }

            let first = &chunk[0];
            let last = &chunk[chunk.len() - 1];

            let high = chunk.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
            let low = chunk.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
            let volume: f64 = chunk.iter().map(|c| c.volume).sum();
            let turnover: f64 = chunk.iter().map(|c| c.turnover).sum();

            Some(Candle {
                timestamp: first.timestamp,
                symbol: first.symbol.clone(),
                open: first.open,
                high,
                low,
                close: last.close,
                volume,
                turnover,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_extract_closes() {
        let candles = vec![
            Candle::new(Utc::now(), "BTC".to_string(), 100.0, 110.0, 90.0, 105.0, 1000.0),
            Candle::new(Utc::now(), "BTC".to_string(), 105.0, 115.0, 100.0, 110.0, 1200.0),
        ];
        let closes = extract_closes(&candles);
        assert_eq!(closes, vec![105.0, 110.0]);
    }

    #[test]
    fn test_resample() {
        let now = Utc::now();
        let candles = vec![
            Candle::new(now, "BTC".to_string(), 100.0, 110.0, 95.0, 105.0, 100.0),
            Candle::new(now, "BTC".to_string(), 105.0, 115.0, 100.0, 110.0, 150.0),
            Candle::new(now, "BTC".to_string(), 110.0, 120.0, 105.0, 115.0, 200.0),
            Candle::new(now, "BTC".to_string(), 115.0, 125.0, 110.0, 120.0, 250.0),
        ];

        let resampled = resample_candles(&candles, 2);
        assert_eq!(resampled.len(), 2);
        assert_eq!(resampled[0].open, 100.0);
        assert_eq!(resampled[0].close, 110.0);
        assert_eq!(resampled[0].high, 115.0);
        assert_eq!(resampled[0].low, 95.0);
        assert_eq!(resampled[0].volume, 250.0);
    }
}
