//! Хранение данных в CSV формате

use crate::types::Candle;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use csv::{Reader, Writer};
use std::fs::File;
use std::path::Path;

/// Сохранить свечи в CSV файл
pub fn save_candles(path: &Path, candles: &[Candle]) -> Result<()> {
    let file = File::create(path).context("Failed to create file")?;
    let mut writer = Writer::from_writer(file);

    // Заголовок
    writer.write_record(["timestamp", "open", "high", "low", "close", "volume"])?;

    for candle in candles {
        writer.write_record([
            candle.timestamp.to_rfc3339(),
            candle.open.to_string(),
            candle.high.to_string(),
            candle.low.to_string(),
            candle.close.to_string(),
            candle.volume.to_string(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

/// Загрузить свечи из CSV файла
pub fn load_candles(path: &Path) -> Result<Vec<Candle>> {
    let file = File::open(path).context("Failed to open file")?;
    let mut reader = Reader::from_reader(file);
    let mut candles = Vec::new();

    for result in reader.records() {
        let record = result?;
        if record.len() < 6 {
            continue;
        }

        let timestamp: DateTime<Utc> = record[0].parse().context("Failed to parse timestamp")?;
        let open: f64 = record[1].parse().context("Failed to parse open")?;
        let high: f64 = record[2].parse().context("Failed to parse high")?;
        let low: f64 = record[3].parse().context("Failed to parse low")?;
        let close: f64 = record[4].parse().context("Failed to parse close")?;
        let volume: f64 = record[5].parse().context("Failed to parse volume")?;

        candles.push(Candle {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        });
    }

    Ok(candles)
}

/// Сохранить временной ряд в CSV
pub fn save_series(path: &Path, timestamps: &[DateTime<Utc>], values: &[f64]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = Writer::from_writer(file);

    writer.write_record(["timestamp", "value"])?;

    for (ts, val) in timestamps.iter().zip(values.iter()) {
        writer.write_record([ts.to_rfc3339(), val.to_string()])?;
    }

    writer.flush()?;
    Ok(())
}

/// Загрузить временной ряд из CSV
pub fn load_series(path: &Path) -> Result<(Vec<DateTime<Utc>>, Vec<f64>)> {
    let file = File::open(path)?;
    let mut reader = Reader::from_reader(file);

    let mut timestamps = Vec::new();
    let mut values = Vec::new();

    for result in reader.records() {
        let record = result?;
        if record.len() < 2 {
            continue;
        }

        timestamps.push(record[0].parse()?);
        values.push(record[1].parse()?);
    }

    Ok((timestamps, values))
}
