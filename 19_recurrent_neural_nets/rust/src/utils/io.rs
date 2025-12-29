//! Функции ввода-вывода для работы с данными

use crate::data::Candle;
use anyhow::Result;
use csv::{Reader, Writer};
use std::fs::File;
use std::path::Path;

/// Сохраняет свечи в CSV файл
pub fn save_candles_csv(candles: &[Candle], path: &str) -> Result<()> {
    let mut writer = Writer::from_path(path)?;

    // Заголовок
    writer.write_record(["timestamp", "open", "high", "low", "close", "volume", "turnover"])?;

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

/// Загружает свечи из CSV файла
pub fn load_candles_csv(path: &str) -> Result<Vec<Candle>> {
    let file = File::open(path)?;
    let mut reader = Reader::from_reader(file);

    let mut candles = Vec::new();

    for result in reader.records() {
        let record = result?;

        if record.len() >= 7 {
            let candle = Candle::new(
                record[0].parse().unwrap_or(0),
                record[1].parse().unwrap_or(0.0),
                record[2].parse().unwrap_or(0.0),
                record[3].parse().unwrap_or(0.0),
                record[4].parse().unwrap_or(0.0),
                record[5].parse().unwrap_or(0.0),
                record[6].parse().unwrap_or(0.0),
            );
            candles.push(candle);
        }
    }

    Ok(candles)
}

/// Сохраняет предсказания в CSV файл
pub fn save_predictions_csv(
    timestamps: &[i64],
    actual: &[f64],
    predicted: &[f64],
    path: &str,
) -> Result<()> {
    let mut writer = Writer::from_path(path)?;

    writer.write_record(["timestamp", "actual", "predicted", "error"])?;

    for i in 0..timestamps.len().min(actual.len()).min(predicted.len()) {
        let error = actual[i] - predicted[i];
        writer.write_record([
            timestamps[i].to_string(),
            actual[i].to_string(),
            predicted[i].to_string(),
            error.to_string(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

/// Проверяет существование файла
pub fn file_exists(path: &str) -> bool {
    Path::new(path).exists()
}

/// Создаёт директорию если не существует
pub fn ensure_dir(path: &str) -> Result<()> {
    let path = Path::new(path);
    if !path.exists() {
        std::fs::create_dir_all(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_save_load_candles() {
        let candles = vec![
            Candle::new(1000, 100.0, 105.0, 95.0, 102.0, 1000.0, 100000.0),
            Candle::new(2000, 102.0, 108.0, 100.0, 107.0, 1200.0, 120000.0),
        ];

        let path = "/tmp/test_candles.csv";
        save_candles_csv(&candles, path).unwrap();

        let loaded = load_candles_csv(path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].timestamp, 1000);
        assert_eq!(loaded[1].close, 107.0);

        fs::remove_file(path).ok();
    }
}
