//! # Utility Functions
//!
//! Вспомогательные функции для сохранения и загрузки данных,
//! визуализации и других общих операций.

use crate::bybit_client::Kline;
use crate::data_processor::Features;
use anyhow::Result;
use csv::{Reader, Writer};
use serde::{de::DeserializeOwned, Serialize};
use std::fs::File;
use std::path::Path;

/// Сохраняет данные в CSV файл
pub fn save_csv<T: Serialize, P: AsRef<Path>>(data: &[T], path: P) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = Writer::from_writer(file);

    for record in data {
        writer.serialize(record)?;
    }

    writer.flush()?;
    Ok(())
}

/// Загружает данные из CSV файла
pub fn load_csv<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<Vec<T>> {
    let file = File::open(path)?;
    let mut reader = Reader::from_reader(file);
    let mut data = Vec::new();

    for result in reader.deserialize() {
        let record: T = result?;
        data.push(record);
    }

    Ok(data)
}

/// Сохраняет свечи в CSV
pub fn save_klines<P: AsRef<Path>>(klines: &[Kline], path: P) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = Writer::from_writer(file);

    writer.write_record(["timestamp", "open", "high", "low", "close", "volume", "turnover"])?;

    for kline in klines {
        writer.write_record(&[
            kline.open_time.to_string(),
            kline.open.to_string(),
            kline.high.to_string(),
            kline.low.to_string(),
            kline.close.to_string(),
            kline.volume.to_string(),
            kline.turnover.to_string(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

/// Загружает свечи из CSV
pub fn load_klines<P: AsRef<Path>>(path: P) -> Result<Vec<Kline>> {
    let file = File::open(path)?;
    let mut reader = Reader::from_reader(file);
    let mut klines = Vec::new();

    for result in reader.records() {
        let record = result?;
        if record.len() >= 7 {
            klines.push(Kline {
                open_time: record[0].parse()?,
                open: record[1].parse()?,
                high: record[2].parse()?,
                low: record[3].parse()?,
                close: record[4].parse()?,
                volume: record[5].parse()?,
                turnover: record[6].parse()?,
            });
        }
    }

    Ok(klines)
}

/// Сохраняет признаки в CSV
pub fn save_features<P: AsRef<Path>>(features: &Features, path: P) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = Writer::from_writer(file);

    // Заголовок
    let mut header = vec!["timestamp".to_string()];
    header.extend(features.names.clone());
    writer.write_record(&header)?;

    // Данные
    for (i, row) in features.data.iter().enumerate() {
        let mut record = vec![features.timestamps[i].to_string()];
        record.extend(row.iter().map(|v| v.to_string()));
        writer.write_record(&record)?;
    }

    writer.flush()?;
    Ok(())
}

/// Загружает признаки из CSV
pub fn load_features<P: AsRef<Path>>(path: P) -> Result<Features> {
    let file = File::open(path)?;
    let mut reader = Reader::from_reader(file);

    // Читаем заголовок
    let headers: Vec<String> = reader
        .headers()?
        .iter()
        .skip(1) // Пропускаем timestamp
        .map(String::from)
        .collect();

    let mut timestamps = Vec::new();
    let mut data = Vec::new();

    for result in reader.records() {
        let record = result?;
        if !record.is_empty() {
            timestamps.push(record[0].parse()?);

            let row: Vec<f64> = record
                .iter()
                .skip(1)
                .filter_map(|v| v.parse().ok())
                .collect();
            data.push(row);
        }
    }

    Ok(Features::new(data, headers, timestamps))
}

/// Генерирует простой ASCII график
pub fn ascii_chart(values: &[f64], width: usize, height: usize) -> String {
    if values.is_empty() {
        return String::new();
    }

    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    if range < 1e-10 {
        return format!("Constant value: {:.4}", min);
    }

    let mut chart = vec![vec![' '; width]; height];

    // Сэмплируем значения
    let step = values.len() as f64 / width as f64;
    for x in 0..width {
        let idx = ((x as f64 * step) as usize).min(values.len() - 1);
        let y = ((values[idx] - min) / range * (height - 1) as f64) as usize;
        let y = (height - 1).saturating_sub(y);
        chart[y][x] = '*';
    }

    // Преобразуем в строку
    let mut result = String::new();
    result.push_str(&format!("{:>10.4} ┤", max));
    for row in &chart {
        result.push_str(&row.iter().collect::<String>());
        result.push('\n');
        result.push_str("           │");
    }
    result.push_str(&format!("{:>10.4} ┤", min));

    result
}

/// Вычисляет скользящее среднее
pub fn moving_average(values: &[f64], window: usize) -> Vec<f64> {
    if values.len() < window || window == 0 {
        return values.to_vec();
    }

    let mut result = Vec::with_capacity(values.len() - window + 1);
    let mut sum: f64 = values[..window].iter().sum();
    result.push(sum / window as f64);

    for i in window..values.len() {
        sum = sum - values[i - window] + values[i];
        result.push(sum / window as f64);
    }

    result
}

/// Вычисляет экспоненциальное скользящее среднее
pub fn ema(values: &[f64], alpha: f64) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(values.len());
    result.push(values[0]);

    for &val in &values[1..] {
        let prev = *result.last().unwrap();
        result.push(alpha * val + (1.0 - alpha) * prev);
    }

    result
}

/// Нормализует значения в диапазон [0, 1]
pub fn normalize_minmax(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }

    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    if range < 1e-10 {
        return vec![0.5; values.len()];
    }

    values.iter().map(|v| (v - min) / range).collect()
}

/// Вычисляет процентные изменения
pub fn pct_change(values: &[f64]) -> Vec<f64> {
    if values.len() < 2 {
        return vec![];
    }

    values
        .windows(2)
        .map(|w| {
            if w[0].abs() > 1e-10 {
                (w[1] - w[0]) / w[0] * 100.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Форматирует число с разделителями тысяч
pub fn format_number(num: f64) -> String {
    let s = format!("{:.2}", num.abs());
    let (int_part, dec_part) = s.split_once('.').unwrap_or((&s, ""));

    let formatted_int: String = int_part
        .chars()
        .rev()
        .enumerate()
        .map(|(i, c)| {
            if i > 0 && i % 3 == 0 {
                format!(",{}", c)
            } else {
                c.to_string()
            }
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    let sign = if num < 0.0 { "-" } else { "" };
    format!("{}{}.{}", sign, formatted_int, dec_part)
}

/// Форматирует временную метку в читаемый вид
pub fn format_timestamp(ts: i64) -> String {
    use chrono::{TimeZone, Utc};
    let dt = Utc.timestamp_millis_opt(ts).unwrap();
    dt.format("%Y-%m-%d %H:%M:%S").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moving_average() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = moving_average(&values, 3);
        assert_eq!(ma.len(), 3);
        assert!((ma[0] - 2.0).abs() < 1e-10);
        assert!((ma[1] - 3.0).abs() < 1e-10);
        assert!((ma[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_minmax() {
        let values = vec![0.0, 50.0, 100.0];
        let normalized = normalize_minmax(&values);
        assert!((normalized[0] - 0.0).abs() < 1e-10);
        assert!((normalized[1] - 0.5).abs() < 1e-10);
        assert!((normalized[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pct_change() {
        let values = vec![100.0, 110.0, 99.0];
        let changes = pct_change(&values);
        assert!((changes[0] - 10.0).abs() < 1e-10);
        assert!((changes[1] - (-10.0)).abs() < 1e-10);
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(1234567.89), "1,234,567.89");
        assert_eq!(format_number(-1234.5), "-1,234.50");
    }
}
