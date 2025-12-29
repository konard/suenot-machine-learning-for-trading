//! Загрузка исторических данных с Bybit
//!
//! Пример использования:
//! ```bash
//! cargo run --bin fetch_data -- --symbol BTCUSDT --interval 15 --days 30
//! ```

use anyhow::Result;
use chrono::{Duration, Utc};
use cnn_crypto_trading::bybit::{BybitClient, KlineInterval};
use std::path::Path;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Настройка логирования
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Параметры (в реальном приложении использовать clap для парсинга аргументов)
    let symbol = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "BTCUSDT".to_string());
    let interval_str = std::env::args()
        .nth(4)
        .unwrap_or_else(|| "15".to_string());
    let days: i64 = std::env::args()
        .nth(6)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    let interval: KlineInterval = interval_str.parse()?;

    info!("Fetching {} data for {} days", symbol, days);

    // Вычисляем временной диапазон
    let end_time = Utc::now();
    let start_time = end_time - Duration::days(days);

    info!(
        "Time range: {} to {}",
        start_time.format("%Y-%m-%d %H:%M"),
        end_time.format("%Y-%m-%d %H:%M")
    );

    // Создаём клиент и загружаем данные
    let client = BybitClient::new();
    let klines = client
        .get_historical_klines(
            &symbol,
            interval,
            start_time.timestamp_millis(),
            Some(end_time.timestamp_millis()),
        )
        .await?;

    info!("Fetched {} klines", klines.len());

    if klines.is_empty() {
        info!("No data received");
        return Ok(());
    }

    // Сохраняем в CSV
    let output_dir = Path::new("data");
    std::fs::create_dir_all(output_dir)?;

    let output_path = output_dir.join(format!("{}_{}_{}d.csv", symbol, interval_str, days));
    let mut writer = csv::Writer::from_path(&output_path)?;

    // Заголовки
    writer.write_record(&[
        "timestamp",
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
    ])?;

    // Данные
    for kline in &klines {
        writer.write_record(&[
            kline.timestamp.to_string(),
            kline.datetime().format("%Y-%m-%d %H:%M:%S").to_string(),
            kline.open.to_string(),
            kline.high.to_string(),
            kline.low.to_string(),
            kline.close.to_string(),
            kline.volume.to_string(),
            kline.turnover.to_string(),
        ])?;
    }

    writer.flush()?;

    info!("Data saved to {:?}", output_path);

    // Выводим статистику
    let first = klines.first().unwrap();
    let last = klines.last().unwrap();

    println!("\n=== Data Summary ===");
    println!("Symbol: {}", symbol);
    println!("Interval: {}", interval_str);
    println!("Total klines: {}", klines.len());
    println!(
        "Date range: {} to {}",
        first.datetime().format("%Y-%m-%d %H:%M"),
        last.datetime().format("%Y-%m-%d %H:%M")
    );
    println!(
        "Price range: ${:.2} - ${:.2}",
        klines.iter().map(|k| k.low).fold(f64::INFINITY, f64::min),
        klines.iter().map(|k| k.high).fold(0.0, f64::max)
    );
    println!(
        "Latest close: ${:.2}",
        last.close
    );

    Ok(())
}
