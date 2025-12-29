//! Загрузка данных с Bybit
//!
//! Использование:
//! ```bash
//! cargo run --bin fetch_data -- BTCUSDT 180
//! cargo run --bin fetch_data -- ETHUSDT 365 --interval 60
//! ```

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::Parser;
use rust_optimal_execution::api::{BybitClient, Interval};
use rust_optimal_execution::utils::save_candles_csv;
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser, Debug)]
#[command(author, version, about = "Fetch historical data from Bybit")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT)
    #[arg(default_value = "BTCUSDT")]
    symbol: String,

    /// Number of days to fetch
    #[arg(default_value = "180")]
    days: i64,

    /// Interval in minutes (1, 5, 15, 60, 240)
    #[arg(long, default_value = "60")]
    interval: u32,

    /// Output directory
    #[arg(long, default_value = "data")]
    output_dir: PathBuf,

    /// Use testnet
    #[arg(long)]
    testnet: bool,
}

fn parse_interval(minutes: u32) -> Interval {
    match minutes {
        1 => Interval::Min1,
        3 => Interval::Min3,
        5 => Interval::Min5,
        15 => Interval::Min15,
        30 => Interval::Min30,
        60 => Interval::Hour1,
        120 => Interval::Hour2,
        240 => Interval::Hour4,
        _ => Interval::Hour1,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Инициализация логирования
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Args::parse();

    info!("Fetching {} data for {} days", args.symbol, args.days);

    // Создаём клиент
    let client = if args.testnet {
        BybitClient::testnet()
    } else {
        BybitClient::new()
    };

    // Определяем период
    let end = Utc::now();
    let start = end - Duration::days(args.days);
    let interval = parse_interval(args.interval);

    info!(
        "Period: {} to {} (interval: {})",
        start.format("%Y-%m-%d"),
        end.format("%Y-%m-%d"),
        interval
    );

    // Загружаем данные
    let candles = client
        .get_all_klines(
            &args.symbol,
            interval,
            start.timestamp_millis(),
            end.timestamp_millis(),
        )
        .await?;

    info!("Fetched {} candles", candles.len());

    if candles.is_empty() {
        info!("No data received");
        return Ok(());
    }

    // Создаём директорию если не существует
    std::fs::create_dir_all(&args.output_dir)?;

    // Сохраняем в CSV
    let filename = format!("{}_{}_{}d.csv", args.symbol, args.interval, args.days);
    let output_path = args.output_dir.join(&filename);

    save_candles_csv(&candles, &output_path)?;
    info!("Saved to {:?}", output_path);

    // Выводим статистику
    if let (Some(first), Some(last)) = (candles.first(), candles.last()) {
        info!("First candle: {} @ {:.2}", first.datetime(), first.close);
        info!("Last candle: {} @ {:.2}", last.datetime(), last.close);

        let price_change = (last.close - first.close) / first.close * 100.0;
        info!("Price change: {:.2}%", price_change);

        let total_volume: f64 = candles.iter().map(|c| c.volume).sum();
        info!("Total volume: {:.2}", total_volume);
    }

    Ok(())
}
