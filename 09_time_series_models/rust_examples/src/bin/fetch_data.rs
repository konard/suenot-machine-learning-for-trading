//! Пример: Получение данных с Bybit
//!
//! Загружает исторические свечи для криптовалютных пар и сохраняет в CSV.
//!
//! Использование:
//! ```
//! cargo run --bin fetch_data -- --symbol BTCUSDT --interval 1h --days 30
//! ```

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::Parser;
use colored::Colorize;
use crypto_time_series::api::{BybitClient, Interval, save_candles};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Fetch cryptocurrency data from Bybit")]
struct Args {
    /// Trading symbol (e.g., BTCUSDT, ETHUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Candle interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
    #[arg(short, long, default_value = "1h")]
    interval: String,

    /// Number of days to fetch
    #[arg(short, long, default_value = "30")]
    days: u32,

    /// Output directory
    #[arg(short, long, default_value = "./data")]
    output: PathBuf,

    /// Fetch multiple symbols
    #[arg(long, value_delimiter = ',')]
    symbols: Option<Vec<String>>,
}

fn parse_interval(s: &str) -> Option<Interval> {
    match s.to_lowercase().as_str() {
        "1m" => Some(Interval::Min1),
        "5m" => Some(Interval::Min5),
        "15m" => Some(Interval::Min15),
        "30m" => Some(Interval::Min30),
        "1h" => Some(Interval::Hour1),
        "4h" => Some(Interval::Hour4),
        "1d" => Some(Interval::Day1),
        "1w" => Some(Interval::Week1),
        _ => None,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "=".repeat(60).blue());
    println!("{}", "Bybit Data Fetcher".bold().blue());
    println!("{}", "=".repeat(60).blue());

    let interval = parse_interval(&args.interval)
        .ok_or_else(|| anyhow::anyhow!("Invalid interval: {}", args.interval))?;

    // Создаём директорию для данных
    std::fs::create_dir_all(&args.output)?;

    let client = BybitClient::new();

    let symbols = args.symbols.unwrap_or_else(|| vec![args.symbol.clone()]);

    let end = Utc::now();
    let start = end - Duration::days(args.days as i64);

    for symbol in &symbols {
        println!(
            "\n{} {} ({} days, {} interval)",
            "Fetching:".green(),
            symbol.bold(),
            args.days,
            args.interval
        );

        match client.get_klines_range(symbol, interval, start, end).await {
            Ok(candles) => {
                println!(
                    "  {} {} candles",
                    "Received:".green(),
                    candles.len()
                );

                if candles.is_empty() {
                    println!("  {} No data available", "Warning:".yellow());
                    continue;
                }

                // Сохраняем в файл
                let filename = format!("{}_{}.csv", symbol, args.interval);
                let filepath = args.output.join(&filename);

                save_candles(&filepath, &candles)?;
                println!(
                    "  {} {}",
                    "Saved to:".green(),
                    filepath.display()
                );

                // Показываем статистику
                let first = candles.first().unwrap();
                let last = candles.last().unwrap();
                println!("  {} {} -> {}", "Period:".cyan(), first.timestamp, last.timestamp);
                println!(
                    "  {} open={:.2}, close={:.2}",
                    "First candle:".cyan(),
                    first.open,
                    first.close
                );
                println!(
                    "  {} open={:.2}, close={:.2}",
                    "Last candle:".cyan(),
                    last.open,
                    last.close
                );

                let change = (last.close - first.open) / first.open * 100.0;
                let change_str = if change >= 0.0 {
                    format!("+{:.2}%", change).green()
                } else {
                    format!("{:.2}%", change).red()
                };
                println!("  {} {}", "Price change:".cyan(), change_str);
            }
            Err(e) => {
                println!("  {} {}", "Error:".red(), e);
            }
        }
    }

    println!("\n{}", "Done!".green().bold());
    Ok(())
}
