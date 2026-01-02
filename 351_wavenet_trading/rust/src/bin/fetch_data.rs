//! Загрузка данных с Bybit
//!
//! Использование:
//! ```
//! cargo run --bin fetch_data -- --symbol BTCUSDT --interval 1h --days 30
//! ```

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::Parser;
use colored::Colorize;
use wavenet_trading::api::{BybitClient, Interval, save_candles};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Fetch cryptocurrency data from Bybit for WaveNet trading")]
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

    /// Fetch multiple symbols (comma-separated)
    #[arg(long, value_delimiter = ',')]
    symbols: Option<Vec<String>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "=".repeat(60).blue());
    println!("{}", "WaveNet Trading - Bybit Data Fetcher".bold().blue());
    println!("{}", "=".repeat(60).blue());

    let interval = Interval::from_str(&args.interval)
        .ok_or_else(|| anyhow::anyhow!("Invalid interval: {}. Use: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w", args.interval))?;

    // Создаём директорию для данных
    std::fs::create_dir_all(&args.output)?;

    let client = BybitClient::new();
    let symbols = args.symbols.unwrap_or_else(|| vec![args.symbol.clone()]);

    let end = Utc::now();
    let start = end - Duration::days(args.days as i64);

    println!("\n{} Fetching data from {} to {}",
        "Period:".cyan(),
        start.format("%Y-%m-%d"),
        end.format("%Y-%m-%d")
    );

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
                    "  {} {} candles received",
                    "✓".green(),
                    candles.len()
                );

                if candles.is_empty() {
                    println!("  {} No data available for this symbol", "⚠".yellow());
                    continue;
                }

                // Сохраняем в файл
                let filename = format!("{}_{}.csv", symbol, args.interval);
                let filepath = args.output.join(&filename);

                save_candles(&filepath, &candles)?;
                println!(
                    "  {} Saved to: {}",
                    "✓".green(),
                    filepath.display()
                );

                // Показываем статистику
                let first = candles.first().unwrap();
                let last = candles.last().unwrap();

                println!("\n  {} Data Summary", "📊".cyan());
                println!("  ├─ First: {} (${:.2})", first.timestamp.format("%Y-%m-%d %H:%M"), first.close);
                println!("  ├─ Last:  {} (${:.2})", last.timestamp.format("%Y-%m-%d %H:%M"), last.close);

                let change = (last.close - first.close) / first.close * 100.0;
                let change_str = if change >= 0.0 {
                    format!("+{:.2}%", change).green()
                } else {
                    format!("{:.2}%", change).red()
                };
                println!("  ├─ Change: {}", change_str);

                // Волатильность
                let returns: Vec<f64> = candles.windows(2)
                    .map(|w| (w[1].close - w[0].close) / w[0].close)
                    .collect();
                let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
                let volatility = variance.sqrt() * (365.0 * 24.0 / interval.minutes() as f64).sqrt() * 100.0;

                println!("  └─ Annualized Volatility: {:.1}%", volatility);
            }
            Err(e) => {
                println!("  {} Error: {}", "✗".red(), e);
            }
        }
    }

    println!("\n{}", "Done!".green().bold());
    println!("Data saved to: {}", args.output.display());

    Ok(())
}
