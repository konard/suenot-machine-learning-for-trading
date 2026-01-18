//! Fetch Data Example
//!
//! Demonstrates how to fetch cryptocurrency data from Bybit.
//!
//! Usage:
//!     cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT --interval 60 --limit 1000

use cross_attention_multi_asset::data::{BybitClient, Candle};
use std::collections::HashMap;

#[derive(Debug)]
struct Args {
    symbols: Vec<String>,
    interval: String,
    limit: usize,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();

    let mut symbols = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string(), "SOLUSDT".to_string()];
    let mut interval = "60".to_string();
    let mut limit = 1000usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--symbols" => {
                if i + 1 < args.len() {
                    symbols = args[i + 1].split(',').map(|s| s.to_string()).collect();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--interval" => {
                if i + 1 < args.len() {
                    interval = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--limit" => {
                if i + 1 < args.len() {
                    limit = args[i + 1].parse().unwrap_or(1000);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }

    Args {
        symbols,
        interval,
        limit,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(60));
    println!("Cross-Attention Multi-Asset Trading - Data Fetch Example");
    println!("{}", "=".repeat(60));

    let args = parse_args();
    println!("\nFetching data with:");
    println!("  Symbols: {:?}", args.symbols);
    println!("  Interval: {} minutes", args.interval);
    println!("  Limit: {} candles", args.limit);

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch data for each symbol
    println!("\n{}", "-".repeat(40));
    println!("Fetching from Bybit API...");
    println!("{}", "-".repeat(40));

    let mut all_data: HashMap<String, Vec<Candle>> = HashMap::new();

    for symbol in &args.symbols {
        print!("Fetching {}... ", symbol);

        match client.fetch_klines(symbol, &args.interval, args.limit).await {
            Ok(candles) => {
                println!("OK ({} candles)", candles.len());

                if !candles.is_empty() {
                    let first = &candles[0];
                    let last = &candles[candles.len() - 1];

                    println!(
                        "  Time range: {} - {}",
                        timestamp_to_string(first.timestamp),
                        timestamp_to_string(last.timestamp)
                    );
                    println!(
                        "  Price range: {:.2} - {:.2}",
                        candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
                        candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max)
                    );
                    println!(
                        "  Avg volume: {:.2}",
                        candles.iter().map(|c| c.volume).sum::<f64>() / candles.len() as f64
                    );
                }

                all_data.insert(symbol.clone(), candles);
            }
            Err(e) => {
                println!("ERROR: {}", e);
            }
        }
    }

    // Summary
    println!("\n{}", "-".repeat(40));
    println!("Data Summary");
    println!("{}", "-".repeat(40));

    for (symbol, candles) in &all_data {
        if !candles.is_empty() {
            // Calculate some basic statistics
            let returns: Vec<f64> = candles
                .windows(2)
                .map(|w| (w[1].close - w[0].close) / w[0].close)
                .collect();

            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
                / returns.len() as f64;
            let std_dev = variance.sqrt();

            println!(
                "{}: {} candles, mean return: {:.4}%, std: {:.4}%",
                symbol,
                candles.len(),
                mean_return * 100.0,
                std_dev * 100.0
            );
        }
    }

    // Calculate correlations between assets
    if all_data.len() >= 2 {
        println!("\n{}", "-".repeat(40));
        println!("Return Correlations");
        println!("{}", "-".repeat(40));

        let symbols: Vec<&String> = all_data.keys().collect();
        let returns: HashMap<&String, Vec<f64>> = all_data
            .iter()
            .map(|(sym, candles)| {
                let r: Vec<f64> = candles
                    .windows(2)
                    .map(|w| (w[1].close - w[0].close) / w[0].close)
                    .collect();
                (sym, r)
            })
            .collect();

        // Print correlation header
        print!("{:>12}", "");
        for sym in &symbols {
            print!("{:>12}", sym);
        }
        println!();

        for sym1 in &symbols {
            print!("{:>12}", sym1);
            for sym2 in &symbols {
                if sym1 == sym2 {
                    print!("{:>12}", "1.000");
                } else {
                    let r1 = &returns[sym1];
                    let r2 = &returns[sym2];
                    let corr = calculate_correlation(r1, r2);
                    print!("{:>12.3}", corr);
                }
            }
            println!();
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("Data fetch complete!");
    println!("{}", "=".repeat(60));

    Ok(())
}

fn timestamp_to_string(ts: i64) -> String {
    // Simple timestamp formatting (milliseconds to date string)
    let seconds = ts / 1000;
    let days = seconds / 86400;
    let hours = (seconds % 86400) / 3600;
    let mins = (seconds % 3600) / 60;

    // Approximate date from Unix epoch
    let year = 1970 + (days / 365);
    let day_of_year = days % 365;

    format!("{}-{:03} {:02}:{:02}", year, day_of_year, hours, mins)
}

fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x > 0.0 && var_y > 0.0 {
        cov / (var_x.sqrt() * var_y.sqrt())
    } else {
        0.0
    }
}
