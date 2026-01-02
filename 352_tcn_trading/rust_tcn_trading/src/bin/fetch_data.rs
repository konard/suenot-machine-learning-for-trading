//! Fetch cryptocurrency data from Bybit
//!
//! Usage:
//!     cargo run --bin fetch_data -- --symbol BTCUSDT --interval 1h --limit 1000

use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use rust_tcn_trading::api::{BybitClient, TimeFrame};
use std::fs::File;
use std::io::Write;

/// Fetch market data from Bybit exchange
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Trading pair symbol (e.g., BTCUSDT, ETHUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Time interval (1m, 5m, 15m, 1h, 4h, 1d)
    #[arg(short, long, default_value = "1h")]
    interval: String,

    /// Number of candles to fetch (max 1000)
    #[arg(short, long, default_value_t = 1000)]
    limit: u32,

    /// Output file path (CSV format)
    #[arg(short, long)]
    output: Option<String>,

    /// Use testnet instead of mainnet
    #[arg(long, default_value_t = false)]
    testnet: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    // Parse timeframe
    let timeframe = TimeFrame::from_str(&args.interval)
        .ok_or_else(|| anyhow::anyhow!("Invalid interval: {}", args.interval))?;

    // Create client
    let client = if args.testnet {
        BybitClient::testnet()
    } else {
        BybitClient::new()
    };

    println!("Fetching {} candles for {} with {} interval...",
             args.limit, args.symbol, args.interval);

    // Fetch data
    let data = client
        .get_klines(&args.symbol, timeframe, Some(args.limit), None, None)
        .await?;

    println!("Fetched {} candles", data.len());

    if data.is_empty() {
        println!("No data returned. Check symbol and interval.");
        return Ok(());
    }

    // Print summary
    let first = data.candles.first().unwrap();
    let last = data.candles.last().unwrap();

    println!("\nData Summary:");
    println!("  Symbol:     {}", data.symbol);
    println!("  From:       {}", first.timestamp);
    println!("  To:         {}", last.timestamp);
    println!("  First close: ${:.2}", first.close);
    println!("  Last close:  ${:.2}", last.close);
    println!("  Change:      {:.2}%", (last.close / first.close - 1.0) * 100.0);

    // Calculate some basic stats
    let closes = data.closes();
    let min_price = closes.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_price = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let avg_price = closes.iter().sum::<f64>() / closes.len() as f64;

    println!("  Min price:   ${:.2}", min_price);
    println!("  Max price:   ${:.2}", max_price);
    println!("  Avg price:   ${:.2}", avg_price);

    // Calculate returns
    let returns = data.log_returns();
    let avg_return = if !returns.is_empty() {
        returns.iter().sum::<f64>() / returns.len() as f64
    } else {
        0.0
    };
    let volatility = if returns.len() > 1 {
        let variance = returns.iter()
            .map(|r| (r - avg_return).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };

    println!("  Avg return:  {:.4}%", avg_return * 100.0);
    println!("  Volatility:  {:.4}%", volatility * 100.0);

    // Save to file if output specified
    if let Some(output_path) = args.output {
        let mut file = File::create(&output_path)?;

        // Write header
        writeln!(file, "timestamp,open,high,low,close,volume,turnover")?;

        // Write data
        for candle in &data.candles {
            writeln!(
                file,
                "{},{},{},{},{},{},{}",
                candle.timestamp.format("%Y-%m-%d %H:%M:%S"),
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
                candle.turnover
            )?;
        }

        println!("\nData saved to: {}", output_path);
    }

    // Fetch additional market info
    println!("\nFetching current ticker...");
    let ticker = client.get_ticker(&args.symbol).await?;
    println!("  Current price: ${:.2}", ticker.last_price);
    println!("  24h high:      ${:.2}", ticker.high_24h);
    println!("  24h low:       ${:.2}", ticker.low_24h);
    println!("  24h change:    {:.2}%", ticker.price_change_pct_24h);
    println!("  24h volume:    {:.2}", ticker.volume_24h);

    println!("\nFetching order book...");
    let orderbook = client.get_orderbook(&args.symbol, Some(5)).await?;
    if let (Some(bid), Some(ask)) = (orderbook.best_bid(), orderbook.best_ask()) {
        println!("  Best bid:   ${:.2}", bid);
        println!("  Best ask:   ${:.2}", ask);
        println!("  Spread:     ${:.4} ({:.4}%)",
                 ask - bid, (ask - bid) / ((bid + ask) / 2.0) * 100.0);
    }

    println!("\nDone! Timestamp: {}", Utc::now());

    Ok(())
}
