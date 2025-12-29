//! Example: Fetch market data from Bybit
//!
//! This example demonstrates how to use the BybitClient to fetch
//! OHLCV (candlestick) data for cryptocurrency pairs.
//!
//! Run with:
//! ```bash
//! cargo run --example fetch_data -- --symbol BTCUSDT --interval 1h --limit 100
//! ```

use anyhow::Result;
use clap::Parser;
use earnings_crypto::api::BybitClient;

#[derive(Parser, Debug)]
#[command(name = "fetch_data")]
#[command(about = "Fetch market data from Bybit exchange")]
struct Args {
    /// Trading pair symbol (e.g., BTCUSDT, ETHUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
    #[arg(short, long, default_value = "1h")]
    interval: String,

    /// Number of candles to fetch (max 1000)
    #[arg(short, long, default_value = "100")]
    limit: usize,

    /// Show order book data
    #[arg(long)]
    orderbook: bool,

    /// Show recent trades
    #[arg(long)]
    trades: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== Bybit Data Fetcher ===\n");

    let client = BybitClient::new();

    // Fetch klines
    println!(
        "Fetching {} {} candles for {}...\n",
        args.limit, args.interval, args.symbol
    );

    let candles = client
        .get_klines(&args.symbol, &args.interval, args.limit)
        .await?;

    println!("Received {} candles\n", candles.len());

    // Display header
    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12} {:>15} {:>10}",
        "Time", "Open", "High", "Low", "Close", "Volume", "Return %"
    );
    println!("{}", "-".repeat(105));

    // Display last 10 candles
    for candle in candles.iter().rev().take(10) {
        println!(
            "{:<20} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2} {:>10.2}%",
            candle.datetime().format("%Y-%m-%d %H:%M"),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
            candle.return_pct() * 100.0
        );
    }

    if candles.len() > 10 {
        println!("\n... and {} more candles", candles.len() - 10);
    }

    // Calculate basic statistics
    if !candles.is_empty() {
        let returns: Vec<f64> = candles.iter().map(|c| c.return_pct()).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

        let avg_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let avg_volume: f64 = volumes.iter().sum::<f64>() / volumes.len() as f64;

        let return_std: f64 = {
            let variance: f64 = returns
                .iter()
                .map(|r| (r - avg_return).powi(2))
                .sum::<f64>()
                / (returns.len() - 1) as f64;
            variance.sqrt()
        };

        let max_return = returns.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_return = returns.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        println!("\n=== Statistics ===");
        println!("Average Return: {:.4}%", avg_return * 100.0);
        println!("Return Std Dev: {:.4}%", return_std * 100.0);
        println!("Max Return: {:.4}%", max_return * 100.0);
        println!("Min Return: {:.4}%", min_return * 100.0);
        println!("Average Volume: {:.2}", avg_volume);
        println!(
            "Price Range: {:.2} - {:.2}",
            candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
            candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max)
        );
    }

    // Optionally fetch order book
    if args.orderbook {
        println!("\n=== Order Book (Top 5) ===\n");
        let orderbook = client.get_orderbook(&args.symbol, 5).await?;

        println!("{:>15} {:>15} | {:>15} {:>15}", "Bid Price", "Bid Qty", "Ask Price", "Ask Qty");
        println!("{}", "-".repeat(65));

        for i in 0..5 {
            let bid = orderbook.bids.get(i);
            let ask = orderbook.asks.get(i);

            println!(
                "{:>15.2} {:>15.4} | {:>15.2} {:>15.4}",
                bid.map(|b| b.price).unwrap_or(0.0),
                bid.map(|b| b.quantity).unwrap_or(0.0),
                ask.map(|a| a.price).unwrap_or(0.0),
                ask.map(|a| a.quantity).unwrap_or(0.0),
            );
        }

        if let Some(spread) = orderbook.spread_pct() {
            println!("\nSpread: {:.4}%", spread);
        }
        if let Some(mid) = orderbook.mid_price() {
            println!("Mid Price: {:.2}", mid);
        }
        println!("Imbalance (5 levels): {:.4}", orderbook.imbalance(5));
    }

    // Optionally fetch recent trades
    if args.trades {
        println!("\n=== Recent Trades (Last 10) ===\n");
        let trades = client.get_recent_trades(&args.symbol, 10).await?;

        println!(
            "{:<20} {:>12} {:>12} {:>8}",
            "Time", "Price", "Quantity", "Side"
        );
        println!("{}", "-".repeat(55));

        for trade in trades.iter().take(10) {
            println!(
                "{:<20} {:>12.2} {:>12.4} {:>8?}",
                trade.datetime().format("%Y-%m-%d %H:%M:%S"),
                trade.price,
                trade.quantity,
                trade.side
            );
        }
    }

    println!("\nDone!");
    Ok(())
}
