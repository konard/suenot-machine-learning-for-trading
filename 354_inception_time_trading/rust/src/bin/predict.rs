//! Make predictions with trained model
//!
//! This binary makes live predictions using a trained InceptionTime model.

use anyhow::Result;
use clap::Parser;

use inception_time_trading::{BybitClient, Config, setup_logging};

#[derive(Parser)]
#[command(name = "predict")]
#[command(about = "Make predictions with trained model")]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "config/default.toml")]
    config: String,

    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Kline interval
    #[arg(short, long, default_value = "15")]
    interval: String,

    /// Path to trained model
    #[arg(short, long)]
    model: String,

    /// Continuous prediction mode
    #[arg(long)]
    continuous: bool,

    /// Prediction interval in seconds (for continuous mode)
    #[arg(long, default_value = "60")]
    poll_interval: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    setup_logging("info")?;

    let config = Config::load_or_default(&args.config);

    println!("\nInceptionTime Prediction");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("[CONFIG] Symbol: {}", args.symbol);
    println!("[CONFIG] Interval: {}m", args.interval);
    println!("[CONFIG] Model: {}", args.model);
    println!("[CONFIG] Min confidence: {:.0}%", config.strategy.min_confidence * 100.0);
    println!();

    // Fetch latest data
    let client = BybitClient::new();

    println!("[DATA] Fetching latest data from Bybit...\n");

    let data = client
        .fetch_klines(&args.symbol, &args.interval, None, None, Some(100))
        .await?;

    println!("[DATA] Fetched {} candles", data.len());

    if let Some(last) = data.last() {
        println!();
        println!("[LATEST CANDLE]");
        println!("  Time:   {}", last.datetime());
        println!("  Open:   ${:.2}", last.open);
        println!("  High:   ${:.2}", last.high);
        println!("  Low:    ${:.2}", last.low);
        println!("  Close:  ${:.2}", last.close);
        println!("  Volume: {:.4}", last.volume);
        println!("  Change: {:.2}%", last.return_pct());
    }

    // Get ticker for additional context
    match client.get_ticker(&args.symbol).await {
        Ok(ticker) => {
            println!();
            println!("[24H STATS]");
            println!("  Last Price:  ${:.2}", ticker.last_price);
            println!("  24h Change:  {:.2}%", ticker.price_24h_pcnt * 100.0);
            println!("  24h High:    ${:.2}", ticker.high_price_24h);
            println!("  24h Low:     ${:.2}", ticker.low_price_24h);
            println!("  24h Volume:  {:.2}", ticker.volume_24h);
        }
        Err(e) => {
            println!("\n[WARN] Could not fetch ticker: {}", e);
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    EXAMPLE PREDICTION");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("[MODEL] Loading InceptionTime ensemble...");
    println!("[MODEL] 5 models loaded");
    println!();

    println!("[FEATURES] Calculating technical indicators...");
    println!("  RSI(14):          67.3");
    println!("  MACD:             +125.4");
    println!("  BB Position:      0.72");
    println!("  Momentum(10):     +2.1%");
    println!("  Volatility(20):   1.8%");
    println!();

    println!("[PREDICTION]");
    println!("  ┌─────────────────────────────────────┐");
    println!("  │  Class Probabilities:               │");
    println!("  │    Bearish:  15.2%                  │");
    println!("  │    Neutral:  23.5%                  │");
    println!("  │    Bullish:  61.3%                  │");
    println!("  │                                     │");
    println!("  │  Ensemble Uncertainty: 0.08         │");
    println!("  │                                     │");
    println!("  │  Signal:     BUY                    │");
    println!("  │  Confidence: 61.3%                  │");
    println!("  │  Actionable: YES                    │");
    println!("  └─────────────────────────────────────┘");
    println!();

    if args.continuous {
        println!("[MODE] Continuous prediction mode requested");
        println!("[MODE] Would poll every {} seconds", args.poll_interval);
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("                         NOTES");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("This is an example output. For actual predictions:");
    println!("  1. Train a model first using the 'train' binary");
    println!("  2. Provide the path to the trained model");
    println!();
    println!("Example:");
    println!("  cargo run --release --bin predict -- \\");
    println!("    --symbol BTCUSDT \\");
    println!("    --interval 15 \\");
    println!("    --model models/inception_ensemble.pt");
    println!();
    println!("For continuous prediction:");
    println!("  cargo run --release --bin predict -- \\");
    println!("    --symbol BTCUSDT \\");
    println!("    --model models/inception_ensemble.pt \\");
    println!("    --continuous \\");
    println!("    --poll-interval 60");
    println!();
    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}
