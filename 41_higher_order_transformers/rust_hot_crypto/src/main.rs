//! HOT Crypto CLI
//!
//! Command-line interface for the Higher Order Transformer crypto trading library

use anyhow::Result;
use clap::{Parser, Subcommand};

use hot_crypto::{
    data::BybitClient,
    get_crypto_universe,
    model::HOTPredictor,
    strategy::SignalGenerator,
    utils::Config,
};

#[derive(Parser)]
#[command(name = "hot-crypto")]
#[command(about = "Higher Order Transformer for crypto trading", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch current prices from Bybit
    Prices {
        /// Symbols to fetch (default: all universe)
        #[arg(short, long)]
        symbols: Option<Vec<String>>,
    },

    /// Generate trading signals
    Signals {
        /// Number of top signals to show
        #[arg(short, long, default_value = "5")]
        top_n: usize,

        /// Minimum confidence threshold
        #[arg(short, long, default_value = "0.6")]
        min_confidence: f64,
    },

    /// Predict price movement for a symbol
    Predict {
        /// Symbol to predict
        #[arg(short, long)]
        symbol: String,

        /// Lookback period in days
        #[arg(short, long, default_value = "60")]
        lookback: u32,
    },

    /// Generate or display configuration
    Config {
        /// Output file path
        #[arg(short, long)]
        output: Option<String>,

        /// Configuration preset (default, conservative, aggressive)
        #[arg(short, long, default_value = "default")]
        preset: String,
    },

    /// Show library information
    Info,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Prices { symbols } => {
            cmd_prices(symbols).await?;
        }
        Commands::Signals { top_n, min_confidence } => {
            cmd_signals(top_n, min_confidence).await?;
        }
        Commands::Predict { symbol, lookback } => {
            cmd_predict(&symbol, lookback).await?;
        }
        Commands::Config { output, preset } => {
            cmd_config(output, &preset)?;
        }
        Commands::Info => {
            cmd_info();
        }
    }

    Ok(())
}

/// Fetch and display current prices
async fn cmd_prices(symbols: Option<Vec<String>>) -> Result<()> {
    let client = BybitClient::new();

    let symbols: Vec<String> = symbols.unwrap_or_else(|| {
        get_crypto_universe()
            .iter()
            .map(|s| s.to_string())
            .collect()
    });

    println!("\n{:-<60}", "");
    println!(" Current Prices (Bybit)");
    println!("{:-<60}", "");
    println!("{:<12} {:>15} {:>15}", "Symbol", "Price", "24h Volume");
    println!("{:-<60}", "");

    for symbol in &symbols {
        match client.get_ticker(symbol).await {
            Ok((price, volume)) => {
                println!(
                    "{:<12} {:>15.2} {:>15.0}",
                    symbol, price, volume
                );
            }
            Err(e) => {
                println!("{:<12} Error: {}", symbol, e);
            }
        }
        // Small delay to avoid rate limiting
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    println!("{:-<60}\n", "");

    Ok(())
}

/// Generate and display trading signals
async fn cmd_signals(top_n: usize, min_confidence: f64) -> Result<()> {
    let client = BybitClient::new();
    let predictor = HOTPredictor::default_crypto();
    let mut generator = SignalGenerator::new();
    generator.config.min_confidence = min_confidence;

    let universe = get_crypto_universe();

    println!("\nFetching data and generating signals...\n");

    // Fetch data for all symbols
    let series_list = client.get_multi_klines(&universe, "D", Some(90)).await?;

    // Generate predictions
    let predictions = predictor.predict_batch(&series_list);

    // Generate signals
    let signals = generator.generate_batch(&predictions);

    // Sort by confidence
    let mut sorted_signals: Vec<_> = signals.iter().collect();
    sorted_signals.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    println!("{:-<70}", "");
    println!(" Trading Signals (min confidence: {:.0}%)", min_confidence * 100.0);
    println!("{:-<70}", "");
    println!(
        "{:<12} {:>8} {:>12} {:>10} {:>25}",
        "Symbol", "Signal", "Confidence", "Position", "Reason"
    );
    println!("{:-<70}", "");

    for signal in sorted_signals.iter().take(top_n) {
        println!(
            "{:<12} {:>8} {:>11.1}% {:>9.1}%  {}",
            signal.symbol,
            signal.signal.as_str(),
            signal.confidence * 100.0,
            signal.position_size * 100.0,
            if signal.reason.len() > 25 {
                &signal.reason[..25]
            } else {
                &signal.reason
            }
        );
    }

    println!("{:-<70}\n", "");

    // Summary
    let buy_count = signals.iter().filter(|s| s.signal == hot_crypto::strategy::Signal::Buy).count();
    let sell_count = signals.iter().filter(|s| s.signal == hot_crypto::strategy::Signal::Sell).count();
    let hold_count = signals.iter().filter(|s| s.signal == hot_crypto::strategy::Signal::Hold).count();

    println!("Summary: {} BUY, {} SELL, {} HOLD\n", buy_count, sell_count, hold_count);

    Ok(())
}

/// Predict price movement for a single symbol
async fn cmd_predict(symbol: &str, lookback: u32) -> Result<()> {
    let client = BybitClient::new();
    let predictor = HOTPredictor::default_crypto();

    println!("\nFetching {} data...", symbol);

    let series = client.get_klines(symbol, "D", None, None, Some(lookback + 30)).await?;

    println!("Generating prediction...\n");

    let prediction = predictor.predict_from_series(&series);

    println!("{:-<50}", "");
    println!(" Prediction for {}", symbol);
    println!("{:-<50}", "");
    println!("Direction:   {}", prediction.prediction.as_str());
    println!("Confidence:  {:.1}%", prediction.confidence * 100.0);
    println!();
    println!("Probabilities:");
    println!("  DOWN:     {:.1}%", prediction.probabilities[0] * 100.0);
    println!("  NEUTRAL:  {:.1}%", prediction.probabilities[1] * 100.0);
    println!("  UP:       {:.1}%", prediction.probabilities[2] * 100.0);
    println!("{:-<50}\n", "");

    // Trading recommendation
    let generator = SignalGenerator::new();
    let signal = generator.generate(&prediction);

    println!("Recommendation: {} (position size: {:.1}%)",
             signal.signal.as_str(),
             signal.position_size * 100.0);
    println!("Reason: {}\n", signal.reason);

    Ok(())
}

/// Generate or display configuration
fn cmd_config(output: Option<String>, preset: &str) -> Result<()> {
    let config = match preset {
        "conservative" => Config::conservative(),
        "aggressive" => Config::aggressive(),
        _ => Config::default(),
    };

    if let Some(path) = output {
        config.to_file(&path)?;
        println!("Configuration saved to: {}", path);
    } else {
        let json = serde_json::to_string_pretty(&config)?;
        println!("{}", json);
    }

    Ok(())
}

/// Display library information
fn cmd_info() {
    println!("\n{:-<50}", "");
    println!(" HOT Crypto - Higher Order Transformer");
    println!("{:-<50}", "");
    println!("Version:     {}", hot_crypto::VERSION);
    println!("Features:");
    println!("  - Higher-order attention mechanism");
    println!("  - CP tensor decomposition");
    println!("  - Kernel attention for efficiency");
    println!("  - Bybit API integration");
    println!();
    println!("Universe ({} assets):", get_crypto_universe().len());
    for symbol in get_crypto_universe() {
        println!("  - {}", symbol);
    }
    println!("{:-<50}\n", "");
}
