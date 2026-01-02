//! Main entry point for InceptionTime Trading System
//!
//! This binary provides a unified CLI interface to all functionality:
//! - Data fetching from Bybit
//! - Model training
//! - Backtesting
//! - Live prediction

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::info;

use inception_time_trading::{
    utils::{Config, setup_logging},
    BybitClient,
};

#[derive(Parser)]
#[command(name = "inception_trading")]
#[command(author = "ML Trading Examples")]
#[command(version = "0.1.0")]
#[command(about = "InceptionTime for cryptocurrency trading", long_about = None)]
struct Cli {
    /// Path to configuration file
    #[arg(short, long, default_value = "config/default.toml")]
    config: String,

    /// Verbosity level
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch data from Bybit
    Fetch {
        /// Trading symbol (e.g., BTCUSDT)
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
        #[arg(short, long, default_value = "15")]
        interval: String,

        /// Number of days of history
        #[arg(short, long, default_value = "90")]
        days: u32,

        /// Output file path
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Train InceptionTime ensemble
    Train {
        /// Path to training data
        #[arg(short, long)]
        data: Option<String>,

        /// Number of epochs
        #[arg(short, long)]
        epochs: Option<u32>,

        /// Output model path
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Run backtest
    Backtest {
        /// Path to model file
        #[arg(short, long)]
        model: String,

        /// Path to test data
        #[arg(short, long)]
        data: String,

        /// Initial capital
        #[arg(short, long, default_value = "100000.0")]
        capital: f64,
    },

    /// Make live predictions
    Predict {
        /// Trading symbol
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Interval
        #[arg(short, long, default_value = "15")]
        interval: String,

        /// Path to model file
        #[arg(short, long)]
        model: String,
    },

    /// Show system information
    Info,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging based on verbosity
    let log_level = match cli.verbose {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    setup_logging(log_level)?;

    // Load configuration
    let config = Config::load(&cli.config)?;
    info!("Loaded configuration from {}", cli.config);

    match cli.command {
        Commands::Fetch {
            symbol,
            interval,
            days,
            output,
        } => {
            fetch_data(&symbol, &interval, days, output).await?;
        }

        Commands::Train {
            data,
            epochs,
            output,
        } => {
            train_model(&config, data, epochs, output).await?;
        }

        Commands::Backtest {
            model,
            data,
            capital,
        } => {
            run_backtest(&config, &model, &data, capital).await?;
        }

        Commands::Predict {
            symbol,
            interval,
            model,
        } => {
            run_prediction(&symbol, &interval, &model).await?;
        }

        Commands::Info => {
            show_info(&config);
        }
    }

    Ok(())
}

async fn fetch_data(
    symbol: &str,
    interval: &str,
    days: u32,
    output: Option<String>,
) -> Result<()> {
    info!("Fetching {} {} data for {} days", symbol, interval, days);

    let client = BybitClient::new();

    // Calculate time range
    let end_time = chrono::Utc::now().timestamp_millis();
    let start_time = end_time - (days as i64 * 24 * 60 * 60 * 1000);

    let dataset = client
        .fetch_historical_klines(symbol, interval, start_time, end_time)
        .await?;

    info!("Fetched {} candles", dataset.len());

    // Save to file
    let output_path = output.unwrap_or_else(|| format!("data/{}_{}.csv", symbol.to_lowercase(), interval));
    dataset.to_csv(&output_path)?;
    info!("Saved data to {}", output_path);

    Ok(())
}

async fn train_model(
    config: &Config,
    data_path: Option<String>,
    epochs: Option<u32>,
    output: Option<String>,
) -> Result<()> {
    info!("Training InceptionTime ensemble");

    let epochs = epochs.unwrap_or(config.training.epochs);
    info!("Training for {} epochs", epochs);

    // Training implementation would go here
    // For demonstration, we show the expected workflow
    println!("\nInceptionTime Training");
    println!("═══════════════════════════════════════════════════════════════\n");
    println!("[CONFIG] Ensemble size: {}", config.model.ensemble_size);
    println!("[CONFIG] Depth: {}", config.model.depth);
    println!("[CONFIG] Kernel sizes: {:?}", config.model.kernel_sizes);
    println!("[CONFIG] Epochs: {}", epochs);
    println!("\n[NOTE] Full training requires libtorch installation.");
    println!("[NOTE] Run with LIBTORCH=/path/to/libtorch cargo run --bin train\n");

    if let Some(path) = &data_path {
        println!("[DATA] Would load training data from: {}", path);
    }

    if let Some(path) = &output {
        println!("[OUTPUT] Would save model to: {}", path);
    }

    Ok(())
}

async fn run_backtest(
    config: &Config,
    model_path: &str,
    data_path: &str,
    capital: f64,
) -> Result<()> {
    info!("Running backtest");

    println!("\nInceptionTime Backtest");
    println!("═══════════════════════════════════════════════════════════════\n");
    println!("[MODEL] Loading from: {}", model_path);
    println!("[DATA] Loading from: {}", data_path);
    println!("[CAPITAL] Initial: ${:.2}", capital);
    println!("\n[CONFIG] Commission: {:.2}%", config.backtest.commission_rate * 100.0);
    println!("[CONFIG] Slippage: {:.2}%", config.backtest.slippage_rate * 100.0);
    println!("[CONFIG] Min confidence: {:.2}", config.strategy.min_confidence);

    println!("\n[NOTE] Backtest engine demonstration.");
    println!("[NOTE] Full backtesting requires trained model.\n");

    // Display example results format
    println!("═══════════════════════════════════════════════════════════════");
    println!("                      BACKTEST RESULTS (Example)");
    println!("═══════════════════════════════════════════════════════════════\n");
    println!("Total Return:       +34.21%");
    println!("Sharpe Ratio:       1.87");
    println!("Sortino Ratio:      2.43");
    println!("Max Drawdown:       -8.34%");
    println!("Win Rate:           58.3%");
    println!("Profit Factor:      1.72");
    println!("Total Trades:       234\n");
    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}

async fn run_prediction(symbol: &str, interval: &str, model_path: &str) -> Result<()> {
    info!("Making predictions for {} {}", symbol, interval);

    let client = BybitClient::new();

    // Fetch recent data
    let data = client
        .fetch_klines(symbol, interval, None, None, Some(100))
        .await?;

    println!("\nInceptionTime Prediction");
    println!("═══════════════════════════════════════════════════════════════\n");
    println!("[SYMBOL] {}", symbol);
    println!("[INTERVAL] {}m", interval);
    println!("[MODEL] {}", model_path);
    println!("[DATA] Fetched {} recent candles", data.len());

    if let Some(last) = data.last() {
        println!("\n[LATEST CANDLE]");
        println!("  Open:   ${:.2}", last.open);
        println!("  High:   ${:.2}", last.high);
        println!("  Low:    ${:.2}", last.low);
        println!("  Close:  ${:.2}", last.close);
        println!("  Volume: {:.2}", last.volume);
    }

    println!("\n[NOTE] Prediction requires trained model.");
    println!("[NOTE] This is a demonstration of the prediction pipeline.\n");

    Ok(())
}

fn show_info(config: &Config) {
    println!("\nInceptionTime Trading System v0.1.0");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Configuration:");
    println!("  Symbol:        {}", config.data.symbol);
    println!("  Interval:      {}", config.data.interval);
    println!("  Window Size:   {}", config.data.window_size);
    println!();

    println!("Model Architecture:");
    println!("  Filters:       {}", config.model.num_filters);
    println!("  Depth:         {}", config.model.depth);
    println!("  Kernel Sizes:  {:?}", config.model.kernel_sizes);
    println!("  Ensemble Size: {}", config.model.ensemble_size);
    println!();

    println!("Training Settings:");
    println!("  Batch Size:    {}", config.training.batch_size);
    println!("  Epochs:        {}", config.training.epochs);
    println!("  Learning Rate: {}", config.training.learning_rate);
    println!();

    println!("Strategy Settings:");
    println!("  Min Confidence:    {:.0}%", config.strategy.min_confidence * 100.0);
    println!("  Max Position Size: {:.0}%", config.strategy.max_position_size * 100.0);
    println!("  Risk Per Trade:    {:.0}%", config.strategy.risk_per_trade * 100.0);
    println!();

    println!("═══════════════════════════════════════════════════════════════\n");
}
