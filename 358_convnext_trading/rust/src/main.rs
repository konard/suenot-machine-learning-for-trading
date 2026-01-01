//! ConvNeXt Trading CLI
//!
//! Command-line interface for ConvNeXt trading operations.

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use convnext_trading::prelude::*;

#[derive(Parser)]
#[command(name = "convnext-trading")]
#[command(about = "ConvNeXt-based cryptocurrency trading system", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbosity level
    #[arg(short, long, default_value = "info")]
    log_level: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch historical data from Bybit
    Fetch {
        /// Trading pair symbol (e.g., BTCUSDT)
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Interval (1m, 5m, 15m, 1h, 4h, 1d)
        #[arg(short, long, default_value = "1h")]
        interval: String,

        /// Number of days to fetch
        #[arg(short, long, default_value = "365")]
        days: u32,

        /// Output file path
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Train a new model
    Train {
        /// Path to training data
        #[arg(short, long)]
        data: String,

        /// Number of training epochs
        #[arg(short, long, default_value = "100")]
        epochs: u32,

        /// Batch size
        #[arg(short, long, default_value = "32")]
        batch_size: usize,

        /// Learning rate
        #[arg(short, long, default_value = "0.0004")]
        learning_rate: f64,

        /// Output model path
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Run backtest
    Backtest {
        /// Path to price data
        #[arg(short, long)]
        data: String,

        /// Path to model weights (optional, uses random weights if not provided)
        #[arg(short, long)]
        model: Option<String>,

        /// Initial capital
        #[arg(short, long, default_value = "10000.0")]
        capital: f64,

        /// Maximum risk per trade (fraction)
        #[arg(short, long, default_value = "0.02")]
        risk: f64,
    },

    /// Generate live signals
    Live {
        /// Trading pair symbol
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Interval
        #[arg(short, long, default_value = "1h")]
        interval: String,

        /// Path to model weights
        #[arg(short, long)]
        model: Option<String>,
    },

    /// Show model information
    Info,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let log_level = match cli.log_level.as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    match cli.command {
        Commands::Fetch {
            symbol,
            interval,
            days,
            output,
        } => {
            info!("Fetching {} {} data for {} days", symbol, interval, days);
            fetch_data(&symbol, &interval, days, output).await?;
        }

        Commands::Train {
            data,
            epochs,
            batch_size,
            learning_rate,
            output,
        } => {
            info!("Training model on {} for {} epochs", data, epochs);
            train_model(&data, epochs, batch_size, learning_rate, output)?;
        }

        Commands::Backtest {
            data,
            model,
            capital,
            risk,
        } => {
            info!("Running backtest on {}", data);
            run_backtest(&data, model, capital, risk)?;
        }

        Commands::Live {
            symbol,
            interval,
            model,
        } => {
            info!("Starting live signal generation for {}", symbol);
            run_live(&symbol, &interval, model).await?;
        }

        Commands::Info => {
            show_info();
        }
    }

    Ok(())
}

async fn fetch_data(symbol: &str, interval: &str, days: u32, output: Option<String>) -> Result<()> {
    let client = BybitClient::new();
    let interval = Interval::from_str(interval)?;

    let end_time = chrono::Utc::now();
    let start_time = end_time - chrono::Duration::days(days as i64);

    info!("Fetching data from {} to {}", start_time, end_time);

    let candles = client
        .get_klines(symbol, interval, start_time.timestamp_millis(), end_time.timestamp_millis())
        .await?;

    info!("Fetched {} candles", candles.len());

    let output_path = output.unwrap_or_else(|| {
        format!("data/{}_{}.json", symbol.to_lowercase(), interval.as_str())
    });

    // Create data directory if needed
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let json = serde_json::to_string_pretty(&candles)?;
    std::fs::write(&output_path, json)?;

    info!("Saved data to {}", output_path);
    Ok(())
}

fn train_model(
    data_path: &str,
    epochs: u32,
    batch_size: usize,
    learning_rate: f64,
    output: Option<String>,
) -> Result<()> {
    info!("Loading data from {}", data_path);

    let data = std::fs::read_to_string(data_path)?;
    let candles: Vec<Candle> = serde_json::from_str(&data)?;

    info!("Loaded {} candles", candles.len());

    // Build features
    let feature_builder = FeatureBuilder::new();
    let features = feature_builder.build(&candles)?;

    info!("Built features with shape: {:?}", features.shape());

    // Create dataset
    let dataset = Dataset::from_features(features, 256)?;

    info!("Created dataset with {} samples", dataset.len());

    // Create model
    let config = ConvNeXtConfig::tiny();
    let mut model = ConvNeXt::new(config);

    info!("Created ConvNeXt-Tiny model");

    // Training loop (simplified - in real scenario would use proper optimizer)
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut n_batches = 0;

        for batch in dataset.batches(batch_size) {
            let (x, y) = batch;
            let output = model.forward(&x);
            let loss = cross_entropy_loss(&output, &y);
            total_loss += loss;
            n_batches += 1;

            // Simplified gradient update
            model.backward(&output, &y, learning_rate);
        }

        let avg_loss = total_loss / n_batches as f64;

        if epoch % 10 == 0 || epoch == epochs - 1 {
            info!("Epoch {}/{}: Loss = {:.6}", epoch + 1, epochs, avg_loss);
        }
    }

    // Save model
    let output_path = output.unwrap_or_else(|| "models/convnext_tiny.json".to_string());
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    model.save(&output_path)?;
    info!("Saved model to {}", output_path);

    Ok(())
}

fn run_backtest(data_path: &str, model_path: Option<String>, capital: f64, risk: f64) -> Result<()> {
    info!("Loading data from {}", data_path);

    let data = std::fs::read_to_string(data_path)?;
    let candles: Vec<Candle> = serde_json::from_str(&data)?;

    // Load or create model
    let model = if let Some(path) = model_path {
        info!("Loading model from {}", path);
        ConvNeXt::load(&path)?
    } else {
        info!("Using randomly initialized model");
        ConvNeXt::new(ConvNeXtConfig::tiny())
    };

    // Create strategy
    let strategy = Strategy::new(model, risk);

    // Run backtest
    let backtest = Backtest::new(strategy, capital);
    let metrics = backtest.run(&candles)?;

    // Print results
    println!("\n=== Backtest Results ===");
    println!("Total Return:     {:.2}%", metrics.total_return * 100.0);
    println!("Sharpe Ratio:     {:.2}", metrics.sharpe_ratio);
    println!("Sortino Ratio:    {:.2}", metrics.sortino_ratio);
    println!("Max Drawdown:     {:.2}%", metrics.max_drawdown * 100.0);
    println!("Win Rate:         {:.2}%", metrics.win_rate * 100.0);
    println!("Profit Factor:    {:.2}", metrics.profit_factor);
    println!("Total Trades:     {}", metrics.total_trades);
    println!("========================\n");

    Ok(())
}

async fn run_live(symbol: &str, interval: &str, model_path: Option<String>) -> Result<()> {
    let client = BybitClient::new();
    let interval_enum = Interval::from_str(interval)?;

    // Load or create model
    let model = if let Some(path) = model_path {
        info!("Loading model from {}", path);
        ConvNeXt::load(&path)?
    } else {
        info!("Using randomly initialized model (for demonstration)");
        ConvNeXt::new(ConvNeXtConfig::tiny())
    };

    let feature_builder = FeatureBuilder::new();

    info!("Starting live signal generation...");
    info!("Press Ctrl+C to stop");

    loop {
        // Fetch recent data
        let end_time = chrono::Utc::now();
        let start_time = end_time - chrono::Duration::hours(256);

        let candles = client
            .get_klines(
                symbol,
                interval_enum,
                start_time.timestamp_millis(),
                end_time.timestamp_millis(),
            )
            .await?;

        if candles.len() < 256 {
            info!("Not enough data, waiting...");
            tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
            continue;
        }

        // Build features
        let features = feature_builder.build(&candles)?;

        // Get last sequence
        let seq_len = 256;
        let n_features = features.shape()[1];
        let start_idx = features.shape()[0].saturating_sub(seq_len);

        let input = features
            .slice(ndarray::s![start_idx.., ..])
            .to_owned()
            .into_shape((1, seq_len, n_features))?
            .permuted_axes([0, 2, 1])
            .to_owned();

        // Generate signal
        let output = model.forward(&input);
        let signal = Signal::from_output(&output);

        let latest_candle = candles.last().unwrap();

        println!(
            "[{}] {} @ ${:.2} - Signal: {:?}",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"),
            symbol,
            latest_candle.close,
            signal
        );

        // Wait for next candle
        let wait_secs = match interval_enum {
            Interval::M1 => 60,
            Interval::M5 => 300,
            Interval::M15 => 900,
            Interval::H1 => 3600,
            Interval::H4 => 14400,
            Interval::D1 => 86400,
        };

        tokio::time::sleep(tokio::time::Duration::from_secs(wait_secs)).await;
    }
}

fn show_info() {
    println!("\n=== ConvNeXt Trading ===");
    println!("Version: {}", convnext_trading::VERSION);
    println!("\nArchitecture Variants:");
    println!("  - ConvNeXt-Tiny:  28M params, [96, 192, 384, 768] channels");
    println!("  - ConvNeXt-Small: 50M params, [96, 192, 384, 768] channels");
    println!("  - ConvNeXt-Base:  89M params, [128, 256, 512, 1024] channels");
    println!("\nSupported Intervals: 1m, 5m, 15m, 1h, 4h, 1d");
    println!("Data Source: Bybit Exchange");
    println!("\nUsage Examples:");
    println!("  Fetch data:      convnext-trading fetch -s BTCUSDT -i 1h -d 365");
    println!("  Train model:     convnext-trading train -d data/btcusdt_1h.json -e 100");
    println!("  Run backtest:    convnext-trading backtest -d data/btcusdt_1h.json");
    println!("  Live signals:    convnext-trading live -s BTCUSDT -i 1h");
    println!("========================\n");
}

fn cross_entropy_loss(output: &ndarray::Array2<f64>, target: &ndarray::Array1<usize>) -> f64 {
    let mut loss = 0.0;
    for (i, &t) in target.iter().enumerate() {
        let p = output[[i, t]].max(1e-10);
        loss -= p.ln();
    }
    loss / target.len() as f64
}
