//! GLOW Trading CLI
//!
//! Command-line interface for GLOW-based cryptocurrency trading

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::{Parser, Subcommand};
use glow_trading::{
    BybitClient, Interval, GLOWModel, GLOWConfig,
    GLOWTrader, TraderConfig, Backtest, BacktestConfig,
    FeatureExtractor, Normalizer, Checkpoint,
};
use indicatif::{ProgressBar, ProgressStyle};
use tracing::{info, warn};

#[derive(Parser)]
#[command(name = "glow-trading")]
#[command(about = "GLOW-based cryptocurrency trading system")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch historical data from Bybit
    Fetch {
        /// Trading symbol (e.g., BTCUSDT)
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Number of days to fetch
        #[arg(short, long, default_value = "30")]
        days: i64,

        /// Output file path
        #[arg(short, long, default_value = "data.csv")]
        output: String,
    },

    /// Train GLOW model
    Train {
        /// Input data file
        #[arg(short, long)]
        input: String,

        /// Output model file
        #[arg(short, long, default_value = "model.bin")]
        output: String,

        /// Number of epochs
        #[arg(short, long, default_value = "100")]
        epochs: usize,

        /// Batch size
        #[arg(short, long, default_value = "256")]
        batch_size: usize,
    },

    /// Run backtest
    Backtest {
        /// Model file
        #[arg(short, long)]
        model: String,

        /// Data file
        #[arg(short, long)]
        data: String,

        /// Initial capital
        #[arg(short, long, default_value = "10000")]
        capital: f64,
    },

    /// Generate trading signal
    Signal {
        /// Model file
        #[arg(short, long)]
        model: String,

        /// Trading symbol
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,
    },

    /// Generate market scenarios
    Scenarios {
        /// Model file
        #[arg(short, long)]
        model: String,

        /// Number of scenarios
        #[arg(short, long, default_value = "1000")]
        num: usize,

        /// Temperature for sampling
        #[arg(short, long, default_value = "1.0")]
        temperature: f64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Fetch { symbol, days, output } => {
            fetch_data(&symbol, days, &output).await?;
        }
        Commands::Train { input, output, epochs, batch_size } => {
            train_model(&input, &output, epochs, batch_size)?;
        }
        Commands::Backtest { model, data, capital } => {
            run_backtest(&model, &data, capital)?;
        }
        Commands::Signal { model, symbol } => {
            generate_signal(&model, &symbol).await?;
        }
        Commands::Scenarios { model, num, temperature } => {
            generate_scenarios(&model, num, temperature)?;
        }
    }

    Ok(())
}

async fn fetch_data(symbol: &str, days: i64, output: &str) -> Result<()> {
    info!("Fetching {} data for {} days", symbol, days);

    let client = BybitClient::new();
    let end_time = Utc::now();
    let start_time = end_time - Duration::days(days);

    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::default_spinner()
        .template("{spinner:.green} {msg}")
        .unwrap());
    pb.set_message("Fetching data from Bybit...");

    let candles = client
        .get_klines(symbol, Interval::OneHour, start_time, end_time)
        .await?;

    pb.finish_with_message(format!("Fetched {} candles", candles.len()));

    // Save to CSV
    let mut wtr = csv::Writer::from_path(output)?;
    wtr.write_record(&["timestamp", "open", "high", "low", "close", "volume", "turnover"])?;

    for candle in &candles {
        wtr.write_record(&[
            candle.timestamp.to_string(),
            candle.open.to_string(),
            candle.high.to_string(),
            candle.low.to_string(),
            candle.close.to_string(),
            candle.volume.to_string(),
            candle.turnover.to_string(),
        ])?;
    }
    wtr.flush()?;

    info!("Data saved to {}", output);
    Ok(())
}

fn train_model(input: &str, output: &str, epochs: usize, batch_size: usize) -> Result<()> {
    info!("Training GLOW model on {}", input);

    // Load data
    let mut rdr = csv::Reader::from_path(input)?;
    let mut candles = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let candle = glow_trading::Candle {
            timestamp: record[0].parse()?,
            open: record[1].parse()?,
            high: record[2].parse()?,
            low: record[3].parse()?,
            close: record[4].parse()?,
            volume: record[5].parse()?,
            turnover: record[6].parse()?,
        };
        candles.push(candle);
    }

    info!("Loaded {} candles", candles.len());

    // Extract features
    let features = FeatureExtractor::extract_features_batch(&candles, 20);
    info!("Extracted {} feature vectors", features.nrows());

    if features.nrows() == 0 {
        warn!("No features extracted. Need more data.");
        return Ok(());
    }

    // Normalize
    let normalizer = Normalizer::fit(&features);
    let normalized = normalizer.transform(&features);

    // Split train/val
    let train_size = (normalized.nrows() as f64 * 0.8) as usize;
    let train_data = normalized.slice(ndarray::s![..train_size, ..]).to_owned();
    let val_data = normalized.slice(ndarray::s![train_size.., ..]).to_owned();

    info!("Train size: {}, Val size: {}", train_data.nrows(), val_data.nrows());

    // Create model
    let config = GLOWConfig {
        num_features: features.ncols(),
        num_levels: 3,
        num_steps: 4,
        hidden_dim: 64,
        learning_rate: 1e-4,
    };
    let mut model = GLOWModel::new(config);

    // Training loop
    let pb = ProgressBar::new(epochs as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg} [{bar:40}] {pos}/{len} ({eta})")
        .unwrap());

    let mut best_val_loss = f64::INFINITY;

    for epoch in 0..epochs {
        // Train step (simplified - just compute loss)
        let train_log_prob = model.log_prob(&train_data);
        let train_loss = -train_log_prob.mean().unwrap_or(0.0);

        // Validation
        let val_log_prob = model.log_prob(&val_data);
        let val_loss = -val_log_prob.mean().unwrap_or(0.0);

        if val_loss < best_val_loss {
            best_val_loss = val_loss;
        }

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            pb.set_message(format!("Train NLL: {:.4}, Val NLL: {:.4}", train_loss, val_loss));
        }
        pb.inc(1);
    }
    pb.finish();

    // Save checkpoint
    let mut checkpoint = Checkpoint::new(model);
    checkpoint.set_normalizer(normalizer);
    checkpoint.save(output)?;

    info!("Model saved to {}", output);
    Ok(())
}

fn run_backtest(model_path: &str, data_path: &str, capital: f64) -> Result<()> {
    info!("Running backtest with model {} on {}", model_path, data_path);

    // Load model
    let checkpoint = Checkpoint::load(model_path)?;
    let model = checkpoint.model;

    // Load data
    let mut rdr = csv::Reader::from_path(data_path)?;
    let mut candles = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let candle = glow_trading::Candle {
            timestamp: record[0].parse()?,
            open: record[1].parse()?,
            high: record[2].parse()?,
            low: record[3].parse()?,
            close: record[4].parse()?,
            volume: record[5].parse()?,
            turnover: record[6].parse()?,
        };
        candles.push(candle);
    }

    // Create trader
    let trader_config = TraderConfig::default();
    let mut trader = GLOWTrader::new(model, trader_config);

    if let Some(normalizer) = checkpoint.normalizer {
        trader.set_normalizer(normalizer);
    }

    // Run backtest
    let backtest_config = BacktestConfig {
        initial_capital: capital,
        ..Default::default()
    };
    let backtest = Backtest::new(backtest_config);
    let result = backtest.run(&mut trader, &candles);

    // Print results
    println!("\n=== Backtest Results ===");
    println!("Total Return: {:.2}%", result.metrics.total_return * 100.0);
    println!("Sharpe Ratio: {:.2}", result.metrics.sharpe_ratio);
    println!("Sortino Ratio: {:.2}", result.metrics.sortino_ratio);
    println!("Max Drawdown: {:.2}%", result.metrics.max_drawdown * 100.0);
    println!("Win Rate: {:.2}%", result.metrics.win_rate * 100.0);
    println!("Profit Factor: {:.2}", result.metrics.profit_factor);
    println!("Number of Trades: {}", result.metrics.num_trades);
    println!("Final Equity: ${:.2}", result.final_equity());

    Ok(())
}

async fn generate_signal(model_path: &str, symbol: &str) -> Result<()> {
    info!("Generating signal for {} using model {}", symbol, model_path);

    // Load model
    let checkpoint = Checkpoint::load(model_path)?;
    let model = checkpoint.model;

    // Create trader
    let trader_config = TraderConfig::default();
    let mut trader = GLOWTrader::new(model, trader_config);

    if let Some(normalizer) = checkpoint.normalizer {
        trader.set_normalizer(normalizer);
    }

    // Fetch recent data
    let client = BybitClient::new();
    let end_time = Utc::now();
    let start_time = end_time - Duration::hours(48);

    let candles = client
        .get_klines(symbol, Interval::OneHour, start_time, end_time)
        .await?;

    // Extract features from latest window
    let mut extractor = FeatureExtractor::new(20);
    let mut latest_features = None;

    for candle in candles {
        if let Some(features) = extractor.add_candle(candle) {
            latest_features = Some(features);
        }
    }

    if let Some(features) = latest_features {
        let signal = trader.generate_signal(&features);

        println!("\n=== Trading Signal for {} ===", symbol);
        println!("Signal: {:.4}", signal.signal);
        println!("Log-Likelihood: {:.4}", signal.log_likelihood);
        println!("In Distribution: {}", signal.in_distribution);
        println!("Regime: {}", signal.regime);
        println!("Confidence: {:.4}", signal.confidence);

        if signal.signal > 0.0 {
            println!("Recommendation: LONG ({:.1}% position)", signal.signal.abs() * 100.0);
        } else if signal.signal < 0.0 {
            println!("Recommendation: SHORT ({:.1}% position)", signal.signal.abs() * 100.0);
        } else {
            println!("Recommendation: NEUTRAL (no position)");
        }
    } else {
        warn!("Not enough data to generate features");
    }

    Ok(())
}

fn generate_scenarios(model_path: &str, num: usize, temperature: f64) -> Result<()> {
    info!("Generating {} scenarios with temperature {}", num, temperature);

    // Load model
    let checkpoint = Checkpoint::load(model_path)?;
    let model = checkpoint.model;

    // Create trader
    let trader_config = TraderConfig::default();
    let mut trader = GLOWTrader::new(model, trader_config);

    if let Some(normalizer) = checkpoint.normalizer {
        trader.set_normalizer(normalizer);
    }

    // Generate scenarios
    let scenarios = trader.generate_scenarios(num, temperature);

    // Compute statistics
    let returns = scenarios.column(0);
    let mean_return = returns.mean().unwrap_or(0.0);
    let std_return = {
        let centered = returns.mapv(|v| v - mean_return);
        let variance = centered.mapv(|v| v * v).mean().unwrap_or(0.0);
        variance.sqrt()
    };

    // VaR and CVaR
    let var_95 = trader.compute_var(num, 0.95);
    let cvar_95 = trader.compute_cvar(num, 0.95);

    println!("\n=== Scenario Analysis ({} scenarios) ===", num);
    println!("Mean Return: {:.4}", mean_return);
    println!("Std Return: {:.4}", std_return);
    println!("VaR (95%): {:.4}", var_95);
    println!("CVaR (95%): {:.4}", cvar_95);

    // Percentiles
    let mut sorted_returns: Vec<f64> = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("\nReturn Distribution:");
    println!("  1st percentile: {:.4}", sorted_returns[(num as f64 * 0.01) as usize]);
    println!("  5th percentile: {:.4}", sorted_returns[(num as f64 * 0.05) as usize]);
    println!("  25th percentile: {:.4}", sorted_returns[(num as f64 * 0.25) as usize]);
    println!("  50th percentile: {:.4}", sorted_returns[(num as f64 * 0.50) as usize]);
    println!("  75th percentile: {:.4}", sorted_returns[(num as f64 * 0.75) as usize]);
    println!("  95th percentile: {:.4}", sorted_returns[(num as f64 * 0.95) as usize]);
    println!("  99th percentile: {:.4}", sorted_returns[(num as f64 * 0.99) as usize]);

    Ok(())
}
