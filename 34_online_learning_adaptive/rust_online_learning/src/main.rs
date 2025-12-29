//! Online Learning CLI Application
//!
//! Demonstrates online learning for adaptive momentum trading on cryptocurrency data.

use clap::{Parser, Subcommand};
use online_learning::api::BybitClient;
use online_learning::backtest::BacktestEngine;
use online_learning::drift::ADWIN;
use online_learning::features::MomentumFeatures;
use online_learning::models::{AdaptiveMomentumWeights, OnlineLinearRegression};
use online_learning::streaming::StreamSimulator;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "online_learning")]
#[command(about = "Online Learning for Adaptive Cryptocurrency Trading")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch cryptocurrency data from Bybit
    Fetch {
        /// Trading symbol (e.g., BTCUSDT)
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Timeframe (e.g., 1h, 4h, 1d)
        #[arg(short, long, default_value = "1h")]
        interval: String,

        /// Number of candles to fetch
        #[arg(short, long, default_value = "500")]
        limit: usize,
    },

    /// Run online regression example
    OnlineRegression {
        /// Trading symbol
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Learning rate
        #[arg(short, long, default_value = "0.01")]
        lr: f64,
    },

    /// Detect concept drift in market data
    DriftDetection {
        /// Trading symbol
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// ADWIN delta parameter
        #[arg(short, long, default_value = "0.002")]
        delta: f64,
    },

    /// Run adaptive momentum strategy
    AdaptiveMomentum {
        /// Trading symbol
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,

        /// Learning rate
        #[arg(short, long, default_value = "0.01")]
        lr: f64,
    },

    /// Compare online vs batch learning
    Compare {
        /// Trading symbol
        #[arg(short, long, default_value = "BTCUSDT")]
        symbol: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let cli = Cli::parse();

    match cli.command {
        Commands::Fetch {
            symbol,
            interval,
            limit,
        } => {
            info!("Fetching {} {} candles for {}", limit, interval, symbol);
            let client = BybitClient::new();
            let candles = client.get_klines(&symbol, &interval, limit).await?;
            info!("Fetched {} candles", candles.len());

            if let Some(last) = candles.last() {
                info!(
                    "Latest: Open={:.2}, High={:.2}, Low={:.2}, Close={:.2}",
                    last.open, last.high, last.low, last.close
                );
            }
        }

        Commands::OnlineRegression { symbol, lr } => {
            info!("Running online regression for {} with lr={}", symbol, lr);

            let client = BybitClient::new();
            let candles = client.get_klines(&symbol, "1h", 500).await?;

            // Create features and model
            let feature_gen = MomentumFeatures::new(vec![12, 24, 48, 96]);
            let mut model = OnlineLinearRegression::new(4, lr);

            let mut predictions = Vec::new();
            let mut actuals = Vec::new();

            // Online learning loop
            for i in 96..candles.len() - 1 {
                let features = feature_gen.compute(&candles[..=i]);
                if let Some(x) = features {
                    // Predict next return
                    let prediction = model.predict(&x);

                    // Get actual return
                    let actual = (candles[i + 1].close - candles[i].close) / candles[i].close;

                    // Learn from observation
                    model.learn(&x, actual);

                    predictions.push(prediction);
                    actuals.push(actual);
                }
            }

            // Calculate metrics
            let mse: f64 = predictions
                .iter()
                .zip(actuals.iter())
                .map(|(p, a)| (p - a).powi(2))
                .sum::<f64>()
                / predictions.len() as f64;

            let direction_accuracy: f64 = predictions
                .iter()
                .zip(actuals.iter())
                .filter(|(p, a)| p.signum() == a.signum())
                .count() as f64
                / predictions.len() as f64;

            info!("MSE: {:.6}", mse);
            info!("Direction Accuracy: {:.2}%", direction_accuracy * 100.0);
        }

        Commands::DriftDetection { symbol, delta } => {
            info!("Running drift detection for {} with delta={}", symbol, delta);

            let client = BybitClient::new();
            let candles = client.get_klines(&symbol, "1h", 1000).await?;

            let mut adwin = ADWIN::new(delta);
            let mut drift_count = 0;

            // Compute returns and check for drift
            for i in 1..candles.len() {
                let ret = (candles[i].close - candles[i - 1].close) / candles[i - 1].close;

                if adwin.update(ret.abs()) {
                    drift_count += 1;
                    info!("Drift detected at candle {} (index {})", i, i);
                }
            }

            info!("Total drifts detected: {}", drift_count);
            info!(
                "Drift frequency: {:.2}%",
                drift_count as f64 / candles.len() as f64 * 100.0
            );
        }

        Commands::AdaptiveMomentum { symbol, lr } => {
            info!("Running adaptive momentum for {} with lr={}", symbol, lr);

            let client = BybitClient::new();
            let candles = client.get_klines(&symbol, "1h", 500).await?;

            // Create adaptive weights model
            let mut weights = AdaptiveMomentumWeights::new(
                4,
                lr,
                vec![
                    "mom_12h".to_string(),
                    "mom_24h".to_string(),
                    "mom_48h".to_string(),
                    "mom_96h".to_string(),
                ],
            );

            let feature_gen = MomentumFeatures::new(vec![12, 24, 48, 96]);
            let mut total_pnl = 0.0;
            let mut trades = 0;

            for i in 96..candles.len() - 1 {
                let features = feature_gen.compute(&candles[..=i]);
                if let Some(signals) = features {
                    // Predict
                    let prediction = weights.predict(&signals);

                    // Generate signal
                    let signal = if prediction > 0.001 {
                        1.0
                    } else if prediction < -0.001 {
                        -1.0
                    } else {
                        0.0
                    };

                    // Get actual return
                    let actual = (candles[i + 1].close - candles[i].close) / candles[i].close;

                    // Trade PnL
                    let pnl = signal * actual;
                    total_pnl += pnl;
                    if signal != 0.0 {
                        trades += 1;
                    }

                    // Learn
                    weights.update(&signals, actual);
                }
            }

            info!("Total PnL: {:.4}", total_pnl);
            info!("Number of trades: {}", trades);
            info!("Average PnL per trade: {:.6}", total_pnl / trades as f64);
            info!("Final weights: {:?}", weights.get_weights());
        }

        Commands::Compare { symbol } => {
            info!("Comparing online vs batch learning for {}", symbol);

            let client = BybitClient::new();
            let candles = client.get_klines(&symbol, "1h", 1000).await?;

            // Run backtest comparison
            let engine = BacktestEngine::new(candles);
            let results = engine.compare_approaches(0.01)?;

            info!("=== Comparison Results ===");
            info!(
                "Online Learning - Sharpe: {:.2}, Total Return: {:.2}%",
                results.online_sharpe,
                results.online_return * 100.0
            );
            info!(
                "Static Model    - Sharpe: {:.2}, Total Return: {:.2}%",
                results.static_sharpe,
                results.static_return * 100.0
            );
            info!(
                "Monthly Retrain - Sharpe: {:.2}, Total Return: {:.2}%",
                results.monthly_sharpe,
                results.monthly_return * 100.0
            );
        }
    }

    Ok(())
}
