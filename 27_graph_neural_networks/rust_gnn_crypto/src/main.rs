//! CLI application for GNN-based cryptocurrency trading.

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::info;
use tracing_subscriber;

use gnn_crypto::{
    data::BybitClient,
    graph::CorrelationGraph,
    model::{GCN, GNNConfig},
    strategy::MomentumStrategy,
    utils::Config,
};

#[derive(Parser)]
#[command(name = "gnn-crypto")]
#[command(about = "Graph Neural Networks for cryptocurrency trading")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Configuration file path
    #[arg(short, long, default_value = "config.toml")]
    config: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch data from Bybit exchange
    Fetch {
        /// Trading symbols (comma-separated)
        #[arg(short, long, default_value = "BTCUSDT,ETHUSDT,SOLUSDT")]
        symbols: String,

        /// Kline interval (1, 5, 15, 30, 60, 240, D)
        #[arg(short, long, default_value = "60")]
        interval: String,

        /// Number of days to fetch
        #[arg(short, long, default_value = "90")]
        days: u32,
    },

    /// Build cryptocurrency graph
    Build {
        /// Graph construction method (correlation, knn, sector)
        #[arg(short, long, default_value = "correlation")]
        method: String,

        /// Correlation threshold
        #[arg(short, long, default_value = "0.5")]
        threshold: f64,

        /// Rolling window size
        #[arg(short, long, default_value = "60")]
        window: usize,
    },

    /// Train GNN model
    Train {
        /// Model architecture (gcn, gat)
        #[arg(short, long, default_value = "gcn")]
        model: String,

        /// Hidden dimension
        #[arg(long, default_value = "64")]
        hidden_dim: i64,

        /// Number of epochs
        #[arg(short, long, default_value = "100")]
        epochs: usize,

        /// Learning rate
        #[arg(short, long, default_value = "0.001")]
        lr: f64,
    },

    /// Backtest trading strategy
    Backtest {
        /// Path to trained model
        #[arg(short, long)]
        model_path: String,

        /// Confidence threshold for signals
        #[arg(short, long, default_value = "0.6")]
        threshold: f64,

        /// Initial capital
        #[arg(short, long, default_value = "100000")]
        capital: f64,
    },

    /// Generate trading signals
    Signals {
        /// Path to trained model
        #[arg(short, long)]
        model_path: String,

        /// Confidence threshold
        #[arg(short, long, default_value = "0.6")]
        threshold: f64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Fetch {
            symbols,
            interval,
            days,
        } => {
            info!("Fetching data from Bybit...");
            let client = BybitClient::new();
            let symbol_list: Vec<&str> = symbols.split(',').collect();

            for symbol in &symbol_list {
                info!("Fetching {} ({} days, {} interval)", symbol, days, interval);
                let data = client
                    .fetch_historical_klines(symbol, &interval, days)
                    .await?;
                info!("  Fetched {} candles", data.len());

                // Save to CSV
                let filename = format!("data/{}_{}.csv", symbol, interval);
                save_to_csv(&data, &filename)?;
                info!("  Saved to {}", filename);
            }
        }

        Commands::Build {
            method,
            threshold,
            window,
        } => {
            info!("Building graph with method: {}", method);
            info!("  Threshold: {}, Window: {}", threshold, window);

            // Load data and build graph
            let graph_builder = CorrelationGraph::new(threshold, window);
            info!("Graph builder created");
            // In real usage, load data and call graph_builder.build(&data)
        }

        Commands::Train {
            model,
            hidden_dim,
            epochs,
            lr,
        } => {
            info!("Training {} model...", model);
            info!("  Hidden dim: {}, Epochs: {}, LR: {}", hidden_dim, epochs, lr);

            let config = GNNConfig {
                num_features: 10,
                hidden_dim,
                num_classes: 3,
                num_layers: 3,
                dropout: 0.3,
                learning_rate: lr,
            };

            match model.as_str() {
                "gcn" => {
                    let _model = GCN::new(&config, tch::Device::Cpu);
                    info!("GCN model created");
                }
                "gat" => {
                    let _model = gnn_crypto::model::GAT::new(&config, tch::Device::Cpu);
                    info!("GAT model created");
                }
                _ => anyhow::bail!("Unknown model: {}", model),
            }
        }

        Commands::Backtest {
            model_path,
            threshold,
            capital,
        } => {
            info!("Running backtest...");
            info!(
                "  Model: {}, Threshold: {}, Capital: {}",
                model_path, threshold, capital
            );
            // Load model and run backtest
        }

        Commands::Signals {
            model_path,
            threshold,
        } => {
            info!("Generating signals...");
            info!("  Model: {}, Threshold: {}", model_path, threshold);
            // Load model and generate signals
        }
    }

    Ok(())
}

fn save_to_csv(data: &[gnn_crypto::OHLCV], filename: &str) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(filename)?;
    writeln!(file, "timestamp,open,high,low,close,volume")?;

    for candle in data {
        writeln!(
            file,
            "{},{},{},{},{},{}",
            candle.timestamp, candle.open, candle.high, candle.low, candle.close, candle.volume
        )?;
    }

    Ok(())
}
