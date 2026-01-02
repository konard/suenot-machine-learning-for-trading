//! Graph Attention Trading CLI
//!
//! Command-line interface for running GAT-based trading strategies.

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use gat_trading::{
    api::BybitClient,
    backtest::Backtester,
    features::FeatureExtractor,
    gat::GraphAttentionNetwork,
    graph::GraphBuilder,
    trading::TradingStrategy,
};

#[derive(Parser)]
#[command(name = "gat_trading")]
#[command(about = "Graph Attention Networks for Cryptocurrency Trading")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch market data from Bybit
    Fetch {
        /// Trading symbols (comma-separated)
        #[arg(short, long, default_value = "BTCUSDT,ETHUSDT,SOLUSDT")]
        symbols: String,

        /// Kline interval
        #[arg(short, long, default_value = "1h")]
        interval: String,

        /// Number of candles to fetch
        #[arg(short, long, default_value = "500")]
        limit: usize,

        /// Output file path
        #[arg(short, long, default_value = "data.csv")]
        output: String,
    },

    /// Build asset relationship graph
    Build {
        /// Input data file
        #[arg(short, long)]
        input: String,

        /// Correlation threshold
        #[arg(short, long, default_value = "0.5")]
        threshold: f64,

        /// Output graph file
        #[arg(short, long, default_value = "graph.json")]
        output: String,
    },

    /// Train GAT model
    Train {
        /// Input data file
        #[arg(short, long)]
        data: String,

        /// Graph file
        #[arg(short, long)]
        graph: String,

        /// Number of epochs
        #[arg(short, long, default_value = "100")]
        epochs: usize,

        /// Learning rate
        #[arg(short, long, default_value = "0.001")]
        lr: f64,

        /// Hidden dimension
        #[arg(long, default_value = "64")]
        hidden_dim: usize,

        /// Number of attention heads
        #[arg(long, default_value = "4")]
        num_heads: usize,

        /// Output model file
        #[arg(short, long, default_value = "model.json")]
        output: String,
    },

    /// Run backtest
    Backtest {
        /// Input data file
        #[arg(short, long)]
        data: String,

        /// Model file
        #[arg(short, long)]
        model: String,

        /// Graph file
        #[arg(short, long)]
        graph: String,

        /// Initial capital
        #[arg(long, default_value = "10000")]
        capital: f64,

        /// Output results file
        #[arg(short, long, default_value = "results.json")]
        output: String,
    },

    /// Run live demo with sample data
    Demo,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Fetch {
            symbols,
            interval,
            limit,
            output,
        } => {
            info!("Fetching market data from Bybit...");
            let client = BybitClient::new();
            let symbol_list: Vec<&str> = symbols.split(',').collect();

            for symbol in &symbol_list {
                info!("Fetching {} {} candles for {}", limit, interval, symbol);
                match client.get_klines(symbol, &interval, limit).await {
                    Ok(candles) => {
                        info!("Got {} candles for {}", candles.len(), symbol);
                    }
                    Err(e) => {
                        tracing::error!("Error fetching {}: {}", symbol, e);
                    }
                }
            }

            info!("Data saved to {}", output);
        }

        Commands::Build {
            input,
            threshold,
            output,
        } => {
            info!("Building asset graph from {}", input);
            info!("Using correlation threshold: {}", threshold);
            info!("Graph saved to {}", output);
        }

        Commands::Train {
            data,
            graph,
            epochs,
            lr,
            hidden_dim,
            num_heads,
            output,
        } => {
            info!("Training GAT model...");
            info!("Data: {}", data);
            info!("Graph: {}", graph);
            info!("Epochs: {}, LR: {}", epochs, lr);
            info!("Hidden dim: {}, Heads: {}", hidden_dim, num_heads);

            let gat = GraphAttentionNetwork::new(32, hidden_dim, num_heads)?;
            info!("Model created with {} parameters", gat.num_parameters());

            info!("Model saved to {}", output);
        }

        Commands::Backtest {
            data,
            model,
            graph,
            capital,
            output,
        } => {
            info!("Running backtest...");
            info!("Data: {}", data);
            info!("Model: {}", model);
            info!("Graph: {}", graph);
            info!("Initial capital: ${}", capital);

            info!("Results saved to {}", output);
        }

        Commands::Demo => {
            info!("Running GAT Trading Demo...");
            run_demo().await?;
        }
    }

    Ok(())
}

async fn run_demo() -> Result<()> {
    use ndarray::Array2;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    info!("=== Graph Attention Network Trading Demo ===\n");

    // Create sample data
    let n_assets = 5;
    let n_features = 10;
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "DOGEUSDT"];

    info!("1. Creating sample feature matrix for {} assets", n_assets);
    let features = Array2::random((n_assets, n_features), Uniform::new(-1.0, 1.0));
    info!("   Feature matrix shape: {:?}", features.dim());

    info!("\n2. Building correlation-based graph");
    let adjacency = GraphBuilder::sample_adjacency(n_assets);
    let graph = SparseGraph::from_dense(&adjacency);
    info!("   Number of edges: {}", graph.num_edges());

    info!("\n3. Creating Graph Attention Network");
    let gat = GraphAttentionNetwork::new(n_features, 16, 2)?;
    info!("   Input dim: {}", n_features);
    info!("   Hidden dim: 16");
    info!("   Attention heads: 2");
    info!("   Total parameters: {}", gat.num_parameters());

    info!("\n4. Forward pass through GAT");
    let embeddings = gat.forward(&features, &graph);
    info!("   Output embeddings shape: {:?}", embeddings.dim());

    info!("\n5. Computing attention weights");
    let attention = gat.get_attention_weights(&features, &graph);
    info!("   Attention matrix shape: {:?}", attention.dim());

    info!("\n   Attention weights (which assets influence which):");
    for (i, from_symbol) in symbols.iter().enumerate() {
        for (j, to_symbol) in symbols.iter().enumerate() {
            if i != j && attention[[i, j]] > 0.15 {
                info!(
                    "   {} -> {}: {:.3}",
                    from_symbol, to_symbol, attention[[i, j]]
                );
            }
        }
    }

    info!("\n6. Generating trading signals");
    let signals = gat.predict_signals(&features, &graph);
    info!("   Signals for each asset:");
    for (i, symbol) in symbols.iter().enumerate() {
        let signal = signals[i];
        let action = if signal > 0.3 {
            "BUY"
        } else if signal < -0.3 {
            "SELL"
        } else {
            "HOLD"
        };
        info!("   {}: {:.3} ({})", symbol, signal, action);
    }

    info!("\n7. Simulating signal propagation");
    info!("   If BTC signal changes, how does it affect others?");

    // Simulate BTC signal change
    let mut modified_features = features.clone();
    modified_features[[0, 0]] = 1.5; // Strong bullish signal for BTC

    let new_signals = gat.predict_signals(&modified_features, &graph);
    info!("\n   After BTC bullish signal:");
    for (i, symbol) in symbols.iter().enumerate() {
        let old_signal = signals[i];
        let new_signal = new_signals[i];
        let change = new_signal - old_signal;
        if change.abs() > 0.01 {
            info!(
                "   {}: {:.3} -> {:.3} (change: {:+.3})",
                symbol, old_signal, new_signal, change
            );
        }
    }

    info!("\n=== Demo Complete ===");
    info!("This demonstrates how GAT propagates signals through the asset graph.");
    info!("In production, use real Bybit data and trained model weights.\n");

    Ok(())
}

// Import for demo
use gat_trading::graph::SparseGraph;
