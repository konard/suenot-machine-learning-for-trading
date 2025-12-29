//! Example: Train GNN model for cryptocurrency prediction.
//!
//! Usage:
//!   cargo run --example train_gnn -- --model gcn --epochs 100

use anyhow::Result;
use clap::Parser;
use gnn_crypto::{
    data::{features::compute_returns, FeatureEngineer},
    graph::{CorrelationGraph, GraphBuilder},
    model::{create_edge_index, create_features, create_labels, GCN, GAT, GNNConfig},
    OHLCV,
};
use std::fs::File;
use std::io::{BufRead, BufReader};
use tch::Device;
use tracing::info;

#[derive(Parser)]
#[command(name = "train_gnn")]
#[command(about = "Train GNN model for cryptocurrency prediction")]
struct Args {
    /// Data directory
    #[arg(short, long, default_value = "data")]
    data_dir: String,

    /// Model architecture (gcn, gat)
    #[arg(short, long, default_value = "gcn")]
    model: String,

    /// Hidden dimension
    #[arg(long, default_value = "64")]
    hidden_dim: i64,

    /// Number of layers
    #[arg(long, default_value = "3")]
    num_layers: usize,

    /// Number of epochs
    #[arg(short, long, default_value = "100")]
    epochs: usize,

    /// Learning rate
    #[arg(short, long, default_value = "0.001")]
    lr: f64,

    /// Dropout
    #[arg(long, default_value = "0.3")]
    dropout: f64,

    /// Correlation threshold for graph
    #[arg(long, default_value = "0.5")]
    threshold: f64,

    /// Window size for features
    #[arg(long, default_value = "20")]
    window: usize,

    /// Checkpoint directory
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: String,
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Create checkpoint directory
    std::fs::create_dir_all(&args.checkpoint_dir)?;

    // Determine device
    let device = if tch::Cuda::is_available() {
        info!("Using CUDA");
        Device::Cuda(0)
    } else {
        info!("Using CPU");
        Device::Cpu
    };

    // Load data
    info!("Loading data from {}...", args.data_dir);
    let symbols_file = format!("{}/symbols.txt", args.data_dir);
    let symbols = load_symbols(&symbols_file)?;
    info!("Found {} symbols", symbols.len());

    let mut all_data: Vec<Vec<OHLCV>> = Vec::new();
    let mut valid_symbols: Vec<String> = Vec::new();

    for symbol in &symbols {
        let filename = format!("{}/{}_60.csv", args.data_dir, symbol);
        if let Ok(data) = load_ohlcv(&filename) {
            if data.len() > args.window + 10 {
                all_data.push(data);
                valid_symbols.push(symbol.clone());
            }
        }
    }

    info!("Loaded {} valid symbols", valid_symbols.len());

    if valid_symbols.len() < 3 {
        anyhow::bail!("Need at least 3 symbols for graph construction");
    }

    // Compute returns
    let returns: Vec<Vec<f64>> = all_data
        .iter()
        .map(|ohlcv| {
            let closes: Vec<f64> = ohlcv.iter().map(|o| o.close).collect();
            compute_returns(&closes)
        })
        .collect();

    // Build graph
    info!("Building correlation graph (threshold={})...", args.threshold);
    let graph_builder = CorrelationGraph::new(args.threshold, args.window);
    let graph = graph_builder.build(&returns, &valid_symbols);
    info!("Graph: {} nodes, {} edges", graph.node_count(), graph.edge_count());

    // Compute features
    info!("Computing features...");
    let feature_engineer = FeatureEngineer::new(args.window);
    let features: Vec<Vec<f64>> = all_data
        .iter()
        .map(|ohlcv| feature_engineer.compute_features(ohlcv))
        .collect::<Result<Vec<_>>>()?;

    let num_features = features[0].len() as i64;
    info!("Features per node: {}", num_features);

    // Create labels (based on future returns)
    let labels: Vec<i64> = returns
        .iter()
        .map(|r| {
            let last_return = r.last().copied().unwrap_or(0.0);
            if last_return > 0.01 {
                2 // Up
            } else if last_return < -0.01 {
                0 // Down
            } else {
                1 // Neutral
            }
        })
        .collect();

    // Create tensors
    let (sources, targets) = graph.to_edge_index();
    let edge_index = create_edge_index(&sources, &targets, device);
    let x = create_features(&features, device);
    let y = create_labels(&labels, device);

    // Create model
    let config = GNNConfig {
        num_features,
        hidden_dim: args.hidden_dim,
        num_classes: 3,
        num_layers: args.num_layers,
        dropout: args.dropout,
        learning_rate: args.lr,
    };

    info!("Creating {} model...", args.model);
    info!("  Hidden dim: {}", config.hidden_dim);
    info!("  Num layers: {}", config.num_layers);
    info!("  Dropout: {}", config.dropout);

    match args.model.as_str() {
        "gcn" => {
            let mut trainer = gnn_crypto::model::gcn::GCNTrainer::new(&config, device);
            train_model(&mut trainer, &x, &edge_index, &y, args.epochs, &args.checkpoint_dir)?;
        }
        "gat" => {
            let mut trainer = gnn_crypto::model::gat::GATTrainer::new(&config, device, 4);
            train_model_gat(&mut trainer, &x, &edge_index, &y, args.epochs, &args.checkpoint_dir)?;
        }
        _ => anyhow::bail!("Unknown model: {}", args.model),
    }

    Ok(())
}

fn train_model(
    trainer: &mut gnn_crypto::model::gcn::GCNTrainer,
    x: &tch::Tensor,
    edge_index: &tch::Tensor,
    y: &tch::Tensor,
    epochs: usize,
    checkpoint_dir: &str,
) -> Result<()> {
    info!("\n=== Training GCN ===");

    let mut best_acc = 0.0;

    for epoch in 0..epochs {
        let loss = trainer.train_epoch(x, edge_index, y, None);
        let (val_loss, accuracy) = trainer.evaluate(x, edge_index, y, None);

        if epoch % 10 == 0 || epoch == epochs - 1 {
            info!(
                "Epoch {}/{}: train_loss={:.4}, val_loss={:.4}, accuracy={:.2}%",
                epoch + 1,
                epochs,
                loss,
                val_loss,
                accuracy * 100.0
            );
        }

        if accuracy > best_acc {
            best_acc = accuracy;
            let path = format!("{}/best_gcn.pt", checkpoint_dir);
            trainer.model().save(&path)?;
        }
    }

    info!("\nTraining complete!");
    info!("Best accuracy: {:.2}%", best_acc * 100.0);
    info!("Model saved to {}/best_gcn.pt", checkpoint_dir);

    Ok(())
}

fn train_model_gat(
    trainer: &mut gnn_crypto::model::gat::GATTrainer,
    x: &tch::Tensor,
    edge_index: &tch::Tensor,
    y: &tch::Tensor,
    epochs: usize,
    checkpoint_dir: &str,
) -> Result<()> {
    info!("\n=== Training GAT ===");

    let mut best_acc = 0.0;

    for epoch in 0..epochs {
        let loss = trainer.train_epoch(x, edge_index, y, None);
        let (val_loss, accuracy) = trainer.evaluate(x, edge_index, y, None);

        if epoch % 10 == 0 || epoch == epochs - 1 {
            info!(
                "Epoch {}/{}: train_loss={:.4}, val_loss={:.4}, accuracy={:.2}%",
                epoch + 1,
                epochs,
                loss,
                val_loss,
                accuracy * 100.0
            );
        }

        if accuracy > best_acc {
            best_acc = accuracy;
            let path = format!("{}/best_gat.pt", checkpoint_dir);
            trainer.model().save(&path)?;
        }
    }

    info!("\nTraining complete!");
    info!("Best accuracy: {:.2}%", best_acc * 100.0);
    info!("Model saved to {}/best_gat.pt", checkpoint_dir);

    Ok(())
}

fn load_symbols(path: &str) -> Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let symbols: Vec<String> = reader
        .lines()
        .filter_map(|line| line.ok())
        .filter(|line| !line.is_empty())
        .collect();
    Ok(symbols)
}

fn load_ohlcv(path: &str) -> Result<Vec<OHLCV>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 6 {
            data.push(OHLCV {
                timestamp: parts[0].parse()?,
                open: parts[1].parse()?,
                high: parts[2].parse()?,
                low: parts[3].parse()?,
                close: parts[4].parse()?,
                volume: parts[5].parse()?,
            });
        }
    }

    Ok(data)
}
