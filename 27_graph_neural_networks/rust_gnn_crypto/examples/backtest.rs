//! Example: Backtest momentum propagation strategy.
//!
//! Usage:
//!   cargo run --example backtest -- --model-path checkpoints/best_gcn.pt

use anyhow::Result;
use clap::Parser;
use gnn_crypto::{
    data::{features::compute_returns, FeatureEngineer},
    graph::{CorrelationGraph, GraphBuilder},
    model::{create_edge_index, create_features, GCN, GNNConfig, GNNModel},
    strategy::{MomentumStrategy, Portfolio, Signal, TradingStrategy},
    OHLCV,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use tch::Device;
use tracing::info;

#[derive(Parser)]
#[command(name = "backtest")]
#[command(about = "Backtest momentum propagation strategy")]
struct Args {
    /// Data directory
    #[arg(short, long, default_value = "data")]
    data_dir: String,

    /// Path to trained model
    #[arg(short, long, default_value = "checkpoints/best_gcn.pt")]
    model_path: String,

    /// Confidence threshold for signals
    #[arg(short, long, default_value = "0.6")]
    threshold: f64,

    /// Initial capital
    #[arg(short, long, default_value = "100000")]
    capital: f64,

    /// Transaction cost (percentage)
    #[arg(long, default_value = "0.001")]
    cost: f64,

    /// Maximum positions
    #[arg(long, default_value = "5")]
    max_positions: usize,

    /// Position size (percentage of capital)
    #[arg(long, default_value = "0.1")]
    position_size: f64,

    /// Window size for features
    #[arg(long, default_value = "20")]
    window: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("=== Momentum Propagation Strategy Backtest ===");
    info!("Model: {}", args.model_path);
    info!("Initial capital: ${:.2}", args.capital);
    info!("Confidence threshold: {}", args.threshold);

    // Load data
    let symbols_file = format!("{}/symbols.txt", args.data_dir);
    let symbols = load_symbols(&symbols_file)?;
    info!("Loaded {} symbols", symbols.len());

    let mut all_data: Vec<Vec<OHLCV>> = Vec::new();
    let mut valid_symbols: Vec<String> = Vec::new();

    for symbol in &symbols {
        let filename = format!("{}/{}_60.csv", args.data_dir, symbol);
        if let Ok(data) = load_ohlcv(&filename) {
            if data.len() > args.window + 100 {
                all_data.push(data);
                valid_symbols.push(symbol.clone());
            }
        }
    }

    info!("Using {} symbols for backtest", valid_symbols.len());

    if valid_symbols.len() < 3 {
        anyhow::bail!("Need at least 3 symbols for backtesting");
    }

    // Split data: first 80% for training graph, last 20% for testing
    let test_start = (all_data[0].len() as f64 * 0.8) as usize;
    info!(
        "Test period: {} to {} ({} candles)",
        test_start,
        all_data[0].len(),
        all_data[0].len() - test_start
    );

    // Build initial graph from training period
    let train_returns: Vec<Vec<f64>> = all_data
        .iter()
        .map(|ohlcv| {
            let closes: Vec<f64> = ohlcv[..test_start].iter().map(|o| o.close).collect();
            compute_returns(&closes)
        })
        .collect();

    let graph_builder = CorrelationGraph::new(0.5, args.window);
    let graph = graph_builder.build(&train_returns, &valid_symbols);
    info!("Graph: {} nodes, {} edges", graph.node_count(), graph.edge_count());

    // Create edge index
    let device = Device::Cpu;
    let (sources, targets) = graph.to_edge_index();
    let edge_index = create_edge_index(&sources, &targets, device);

    // Load or create model
    let num_features = 11; // Number of features from FeatureEngineer
    let config = GNNConfig {
        num_features,
        hidden_dim: 64,
        num_classes: 3,
        num_layers: 3,
        dropout: 0.0, // No dropout during inference
        learning_rate: 0.001,
    };

    let mut model = GCN::new(&config, device);

    if std::path::Path::new(&args.model_path).exists() {
        info!("Loading model from {}", args.model_path);
        model.load(&args.model_path)?;
    } else {
        info!("Model not found, using random weights (for demonstration)");
    }

    // Create strategy
    let strategy = MomentumStrategy::new(model, args.threshold);

    // Initialize portfolio
    let mut portfolio = Portfolio::new(
        args.capital,
        args.cost,
        args.max_positions,
        args.position_size,
    );

    let feature_engineer = FeatureEngineer::new(args.window);

    // Run backtest
    info!("\n=== Running Backtest ===");

    let mut total_signals = 0;
    let mut winning_signals = 0;

    for t in test_start..all_data[0].len() {
        // Get current prices
        let current_prices: HashMap<String, f64> = valid_symbols
            .iter()
            .zip(all_data.iter())
            .filter_map(|(sym, data)| {
                data.get(t).map(|ohlcv| (sym.clone(), ohlcv.close))
            })
            .collect();

        // Compute features at current time
        let features: Vec<Vec<f64>> = all_data
            .iter()
            .filter_map(|ohlcv| {
                if t >= args.window && t < ohlcv.len() {
                    let window_data = &ohlcv[t - args.window..t];
                    feature_engineer.compute_features(window_data).ok()
                } else {
                    None
                }
            })
            .collect();

        if features.len() != valid_symbols.len() {
            continue;
        }

        let x = create_features(&features, device);

        // Generate signals
        let signals = strategy.generate_signals(&x, &edge_index, &valid_symbols);
        total_signals += signals.len();

        // Process signals
        for signal in &signals {
            if let Some(&price) = current_prices.get(&signal.symbol) {
                // Check existing positions
                if portfolio.positions.contains_key(&signal.symbol) {
                    // Close if signal direction changed
                    let position = &portfolio.positions[&signal.symbol];
                    if (position.direction > 0 && signal.direction < 0)
                        || (position.direction < 0 && signal.direction > 0)
                    {
                        if let Some(trade) = portfolio.close_position(&signal.symbol, t as i64, price) {
                            if trade.pnl.unwrap_or(0.0) > 0.0 {
                                winning_signals += 1;
                            }
                        }
                    }
                } else {
                    // Open new position
                    portfolio.open_position(
                        &signal.symbol,
                        t as i64,
                        price,
                        signal.direction,
                        signal.confidence,
                    );
                }
            }
        }

        // Close positions with weak signals
        let symbols_to_close: Vec<String> = portfolio
            .positions
            .keys()
            .filter(|sym| {
                !signals.iter().any(|s| &s.symbol == *sym)
            })
            .cloned()
            .collect();

        for symbol in symbols_to_close {
            if let Some(&price) = current_prices.get(&symbol) {
                if let Some(trade) = portfolio.close_position(&symbol, t as i64, price) {
                    if trade.pnl.unwrap_or(0.0) > 0.0 {
                        winning_signals += 1;
                    }
                }
            }
        }

        // Update equity curve
        portfolio.update_equity(&current_prices);
    }

    // Close remaining positions at end
    let final_prices: HashMap<String, f64> = valid_symbols
        .iter()
        .zip(all_data.iter())
        .filter_map(|(sym, data)| {
            data.last().map(|ohlcv| (sym.clone(), ohlcv.close))
        })
        .collect();

    for symbol in portfolio.positions.keys().cloned().collect::<Vec<_>>() {
        if let Some(&price) = final_prices.get(&symbol) {
            if let Some(trade) = portfolio.close_position(&symbol, all_data[0].len() as i64, price) {
                if trade.pnl.unwrap_or(0.0) > 0.0 {
                    winning_signals += 1;
                }
            }
        }
    }

    // Calculate and display metrics
    let metrics = portfolio.calculate_metrics();

    println!("\n{}", metrics);

    println!("\nAdditional Statistics:");
    println!("  Total signals generated: {}", total_signals);
    println!(
        "  Signal win rate: {:.2}%",
        if metrics.num_trades > 0 {
            winning_signals as f64 / metrics.num_trades as f64 * 100.0
        } else {
            0.0
        }
    );

    // Show equity curve summary
    if let (Some(&start), Some(&end)) = (portfolio.equity_curve.first(), portfolio.equity_curve.last()) {
        println!("\nEquity Curve:");
        println!("  Start: ${:.2}", start);
        println!("  End:   ${:.2}", end);
        println!("  Change: ${:.2} ({:+.2}%)", end - start, (end - start) / start * 100.0);
    }

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
