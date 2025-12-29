//! Example: Build cryptocurrency correlation graph.
//!
//! Usage:
//!   cargo run --example build_graph -- --data-dir data --threshold 0.5

use anyhow::Result;
use clap::Parser;
use gnn_crypto::{
    data::features::{compute_returns, pearson_correlation},
    graph::{detect_lead_lag, CorrelationGraph, CryptoGraph, GraphBuilder, KNNGraph},
    OHLCV,
};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use tracing::info;

#[derive(Parser)]
#[command(name = "build_graph")]
#[command(about = "Build cryptocurrency correlation graph")]
struct Args {
    /// Data directory
    #[arg(short, long, default_value = "data")]
    data_dir: String,

    /// Graph construction method (correlation, knn)
    #[arg(short, long, default_value = "correlation")]
    method: String,

    /// Correlation threshold
    #[arg(short, long, default_value = "0.5")]
    threshold: f64,

    /// Rolling window for correlation
    #[arg(short, long, default_value = "60")]
    window: usize,

    /// Number of neighbors for k-NN
    #[arg(short, long, default_value = "5")]
    k: usize,

    /// Detect lead-lag relationships
    #[arg(long)]
    lead_lag: bool,

    /// Output file for graph edges
    #[arg(short, long)]
    output: Option<String>,
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Load symbol list
    let symbols_file = format!("{}/symbols.txt", args.data_dir);
    let symbols = load_symbols(&symbols_file)?;
    info!("Loaded {} symbols", symbols.len());

    // Load OHLCV data for each symbol
    let mut all_data: Vec<Vec<OHLCV>> = Vec::new();
    let mut valid_symbols: Vec<String> = Vec::new();

    for symbol in &symbols {
        let filename = format!("{}/{}_60.csv", args.data_dir, symbol);
        match load_ohlcv(&filename) {
            Ok(data) => {
                if data.len() > args.window {
                    all_data.push(data);
                    valid_symbols.push(symbol.clone());
                }
            }
            Err(e) => {
                info!("Skipping {}: {}", symbol, e);
            }
        }
    }

    info!("Loaded data for {} symbols", valid_symbols.len());

    // Compute returns
    let returns: Vec<Vec<f64>> = all_data
        .iter()
        .map(|ohlcv| {
            let closes: Vec<f64> = ohlcv.iter().map(|o| o.close).collect();
            compute_returns(&closes)
        })
        .collect();

    // Build graph
    let graph: CryptoGraph = match args.method.as_str() {
        "correlation" => {
            info!(
                "Building correlation graph (threshold={}, window={})",
                args.threshold, args.window
            );
            let builder = CorrelationGraph::new(args.threshold, args.window);
            builder.build(&returns, &valid_symbols)
        }
        "knn" => {
            info!("Building k-NN graph (k={})", args.k);
            let builder = KNNGraph::new(args.k);
            builder.build(&returns, &valid_symbols)
        }
        _ => {
            anyhow::bail!("Unknown method: {}", args.method);
        }
    };

    // Print graph statistics
    println!("\n=== Graph Statistics ===");
    println!("Nodes: {}", graph.node_count());
    println!("Edges: {}", graph.edge_count());
    println!("Density: {:.4}", graph.density());

    // Find hubs
    println!("\n=== Top Hub Nodes ===");
    for (symbol, degree) in graph.find_hubs(10) {
        println!("  {} - degree: {}", symbol, degree);
    }

    // Analyze correlations
    println!("\n=== Strongest Correlations ===");
    let mut correlations: Vec<(String, String, f64)> = Vec::new();
    for i in 0..valid_symbols.len() {
        for j in (i + 1)..valid_symbols.len() {
            let corr = pearson_correlation(&returns[i], &returns[j]);
            correlations.push((valid_symbols[i].clone(), valid_symbols[j].clone(), corr));
        }
    }
    correlations.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());

    for (sym1, sym2, corr) in correlations.iter().take(10) {
        println!("  {} <-> {}: {:.4}", sym1, sym2, corr);
    }

    // Detect lead-lag relationships
    if args.lead_lag {
        println!("\n=== Lead-Lag Relationships ===");
        let pairs = detect_lead_lag(&returns, &valid_symbols, 10);

        for pair in pairs.iter().take(20) {
            println!(
                "  {} -> {} (lag: {}, corr: {:.4})",
                pair.leader, pair.lagger, pair.lag, pair.correlation
            );
        }
    }

    // Save graph edges
    if let Some(output) = args.output {
        save_graph_edges(&graph, &output)?;
        info!("Graph edges saved to {}", output);
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
            continue; // Skip header
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

fn save_graph_edges(graph: &CryptoGraph, path: &str) -> Result<()> {
    use std::io::Write;

    let mut file = File::create(path)?;
    writeln!(file, "source,target,weight")?;

    for symbol in graph.symbols() {
        for neighbor in graph.neighbors(&symbol) {
            if symbol < neighbor {
                // Avoid duplicates
                let weight = graph.edge_weight(&symbol, &neighbor).unwrap_or(0.0);
                writeln!(file, "{},{},{:.6}", symbol, neighbor, weight)?;
            }
        }
    }

    Ok(())
}
