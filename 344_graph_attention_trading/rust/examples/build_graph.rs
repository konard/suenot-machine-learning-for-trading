//! Build asset relationship graphs
//!
//! This example demonstrates different methods for constructing
//! asset graphs for Graph Attention Networks.
//!
//! Run with: cargo run --example build_graph

use anyhow::Result;
use gat_trading::graph::{GraphBuilder, SparseGraph};
use ndarray::Array2;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

fn main() -> Result<()> {
    println!("=== Asset Graph Construction ===\n");

    // Simulate return data for 10 assets over 200 periods
    let n_assets = 10;
    let n_periods = 200;
    let returns = Array2::random((n_periods, n_assets), StandardNormal);

    let symbols = vec![
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "NEARUSDT",
        "UNIUSDT", "AAVEUSDT", "DOGEUSDT", "SHIBUSDT", "ARBUSDT",
    ];

    // 1. Correlation-based graph
    println!("1. Correlation-based Graph (threshold = 0.3)");
    let corr_graph = GraphBuilder::from_correlation(&returns, 0.3)?;
    println!("   Nodes: {}, Edges: {}", corr_graph.num_nodes(), corr_graph.num_edges());
    println!("   Density: {:.2}%", corr_graph.density() * 100.0);

    // Show connections
    println!("   Connections:");
    for (i, symbol) in symbols.iter().enumerate() {
        let neighbors = corr_graph.neighbors(i);
        if !neighbors.is_empty() {
            let neighbor_symbols: Vec<&str> = neighbors
                .iter()
                .map(|&j| symbols[j])
                .collect();
            println!("   {} -> {:?}", symbol, neighbor_symbols);
        }
    }

    // 2. k-NN graph
    println!("\n2. k-NN Graph (k = 3)");
    let knn_graph = GraphBuilder::from_knn(&returns, 3)?;
    println!("   Nodes: {}, Edges: {}", knn_graph.num_nodes(), knn_graph.num_edges());
    println!("   Each node has exactly {} neighbors", knn_graph.degree(0));

    // 3. Sector-based graph
    println!("\n3. Sector-based Graph");
    let symbol_refs: Vec<&str> = symbols.iter().map(|s| s.as_str()).collect();
    let sector_graph = GraphBuilder::from_sectors(&symbol_refs)?;
    println!("   Nodes: {}, Edges: {}", sector_graph.num_nodes(), sector_graph.num_edges());

    println!("   Sector connections:");
    // Layer 1: BTC, ETH, SOL, AVAX, NEAR
    println!("   Layer 1 (BTC, ETH, SOL, AVAX, NEAR): all connected");
    // DeFi: UNI, AAVE
    println!("   DeFi (UNI, AAVE): connected");
    // Meme: DOGE, SHIB
    println!("   Meme (DOGE, SHIB): connected");

    // 4. Fully connected graph
    println!("\n4. Fully Connected Graph");
    let full_graph = GraphBuilder::fully_connected(n_assets);
    println!("   Nodes: {}, Edges: {}", full_graph.num_nodes(), full_graph.num_edges());
    println!("   Density: {:.2}% (fully connected)", full_graph.density() * 100.0);

    // 5. Hybrid graph
    println!("\n5. Hybrid Graph (correlation + sector)");
    let hybrid_graph = GraphBuilder::hybrid(&returns, &symbol_refs, 0.2, 0.3)?;
    println!("   Nodes: {}, Edges: {}", hybrid_graph.num_nodes(), hybrid_graph.num_edges());
    println!("   Density: {:.2}%", hybrid_graph.density() * 100.0);

    // Graph operations
    println!("\n=== Graph Operations ===");

    // Make symmetric
    let sym_graph = corr_graph.make_symmetric();
    println!("Symmetric graph edges: {} (was {})", sym_graph.num_edges(), corr_graph.num_edges());

    // Add self-loops
    let loop_graph = corr_graph.add_self_loops(1.0);
    println!("Graph with self-loops: {} edges", loop_graph.num_edges());

    // Normalize
    let norm_graph = corr_graph.normalize();
    println!("Normalized graph: row sums = 1.0");

    // Subgraph
    let subset = vec![0, 1, 2]; // BTC, ETH, SOL
    let subgraph = corr_graph.subgraph(&subset);
    println!(
        "Subgraph (BTC, ETH, SOL): {} nodes, {} edges",
        subgraph.num_nodes(),
        subgraph.num_edges()
    );

    // Convert to dense
    println!("\n=== Dense Adjacency Matrix (first 5x5) ===");
    let dense = corr_graph.to_dense();
    for i in 0..5 {
        let row: Vec<String> = (0..5)
            .map(|j| format!("{:.1}", dense[[i, j]]))
            .collect();
        println!("   {}", row.join(" "));
    }

    println!("\n=== Done ===");

    Ok(())
}
