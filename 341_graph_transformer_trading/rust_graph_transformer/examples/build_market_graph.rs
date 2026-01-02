//! Example: Build a market graph from cryptocurrency data
//!
//! This example shows how to construct a market graph where nodes are
//! cryptocurrencies and edges represent relationships between them.

use graph_transformer_trading::prelude::*;
use graph_transformer_trading::graph::builder::create_test_graph;
use graph_transformer_trading::graph::node::CryptoSector;

fn main() {
    println!("=== Building Market Graph ===\n");

    // Method 1: Create a simple test graph
    println!("1. Creating test graph with sample data...\n");
    let graph = create_test_graph();

    println!("Graph Statistics:");
    println!("{}", graph.stats());
    println!();

    // Print nodes
    println!("Nodes (Cryptocurrencies):");
    println!("{:-<60}", "");
    for node in &graph.nodes {
        let sector = CryptoSector::from_symbol(&node.symbol);
        println!(
            "  {} | Price: ${:.2} | Change: {:+.2}% | Sector: {:?}",
            node.symbol, node.price, node.price_change_24h, sector
        );
    }

    // Print edges
    println!("\nEdges (Relationships):");
    println!("{:-<60}", "");
    for edge in &graph.edges {
        let src = &graph.nodes[edge.source].symbol;
        let dst = &graph.nodes[edge.target].symbol;
        println!(
            "  {} <-> {} | Type: {:?} | Weight: {:.3}",
            src, dst, edge.edge_type, edge.weight
        );
    }

    // Method 2: Build from tickers (simulated)
    println!("\n2. Building graph from simulated ticker data...\n");

    let tickers = vec![
        create_mock_ticker("BTCUSDT", 45000.0, 2.5, 1000000.0),
        create_mock_ticker("ETHUSDT", 2500.0, 3.0, 500000.0),
        create_mock_ticker("SOLUSDT", 100.0, 5.0, 200000.0),
        create_mock_ticker("BNBUSDT", 300.0, 1.5, 150000.0),
        create_mock_ticker("XRPUSDT", 0.5, -2.0, 300000.0),
        create_mock_ticker("ADAUSDT", 0.4, -1.5, 250000.0),
        create_mock_ticker("DOGEUSDT", 0.08, 10.0, 100000.0),
        create_mock_ticker("AVAXUSDT", 35.0, 4.0, 80000.0),
    ];

    let builder = MarketGraphBuilder::new()
        .correlation_threshold(0.5)
        .include_sector_edges(true);

    let market_graph = builder.build_from_tickers(&tickers);

    println!("Market Graph Statistics:");
    println!("{}", market_graph.stats());

    // Analyze graph structure
    println!("\nNode Degrees:");
    for node in &market_graph.nodes {
        let degree = market_graph.degree(node.id);
        let neighbors: Vec<&str> = market_graph
            .neighbors(node.id)
            .iter()
            .map(|&i| market_graph.nodes[i].symbol.as_str())
            .collect();

        println!(
            "  {} - Degree: {} - Neighbors: {:?}",
            node.symbol, degree, neighbors
        );
    }

    // Get adjacency matrix
    println!("\nAdjacency Matrix (first 5x5):");
    let adj = market_graph.adjacency_matrix();
    for i in 0..5.min(adj.nrows()) {
        print!("  ");
        for j in 0..5.min(adj.ncols()) {
            print!("{:.2} ", adj[[i, j]]);
        }
        println!();
    }

    // Compute graph metrics
    println!("\nGraph Laplacian (diagonal elements = degrees):");
    let laplacian = market_graph.laplacian_matrix();
    for i in 0..5.min(laplacian.nrows()) {
        println!("  {} - Degree: {:.0}", market_graph.nodes[i].symbol, laplacian[[i, i]]);
    }

    println!("\n=== Graph building complete! ===");
}

fn create_mock_ticker(symbol: &str, price: f64, change: f64, volume: f64) -> graph_transformer_trading::api::types::Ticker {
    graph_transformer_trading::api::types::Ticker {
        symbol: symbol.to_string(),
        last_price: price,
        high_24h: price * 1.05,
        low_24h: price * 0.95,
        price_change_24h: change / 100.0,
        volume_24h: volume,
        turnover_24h: price * volume,
        bid_price: price * 0.999,
        ask_price: price * 1.001,
    }
}
