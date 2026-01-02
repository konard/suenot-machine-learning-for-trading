//! Basic graph construction example.
//!
//! This example demonstrates how to create and analyze a simple market graph.

use graph_generation_trading::graph::{GraphBuilder, GraphMetrics, MarketGraph, CorrelationMethod};

fn main() {
    println!("=== Basic Graph Construction Example ===\n");

    // Create a simple market graph manually
    let mut graph = MarketGraph::new();

    // Add edges with correlation weights
    println!("Adding edges to the graph...");
    graph.add_edge("BTCUSDT", "ETHUSDT", 0.85);
    graph.add_edge("BTCUSDT", "SOLUSDT", 0.72);
    graph.add_edge("ETHUSDT", "SOLUSDT", 0.78);
    graph.add_edge("BTCUSDT", "AVAXUSDT", 0.65);
    graph.add_edge("ETHUSDT", "AVAXUSDT", 0.70);
    graph.add_edge("SOLUSDT", "AVAXUSDT", 0.82);
    graph.add_edge("DOGEUSDT", "SHIBUSDT", 0.90);

    // Display graph statistics
    println!("\n--- Graph Statistics ---");
    println!("Nodes: {}", graph.node_count());
    println!("Edges: {}", graph.edge_count());
    println!("Density: {:.4}", graph.density());
    println!("Average edge weight: {:.4}", graph.average_edge_weight());

    // Get neighbors for each node
    println!("\n--- Node Neighbors ---");
    for symbol in graph.symbols() {
        let neighbors = graph.neighbors(&symbol);
        let degree = graph.degree(&symbol);
        println!("{}: {} neighbors (degree={})", symbol, neighbors.join(", "), degree);
    }

    // Calculate centrality metrics
    println!("\n--- Centrality Metrics ---");
    let metrics = GraphMetrics::new(&graph);

    println!("\nDegree Centrality:");
    let degree_centrality = metrics.degree_centrality();
    let mut sorted: Vec<_> = degree_centrality.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (symbol, centrality) in sorted {
        println!("  {}: {:.4}", symbol, centrality);
    }

    println!("\nBetweenness Centrality:");
    let betweenness = metrics.betweenness_centrality();
    let mut sorted: Vec<_> = betweenness.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (symbol, centrality) in sorted {
        println!("  {}: {:.4}", symbol, centrality);
    }

    // Detect hub nodes
    println!("\n--- Hub Detection ---");
    let hubs = metrics.detect_hubs(3);
    println!("Top 3 hub nodes:");
    for (symbol, score) in hubs {
        println!("  {}: {:.4}", symbol, score);
    }

    // Additional metrics
    println!("\n--- Additional Metrics ---");
    println!("Average clustering coefficient: {:.4}", metrics.average_clustering());
    println!("Graph diameter: {}", metrics.diameter());
    println!("Average path length: {:.4}", metrics.average_path_length());

    // Filter graph by threshold
    println!("\n--- Filtered Graph (threshold=0.75) ---");
    let mut filtered_graph = graph.clone();
    filtered_graph.filter_by_threshold(0.75);
    println!("Edges after filtering: {}", filtered_graph.edge_count());

    for (s1, s2, weight) in filtered_graph.edges() {
        println!("  {} -- {} : {:.2}", s1, s2, weight);
    }

    // Connected components
    println!("\n--- Connected Components ---");
    let components = graph.connected_components();
    println!("Number of components: {}", components.len());
    for (i, component) in components.iter().enumerate() {
        println!("  Component {}: {}", i + 1, component.join(", "));
    }

    println!("\n=== Example Complete ===");
}
