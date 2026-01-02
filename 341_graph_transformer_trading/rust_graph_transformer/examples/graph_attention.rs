//! Example: Graph Attention mechanism visualization
//!
//! This example demonstrates how graph attention works and how
//! to interpret attention weights.

use graph_transformer_trading::prelude::*;
use graph_transformer_trading::graph::builder::create_test_graph;
use graph_transformer_trading::transformer::attention::GraphAttention;
use ndarray::Array2;

fn main() {
    println!("=== Graph Attention Mechanism Example ===\n");

    // Create a test graph
    println!("1. Creating market graph...");
    let graph = create_test_graph();

    println!("\nGraph structure:");
    for edge in &graph.edges {
        println!(
            "   {} -- {} (weight: {:.2}, type: {:?})",
            graph.nodes[edge.source].symbol,
            graph.nodes[edge.target].symbol,
            edge.weight,
            edge.edge_type
        );
    }

    // Create attention layer
    println!("\n2. Creating Graph Attention layer...");
    let hidden_dim = 16;
    let num_heads = 2;
    let attention = GraphAttention::new(hidden_dim, num_heads, 0.0, None);

    println!("   Hidden dim: {}", hidden_dim);
    println!("   Number of heads: {}", num_heads);
    println!("   Head dim: {}", attention.head_dim);

    // Create random node features
    println!("\n3. Creating node features...");
    let num_nodes = graph.num_nodes();
    let features = Array2::from_shape_fn((num_nodes, hidden_dim), |_| {
        rand::random::<f64>() * 2.0 - 1.0
    });

    // Get edge index
    let edge_index = graph.edge_index_coo();
    println!("   Edge index shape: {:?}", edge_index.shape());

    // Compute attention
    println!("\n4. Computing attention weights...");
    let attention_weights = attention.get_attention_weights(&features, &edge_index);

    // Display attention weights as a matrix
    println!("\nAttention weight matrix:");
    println!("{:-<60}", "");

    // Header
    print!("{:>12}", "");
    for node in &graph.nodes {
        print!("{:>10}", &node.symbol[..node.symbol.len().min(8)]);
    }
    println!();

    // Rows
    for i in 0..num_nodes {
        print!("{:>12}", graph.nodes[i].symbol);
        for j in 0..num_nodes {
            if let Some(&weight) = attention_weights.get(&(j, i)) {
                print!("{:>10.4}", weight);
            } else {
                print!("{:>10}", "-");
            }
        }
        println!();
    }

    // Analyze attention patterns
    println!("\n5. Attention pattern analysis:");

    for (i, node) in graph.nodes.iter().enumerate() {
        let incoming: Vec<(&str, f64)> = attention_weights
            .iter()
            .filter(|((_, dst), _)| *dst == i)
            .map(|((src, _), weight)| (graph.nodes[*src].symbol.as_str(), *weight))
            .collect();

        if !incoming.is_empty() {
            println!("\n   {} receives attention from:", node.symbol);
            for (src, weight) in &incoming {
                let bar_len = (weight * 30.0) as usize;
                let bar = "█".repeat(bar_len);
                println!("      {:>10} [{:<30}] {:.4}", src, bar, weight);
            }
        }
    }

    // Forward pass
    println!("\n6. Running attention forward pass...");
    let output = attention.forward(&features, &edge_index, None);
    println!("   Input shape: {:?}", features.shape());
    println!("   Output shape: {:?}", output.shape());

    // Compare input and output for first node
    println!("\n7. Feature transformation (BTC):");
    println!("   Input features (first 8): {:?}",
             &features.row(0).to_vec()[..8.min(hidden_dim)]);
    println!("   Output features (first 8): {:?}",
             &output.row(0).to_vec()[..8.min(hidden_dim)]);

    // Show how attention aggregates information
    println!("\n8. Information aggregation example:");
    println!("   BTC's output is a weighted combination of its neighbors:");

    let btc_idx = 0;
    let neighbors = graph.neighbors(btc_idx);

    for &neighbor_idx in neighbors {
        if let Some(&weight) = attention_weights.get(&(neighbor_idx, btc_idx)) {
            println!(
                "      {:.2}% from {}",
                weight * 100.0,
                graph.nodes[neighbor_idx].symbol
            );
        }
    }

    println!("\n=== Attention analysis complete! ===");
}
