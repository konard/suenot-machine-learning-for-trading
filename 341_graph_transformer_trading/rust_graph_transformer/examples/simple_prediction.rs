//! Example: Simple prediction using Graph Transformer
//!
//! This example demonstrates how to use the Graph Transformer model
//! to make predictions on cryptocurrency price movements.

use graph_transformer_trading::prelude::*;
use graph_transformer_trading::graph::builder::create_test_graph;
use graph_transformer_trading::transformer::model::get_predictions;
use ndarray::Array2;

fn main() {
    println!("=== Graph Transformer Prediction Example ===\n");

    // Create a test graph
    println!("1. Creating market graph...");
    let graph = create_test_graph();
    println!("   Nodes: {}, Edges: {}", graph.num_nodes(), graph.num_edges());

    // Create model with minimal config for demonstration
    println!("\n2. Creating Graph Transformer model...");
    let config = GraphTransformerConfig::minimal();
    println!("   Hidden dim: {}", config.hidden_dim);
    println!("   Num layers: {}", config.num_layers);
    println!("   Num heads: {}", config.num_heads);

    let model = GraphTransformer::new(config.clone());

    // Create random features for demonstration
    println!("\n3. Creating node features...");
    let num_nodes = graph.num_nodes();
    let features = Array2::from_shape_fn((num_nodes, config.input_dim), |_| {
        rand::random::<f64>() * 2.0 - 1.0
    });
    println!("   Feature matrix shape: {:?}", features.shape());

    // Run prediction
    println!("\n4. Running Graph Transformer...");
    let (direction_probs, returns) = model.predict(&features, &graph);

    // Display results
    println!("\n5. Predictions:");
    println!("{:-<70}", "");
    println!(
        "{:<12} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Symbol", "P(Up)", "P(Neutral)", "P(Down)", "Exp.Ret", "Signal"
    );
    println!("{:-<70}", "");

    for (i, node) in graph.nodes.iter().enumerate() {
        let prob_up = direction_probs[[i, 0]];
        let prob_neutral = direction_probs[[i, 1]];
        let prob_down = direction_probs[[i, 2]];
        let exp_return = returns[i];

        let signal = if prob_up > 0.55 {
            "LONG"
        } else if prob_down > 0.55 {
            "SHORT"
        } else {
            "HOLD"
        };

        println!(
            "{:<12} {:>9.2}% {:>9.2}% {:>9.2}% {:>9.2}% {:>10}",
            node.symbol,
            prob_up * 100.0,
            prob_neutral * 100.0,
            prob_down * 100.0,
            exp_return * 100.0,
            signal
        );
    }

    // Get predictions with symbol info
    println!("\n6. Detailed predictions:");
    let predictions = get_predictions(&model, &features, &graph);

    for pred in &predictions {
        println!(
            "\n{} - Predicted: {} (confidence: {:.1}%)",
            pred.symbol,
            pred.predicted_direction(),
            pred.confidence() * 100.0
        );
    }

    // Generate trading signals
    println!("\n7. Trading signals:");
    let signal_gen = SignalGenerator::new(0.55);
    let signals = signal_gen.from_predictions(&predictions);

    if signals.is_empty() {
        println!("   No strong signals detected.");
    } else {
        for signal in &signals {
            println!(
                "   {:?} {} - Confidence: {:.1}%, Expected Return: {:.2}%",
                signal.signal_type,
                signal.symbol,
                signal.confidence * 100.0,
                signal.expected_return * 100.0
            );
        }
    }

    // Show attention weights
    println!("\n8. Attention analysis (first layer):");
    let attention_weights = model.get_attention_weights(&features, &graph);

    if let Some(layer_weights) = attention_weights.first() {
        let mut sorted_weights: Vec<_> = layer_weights.iter().collect();
        sorted_weights.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        println!("   Top 5 attention connections:");
        for ((src, dst), weight) in sorted_weights.iter().take(5) {
            let src_symbol = &graph.nodes[*src].symbol;
            let dst_symbol = &graph.nodes[*dst].symbol;
            println!("   {} -> {}: {:.4}", src_symbol, dst_symbol, weight);
        }
    }

    println!("\n=== Prediction complete! ===");
}
