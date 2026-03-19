//! # GNN LOB Trading Example
//!
//! Fetches the BTCUSDT order book from Bybit, constructs a graph,
//! runs a GNN forward pass, and predicts mid-price direction.

use gnn_lob::{
    BybitClient, GATLayer, LOBGraph, MLPHead, MidPricePredictor, PriceLevel,
    ReadoutMethod, Side, graph_readout,
};

fn main() {
    println!("=== GNN LOB Trading Example ===\n");

    // --- Step 1: Fetch orderbook from Bybit ---
    println!("[1] Fetching BTCUSDT orderbook from Bybit...");
    let client = BybitClient::new();

    let (bids, asks) = match client.fetch_orderbook("BTCUSDT", 50) {
        Ok((b, a)) => {
            println!("    Fetched {} bid levels, {} ask levels", b.len(), a.len());
            (b, a)
        }
        Err(e) => {
            println!("    Failed to fetch from Bybit: {}", e);
            println!("    Using synthetic orderbook data instead.\n");
            generate_synthetic_orderbook()
        }
    };

    // Print top 5 levels
    println!("\n    Top 5 Bids:");
    for level in bids.iter().take(5) {
        println!("      Price: {:.2}  Qty: {:.6}", level.price, level.quantity);
    }
    println!("    Top 5 Asks:");
    for level in asks.iter().take(5) {
        println!("      Price: {:.2}  Qty: {:.6}", level.price, level.quantity);
    }

    let mid_price = if !bids.is_empty() && !asks.is_empty() {
        (bids[0].price + asks[0].price) / 2.0
    } else {
        0.0
    };
    println!("\n    Current mid-price: {:.2}", mid_price);

    // --- Step 2: Build LOB graph ---
    println!("\n[2] Building LOB graph (k=5 nearest neighbors)...");
    let graph = LOBGraph::from_orderbook(&bids, &asks, 5);
    println!("    Nodes: {}", graph.num_nodes);
    println!("    Features per node: {}", graph.num_features);

    let total_edges: f64 = graph.adjacency.iter().sum();
    let self_loops = graph.num_nodes as f64;
    let undirected_edges = (total_edges - self_loops) / 2.0;
    println!("    Edges (undirected, excl. self-loops): {:.0}", undirected_edges);

    let degrees = graph.degree_matrix();
    let avg_degree = degrees.iter().sum::<f64>() / graph.num_nodes as f64;
    println!("    Average degree: {:.2}", avg_degree);

    // --- Step 3: Run GCN-based predictor ---
    println!("\n[3] Running GCN-based MidPricePredictor...");
    let predictor = MidPricePredictor::new(graph.num_features, 16);
    let probs = predictor.predict(&graph);
    let (direction, confidence) = predictor.predict_direction(&graph);

    println!("    Probabilities: DOWN={:.4}  STATIONARY={:.4}  UP={:.4}",
        probs[0], probs[1], probs[2]);
    println!("    Prediction: {} (confidence: {:.4})", direction, confidence);

    // --- Step 4: Run GAT-based pipeline ---
    println!("\n[4] Running GAT-based pipeline (2 heads)...");
    let gat1 = GATLayer::new(graph.num_features, 8, 2);
    let h1 = gat1.forward_mean(&graph.features, &graph.adjacency);

    let gat2 = GATLayer::new(8, 8, 2);
    let h2 = gat2.forward_mean(&h1, &graph.adjacency);

    let graph_emb = graph_readout(&h2, ReadoutMethod::Mean);
    let mlp = MLPHead::new(8, 4);
    let gat_probs = mlp.forward(&graph_emb);

    let labels = ["DOWN", "STATIONARY", "UP"];
    let mut best_idx = 0;
    for i in 1..3 {
        if gat_probs[i] > gat_probs[best_idx] {
            best_idx = i;
        }
    }
    println!("    Probabilities: DOWN={:.4}  STATIONARY={:.4}  UP={:.4}",
        gat_probs[0], gat_probs[1], gat_probs[2]);
    println!("    Prediction: {} (confidence: {:.4})", labels[best_idx], gat_probs[best_idx]);

    // --- Step 5: Compare GCN vs GAT ---
    println!("\n[5] Comparison:");
    println!("    GCN prediction: {} ({:.2}%)", direction, confidence * 100.0);
    println!("    GAT prediction: {} ({:.2}%)", labels[best_idx], gat_probs[best_idx] * 100.0);

    println!("\n=== Done ===");
    println!("\nNote: These are randomly initialized models for demonstration.");
    println!("In production, weights would be trained on historical LOB data.");
}

/// Generate synthetic orderbook data for testing when Bybit API is unavailable.
fn generate_synthetic_orderbook() -> (Vec<PriceLevel>, Vec<PriceLevel>) {
    let base_price = 50000.0;
    let mut bids = Vec::new();
    let mut asks = Vec::new();

    for i in 0..20 {
        bids.push(PriceLevel {
            price: base_price - (i as f64 + 1.0) * 10.0,
            quantity: 0.5 + (i as f64 * 0.3),
            side: Side::Bid,
        });
        asks.push(PriceLevel {
            price: base_price + (i as f64 + 1.0) * 10.0,
            quantity: 0.4 + (i as f64 * 0.25),
            side: Side::Ask,
        });
    }

    (bids, asks)
}
