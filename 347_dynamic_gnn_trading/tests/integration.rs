//! Integration tests for Dynamic GNN Trading

use dynamic_gnn_trading::graph::{DynamicGraph, GraphConfig, NodeFeatures, EdgeFeatures};
use dynamic_gnn_trading::gnn::{DynamicGNN, GNNConfig};
use dynamic_gnn_trading::strategy::{SignalGenerator, TradingStrategy, StrategyConfig};

#[test]
fn test_full_pipeline() {
    // Create graph
    let config = GraphConfig::default();
    let mut graph = DynamicGraph::with_config(config);

    // Add nodes
    graph.add_node("BTC", NodeFeatures::new(50000.0, 1000000.0, 1000));
    graph.add_node("ETH", NodeFeatures::new(3000.0, 500000.0, 1000));
    graph.add_node("SOL", NodeFeatures::new(100.0, 200000.0, 1000));

    // Add edges
    graph.add_edge("BTC", "ETH", EdgeFeatures::with_correlation(0.85, 1000));
    graph.add_edge("ETH", "SOL", EdgeFeatures::with_correlation(0.75, 1000));

    // Verify graph structure
    assert_eq!(graph.node_count(), 3);
    assert_eq!(graph.edge_count(), 2);

    // Create GNN
    let gnn_config = GNNConfig {
        input_dim: NodeFeatures::feature_dim(),
        hidden_dims: vec![16, 8],
        output_dim: 4,
        num_heads: 2,
        ..Default::default()
    };
    let mut model = DynamicGNN::new(gnn_config);

    // Run forward pass
    let (features, node_ids) = graph.feature_matrix();
    let (adjacency, _) = graph.adjacency_matrix();

    let output = model.forward(&features, &adjacency, None);

    assert_eq!(output.nrows(), 3);
    assert_eq!(output.ncols(), 4);
    assert_eq!(node_ids.len(), 3);
}

#[test]
fn test_signal_generation() {
    let mut gen = SignalGenerator::new();

    // Bullish signal
    let signal = gen.generate("BTCUSDT", 50000.0, (0.1, 0.2, 0.7), 0.8, 1000);
    assert!(signal.is_actionable());

    // Bearish signal
    let signal = gen.generate("BTCUSDT", 50000.0, (0.7, 0.2, 0.1), 0.8, 2000);
    assert!(signal.is_actionable());

    // Neutral signal
    let signal = gen.generate("BTCUSDT", 50000.0, (0.3, 0.4, 0.3), 0.8, 3000);
    assert!(!signal.is_actionable());

    // Check stats
    let stats = gen.stats();
    assert_eq!(stats.total_signals, 3);
}

#[test]
fn test_trading_strategy() {
    let config = StrategyConfig {
        min_confidence: 0.5,
        max_position_pct: 0.1,
        stop_loss_pct: 0.02,
        take_profit_pct: 0.03,
        max_positions: 5,
        trade_cooldown: 60,
        ..Default::default()
    };

    let mut strategy = TradingStrategy::new(config);

    // Strong buy signal should generate order
    let order = strategy.process_predictions(
        "BTCUSDT",
        50000.0,
        (0.1, 0.2, 0.7),
        0.8,
        1000,
    );

    assert!(order.is_some());
}

#[test]
fn test_graph_evolution() {
    let mut graph = DynamicGraph::new();

    // Initial state
    graph.add_node("A", NodeFeatures::new(100.0, 1000.0, 0));
    graph.add_node("B", NodeFeatures::new(200.0, 2000.0, 0));
    graph.add_edge("A", "B", EdgeFeatures::with_correlation(0.9, 0));

    // Update features
    graph.tick(1000);
    graph.update_node("A", NodeFeatures::new(110.0, 1100.0, 1000));
    graph.update_node("B", NodeFeatures::new(190.0, 1900.0, 1000));

    // Take snapshot
    let snapshot = graph.snapshot();
    assert_eq!(snapshot.node_order.len(), 2);

    // Check node was updated
    let node = graph.get_node("A").unwrap();
    assert_eq!(node.features.price, 110.0);
}

#[test]
fn test_gnn_embeddings() {
    let gnn_config = GNNConfig {
        input_dim: 4,
        hidden_dims: vec![8],
        output_dim: 4,
        num_heads: 2,
        ..Default::default()
    };

    let mut model = DynamicGNN::new(gnn_config);

    // Create simple graph data
    let features = ndarray::Array2::from_shape_fn((3, 4), |_| rand::random::<f64>());
    let adjacency = ndarray::Array2::from_shape_fn((3, 3), |(i, j)| {
        if i != j { 0.5 } else { 0.0 }
    });

    let embeddings = model.get_embeddings(&features, &adjacency);

    assert_eq!(embeddings.shape(), &[3, 8]); // Last hidden dim
}
