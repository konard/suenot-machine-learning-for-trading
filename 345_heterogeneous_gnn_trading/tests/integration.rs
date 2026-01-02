//! Integration tests for Heterogeneous GNN Trading

use heterogeneous_gnn_trading::prelude::*;
use heterogeneous_gnn_trading::gnn::HGNNConfig;
use heterogeneous_gnn_trading::graph::{HeterogeneousGraph, GraphSchema, NodeType, EdgeType};

/// Create a test graph with multiple node and edge types
fn create_test_graph() -> HeterogeneousGraph {
    let schema = GraphSchema::trading_schema();
    let mut graph = HeterogeneousGraph::new(schema);

    // Add asset nodes
    let assets = [
        ("BTCUSDT", 50000.0, 1_000_000.0, 0.02),
        ("ETHUSDT", 3000.0, 500_000.0, 0.03),
        ("SOLUSDT", 100.0, 200_000.0, 0.05),
    ];

    for (symbol, price, volume, vol) in &assets {
        let features = AssetFeatures::new(*price, *volume, *vol);
        graph.add_node(*symbol, NodeType::Asset, features.into());
    }

    // Add exchange node
    let exchange_features = ExchangeFeatures::new(5_000_000.0, 100, 0.95);
    graph.add_node("Bybit", NodeType::Exchange, exchange_features.into());

    // Add wallet node
    let wallet_features = WalletFeatures::new(10_000_000.0, 500, true);
    graph.add_node("Whale_001", NodeType::Wallet, wallet_features.into());

    // Add correlation edges
    graph.add_edge("BTCUSDT", "ETHUSDT", EdgeType::Correlation, EdgeFeatures::with_correlation(0.85, 1000));
    graph.add_edge("BTCUSDT", "SOLUSDT", EdgeType::Correlation, EdgeFeatures::with_correlation(0.72, 1000));
    graph.add_edge("ETHUSDT", "SOLUSDT", EdgeType::Correlation, EdgeFeatures::with_correlation(0.78, 1000));

    // Add TradesOn edges
    for (symbol, _, _, _) in &assets {
        graph.add_edge(*symbol, "Bybit", EdgeType::TradesOn, EdgeFeatures::with_volume(1_000_000.0, 1000));
    }

    // Add Holds edges
    graph.add_edge("Whale_001", "BTCUSDT", EdgeType::Holds, EdgeFeatures::with_holding(100.0, 1000));
    graph.add_edge("Whale_001", "ETHUSDT", EdgeType::Holds, EdgeFeatures::with_holding(500.0, 1000));

    graph
}

#[test]
fn test_graph_creation() {
    let graph = create_test_graph();

    assert!(graph.num_nodes() > 0);
    assert!(graph.num_edges() > 0);

    let stats = graph.stats();
    assert_eq!(stats.node_count, 5);  // 3 assets + 1 exchange + 1 wallet
    assert!(stats.edge_count >= 6);  // At least 3 correlation + 3 trades_on edges
}

#[test]
fn test_node_types() {
    let graph = create_test_graph();

    let assets = graph.get_nodes_by_type(NodeType::Asset);
    assert_eq!(assets.len(), 3);

    let exchanges = graph.get_nodes_by_type(NodeType::Exchange);
    assert_eq!(exchanges.len(), 1);

    let wallets = graph.get_nodes_by_type(NodeType::Wallet);
    assert_eq!(wallets.len(), 1);
}

#[test]
fn test_edge_types() {
    let graph = create_test_graph();

    let correlation_edges = graph.get_edges_by_type(EdgeType::Correlation);
    assert_eq!(correlation_edges.len(), 3);

    let trades_on_edges = graph.get_edges_by_type(EdgeType::TradesOn);
    assert_eq!(trades_on_edges.len(), 3);

    let holds_edges = graph.get_edges_by_type(EdgeType::Holds);
    assert_eq!(holds_edges.len(), 2);
}

#[test]
fn test_neighbors() {
    let graph = create_test_graph();

    // BTC should have correlation neighbors
    let btc_corr = graph.get_neighbors_by_edge_type("BTCUSDT", EdgeType::Correlation);
    assert_eq!(btc_corr.len(), 2);  // ETH and SOL

    // BTC should have exchange neighbor
    let btc_exchange = graph.get_neighbors_by_edge_type("BTCUSDT", EdgeType::TradesOn);
    assert_eq!(btc_exchange.len(), 1);

    // Whale should have asset neighbors
    let whale_holds = graph.get_neighbors_by_edge_type("Whale_001", EdgeType::Holds);
    assert_eq!(whale_holds.len(), 2);
}

#[test]
fn test_schema_validation() {
    let schema = GraphSchema::trading_schema();

    // Valid relations
    assert!(schema.is_valid_relation(NodeType::Asset, EdgeType::Correlation, NodeType::Asset));
    assert!(schema.is_valid_relation(NodeType::Asset, EdgeType::TradesOn, NodeType::Exchange));
    assert!(schema.is_valid_relation(NodeType::Wallet, EdgeType::Holds, NodeType::Asset));

    // Invalid relations
    assert!(!schema.is_valid_relation(NodeType::Exchange, EdgeType::Correlation, NodeType::Asset));
    assert!(!schema.is_valid_relation(NodeType::Exchange, EdgeType::Holds, NodeType::Asset));
}

#[test]
fn test_metapaths() {
    let schema = GraphSchema::trading_schema();
    let metapaths = schema.metapaths();

    assert!(!metapaths.is_empty());

    // Check for specific metapaths
    let asset_corr = schema.get_metapath("asset_correlation");
    assert!(asset_corr.is_some());

    let same_exchange = schema.get_metapath("same_exchange");
    assert!(same_exchange.is_some());
}

#[test]
fn test_hgnn_model_creation() {
    let config = HGNNConfig::default();
    let model = HeterogeneousGNN::new(config);

    assert!(model.param_count() > 0);
}

#[test]
fn test_hgnn_embeddings() {
    let graph = create_test_graph();
    let config = HGNNConfig::default();
    let model = HeterogeneousGNN::new(config);

    let embeddings = model.get_embeddings(&graph);

    // Should have embeddings for all nodes
    assert!(embeddings.contains_key("BTCUSDT"));
    assert!(embeddings.contains_key("ETHUSDT"));
    assert!(embeddings.contains_key("Bybit"));
    assert!(embeddings.contains_key("Whale_001"));
}

#[test]
fn test_direction_prediction() {
    let graph = create_test_graph();
    let config = HGNNConfig::default();
    let model = HeterogeneousGNN::new(config);

    let embeddings = model.get_embeddings(&graph);

    if let Some(btc_emb) = embeddings.get("BTCUSDT") {
        let (p_down, p_neutral, p_up) = model.predict_direction(btc_emb);

        // Probabilities should sum to approximately 1
        let sum = p_down + p_neutral + p_up;
        assert!((sum - 1.0).abs() < 0.1);

        // All probabilities should be non-negative
        assert!(p_down >= 0.0);
        assert!(p_neutral >= 0.0);
        assert!(p_up >= 0.0);
    }
}

#[test]
fn test_edge_prediction() {
    let graph = create_test_graph();
    let config = HGNNConfig::default();
    let model = HeterogeneousGNN::new(config);

    let embeddings = model.get_embeddings(&graph);

    if let (Some(btc_emb), Some(eth_emb)) = (embeddings.get("BTCUSDT"), embeddings.get("ETHUSDT")) {
        let edge_prob = model.predict_edge(btc_emb, eth_emb);

        // Probability should be between 0 and 1
        assert!(edge_prob >= 0.0 && edge_prob <= 1.0);
    }
}

#[test]
fn test_signal_generation() {
    use heterogeneous_gnn_trading::strategy::{Signal, SignalType, SignalAggregator};

    let signal1 = Signal::new("BTCUSDT", SignalType::Buy, 0.6)
        .with_confidence(0.8)
        .with_source("metapath1");

    let signal2 = Signal::new("BTCUSDT", SignalType::StrongBuy, 0.8)
        .with_confidence(0.7)
        .with_source("metapath2");

    let aggregator = SignalAggregator::new();
    let result = aggregator.aggregate(&[signal1, signal2]);

    assert!(result.is_some());
    let signal = result.unwrap();
    assert_eq!(signal.symbol, "BTCUSDT");
}

#[test]
fn test_performance_metrics() {
    use heterogeneous_gnn_trading::utils::Metrics;

    let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.003];
    let metrics = Metrics::from_returns(&returns, 0.02);

    assert!(metrics.total_return != 0.0);
    assert!(metrics.sharpe_ratio.is_finite());
    assert!(metrics.max_drawdown >= 0.0);
    assert!(metrics.win_rate >= 0.0 && metrics.win_rate <= 1.0);
}

#[test]
fn test_feature_matrices() {
    let graph = create_test_graph();

    let (asset_features, asset_ids) = graph.feature_matrix_by_type(NodeType::Asset);
    assert_eq!(asset_ids.len(), 3);
    assert_eq!(asset_features.nrows(), 3);

    let (exchange_features, exchange_ids) = graph.feature_matrix_by_type(NodeType::Exchange);
    assert_eq!(exchange_ids.len(), 1);
    assert_eq!(exchange_features.nrows(), 1);
}

#[test]
fn test_adjacency_matrix() {
    let graph = create_test_graph();

    let (adj_matrix, node_ids) = graph.adjacency_matrix_by_type(EdgeType::Correlation);
    assert!(!node_ids.is_empty());

    // Matrix should be square
    assert_eq!(adj_matrix.nrows(), adj_matrix.ncols());

    // Correlation is bidirectional, so matrix should be symmetric
    for i in 0..adj_matrix.nrows() {
        for j in 0..adj_matrix.ncols() {
            assert!((adj_matrix[[i, j]] - adj_matrix[[j, i]]).abs() < 1e-10);
        }
    }
}

#[test]
fn test_graph_snapshot() {
    let mut graph = create_test_graph();

    graph.tick(1000);
    let snapshot = graph.snapshot();

    assert_eq!(snapshot.timestamp, 1000);
    assert_eq!(snapshot.node_count, graph.num_nodes());
    assert_eq!(snapshot.edge_count, graph.num_edges());
}
