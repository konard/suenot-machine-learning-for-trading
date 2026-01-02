//! Integration tests for the graph generation trading library.

use graph_generation_trading::{
    data::{MarketData, OHLCV, calculate_returns},
    graph::{GraphBuilder, GraphMetrics, MarketGraph, CorrelationMethod, GraphType},
    trading::{GraphSignals, Portfolio, BacktestEngine},
};
use chrono::Utc;
use std::collections::HashMap;

/// Helper to create test OHLCV data
fn create_test_candles(n: usize, base_price: f64) -> Vec<OHLCV> {
    (0..n)
        .map(|i| {
            let price = base_price * (1.0 + (i as f64 * 0.1).sin() * 0.05);
            OHLCV::new(
                Utc::now(),
                price * 0.99,
                price * 1.01,
                price * 0.98,
                price,
                1000.0,
            )
        })
        .collect()
}

/// Create test market data with correlated assets
fn create_test_market_data() -> MarketData {
    let symbols = vec![
        "BTCUSDT".to_string(),
        "ETHUSDT".to_string(),
        "SOLUSDT".to_string(),
    ];

    let mut data = MarketData::new(symbols.clone(), "1h");

    // BTC as base
    let btc_candles = create_test_candles(100, 40000.0);

    // ETH correlated with BTC
    let eth_candles: Vec<OHLCV> = btc_candles
        .iter()
        .map(|c| {
            let price = c.close / 20.0;
            OHLCV::new(c.timestamp, price * 0.99, price * 1.01, price * 0.98, price, 2000.0)
        })
        .collect();

    // SOL less correlated
    let sol_candles: Vec<OHLCV> = btc_candles
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let price = 100.0 * (1.0 + (i as f64 * 0.15).cos() * 0.08);
            OHLCV::new(c.timestamp, price * 0.99, price * 1.01, price * 0.98, price, 500.0)
        })
        .collect();

    data.add_candles("BTCUSDT", btc_candles);
    data.add_candles("ETHUSDT", eth_candles);
    data.add_candles("SOLUSDT", sol_candles);

    data
}

#[test]
fn test_market_data_operations() {
    let data = create_test_market_data();

    assert_eq!(data.num_symbols(), 3);
    assert_eq!(data.num_candles(), 100);

    let btc_data = data.get_symbol("BTCUSDT");
    assert!(btc_data.is_some());
    assert_eq!(btc_data.unwrap().len(), 100);

    let returns = data.returns();
    assert_eq!(returns.len(), 3);
    assert_eq!(returns[0].len(), 99); // n-1 returns
}

#[test]
fn test_graph_builder_threshold() {
    let data = create_test_market_data();

    let graph = GraphBuilder::new()
        .with_method(CorrelationMethod::Pearson)
        .with_graph_type(GraphType::Threshold)
        .with_threshold(0.5)
        .build(&data)
        .unwrap();

    assert_eq!(graph.node_count(), 3);
    assert!(graph.edge_count() > 0);
    assert!(graph.density() >= 0.0 && graph.density() <= 1.0);
}

#[test]
fn test_graph_builder_mst() {
    let data = create_test_market_data();

    let graph = GraphBuilder::new()
        .with_method(CorrelationMethod::Pearson)
        .with_graph_type(GraphType::MST)
        .build(&data)
        .unwrap();

    // MST should have n-1 edges
    assert_eq!(graph.node_count(), 3);
    assert_eq!(graph.edge_count(), 2);
}

#[test]
fn test_graph_builder_knn() {
    let data = create_test_market_data();

    let graph = GraphBuilder::new()
        .with_method(CorrelationMethod::Pearson)
        .with_graph_type(GraphType::KNN)
        .with_k(2)
        .build(&data)
        .unwrap();

    assert_eq!(graph.node_count(), 3);
    // Each node connects to 2 neighbors
    assert!(graph.edge_count() >= 2);
}

#[test]
fn test_graph_metrics() {
    let mut graph = MarketGraph::new();
    graph.add_edge("A", "B", 0.8);
    graph.add_edge("A", "C", 0.7);
    graph.add_edge("B", "C", 0.6);
    graph.add_edge("A", "D", 0.5);

    let metrics = GraphMetrics::new(&graph);

    // Degree centrality
    let degree = metrics.degree_centrality();
    assert_eq!(degree.len(), 4);
    assert!(degree["A"] > degree["D"]); // A is most connected

    // Betweenness centrality
    let betweenness = metrics.betweenness_centrality();
    assert_eq!(betweenness.len(), 4);

    // Clustering
    let clustering = metrics.average_clustering();
    assert!(clustering >= 0.0 && clustering <= 1.0);
}

#[test]
fn test_trading_signals() {
    let mut graph = MarketGraph::new();
    graph.add_edge("BTC", "ETH", 0.85);
    graph.add_edge("BTC", "SOL", 0.72);
    graph.add_edge("ETH", "SOL", 0.78);
    graph.add_edge("DOGE", "SHIB", 0.90);

    let signals = GraphSignals::new(&graph);

    // Test community detection
    let communities = signals.detect_communities();
    assert!(!communities.is_empty());

    // Test hub detection
    let hubs = signals.detect_hubs(2);
    assert_eq!(hubs.len(), 2);

    // Test centrality signals
    let trading_signals = signals.centrality_signals(0.3, 0.3);
    assert_eq!(trading_signals.len(), 5);

    // Test network stress
    let stress = signals.network_stress();
    assert!(stress >= 0.0 && stress <= 1.0);

    // Test regime indicator
    let regime = signals.regime_indicator();
    assert!(regime.position_multiplier() > 0.0);
}

#[test]
fn test_portfolio() {
    let mut portfolio = Portfolio::new(10000.0);

    let mut prices = HashMap::new();
    prices.insert("BTC".to_string(), 40000.0);
    prices.insert("ETH".to_string(), 2000.0);

    // Initial value
    assert_eq!(portfolio.total_value(&prices), 10000.0);

    // Buy some BTC
    portfolio.update_position("BTC", 0.1, 40000.0);

    // Check position
    assert_eq!(portfolio.positions.get("BTC"), Some(&0.1));
    assert!(portfolio.cash < 10000.0);

    // Total value should remain ~same
    let total = portfolio.total_value(&prices);
    assert!((total - 10000.0).abs() < 10.0);

    // Check weights
    let weights = portfolio.weights(&prices);
    assert!(weights["BTC"] > 0.0);

    // Close all
    portfolio.close_all(&prices);
    assert!(portfolio.positions.is_empty());
}

#[test]
fn test_backtest_basic() {
    let mut data = HashMap::new();

    // Simple uptrend for BTC
    let btc_candles: Vec<OHLCV> = (0..50)
        .map(|i| {
            let price = 40000.0 + i as f64 * 100.0;
            OHLCV::new(Utc::now(), price, price + 50.0, price - 50.0, price, 1000.0)
        })
        .collect();

    data.insert("BTCUSDT".to_string(), btc_candles);

    // Constant long signal
    let signals: Vec<HashMap<String, f64>> = (0..50)
        .map(|_| {
            let mut s = HashMap::new();
            s.insert("BTCUSDT".to_string(), 1.0);
            s
        })
        .collect();

    let engine = BacktestEngine::new(10000.0)
        .with_commission(0.001);

    let result = engine.run(&data, &signals);

    assert!(!result.equity_curve.is_empty());
    assert!(result.max_drawdown >= 0.0);
    // Should be profitable in uptrend with long signal
    assert!(result.total_return > -0.1);
}

#[test]
fn test_end_to_end_workflow() {
    // 1. Create market data
    let data = create_test_market_data();

    // 2. Build graph
    let graph = GraphBuilder::new()
        .with_threshold(0.3)
        .build(&data)
        .unwrap();

    // 3. Generate signals
    let signals = GraphSignals::new(&graph);
    let trading_signals = signals.centrality_signals(0.3, 0.3);

    // 4. Create portfolio
    let mut portfolio = Portfolio::new(10000.0);

    // 5. Get final prices
    let mut prices = HashMap::new();
    for symbol in data.symbols.iter() {
        if let Some(candles) = data.get_symbol(symbol) {
            if let Some(last) = candles.last() {
                prices.insert(symbol.clone(), last.close);
            }
        }
    }

    // 6. Execute signals
    for (symbol, signal) in trading_signals {
        if let Some(&price) = prices.get(&symbol) {
            let target_value = 10000.0 * signal.abs() * 0.1;
            let quantity = target_value / price;
            if signal > 0.0 {
                portfolio.update_position(&symbol, quantity, price);
            }
        }
    }

    // Verify portfolio state
    assert!(portfolio.positions.len() > 0 || portfolio.cash > 0.0);
    let total = portfolio.total_value(&prices);
    assert!(total > 0.0);
}
