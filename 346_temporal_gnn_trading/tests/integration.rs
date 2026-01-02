//! Integration tests for Temporal GNN Trading

use temporal_gnn_trading::prelude::*;
use temporal_gnn_trading::{TemporalGNN, TGNConfig, MarketEvent};

#[test]
fn test_tgn_full_pipeline() {
    // Create TGN model
    let config = TGNConfig {
        memory_dim: 32,
        time_dim: 16,
        message_dim: 32,
        embedding_dim: 16,
        num_heads: 2,
        ..Default::default()
    };

    let mut tgn = TemporalGNN::new(config);

    // Create sample events
    let events = vec![
        MarketEvent {
            source: "trade_1".to_string(),
            target: "BTCUSDT".to_string(),
            timestamp: 1000,
            features: EventFeatures {
                price: 50000.0,
                volume: 1.0,
                side: 1.0,
                ..Default::default()
            },
        },
        MarketEvent {
            source: "trade_2".to_string(),
            target: "ETHUSDT".to_string(),
            timestamp: 2000,
            features: EventFeatures {
                price: 3000.0,
                volume: 10.0,
                side: 1.0,
                ..Default::default()
            },
        },
        MarketEvent {
            source: "BTCUSDT".to_string(),
            target: "ETHUSDT".to_string(),
            timestamp: 3000,
            features: EventFeatures {
                price: 50100.0,
                volume: 0.5,
                side: 1.0,
                ..Default::default()
            },
        },
    ];

    // Process events
    let signals = tgn.process_batch(&events);

    // Verify model state
    let stats = tgn.stats();
    assert_eq!(stats.num_nodes, 4); // trade_1, trade_2, BTCUSDT, ETHUSDT
    assert!(stats.total_events > 0);

    // Check memory exists for processed nodes
    assert!(tgn.get_memory("BTCUSDT").is_some());
    assert!(tgn.get_memory("ETHUSDT").is_some());

    // Check embeddings can be computed
    let btc_embedding = tgn.get_embedding("BTCUSDT");
    assert!(btc_embedding.is_some());
    assert_eq!(btc_embedding.unwrap().len(), 16); // embedding_dim
}

#[test]
fn test_memory_module() {
    use temporal_gnn_trading::memory::{Memory, MemoryConfig};
    use ndarray::Array1;

    let config = MemoryConfig {
        dim: 16,
        decay_rate: 0.01,
    };

    let mut memory = Memory::new(config);

    // Initialize node
    memory.initialize_node(0);

    // Initial state should be zeros
    let initial_state = memory.get_state(0);
    assert!(initial_state.iter().all(|&x| x == 0.0));

    // Update memory
    let message = Array1::from_vec(vec![1.0; 16]);
    memory.update(0, &message, 1000);

    // State should be updated
    let updated_state = memory.get_state(0);
    assert!(updated_state.iter().any(|&x| x != 0.0));

    // Check time since update
    let time_since = memory.time_since_update(0, 2000);
    assert!((time_since - 1.0).abs() < 0.001); // 1 second
}

#[test]
fn test_message_module() {
    use temporal_gnn_trading::message::{MessageFunction, MessageAggregator, Message, AggregationType};
    use ndarray::Array1;

    // Test message function
    let msg_fn = MessageFunction::new(32, 16);

    let source = Array1::from_vec(vec![1.0; 8]);
    let target = Array1::from_vec(vec![2.0; 8]);
    let time = Array1::from_vec(vec![0.5; 8]);
    let features = Array1::from_vec(vec![0.1; 8]);

    let message = msg_fn.compute(&source, &target, &time, &features);
    assert_eq!(message.len(), 16);

    // Test message aggregator
    let aggregator = MessageAggregator::new(AggregationType::Mean);

    let messages = vec![
        Message::new(0, 1, Array1::from_vec(vec![1.0, 2.0, 3.0]), 100),
        Message::new(2, 1, Array1::from_vec(vec![4.0, 5.0, 6.0]), 200),
    ];

    let aggregated = aggregator.aggregate(&messages);
    assert_eq!(aggregated.len(), 3);
    assert!((aggregated[0] - 2.5).abs() < 0.001);
}

#[test]
fn test_temporal_encoding() {
    use temporal_gnn_trading::temporal::{TimeEncoder, Time2Vec};

    // Test TimeEncoder
    let encoder = TimeEncoder::new(16);
    let encoding = encoder.encode(5000.0);
    assert_eq!(encoding.len(), 16);

    // Check normalization
    let norm: f64 = encoding.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 0.001);

    // Test Time2Vec
    let t2v = Time2Vec::new(8);
    let encoding = t2v.encode(1000.0);
    assert_eq!(encoding.len(), 8);
}

#[test]
fn test_embedding_module() {
    use temporal_gnn_trading::embedding::{GraphConv, NodeEmbedding, EmbeddingConfig};
    use ndarray::Array1;

    // Test GraphConv
    let layer = GraphConv::new(16, 8);
    let features = Array1::from_shape_fn(16, |_| rand::random::<f64>());
    let output = layer.forward_single(&features);
    assert_eq!(output.len(), 8);

    // Test with neighbors
    let neighbors: Vec<Array1<f64>> = (0..3)
        .map(|_| Array1::from_shape_fn(16, |_| rand::random::<f64>()))
        .collect();
    let output = layer.forward_with_neighbors(&features, &neighbors, None);
    assert_eq!(output.len(), 8);

    // Test NodeEmbedding
    let config = EmbeddingConfig {
        input_dim: 16,
        output_dim: 8,
        num_layers: 2,
        skip_connection: true,
        dropout: 0.0,
    };

    let embedding = NodeEmbedding::new(config);
    let output = embedding.embed(&features, &neighbors);
    assert_eq!(output.len(), 8);
}

#[test]
fn test_trading_strategy() {
    use temporal_gnn_trading::strategy::{TradingStrategy, StrategyConfig, Signal, SignalType};

    let config = StrategyConfig {
        min_confidence: 0.5,
        max_position_size: 0.1,
        stop_loss_pct: 0.02,
        take_profit_pct: 0.04,
        cooldown_ms: 0, // No cooldown for testing
        ..Default::default()
    };

    let mut strategy = TradingStrategy::new(config);

    // Create a signal
    let signal = Signal::long("BTCUSDT", 0.8, 0.7, 1000);

    // Process signal
    let action = strategy.process_signal(&signal, 50000.0);
    assert!(action.is_some());

    // Open position
    strategy.open_position("BTCUSDT", SignalType::Long, 0.05, 50000.0, 1000);

    // Check position exists
    assert!(strategy.get_position("BTCUSDT").is_some());

    // Close position
    let record = strategy.close_position("BTCUSDT", 51000.0, 2000, "take_profit");
    assert!(record.is_some());
    assert!(record.unwrap().pnl > 0.0);

    // Check position is closed
    assert!(strategy.get_position("BTCUSDT").is_none());

    // Check metrics
    let metrics = strategy.calculate_metrics();
    assert_eq!(metrics.total_trades, 1);
    assert_eq!(metrics.winning_trades, 1);
}

#[test]
fn test_performance_tracker() {
    use temporal_gnn_trading::utils::PerformanceTracker;

    let mut tracker = PerformanceTracker::new(10000.0, 100);

    // Add some returns
    tracker.update(100.0);  // +1%
    tracker.update(50.0);   // +0.5%
    tracker.update(-75.0);  // -0.75%
    tracker.update(200.0);  // +2%

    // Check equity
    assert!((tracker.current_equity() - 10275.0).abs() < 0.01);

    // Check return
    assert!(tracker.total_return() > 0.0);

    // Check drawdown
    assert!(tracker.max_drawdown() >= 0.0);

    // Check metrics
    let metrics = tracker.get_metrics(0.0, 252.0);
    assert!(metrics.num_periods == 5); // Initial + 4 updates
}

#[test]
fn test_data_structures() {
    use temporal_gnn_trading::data::{Kline, Ticker, OrderBook, OrderBookLevel, Trade};

    // Test Kline
    let kline = Kline {
        timestamp: 1000,
        open: 100.0,
        high: 110.0,
        low: 95.0,
        close: 105.0,
        volume: 1000.0,
        turnover: 100000.0,
    };

    assert_eq!(kline.return_oc(), 0.05);
    assert_eq!(kline.range(), 15.0);
    assert!(kline.is_bullish());

    // Test Ticker
    let ticker = Ticker {
        symbol: "BTCUSDT".to_string(),
        last_price: 50000.0,
        high_24h: 51000.0,
        low_24h: 49000.0,
        volume_24h: 10000.0,
        turnover_24h: 500000000.0,
        price_change_24h: 2.0,
        bid_price: 49995.0,
        ask_price: 50005.0,
        timestamp: 1000,
    };

    assert_eq!(ticker.spread(), 10.0);
    assert_eq!(ticker.mid_price(), 50000.0);

    // Test OrderBook
    let ob = OrderBook::new(
        "BTCUSDT",
        vec![
            OrderBookLevel { price: 49999.0, size: 10.0 },
            OrderBookLevel { price: 49998.0, size: 5.0 },
        ],
        vec![
            OrderBookLevel { price: 50001.0, size: 8.0 },
            OrderBookLevel { price: 50002.0, size: 2.0 },
        ],
    );

    assert_eq!(ob.best_bid(), Some(49999.0));
    assert_eq!(ob.best_ask(), Some(50001.0));
    assert_eq!(ob.spread(), Some(2.0));

    // Test Trade
    let trade = Trade {
        id: "1".to_string(),
        symbol: "BTCUSDT".to_string(),
        price: 50000.0,
        size: 1.0,
        side: "Buy".to_string(),
        timestamp: 1000,
    };

    assert!(trade.is_buy());
    assert_eq!(trade.signed_size(), 1.0);
}

#[test]
fn test_event_features() {
    use temporal_gnn_trading::data::EventFeatures;

    let features = EventFeatures {
        price: 50000.0,
        volume: 100.0,
        side: 1.0,
        price_change: 0.01,
        volume_ratio: 1.5,
        imbalance: 0.3,
        spread: 10.0,
        volatility: 0.02,
    };

    let vector = features.to_vector();
    assert_eq!(vector.len(), 8);
}

#[test]
fn test_rolling_stats() {
    use temporal_gnn_trading::utils::RollingStats;

    let mut stats = RollingStats::new(5);

    // Add values
    for i in 1..=10 {
        stats.update(i as f64);
    }

    // Should only contain last 5 values
    assert!(stats.is_full());
    assert!((stats.mean() - 8.0).abs() < 0.001); // Mean of [6,7,8,9,10]
    assert!(stats.std_dev() > 0.0);
}

#[test]
fn test_ema() {
    use temporal_gnn_trading::utils::EMA;

    let mut ema = EMA::new(5);

    // First value initializes
    ema.update(100.0);
    assert_eq!(ema.value(), 100.0);

    // Subsequent values are smoothed
    ema.update(110.0);
    assert!(ema.value() > 100.0 && ema.value() < 110.0);
}
