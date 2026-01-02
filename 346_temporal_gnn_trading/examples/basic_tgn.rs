//! Basic Temporal GNN Example
//!
//! This example demonstrates the core functionality of Temporal GNN
//! for processing market events and generating trading signals.

use temporal_gnn_trading::prelude::*;
use temporal_gnn_trading::{TemporalGNN, TGNConfig, MarketEvent};

fn main() {
    println!("=== Temporal GNN Trading - Basic Example ===\n");

    // Create TGN model with default configuration
    let config = TGNConfig {
        memory_dim: 64,
        time_dim: 32,
        message_dim: 64,
        embedding_dim: 32,
        num_heads: 4,
        ..Default::default()
    };

    println!("Creating TGN model with configuration:");
    println!("  Memory dimension: {}", config.memory_dim);
    println!("  Time encoding dimension: {}", config.time_dim);
    println!("  Embedding dimension: {}", config.embedding_dim);
    println!("  Number of attention heads: {}", config.num_heads);
    println!();

    let mut tgn = TemporalGNN::new(config);

    // Simulate a sequence of market events
    let events = generate_sample_events();

    println!("Processing {} market events...\n", events.len());

    let mut signals: Vec<Signal> = Vec::new();

    for (i, event) in events.iter().enumerate() {
        // Process event and get potential signal
        if let Some(signal) = tgn.process_event(event) {
            println!(
                "Event {}: {} -> {} @ t={} => Signal: {:?} (strength: {:.3}, confidence: {:.3})",
                i + 1,
                event.source,
                event.target,
                event.timestamp,
                signal.signal_type,
                signal.strength,
                signal.confidence
            );
            signals.push(signal);
        } else {
            println!(
                "Event {}: {} -> {} @ t={} => No signal",
                i + 1,
                event.source,
                event.target,
                event.timestamp
            );
        }
    }

    println!("\n=== Model Statistics ===");
    let stats = tgn.stats();
    println!("  Number of nodes: {}", stats.num_nodes);
    println!("  Total events processed: {}", stats.total_events);
    println!("  Signals generated: {}", signals.len());

    // Demonstrate memory retrieval
    println!("\n=== Node Memories ===");
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"] {
        if let Some(memory) = tgn.get_memory(symbol) {
            let memory_norm: f64 = memory.iter().map(|x| x * x).sum::<f64>().sqrt();
            println!("  {}: memory norm = {:.4}", symbol, memory_norm);
        }
    }

    // Demonstrate embedding retrieval
    println!("\n=== Node Embeddings ===");
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"] {
        if let Some(embedding) = tgn.get_embedding(symbol) {
            let embedding_norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
            println!("  {}: embedding norm = {:.4}", symbol, embedding_norm);
        }
    }

    println!("\n=== Example Complete ===");
}

/// Generate sample market events for demonstration
fn generate_sample_events() -> Vec<MarketEvent> {
    let base_time = 1704067200000u64; // 2024-01-01 00:00:00 UTC

    vec![
        // BTC large buy triggers market activity
        MarketEvent {
            source: "trade_1".to_string(),
            target: "BTCUSDT".to_string(),
            timestamp: base_time,
            features: EventFeatures {
                price: 42000.0,
                volume: 10.0,
                side: 1.0, // Buy
                price_change: 0.005,
                volume_ratio: 2.5,
                imbalance: 0.3,
                spread: 5.0,
                volatility: 0.02,
            },
        },
        // ETH follows with a buy
        MarketEvent {
            source: "trade_2".to_string(),
            target: "ETHUSDT".to_string(),
            timestamp: base_time + 3000, // 3 seconds later
            features: EventFeatures {
                price: 2200.0,
                volume: 50.0,
                side: 1.0,
                price_change: 0.003,
                volume_ratio: 1.8,
                imbalance: 0.2,
                spread: 0.5,
                volatility: 0.025,
            },
        },
        // SOL reacts
        MarketEvent {
            source: "trade_3".to_string(),
            target: "SOLUSDT".to_string(),
            timestamp: base_time + 5000,
            features: EventFeatures {
                price: 100.0,
                volume: 500.0,
                side: 1.0,
                price_change: 0.008,
                volume_ratio: 3.0,
                imbalance: 0.4,
                spread: 0.1,
                volatility: 0.035,
            },
        },
        // Cross-asset event: BTC affects ETH
        MarketEvent {
            source: "BTCUSDT".to_string(),
            target: "ETHUSDT".to_string(),
            timestamp: base_time + 10000,
            features: EventFeatures {
                price: 42100.0,
                volume: 5.0,
                side: 1.0,
                price_change: 0.002,
                volume_ratio: 1.2,
                imbalance: 0.15,
                spread: 5.0,
                volatility: 0.018,
            },
        },
        // Another BTC trade
        MarketEvent {
            source: "trade_4".to_string(),
            target: "BTCUSDT".to_string(),
            timestamp: base_time + 15000,
            features: EventFeatures {
                price: 42150.0,
                volume: 8.0,
                side: 1.0,
                price_change: 0.001,
                volume_ratio: 1.5,
                imbalance: 0.25,
                spread: 4.5,
                volatility: 0.015,
            },
        },
        // Sell pressure on ETH
        MarketEvent {
            source: "trade_5".to_string(),
            target: "ETHUSDT".to_string(),
            timestamp: base_time + 20000,
            features: EventFeatures {
                price: 2190.0,
                volume: 100.0,
                side: -1.0, // Sell
                price_change: -0.004,
                volume_ratio: 2.2,
                imbalance: -0.3,
                spread: 0.8,
                volatility: 0.028,
            },
        },
        // SOL also sees selling
        MarketEvent {
            source: "trade_6".to_string(),
            target: "SOLUSDT".to_string(),
            timestamp: base_time + 22000,
            features: EventFeatures {
                price: 99.0,
                volume: 800.0,
                side: -1.0,
                price_change: -0.01,
                volume_ratio: 3.5,
                imbalance: -0.4,
                spread: 0.15,
                volatility: 0.04,
            },
        },
        // Market stabilizes - BTC small trade
        MarketEvent {
            source: "trade_7".to_string(),
            target: "BTCUSDT".to_string(),
            timestamp: base_time + 30000,
            features: EventFeatures {
                price: 42050.0,
                volume: 2.0,
                side: 1.0,
                price_change: 0.0005,
                volume_ratio: 0.8,
                imbalance: 0.05,
                spread: 4.0,
                volatility: 0.012,
            },
        },
        // Correlation event
        MarketEvent {
            source: "ETHUSDT".to_string(),
            target: "SOLUSDT".to_string(),
            timestamp: base_time + 35000,
            features: EventFeatures {
                price: 2195.0,
                volume: 30.0,
                side: 1.0,
                price_change: 0.002,
                volume_ratio: 1.1,
                imbalance: 0.1,
                spread: 0.5,
                volatility: 0.022,
            },
        },
        // Final large BTC move
        MarketEvent {
            source: "trade_8".to_string(),
            target: "BTCUSDT".to_string(),
            timestamp: base_time + 40000,
            features: EventFeatures {
                price: 42300.0,
                volume: 15.0,
                side: 1.0,
                price_change: 0.006,
                volume_ratio: 4.0,
                imbalance: 0.5,
                spread: 6.0,
                volatility: 0.025,
            },
        },
    ]
}
