//! # Temporal Graph Neural Networks for Trading
//!
//! This library implements Temporal GNNs (TGN) for cryptocurrency trading on Bybit.
//! TGN processes continuous-time event streams, maintaining memory of past interactions
//! to make predictions about future market movements.
//!
//! ## Key Features
//!
//! - **Temporal Memory**: Maintains per-node memory states that evolve over time
//! - **Event-Driven Processing**: Handles trades and market events as they occur
//! - **Time Encoding**: Captures temporal patterns using learnable time encodings
//! - **Real-Time Signals**: Generates trading signals with low latency
//!
//! ## Modules
//!
//! - `memory` - Node memory states and update mechanisms
//! - `message` - Message computation and aggregation
//! - `temporal` - Time encoding and temporal attention
//! - `embedding` - Graph convolution and node embeddings
//! - `data` - Bybit API client and data processing
//! - `strategy` - Trading signal generation and execution
//! - `utils` - Utilities and metrics
//!
//! ## Example
//!
//! ```rust,no_run
//! use temporal_gnn_trading::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a TGN model for crypto trading
//!     let config = TGNConfig::default();
//!     let mut tgn = TemporalGNN::new(config);
//!
//!     // Process a market event
//!     let event = MarketEvent {
//!         source: "BTCUSDT".to_string(),
//!         target: "ETHUSDT".to_string(),
//!         timestamp: 1704067200000,
//!         features: EventFeatures::default(),
//!     };
//!
//!     // Update model and get predictions
//!     let signal = tgn.process_event(&event);
//!
//!     Ok(())
//! }
//! ```

pub mod memory;
pub mod message;
pub mod temporal;
pub mod embedding;
pub mod data;
pub mod strategy;
pub mod utils;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::memory::{Memory, MemoryState, MemoryConfig, MemoryUpdater};
    pub use crate::message::{Message, MessageFunction, MessageAggregator};
    pub use crate::temporal::{TimeEncoder, TemporalAttention, Time2Vec};
    pub use crate::embedding::{GraphConv, NodeEmbedding, EmbeddingConfig};
    pub use crate::data::{BybitClient, BybitConfig, MarketEvent, EventFeatures, Kline, Ticker, OrderBook};
    pub use crate::strategy::{Signal, SignalType, TradingStrategy, StrategyConfig};
    pub use crate::utils::{Metrics, PerformanceTracker};
    pub use crate::{TemporalGNN, TGNConfig};
}

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use hashbrown::HashMap as FastHashMap;

/// TGN model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TGNConfig {
    /// Memory dimension per node
    pub memory_dim: usize,
    /// Time encoding dimension
    pub time_dim: usize,
    /// Message function hidden dimension
    pub message_dim: usize,
    /// Embedding output dimension
    pub embedding_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Memory decay rate
    pub memory_decay: f64,
    /// Whether to use time encoding
    pub use_time_encoding: bool,
    /// Number of temporal neighbors to consider
    pub num_neighbors: usize,
    /// Learning rate
    pub learning_rate: f64,
}

impl Default for TGNConfig {
    fn default() -> Self {
        Self {
            memory_dim: 128,
            time_dim: 64,
            message_dim: 128,
            embedding_dim: 64,
            num_heads: 4,
            dropout: 0.1,
            memory_decay: 0.01,
            use_time_encoding: true,
            num_neighbors: 10,
            learning_rate: 0.001,
        }
    }
}

/// Market event representing an interaction in the temporal graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketEvent {
    /// Source node (e.g., trading pair or event type)
    pub source: String,
    /// Target node (e.g., affected asset)
    pub target: String,
    /// Event timestamp in milliseconds
    pub timestamp: u64,
    /// Event features
    pub features: data::EventFeatures,
}

/// Temporal Graph Neural Network for trading
#[derive(Debug)]
pub struct TemporalGNN {
    /// Model configuration
    pub config: TGNConfig,
    /// Node memory states
    memory: memory::Memory,
    /// Message function
    message_fn: message::MessageFunction,
    /// Message aggregator
    aggregator: message::MessageAggregator,
    /// Time encoder
    time_encoder: temporal::TimeEncoder,
    /// Temporal attention
    temporal_attention: temporal::TemporalAttention,
    /// Graph convolution for embeddings
    graph_conv: embedding::GraphConv,
    /// Node ID to index mapping
    node_index: FastHashMap<String, usize>,
    /// Event history for each node
    event_history: FastHashMap<usize, Vec<(u64, Array1<f64>)>>,
    /// Current timestamp
    current_time: u64,
}

impl TemporalGNN {
    /// Create a new Temporal GNN model
    pub fn new(config: TGNConfig) -> Self {
        let memory = memory::Memory::new(memory::MemoryConfig {
            dim: config.memory_dim,
            decay_rate: config.memory_decay,
        });

        let message_fn = message::MessageFunction::new(
            config.memory_dim * 2 + config.time_dim,
            config.message_dim,
        );

        let aggregator = message::MessageAggregator::new(
            message::AggregationType::LastMessage,
        );

        let time_encoder = temporal::TimeEncoder::new(config.time_dim);

        let temporal_attention = temporal::TemporalAttention::new(
            config.memory_dim,
            config.num_heads,
        );

        let graph_conv = embedding::GraphConv::new(
            config.memory_dim,
            config.embedding_dim,
        );

        Self {
            config,
            memory,
            message_fn,
            aggregator,
            time_encoder,
            temporal_attention,
            graph_conv,
            node_index: FastHashMap::new(),
            event_history: FastHashMap::new(),
            current_time: 0,
        }
    }

    /// Get or create node index
    fn get_node_index(&mut self, node_id: &str) -> usize {
        if let Some(&idx) = self.node_index.get(node_id) {
            idx
        } else {
            let idx = self.node_index.len();
            self.node_index.insert(node_id.to_string(), idx);
            self.memory.initialize_node(idx);
            idx
        }
    }

    /// Process a market event and update model state
    pub fn process_event(&mut self, event: &MarketEvent) -> Option<strategy::Signal> {
        // Update current time
        self.current_time = event.timestamp;

        // Get node indices
        let src_idx = self.get_node_index(&event.source);
        let dst_idx = self.get_node_index(&event.target);

        // Get current memory states
        let src_memory = self.memory.get_state(src_idx);
        let dst_memory = self.memory.get_state(dst_idx);

        // Compute time encoding
        let time_delta_src = self.memory.time_since_update(src_idx, event.timestamp);
        let time_delta_dst = self.memory.time_since_update(dst_idx, event.timestamp);
        let time_encoding_src = self.time_encoder.encode(time_delta_src);
        let time_encoding_dst = self.time_encoder.encode(time_delta_dst);

        // Compute messages
        let src_message = self.message_fn.compute(
            &src_memory,
            &dst_memory,
            &time_encoding_src,
            &event.features.to_vector(),
        );

        let dst_message = self.message_fn.compute(
            &dst_memory,
            &src_memory,
            &time_encoding_dst,
            &event.features.to_vector(),
        );

        // Update memory states
        self.memory.update(src_idx, &src_message, event.timestamp);
        self.memory.update(dst_idx, &dst_message, event.timestamp);

        // Store event in history
        self.event_history
            .entry(src_idx)
            .or_insert_with(Vec::new)
            .push((event.timestamp, event.features.to_vector()));
        self.event_history
            .entry(dst_idx)
            .or_insert_with(Vec::new)
            .push((event.timestamp, event.features.to_vector()));

        // Compute embeddings for prediction
        let src_embedding = self.compute_embedding(src_idx);
        let dst_embedding = self.compute_embedding(dst_idx);

        // Generate trading signal
        self.generate_signal(&event.target, &dst_embedding)
    }

    /// Compute node embedding using temporal neighbors
    fn compute_embedding(&self, node_idx: usize) -> Array1<f64> {
        let memory_state = self.memory.get_state(node_idx);

        // Get temporal neighbors
        let neighbors = self.get_temporal_neighbors(node_idx, self.config.num_neighbors);

        if neighbors.is_empty() {
            // No neighbors, just use memory state projected through graph conv
            return self.graph_conv.forward_single(&memory_state);
        }

        // Gather neighbor memories
        let neighbor_memories: Vec<Array1<f64>> = neighbors
            .iter()
            .map(|&(idx, _)| self.memory.get_state(idx))
            .collect();

        // Apply temporal attention
        let aggregated = self.temporal_attention.aggregate(&memory_state, &neighbor_memories);

        // Apply graph convolution
        self.graph_conv.forward_single(&aggregated)
    }

    /// Get temporal neighbors sorted by recency
    fn get_temporal_neighbors(&self, node_idx: usize, k: usize) -> Vec<(usize, u64)> {
        // For simplicity, return nodes that have recent events
        let mut neighbors: Vec<(usize, u64)> = self.memory
            .get_all_last_update_times()
            .into_iter()
            .filter(|&(idx, _)| idx != node_idx)
            .collect();

        // Sort by most recent
        neighbors.sort_by(|a, b| b.1.cmp(&a.1));

        neighbors.into_iter().take(k).collect()
    }

    /// Generate trading signal from embedding
    fn generate_signal(&self, symbol: &str, embedding: &Array1<f64>) -> Option<strategy::Signal> {
        // Simple signal generation based on embedding
        let signal_score: f64 = embedding.iter().sum::<f64>() / embedding.len() as f64;

        // Apply sigmoid-like transformation
        let prob = 1.0 / (1.0 + (-signal_score).exp());

        // Generate signal based on probability
        if prob > 0.6 {
            Some(strategy::Signal {
                symbol: symbol.to_string(),
                signal_type: strategy::SignalType::Long,
                strength: prob,
                confidence: (prob - 0.5).abs() * 2.0,
                timestamp: self.current_time,
            })
        } else if prob < 0.4 {
            Some(strategy::Signal {
                symbol: symbol.to_string(),
                signal_type: strategy::SignalType::Short,
                strength: 1.0 - prob,
                confidence: (prob - 0.5).abs() * 2.0,
                timestamp: self.current_time,
            })
        } else {
            None
        }
    }

    /// Process batch of events
    pub fn process_batch(&mut self, events: &[MarketEvent]) -> Vec<Option<strategy::Signal>> {
        events.iter().map(|e| self.process_event(e)).collect()
    }

    /// Get current memory state for a node
    pub fn get_memory(&self, node_id: &str) -> Option<Array1<f64>> {
        self.node_index.get(node_id).map(|&idx| self.memory.get_state(idx))
    }

    /// Get current embedding for a node
    pub fn get_embedding(&self, node_id: &str) -> Option<Array1<f64>> {
        self.node_index.get(node_id).map(|&idx| self.compute_embedding(idx))
    }

    /// Reset all memory states
    pub fn reset(&mut self) {
        self.memory.reset();
        self.event_history.clear();
        self.current_time = 0;
    }

    /// Get model statistics
    pub fn stats(&self) -> TGNStats {
        TGNStats {
            num_nodes: self.node_index.len(),
            total_events: self.event_history.values().map(|v| v.len()).sum(),
            current_time: self.current_time,
            memory_dim: self.config.memory_dim,
            embedding_dim: self.config.embedding_dim,
        }
    }
}

/// TGN model statistics
#[derive(Debug, Clone)]
pub struct TGNStats {
    /// Number of nodes in the graph
    pub num_nodes: usize,
    /// Total events processed
    pub total_events: usize,
    /// Current model time
    pub current_time: u64,
    /// Memory dimension
    pub memory_dim: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
}

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tgn_creation() {
        let config = TGNConfig::default();
        let tgn = TemporalGNN::new(config);
        assert_eq!(tgn.stats().num_nodes, 0);
    }

    #[test]
    fn test_process_event() {
        let config = TGNConfig::default();
        let mut tgn = TemporalGNN::new(config);

        let event = MarketEvent {
            source: "BTCUSDT".to_string(),
            target: "ETHUSDT".to_string(),
            timestamp: 1704067200000,
            features: data::EventFeatures::default(),
        };

        let _signal = tgn.process_event(&event);
        assert_eq!(tgn.stats().num_nodes, 2);
        assert_eq!(tgn.stats().total_events, 2);
    }

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
