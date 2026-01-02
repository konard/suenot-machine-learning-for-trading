//! Message module for TGN
//!
//! This module implements message computation and aggregation
//! for passing information between nodes during events.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Message aggregation types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AggregationType {
    /// Use only the most recent message
    LastMessage,
    /// Average all messages
    Mean,
    /// Sum all messages
    Sum,
    /// Attention-weighted aggregation
    Attention,
}

/// A single message between nodes
#[derive(Debug, Clone)]
pub struct Message {
    /// Source node index
    pub source: usize,
    /// Target node index
    pub target: usize,
    /// Message vector
    pub vector: Array1<f64>,
    /// Timestamp
    pub timestamp: u64,
}

impl Message {
    /// Create a new message
    pub fn new(source: usize, target: usize, vector: Array1<f64>, timestamp: u64) -> Self {
        Self {
            source,
            target,
            vector,
            timestamp,
        }
    }
}

/// Message function for computing messages from events
#[derive(Debug)]
pub struct MessageFunction {
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// First layer weights
    w1: Array2<f64>,
    /// First layer bias
    b1: Array1<f64>,
    /// Second layer weights
    w2: Array2<f64>,
    /// Second layer bias
    b2: Array1<f64>,
}

impl MessageFunction {
    /// Create a new message function
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let hidden_dim = (input_dim + output_dim) / 2;

        Self {
            input_dim,
            output_dim,
            w1: Array2::from_shape_fn((input_dim, hidden_dim), |_| {
                rand::random::<f64>() * 0.1 - 0.05
            }),
            b1: Array1::zeros(hidden_dim),
            w2: Array2::from_shape_fn((hidden_dim, output_dim), |_| {
                rand::random::<f64>() * 0.1 - 0.05
            }),
            b2: Array1::zeros(output_dim),
        }
    }

    /// Compute message from source memory, target memory, time encoding, and event features
    pub fn compute(
        &self,
        source_memory: &Array1<f64>,
        target_memory: &Array1<f64>,
        time_encoding: &Array1<f64>,
        event_features: &Array1<f64>,
    ) -> Array1<f64> {
        // Concatenate all inputs
        let total_dim = source_memory.len() + target_memory.len() +
                        time_encoding.len() + event_features.len();

        // Pad or truncate to match input dimension
        let mut input = Array1::zeros(self.input_dim);
        let mut offset = 0;

        for arr in [source_memory, target_memory, time_encoding, event_features] {
            let copy_len = arr.len().min(self.input_dim - offset);
            if copy_len > 0 && offset < self.input_dim {
                input.slice_mut(ndarray::s![offset..offset + copy_len])
                    .assign(&arr.slice(ndarray::s![..copy_len]));
                offset += copy_len;
            }
            if offset >= self.input_dim {
                break;
            }
        }

        // First layer with ReLU
        let h1 = input.dot(&self.w1) + &self.b1;
        let h1_relu: Array1<f64> = h1.mapv(|x| x.max(0.0));

        // Second layer
        let output = h1_relu.dot(&self.w2) + &self.b2;

        output
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
}

/// Message aggregator for combining multiple messages
#[derive(Debug)]
pub struct MessageAggregator {
    /// Aggregation type
    aggregation_type: AggregationType,
    /// Attention weights (if using attention)
    attention_weights: Option<Array2<f64>>,
}

impl MessageAggregator {
    /// Create a new message aggregator
    pub fn new(aggregation_type: AggregationType) -> Self {
        Self {
            aggregation_type,
            attention_weights: None,
        }
    }

    /// Initialize attention weights if needed
    pub fn init_attention(&mut self, dim: usize) {
        if matches!(self.aggregation_type, AggregationType::Attention) {
            self.attention_weights = Some(Array2::from_shape_fn((dim, dim), |_| {
                rand::random::<f64>() * 0.1 - 0.05
            }));
        }
    }

    /// Aggregate multiple messages for a node
    pub fn aggregate(&self, messages: &[Message]) -> Array1<f64> {
        if messages.is_empty() {
            return Array1::zeros(0);
        }

        match self.aggregation_type {
            AggregationType::LastMessage => {
                // Return the most recent message
                messages
                    .iter()
                    .max_by_key(|m| m.timestamp)
                    .map(|m| m.vector.clone())
                    .unwrap()
            }
            AggregationType::Mean => {
                // Average all messages
                let sum: Array1<f64> = messages
                    .iter()
                    .fold(Array1::zeros(messages[0].vector.len()), |acc, m| acc + &m.vector);
                sum / messages.len() as f64
            }
            AggregationType::Sum => {
                // Sum all messages
                messages
                    .iter()
                    .fold(Array1::zeros(messages[0].vector.len()), |acc, m| acc + &m.vector)
            }
            AggregationType::Attention => {
                // Attention-weighted aggregation
                self.attention_aggregate(messages)
            }
        }
    }

    /// Attention-based aggregation
    fn attention_aggregate(&self, messages: &[Message]) -> Array1<f64> {
        if messages.is_empty() {
            return Array1::zeros(0);
        }

        let dim = messages[0].vector.len();

        // Simple attention: use softmax over message norms
        let scores: Vec<f64> = messages
            .iter()
            .map(|m| m.vector.dot(&m.vector).sqrt())
            .collect();

        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();
        let weights: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        // Weighted sum
        let mut result = Array1::zeros(dim);
        for (msg, &weight) in messages.iter().zip(weights.iter()) {
            result = result + &msg.vector * weight;
        }

        result
    }

    /// Aggregate messages grouped by target node
    pub fn aggregate_by_target(&self, messages: &[Message]) -> std::collections::HashMap<usize, Array1<f64>> {
        let mut grouped: std::collections::HashMap<usize, Vec<&Message>> = std::collections::HashMap::new();

        for msg in messages {
            grouped.entry(msg.target).or_insert_with(Vec::new).push(msg);
        }

        grouped
            .into_iter()
            .map(|(target, msgs)| {
                let owned_msgs: Vec<Message> = msgs.into_iter().cloned().collect();
                (target, self.aggregate(&owned_msgs))
            })
            .collect()
    }
}

/// Raw message store for batching
#[derive(Debug, Default)]
pub struct MessageStore {
    /// Stored messages
    messages: Vec<Message>,
}

impl MessageStore {
    /// Create new message store
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// Add a message
    pub fn add(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Get all messages for a target node
    pub fn get_for_target(&self, target: usize) -> Vec<&Message> {
        self.messages
            .iter()
            .filter(|m| m.target == target)
            .collect()
    }

    /// Get all messages since a timestamp
    pub fn get_since(&self, timestamp: u64) -> Vec<&Message> {
        self.messages
            .iter()
            .filter(|m| m.timestamp >= timestamp)
            .collect()
    }

    /// Clear all messages
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Get number of stored messages
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_function() {
        let msg_fn = MessageFunction::new(16, 8);

        let source = Array1::from_vec(vec![1.0; 4]);
        let target = Array1::from_vec(vec![2.0; 4]);
        let time = Array1::from_vec(vec![0.5; 4]);
        let features = Array1::from_vec(vec![0.1; 4]);

        let output = msg_fn.compute(&source, &target, &time, &features);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_message_aggregation() {
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
    fn test_last_message_aggregation() {
        let aggregator = MessageAggregator::new(AggregationType::LastMessage);

        let messages = vec![
            Message::new(0, 1, Array1::from_vec(vec![1.0, 2.0, 3.0]), 100),
            Message::new(2, 1, Array1::from_vec(vec![4.0, 5.0, 6.0]), 200),
        ];

        let aggregated = aggregator.aggregate(&messages);
        assert_eq!(aggregated[0], 4.0); // Should be the message with timestamp 200
    }

    #[test]
    fn test_message_store() {
        let mut store = MessageStore::new();

        store.add(Message::new(0, 1, Array1::from_vec(vec![1.0]), 100));
        store.add(Message::new(2, 1, Array1::from_vec(vec![2.0]), 200));
        store.add(Message::new(0, 2, Array1::from_vec(vec![3.0]), 150));

        assert_eq!(store.len(), 3);
        assert_eq!(store.get_for_target(1).len(), 2);
        assert_eq!(store.get_since(150).len(), 2);
    }
}
