//! Temporal graph models for time-varying market graphs.

use super::{FeatureMatrix, ModelOutput};
use crate::graph::MarketGraph;

/// Temporal graph snapshot
#[derive(Debug, Clone)]
pub struct GraphSnapshot {
    /// The graph at this time step
    pub graph: MarketGraph,
    /// Node features at this time step
    pub features: FeatureMatrix,
    /// Timestamp index
    pub time_idx: usize,
}

/// Temporal Graph Model combining spatial and temporal patterns
#[derive(Debug)]
pub struct TemporalGraphModel {
    /// Hidden state dimension
    hidden_dim: usize,
    /// Number of GCN layers
    num_gcn_layers: usize,
    /// Sequence length for temporal modeling
    seq_len: usize,
    /// Current hidden states per node
    hidden_states: Vec<Vec<f64>>,
}

impl TemporalGraphModel {
    /// Create a new temporal graph model
    pub fn new(hidden_dim: usize, num_gcn_layers: usize, seq_len: usize) -> Self {
        Self {
            hidden_dim,
            num_gcn_layers,
            seq_len,
            hidden_states: Vec::new(),
        }
    }

    /// Process a sequence of graph snapshots
    pub fn forward(&mut self, snapshots: &[GraphSnapshot]) -> ModelOutput {
        if snapshots.is_empty() {
            return ModelOutput::new();
        }

        let last_snapshot = snapshots.last().unwrap();
        let symbols = last_snapshot.graph.symbols();
        let n = symbols.len();

        // Initialize hidden states if needed
        if self.hidden_states.len() != n {
            self.hidden_states = vec![vec![0.0; self.hidden_dim]; n];
        }

        // Process each snapshot
        for snapshot in snapshots {
            self.process_snapshot(snapshot);
        }

        // Generate predictions from final hidden states
        let mut output = ModelOutput::new();
        for (i, symbol) in symbols.iter().enumerate() {
            if i < self.hidden_states.len() && !self.hidden_states[i].is_empty() {
                // Simple prediction: sum of hidden state
                let prediction: f64 = self.hidden_states[i].iter().sum::<f64>()
                    / self.hidden_states[i].len() as f64;
                let confidence = prediction.abs().tanh();
                output.set_prediction(symbol, prediction, confidence);
            }
        }

        output
    }

    /// Process a single snapshot and update hidden states
    fn process_snapshot(&mut self, snapshot: &GraphSnapshot) {
        let symbols = snapshot.graph.symbols();
        let adjacency = snapshot.graph.adjacency_matrix();
        let n = symbols.len();

        // Ensure hidden states match current graph size
        while self.hidden_states.len() < n {
            self.hidden_states.push(vec![0.0; self.hidden_dim]);
        }

        // Get current features
        let features = snapshot.features.to_matrix(&symbols);

        // Simple GRU-like update for each node
        for i in 0..n {
            let mut aggregated = vec![0.0; self.hidden_dim];
            let mut neighbor_count = 0.0;

            // Aggregate neighbor hidden states
            for j in 0..n {
                if adjacency[i][j] > 0.0 || i == j {
                    let weight = if i == j { 1.0 } else { adjacency[i][j] };
                    neighbor_count += weight;

                    for k in 0..self.hidden_dim.min(self.hidden_states[j].len()) {
                        aggregated[k] += weight * self.hidden_states[j][k];
                    }
                }
            }

            // Normalize
            if neighbor_count > 0.0 {
                for val in &mut aggregated {
                    *val /= neighbor_count;
                }
            }

            // GRU-like update: h = (1-z)*h + z*tanh(Wh*agg + Wx*x)
            let z = 0.5; // Update gate (simplified)
            for k in 0..self.hidden_dim {
                let input_contrib = if k < features[i].len() {
                    features[i][k]
                } else {
                    0.0
                };

                let new_val = (aggregated[k] + input_contrib).tanh();
                self.hidden_states[i][k] = (1.0 - z) * self.hidden_states[i][k] + z * new_val;
            }
        }
    }

    /// Reset hidden states
    pub fn reset(&mut self) {
        self.hidden_states.clear();
    }

    /// Get current hidden states
    pub fn get_hidden_states(&self) -> &Vec<Vec<f64>> {
        &self.hidden_states
    }
}

/// Create graph snapshots from time series data
pub fn create_snapshots(
    graphs: Vec<MarketGraph>,
    features: Vec<FeatureMatrix>,
) -> Vec<GraphSnapshot> {
    graphs
        .into_iter()
        .zip(features.into_iter())
        .enumerate()
        .map(|(i, (graph, feat))| GraphSnapshot {
            graph,
            features: feat,
            time_idx: i,
        })
        .collect()
}

/// Rolling window graph generator
pub struct RollingGraphGenerator {
    window_size: usize,
    step_size: usize,
}

impl RollingGraphGenerator {
    pub fn new(window_size: usize, step_size: usize) -> Self {
        Self {
            window_size,
            step_size,
        }
    }

    /// Generate rolling window indices
    pub fn generate_windows(&self, total_len: usize) -> Vec<(usize, usize)> {
        let mut windows = Vec::new();
        let mut start = 0;

        while start + self.window_size <= total_len {
            windows.push((start, start + self.window_size));
            start += self.step_size;
        }

        windows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_model() {
        let mut graph1 = MarketGraph::new();
        graph1.add_edge("BTC", "ETH", 0.8);

        let mut features1 = FeatureMatrix::new(3);
        features1.set("BTC", vec![0.1, 0.2, 0.3]);
        features1.set("ETH", vec![0.2, 0.3, 0.4]);

        let snapshots = vec![GraphSnapshot {
            graph: graph1,
            features: features1,
            time_idx: 0,
        }];

        let mut model = TemporalGraphModel::new(4, 2, 10);
        let output = model.forward(&snapshots);

        assert!(output.predictions.contains_key("BTC"));
    }

    #[test]
    fn test_rolling_windows() {
        let generator = RollingGraphGenerator::new(10, 5);
        let windows = generator.generate_windows(30);

        assert_eq!(windows.len(), 5); // (0,10), (5,15), (10,20), (15,25), (20,30)
    }
}
