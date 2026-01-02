//! E-GNN Configuration
//!
//! Configuration structures for Equivariant GNN models.

use serde::{Deserialize, Serialize};

/// Configuration for E-GNN model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EGNNConfig {
    /// Input feature dimension
    pub input_dim: usize,

    /// Hidden layer dimension
    pub hidden_dim: usize,

    /// Coordinate dimension
    pub coord_dim: usize,

    /// Number of E-GNN layers
    pub num_layers: usize,

    /// Output classes (e.g., 3 for Long/Hold/Short)
    pub output_classes: usize,

    /// Dropout rate
    pub dropout: f64,

    /// Whether to update coordinates
    pub update_coords: bool,

    /// Edge feature dimension
    pub edge_dim: usize,
}

impl Default for EGNNConfig {
    fn default() -> Self {
        Self {
            input_dim: 10,
            hidden_dim: 64,
            coord_dim: 3,
            num_layers: 4,
            output_classes: 3,
            dropout: 0.1,
            update_coords: true,
            edge_dim: 3,
        }
    }
}

impl EGNNConfig {
    /// Create a new configuration
    pub fn new(input_dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            num_layers,
            ..Default::default()
        }
    }

    /// Set coordinate dimension
    pub fn with_coord_dim(mut self, dim: usize) -> Self {
        self.coord_dim = dim;
        self
    }

    /// Set output classes
    pub fn with_output_classes(mut self, classes: usize) -> Self {
        self.output_classes = classes;
        self
    }

    /// Set dropout rate
    pub fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout = rate;
        self
    }
}
