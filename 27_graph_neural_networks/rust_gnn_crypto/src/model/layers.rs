//! Graph neural network layers.

use tch::{nn, Tensor};

/// Graph Convolutional Layer.
///
/// Implements: H' = σ(D^(-1/2) A D^(-1/2) H W)
pub struct GraphConvLayer {
    linear: nn::Linear,
    bias: Tensor,
}

impl GraphConvLayer {
    /// Create a new GCN layer.
    pub fn new(vs: &nn::Path, in_features: i64, out_features: i64) -> Self {
        let linear = nn::linear(vs / "linear", in_features, out_features, Default::default());
        let bias = vs.zeros("bias", &[out_features]);
        Self { linear, bias }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Node features [num_nodes, in_features]
    /// * `edge_index` - Edge indices [2, num_edges]
    /// * `edge_weight` - Optional edge weights [num_edges]
    pub fn forward(
        &self,
        x: &Tensor,
        edge_index: &Tensor,
        edge_weight: Option<&Tensor>,
    ) -> Tensor {
        let num_nodes = x.size()[0];

        // Transform features
        let h = x.apply(&self.linear);

        // Message passing: aggregate neighbor features
        let source = edge_index.select(0, 0);
        let target = edge_index.select(0, 1);

        // Get source node features
        let source_features = h.index_select(0, &source);

        // Apply edge weights if provided
        let weighted_features = if let Some(weights) = edge_weight {
            let weights_expanded = weights.unsqueeze(-1);
            source_features * weights_expanded
        } else {
            source_features
        };

        // Scatter add to aggregate at target nodes
        let aggregated = Tensor::zeros(&[num_nodes, h.size()[1]], (h.kind(), h.device()));
        let target_expanded = target.unsqueeze(-1).expand_as(&weighted_features);
        let result = aggregated.scatter_add(0, &target_expanded, &weighted_features);

        // Degree normalization (simplified)
        let degree = self.compute_degree(&target, num_nodes);
        let degree_inv_sqrt = (degree + 1e-6).pow_tensor_scalar(-0.5);
        let normalized = result * degree_inv_sqrt.unsqueeze(-1);

        // Add bias
        normalized + &self.bias
    }

    /// Compute node degrees from edge index.
    fn compute_degree(&self, target: &Tensor, num_nodes: i64) -> Tensor {
        let ones = Tensor::ones(&[target.size()[0]], (tch::Kind::Float, target.device()));
        let degree = Tensor::zeros(&[num_nodes], (tch::Kind::Float, target.device()));
        degree.scatter_add(0, target, &ones)
    }
}

/// Graph Attention Layer.
///
/// Implements attention-weighted neighbor aggregation.
pub struct GraphAttentionLayer {
    linear: nn::Linear,
    attention_src: Tensor,
    attention_dst: Tensor,
    num_heads: i64,
    head_dim: i64,
    leaky_relu_slope: f64,
}

impl GraphAttentionLayer {
    /// Create a new GAT layer.
    pub fn new(
        vs: &nn::Path,
        in_features: i64,
        out_features: i64,
        num_heads: i64,
    ) -> Self {
        let head_dim = out_features / num_heads;

        let linear = nn::linear(
            vs / "linear",
            in_features,
            out_features,
            Default::default(),
        );

        // Attention parameters
        let attention_src = vs.zeros("attn_src", &[num_heads, head_dim]);
        let attention_dst = vs.zeros("attn_dst", &[num_heads, head_dim]);

        Self {
            linear,
            attention_src,
            attention_dst,
            num_heads,
            head_dim,
            leaky_relu_slope: 0.2,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: &Tensor, edge_index: &Tensor) -> Tensor {
        let num_nodes = x.size()[0];

        // Linear transformation
        let h = x.apply(&self.linear);

        // Reshape for multi-head attention [num_nodes, num_heads, head_dim]
        let h_reshaped = h.reshape(&[num_nodes, self.num_heads, self.head_dim]);

        // Compute attention scores
        let source = edge_index.select(0, 0);
        let target = edge_index.select(0, 1);

        let h_src = h_reshaped.index_select(0, &source);
        let h_dst = h_reshaped.index_select(0, &target);

        // Attention coefficients
        let alpha_src = (h_src * &self.attention_src).sum_dim_intlist(-1, false, tch::Kind::Float);
        let alpha_dst = (h_dst * &self.attention_dst).sum_dim_intlist(-1, false, tch::Kind::Float);
        let alpha = alpha_src + alpha_dst;

        // LeakyReLU
        let alpha = alpha.leaky_relu_with_negslope(self.leaky_relu_slope);

        // Softmax over neighbors (simplified - using global softmax)
        let alpha = alpha.softmax(-1, tch::Kind::Float);

        // Apply attention weights
        let alpha_expanded = alpha.unsqueeze(-1);
        let weighted = h_src * alpha_expanded;

        // Aggregate
        let output = Tensor::zeros(
            &[num_nodes, self.num_heads, self.head_dim],
            (h.kind(), h.device()),
        );
        let target_expanded = target
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand_as(&weighted);
        let aggregated = output.scatter_add(0, &target_expanded, &weighted);

        // Reshape back [num_nodes, num_heads * head_dim]
        aggregated.reshape(&[num_nodes, self.num_heads * self.head_dim])
    }

    /// Forward pass with attention weights returned.
    pub fn forward_with_attention(&self, x: &Tensor, edge_index: &Tensor) -> (Tensor, Tensor) {
        let num_nodes = x.size()[0];

        // Linear transformation
        let h = x.apply(&self.linear);
        let h_reshaped = h.reshape(&[num_nodes, self.num_heads, self.head_dim]);

        let source = edge_index.select(0, 0);
        let target = edge_index.select(0, 1);

        let h_src = h_reshaped.index_select(0, &source);
        let h_dst = h_reshaped.index_select(0, &target);

        // Attention coefficients
        let alpha_src = (h_src * &self.attention_src).sum_dim_intlist(-1, false, tch::Kind::Float);
        let alpha_dst = (h_dst * &self.attention_dst).sum_dim_intlist(-1, false, tch::Kind::Float);
        let alpha = (alpha_src + alpha_dst).leaky_relu_with_negslope(self.leaky_relu_slope);
        let attention_weights = alpha.softmax(-1, tch::Kind::Float);

        // Apply attention weights
        let alpha_expanded = attention_weights.unsqueeze(-1);
        let weighted = h_src * alpha_expanded;

        // Aggregate
        let output = Tensor::zeros(
            &[num_nodes, self.num_heads, self.head_dim],
            (h.kind(), h.device()),
        );
        let target_expanded = target
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand_as(&weighted);
        let aggregated = output.scatter_add(0, &target_expanded, &weighted);

        let out = aggregated.reshape(&[num_nodes, self.num_heads * self.head_dim]);
        (out, attention_weights)
    }
}

/// Dropout layer for graphs (drops edges).
pub struct GraphDropout {
    p: f64,
}

impl GraphDropout {
    pub fn new(p: f64) -> Self {
        Self { p }
    }

    /// Apply dropout to edge index (randomly remove edges during training).
    pub fn forward(&self, edge_index: &Tensor, training: bool) -> Tensor {
        if !training || self.p == 0.0 {
            return edge_index.shallow_clone();
        }

        let num_edges = edge_index.size()[1];
        let mask = Tensor::rand(&[num_edges], (tch::Kind::Float, edge_index.device()));
        let keep_mask = mask.ge(self.p);

        // Create indices of edges to keep
        let keep_indices = keep_mask.nonzero().squeeze_dim(-1);

        // Select kept edges
        edge_index.index_select(1, &keep_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_gcn_layer() {
        let vs = nn::VarStore::new(Device::Cpu);
        let layer = GraphConvLayer::new(&vs.root(), 10, 32);

        let x = Tensor::randn(&[5, 10], (tch::Kind::Float, Device::Cpu));
        let edge_index = Tensor::from_slice2(&[[0i64, 1, 2, 3], [1, 2, 3, 4]]);

        let output = layer.forward(&x, &edge_index, None);
        assert_eq!(output.size(), vec![5, 32]);
    }

    #[test]
    fn test_gat_layer() {
        let vs = nn::VarStore::new(Device::Cpu);
        let layer = GraphAttentionLayer::new(&vs.root(), 10, 32, 4);

        let x = Tensor::randn(&[5, 10], (tch::Kind::Float, Device::Cpu));
        let edge_index = Tensor::from_slice2(&[[0i64, 1, 2, 3], [1, 2, 3, 4]]);

        let output = layer.forward(&x, &edge_index);
        assert_eq!(output.size(), vec![5, 32]);
    }
}
