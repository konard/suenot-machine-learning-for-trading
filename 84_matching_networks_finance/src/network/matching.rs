//! Main Matching Network implementation
//!
//! Combines embedding network, attention module, and optional FCE
//! for end-to-end pattern classification.

use ndarray::{Array1, Array2};
use super::{EmbeddingNetwork, AttentionModule, FullContextEmbedding, DistanceFunction};

/// Matching Network for few-shot classification
///
/// The network computes:
/// ```text
/// P(ŷ|x̂,S) = Σᵢ a(x̂,xᵢ)yᵢ
/// ```
///
/// Where attention weights are computed as softmax of similarities
/// between query and support set embeddings.
#[derive(Debug, Clone)]
pub struct MatchingNetwork {
    /// Embedding network for feature transformation
    embedding_network: EmbeddingNetwork,
    /// Attention module for weighted voting
    attention_module: AttentionModule,
    /// Optional Full Context Embedding
    fce: Option<FullContextEmbedding>,
    /// Number of classes (set dynamically)
    num_classes: Option<usize>,
}

impl MatchingNetwork {
    /// Create a new Matching Network
    ///
    /// # Arguments
    /// * `input_dim` - Dimension of input features
    /// * `hidden_dim` - Hidden layer dimension
    /// * `embedding_dim` - Output embedding dimension
    /// * `use_fce` - Whether to use Full Context Embeddings
    ///
    /// # Example
    /// ```
    /// use matching_networks_finance::MatchingNetwork;
    ///
    /// let network = MatchingNetwork::new(20, 64, 32, true);
    /// ```
    pub fn new(input_dim: usize, hidden_dim: usize, embedding_dim: usize, use_fce: bool) -> Self {
        let embedding_network = EmbeddingNetwork::new(
            input_dim,
            &[hidden_dim, hidden_dim],
            embedding_dim,
            true, // use batch normalization
        );

        let attention_module = AttentionModule::new(DistanceFunction::Cosine, 1.0);

        let fce = if use_fce {
            Some(FullContextEmbedding::new(embedding_dim, 3))
        } else {
            None
        };

        Self {
            embedding_network,
            attention_module,
            fce,
            num_classes: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        input_dim: usize,
        hidden_dims: &[usize],
        embedding_dim: usize,
        distance_fn: DistanceFunction,
        temperature: f64,
        use_fce: bool,
        fce_steps: usize,
    ) -> Self {
        let embedding_network = EmbeddingNetwork::new(
            input_dim,
            hidden_dims,
            embedding_dim,
            true,
        );

        let attention_module = AttentionModule::new(distance_fn, temperature);

        let fce = if use_fce {
            Some(FullContextEmbedding::new(embedding_dim, fce_steps))
        } else {
            None
        };

        Self {
            embedding_network,
            attention_module,
            fce,
            num_classes: None,
        }
    }

    /// Forward pass for classification
    ///
    /// # Arguments
    /// * `support_features` - Support set features [num_support, input_dim]
    /// * `support_labels` - Support set labels [num_support]
    /// * `query_features` - Query features [num_queries, input_dim]
    ///
    /// # Returns
    /// Class probabilities [num_queries, num_classes]
    pub fn forward(
        &self,
        support_features: &Array2<f64>,
        support_labels: &Array1<usize>,
        query_features: &Array2<f64>,
    ) -> Array2<f64> {
        // Compute basic embeddings
        let mut support_embeddings = self.embedding_network.forward(support_features);
        let mut query_embeddings = self.embedding_network.forward(query_features);

        // Apply FCE if enabled
        if let Some(fce) = &self.fce {
            support_embeddings = fce.encode_support(&support_embeddings);
            query_embeddings = fce.encode_query(&query_embeddings, &support_embeddings);
        }

        // Compute attention weights
        let attention = self.attention_module.compute_attention(
            &query_embeddings,
            &support_embeddings,
        );

        // Aggregate predictions by class
        self.aggregate_by_class(&attention, support_labels)
    }

    /// Classify queries and return predicted class indices
    ///
    /// # Arguments
    /// * `support_features` - Support set features
    /// * `support_labels` - Support set labels
    /// * `query_features` - Query features
    ///
    /// # Returns
    /// Predicted class indices for each query
    pub fn predict(
        &self,
        support_features: &Array2<f64>,
        support_labels: &Array1<usize>,
        query_features: &Array2<f64>,
    ) -> Array1<usize> {
        let probabilities = self.forward(support_features, support_labels, query_features);

        // Find argmax for each query
        let mut predictions = Array1::zeros(probabilities.nrows());
        for (i, row) in probabilities.rows().into_iter().enumerate() {
            let (max_idx, _) = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            predictions[i] = max_idx;
        }

        predictions
    }

    /// Predict with confidence scores
    ///
    /// # Returns
    /// Tuple of (predictions, confidences)
    pub fn predict_with_confidence(
        &self,
        support_features: &Array2<f64>,
        support_labels: &Array1<usize>,
        query_features: &Array2<f64>,
    ) -> (Array1<usize>, Array1<f64>) {
        let probabilities = self.forward(support_features, support_labels, query_features);

        let mut predictions = Array1::zeros(probabilities.nrows());
        let mut confidences = Array1::zeros(probabilities.nrows());

        for (i, row) in probabilities.rows().into_iter().enumerate() {
            let (max_idx, &max_prob) = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            predictions[i] = max_idx;
            confidences[i] = max_prob;
        }

        (predictions, confidences)
    }

    /// Aggregate attention weights by class
    fn aggregate_by_class(
        &self,
        attention: &Array2<f64>,
        labels: &Array1<usize>,
    ) -> Array2<f64> {
        let num_queries = attention.nrows();
        let num_classes = labels.iter().max().map(|&m| m + 1).unwrap_or(0);

        let mut class_probs = Array2::zeros((num_queries, num_classes));

        for (q, attention_row) in attention.rows().into_iter().enumerate() {
            for (s, &attention_weight) in attention_row.iter().enumerate() {
                let class_idx = labels[s];
                class_probs[[q, class_idx]] += attention_weight;
            }
        }

        class_probs
    }

    /// Get embeddings for support set
    pub fn embed_support(&self, support_features: &Array2<f64>) -> Array2<f64> {
        let mut embeddings = self.embedding_network.forward(support_features);

        if let Some(fce) = &self.fce {
            embeddings = fce.encode_support(&embeddings);
        }

        embeddings
    }

    /// Get embeddings for queries
    pub fn embed_query(
        &self,
        query_features: &Array2<f64>,
        support_embeddings: &Array2<f64>,
    ) -> Array2<f64> {
        let mut embeddings = self.embedding_network.forward(query_features);

        if let Some(fce) = &self.fce {
            embeddings = fce.encode_query(&embeddings, support_embeddings);
        }

        embeddings
    }

    /// Interpret prediction - return top-k most similar support examples
    pub fn interpret(
        &self,
        support_features: &Array2<f64>,
        support_labels: &Array1<usize>,
        query_features: &Array2<f64>,
        top_k: usize,
    ) -> Vec<Vec<(usize, f64, usize)>> {
        // Get embeddings
        let mut support_embeddings = self.embedding_network.forward(support_features);
        let mut query_embeddings = self.embedding_network.forward(query_features);

        if let Some(fce) = &self.fce {
            support_embeddings = fce.encode_support(&support_embeddings);
            query_embeddings = fce.encode_query(&query_embeddings, &support_embeddings);
        }

        // Compute attention
        let attention = self.attention_module.compute_attention(
            &query_embeddings,
            &support_embeddings,
        );

        // Get top-k for each query
        let mut results = Vec::with_capacity(attention.nrows());

        for row in attention.rows() {
            let mut indexed: Vec<_> = row
                .iter()
                .enumerate()
                .map(|(i, &w)| (i, w, support_labels[i]))
                .collect();

            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(top_k);

            results.push(indexed);
        }

        results
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_network.output_dim()
    }

    /// Get the input dimension
    pub fn input_dim(&self) -> usize {
        self.embedding_network.input_dim()
    }

    /// Check if FCE is enabled
    pub fn uses_fce(&self) -> bool {
        self.fce.is_some()
    }

    /// Get reference to embedding network
    pub fn embedding_network(&self) -> &EmbeddingNetwork {
        &self.embedding_network
    }

    /// Get reference to attention module
    pub fn attention_module(&self) -> &AttentionModule {
        &self.attention_module
    }

    /// Set temperature for attention
    pub fn set_temperature(&mut self, temperature: f64) {
        self.attention_module.set_temperature(temperature);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_matching_network_creation() {
        let network = MatchingNetwork::new(20, 64, 32, true);
        assert_eq!(network.input_dim(), 20);
        assert_eq!(network.embedding_dim(), 32);
        assert!(network.uses_fce());
    }

    #[test]
    fn test_forward_pass() {
        let network = MatchingNetwork::new(10, 32, 16, false);

        let support_features = Array2::from_shape_fn((6, 10), |_| rand::random::<f64>());
        let support_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let query_features = Array2::from_shape_fn((3, 10), |_| rand::random::<f64>());

        let probabilities = network.forward(&support_features, &support_labels, &query_features);

        assert_eq!(probabilities.shape(), &[3, 3]);

        // Probabilities should sum to 1 for each query
        for row in probabilities.rows() {
            assert_relative_eq!(row.sum(), 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_predict() {
        let network = MatchingNetwork::new(10, 32, 16, true);

        let support_features = Array2::from_shape_fn((4, 10), |_| rand::random::<f64>());
        let support_labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let query_features = Array2::from_shape_fn((5, 10), |_| rand::random::<f64>());

        let predictions = network.predict(&support_features, &support_labels, &query_features);

        assert_eq!(predictions.len(), 5);

        // All predictions should be valid class indices
        for &pred in predictions.iter() {
            assert!(pred < 2);
        }
    }

    #[test]
    fn test_predict_with_confidence() {
        let network = MatchingNetwork::new(10, 32, 16, false);

        let support_features = Array2::from_shape_fn((4, 10), |_| rand::random::<f64>());
        let support_labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let query_features = Array2::from_shape_fn((3, 10), |_| rand::random::<f64>());

        let (predictions, confidences) = network.predict_with_confidence(
            &support_features,
            &support_labels,
            &query_features,
        );

        assert_eq!(predictions.len(), 3);
        assert_eq!(confidences.len(), 3);

        // Confidences should be between 0 and 1
        for &conf in confidences.iter() {
            assert!(conf >= 0.0 && conf <= 1.0);
        }
    }

    #[test]
    fn test_interpret() {
        let network = MatchingNetwork::new(10, 32, 16, false);

        let support_features = Array2::from_shape_fn((6, 10), |_| rand::random::<f64>());
        let support_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let query_features = Array2::from_shape_fn((2, 10), |_| rand::random::<f64>());

        let interpretations = network.interpret(
            &support_features,
            &support_labels,
            &query_features,
            3,
        );

        assert_eq!(interpretations.len(), 2);
        for interp in &interpretations {
            assert_eq!(interp.len(), 3);
            // Weights should be in descending order
            for i in 0..interp.len() - 1 {
                assert!(interp[i].1 >= interp[i + 1].1);
            }
        }
    }

    #[test]
    fn test_with_config() {
        let network = MatchingNetwork::with_config(
            20,
            &[64, 32],
            16,
            DistanceFunction::Euclidean,
            0.5,
            true,
            5,
        );

        assert_eq!(network.input_dim(), 20);
        assert_eq!(network.embedding_dim(), 16);
        assert!(network.uses_fce());
    }
}
