//! Aggregation functions for MPNN.
//!
//! Aggregation functions combine messages from multiple neighbors into a single representation.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Aggregator that combines multiple messages.
pub struct Aggregator {
    /// Type of aggregation
    pub agg_type: AggregatorType,
    /// Optional learnable parameters
    pub params: Option<AggregatorParams>,
}

/// Types of aggregation functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregatorType {
    /// Sum all messages
    Sum,
    /// Average all messages
    Mean,
    /// Element-wise maximum
    Max,
    /// Element-wise minimum
    Min,
    /// Learnable weighted sum
    WeightedSum,
    /// LSTM-based aggregation
    LSTM,
    /// Set2Set pooling
    Set2Set,
    /// Principal Neighborhood Aggregation
    PNA,
}

/// Learnable aggregator parameters.
pub struct AggregatorParams {
    /// Attention weights for weighted sum
    pub attention_weights: Option<Array2<f64>>,
    /// LSTM cell weights
    pub lstm_weights: Option<LSTMWeights>,
    /// Number of Set2Set iterations
    pub set2set_iterations: usize,
}

/// LSTM weights for LSTM aggregation.
pub struct LSTMWeights {
    pub w_i: Array2<f64>,
    pub w_f: Array2<f64>,
    pub w_c: Array2<f64>,
    pub w_o: Array2<f64>,
    pub u_i: Array2<f64>,
    pub u_f: Array2<f64>,
    pub u_c: Array2<f64>,
    pub u_o: Array2<f64>,
}

impl Aggregator {
    /// Create a sum aggregator.
    pub fn sum() -> Self {
        Self {
            agg_type: AggregatorType::Sum,
            params: None,
        }
    }

    /// Create a mean aggregator.
    pub fn mean() -> Self {
        Self {
            agg_type: AggregatorType::Mean,
            params: None,
        }
    }

    /// Create a max aggregator.
    pub fn max() -> Self {
        Self {
            agg_type: AggregatorType::Max,
            params: None,
        }
    }

    /// Create a min aggregator.
    pub fn min() -> Self {
        Self {
            agg_type: AggregatorType::Min,
            params: None,
        }
    }

    /// Create a learnable weighted sum aggregator.
    pub fn weighted_sum(hidden_dim: usize) -> Self {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        Self {
            agg_type: AggregatorType::WeightedSum,
            params: Some(AggregatorParams {
                attention_weights: Some(Array2::from_shape_fn((hidden_dim, 1), |_| {
                    normal.sample(&mut rng)
                })),
                lstm_weights: None,
                set2set_iterations: 0,
            }),
        }
    }

    /// Aggregate a list of messages.
    pub fn aggregate(&self, messages: &[Array1<f64>]) -> Array1<f64> {
        if messages.is_empty() {
            return Array1::zeros(0);
        }

        let dim = messages[0].len();
        let n = messages.len() as f64;

        match self.agg_type {
            AggregatorType::Sum => {
                let mut result = Array1::zeros(dim);
                for msg in messages {
                    result = &result + msg;
                }
                result
            }

            AggregatorType::Mean => {
                let mut result = Array1::zeros(dim);
                for msg in messages {
                    result = &result + msg;
                }
                result / n
            }

            AggregatorType::Max => {
                let mut result = Array1::from_elem(dim, f64::NEG_INFINITY);
                for msg in messages {
                    for (i, &v) in msg.iter().enumerate() {
                        result[i] = result[i].max(v);
                    }
                }
                // Replace -inf with 0
                result.mapv(|x| if x.is_finite() { x } else { 0.0 })
            }

            AggregatorType::Min => {
                let mut result = Array1::from_elem(dim, f64::INFINITY);
                for msg in messages {
                    for (i, &v) in msg.iter().enumerate() {
                        result[i] = result[i].min(v);
                    }
                }
                // Replace inf with 0
                result.mapv(|x| if x.is_finite() { x } else { 0.0 })
            }

            AggregatorType::WeightedSum => {
                if let Some(ref params) = self.params {
                    if let Some(ref att_w) = params.attention_weights {
                        // Compute attention scores
                        let mut scores = Vec::with_capacity(messages.len());
                        for msg in messages {
                            if msg.len() == att_w.nrows() {
                                let score = msg.dot(&att_w.column(0).to_owned());
                                scores.push(score);
                            } else {
                                scores.push(0.0);
                            }
                        }

                        // Softmax
                        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let exp_scores: Vec<f64> =
                            scores.iter().map(|s| (s - max_score).exp()).collect();
                        let sum_exp: f64 = exp_scores.iter().sum();

                        let weights: Vec<f64> = exp_scores.iter().map(|e| e / sum_exp).collect();

                        // Weighted sum
                        let mut result = Array1::zeros(dim);
                        for (msg, &w) in messages.iter().zip(weights.iter()) {
                            result = &result + &(msg * w);
                        }
                        result
                    } else {
                        // Fallback to mean
                        let mut result = Array1::zeros(dim);
                        for msg in messages {
                            result = &result + msg;
                        }
                        result / n
                    }
                } else {
                    // Fallback to mean
                    let mut result = Array1::zeros(dim);
                    for msg in messages {
                        result = &result + msg;
                    }
                    result / n
                }
            }

            AggregatorType::LSTM => {
                // Simplified LSTM aggregation (process messages sequentially)
                self.lstm_aggregate(messages)
            }

            AggregatorType::Set2Set => {
                // Set2Set pooling
                self.set2set_aggregate(messages)
            }

            AggregatorType::PNA => {
                // Principal Neighborhood Aggregation: combine multiple aggregators
                self.pna_aggregate(messages)
            }
        }
    }

    /// LSTM-based aggregation.
    fn lstm_aggregate(&self, messages: &[Array1<f64>]) -> Array1<f64> {
        if messages.is_empty() {
            return Array1::zeros(0);
        }

        let dim = messages[0].len();

        // Simple RNN-like aggregation (not full LSTM for simplicity)
        let mut hidden = Array1::zeros(dim);

        for msg in messages {
            // Simple update: h = tanh(h + msg)
            for i in 0..dim {
                hidden[i] = (hidden[i] + msg[i]).tanh();
            }
        }

        hidden
    }

    /// Set2Set pooling aggregation.
    fn set2set_aggregate(&self, messages: &[Array1<f64>]) -> Array1<f64> {
        if messages.is_empty() {
            return Array1::zeros(0);
        }

        let dim = messages[0].len();
        let iterations = self
            .params
            .as_ref()
            .map(|p| p.set2set_iterations)
            .unwrap_or(3);

        let mut query = Array1::zeros(dim);

        for _ in 0..iterations {
            // Compute attention scores
            let mut scores = Vec::with_capacity(messages.len());
            for msg in messages {
                let score: f64 = query.iter().zip(msg.iter()).map(|(a, b)| a * b).sum();
                scores.push(score);
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();

            // Weighted read
            let mut read = Array1::zeros(dim);
            for (msg, &exp_s) in messages.iter().zip(exp_scores.iter()) {
                let weight = exp_s / sum_exp;
                read = &read + &(msg * weight);
            }

            // Update query (simplified)
            query = &query + &read;
        }

        query
    }

    /// PNA: Principal Neighborhood Aggregation.
    fn pna_aggregate(&self, messages: &[Array1<f64>]) -> Array1<f64> {
        if messages.is_empty() {
            return Array1::zeros(0);
        }

        let dim = messages[0].len();
        let n = messages.len() as f64;

        // Compute multiple aggregations
        let mut sum_agg = Array1::zeros(dim);
        let mut max_agg = Array1::from_elem(dim, f64::NEG_INFINITY);
        let mut min_agg = Array1::from_elem(dim, f64::INFINITY);

        for msg in messages {
            for i in 0..dim {
                sum_agg[i] += msg[i];
                max_agg[i] = max_agg[i].max(msg[i]);
                min_agg[i] = min_agg[i].min(msg[i]);
            }
        }

        let mean_agg = &sum_agg / n;

        // Compute standard deviation
        let mut var_agg = Array1::zeros(dim);
        for msg in messages {
            for i in 0..dim {
                var_agg[i] += (msg[i] - mean_agg[i]).powi(2);
            }
        }
        let std_agg = var_agg.mapv(|v| (v / n).sqrt());

        // Concatenate all aggregations
        let total_dim = dim * 4;
        let mut result = Array1::zeros(total_dim);

        for i in 0..dim {
            result[i] = mean_agg[i];
            result[dim + i] = if max_agg[i].is_finite() {
                max_agg[i]
            } else {
                0.0
            };
            result[2 * dim + i] = if min_agg[i].is_finite() {
                min_agg[i]
            } else {
                0.0
            };
            result[3 * dim + i] = std_agg[i];
        }

        result
    }
}

/// Trait for custom aggregation functions.
pub trait Aggregate {
    /// Aggregate messages into a single representation.
    fn aggregate(&self, messages: &[Array1<f64>]) -> Array1<f64>;
}

/// Softmax aggregation with learnable temperature.
pub struct SoftmaxAggregator {
    /// Temperature parameter
    pub temperature: f64,
}

impl SoftmaxAggregator {
    /// Create a new softmax aggregator.
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }
}

impl Aggregate for SoftmaxAggregator {
    fn aggregate(&self, messages: &[Array1<f64>]) -> Array1<f64> {
        if messages.is_empty() {
            return Array1::zeros(0);
        }

        let dim = messages[0].len();

        // Compute norms as "scores"
        let scores: Vec<f64> = messages
            .iter()
            .map(|m| m.iter().map(|x| x.powi(2)).sum::<f64>().sqrt() / self.temperature)
            .collect();

        // Softmax
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        // Weighted sum
        let mut result = Array1::zeros(dim);
        for (msg, &exp_s) in messages.iter().zip(exp_scores.iter()) {
            let weight = exp_s / sum_exp;
            result = &result + &(msg * weight);
        }

        result
    }
}

/// Power mean aggregation.
pub struct PowerMeanAggregator {
    /// Power parameter (p=1 is mean, p->inf is max, p->-inf is min)
    pub power: f64,
}

impl PowerMeanAggregator {
    /// Create a new power mean aggregator.
    pub fn new(power: f64) -> Self {
        Self { power }
    }
}

impl Aggregate for PowerMeanAggregator {
    fn aggregate(&self, messages: &[Array1<f64>]) -> Array1<f64> {
        if messages.is_empty() {
            return Array1::zeros(0);
        }

        let dim = messages[0].len();
        let n = messages.len() as f64;

        if self.power.abs() < 0.01 {
            // Geometric mean for p close to 0
            let mut result = Array1::ones(dim);
            for msg in messages {
                for i in 0..dim {
                    result[i] *= msg[i].abs().max(1e-10);
                }
            }
            result.mapv(|x| x.powf(1.0 / n))
        } else {
            // General power mean
            let mut result = Array1::zeros(dim);
            for msg in messages {
                for i in 0..dim {
                    result[i] += msg[i].abs().powf(self.power);
                }
            }
            result.mapv(|x| (x / n).powf(1.0 / self.power))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sum_aggregation() {
        let agg = Aggregator::sum();
        let messages = vec![array![1.0, 2.0], array![3.0, 4.0], array![5.0, 6.0]];

        let result = agg.aggregate(&messages);
        assert_eq!(result, array![9.0, 12.0]);
    }

    #[test]
    fn test_mean_aggregation() {
        let agg = Aggregator::mean();
        let messages = vec![array![1.0, 2.0], array![3.0, 4.0], array![5.0, 6.0]];

        let result = agg.aggregate(&messages);
        assert_eq!(result, array![3.0, 4.0]);
    }

    #[test]
    fn test_max_aggregation() {
        let agg = Aggregator::max();
        let messages = vec![array![1.0, 6.0], array![3.0, 4.0], array![5.0, 2.0]];

        let result = agg.aggregate(&messages);
        assert_eq!(result, array![5.0, 6.0]);
    }

    #[test]
    fn test_pna_aggregation() {
        let agg = Aggregator {
            agg_type: AggregatorType::PNA,
            params: None,
        };
        let messages = vec![array![1.0, 2.0], array![3.0, 4.0]];

        let result = agg.aggregate(&messages);
        assert_eq!(result.len(), 8); // 4 aggregations * 2 dims
    }

    #[test]
    fn test_empty_aggregation() {
        let agg = Aggregator::mean();
        let messages: Vec<Array1<f64>> = vec![];

        let result = agg.aggregate(&messages);
        assert_eq!(result.len(), 0);
    }
}
