//! # Active Learning for Trading
//!
//! This library implements active learning strategies for efficient labeling of
//! trading data. In financial markets, obtaining high-quality labels (e.g.,
//! regime labels, event tags, expert annotations) is expensive. Active learning
//! selects the most informative unlabeled samples for annotation, dramatically
//! reducing labeling cost while maintaining model performance.
//!
//! ## Key Components
//! - `UncertaintySampler`: Selects samples where the model is least confident
//! - `DiversitySampler`: Selects samples that are maximally diverse in feature space
//! - `QueryByCommittee`: Uses disagreement among an ensemble of models
//! - `ActiveLearningPipeline`: Orchestrates the full active learning loop
//! - `SimpleClassifier`: Lightweight linear classifier for demonstration

use ndarray::{Array1, Array2, Axis};
use rand::Rng;

// ---------------------------------------------------------------------------
// Distance & Utility Functions
// ---------------------------------------------------------------------------

/// Compute the Euclidean distance between two vectors.
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have equal length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute the cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have equal length");
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Compute softmax probabilities from logits.
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = logits.iter().map(|&z| (z - max_logit).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();
    exp_vals.iter().map(|&e| e / sum).collect()
}

/// Compute Shannon entropy of a probability distribution.
///
/// H(p) = -sum(p_i * log(p_i))
///
/// Higher entropy means greater uncertainty.
pub fn entropy(probs: &[f64]) -> f64 {
    -probs
        .iter()
        .map(|&p| {
            if p > 1e-10 {
                p * p.ln()
            } else {
                0.0
            }
        })
        .sum::<f64>()
}

/// Compute the margin between the top two class probabilities.
///
/// Small margin means the model is uncertain between two classes.
pub fn margin(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return 1.0;
    }
    let mut sorted: Vec<f64> = probs.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    sorted[0] - sorted[1]
}

// ---------------------------------------------------------------------------
// Simple Linear Classifier
// ---------------------------------------------------------------------------

/// A simple single-layer linear classifier: y = softmax(Wx + b).
///
/// Used for demonstration of active learning strategies.
/// In production, replace with a deep neural network.
#[derive(Debug, Clone)]
pub struct SimpleClassifier {
    /// Weight matrix: shape (num_classes, num_features)
    pub weights: Array2<f64>,
    /// Bias vector: shape (num_classes,)
    pub bias: Array1<f64>,
    /// Learning rate for SGD updates
    pub learning_rate: f64,
    /// Number of classes
    pub num_classes: usize,
    /// Number of input features
    pub num_features: usize,
}

impl SimpleClassifier {
    /// Create a new classifier with random weights.
    pub fn new(num_features: usize, num_classes: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / num_features as f64).sqrt();
        let weights = Array2::from_shape_fn((num_classes, num_features), |_| {
            rng.gen_range(-scale..scale)
        });
        let bias = Array1::zeros(num_classes);
        Self {
            weights,
            bias,
            learning_rate,
            num_classes,
            num_features,
        }
    }

    /// Compute logits for a single input sample.
    pub fn logits(&self, x: &Array1<f64>) -> Vec<f64> {
        let z = self.weights.dot(x) + &self.bias;
        z.to_vec()
    }

    /// Predict class probabilities for a single input.
    pub fn predict_proba(&self, x: &Array1<f64>) -> Vec<f64> {
        softmax(&self.logits(x))
    }

    /// Predict the most probable class.
    pub fn predict(&self, x: &Array1<f64>) -> usize {
        let probs = self.predict_proba(x);
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Train on a single sample using SGD with cross-entropy loss.
    pub fn train_step(&mut self, x: &Array1<f64>, label: usize) {
        let probs = self.predict_proba(x);
        // Gradient of cross-entropy w.r.t. logits: p_j - 1{j == label}
        for c in 0..self.num_classes {
            let grad = if c == label {
                probs[c] - 1.0
            } else {
                probs[c]
            };
            for f in 0..self.num_features {
                self.weights[[c, f]] -= self.learning_rate * grad * x[f];
            }
            self.bias[c] -= self.learning_rate * grad;
        }
    }

    /// Train on a batch of labeled samples for multiple epochs.
    pub fn train_batch(
        &mut self,
        features: &Array2<f64>,
        labels: &[usize],
        epochs: usize,
    ) {
        assert_eq!(features.nrows(), labels.len());
        for _ in 0..epochs {
            for i in 0..features.nrows() {
                let x = features.row(i).to_owned();
                self.train_step(&x, labels[i]);
            }
        }
    }

    /// Evaluate accuracy on a test set.
    pub fn accuracy(&self, features: &Array2<f64>, labels: &[usize]) -> f64 {
        let correct = features
            .axis_iter(Axis(0))
            .zip(labels.iter())
            .filter(|(x, &label)| self.predict(&x.to_owned()) == label)
            .count();
        correct as f64 / labels.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Uncertainty Sampling
// ---------------------------------------------------------------------------

/// Uncertainty sampling strategy for active learning.
///
/// Selects the samples where the model is most uncertain. Three uncertainty
/// measures are supported:
/// - **Least Confidence**: 1 - max(p)
/// - **Margin**: smallest difference between top-2 class probabilities
/// - **Entropy**: Shannon entropy of the predicted distribution
#[derive(Debug, Clone)]
pub struct UncertaintySampler {
    /// Which uncertainty measure to use
    pub strategy: UncertaintyStrategy,
}

/// Available uncertainty measures for sample selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UncertaintyStrategy {
    /// Select samples with lowest maximum predicted probability
    LeastConfidence,
    /// Select samples with smallest margin between top-2 probabilities
    MarginSampling,
    /// Select samples with highest prediction entropy
    EntropySampling,
}

impl UncertaintySampler {
    pub fn new(strategy: UncertaintyStrategy) -> Self {
        Self { strategy }
    }

    /// Score a single sample's uncertainty (higher = more informative).
    pub fn score(&self, probs: &[f64]) -> f64 {
        match self.strategy {
            UncertaintyStrategy::LeastConfidence => {
                let max_p = probs
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                1.0 - max_p
            }
            UncertaintyStrategy::MarginSampling => {
                // Low margin = high uncertainty, so we negate
                1.0 - margin(probs)
            }
            UncertaintyStrategy::EntropySampling => entropy(probs),
        }
    }

    /// Select the top-k most uncertain samples from the pool.
    ///
    /// Returns indices into the pool sorted by descending uncertainty.
    pub fn select(
        &self,
        model: &SimpleClassifier,
        pool: &Array2<f64>,
        k: usize,
    ) -> Vec<usize> {
        let mut scores: Vec<(usize, f64)> = pool
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, x)| {
                let probs = model.predict_proba(&x.to_owned());
                (i, self.score(&probs))
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.iter().take(k).map(|(idx, _)| *idx).collect()
    }
}

// ---------------------------------------------------------------------------
// Diversity Sampling
// ---------------------------------------------------------------------------

/// Diversity-based sampling that selects maximally spread-out samples.
///
/// Uses a greedy furthest-first traversal in feature space to ensure selected
/// samples cover different regions of the input distribution. This avoids
/// redundancy that can occur with pure uncertainty sampling.
#[derive(Debug, Clone)]
pub struct DiversitySampler;

impl DiversitySampler {
    pub fn new() -> Self {
        Self
    }

    /// Select k diverse samples from the pool using greedy furthest-first.
    ///
    /// Algorithm:
    /// 1. Start with a random seed sample
    /// 2. Iteratively add the sample that is farthest from the selected set
    pub fn select(
        &self,
        pool: &Array2<f64>,
        k: usize,
    ) -> Vec<usize> {
        let n = pool.nrows();
        if k >= n {
            return (0..n).collect();
        }

        let mut rng = rand::thread_rng();
        let mut selected = vec![rng.gen_range(0..n)];
        let mut min_distances = vec![f64::INFINITY; n];

        for _ in 1..k {
            // Update minimum distances to the selected set
            let last_selected = selected[selected.len() - 1];
            let last_point = pool.row(last_selected);
            for i in 0..n {
                if selected.contains(&i) {
                    min_distances[i] = 0.0;
                    continue;
                }
                let d = euclidean_distance(
                    last_point.as_slice().unwrap(),
                    pool.row(i).as_slice().unwrap(),
                );
                if d < min_distances[i] {
                    min_distances[i] = d;
                }
            }
            // Pick the point with the largest minimum distance
            let next = min_distances
                .iter()
                .enumerate()
                .filter(|(i, _)| !selected.contains(i))
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap();
            selected.push(next);
        }
        selected
    }
}

impl Default for DiversitySampler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Query by Committee
// ---------------------------------------------------------------------------

/// Query by Committee (QBC) active learning strategy.
///
/// Maintains a committee of models and selects samples where the committee
/// members disagree the most. The disagreement is measured by the variance
/// of predicted class distributions across committee members (vote entropy).
#[derive(Debug, Clone)]
pub struct QueryByCommittee {
    /// The committee of classifiers
    pub committee: Vec<SimpleClassifier>,
}

impl QueryByCommittee {
    /// Create a committee of `n` classifiers with different random seeds.
    pub fn new(n: usize, num_features: usize, num_classes: usize, lr: f64) -> Self {
        let committee = (0..n)
            .map(|_| SimpleClassifier::new(num_features, num_classes, lr))
            .collect();
        Self { committee }
    }

    /// Train all committee members on the same labeled data, but each starts
    /// from different random initialisation.
    pub fn train_all(
        &mut self,
        features: &Array2<f64>,
        labels: &[usize],
        epochs: usize,
    ) {
        for model in &mut self.committee {
            model.train_batch(features, labels, epochs);
        }
    }

    /// Compute vote entropy for a single sample.
    ///
    /// Each committee member casts a vote (predicted class), then we compute
    /// the entropy of the vote distribution.
    pub fn vote_entropy(&self, x: &Array1<f64>) -> f64 {
        let num_classes = self.committee[0].num_classes;
        let mut vote_counts = vec![0.0; num_classes];
        for model in &self.committee {
            let pred = model.predict(x);
            vote_counts[pred] += 1.0;
        }
        let n = self.committee.len() as f64;
        let probs: Vec<f64> = vote_counts.iter().map(|&v| v / n).collect();
        entropy(&probs)
    }

    /// Compute average KL divergence (consensus entropy) for a sample.
    ///
    /// Measures how much individual committee member distributions differ
    /// from the committee consensus.
    pub fn consensus_entropy(&self, x: &Array1<f64>) -> f64 {
        let predictions: Vec<Vec<f64>> = self
            .committee
            .iter()
            .map(|m| m.predict_proba(x))
            .collect();
        let num_classes = predictions[0].len();
        let n = predictions.len() as f64;

        // Compute consensus (average) distribution
        let mut consensus = vec![0.0; num_classes];
        for pred in &predictions {
            for (c, &p) in pred.iter().enumerate() {
                consensus[c] += p / n;
            }
        }

        // Average KL divergence of each member from consensus
        let mut total_kl = 0.0;
        for pred in &predictions {
            for (c, &p) in pred.iter().enumerate() {
                if p > 1e-10 && consensus[c] > 1e-10 {
                    total_kl += p * (p / consensus[c]).ln();
                }
            }
        }
        total_kl / n
    }

    /// Select the top-k samples with highest committee disagreement.
    pub fn select(
        &self,
        pool: &Array2<f64>,
        k: usize,
    ) -> Vec<usize> {
        let mut scores: Vec<(usize, f64)> = pool
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, x)| (i, self.vote_entropy(&x.to_owned())))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.iter().take(k).map(|(idx, _)| *idx).collect()
    }
}

// ---------------------------------------------------------------------------
// Active Learning Pipeline
// ---------------------------------------------------------------------------

/// Configuration for the active learning pipeline.
#[derive(Debug, Clone)]
pub struct ActiveLearningConfig {
    /// Number of samples to query per round
    pub query_batch_size: usize,
    /// Number of active learning rounds
    pub num_rounds: usize,
    /// Number of training epochs per round
    pub epochs_per_round: usize,
    /// Initial seed set size (randomly labeled)
    pub initial_seed_size: usize,
}

impl Default for ActiveLearningConfig {
    fn default() -> Self {
        Self {
            query_batch_size: 10,
            num_rounds: 10,
            initial_seed_size: 20,
            epochs_per_round: 5,
        }
    }
}

/// Which acquisition function to use in the pipeline.
#[derive(Debug, Clone, Copy)]
pub enum AcquisitionFunction {
    /// Uncertainty sampling with entropy
    UncertaintyEntropy,
    /// Uncertainty sampling with least confidence
    UncertaintyLeastConfidence,
    /// Uncertainty sampling with margin
    UncertaintyMargin,
    /// Diversity-based selection
    Diversity,
    /// Random baseline
    Random,
}

/// The active learning pipeline orchestrator.
///
/// Manages the iterative process:
/// 1. Train model on current labeled set
/// 2. Score unlabeled pool with acquisition function
/// 3. Query top-k samples for labeling
/// 4. Add newly labeled samples to training set
/// 5. Repeat
#[derive(Debug, Clone)]
pub struct ActiveLearningPipeline {
    pub config: ActiveLearningConfig,
    pub acquisition: AcquisitionFunction,
    /// Indices of the labeled set (into the full dataset)
    pub labeled_indices: Vec<usize>,
    /// Indices of the unlabeled pool
    pub pool_indices: Vec<usize>,
    /// Accuracy history per round
    pub accuracy_history: Vec<f64>,
}

impl ActiveLearningPipeline {
    /// Create a new pipeline.
    pub fn new(
        config: ActiveLearningConfig,
        acquisition: AcquisitionFunction,
        total_samples: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let mut all_indices: Vec<usize> = (0..total_samples).collect();
        // Shuffle to get random initial seed
        for i in (1..all_indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            all_indices.swap(i, j);
        }
        let seed_size = config.initial_seed_size.min(total_samples);
        let labeled_indices = all_indices[..seed_size].to_vec();
        let pool_indices = all_indices[seed_size..].to_vec();
        Self {
            config,
            acquisition,
            labeled_indices,
            pool_indices,
            accuracy_history: Vec::new(),
        }
    }

    /// Run one round of active learning.
    ///
    /// Returns the indices selected for labeling this round.
    pub fn query_round(
        &mut self,
        model: &SimpleClassifier,
        all_features: &Array2<f64>,
    ) -> Vec<usize> {
        if self.pool_indices.is_empty() {
            return Vec::new();
        }

        let k = self.config.query_batch_size.min(self.pool_indices.len());

        // Build pool feature matrix
        let num_features = all_features.ncols();
        let pool_features = Array2::from_shape_fn((self.pool_indices.len(), num_features), |(i, j)| {
            all_features[[self.pool_indices[i], j]]
        });

        let selected_pool_indices = match self.acquisition {
            AcquisitionFunction::UncertaintyEntropy => {
                let sampler = UncertaintySampler::new(UncertaintyStrategy::EntropySampling);
                sampler.select(model, &pool_features, k)
            }
            AcquisitionFunction::UncertaintyLeastConfidence => {
                let sampler = UncertaintySampler::new(UncertaintyStrategy::LeastConfidence);
                sampler.select(model, &pool_features, k)
            }
            AcquisitionFunction::UncertaintyMargin => {
                let sampler = UncertaintySampler::new(UncertaintyStrategy::MarginSampling);
                sampler.select(model, &pool_features, k)
            }
            AcquisitionFunction::Diversity => {
                let sampler = DiversitySampler::new();
                sampler.select(&pool_features, k)
            }
            AcquisitionFunction::Random => {
                let mut rng = rand::thread_rng();
                let mut indices: Vec<usize> = (0..self.pool_indices.len()).collect();
                for i in (1..indices.len()).rev() {
                    let j = rng.gen_range(0..=i);
                    indices.swap(i, j);
                }
                indices[..k].to_vec()
            }
        };

        // Map back to global indices
        let global_indices: Vec<usize> = selected_pool_indices
            .iter()
            .map(|&i| self.pool_indices[i])
            .collect();

        // Move from pool to labeled
        for &gi in &global_indices {
            self.labeled_indices.push(gi);
        }
        // Remove from pool (in reverse order of pool indices to avoid shifting)
        let mut to_remove: Vec<usize> = selected_pool_indices.clone();
        to_remove.sort_unstable_by(|a, b| b.cmp(a));
        for idx in to_remove {
            self.pool_indices.swap_remove(idx);
        }

        global_indices
    }

    /// Run the full active learning loop.
    ///
    /// Returns accuracy history across rounds.
    pub fn run(
        &mut self,
        all_features: &Array2<f64>,
        all_labels: &[usize],
        test_features: &Array2<f64>,
        test_labels: &[usize],
        num_classes: usize,
    ) -> Vec<f64> {
        let num_features = all_features.ncols();
        self.accuracy_history.clear();

        for _round in 0..self.config.num_rounds {
            // Build current labeled training set
            let n_labeled = self.labeled_indices.len();
            let train_features =
                Array2::from_shape_fn((n_labeled, num_features), |(i, j)| {
                    all_features[[self.labeled_indices[i], j]]
                });
            let train_labels: Vec<usize> = self
                .labeled_indices
                .iter()
                .map(|&i| all_labels[i])
                .collect();

            // Train model
            let mut model =
                SimpleClassifier::new(num_features, num_classes, 0.01);
            model.train_batch(&train_features, &train_labels, self.config.epochs_per_round);

            // Evaluate
            let acc = model.accuracy(test_features, test_labels);
            self.accuracy_history.push(acc);

            // Query next batch
            self.query_round(&model, all_features);
        }

        self.accuracy_history.clone()
    }

    /// Get the number of labeled samples at each round.
    pub fn labeled_count_per_round(&self) -> Vec<usize> {
        let seed = self.config.initial_seed_size;
        let batch = self.config.query_batch_size;
        (0..self.config.num_rounds)
            .map(|r| seed + r * batch)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Trading-Specific Helpers
// ---------------------------------------------------------------------------

/// Generate trading features from OHLCV candle data.
///
/// Features:
/// 1. Log return: ln(close_t / close_{t-1})
/// 2. Body ratio: |close - open| / (high - low)
/// 3. Upper shadow ratio: (high - max(open, close)) / (high - low)
/// 4. Lower shadow ratio: (min(open, close) - low) / (high - low)
/// 5. Volume change ratio: volume_t / volume_{t-1} - 1
pub fn compute_trading_features(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
) -> Array2<f64> {
    let n = close.len();
    assert!(n >= 2, "Need at least 2 candles");
    let rows = n - 1;
    let mut features = Array2::zeros((rows, 5));

    for i in 1..n {
        let range = (high[i] - low[i]).max(1e-10);
        features[[i - 1, 0]] = (close[i] / close[i - 1].max(1e-10)).ln();
        features[[i - 1, 1]] = (close[i] - open[i]).abs() / range;
        features[[i - 1, 2]] = (high[i] - close[i].max(open[i])) / range;
        features[[i - 1, 3]] = (close[i].min(open[i]) - low[i]) / range;
        features[[i - 1, 4]] = volume[i] / volume[i - 1].max(1e-10) - 1.0;
    }
    features
}

/// Generate labels based on forward returns.
///
/// - Class 0 (Short): forward return < -threshold
/// - Class 1 (Hold):  |forward return| <= threshold
/// - Class 2 (Long):  forward return > threshold
pub fn generate_trading_labels(close: &[f64], lookahead: usize, threshold: f64) -> Vec<usize> {
    let n = close.len();
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        if i + lookahead < n {
            let fwd_return = (close[i + lookahead] / close[i].max(1e-10)) - 1.0;
            if fwd_return > threshold {
                labels.push(2); // Long
            } else if fwd_return < -threshold {
                labels.push(0); // Short
            } else {
                labels.push(1); // Hold
            }
        } else {
            labels.push(1); // Default to Hold for tail
        }
    }
    labels
}

/// Simulate a simple trading strategy from predicted signals.
///
/// Returns cumulative PnL as a Vec<f64>.
pub fn simulate_trading(predictions: &[usize], close: &[f64]) -> Vec<f64> {
    let mut pnl = vec![0.0];
    let mut cumulative = 0.0;
    for i in 1..predictions.len().min(close.len()) {
        let ret = (close[i] / close[i - 1].max(1e-10)) - 1.0;
        let position = match predictions[i - 1] {
            0 => -1.0, // Short
            2 => 1.0,  // Long
            _ => 0.0,  // Hold
        };
        cumulative += position * ret;
        pnl.push(cumulative);
    }
    pnl
}

/// Compute the Sharpe ratio from a series of returns.
pub fn sharpe_ratio(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance =
        returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std_dev = variance.sqrt();
    if std_dev < 1e-10 {
        return 0.0;
    }
    // Annualized (assuming daily returns, 252 trading days)
    mean / std_dev * (252.0_f64).sqrt()
}

/// Compute maximum drawdown from a cumulative PnL curve.
pub fn max_drawdown(pnl: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0;
    for &val in pnl {
        if val > peak {
            peak = val;
        }
        let dd = peak - val;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Softmax should sum to 1.0");
    }

    #[test]
    fn test_entropy_uniform() {
        // Uniform distribution over 4 classes => max entropy = ln(4)
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let e = entropy(&probs);
        let expected = (4.0_f64).ln();
        assert!(
            (e - expected).abs() < 1e-10,
            "Entropy of uniform should be ln(4)"
        );
    }

    #[test]
    fn test_entropy_certain() {
        // Certain distribution => entropy = 0
        let probs = vec![1.0, 0.0, 0.0];
        let e = entropy(&probs);
        assert!(e.abs() < 1e-10, "Entropy of certain dist should be 0");
    }

    #[test]
    fn test_margin_calculation() {
        let probs = vec![0.6, 0.3, 0.1];
        let m = margin(&probs);
        assert!((m - 0.3).abs() < 1e-10, "Margin should be 0.6 - 0.3 = 0.3");
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let d = euclidean_distance(&a, &b);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_classifier_predict_produces_valid_class() {
        let clf = SimpleClassifier::new(4, 3, 0.01);
        let x = Array1::from_vec(vec![0.1, -0.2, 0.5, 0.3]);
        let pred = clf.predict(&x);
        assert!(pred < 3, "Prediction must be a valid class index");
    }

    #[test]
    fn test_classifier_proba_sums_to_one() {
        let clf = SimpleClassifier::new(4, 3, 0.01);
        let x = Array1::from_vec(vec![0.1, -0.2, 0.5, 0.3]);
        let probs = clf.predict_proba(&x);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_uncertainty_sampler_selects_correct_count() {
        let clf = SimpleClassifier::new(3, 2, 0.01);
        let pool = Array2::from_shape_fn((20, 3), |(i, j)| (i * 3 + j) as f64 * 0.1);
        let sampler = UncertaintySampler::new(UncertaintyStrategy::EntropySampling);
        let selected = sampler.select(&clf, &pool, 5);
        assert_eq!(selected.len(), 5);
    }

    #[test]
    fn test_diversity_sampler_selects_unique() {
        let pool = Array2::from_shape_fn((20, 3), |(i, j)| (i * 3 + j) as f64 * 0.1);
        let sampler = DiversitySampler::new();
        let selected = sampler.select(&pool, 5);
        assert_eq!(selected.len(), 5);
        // All indices should be unique
        let mut unique = selected.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 5);
    }

    #[test]
    fn test_trading_labels_generation() {
        let close = vec![100.0, 102.0, 98.0, 101.0, 103.0];
        let labels = generate_trading_labels(&close, 1, 0.015);
        assert_eq!(labels.len(), 5);
        // 100->102 = +2% > 1.5% => Long (2)
        assert_eq!(labels[0], 2);
        // 102->98 = -3.9% < -1.5% => Short (0)
        assert_eq!(labels[1], 0);
    }

    #[test]
    fn test_compute_trading_features_shape() {
        let open = vec![100.0, 101.0, 99.0, 102.0];
        let high = vec![103.0, 104.0, 101.0, 105.0];
        let low = vec![99.0, 100.0, 97.0, 100.0];
        let close = vec![102.0, 103.0, 100.0, 104.0];
        let volume = vec![1000.0, 1200.0, 800.0, 1500.0];
        let feats = compute_trading_features(&open, &high, &low, &close, &volume);
        assert_eq!(feats.nrows(), 3);
        assert_eq!(feats.ncols(), 5);
    }

    #[test]
    fn test_sharpe_ratio_positive() {
        let returns = vec![0.01, 0.02, 0.015, 0.005, 0.01];
        let sr = sharpe_ratio(&returns);
        assert!(sr > 0.0, "Sharpe ratio should be positive for positive returns");
    }

    #[test]
    fn test_max_drawdown() {
        let pnl = vec![0.0, 0.1, 0.2, 0.15, 0.05, 0.25];
        let dd = max_drawdown(&pnl);
        // Peak at 0.2, trough at 0.05 => dd = 0.15
        assert!((dd - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_qbc_vote_entropy() {
        let qbc = QueryByCommittee::new(5, 3, 2, 0.01);
        let x = Array1::from_vec(vec![0.1, -0.2, 0.5]);
        let ve = qbc.vote_entropy(&x);
        // Vote entropy should be non-negative
        assert!(ve >= 0.0);
    }
}
