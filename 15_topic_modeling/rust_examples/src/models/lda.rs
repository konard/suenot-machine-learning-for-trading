//! Latent Dirichlet Allocation (LDA)
//!
//! LDA is a generative probabilistic model for topic modeling.
//! This implementation uses collapsed Gibbs sampling.

use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_distr::Dirichlet;
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during LDA computation
#[derive(Error, Debug)]
pub enum LdaError {
    #[error("Matrix dimensions mismatch")]
    DimensionMismatch,

    #[error("Number of topics must be positive")]
    InvalidTopicCount,

    #[error("Model not fitted yet")]
    NotFitted,

    #[error("Invalid hyperparameter: {0}")]
    InvalidParameter(String),

    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Topic representation with words and probabilities
#[derive(Debug, Clone)]
pub struct LdaTopic {
    /// Topic index
    pub index: usize,
    /// Top words with their probabilities
    pub top_words: Vec<(String, f64)>,
    /// Topic prevalence in corpus
    pub prevalence: f64,
}

/// LDA model configuration
#[derive(Debug, Clone)]
pub struct LdaConfig {
    /// Number of topics
    pub n_topics: usize,
    /// Document-topic prior (alpha)
    pub alpha: f64,
    /// Topic-word prior (beta/eta)
    pub beta: f64,
    /// Number of Gibbs sampling iterations
    pub n_iterations: usize,
    /// Burn-in period (iterations to discard)
    pub burn_in: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for LdaConfig {
    fn default() -> Self {
        Self {
            n_topics: 10,
            alpha: 0.1,
            beta: 0.01,
            n_iterations: 1000,
            burn_in: 100,
            random_seed: None,
        }
    }
}

impl LdaConfig {
    /// Create a new configuration with specified number of topics
    pub fn new(n_topics: usize) -> Self {
        Self {
            n_topics,
            ..Default::default()
        }
    }

    /// Set alpha (document-topic prior)
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set beta (topic-word prior)
    pub fn beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Set number of iterations
    pub fn n_iterations(mut self, n: usize) -> Self {
        self.n_iterations = n;
        self
    }

    /// Set burn-in period
    pub fn burn_in(mut self, n: usize) -> Self {
        self.burn_in = n;
        self
    }

    /// Set random seed
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }
}

/// Latent Dirichlet Allocation model
///
/// Uses collapsed Gibbs sampling for inference.
#[derive(Debug)]
pub struct LDA {
    /// Model configuration
    config: LdaConfig,
    /// Topic-word counts: n_topics x n_words
    topic_word_counts: Option<Array2<f64>>,
    /// Document-topic counts: n_docs x n_topics
    doc_topic_counts: Option<Array2<f64>>,
    /// Topic counts (sum over words)
    topic_counts: Option<Array1<f64>>,
    /// Vocabulary mapping
    vocabulary: Option<HashMap<String, usize>>,
    /// Inverse vocabulary
    terms: Option<Vec<String>>,
    /// Number of words in vocabulary
    n_words: usize,
    /// Log likelihood history
    log_likelihood_history: Vec<f64>,
    /// Topic coherence scores
    coherence_scores: Option<Vec<f64>>,
}

impl LDA {
    /// Create a new LDA model
    pub fn new(config: LdaConfig) -> Result<Self, LdaError> {
        if config.n_topics == 0 {
            return Err(LdaError::InvalidTopicCount);
        }
        if config.alpha <= 0.0 {
            return Err(LdaError::InvalidParameter("alpha must be positive".into()));
        }
        if config.beta <= 0.0 {
            return Err(LdaError::InvalidParameter("beta must be positive".into()));
        }

        Ok(Self {
            config,
            topic_word_counts: None,
            doc_topic_counts: None,
            topic_counts: None,
            vocabulary: None,
            terms: None,
            n_words: 0,
            log_likelihood_history: Vec::new(),
            coherence_scores: None,
        })
    }

    /// Create a simple LDA model with just topic count
    pub fn simple(n_topics: usize) -> Result<Self, LdaError> {
        Self::new(LdaConfig::new(n_topics))
    }

    /// Fit the model using Gibbs sampling
    ///
    /// # Arguments
    /// * `dtm` - Document-term matrix (documents x terms) with word counts
    /// * `vocabulary` - Term to index mapping
    /// * `terms` - Index to term mapping
    pub fn fit(
        &mut self,
        dtm: &Array2<f64>,
        vocabulary: HashMap<String, usize>,
        terms: Vec<String>,
    ) -> Result<(), LdaError> {
        let n_docs = dtm.nrows();
        self.n_words = dtm.ncols();
        let n_topics = self.config.n_topics;

        if n_docs == 0 || self.n_words == 0 {
            return Err(LdaError::DimensionMismatch);
        }

        // Initialize RNG
        let mut rng = match self.config.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Convert DTM to word list format for Gibbs sampling
        // words[doc] = [(word_idx, count), ...]
        let mut doc_words: Vec<Vec<(usize, usize)>> = Vec::with_capacity(n_docs);
        for doc_idx in 0..n_docs {
            let mut words = Vec::new();
            for word_idx in 0..self.n_words {
                let count = dtm[[doc_idx, word_idx]] as usize;
                if count > 0 {
                    words.push((word_idx, count));
                }
            }
            doc_words.push(words);
        }

        // Initialize topic assignments randomly
        // topic_assignments[doc][word_position] = topic
        let mut topic_assignments: Vec<Vec<usize>> = Vec::with_capacity(n_docs);
        let mut topic_word_counts = Array2::zeros((n_topics, self.n_words));
        let mut doc_topic_counts = Array2::zeros((n_docs, n_topics));
        let mut topic_counts = Array1::zeros(n_topics);

        for (doc_idx, words) in doc_words.iter().enumerate() {
            let mut doc_assignments = Vec::new();
            for &(word_idx, count) in words {
                for _ in 0..count {
                    let topic = rng.gen_range(0..n_topics);
                    doc_assignments.push(topic);

                    topic_word_counts[[topic, word_idx]] += 1.0;
                    doc_topic_counts[[doc_idx, topic]] += 1.0;
                    topic_counts[topic] += 1.0;
                }
            }
            topic_assignments.push(doc_assignments);
        }

        // Gibbs sampling
        self.log_likelihood_history.clear();
        let alpha = self.config.alpha;
        let beta = self.config.beta;
        let beta_sum = beta * self.n_words as f64;

        for iter in 0..self.config.n_iterations {
            for (doc_idx, words) in doc_words.iter().enumerate() {
                let mut word_pos = 0;
                for &(word_idx, count) in words {
                    for _ in 0..count {
                        let old_topic = topic_assignments[doc_idx][word_pos];

                        // Remove word from counts
                        topic_word_counts[[old_topic, word_idx]] -= 1.0;
                        doc_topic_counts[[doc_idx, old_topic]] -= 1.0;
                        topic_counts[old_topic] -= 1.0;

                        // Sample new topic
                        let new_topic = self.sample_topic(
                            word_idx,
                            doc_idx,
                            &topic_word_counts,
                            &doc_topic_counts,
                            &topic_counts,
                            alpha,
                            beta,
                            beta_sum,
                            &mut rng,
                        );

                        // Add word to counts with new topic
                        topic_word_counts[[new_topic, word_idx]] += 1.0;
                        doc_topic_counts[[doc_idx, new_topic]] += 1.0;
                        topic_counts[new_topic] += 1.0;

                        topic_assignments[doc_idx][word_pos] = new_topic;
                        word_pos += 1;
                    }
                }
            }

            // Compute log-likelihood after burn-in
            if iter >= self.config.burn_in {
                let ll = self.compute_log_likelihood(
                    &topic_word_counts,
                    &doc_topic_counts,
                    &topic_counts,
                    alpha,
                    beta,
                    beta_sum,
                );
                self.log_likelihood_history.push(ll);
            }
        }

        self.topic_word_counts = Some(topic_word_counts);
        self.doc_topic_counts = Some(doc_topic_counts);
        self.topic_counts = Some(topic_counts);
        self.vocabulary = Some(vocabulary);
        self.terms = Some(terms);

        // Compute coherence scores
        self.compute_coherence(dtm)?;

        Ok(())
    }

    /// Sample a new topic for a word
    fn sample_topic(
        &self,
        word_idx: usize,
        doc_idx: usize,
        topic_word_counts: &Array2<f64>,
        doc_topic_counts: &Array2<f64>,
        topic_counts: &Array1<f64>,
        alpha: f64,
        beta: f64,
        beta_sum: f64,
        rng: &mut StdRng,
    ) -> usize {
        let n_topics = self.config.n_topics;
        let mut probs = Vec::with_capacity(n_topics);
        let mut total = 0.0;

        for topic in 0..n_topics {
            // P(topic | doc) * P(word | topic)
            let doc_topic = (doc_topic_counts[[doc_idx, topic]] + alpha)
                / (doc_topic_counts.row(doc_idx).sum() + n_topics as f64 * alpha);

            let topic_word =
                (topic_word_counts[[topic, word_idx]] + beta) / (topic_counts[topic] + beta_sum);

            let prob = doc_topic * topic_word;
            total += prob;
            probs.push(prob);
        }

        // Sample from distribution
        let threshold = rng.gen::<f64>() * total;
        let mut cumsum = 0.0;
        for (topic, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if cumsum >= threshold {
                return topic;
            }
        }

        n_topics - 1
    }

    /// Compute log-likelihood of the model
    fn compute_log_likelihood(
        &self,
        topic_word_counts: &Array2<f64>,
        doc_topic_counts: &Array2<f64>,
        topic_counts: &Array1<f64>,
        alpha: f64,
        beta: f64,
        beta_sum: f64,
    ) -> f64 {
        let mut ll = 0.0;
        let n_topics = self.config.n_topics;

        // Log-likelihood of word-topic distribution
        for topic in 0..n_topics {
            for word_idx in 0..self.n_words {
                let count = topic_word_counts[[topic, word_idx]];
                if count > 0.0 {
                    let prob =
                        (topic_word_counts[[topic, word_idx]] + beta) / (topic_counts[topic] + beta_sum);
                    ll += count * prob.ln();
                }
            }
        }

        // Log-likelihood of document-topic distribution
        for doc_idx in 0..doc_topic_counts.nrows() {
            let doc_total = doc_topic_counts.row(doc_idx).sum();
            for topic in 0..n_topics {
                let count = doc_topic_counts[[doc_idx, topic]];
                if count > 0.0 {
                    let prob = (count + alpha) / (doc_total + n_topics as f64 * alpha);
                    ll += count * prob.ln();
                }
            }
        }

        ll
    }

    /// Compute topic coherence scores (PMI-based)
    fn compute_coherence(&mut self, dtm: &Array2<f64>) -> Result<(), LdaError> {
        let topic_word_counts = self
            .topic_word_counts
            .as_ref()
            .ok_or(LdaError::NotFitted)?;
        let n_topics = self.config.n_topics;
        let n_top_words = 10;

        let mut coherences = Vec::with_capacity(n_topics);
        let n_docs = dtm.nrows() as f64;

        // Compute document frequency for each word
        let mut doc_freq: Vec<f64> = Vec::with_capacity(self.n_words);
        for word_idx in 0..self.n_words {
            let df = dtm.column(word_idx).iter().filter(|&&x| x > 0.0).count() as f64;
            doc_freq.push(df);
        }

        // Compute co-occurrence matrix (for top words only, to save memory)
        for topic in 0..n_topics {
            // Get top words for this topic
            let topic_vec = topic_word_counts.row(topic);
            let mut word_weights: Vec<(usize, f64)> = topic_vec
                .iter()
                .enumerate()
                .map(|(idx, &w)| (idx, w))
                .collect();
            word_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            word_weights.truncate(n_top_words);

            let top_word_indices: Vec<usize> = word_weights.iter().map(|(idx, _)| *idx).collect();

            // Compute pairwise PMI
            let mut pmi_sum = 0.0;
            let mut pair_count = 0;

            for i in 0..top_word_indices.len() {
                for j in (i + 1)..top_word_indices.len() {
                    let w1 = top_word_indices[i];
                    let w2 = top_word_indices[j];

                    // Count co-occurrences
                    let mut cooc = 0.0;
                    for doc_idx in 0..dtm.nrows() {
                        if dtm[[doc_idx, w1]] > 0.0 && dtm[[doc_idx, w2]] > 0.0 {
                            cooc += 1.0;
                        }
                    }

                    // PMI = log(P(w1,w2) / (P(w1) * P(w2)))
                    let p_w1 = (doc_freq[w1] + 1.0) / (n_docs + 2.0);
                    let p_w2 = (doc_freq[w2] + 1.0) / (n_docs + 2.0);
                    let p_w1_w2 = (cooc + 1.0) / (n_docs + 2.0);

                    let pmi = (p_w1_w2 / (p_w1 * p_w2)).ln();
                    pmi_sum += pmi;
                    pair_count += 1;
                }
            }

            let coherence = if pair_count > 0 {
                pmi_sum / pair_count as f64
            } else {
                0.0
            };
            coherences.push(coherence);
        }

        self.coherence_scores = Some(coherences);
        Ok(())
    }

    /// Get document-topic distribution
    ///
    /// Returns probabilities for each document belonging to each topic.
    pub fn get_document_topics(&self) -> Result<Array2<f64>, LdaError> {
        let doc_topic_counts = self
            .doc_topic_counts
            .as_ref()
            .ok_or(LdaError::NotFitted)?;

        let n_docs = doc_topic_counts.nrows();
        let n_topics = self.config.n_topics;
        let alpha = self.config.alpha;

        let mut doc_topics = Array2::zeros((n_docs, n_topics));

        for doc_idx in 0..n_docs {
            let doc_total = doc_topic_counts.row(doc_idx).sum() + n_topics as f64 * alpha;
            for topic in 0..n_topics {
                doc_topics[[doc_idx, topic]] =
                    (doc_topic_counts[[doc_idx, topic]] + alpha) / doc_total;
            }
        }

        Ok(doc_topics)
    }

    /// Get topic-word distribution
    ///
    /// Returns probabilities for each word in each topic.
    pub fn get_topic_words(&self) -> Result<Array2<f64>, LdaError> {
        let topic_word_counts = self
            .topic_word_counts
            .as_ref()
            .ok_or(LdaError::NotFitted)?;
        let topic_counts = self.topic_counts.as_ref().ok_or(LdaError::NotFitted)?;

        let n_topics = self.config.n_topics;
        let beta = self.config.beta;
        let beta_sum = beta * self.n_words as f64;

        let mut topic_words = Array2::zeros((n_topics, self.n_words));

        for topic in 0..n_topics {
            for word_idx in 0..self.n_words {
                topic_words[[topic, word_idx]] =
                    (topic_word_counts[[topic, word_idx]] + beta) / (topic_counts[topic] + beta_sum);
            }
        }

        Ok(topic_words)
    }

    /// Get topics with top words
    ///
    /// # Arguments
    /// * `n_words` - Number of top words per topic
    pub fn get_topics(&self, n_words: usize) -> Result<Vec<LdaTopic>, LdaError> {
        let topic_word_counts = self
            .topic_word_counts
            .as_ref()
            .ok_or(LdaError::NotFitted)?;
        let topic_counts = self.topic_counts.as_ref().ok_or(LdaError::NotFitted)?;
        let terms = self.terms.as_ref().ok_or(LdaError::NotFitted)?;

        let n_topics = self.config.n_topics;
        let beta = self.config.beta;
        let beta_sum = beta * self.n_words as f64;
        let total_words: f64 = topic_counts.iter().sum();

        let mut topics = Vec::with_capacity(n_topics);

        for topic_idx in 0..n_topics {
            let topic_total = topic_counts[topic_idx] + beta_sum;

            // Get top words by probability
            let mut word_probs: Vec<(usize, f64)> = (0..self.n_words)
                .map(|word_idx| {
                    let prob = (topic_word_counts[[topic_idx, word_idx]] + beta) / topic_total;
                    (word_idx, prob)
                })
                .collect();

            word_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            word_probs.truncate(n_words);

            let top_words: Vec<(String, f64)> = word_probs
                .into_iter()
                .filter_map(|(idx, prob)| terms.get(idx).map(|term| (term.clone(), prob)))
                .collect();

            let prevalence = topic_counts[topic_idx] / total_words;

            topics.push(LdaTopic {
                index: topic_idx,
                top_words,
                prevalence,
            });
        }

        Ok(topics)
    }

    /// Transform new documents into topic space
    ///
    /// Uses the fitted model to infer topic distributions for new documents.
    pub fn transform(&self, dtm: &Array2<f64>) -> Result<Array2<f64>, LdaError> {
        let topic_word_counts = self
            .topic_word_counts
            .as_ref()
            .ok_or(LdaError::NotFitted)?;
        let topic_counts = self.topic_counts.as_ref().ok_or(LdaError::NotFitted)?;

        let n_docs = dtm.nrows();
        let n_topics = self.config.n_topics;
        let alpha = self.config.alpha;
        let beta = self.config.beta;
        let beta_sum = beta * self.n_words as f64;

        let mut doc_topics = Array2::zeros((n_docs, n_topics));

        // For each new document, infer topic distribution
        for doc_idx in 0..n_docs {
            // Initialize topic counts for this document
            let mut local_topic_counts = Array1::zeros(n_topics);

            // Get words in document
            let mut doc_word_list: Vec<usize> = Vec::new();
            for word_idx in 0..dtm.ncols().min(self.n_words) {
                let count = dtm[[doc_idx, word_idx]] as usize;
                for _ in 0..count {
                    doc_word_list.push(word_idx);
                }
            }

            if doc_word_list.is_empty() {
                // Uniform distribution for empty documents
                for topic in 0..n_topics {
                    doc_topics[[doc_idx, topic]] = 1.0 / n_topics as f64;
                }
                continue;
            }

            // Simple inference: assign each word to most likely topic
            for &word_idx in &doc_word_list {
                let mut best_topic = 0;
                let mut best_prob = 0.0;

                for topic in 0..n_topics {
                    let prob =
                        (topic_word_counts[[topic, word_idx]] + beta) / (topic_counts[topic] + beta_sum);
                    if prob > best_prob {
                        best_prob = prob;
                        best_topic = topic;
                    }
                }

                local_topic_counts[best_topic] += 1.0;
            }

            // Normalize to get distribution
            let total = local_topic_counts.sum() + n_topics as f64 * alpha;
            for topic in 0..n_topics {
                doc_topics[[doc_idx, topic]] = (local_topic_counts[topic] + alpha) / total;
            }
        }

        Ok(doc_topics)
    }

    /// Get perplexity of the model
    ///
    /// Lower perplexity indicates better fit.
    pub fn perplexity(&self, dtm: &Array2<f64>) -> Result<f64, LdaError> {
        let doc_topics = self.get_document_topics()?;
        let topic_words = self.get_topic_words()?;

        let mut log_likelihood = 0.0;
        let mut total_words = 0.0;

        for doc_idx in 0..dtm.nrows() {
            for word_idx in 0..dtm.ncols().min(self.n_words) {
                let count = dtm[[doc_idx, word_idx]];
                if count > 0.0 {
                    // P(word | document) = sum over topics of P(word|topic) * P(topic|doc)
                    let mut prob = 0.0;
                    for topic in 0..self.config.n_topics {
                        prob += topic_words[[topic, word_idx]] * doc_topics[[doc_idx, topic]];
                    }
                    log_likelihood += count * prob.ln();
                    total_words += count;
                }
            }
        }

        Ok((-log_likelihood / total_words).exp())
    }

    /// Get coherence scores for each topic
    pub fn get_coherence_scores(&self) -> Option<&Vec<f64>> {
        self.coherence_scores.as_ref()
    }

    /// Get average coherence across all topics
    pub fn average_coherence(&self) -> Option<f64> {
        self.coherence_scores
            .as_ref()
            .map(|scores| scores.iter().sum::<f64>() / scores.len() as f64)
    }

    /// Get log-likelihood history during training
    pub fn log_likelihood_history(&self) -> &[f64] {
        &self.log_likelihood_history
    }

    /// Get configuration
    pub fn config(&self) -> &LdaConfig {
        &self.config
    }

    /// Get dominant topic for each document
    pub fn dominant_topics(&self) -> Result<Vec<usize>, LdaError> {
        let doc_topics = self.get_document_topics()?;

        let mut dominant = Vec::with_capacity(doc_topics.nrows());
        for doc_idx in 0..doc_topics.nrows() {
            let mut best_topic = 0;
            let mut best_prob = 0.0;
            for topic in 0..self.config.n_topics {
                if doc_topics[[doc_idx, topic]] > best_prob {
                    best_prob = doc_topics[[doc_idx, topic]];
                    best_topic = topic;
                }
            }
            dominant.push(best_topic);
        }

        Ok(dominant)
    }
}

/// Display implementation for LdaTopic
impl std::fmt::Display for LdaTopic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Topic {}: (prevalence: {:.2}%) [",
            self.index,
            self.prevalence * 100.0
        )?;
        for (i, (word, prob)) in self.top_words.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}: {:.3}", word, prob)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> (Array2<f64>, HashMap<String, usize>, Vec<String>) {
        // Simple test documents with clear topics
        // Topic 1: bitcoin, trading, price
        // Topic 2: ethereum, contract, smart
        let matrix = Array2::from_shape_vec(
            (6, 6),
            vec![
                3.0, 2.0, 2.0, 0.0, 0.0, 0.0, // Doc 1: topic 1
                2.0, 3.0, 1.0, 0.0, 0.0, 0.0, // Doc 2: topic 1
                1.0, 2.0, 3.0, 0.0, 0.0, 0.0, // Doc 3: topic 1
                0.0, 0.0, 0.0, 3.0, 2.0, 2.0, // Doc 4: topic 2
                0.0, 0.0, 0.0, 2.0, 3.0, 1.0, // Doc 5: topic 2
                0.0, 0.0, 0.0, 1.0, 2.0, 3.0, // Doc 6: topic 2
            ],
        )
        .unwrap();

        let terms = vec![
            "bitcoin".to_string(),
            "trading".to_string(),
            "price".to_string(),
            "ethereum".to_string(),
            "contract".to_string(),
            "smart".to_string(),
        ];

        let vocabulary: HashMap<String, usize> =
            terms.iter().enumerate().map(|(i, t)| (t.clone(), i)).collect();

        (matrix, vocabulary, terms)
    }

    #[test]
    fn test_lda_creation() {
        let lda = LDA::simple(5);
        assert!(lda.is_ok());

        let config = LdaConfig::new(0);
        let lda = LDA::new(config);
        assert!(lda.is_err());
    }

    #[test]
    fn test_lda_fit() {
        let (matrix, vocab, terms) = create_test_data();
        let config = LdaConfig::new(2)
            .n_iterations(100)
            .burn_in(10)
            .random_seed(42);

        let mut lda = LDA::new(config).unwrap();
        let result = lda.fit(&matrix, vocab, terms);
        assert!(result.is_ok());
    }

    #[test]
    fn test_lda_get_topics() {
        let (matrix, vocab, terms) = create_test_data();
        let config = LdaConfig::new(2)
            .n_iterations(100)
            .burn_in(10)
            .random_seed(42);

        let mut lda = LDA::new(config).unwrap();
        lda.fit(&matrix, vocab, terms).unwrap();

        let topics = lda.get_topics(3).unwrap();
        assert_eq!(topics.len(), 2);
        assert!(topics[0].top_words.len() <= 3);
    }

    #[test]
    fn test_dominant_topics() {
        let (matrix, vocab, terms) = create_test_data();
        let config = LdaConfig::new(2)
            .n_iterations(200)
            .burn_in(50)
            .random_seed(42);

        let mut lda = LDA::new(config).unwrap();
        lda.fit(&matrix, vocab, terms).unwrap();

        let dominant = lda.dominant_topics().unwrap();
        assert_eq!(dominant.len(), 6);

        // Documents 0-2 should have same dominant topic
        assert_eq!(dominant[0], dominant[1]);
        assert_eq!(dominant[1], dominant[2]);

        // Documents 3-5 should have same dominant topic
        assert_eq!(dominant[3], dominant[4]);
        assert_eq!(dominant[4], dominant[5]);

        // Two groups should have different topics
        assert_ne!(dominant[0], dominant[3]);
    }
}
