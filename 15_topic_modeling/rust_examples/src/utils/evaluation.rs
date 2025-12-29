//! Evaluation metrics for topic models

use ndarray::Array2;
use std::collections::{HashMap, HashSet};

/// Topic model evaluation metrics
pub struct Evaluator {
    /// Document-term matrix for coherence computation
    dtm: Option<Array2<f64>>,
    /// Vocabulary
    vocabulary: Option<Vec<String>>,
}

impl Evaluator {
    /// Create a new evaluator
    pub fn new() -> Self {
        Self {
            dtm: None,
            vocabulary: None,
        }
    }

    /// Set the document-term matrix for coherence computation
    pub fn with_dtm(mut self, dtm: Array2<f64>, vocabulary: Vec<String>) -> Self {
        self.dtm = Some(dtm);
        self.vocabulary = Some(vocabulary);
        self
    }

    /// Compute UMass coherence for a topic
    ///
    /// UMass coherence uses document co-occurrence to measure topic quality.
    /// Higher (less negative) values indicate more coherent topics.
    pub fn umass_coherence(&self, top_words: &[String]) -> Option<f64> {
        let dtm = self.dtm.as_ref()?;
        let vocabulary = self.vocabulary.as_ref()?;

        let vocab_map: HashMap<&str, usize> = vocabulary
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i))
            .collect();

        let word_indices: Vec<usize> = top_words
            .iter()
            .filter_map(|w| vocab_map.get(w.as_str()).copied())
            .collect();

        if word_indices.len() < 2 {
            return None;
        }

        let n_docs = dtm.nrows() as f64;
        let epsilon = 1.0; // Smoothing factor

        let mut coherence = 0.0;
        let mut pair_count = 0;

        for (i, &w1_idx) in word_indices.iter().enumerate() {
            for &w2_idx in word_indices.iter().skip(i + 1) {
                // Count documents containing w2
                let d_w2 = dtm
                    .column(w2_idx)
                    .iter()
                    .filter(|&&x| x > 0.0)
                    .count() as f64;

                // Count documents containing both w1 and w2
                let d_w1_w2 = (0..dtm.nrows())
                    .filter(|&doc| dtm[[doc, w1_idx]] > 0.0 && dtm[[doc, w2_idx]] > 0.0)
                    .count() as f64;

                coherence += ((d_w1_w2 + epsilon) / d_w2).ln();
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            Some(coherence / pair_count as f64)
        } else {
            None
        }
    }

    /// Compute CV coherence (based on normalized PMI)
    ///
    /// Uses sliding window for word co-occurrence.
    pub fn cv_coherence(&self, top_words: &[String], window_size: usize) -> Option<f64> {
        let vocabulary = self.vocabulary.as_ref()?;

        let vocab_set: HashSet<&str> = vocabulary.iter().map(|s| s.as_str()).collect();
        let words_in_vocab: Vec<&str> = top_words
            .iter()
            .filter(|w| vocab_set.contains(w.as_str()))
            .map(|s| s.as_str())
            .collect();

        if words_in_vocab.len() < 2 {
            return None;
        }

        // For CV coherence, we would need the original text to compute
        // sliding window co-occurrences. This is a simplified version
        // using document co-occurrence instead.
        self.umass_coherence(top_words)
    }

    /// Compute topic diversity
    ///
    /// Measures how different topics are from each other.
    /// Higher values indicate more diverse topics.
    pub fn topic_diversity(topics: &[Vec<String>]) -> f64 {
        if topics.is_empty() {
            return 0.0;
        }

        let all_words: Vec<&str> = topics.iter().flatten().map(|s| s.as_str()).collect();

        let unique_words: HashSet<&str> = all_words.iter().copied().collect();

        if all_words.is_empty() {
            return 0.0;
        }

        unique_words.len() as f64 / all_words.len() as f64
    }

    /// Compute topic overlap between two topics
    ///
    /// Returns Jaccard similarity between word sets.
    pub fn topic_overlap(topic1: &[String], topic2: &[String]) -> f64 {
        let set1: HashSet<&str> = topic1.iter().map(|s| s.as_str()).collect();
        let set2: HashSet<&str> = topic2.iter().map(|s| s.as_str()).collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        if union == 0 {
            return 0.0;
        }

        intersection as f64 / union as f64
    }

    /// Compute average topic overlap matrix
    pub fn topic_overlap_matrix(topics: &[Vec<String>]) -> Array2<f64> {
        let n = topics.len();
        let mut matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                matrix[[i, j]] = Self::topic_overlap(&topics[i], &topics[j]);
            }
        }

        matrix
    }

    /// Compute word intrusion score
    ///
    /// Measures how well an intruder word can be identified among topic words.
    /// This is typically done with human evaluation, but we can approximate
    /// by checking word frequency in the topic distribution.
    pub fn word_intrusion_approximation(
        topic_words: &[(String, f64)],
        intruder_word: &str,
    ) -> f64 {
        // Find the probability of the intruder word if it exists
        let intruder_prob = topic_words
            .iter()
            .find(|(w, _)| w == intruder_word)
            .map(|(_, p)| *p)
            .unwrap_or(0.0);

        // Average probability of topic words
        let avg_prob: f64 = topic_words.iter().map(|(_, p)| p).sum::<f64>()
            / topic_words.len().max(1) as f64;

        // Score: how much lower is the intruder compared to average
        if avg_prob > 0.0 {
            1.0 - (intruder_prob / avg_prob).min(1.0)
        } else {
            0.0
        }
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for a topic model
#[derive(Debug, Clone)]
pub struct ModelSummary {
    /// Number of topics
    pub n_topics: usize,
    /// Average coherence score
    pub avg_coherence: Option<f64>,
    /// Topic diversity
    pub diversity: f64,
    /// Per-topic coherence scores
    pub topic_coherences: Vec<Option<f64>>,
    /// Perplexity (if available)
    pub perplexity: Option<f64>,
}

impl ModelSummary {
    /// Create a summary from topic model results
    pub fn from_topics(
        topics: &[Vec<String>],
        evaluator: &Evaluator,
        perplexity: Option<f64>,
    ) -> Self {
        let n_topics = topics.len();

        let topic_coherences: Vec<Option<f64>> = topics
            .iter()
            .map(|words| evaluator.umass_coherence(words))
            .collect();

        let coherence_values: Vec<f64> = topic_coherences.iter().filter_map(|&c| c).collect();

        let avg_coherence = if coherence_values.is_empty() {
            None
        } else {
            Some(coherence_values.iter().sum::<f64>() / coherence_values.len() as f64)
        };

        let diversity = Evaluator::topic_diversity(topics);

        Self {
            n_topics,
            avg_coherence,
            diversity,
            topic_coherences,
            perplexity,
        }
    }

    /// Print summary to console
    pub fn print(&self) {
        println!("=== Topic Model Summary ===");
        println!("Number of topics: {}", self.n_topics);

        if let Some(coh) = self.avg_coherence {
            println!("Average coherence: {:.4}", coh);
        }

        println!("Topic diversity: {:.4}", self.diversity);

        if let Some(perp) = self.perplexity {
            println!("Perplexity: {:.2}", perp);
        }

        println!("\nPer-topic coherence:");
        for (i, coh) in self.topic_coherences.iter().enumerate() {
            match coh {
                Some(c) => println!("  Topic {}: {:.4}", i, c),
                None => println!("  Topic {}: N/A", i),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_topic_diversity() {
        let topics = vec![
            vec!["bitcoin".to_string(), "trading".to_string()],
            vec!["ethereum".to_string(), "contract".to_string()],
        ];

        let diversity = Evaluator::topic_diversity(&topics);
        assert_eq!(diversity, 1.0); // All unique words

        let topics_overlap = vec![
            vec!["bitcoin".to_string(), "trading".to_string()],
            vec!["bitcoin".to_string(), "ethereum".to_string()],
        ];

        let diversity_overlap = Evaluator::topic_diversity(&topics_overlap);
        assert!(diversity_overlap < 1.0); // Some overlap
    }

    #[test]
    fn test_topic_overlap() {
        let topic1 = vec!["bitcoin".to_string(), "trading".to_string()];
        let topic2 = vec!["bitcoin".to_string(), "ethereum".to_string()];

        let overlap = Evaluator::topic_overlap(&topic1, &topic2);
        assert!((overlap - 1.0 / 3.0).abs() < 0.001); // 1 common, 3 total
    }

    #[test]
    fn test_umass_coherence() {
        let dtm = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap();

        let vocab = vec![
            "word1".to_string(),
            "word2".to_string(),
            "word3".to_string(),
        ];

        let evaluator = Evaluator::new().with_dtm(dtm, vocab);

        // Words 1 and 2 always appear together
        let top_words = vec!["word1".to_string(), "word2".to_string()];
        let coherence = evaluator.umass_coherence(&top_words);

        assert!(coherence.is_some());
    }
}
