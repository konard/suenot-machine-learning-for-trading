//! Latent Semantic Indexing (LSI)
//!
//! LSI uses Singular Value Decomposition (SVD) to find latent topics
//! in a document-term matrix.

use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during LSI computation
#[derive(Error, Debug)]
pub enum LsiError {
    #[error("Matrix dimensions mismatch")]
    DimensionMismatch,

    #[error("Number of topics must be positive")]
    InvalidTopicCount,

    #[error("Model not fitted yet")]
    NotFitted,

    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Topic representation with words and weights
#[derive(Debug, Clone)]
pub struct Topic {
    /// Topic index
    pub index: usize,
    /// Top words with their weights
    pub top_words: Vec<(String, f64)>,
    /// Explained variance ratio
    pub variance_ratio: f64,
}

/// Latent Semantic Indexing model
///
/// Uses truncated SVD to decompose the document-term matrix
/// into document-topic and topic-term matrices.
#[derive(Debug)]
pub struct LSI {
    /// Number of topics/components
    n_topics: usize,
    /// Term weights matrix (V^T from SVD): topics x terms
    term_weights: Option<Array2<f64>>,
    /// Singular values
    singular_values: Option<Array1<f64>>,
    /// Document weights matrix (U * S from SVD): documents x topics
    doc_weights: Option<Array2<f64>>,
    /// Vocabulary mapping
    vocabulary: Option<HashMap<String, usize>>,
    /// Inverse vocabulary
    terms: Option<Vec<String>>,
    /// Explained variance ratios
    variance_ratios: Option<Vec<f64>>,
}

impl LSI {
    /// Create a new LSI model
    ///
    /// # Arguments
    /// * `n_topics` - Number of topics to extract
    pub fn new(n_topics: usize) -> Result<Self, LsiError> {
        if n_topics == 0 {
            return Err(LsiError::InvalidTopicCount);
        }

        Ok(Self {
            n_topics,
            term_weights: None,
            singular_values: None,
            doc_weights: None,
            vocabulary: None,
            terms: None,
            variance_ratios: None,
        })
    }

    /// Fit the model using power iteration method for SVD
    ///
    /// This is a simplified SVD implementation that doesn't require
    /// external LAPACK libraries.
    ///
    /// # Arguments
    /// * `dtm` - Document-term matrix (documents x terms)
    /// * `vocabulary` - Term to index mapping
    /// * `terms` - Index to term mapping
    pub fn fit(
        &mut self,
        dtm: &Array2<f64>,
        vocabulary: HashMap<String, usize>,
        terms: Vec<String>,
    ) -> Result<(), LsiError> {
        let (n_docs, n_terms) = (dtm.nrows(), dtm.ncols());

        if n_terms == 0 || n_docs == 0 {
            return Err(LsiError::DimensionMismatch);
        }

        let actual_n_topics = self.n_topics.min(n_docs).min(n_terms);

        // Compute truncated SVD using power iteration
        let (u, s, vt) = self.truncated_svd(dtm, actual_n_topics)?;

        // Compute variance ratios
        let total_variance: f64 = s.iter().map(|x| x * x).sum();
        let variance_ratios: Vec<f64> = s.iter().map(|x| (x * x) / total_variance).collect();

        // Document weights: U * S
        let doc_weights = &u * &s;

        self.term_weights = Some(vt);
        self.singular_values = Some(s);
        self.doc_weights = Some(doc_weights);
        self.vocabulary = Some(vocabulary);
        self.terms = Some(terms);
        self.variance_ratios = Some(variance_ratios);

        Ok(())
    }

    /// Perform truncated SVD using power iteration
    fn truncated_svd(
        &self,
        matrix: &Array2<f64>,
        k: usize,
    ) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>), LsiError> {
        let (m, n) = (matrix.nrows(), matrix.ncols());
        let mut u = Array2::zeros((m, k));
        let mut s = Array1::zeros(k);
        let mut vt = Array2::zeros((k, n));

        // Work matrix (will be modified)
        let mut work = matrix.to_owned();

        for i in 0..k {
            // Power iteration to find singular vector
            let (sigma, left_vec, right_vec) = self.power_iteration(&work, 100)?;

            s[i] = sigma;
            u.column_mut(i).assign(&left_vec);
            vt.row_mut(i).assign(&right_vec);

            // Deflate: remove this component from the matrix
            for row in 0..m {
                for col in 0..n {
                    work[[row, col]] -= sigma * left_vec[row] * right_vec[col];
                }
            }
        }

        Ok((u, s, vt))
    }

    /// Power iteration to find the largest singular value and vectors
    fn power_iteration(
        &self,
        matrix: &Array2<f64>,
        max_iter: usize,
    ) -> Result<(f64, Array1<f64>, Array1<f64>), LsiError> {
        let (m, n) = (matrix.nrows(), matrix.ncols());

        // Initialize random vector
        let mut v: Array1<f64> = Array1::from_iter((0..n).map(|i| ((i + 1) as f64).sin()));
        let norm = (v.iter().map(|x| x * x).sum::<f64>()).sqrt();
        v /= norm;

        let mut sigma = 0.0;

        for _ in 0..max_iter {
            // u = A * v
            let mut u = Array1::zeros(m);
            for i in 0..m {
                for j in 0..n {
                    u[i] += matrix[[i, j]] * v[j];
                }
            }

            // Normalize u
            let norm_u = (u.iter().map(|x| x * x).sum::<f64>()).sqrt();
            if norm_u < 1e-10 {
                break;
            }
            u /= norm_u;

            // v = A^T * u
            let mut v_new = Array1::zeros(n);
            for j in 0..n {
                for i in 0..m {
                    v_new[j] += matrix[[i, j]] * u[i];
                }
            }

            // Compute sigma and normalize v
            sigma = (v_new.iter().map(|x| x * x).sum::<f64>()).sqrt();
            if sigma < 1e-10 {
                break;
            }
            v = v_new / sigma;
        }

        // Final u computation
        let mut u = Array1::zeros(m);
        for i in 0..m {
            for j in 0..n {
                u[i] += matrix[[i, j]] * v[j];
            }
        }
        let norm_u = (u.iter().map(|x| x * x).sum::<f64>()).sqrt();
        if norm_u > 1e-10 {
            u /= norm_u;
        }

        Ok((sigma, u, v))
    }

    /// Transform new documents into topic space
    ///
    /// # Arguments
    /// * `dtm` - Document-term matrix for new documents
    pub fn transform(&self, dtm: &Array2<f64>) -> Result<Array2<f64>, LsiError> {
        let term_weights = self.term_weights.as_ref().ok_or(LsiError::NotFitted)?;
        let singular_values = self.singular_values.as_ref().ok_or(LsiError::NotFitted)?;

        let n_docs = dtm.nrows();
        let n_topics = term_weights.nrows();

        // Project: doc_topics = DTM * V^T^T * S^-1 = DTM * V * S^-1
        let mut doc_topics = Array2::zeros((n_docs, n_topics));

        for doc_idx in 0..n_docs {
            for topic_idx in 0..n_topics {
                let mut sum = 0.0;
                for term_idx in 0..dtm.ncols() {
                    sum += dtm[[doc_idx, term_idx]] * term_weights[[topic_idx, term_idx]];
                }
                // Divide by singular value (if non-zero)
                if singular_values[topic_idx].abs() > 1e-10 {
                    doc_topics[[doc_idx, topic_idx]] = sum;
                }
            }
        }

        Ok(doc_topics)
    }

    /// Get document-topic matrix from training
    pub fn get_document_topics(&self) -> Result<&Array2<f64>, LsiError> {
        self.doc_weights.as_ref().ok_or(LsiError::NotFitted)
    }

    /// Get topics with top words
    ///
    /// # Arguments
    /// * `n_words` - Number of top words per topic
    pub fn get_topics(&self, n_words: usize) -> Result<Vec<Topic>, LsiError> {
        let term_weights = self.term_weights.as_ref().ok_or(LsiError::NotFitted)?;
        let terms = self.terms.as_ref().ok_or(LsiError::NotFitted)?;
        let variance_ratios = self.variance_ratios.as_ref().ok_or(LsiError::NotFitted)?;

        let n_topics = term_weights.nrows();
        let mut topics = Vec::with_capacity(n_topics);

        for topic_idx in 0..n_topics {
            let topic_vec = term_weights.row(topic_idx);

            // Get top words by absolute weight
            let mut word_weights: Vec<(usize, f64)> = topic_vec
                .iter()
                .enumerate()
                .map(|(idx, &weight)| (idx, weight.abs()))
                .collect();

            word_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            word_weights.truncate(n_words);

            let top_words: Vec<(String, f64)> = word_weights
                .into_iter()
                .filter_map(|(idx, _)| {
                    terms
                        .get(idx)
                        .map(|term| (term.clone(), topic_vec[idx]))
                })
                .collect();

            topics.push(Topic {
                index: topic_idx,
                top_words,
                variance_ratio: variance_ratios[topic_idx],
            });
        }

        Ok(topics)
    }

    /// Get topic for a specific document
    ///
    /// # Arguments
    /// * `doc_idx` - Document index
    pub fn get_document_topic_distribution(
        &self,
        doc_idx: usize,
    ) -> Result<Vec<(usize, f64)>, LsiError> {
        let doc_weights = self.doc_weights.as_ref().ok_or(LsiError::NotFitted)?;

        if doc_idx >= doc_weights.nrows() {
            return Err(LsiError::DimensionMismatch);
        }

        let doc_vec = doc_weights.row(doc_idx);
        let mut topic_weights: Vec<(usize, f64)> = doc_vec
            .iter()
            .enumerate()
            .map(|(idx, &weight)| (idx, weight))
            .collect();

        topic_weights.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        Ok(topic_weights)
    }

    /// Compute cosine similarity between documents in topic space
    pub fn document_similarity(&self, doc1_idx: usize, doc2_idx: usize) -> Result<f64, LsiError> {
        let doc_weights = self.doc_weights.as_ref().ok_or(LsiError::NotFitted)?;

        if doc1_idx >= doc_weights.nrows() || doc2_idx >= doc_weights.nrows() {
            return Err(LsiError::DimensionMismatch);
        }

        let vec1 = doc_weights.row(doc1_idx);
        let vec2 = doc_weights.row(doc2_idx);

        let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1 = (vec1.iter().map(|x| x * x).sum::<f64>()).sqrt();
        let norm2 = (vec2.iter().map(|x| x * x).sum::<f64>()).sqrt();

        if norm1 < 1e-10 || norm2 < 1e-10 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm1 * norm2))
    }

    /// Find most similar documents to a given document
    pub fn most_similar_documents(
        &self,
        doc_idx: usize,
        n: usize,
    ) -> Result<Vec<(usize, f64)>, LsiError> {
        let doc_weights = self.doc_weights.as_ref().ok_or(LsiError::NotFitted)?;
        let n_docs = doc_weights.nrows();

        let mut similarities: Vec<(usize, f64)> = (0..n_docs)
            .filter(|&i| i != doc_idx)
            .filter_map(|i| self.document_similarity(doc_idx, i).ok().map(|sim| (i, sim)))
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(n);

        Ok(similarities)
    }

    /// Get total explained variance
    pub fn explained_variance_ratio(&self) -> Result<f64, LsiError> {
        let ratios = self.variance_ratios.as_ref().ok_or(LsiError::NotFitted)?;
        Ok(ratios.iter().sum())
    }

    /// Get number of topics
    pub fn n_topics(&self) -> usize {
        self.n_topics
    }
}

/// Display implementation for Topic
impl std::fmt::Display for Topic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Topic {}: (var: {:.2}%) [",
            self.index,
            self.variance_ratio * 100.0
        )?;
        for (i, (word, weight)) in self.top_words.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}: {:.3}", word, weight)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_matrix() -> (Array2<f64>, HashMap<String, usize>, Vec<String>) {
        // Simple test matrix
        let matrix = Array2::from_shape_vec(
            (4, 5),
            vec![
                1.0, 1.0, 0.0, 0.0, 0.0, // Doc about topic 1
                1.0, 0.0, 1.0, 0.0, 0.0, // Doc about topic 1
                0.0, 0.0, 0.0, 1.0, 1.0, // Doc about topic 2
                0.0, 0.0, 0.0, 1.0, 0.0, // Doc about topic 2
            ],
        )
        .unwrap();

        let terms = vec![
            "bitcoin".to_string(),
            "trading".to_string(),
            "analysis".to_string(),
            "ethereum".to_string(),
            "contract".to_string(),
        ];

        let vocabulary: HashMap<String, usize> =
            terms.iter().enumerate().map(|(i, t)| (t.clone(), i)).collect();

        (matrix, vocabulary, terms)
    }

    #[test]
    fn test_lsi_creation() {
        let lsi = LSI::new(5);
        assert!(lsi.is_ok());

        let lsi = LSI::new(0);
        assert!(lsi.is_err());
    }

    #[test]
    fn test_lsi_fit() {
        let (matrix, vocab, terms) = create_test_matrix();
        let mut lsi = LSI::new(2).unwrap();

        let result = lsi.fit(&matrix, vocab, terms);
        assert!(result.is_ok());
    }

    #[test]
    fn test_lsi_get_topics() {
        let (matrix, vocab, terms) = create_test_matrix();
        let mut lsi = LSI::new(2).unwrap();
        lsi.fit(&matrix, vocab, terms).unwrap();

        let topics = lsi.get_topics(3).unwrap();
        assert_eq!(topics.len(), 2);
        assert!(topics[0].top_words.len() <= 3);
    }

    #[test]
    fn test_document_similarity() {
        let (matrix, vocab, terms) = create_test_matrix();
        let mut lsi = LSI::new(2).unwrap();
        lsi.fit(&matrix, vocab, terms).unwrap();

        // Documents 0 and 1 should be similar (both topic 1)
        let sim_01 = lsi.document_similarity(0, 1).unwrap();
        // Documents 2 and 3 should be similar (both topic 2)
        let sim_23 = lsi.document_similarity(2, 3).unwrap();
        // Documents 0 and 2 should be less similar (different topics)
        let sim_02 = lsi.document_similarity(0, 2).unwrap();

        assert!(sim_01 > sim_02);
        assert!(sim_23 > sim_02);
    }
}
