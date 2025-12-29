//! Text vectorization for topic modeling
//!
//! Provides TF-IDF and count vectorization for converting
//! tokenized text into numerical matrices.

use hashbrown::HashMap;
use ndarray::{Array1, Array2};
use std::collections::HashSet;

/// Term frequency computation methods
#[derive(Debug, Clone, Copy)]
pub enum TfMethod {
    /// Raw term count
    Raw,
    /// Boolean: 1 if term present, 0 otherwise
    Binary,
    /// Log-normalized: 1 + log(tf)
    LogNorm,
    /// Double normalization: 0.5 + 0.5 * (tf / max_tf)
    DoubleNorm,
}

/// Inverse document frequency computation methods
#[derive(Debug, Clone, Copy)]
pub enum IdfMethod {
    /// Standard IDF: log(N / df)
    Standard,
    /// Smooth IDF: log(N / (1 + df)) + 1
    Smooth,
    /// Probabilistic IDF: log((N - df) / df)
    Probabilistic,
}

/// TF-IDF Vectorizer
///
/// Converts text documents into TF-IDF feature matrices.
#[derive(Debug, Clone)]
pub struct TfIdfVectorizer {
    /// Vocabulary: term -> index mapping
    vocabulary: HashMap<String, usize>,
    /// Inverse vocabulary: index -> term
    inverse_vocabulary: Vec<String>,
    /// Document frequencies for each term
    document_frequencies: Vec<usize>,
    /// Total number of documents seen during fitting
    n_documents: usize,
    /// TF computation method
    tf_method: TfMethod,
    /// IDF computation method
    idf_method: IdfMethod,
    /// Minimum document frequency for term inclusion
    min_df: usize,
    /// Maximum document frequency ratio for term inclusion
    max_df_ratio: f64,
    /// Maximum vocabulary size
    max_features: Option<usize>,
    /// IDF values (computed during fit)
    idf_values: Vec<f64>,
    /// Whether the vectorizer has been fitted
    is_fitted: bool,
}

impl TfIdfVectorizer {
    /// Create a new TF-IDF vectorizer with default settings
    pub fn new() -> Self {
        Self {
            vocabulary: HashMap::new(),
            inverse_vocabulary: Vec::new(),
            document_frequencies: Vec::new(),
            n_documents: 0,
            tf_method: TfMethod::Raw,
            idf_method: IdfMethod::Smooth,
            min_df: 1,
            max_df_ratio: 1.0,
            max_features: None,
            idf_values: Vec::new(),
            is_fitted: false,
        }
    }

    /// Set TF computation method
    pub fn tf_method(mut self, method: TfMethod) -> Self {
        self.tf_method = method;
        self
    }

    /// Set IDF computation method
    pub fn idf_method(mut self, method: IdfMethod) -> Self {
        self.idf_method = method;
        self
    }

    /// Set minimum document frequency
    pub fn min_df(mut self, min_df: usize) -> Self {
        self.min_df = min_df;
        self
    }

    /// Set maximum document frequency ratio
    pub fn max_df_ratio(mut self, ratio: f64) -> Self {
        self.max_df_ratio = ratio;
        self
    }

    /// Set maximum vocabulary size
    pub fn max_features(mut self, max: usize) -> Self {
        self.max_features = Some(max);
        self
    }

    /// Fit the vectorizer on tokenized documents
    pub fn fit(&mut self, tokenized_docs: &[Vec<String>]) {
        self.n_documents = tokenized_docs.len();

        // Count document frequencies
        let mut term_doc_freq: HashMap<String, usize> = HashMap::new();

        for doc in tokenized_docs {
            let unique_terms: HashSet<&String> = doc.iter().collect();
            for term in unique_terms {
                *term_doc_freq.entry(term.clone()).or_insert(0) += 1;
            }
        }

        // Filter by document frequency
        let max_df = (self.n_documents as f64 * self.max_df_ratio) as usize;
        let mut filtered_terms: Vec<(String, usize)> = term_doc_freq
            .into_iter()
            .filter(|(_, df)| *df >= self.min_df && *df <= max_df)
            .collect();

        // Sort by document frequency (descending) for max_features selection
        filtered_terms.sort_by(|a, b| b.1.cmp(&a.1));

        // Apply max_features limit
        if let Some(max) = self.max_features {
            filtered_terms.truncate(max);
        }

        // Sort alphabetically for consistent vocabulary ordering
        filtered_terms.sort_by(|a, b| a.0.cmp(&b.0));

        // Build vocabulary
        self.vocabulary.clear();
        self.inverse_vocabulary.clear();
        self.document_frequencies.clear();

        for (idx, (term, df)) in filtered_terms.into_iter().enumerate() {
            self.vocabulary.insert(term.clone(), idx);
            self.inverse_vocabulary.push(term);
            self.document_frequencies.push(df);
        }

        // Compute IDF values
        self.compute_idf_values();
        self.is_fitted = true;
    }

    /// Compute IDF values for all terms
    fn compute_idf_values(&mut self) {
        self.idf_values = self
            .document_frequencies
            .iter()
            .map(|&df| self.compute_idf(df))
            .collect();
    }

    /// Compute IDF for a single term
    fn compute_idf(&self, df: usize) -> f64 {
        let n = self.n_documents as f64;
        let df = df as f64;

        match self.idf_method {
            IdfMethod::Standard => (n / df).ln(),
            IdfMethod::Smooth => (n / (1.0 + df)).ln() + 1.0,
            IdfMethod::Probabilistic => ((n - df) / df).ln().max(0.0),
        }
    }

    /// Compute TF for a term count
    fn compute_tf(&self, count: usize, max_count: usize) -> f64 {
        let count = count as f64;

        match self.tf_method {
            TfMethod::Raw => count,
            TfMethod::Binary => {
                if count > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            TfMethod::LogNorm => {
                if count > 0.0 {
                    1.0 + count.ln()
                } else {
                    0.0
                }
            }
            TfMethod::DoubleNorm => {
                let max = max_count as f64;
                if max > 0.0 {
                    0.5 + 0.5 * (count / max)
                } else {
                    0.0
                }
            }
        }
    }

    /// Transform tokenized documents into TF-IDF matrix
    ///
    /// Returns a matrix of shape (n_documents, n_features)
    pub fn transform(&self, tokenized_docs: &[Vec<String>]) -> Array2<f64> {
        assert!(self.is_fitted, "Vectorizer must be fitted before transform");

        let n_docs = tokenized_docs.len();
        let n_features = self.vocabulary.len();

        let mut matrix = Array2::zeros((n_docs, n_features));

        for (doc_idx, doc) in tokenized_docs.iter().enumerate() {
            // Count term frequencies in this document
            let mut term_counts: HashMap<&String, usize> = HashMap::new();
            for term in doc {
                *term_counts.entry(term).or_insert(0) += 1;
            }

            let max_count = *term_counts.values().max().unwrap_or(&1);

            // Compute TF-IDF for each term
            for (term, &count) in &term_counts {
                if let Some(&term_idx) = self.vocabulary.get(*term) {
                    let tf = self.compute_tf(count, max_count);
                    let idf = self.idf_values[term_idx];
                    matrix[[doc_idx, term_idx]] = tf * idf;
                }
            }
        }

        matrix
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, tokenized_docs: &[Vec<String>]) -> Array2<f64> {
        self.fit(tokenized_docs);
        self.transform(tokenized_docs)
    }

    /// Get the vocabulary
    pub fn get_vocabulary(&self) -> &HashMap<String, usize> {
        &self.vocabulary
    }

    /// Get term by index
    pub fn get_term(&self, index: usize) -> Option<&String> {
        self.inverse_vocabulary.get(index)
    }

    /// Get vocabulary size
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// Get top terms by IDF (most discriminative)
    pub fn top_terms_by_idf(&self, n: usize) -> Vec<(String, f64)> {
        let mut terms_idf: Vec<(String, f64)> = self
            .inverse_vocabulary
            .iter()
            .zip(self.idf_values.iter())
            .map(|(t, &idf)| (t.clone(), idf))
            .collect();

        terms_idf.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        terms_idf.truncate(n);
        terms_idf
    }
}

impl Default for TfIdfVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Count Vectorizer (Bag of Words)
///
/// Converts text documents into term count matrices.
#[derive(Debug, Clone)]
pub struct CountVectorizer {
    /// Vocabulary: term -> index mapping
    vocabulary: HashMap<String, usize>,
    /// Inverse vocabulary: index -> term
    inverse_vocabulary: Vec<String>,
    /// Minimum document frequency
    min_df: usize,
    /// Maximum document frequency ratio
    max_df_ratio: f64,
    /// Maximum vocabulary size
    max_features: Option<usize>,
    /// Whether the vectorizer has been fitted
    is_fitted: bool,
}

impl CountVectorizer {
    /// Create a new count vectorizer
    pub fn new() -> Self {
        Self {
            vocabulary: HashMap::new(),
            inverse_vocabulary: Vec::new(),
            min_df: 1,
            max_df_ratio: 1.0,
            max_features: None,
            is_fitted: false,
        }
    }

    /// Set minimum document frequency
    pub fn min_df(mut self, min_df: usize) -> Self {
        self.min_df = min_df;
        self
    }

    /// Set maximum document frequency ratio
    pub fn max_df_ratio(mut self, ratio: f64) -> Self {
        self.max_df_ratio = ratio;
        self
    }

    /// Set maximum vocabulary size
    pub fn max_features(mut self, max: usize) -> Self {
        self.max_features = Some(max);
        self
    }

    /// Fit the vectorizer
    pub fn fit(&mut self, tokenized_docs: &[Vec<String>]) {
        let n_docs = tokenized_docs.len();

        // Count document frequencies
        let mut term_doc_freq: HashMap<String, usize> = HashMap::new();
        let mut term_total_freq: HashMap<String, usize> = HashMap::new();

        for doc in tokenized_docs {
            let unique_terms: HashSet<&String> = doc.iter().collect();
            for term in &unique_terms {
                *term_doc_freq.entry((*term).clone()).or_insert(0) += 1;
            }
            for term in doc {
                *term_total_freq.entry(term.clone()).or_insert(0) += 1;
            }
        }

        // Filter by document frequency
        let max_df = (n_docs as f64 * self.max_df_ratio) as usize;
        let mut filtered_terms: Vec<(String, usize)> = term_doc_freq
            .into_iter()
            .filter(|(_, df)| *df >= self.min_df && *df <= max_df)
            .map(|(term, _)| {
                let total_freq = term_total_freq.get(&term).cloned().unwrap_or(0);
                (term, total_freq)
            })
            .collect();

        // Sort by frequency (descending)
        filtered_terms.sort_by(|a, b| b.1.cmp(&a.1));

        // Apply max_features limit
        if let Some(max) = self.max_features {
            filtered_terms.truncate(max);
        }

        // Sort alphabetically
        filtered_terms.sort_by(|a, b| a.0.cmp(&b.0));

        // Build vocabulary
        self.vocabulary.clear();
        self.inverse_vocabulary.clear();

        for (idx, (term, _)) in filtered_terms.into_iter().enumerate() {
            self.vocabulary.insert(term.clone(), idx);
            self.inverse_vocabulary.push(term);
        }

        self.is_fitted = true;
    }

    /// Transform tokenized documents into count matrix
    pub fn transform(&self, tokenized_docs: &[Vec<String>]) -> Array2<f64> {
        assert!(self.is_fitted, "Vectorizer must be fitted before transform");

        let n_docs = tokenized_docs.len();
        let n_features = self.vocabulary.len();

        let mut matrix = Array2::zeros((n_docs, n_features));

        for (doc_idx, doc) in tokenized_docs.iter().enumerate() {
            for term in doc {
                if let Some(&term_idx) = self.vocabulary.get(term) {
                    matrix[[doc_idx, term_idx]] += 1.0;
                }
            }
        }

        matrix
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, tokenized_docs: &[Vec<String>]) -> Array2<f64> {
        self.fit(tokenized_docs);
        self.transform(tokenized_docs)
    }

    /// Get the vocabulary
    pub fn get_vocabulary(&self) -> &HashMap<String, usize> {
        &self.vocabulary
    }

    /// Get term by index
    pub fn get_term(&self, index: usize) -> Option<&String> {
        self.inverse_vocabulary.get(index)
    }

    /// Get vocabulary size
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }
}

impl Default for CountVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Document-Term Matrix wrapper with metadata
#[derive(Debug)]
pub struct DocumentTermMatrix {
    /// The actual matrix (n_documents x n_terms)
    pub matrix: Array2<f64>,
    /// Vocabulary mapping
    pub vocabulary: HashMap<String, usize>,
    /// Inverse vocabulary
    pub terms: Vec<String>,
    /// Document IDs
    pub document_ids: Vec<String>,
}

impl DocumentTermMatrix {
    /// Create a new DTM from components
    pub fn new(
        matrix: Array2<f64>,
        vocabulary: HashMap<String, usize>,
        terms: Vec<String>,
        document_ids: Vec<String>,
    ) -> Self {
        Self {
            matrix,
            vocabulary,
            terms,
            document_ids,
        }
    }

    /// Get matrix dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.matrix.nrows(), self.matrix.ncols())
    }

    /// Get number of documents
    pub fn n_documents(&self) -> usize {
        self.matrix.nrows()
    }

    /// Get number of terms
    pub fn n_terms(&self) -> usize {
        self.matrix.ncols()
    }

    /// Get document vector by index
    pub fn get_document(&self, idx: usize) -> Option<Array1<f64>> {
        if idx < self.n_documents() {
            Some(self.matrix.row(idx).to_owned())
        } else {
            None
        }
    }

    /// Get term vector (all documents for one term)
    pub fn get_term_vector(&self, term: &str) -> Option<Array1<f64>> {
        self.vocabulary
            .get(term)
            .map(|&idx| self.matrix.column(idx).to_owned())
    }

    /// Get top terms for a document
    pub fn top_terms_for_document(&self, doc_idx: usize, n: usize) -> Vec<(String, f64)> {
        if doc_idx >= self.n_documents() {
            return vec![];
        }

        let doc_vec = self.matrix.row(doc_idx);
        let mut term_scores: Vec<(usize, f64)> = doc_vec
            .iter()
            .enumerate()
            .filter(|(_, &score)| score > 0.0)
            .map(|(idx, &score)| (idx, score))
            .collect();

        term_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        term_scores.truncate(n);

        term_scores
            .into_iter()
            .filter_map(|(idx, score)| self.terms.get(idx).map(|term| (term.clone(), score)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tfidf_vectorizer() {
        let docs = vec![
            vec!["bitcoin".to_string(), "trading".to_string()],
            vec![
                "ethereum".to_string(),
                "smart".to_string(),
                "contracts".to_string(),
            ],
            vec![
                "bitcoin".to_string(),
                "ethereum".to_string(),
                "comparison".to_string(),
            ],
        ];

        let mut vectorizer = TfIdfVectorizer::new();
        let matrix = vectorizer.fit_transform(&docs);

        assert_eq!(matrix.nrows(), 3);
        assert!(vectorizer.vocabulary_size() > 0);
    }

    #[test]
    fn test_count_vectorizer() {
        let docs = vec![
            vec![
                "hello".to_string(),
                "world".to_string(),
                "hello".to_string(),
            ],
            vec!["world".to_string(), "test".to_string()],
        ];

        let mut vectorizer = CountVectorizer::new();
        let matrix = vectorizer.fit_transform(&docs);

        assert_eq!(matrix.nrows(), 2);
        assert_eq!(vectorizer.vocabulary_size(), 3);
    }

    #[test]
    fn test_dtm() {
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.0, 0.0, 1.0, 3.0]).unwrap();

        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("b".to_string(), 1);
        vocab.insert("c".to_string(), 2);

        let terms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let doc_ids = vec!["doc1".to_string(), "doc2".to_string()];

        let dtm = DocumentTermMatrix::new(matrix, vocab, terms, doc_ids);

        assert_eq!(dtm.shape(), (2, 3));

        let top_terms = dtm.top_terms_for_document(0, 2);
        assert_eq!(top_terms.len(), 2);
        assert_eq!(top_terms[0].0, "b"); // Highest score
    }
}
