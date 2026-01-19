//! Document retrieval with keyword-based search.

use crate::document::Document;
// Note: RagError and Result are available via crate::error if needed
use chrono::{DateTime, Utc};
use std::collections::HashSet;

/// Result of a document search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The matched document
    pub document: Document,
    /// Relevance score (0-1, higher is better)
    pub score: f64,
    /// Relevant text highlights
    pub highlights: Vec<String>,
}

impl SearchResult {
    /// Create a new search result.
    pub fn new(document: Document, score: f64, highlights: Vec<String>) -> Self {
        Self {
            document,
            score,
            highlights,
        }
    }
}

/// Document retriever with keyword-based search.
///
/// Uses TF-IDF style scoring for document retrieval.
/// For production use with embeddings, integrate with a vector database.
pub struct DocumentRetriever {
    documents: Vec<Document>,
}

impl DocumentRetriever {
    /// Create a new empty retriever.
    pub fn new() -> Self {
        Self {
            documents: Vec::new(),
        }
    }

    /// Add a single document to the index.
    pub fn add_document(&mut self, document: Document) {
        self.documents.push(document);
    }

    /// Add multiple documents to the index.
    pub fn add_documents(&mut self, documents: Vec<Document>) -> usize {
        let count = documents.len();
        self.documents.extend(documents);
        count
    }

    /// Get the number of indexed documents.
    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Clear all documents from the index.
    pub fn clear(&mut self) {
        self.documents.clear();
    }

    /// Search for relevant documents.
    ///
    /// # Arguments
    ///
    /// * `query` - Search query string
    /// * `top_k` - Number of results to return
    /// * `ticker` - Optional ticker filter
    ///
    /// # Returns
    ///
    /// Vector of search results sorted by relevance.
    pub fn search(
        &self,
        query: &str,
        top_k: usize,
        ticker: Option<&str>,
    ) -> Vec<SearchResult> {
        self.search_with_filters(query, top_k, ticker, None, None, None)
    }

    /// Search with additional filters.
    pub fn search_with_filters(
        &self,
        query: &str,
        top_k: usize,
        ticker: Option<&str>,
        doc_type: Option<&str>,
        min_date: Option<DateTime<Utc>>,
        max_date: Option<DateTime<Utc>>,
    ) -> Vec<SearchResult> {
        if self.documents.is_empty() {
            return Vec::new();
        }

        // Tokenize query
        let query_words: HashSet<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        if query_words.is_empty() {
            return Vec::new();
        }

        // Score each document
        let mut results: Vec<SearchResult> = self
            .documents
            .iter()
            .filter(|doc| {
                // Apply filters
                if !doc.matches_ticker(ticker) {
                    return false;
                }
                if !doc.in_date_range(min_date, max_date) {
                    return false;
                }
                if let Some(dt) = doc_type {
                    if doc.doc_type.to_string() != dt {
                        return false;
                    }
                }
                true
            })
            .map(|doc| {
                let score = self.compute_score(&query_words, &doc.text);
                let highlights = self.extract_highlights(&query_words, &doc.text);
                SearchResult::new(doc.clone(), score, highlights)
            })
            .filter(|r| r.score > 0.0)
            .collect();

        // Sort by score (descending)
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k
        results.truncate(top_k);
        results
    }

    /// Compute relevance score using keyword overlap.
    fn compute_score(&self, query_words: &HashSet<String>, text: &str) -> f64 {
        let text_lower = text.to_lowercase();
        let text_words: HashSet<&str> = text_lower.split_whitespace().collect();

        let mut matches = 0;
        for word in query_words {
            if text_words.contains(word.as_str()) || text_lower.contains(word) {
                matches += 1;
            }
        }

        if query_words.is_empty() {
            0.0
        } else {
            matches as f64 / query_words.len() as f64
        }
    }

    /// Extract relevant text snippets.
    fn extract_highlights(&self, query_words: &HashSet<String>, text: &str) -> Vec<String> {
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        let mut scored: Vec<(usize, &str)> = sentences
            .iter()
            .map(|&s| {
                let s_lower = s.to_lowercase();
                let score = query_words
                    .iter()
                    .filter(|w| s_lower.contains(w.as_str()))
                    .count();
                (score, s)
            })
            .filter(|(score, _)| *score > 0)
            .collect();

        scored.sort_by(|a, b| b.0.cmp(&a.0));

        scored
            .into_iter()
            .take(3)
            .map(|(_, s)| s.trim().to_string())
            .collect()
    }

    /// Get all documents for a ticker.
    pub fn get_by_ticker(&self, ticker: &str) -> Vec<&Document> {
        self.documents
            .iter()
            .filter(|d| d.matches_ticker(Some(ticker)))
            .collect()
    }

    /// Get document by ID.
    pub fn get_by_id(&self, id: &str) -> Option<&Document> {
        self.documents.iter().find(|d| d.id == id)
    }
}

impl Default for DocumentRetriever {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_docs() -> Vec<Document> {
        vec![
            Document::new(
                "Tesla reported record Q4 deliveries beating expectations",
                Some("TSLA".to_string()),
                "Reuters",
                Utc::now(),
            ),
            Document::new(
                "Apple iPhone sales strong in holiday quarter",
                Some("AAPL".to_string()),
                "Bloomberg",
                Utc::now(),
            ),
            Document::new(
                "Federal Reserve holds interest rates steady",
                None,
                "Fed",
                Utc::now(),
            ),
        ]
    }

    #[test]
    fn test_add_documents() {
        let mut retriever = DocumentRetriever::new();
        let docs = create_test_docs();

        let count = retriever.add_documents(docs);
        assert_eq!(count, 3);
        assert_eq!(retriever.document_count(), 3);
    }

    #[test]
    fn test_search() {
        let mut retriever = DocumentRetriever::new();
        retriever.add_documents(create_test_docs());

        let results = retriever.search("Tesla deliveries", 5, None);
        assert!(!results.is_empty());
        assert!(results[0].document.ticker == Some("TSLA".to_string()));
    }

    #[test]
    fn test_search_with_ticker_filter() {
        let mut retriever = DocumentRetriever::new();
        retriever.add_documents(create_test_docs());

        let results = retriever.search("sales", 5, Some("AAPL"));
        assert!(!results.is_empty());
        for r in results {
            assert_eq!(r.document.ticker, Some("AAPL".to_string()));
        }
    }

    #[test]
    fn test_empty_search() {
        let retriever = DocumentRetriever::new();
        let results = retriever.search("test", 5, None);
        assert!(results.is_empty());
    }
}
