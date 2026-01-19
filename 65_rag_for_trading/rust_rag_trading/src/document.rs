//! Document types and structures for RAG.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Type of financial document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocumentType {
    /// News article
    News,
    /// SEC filing (10-K, 10-Q, 8-K, etc.)
    Filing,
    /// Earnings call transcript
    Earnings,
    /// Research report
    Research,
    /// Social media post
    Social,
    /// Other document type
    Other,
}

impl Default for DocumentType {
    fn default() -> Self {
        DocumentType::News
    }
}

impl std::fmt::Display for DocumentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DocumentType::News => write!(f, "news"),
            DocumentType::Filing => write!(f, "filing"),
            DocumentType::Earnings => write!(f, "earnings"),
            DocumentType::Research => write!(f, "research"),
            DocumentType::Social => write!(f, "social"),
            DocumentType::Other => write!(f, "other"),
        }
    }
}

/// A financial document with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique document identifier
    pub id: String,
    /// Document text content
    pub text: String,
    /// Associated ticker symbol (optional)
    pub ticker: Option<String>,
    /// Document source (e.g., "Reuters", "Bloomberg")
    pub source: String,
    /// Document publication/filing date
    pub date: DateTime<Utc>,
    /// Document type
    pub doc_type: DocumentType,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Document {
    /// Create a new document with auto-generated ID.
    ///
    /// # Arguments
    ///
    /// * `text` - Document text content
    /// * `ticker` - Optional ticker symbol
    /// * `source` - Document source
    /// * `date` - Publication date
    ///
    /// # Example
    ///
    /// ```
    /// use rag_trading::Document;
    /// use chrono::Utc;
    ///
    /// let doc = Document::new(
    ///     "Tesla reports record deliveries...",
    ///     Some("TSLA".to_string()),
    ///     "Reuters",
    ///     Utc::now()
    /// );
    /// ```
    pub fn new(
        text: impl Into<String>,
        ticker: Option<String>,
        source: impl Into<String>,
        date: DateTime<Utc>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string()[..12].to_string(),
            text: text.into(),
            ticker,
            source: source.into(),
            date,
            doc_type: DocumentType::News,
            metadata: HashMap::new(),
        }
    }

    /// Create a document with all fields specified.
    pub fn with_all(
        id: impl Into<String>,
        text: impl Into<String>,
        ticker: Option<String>,
        source: impl Into<String>,
        date: DateTime<Utc>,
        doc_type: DocumentType,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            ticker,
            source: source.into(),
            date,
            doc_type,
            metadata,
        }
    }

    /// Set the document type.
    pub fn with_type(mut self, doc_type: DocumentType) -> Self {
        self.doc_type = doc_type;
        self
    }

    /// Add metadata to the document.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get a text snippet (first n characters).
    pub fn snippet(&self, max_len: usize) -> String {
        if self.text.len() <= max_len {
            self.text.clone()
        } else {
            format!("{}...", &self.text[..max_len])
        }
    }

    /// Check if document matches a ticker filter.
    pub fn matches_ticker(&self, ticker: Option<&str>) -> bool {
        match (ticker, &self.ticker) {
            (None, _) => true,
            (Some(t), Some(dt)) => t.eq_ignore_ascii_case(dt),
            (Some(_), None) => false,
        }
    }

    /// Check if document is within a date range.
    pub fn in_date_range(
        &self,
        min_date: Option<DateTime<Utc>>,
        max_date: Option<DateTime<Utc>>,
    ) -> bool {
        if let Some(min) = min_date {
            if self.date < min {
                return false;
            }
        }
        if let Some(max) = max_date {
            if self.date > max {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_creation() {
        let doc = Document::new(
            "Tesla reports record deliveries",
            Some("TSLA".to_string()),
            "Reuters",
            Utc::now(),
        );

        assert_eq!(doc.ticker, Some("TSLA".to_string()));
        assert_eq!(doc.source, "Reuters");
        assert!(!doc.id.is_empty());
    }

    #[test]
    fn test_document_snippet() {
        let doc = Document::new(
            "This is a long document that should be truncated",
            None,
            "Test",
            Utc::now(),
        );

        let snippet = doc.snippet(20);
        assert_eq!(snippet, "This is a long docum...");
    }

    #[test]
    fn test_ticker_matching() {
        let doc = Document::new("Test", Some("AAPL".to_string()), "Test", Utc::now());

        assert!(doc.matches_ticker(Some("AAPL")));
        assert!(doc.matches_ticker(Some("aapl")));
        assert!(!doc.matches_ticker(Some("TSLA")));
        assert!(doc.matches_ticker(None));
    }
}
