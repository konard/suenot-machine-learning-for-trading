//! Cross-Modal Contrastive Learning — Rust inference library.
//!
//! Provides high-performance embedding similarity search for production
//! trading systems. Works with pre-computed embeddings from the Python
//! CLIP-style dual-encoder model.
//!
//! Supports both stock market and cryptocurrency (Bybit) data.

use ndarray::Array1;

/// Represents a pre-computed embedding from the Neural Network
/// (either Text or Price encoder output).
#[derive(Debug, Clone)]
pub struct Embedding {
    pub vector: Array1<f32>,
}

impl Embedding {
    pub fn new(data: Vec<f32>) -> Self {
        Embedding {
            vector: Array1::from(data),
        }
    }

    /// L2-normalize the embedding vector in-place.
    pub fn normalize(&mut self) {
        let norm = self.vector.dot(&self.vector).sqrt();
        if norm > 1e-8 {
            self.vector.mapv_inplace(|x| x / norm);
        }
    }

    /// Computes cosine similarity with another embedding.
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        let dot_product = self.vector.dot(&other.vector);
        let norm_a = self.vector.dot(&self.vector).sqrt();
        let norm_b = other.vector.dot(&other.vector).sqrt();

        if norm_a < 1e-8 || norm_b < 1e-8 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

/// Metadata for a stored chart embedding (supports both stock and crypto).
#[derive(Debug, Clone)]
pub struct ChartMetadata {
    pub chart_id: usize,
    pub symbol: String,
    pub market_type: MarketType,
    pub description: String,
}

/// Market type for the chart (Stock or Crypto/Bybit).
#[derive(Debug, Clone, PartialEq)]
pub enum MarketType {
    Stock,
    Crypto,
}

/// Search result from cross-modal retrieval.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chart_id: usize,
    pub symbol: String,
    pub market_type: MarketType,
    pub similarity: f32,
    pub description: String,
}

/// Cross-Modal Search Engine for zero-shot retrieval.
///
/// Stores pre-computed price chart embeddings and retrieves them
/// by cosine similarity against text query embeddings.
pub struct CrossModalSearchEngine {
    gallery: Vec<(ChartMetadata, Embedding)>,
}

impl CrossModalSearchEngine {
    pub fn new() -> Self {
        CrossModalSearchEngine {
            gallery: Vec::new(),
        }
    }

    /// Add a price chart embedding with metadata to the gallery.
    pub fn add_chart(&mut self, metadata: ChartMetadata, embedding: Embedding) {
        self.gallery.push((metadata, embedding));
    }

    /// Convenience: add a chart by ID and embedding only (backward compat).
    pub fn add_price_chart(&mut self, chart_id: usize, embedding: Embedding) {
        let meta = ChartMetadata {
            chart_id,
            symbol: format!("CHART_{}", chart_id),
            market_type: MarketType::Stock,
            description: String::new(),
        };
        self.gallery.push((meta, embedding));
    }

    /// Zero-Shot Retrieval: find the top-K most similar price charts
    /// for a given text query embedding.
    pub fn search(&self, text_query: &Embedding, top_k: usize) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = self
            .gallery
            .iter()
            .map(|(meta, price_emb)| SearchResult {
                chart_id: meta.chart_id,
                symbol: meta.symbol.clone(),
                market_type: meta.market_type.clone(),
                similarity: text_query.cosine_similarity(price_emb),
                description: meta.description.clone(),
            })
            .collect();

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.into_iter().take(top_k).collect()
    }

    /// Filter search results by market type (Stock or Crypto).
    pub fn search_by_market(
        &self,
        text_query: &Embedding,
        market_type: MarketType,
        top_k: usize,
    ) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = self
            .gallery
            .iter()
            .filter(|(meta, _)| meta.market_type == market_type)
            .map(|(meta, price_emb)| SearchResult {
                chart_id: meta.chart_id,
                symbol: meta.symbol.clone(),
                market_type: meta.market_type.clone(),
                similarity: text_query.cosine_similarity(price_emb),
                description: meta.description.clone(),
            })
            .collect();

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.into_iter().take(top_k).collect()
    }

    /// Compute the divergence score between a price embedding and a text embedding.
    /// High divergence = narrative doesn't match price action (potential trading signal).
    pub fn divergence_score(&self, price_emb: &Embedding, text_emb: &Embedding) -> f32 {
        1.0 - price_emb.cosine_similarity(text_emb)
    }

    /// Number of charts in the gallery.
    pub fn gallery_size(&self) -> usize {
        self.gallery.len()
    }
}

impl Default for CrossModalSearchEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let emb_a = Embedding::new(vec![1.0, 0.0, 0.0]);
        let emb_b = Embedding::new(vec![1.0, 0.0, 0.0]);
        let emb_c = Embedding::new(vec![0.0, 1.0, 0.0]);
        let emb_d = Embedding::new(vec![-1.0, 0.0, 0.0]);

        assert!((emb_a.cosine_similarity(&emb_b) - 1.0).abs() < 1e-5);
        assert!((emb_a.cosine_similarity(&emb_c) - 0.0).abs() < 1e-5);
        assert!((emb_a.cosine_similarity(&emb_d) - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_normalize() {
        let mut emb = Embedding::new(vec![3.0, 4.0, 0.0]);
        emb.normalize();
        let norm = emb.vector.dot(&emb.vector).sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        assert!((emb.vector[0] - 0.6).abs() < 1e-5);
        assert!((emb.vector[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_zero_shot_retrieval() {
        let mut engine = CrossModalSearchEngine::new();

        engine.add_price_chart(1, Embedding::new(vec![0.9, 0.1, 0.0]));
        engine.add_price_chart(2, Embedding::new(vec![0.1, 0.9, 0.0]));
        engine.add_price_chart(3, Embedding::new(vec![0.0, 0.1, 0.9]));
        engine.add_price_chart(4, Embedding::new(vec![0.5, 0.5, 0.0]));

        let text_query = Embedding::new(vec![0.0, 0.95, 0.0]);
        let top_results = engine.search(&text_query, 1);

        assert_eq!(top_results.len(), 1);
        assert_eq!(top_results[0].chart_id, 2);
    }

    #[test]
    fn test_cross_market_search() {
        let mut engine = CrossModalSearchEngine::new();

        // Crypto charts (Bybit)
        engine.add_chart(
            ChartMetadata {
                chart_id: 1,
                symbol: "BTCUSDT".to_string(),
                market_type: MarketType::Crypto,
                description: "BTC pump".to_string(),
            },
            Embedding::new(vec![0.9, 0.1, 0.0]),
        );
        engine.add_chart(
            ChartMetadata {
                chart_id: 2,
                symbol: "ETHUSDT".to_string(),
                market_type: MarketType::Crypto,
                description: "ETH crash".to_string(),
            },
            Embedding::new(vec![0.1, 0.9, 0.0]),
        );

        // Stock chart
        engine.add_chart(
            ChartMetadata {
                chart_id: 3,
                symbol: "AAPL".to_string(),
                market_type: MarketType::Stock,
                description: "AAPL earnings beat".to_string(),
            },
            Embedding::new(vec![0.8, 0.2, 0.0]),
        );

        // Search only crypto charts
        let query = Embedding::new(vec![0.9, 0.0, 0.0]);
        let crypto_results = engine.search_by_market(&query, MarketType::Crypto, 1);
        assert_eq!(crypto_results.len(), 1);
        assert_eq!(crypto_results[0].symbol, "BTCUSDT");

        // Search all charts
        let all_results = engine.search(&query, 2);
        assert_eq!(all_results.len(), 2);
    }

    #[test]
    fn test_divergence_score() {
        let engine = CrossModalSearchEngine::new();

        // Aligned: price and text point the same direction
        let price = Embedding::new(vec![1.0, 0.0, 0.0]);
        let text_aligned = Embedding::new(vec![0.9, 0.1, 0.0]);
        let div_low = engine.divergence_score(&price, &text_aligned);

        // Diverged: price and text point opposite directions
        let text_diverged = Embedding::new(vec![-0.9, 0.1, 0.0]);
        let div_high = engine.divergence_score(&price, &text_diverged);

        assert!(div_low < 0.1, "Aligned pair should have low divergence");
        assert!(div_high > 1.5, "Diverged pair should have high divergence");
    }
}
