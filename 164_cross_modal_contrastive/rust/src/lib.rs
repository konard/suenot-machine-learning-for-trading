use ndarray::{Array1, ArrayView1};

/// Represents a pre-computed embedding from the Neural Network (either Text or Price)
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

    /// Computes exactly the same Cosine Similarity as PyTorch F.cosine_similarity
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        let dot_product = self.vector.dot(&other.vector);
        let norm_a = self.vector.dot(&self.vector).sqrt();
        let norm_b = other.vector.dot(&other.vector).sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

pub struct CrossModalSearchEngine {
    /// A gallery of pre-computed price charts represented as embeddings
    gallery: Vec<(usize, Embedding)>, 
}

impl CrossModalSearchEngine {
    pub fn new() -> Self {
        CrossModalSearchEngine {
            gallery: Vec::new(),
        }
    }

    /// Adds a price chart embedding to the gallery
    pub fn add_price_chart(&mut self, chart_id: usize, embedding: Embedding) {
        self.gallery.push((chart_id, embedding));
    }

    /// Zero-Shot Retrieval:
    /// Given a Text Embedding (e.g. "Bullish News"), searches the gallery for the most similar Price Chart.
    pub fn search(&self, text_query: &Embedding, top_k: usize) -> Vec<(usize, f32)> {
        let mut results: Vec<(usize, f32)> = self.gallery.iter().map(|(id, price_emb)| {
            let sim = text_query.cosine_similarity(price_emb);
            (*id, sim)
        }).collect();

        // Sort by similarity descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        results.into_iter().take(top_k).collect()
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
    fn test_zero_shot_retrieval() {
        let mut engine = CrossModalSearchEngine::new();
        
        // Mock 4 price charts
        engine.add_price_chart(1, Embedding::new(vec![0.9, 0.1, 0.0])); // Strong match for Q1
        engine.add_price_chart(2, Embedding::new(vec![0.1, 0.9, 0.0])); // Strong match for Q2
        engine.add_price_chart(3, Embedding::new(vec![0.0, 0.1, 0.9]));
        engine.add_price_chart(4, Embedding::new(vec![0.5, 0.5, 0.0]));

        // Text query asking for Chart 2's specific feature
        let text_query = Embedding::new(vec![0.0, 0.95, 0.0]);
        
        let top_results = engine.search(&text_query, 1);
        
        assert_eq!(top_results.len(), 1);
        assert_eq!(top_results[0].0, 2); // Chart 2 should mathematically be the closest
    }
}
