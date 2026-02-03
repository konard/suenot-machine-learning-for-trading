//! Neural network-based counterfactual generator.

use rand::Rng;
use rand_distr::StandardNormal;

/// Neural network-based counterfactual generator.
///
/// Uses an encoder-decoder architecture to generate counterfactual
/// explanations conditioned on a target class.
pub struct CounterfactualGenerator {
    input_dim: usize,
    hidden_dim: usize,
    num_classes: usize,
    // Encoder weights
    enc_w1: Vec<Vec<f64>>,
    enc_b1: Vec<f64>,
    enc_w2: Vec<Vec<f64>>,
    enc_b2: Vec<f64>,
    // Class embeddings
    class_emb: Vec<Vec<f64>>,
    // Decoder weights
    dec_w1: Vec<Vec<f64>>,
    dec_b1: Vec<f64>,
    dec_w2: Vec<Vec<f64>>,
    dec_b2: Vec<f64>,
}

impl CounterfactualGenerator {
    /// Create a new generator with random weights.
    pub fn new(input_dim: usize, hidden_dim: usize, num_classes: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let scale = |fan_in: usize, fan_out: usize| (2.0 / (fan_in + fan_out) as f64).sqrt();

        Self {
            input_dim,
            hidden_dim,
            num_classes,
            // Encoder
            enc_w1: Self::init_weights(&mut rng, hidden_dim, input_dim, scale(input_dim, hidden_dim)),
            enc_b1: vec![0.0; hidden_dim],
            enc_w2: Self::init_weights(&mut rng, hidden_dim, hidden_dim, scale(hidden_dim, hidden_dim)),
            enc_b2: vec![0.0; hidden_dim],
            // Class embeddings
            class_emb: Self::init_weights(&mut rng, num_classes, hidden_dim, scale(num_classes, hidden_dim)),
            // Decoder (takes 2*hidden_dim as input: encoded + class embedding)
            dec_w1: Self::init_weights(&mut rng, hidden_dim, hidden_dim * 2, scale(hidden_dim * 2, hidden_dim)),
            dec_b1: vec![0.0; hidden_dim],
            dec_w2: Self::init_weights(&mut rng, input_dim, hidden_dim, scale(hidden_dim, input_dim)),
            dec_b2: vec![0.0; input_dim],
        }
    }

    fn init_weights<R: Rng>(rng: &mut R, rows: usize, cols: usize, scale: f64) -> Vec<Vec<f64>> {
        (0..rows)
            .map(|_| {
                (0..cols)
                    .map(|_| rng.sample::<f64, _>(StandardNormal) * scale)
                    .collect()
            })
            .collect()
    }

    /// Generate counterfactual for given input and target class.
    pub fn generate(&self, x: &[f64], target_class: usize) -> Vec<f64> {
        // Encode input
        let h1 = self.linear(&self.enc_w1, &self.enc_b1, x);
        let h1 = self.relu(&h1);
        let z = self.linear(&self.enc_w2, &self.enc_b2, &h1);
        let z = self.relu(&z);

        // Get class embedding
        let class_emb = &self.class_emb[target_class];

        // Concatenate encoded representation and class embedding
        let mut z_combined = z.clone();
        z_combined.extend(class_emb.iter());

        // Decode
        let h_dec = self.linear(&self.dec_w1, &self.dec_b1, &z_combined);
        let h_dec = self.relu(&h_dec);
        let delta = self.linear(&self.dec_w2, &self.dec_b2, &h_dec);

        // Generate counterfactual as perturbation
        x.iter().zip(delta.iter()).map(|(&x, &d)| x + d).collect()
    }

    fn linear(&self, w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
        w.iter()
            .zip(b.iter())
            .map(|(row, &bias)| {
                row.iter().zip(x.iter()).map(|(&w, &x)| w * x).sum::<f64>() + bias
            })
            .collect()
    }

    fn relu(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| v.max(0.0)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_creation() {
        let gen = CounterfactualGenerator::new(6, 64, 3);
        assert_eq!(gen.input_dim, 6);
        assert_eq!(gen.hidden_dim, 64);
        assert_eq!(gen.num_classes, 3);
    }

    #[test]
    fn test_generate() {
        let gen = CounterfactualGenerator::new(6, 64, 3);
        let x = vec![0.5, 0.1, -0.2, 0.3, 0.0, -0.1];
        let cf = gen.generate(&x, 2);
        assert_eq!(cf.len(), 6);
    }
}
