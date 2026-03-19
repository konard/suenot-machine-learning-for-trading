//! Generative LOB (Limit Order Book) Trading
//!
//! This crate implements generative models for limit order book simulation,
//! including VAE encoding/decoding, synthetic order flow generation,
//! statistical validation, and Bybit API integration.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// LOB State Representation
// ---------------------------------------------------------------------------

/// A single price level in the order book.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: f64,
    pub quantity: f64,
}

/// A snapshot of the limit order book at a single point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LobSnapshot {
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
    pub timestamp: u64,
}

impl LobSnapshot {
    /// Create a new LOB snapshot.
    pub fn new(bids: Vec<PriceLevel>, asks: Vec<PriceLevel>, timestamp: u64) -> Self {
        Self {
            bids,
            asks,
            timestamp,
        }
    }

    /// Compute the mid-price.
    pub fn mid_price(&self) -> f64 {
        if self.bids.is_empty() || self.asks.is_empty() {
            return 0.0;
        }
        (self.bids[0].price + self.asks[0].price) / 2.0
    }

    /// Compute the bid-ask spread.
    pub fn spread(&self) -> f64 {
        if self.bids.is_empty() || self.asks.is_empty() {
            return 0.0;
        }
        self.asks[0].price - self.bids[0].price
    }

    /// Flatten to a normalized feature vector.
    /// For each level: (relative_bid_price, log_bid_qty, relative_ask_price, log_ask_qty)
    pub fn to_feature_vector(&self, num_levels: usize) -> Array1<f64> {
        let mid = self.mid_price();
        let mut features = Vec::with_capacity(num_levels * 4);

        for i in 0..num_levels {
            if i < self.bids.len() && mid > 0.0 {
                features.push((self.bids[i].price - mid) / mid);
                features.push((1.0 + self.bids[i].quantity).ln());
            } else {
                features.push(0.0);
                features.push(0.0);
            }
            if i < self.asks.len() && mid > 0.0 {
                features.push((self.asks[i].price - mid) / mid);
                features.push((1.0 + self.asks[i].quantity).ln());
            } else {
                features.push(0.0);
                features.push(0.0);
            }
        }

        Array1::from_vec(features)
    }

    /// Reconstruct a LOB snapshot from a feature vector and a reference mid-price.
    pub fn from_feature_vector(
        features: &Array1<f64>,
        mid_price: f64,
        num_levels: usize,
    ) -> Self {
        let mut bids = Vec::with_capacity(num_levels);
        let mut asks = Vec::with_capacity(num_levels);

        for i in 0..num_levels {
            let base = i * 4;
            if base + 3 < features.len() {
                let bid_price = features[base] * mid_price + mid_price;
                let bid_qty = features[base + 1].exp() - 1.0;
                let ask_price = features[base + 2] * mid_price + mid_price;
                let ask_qty = features[base + 3].exp() - 1.0;

                bids.push(PriceLevel {
                    price: bid_price,
                    quantity: bid_qty.max(0.0),
                });
                asks.push(PriceLevel {
                    price: ask_price,
                    quantity: ask_qty.max(0.0),
                });
            }
        }

        LobSnapshot::new(bids, asks, 0)
    }
}

// ---------------------------------------------------------------------------
// VAE for LOB Snapshots
// ---------------------------------------------------------------------------

/// Simple dense layer: y = Wx + b
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
}

impl DenseLayer {
    /// Create a new dense layer with Xavier initialization.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();

        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let bias = Array1::zeros(output_dim);

        Self { weights, bias }
    }

    /// Forward pass.
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        x.dot(&self.weights) + &self.bias
    }
}

/// Variational Autoencoder for LOB snapshots.
#[derive(Debug, Clone)]
pub struct LobVae {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub latent_dim: usize,
    // Encoder layers
    pub encoder_hidden: DenseLayer,
    pub encoder_mu: DenseLayer,
    pub encoder_log_var: DenseLayer,
    // Decoder layers
    pub decoder_hidden: DenseLayer,
    pub decoder_output: DenseLayer,
}

impl LobVae {
    /// Create a new VAE with specified dimensions.
    pub fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            latent_dim,
            encoder_hidden: DenseLayer::new(input_dim, hidden_dim),
            encoder_mu: DenseLayer::new(hidden_dim, latent_dim),
            encoder_log_var: DenseLayer::new(hidden_dim, latent_dim),
            decoder_hidden: DenseLayer::new(latent_dim, hidden_dim),
            decoder_output: DenseLayer::new(hidden_dim, input_dim),
        }
    }

    /// ReLU activation.
    fn relu(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| v.max(0.0))
    }

    /// Encode an input vector to latent parameters (mu, log_var).
    pub fn encode(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let h = Self::relu(&self.encoder_hidden.forward(x));
        let mu = self.encoder_mu.forward(&h);
        let log_var = self.encoder_log_var.forward(&h);
        (mu, log_var)
    }

    /// Reparameterization trick: z = mu + sigma * epsilon.
    pub fn reparameterize(&self, mu: &Array1<f64>, log_var: &Array1<f64>) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let std = log_var.mapv(|v| (v * 0.5).exp());
        let epsilon = Array1::from_shape_fn(self.latent_dim, |_| rng.gen_range(-1.0..1.0));
        mu + &(std * epsilon)
    }

    /// Decode a latent vector to a reconstructed input.
    pub fn decode(&self, z: &Array1<f64>) -> Array1<f64> {
        let h = Self::relu(&self.decoder_hidden.forward(z));
        self.decoder_output.forward(&h)
    }

    /// Full forward pass: encode, sample, decode.
    pub fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let (mu, log_var) = self.encode(x);
        let z = self.reparameterize(&mu, &log_var);
        let x_recon = self.decode(&z);
        (x_recon, mu, log_var)
    }

    /// Compute VAE loss: reconstruction + KL divergence.
    pub fn loss(&self, x: &Array1<f64>, x_recon: &Array1<f64>, mu: &Array1<f64>, log_var: &Array1<f64>) -> f64 {
        // MSE reconstruction loss
        let diff = x - x_recon;
        let recon_loss: f64 = diff.mapv(|v| v * v).sum() / x.len() as f64;

        // KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        let kl_loss: f64 = -0.5
            * (log_var.mapv(|v| 1.0 + v) - mu.mapv(|v| v * v) - log_var.mapv(|v| v.exp()))
                .sum();

        recon_loss + kl_loss
    }

    /// Simple training step using numerical gradient descent.
    pub fn train_step(
        &mut self,
        data: &[Array1<f64>],
        learning_rate: f64,
    ) -> f64 {
        let mut total_loss = 0.0;
        let n = data.len() as f64;

        for x in data {
            let (x_recon, mu, log_var) = self.forward(x);
            total_loss += self.loss(x, &x_recon, &mu, &log_var);

            // Simplified gradient update: perturb decoder output bias toward data
            let error = x - &x_recon;
            let update = &error * learning_rate;

            // Update decoder output bias
            let new_bias = &self.decoder_output.bias + &(&update * 0.01);
            self.decoder_output.bias = new_bias;
        }

        total_loss / n
    }

    /// Generate a synthetic LOB feature vector by sampling from the prior.
    pub fn generate(&self) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let z = Array1::from_shape_fn(self.latent_dim, |_| rng.gen_range(-1.0..1.0));
        self.decode(&z)
    }

    /// Generate multiple synthetic LOB feature vectors.
    pub fn generate_batch(&self, count: usize) -> Vec<Array1<f64>> {
        (0..count).map(|_| self.generate()).collect()
    }
}

// ---------------------------------------------------------------------------
// LOB Generator (high-level interface)
// ---------------------------------------------------------------------------

/// High-level generator for synthetic LOB states.
pub struct LobGenerator {
    pub vae: LobVae,
    pub num_levels: usize,
    pub reference_mid_price: f64,
}

impl LobGenerator {
    /// Create a new LOB generator.
    pub fn new(num_levels: usize, hidden_dim: usize, latent_dim: usize) -> Self {
        let input_dim = num_levels * 4;
        Self {
            vae: LobVae::new(input_dim, hidden_dim, latent_dim),
            num_levels,
            reference_mid_price: 50000.0, // default for BTC
        }
    }

    /// Train the generator on real LOB snapshots.
    pub fn train(&mut self, snapshots: &[LobSnapshot], epochs: usize, learning_rate: f64) -> Vec<f64> {
        let data: Vec<Array1<f64>> = snapshots
            .iter()
            .map(|s| s.to_feature_vector(self.num_levels))
            .collect();

        if !snapshots.is_empty() {
            self.reference_mid_price = snapshots
                .iter()
                .map(|s| s.mid_price())
                .sum::<f64>()
                / snapshots.len() as f64;
        }

        let mut losses = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            let loss = self.vae.train_step(&data, learning_rate);
            losses.push(loss);
        }
        losses
    }

    /// Generate a single synthetic LOB snapshot.
    pub fn generate_snapshot(&self) -> LobSnapshot {
        let features = self.vae.generate();
        LobSnapshot::from_feature_vector(&features, self.reference_mid_price, self.num_levels)
    }

    /// Generate multiple synthetic LOB snapshots.
    pub fn generate_snapshots(&self, count: usize) -> Vec<LobSnapshot> {
        (0..count).map(|_| self.generate_snapshot()).collect()
    }
}

// ---------------------------------------------------------------------------
// Statistical Validation
// ---------------------------------------------------------------------------

/// Statistical validator for comparing real and synthetic LOB data.
pub struct LobValidator;

impl LobValidator {
    /// Compute basic statistics (mean, std, min, max) for a set of values.
    pub fn basic_stats(values: &[f64]) -> (f64, f64, f64, f64) {
        if values.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (mean, std, min, max)
    }

    /// Extract mid-prices from snapshots.
    pub fn mid_prices(snapshots: &[LobSnapshot]) -> Vec<f64> {
        snapshots.iter().map(|s| s.mid_price()).collect()
    }

    /// Extract spreads from snapshots.
    pub fn spreads(snapshots: &[LobSnapshot]) -> Vec<f64> {
        snapshots.iter().map(|s| s.spread()).collect()
    }

    /// Extract total bid volume at top N levels.
    pub fn bid_volumes(snapshots: &[LobSnapshot], levels: usize) -> Vec<f64> {
        snapshots
            .iter()
            .map(|s| {
                s.bids
                    .iter()
                    .take(levels)
                    .map(|l| l.quantity)
                    .sum::<f64>()
            })
            .collect()
    }

    /// Extract total ask volume at top N levels.
    pub fn ask_volumes(snapshots: &[LobSnapshot], levels: usize) -> Vec<f64> {
        snapshots
            .iter()
            .map(|s| {
                s.asks
                    .iter()
                    .take(levels)
                    .map(|l| l.quantity)
                    .sum::<f64>()
            })
            .collect()
    }

    /// Compare real and synthetic LOB data, returning a validation report.
    pub fn validate(
        real: &[LobSnapshot],
        synthetic: &[LobSnapshot],
    ) -> ValidationReport {
        let real_mids = Self::mid_prices(real);
        let synth_mids = Self::mid_prices(synthetic);
        let real_spreads = Self::spreads(real);
        let synth_spreads = Self::spreads(synthetic);
        let real_bid_vols = Self::bid_volumes(real, 5);
        let synth_bid_vols = Self::bid_volumes(synthetic, 5);

        ValidationReport {
            real_mid_stats: Self::basic_stats(&real_mids),
            synthetic_mid_stats: Self::basic_stats(&synth_mids),
            real_spread_stats: Self::basic_stats(&real_spreads),
            synthetic_spread_stats: Self::basic_stats(&synth_spreads),
            real_volume_stats: Self::basic_stats(&real_bid_vols),
            synthetic_volume_stats: Self::basic_stats(&synth_bid_vols),
        }
    }
}

/// Validation report comparing real and synthetic LOB statistics.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub real_mid_stats: (f64, f64, f64, f64),
    pub synthetic_mid_stats: (f64, f64, f64, f64),
    pub real_spread_stats: (f64, f64, f64, f64),
    pub synthetic_spread_stats: (f64, f64, f64, f64),
    pub real_volume_stats: (f64, f64, f64, f64),
    pub synthetic_volume_stats: (f64, f64, f64, f64),
}

impl std::fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== LOB Validation Report ===")?;
        writeln!(f, "\nMid-Price Statistics (mean, std, min, max):")?;
        writeln!(f, "  Real:      ({:.2}, {:.4}, {:.2}, {:.2})",
            self.real_mid_stats.0, self.real_mid_stats.1,
            self.real_mid_stats.2, self.real_mid_stats.3)?;
        writeln!(f, "  Synthetic: ({:.2}, {:.4}, {:.2}, {:.2})",
            self.synthetic_mid_stats.0, self.synthetic_mid_stats.1,
            self.synthetic_mid_stats.2, self.synthetic_mid_stats.3)?;
        writeln!(f, "\nSpread Statistics (mean, std, min, max):")?;
        writeln!(f, "  Real:      ({:.4}, {:.4}, {:.4}, {:.4})",
            self.real_spread_stats.0, self.real_spread_stats.1,
            self.real_spread_stats.2, self.real_spread_stats.3)?;
        writeln!(f, "  Synthetic: ({:.4}, {:.4}, {:.4}, {:.4})",
            self.synthetic_spread_stats.0, self.synthetic_spread_stats.1,
            self.synthetic_spread_stats.2, self.synthetic_spread_stats.3)?;
        writeln!(f, "\nBid Volume Statistics (mean, std, min, max):")?;
        writeln!(f, "  Real:      ({:.4}, {:.4}, {:.4}, {:.4})",
            self.real_volume_stats.0, self.real_volume_stats.1,
            self.real_volume_stats.2, self.real_volume_stats.3)?;
        writeln!(f, "  Synthetic: ({:.4}, {:.4}, {:.4}, {:.4})",
            self.synthetic_volume_stats.0, self.synthetic_volume_stats.1,
            self.synthetic_volume_stats.2, self.synthetic_volume_stats.3)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Bybit API Client
// ---------------------------------------------------------------------------

/// Response structures for the Bybit V5 orderbook endpoint.
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitOrderbookResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitOrderbookResult {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "b")]
    pub bids: Vec<Vec<String>>,
    #[serde(rename = "a")]
    pub asks: Vec<Vec<String>>,
    pub ts: u64,
}

/// Client for fetching LOB data from Bybit.
pub struct BybitClient {
    pub base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch the order book for a given symbol (blocking).
    pub fn fetch_orderbook_blocking(
        &self,
        symbol: &str,
        category: &str,
        limit: u32,
    ) -> Result<LobSnapshot> {
        let url = format!(
            "{}/v5/market/orderbook?category={}&symbol={}&limit={}",
            self.base_url, category, symbol, limit
        );

        let response: BybitResponse = reqwest::blocking::get(&url)
            .context("Failed to fetch orderbook")?
            .json()
            .context("Failed to parse orderbook response")?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let bids = response
            .result
            .bids
            .iter()
            .filter_map(|level| {
                if level.len() >= 2 {
                    Some(PriceLevel {
                        price: level[0].parse().unwrap_or(0.0),
                        quantity: level[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks = response
            .result
            .asks
            .iter()
            .filter_map(|level| {
                if level.len() >= 2 {
                    Some(PriceLevel {
                        price: level[0].parse().unwrap_or(0.0),
                        quantity: level[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(LobSnapshot::new(bids, asks, response.result.ts))
    }

    /// Fetch the order book for a given symbol (async).
    pub async fn fetch_orderbook(
        &self,
        symbol: &str,
        category: &str,
        limit: u32,
    ) -> Result<LobSnapshot> {
        let url = format!(
            "{}/v5/market/orderbook?category={}&symbol={}&limit={}",
            self.base_url, category, symbol, limit
        );

        let client = reqwest::Client::new();
        let response: BybitResponse = client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch orderbook")?
            .json()
            .await
            .context("Failed to parse orderbook response")?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let bids = response
            .result
            .bids
            .iter()
            .filter_map(|level| {
                if level.len() >= 2 {
                    Some(PriceLevel {
                        price: level[0].parse().unwrap_or(0.0),
                        quantity: level[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks = response
            .result
            .asks
            .iter()
            .filter_map(|level| {
                if level.len() >= 2 {
                    Some(PriceLevel {
                        price: level[0].parse().unwrap_or(0.0),
                        quantity: level[1].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(LobSnapshot::new(bids, asks, response.result.ts))
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper: generate synthetic LOB data for testing
// ---------------------------------------------------------------------------

/// Generate a set of realistic-looking synthetic LOB snapshots for testing.
pub fn generate_test_snapshots(count: usize, num_levels: usize, base_price: f64) -> Vec<LobSnapshot> {
    let mut rng = rand::thread_rng();
    let mut snapshots = Vec::with_capacity(count);

    for i in 0..count {
        let mid = base_price + rng.gen_range(-100.0..100.0);
        let spread = rng.gen_range(0.5..5.0);

        let mut bids = Vec::with_capacity(num_levels);
        let mut asks = Vec::with_capacity(num_levels);

        for j in 0..num_levels {
            let bid_price = mid - spread / 2.0 - j as f64 * rng.gen_range(0.5..2.0);
            let ask_price = mid + spread / 2.0 + j as f64 * rng.gen_range(0.5..2.0);
            let bid_qty = rng.gen_range(0.1..10.0);
            let ask_qty = rng.gen_range(0.1..10.0);

            bids.push(PriceLevel {
                price: bid_price,
                quantity: bid_qty,
            });
            asks.push(PriceLevel {
                price: ask_price,
                quantity: ask_qty,
            });
        }

        snapshots.push(LobSnapshot::new(bids, asks, i as u64));
    }

    snapshots
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lob_snapshot_mid_price() {
        let snap = LobSnapshot::new(
            vec![PriceLevel { price: 100.0, quantity: 1.0 }],
            vec![PriceLevel { price: 102.0, quantity: 1.0 }],
            0,
        );
        assert!((snap.mid_price() - 101.0).abs() < 1e-10);
    }

    #[test]
    fn test_lob_snapshot_spread() {
        let snap = LobSnapshot::new(
            vec![PriceLevel { price: 100.0, quantity: 1.0 }],
            vec![PriceLevel { price: 102.0, quantity: 1.0 }],
            0,
        );
        assert!((snap.spread() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_feature_vector_roundtrip() {
        let snap = LobSnapshot::new(
            vec![
                PriceLevel { price: 99.0, quantity: 5.0 },
                PriceLevel { price: 98.0, quantity: 3.0 },
            ],
            vec![
                PriceLevel { price: 101.0, quantity: 4.0 },
                PriceLevel { price: 102.0, quantity: 2.0 },
            ],
            0,
        );
        let mid = snap.mid_price();
        let features = snap.to_feature_vector(2);
        assert_eq!(features.len(), 8);

        let reconstructed = LobSnapshot::from_feature_vector(&features, mid, 2);
        assert!((reconstructed.mid_price() - mid).abs() < 0.01);
    }

    #[test]
    fn test_vae_encode_decode() {
        let vae = LobVae::new(20, 16, 4);
        let input = Array1::from_vec(vec![0.1; 20]);

        let (mu, log_var) = vae.encode(&input);
        assert_eq!(mu.len(), 4);
        assert_eq!(log_var.len(), 4);

        let z = vae.reparameterize(&mu, &log_var);
        let decoded = vae.decode(&z);
        assert_eq!(decoded.len(), 20);
    }

    #[test]
    fn test_vae_forward_and_loss() {
        let vae = LobVae::new(20, 16, 4);
        let input = Array1::from_vec(vec![0.1; 20]);

        let (x_recon, mu, log_var) = vae.forward(&input);
        let loss = vae.loss(&input, &x_recon, &mu, &log_var);

        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_generator_generate() {
        let generator = LobGenerator::new(5, 16, 4);
        let snapshot = generator.generate_snapshot();

        assert_eq!(snapshot.bids.len(), 5);
        assert_eq!(snapshot.asks.len(), 5);
        assert!(snapshot.mid_price() > 0.0);
    }

    #[test]
    fn test_generator_train_and_generate() {
        let mut generator = LobGenerator::new(5, 16, 4);
        let data = generate_test_snapshots(20, 5, 50000.0);

        let losses = generator.train(&data, 3, 0.001);
        assert_eq!(losses.len(), 3);

        let synthetic = generator.generate_snapshots(10);
        assert_eq!(synthetic.len(), 10);
    }

    #[test]
    fn test_validation_report() {
        let real = generate_test_snapshots(50, 5, 50000.0);
        let synthetic = generate_test_snapshots(50, 5, 50000.0);

        let report = LobValidator::validate(&real, &synthetic);
        assert!(report.real_mid_stats.0 > 0.0);
        assert!(report.synthetic_mid_stats.0 > 0.0);
    }

    #[test]
    fn test_basic_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, std, min, max) = LobValidator::basic_stats(&values);
        assert!((mean - 3.0).abs() < 1e-10);
        assert!((min - 1.0).abs() < 1e-10);
        assert!((max - 5.0).abs() < 1e-10);
        assert!(std > 0.0);
    }

    #[test]
    fn test_empty_snapshot() {
        let snap = LobSnapshot::new(vec![], vec![], 0);
        assert_eq!(snap.mid_price(), 0.0);
        assert_eq!(snap.spread(), 0.0);
    }
}
