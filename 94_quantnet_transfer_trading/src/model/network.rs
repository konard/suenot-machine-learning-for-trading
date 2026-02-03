//! QuantNet encoder-decoder network with asset-specific trading heads.
//!
//! Architecture:
//!   Input -> [Encoder] -> Latent -> [Decoder] -> Reconstruction
//!                       |
//!                       +-> TradingHead("BTCUSDT") -> signal
//!                       +-> TradingHead("ETHUSDT") -> signal
//!                       +-> ...

use rand::Rng;

/// Output from a trading head for a single asset.
#[derive(Debug, Clone)]
pub struct TradingOutput {
    /// Predicted position signal in [-1, 1]
    pub signal: f64,
    /// Predicted return magnitude
    pub return_pred: f64,
    /// Confidence score in [0, 1]
    pub confidence: f64,
}

/// Predictions from the QuantNet model for all assets.
#[derive(Debug, Clone)]
pub struct QuantNetPredictions {
    /// Per-asset trading outputs, keyed by asset name
    pub asset_outputs: Vec<(String, TradingOutput)>,
    /// Reconstructed input from decoder
    pub reconstruction: Vec<f64>,
    /// Latent representation
    pub latent: Vec<f64>,
}

/// A single linear layer: y = Wx + b with Xavier initialization.
#[derive(Debug, Clone)]
struct LinearLayer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    input_size: usize,
    output_size: usize,
}

impl LinearLayer {
    /// Create a new linear layer with Xavier (Glorot) initialization.
    fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        // Xavier uniform: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
        let limit = (6.0 / (input_size + output_size) as f64).sqrt();

        let weights = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.gen_range(-limit..limit))
                    .collect()
            })
            .collect();
        let biases = vec![0.0; output_size];

        Self { weights, biases, input_size, output_size }
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.input_size);
        (0..self.output_size)
            .map(|o| {
                let dot: f64 = self.weights[o].iter().zip(input).map(|(w, x)| w * x).sum();
                dot + self.biases[o]
            })
            .collect()
    }

    fn parameters(&self) -> Vec<f64> {
        let mut params = Vec::new();
        for row in &self.weights {
            params.extend(row);
        }
        params.extend(&self.biases);
        params
    }

    fn set_parameters(&mut self, params: &[f64]) -> usize {
        let mut idx = 0;
        for row in &mut self.weights {
            for w in row.iter_mut() {
                *w = params[idx];
                idx += 1;
            }
        }
        for b in &mut self.biases {
            *b = params[idx];
            idx += 1;
        }
        idx
    }

    fn num_parameters(&self) -> usize {
        self.input_size * self.output_size + self.output_size
    }
}

/// GELU activation: x * Phi(x) approximation.
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// Encoder: maps input features to a latent representation.
#[derive(Debug, Clone)]
struct Encoder {
    layers: Vec<LinearLayer>,
}

impl Encoder {
    fn new(input_size: usize, hidden_size: usize, latent_size: usize) -> Self {
        let layers = vec![
            LinearLayer::new(input_size, hidden_size),
            LinearLayer::new(hidden_size, hidden_size),
            LinearLayer::new(hidden_size, latent_size),
        ];
        Self { layers }
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut x = input.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            // Apply GELU activation on all layers except the last
            if i < self.layers.len() - 1 {
                x.iter_mut().for_each(|v| *v = gelu(*v));
            }
        }
        x
    }

    fn parameters(&self) -> Vec<f64> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    fn set_parameters(&mut self, params: &[f64]) -> usize {
        let mut offset = 0;
        for layer in &mut self.layers {
            offset += layer.set_parameters(&params[offset..]);
        }
        offset
    }

    fn num_parameters(&self) -> usize {
        self.layers.iter().map(|l| l.num_parameters()).sum()
    }
}

/// Decoder: mirrors the encoder to reconstruct inputs from latent space.
#[derive(Debug, Clone)]
struct Decoder {
    layers: Vec<LinearLayer>,
}

impl Decoder {
    fn new(latent_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let layers = vec![
            LinearLayer::new(latent_size, hidden_size),
            LinearLayer::new(hidden_size, hidden_size),
            LinearLayer::new(hidden_size, output_size),
        ];
        Self { layers }
    }

    fn forward(&self, latent: &[f64]) -> Vec<f64> {
        let mut x = latent.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            if i < self.layers.len() - 1 {
                x.iter_mut().for_each(|v| *v = gelu(*v));
            }
        }
        x
    }

    fn parameters(&self) -> Vec<f64> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    fn set_parameters(&mut self, params: &[f64]) -> usize {
        let mut offset = 0;
        for layer in &mut self.layers {
            offset += layer.set_parameters(&params[offset..]);
        }
        offset
    }

    fn num_parameters(&self) -> usize {
        self.layers.iter().map(|l| l.num_parameters()).sum()
    }
}

/// Asset-specific trading head that produces trading signals from the latent space.
#[derive(Debug, Clone)]
struct TradingHead {
    asset_name: String,
    hidden: LinearLayer,
    signal_layer: LinearLayer,
    return_layer: LinearLayer,
    confidence_layer: LinearLayer,
}

impl TradingHead {
    fn new(asset_name: &str, latent_size: usize, head_hidden: usize) -> Self {
        Self {
            asset_name: asset_name.to_string(),
            hidden: LinearLayer::new(latent_size, head_hidden),
            signal_layer: LinearLayer::new(head_hidden, 1),
            return_layer: LinearLayer::new(head_hidden, 1),
            confidence_layer: LinearLayer::new(head_hidden, 1),
        }
    }

    fn forward(&self, latent: &[f64]) -> TradingOutput {
        // Hidden layer with GELU activation
        let h: Vec<f64> = self.hidden.forward(latent)
            .into_iter()
            .map(gelu)
            .collect();

        // Signal in [-1, 1] via tanh
        let raw_signal = self.signal_layer.forward(&h)[0];
        let signal = raw_signal.tanh();

        // Return prediction (unbounded)
        let return_pred = self.return_layer.forward(&h)[0];

        // Confidence in [0, 1] via sigmoid
        let raw_conf = self.confidence_layer.forward(&h)[0];
        let confidence = 1.0 / (1.0 + (-raw_conf).exp());

        TradingOutput { signal, return_pred, confidence }
    }

    fn parameters(&self) -> Vec<f64> {
        let mut params = self.hidden.parameters();
        params.extend(self.signal_layer.parameters());
        params.extend(self.return_layer.parameters());
        params.extend(self.confidence_layer.parameters());
        params
    }

    fn set_parameters(&mut self, params: &[f64]) -> usize {
        let mut offset = 0;
        offset += self.hidden.set_parameters(&params[offset..]);
        offset += self.signal_layer.set_parameters(&params[offset..]);
        offset += self.return_layer.set_parameters(&params[offset..]);
        offset += self.confidence_layer.set_parameters(&params[offset..]);
        offset
    }

    fn num_parameters(&self) -> usize {
        self.hidden.num_parameters()
            + self.signal_layer.num_parameters()
            + self.return_layer.num_parameters()
            + self.confidence_layer.num_parameters()
    }
}

/// QuantNet model combining encoder, decoder, and asset-specific trading heads.
///
/// The encoder maps features to a shared latent space. The decoder reconstructs
/// inputs for representation learning. Trading heads produce asset-specific signals.
#[derive(Debug, Clone)]
pub struct QuantNetModel {
    encoder: Encoder,
    decoder: Decoder,
    trading_heads: Vec<TradingHead>,
    input_size: usize,
    latent_size: usize,
}

impl QuantNetModel {
    /// Create a new QuantNet model.
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Hidden layer size for encoder/decoder
    /// * `latent_size` - Dimension of the latent space
    /// * `asset_names` - Names of assets for trading heads
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        latent_size: usize,
        asset_names: &[&str],
    ) -> Self {
        let head_hidden = latent_size;

        let trading_heads = asset_names
            .iter()
            .map(|name| TradingHead::new(name, latent_size, head_hidden))
            .collect();

        Self {
            encoder: Encoder::new(input_size, hidden_size, latent_size),
            decoder: Decoder::new(latent_size, hidden_size, input_size),
            trading_heads,
            input_size,
            latent_size,
        }
    }

    /// Forward pass for a single sample.
    pub fn forward(&self, input: &[f64]) -> QuantNetPredictions {
        let latent = self.encoder.forward(input);
        let reconstruction = self.decoder.forward(&latent);

        let asset_outputs = self.trading_heads
            .iter()
            .map(|head| (head.asset_name.clone(), head.forward(&latent)))
            .collect();

        QuantNetPredictions { asset_outputs, reconstruction, latent }
    }

    /// Forward pass for a batch of samples.
    pub fn forward_batch(&self, inputs: &[Vec<f64>]) -> Vec<QuantNetPredictions> {
        inputs.iter().map(|input| self.forward(input)).collect()
    }

    /// Encode input to latent space (for transfer learning).
    pub fn encode(&self, input: &[f64]) -> Vec<f64> {
        self.encoder.forward(input)
    }

    /// Decode from latent space to input space.
    pub fn decode(&self, latent: &[f64]) -> Vec<f64> {
        self.decoder.forward(latent)
    }

    /// Get trading signal for a specific asset.
    pub fn predict_asset(&self, input: &[f64], asset_index: usize) -> Option<TradingOutput> {
        let latent = self.encoder.forward(input);
        self.trading_heads.get(asset_index).map(|head| head.forward(&latent))
    }

    /// Compute Sharpe ratio loss for a sequence of returns and signals.
    ///
    /// Loss = -Sharpe = -(mean(r * s) / std(r * s))
    /// where r = actual returns, s = predicted signals.
    pub fn sharpe_loss(returns: &[f64], signals: &[f64]) -> f64 {
        if returns.len() < 2 || returns.len() != signals.len() {
            return 0.0;
        }

        let portfolio_returns: Vec<f64> = returns
            .iter()
            .zip(signals)
            .map(|(r, s)| r * s)
            .collect();

        let n = portfolio_returns.len() as f64;
        let mean: f64 = portfolio_returns.iter().sum::<f64>() / n;
        let variance: f64 = portfolio_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / n;
        let std = variance.sqrt();

        if std < 1e-10 {
            return 0.0;
        }

        // Negative Sharpe (since we minimize the loss)
        -(mean / std) * (252.0_f64).sqrt()
    }

    /// Get all model parameters as a flat vector.
    pub fn parameters(&self) -> Vec<f64> {
        let mut params = self.encoder.parameters();
        params.extend(self.decoder.parameters());
        for head in &self.trading_heads {
            params.extend(head.parameters());
        }
        params
    }

    /// Set model parameters from a flat vector.
    pub fn set_parameters(&mut self, params: &[f64]) {
        let mut offset = 0;
        offset += self.encoder.set_parameters(&params[offset..]);
        offset += self.decoder.set_parameters(&params[offset..]);
        for head in &mut self.trading_heads {
            offset += head.set_parameters(&params[offset..]);
        }
    }

    /// Get only the encoder parameters.
    pub fn encoder_parameters(&self) -> Vec<f64> {
        self.encoder.parameters()
    }

    /// Get only the decoder parameters.
    pub fn decoder_parameters(&self) -> Vec<f64> {
        self.decoder.parameters()
    }

    /// Total number of trainable parameters.
    pub fn num_parameters(&self) -> usize {
        self.encoder.num_parameters()
            + self.decoder.num_parameters()
            + self.trading_heads.iter().map(|h| h.num_parameters()).sum::<usize>()
    }

    /// Number of encoder parameters.
    pub fn num_encoder_parameters(&self) -> usize {
        self.encoder.num_parameters()
    }

    /// Number of decoder parameters.
    pub fn num_decoder_parameters(&self) -> usize {
        self.decoder.num_parameters()
    }

    /// Get the input size.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get the latent space dimension.
    pub fn latent_size(&self) -> usize {
        self.latent_size
    }

    /// Get the number of asset heads.
    pub fn num_assets(&self) -> usize {
        self.trading_heads.len()
    }

    /// Get the asset names.
    pub fn asset_names(&self) -> Vec<String> {
        self.trading_heads.iter().map(|h| h.asset_name.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let model = QuantNetModel::new(10, 32, 16, &["BTCUSDT", "ETHUSDT"]);
        assert!(model.num_parameters() > 0);
        assert_eq!(model.input_size(), 10);
        assert_eq!(model.latent_size(), 16);
        assert_eq!(model.num_assets(), 2);
    }

    #[test]
    fn test_forward_pass() {
        let model = QuantNetModel::new(5, 16, 8, &["BTCUSDT", "ETHUSDT", "SOLUSDT"]);
        let input = vec![0.1, -0.2, 0.3, 0.0, 0.5];
        let preds = model.forward(&input);

        assert_eq!(preds.asset_outputs.len(), 3);
        assert_eq!(preds.reconstruction.len(), 5);
        assert_eq!(preds.latent.len(), 8);

        // Signals should be in [-1, 1] (tanh output)
        for (_, output) in &preds.asset_outputs {
            assert!(output.signal >= -1.0 && output.signal <= 1.0);
            assert!(output.confidence >= 0.0 && output.confidence <= 1.0);
        }
    }

    #[test]
    fn test_batch_forward() {
        let model = QuantNetModel::new(3, 8, 4, &["BTC", "ETH"]);
        let inputs = vec![
            vec![0.1, 0.2, 0.3],
            vec![-0.1, 0.0, 0.5],
            vec![0.5, -0.3, 0.1],
        ];
        let batch_preds = model.forward_batch(&inputs);
        assert_eq!(batch_preds.len(), 3);
    }

    #[test]
    fn test_encode_decode() {
        let model = QuantNetModel::new(5, 16, 8, &["BTC"]);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let latent = model.encode(&input);
        assert_eq!(latent.len(), 8);
        let decoded = model.decode(&latent);
        assert_eq!(decoded.len(), 5);
    }

    #[test]
    fn test_parameter_roundtrip() {
        let model = QuantNetModel::new(5, 16, 8, &["BTC"]);
        let params = model.parameters();
        let mut model2 = QuantNetModel::new(5, 16, 8, &["BTC"]);
        model2.set_parameters(&params);

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p1 = model.forward(&input);
        let p2 = model2.forward(&input);

        assert!((p1.asset_outputs[0].1.signal - p2.asset_outputs[0].1.signal).abs() < 1e-10);
    }

    #[test]
    fn test_sharpe_loss() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015];
        let signals = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let loss = QuantNetModel::sharpe_loss(&returns, &signals);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_gelu_activation() {
        // GELU(0) should be ~0
        assert!((gelu(0.0)).abs() < 1e-7);
        // GELU for large positive should be close to identity
        assert!((gelu(3.0) - 3.0).abs() < 0.01);
        // GELU for large negative should be close to 0
        assert!(gelu(-3.0).abs() < 0.01);
    }

    #[test]
    fn test_predict_asset() {
        let model = QuantNetModel::new(5, 16, 8, &["BTC", "ETH"]);
        let input = vec![0.1, -0.2, 0.3, 0.0, 0.5];

        let btc = model.predict_asset(&input, 0);
        assert!(btc.is_some());
        let btc = btc.unwrap();
        assert!(btc.signal >= -1.0 && btc.signal <= 1.0);

        let none = model.predict_asset(&input, 5);
        assert!(none.is_none());
    }
}
