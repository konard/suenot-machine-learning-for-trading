//! Configuration for the Disentangled VAE

use serde::{Deserialize, Serialize};

/// Disentanglement method
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DisentanglementMethod {
    /// Beta-VAE: increase KL weight with beta > 1
    BetaVAE,
    /// FactorVAE: adds total correlation penalty via discriminator
    FactorVAE,
    /// DIP-VAE: penalizes covariance of inferred moments
    DIPVAE,
    /// Beta-TC-VAE: decomposes KL into index-code MI, TC, and dimension-wise KL
    BetaTCVAE,
}

impl std::fmt::Display for DisentanglementMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DisentanglementMethod::BetaVAE => write!(f, "BetaVAE"),
            DisentanglementMethod::FactorVAE => write!(f, "FactorVAE"),
            DisentanglementMethod::DIPVAE => write!(f, "DIP-VAE"),
            DisentanglementMethod::BetaTCVAE => write!(f, "BetaTCVAE"),
        }
    }
}

/// Configuration for the Disentangled VAE model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisentangledVAEConfig {
    /// Input feature dimension (per time step)
    pub input_dim: usize,
    /// Latent space dimensionality
    pub latent_dim: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Number of hidden layers in encoder/decoder
    pub num_hidden_layers: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Output dimension (for prediction head)
    pub output_dim: usize,
    /// Beta weight for KL divergence
    pub beta: f64,
    /// Disentanglement method
    pub method: DisentanglementMethod,
    /// Learning rate
    pub learning_rate: f64,
    /// Dropout rate
    pub dropout: f64,
    /// DIP-VAE lambda for diagonal covariance penalty
    pub dip_lambda_diag: f64,
    /// DIP-VAE lambda for off-diagonal covariance penalty
    pub dip_lambda_offdiag: f64,
}

impl Default for DisentangledVAEConfig {
    fn default() -> Self {
        Self {
            input_dim: 8,
            latent_dim: crate::defaults::LATENT_DIM,
            hidden_dim: crate::defaults::HIDDEN_DIM,
            num_hidden_layers: 2,
            seq_len: crate::defaults::SEQ_LEN,
            output_dim: 1,
            beta: crate::defaults::BETA,
            method: DisentanglementMethod::BetaVAE,
            learning_rate: crate::defaults::LEARNING_RATE,
            dropout: 0.1,
            dip_lambda_diag: 10.0,
            dip_lambda_offdiag: 5.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DisentangledVAEConfig::default();
        assert_eq!(config.latent_dim, 8);
        assert_eq!(config.hidden_dim, 128);
        assert!((config.beta - 4.0).abs() < 0.001);
        assert_eq!(config.method, DisentanglementMethod::BetaVAE);
    }

    #[test]
    fn test_method_display() {
        assert_eq!(format!("{}", DisentanglementMethod::BetaVAE), "BetaVAE");
        assert_eq!(format!("{}", DisentanglementMethod::FactorVAE), "FactorVAE");
        assert_eq!(format!("{}", DisentanglementMethod::DIPVAE), "DIP-VAE");
        assert_eq!(format!("{}", DisentanglementMethod::BetaTCVAE), "BetaTCVAE");
    }
}
