//! # Spiking Neural Networks for Trading
//!
//! This library implements Spiking Neural Networks (SNNs) for cryptocurrency
//! trading applications, specifically designed to work with Bybit exchange data.
//!
//! ## Modules
//!
//! - `neuron`: Neuron models (LIF, Izhikevich)
//! - `network`: Network architecture and layers
//! - `encoding`: Spike encoding schemes
//! - `learning`: Learning rules (STDP, R-STDP)
//! - `trading`: Trading strategies and signals
//! - `data`: Bybit data fetching and processing

pub mod neuron;
pub mod network;
pub mod encoding;
pub mod learning;
pub mod trading;
pub mod data;

pub use neuron::{LIFNeuron, IzhikevichNeuron, Spike, Neuron};
pub use network::{SNNLayer, SNNNetwork, LayerConfig};
pub use encoding::{SpikeEncoder, RateEncoder, TemporalEncoder, DeltaEncoder};
pub use learning::{STDP, RewardModulatedSTDP, LearningRule};
pub use trading::{TradingSignal, TradingStrategy, SNNTradingStrategy, StrategyParams};
pub use data::{BybitClient, Candle, OrderBook, Trade};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default simulation timestep in milliseconds
pub const DEFAULT_DT: f64 = 1.0;

/// Result type for this library
pub type Result<T> = std::result::Result<T, Error>;

/// Library error type
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid configuration: {0}")]
    Config(String),

    #[error("Simulation error: {0}")]
    Simulation(String),

    #[error("Data error: {0}")]
    Data(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
