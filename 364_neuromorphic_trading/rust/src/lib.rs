//! # Neuromorphic Trading Library
//!
//! A high-performance spiking neural network library for algorithmic trading,
//! designed for ultra-low-latency cryptocurrency market analysis and execution.
//!
//! ## Features
//!
//! - **Spiking Neural Networks**: LIF and Izhikevich neuron models
//! - **Spike Encoding**: Rate, temporal, and delta modulation encoders
//! - **Learning Rules**: STDP and reward-modulated learning
//! - **Exchange Integration**: Bybit WebSocket and REST API
//! - **Trading Strategies**: Neuromorphic signal generation
//!
//! ## Example
//!
//! ```rust,no_run
//! use neuromorphic_trading::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create a simple SNN
//!     let mut network = SpikingNetwork::new(NetworkConfig {
//!         input_size: 128,
//!         hidden_sizes: vec![64],
//!         output_size: 3,
//!         ..Default::default()
//!     });
//!
//!     // Create encoder for market data
//!     let encoder = RateEncoder::new(EncoderConfig::default());
//!
//!     // Create decoder for trading signals
//!     let decoder = TradingDecoder::new(DecoderConfig::default());
//!
//!     // Process market data
//!     let market_data = MarketData {
//!         bid_prices: vec![50000.0; 8],
//!         ask_prices: vec![50001.0; 8],
//!         bid_volumes: vec![1.0; 8],
//!         ask_volumes: vec![1.0; 8],
//!         timestamp: chrono::Utc::now(),
//!     };
//!
//!     let spikes = encoder.encode(&market_data);
//!     let output_spikes = network.step(&spikes, 1.0);
//!     let signal = decoder.decode(&output_spikes);
//!
//!     println!("Trading signal: {:?}", signal);
//!     Ok(())
//! }
//! ```

pub mod neuron;
pub mod network;
pub mod encoder;
pub mod decoder;
pub mod exchange;
pub mod strategy;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::neuron::{
        Neuron, NeuronState, SpikeEvent,
        lif::{LIFNeuron, LIFConfig},
        izhikevich::{IzhikevichNeuron, IzhikevichConfig, NeuronType as IzhikevichType},
        synapse::{Synapse, SynapseConfig},
    };
    pub use crate::network::{
        SpikingNetwork, NetworkConfig,
        layer::{Layer, LayerConfig},
        learning::{STDPConfig, LearningRule},
    };
    pub use crate::encoder::{
        Encoder, EncoderConfig, MarketData,
        rate::RateEncoder,
        temporal::TemporalEncoder,
        delta::DeltaEncoder,
    };
    pub use crate::decoder::{
        Decoder, DecoderConfig, TradingSignal,
        trading::TradingDecoder,
    };
    pub use crate::exchange::{
        ExchangeClient, OrderBook, Trade, Ticker,
        bybit::{BybitClient, BybitConfig},
    };
    pub use crate::strategy::{
        TradingStrategy, StrategyConfig,
        neuromorphic::NeuromorphicStrategy,
    };
}

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default simulation timestep in milliseconds
pub const DEFAULT_TIMESTEP_MS: f64 = 1.0;

/// Default membrane time constant in milliseconds
pub const DEFAULT_TAU_M: f64 = 20.0;

/// Default spike threshold
pub const DEFAULT_THRESHOLD: f64 = 1.0;

/// Default reset potential
pub const DEFAULT_RESET: f64 = 0.0;

/// Default resting potential
pub const DEFAULT_REST: f64 = 0.0;
