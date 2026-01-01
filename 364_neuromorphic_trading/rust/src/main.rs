//! Neuromorphic Trading CLI Application
//!
//! Command-line interface for running neuromorphic trading strategies
//! on cryptocurrency markets using Bybit exchange.

use anyhow::Result;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

use neuromorphic_trading::prelude::*;

/// CLI configuration
#[derive(Debug)]
struct CliConfig {
    /// Trading symbol (e.g., "BTCUSDT")
    symbol: String,
    /// Bybit API key (optional for public data)
    api_key: Option<String>,
    /// Bybit API secret (optional for public data)
    api_secret: Option<String>,
    /// Enable testnet
    testnet: bool,
    /// Simulation mode (no real trades)
    simulation: bool,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            symbol: "BTCUSDT".to_string(),
            api_key: None,
            api_secret: None,
            testnet: true,
            simulation: true,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("=================================================");
    info!("  Neuromorphic Trading System v{}", neuromorphic_trading::VERSION);
    info!("  Spiking Neural Networks for Crypto Trading");
    info!("=================================================");

    // Load configuration
    let config = load_config()?;

    info!("Symbol: {}", config.symbol);
    info!("Mode: {}", if config.simulation { "Simulation" } else { "Live" });
    info!("Network: {}", if config.testnet { "Testnet" } else { "Mainnet" });

    // Create SNN network
    let network_config = NetworkConfig {
        input_size: 128,
        hidden_sizes: vec![64, 32],
        output_size: 3,
        tau_m: 20.0,
        threshold: 1.0,
        reset: 0.0,
        rest: 0.0,
    };
    let mut network = SpikingNetwork::new(network_config);
    info!("Created SNN with {} input, {:?} hidden, {} output neurons",
          128, vec![64, 32], 3);

    // Create encoder and decoder
    let encoder = RateEncoder::new(EncoderConfig::default());
    let decoder = TradingDecoder::new(DecoderConfig::default());

    // Create Bybit client
    let bybit_config = BybitConfig {
        api_key: config.api_key,
        api_secret: config.api_secret,
        testnet: config.testnet,
    };
    let client = BybitClient::new(bybit_config);

    info!("Starting market data stream...");

    // Create neuromorphic strategy
    let strategy_config = StrategyConfig {
        confidence_threshold: 0.6,
        max_position_size: 0.01,
        spike_rate_threshold: 100.0,
    };
    let strategy = NeuromorphicStrategy::new(strategy_config);

    // Main trading loop
    run_trading_loop(&client, &mut network, &encoder, &decoder, &strategy, &config).await?;

    Ok(())
}

/// Load configuration from environment or defaults
fn load_config() -> Result<CliConfig> {
    dotenv::dotenv().ok();

    let config = CliConfig {
        symbol: std::env::var("SYMBOL").unwrap_or_else(|_| "BTCUSDT".to_string()),
        api_key: std::env::var("BYBIT_API_KEY").ok(),
        api_secret: std::env::var("BYBIT_API_SECRET").ok(),
        testnet: std::env::var("TESTNET")
            .map(|v| v.to_lowercase() == "true")
            .unwrap_or(true),
        simulation: std::env::var("SIMULATION")
            .map(|v| v.to_lowercase() == "true")
            .unwrap_or(true),
    };

    Ok(config)
}

/// Main trading loop
async fn run_trading_loop(
    client: &BybitClient,
    network: &mut SpikingNetwork,
    encoder: &RateEncoder,
    decoder: &TradingDecoder,
    strategy: &NeuromorphicStrategy,
    config: &CliConfig,
) -> Result<()> {
    info!("Fetching initial orderbook for {}...", config.symbol);

    // Demo: Fetch orderbook and process
    match client.get_orderbook(&config.symbol, 25).await {
        Ok(orderbook) => {
            info!("Received orderbook with {} bids and {} asks",
                  orderbook.bids.len(), orderbook.asks.len());

            // Convert orderbook to market data
            let market_data = orderbook_to_market_data(&orderbook);

            // Encode to spikes
            let input_spikes = encoder.encode(&market_data);
            info!("Encoded {} input spikes", input_spikes.len());

            // Process through SNN
            let output_spikes = network.step(&input_spikes, 1.0);
            info!("Generated {} output spikes", output_spikes.len());

            // Decode to trading signal
            let signal = decoder.decode(&output_spikes);
            info!("Trading signal: {:?}", signal);

            // Validate with strategy
            let network_state = network.get_state();
            if strategy.validate_signal(&signal, &network_state) {
                match signal {
                    TradingSignal::Buy { confidence, urgency } => {
                        info!("SIGNAL: BUY (confidence: {:.2}%, urgency: {:.2})",
                              confidence * 100.0, urgency);
                        if !config.simulation {
                            warn!("Live trading not implemented in this demo");
                        }
                    }
                    TradingSignal::Sell { confidence, urgency } => {
                        info!("SIGNAL: SELL (confidence: {:.2}%, urgency: {:.2})",
                              confidence * 100.0, urgency);
                        if !config.simulation {
                            warn!("Live trading not implemented in this demo");
                        }
                    }
                    TradingSignal::Hold => {
                        info!("SIGNAL: HOLD");
                    }
                }
            } else {
                info!("Signal rejected by risk management");
            }
        }
        Err(e) => {
            warn!("Failed to fetch orderbook: {}", e);
            info!("Running in offline demo mode with synthetic data...");

            // Generate synthetic data for demo
            run_synthetic_demo(network, encoder, decoder, strategy)?;
        }
    }

    info!("Demo completed. In production, this would run continuously.");

    Ok(())
}

/// Convert orderbook to market data format
fn orderbook_to_market_data(orderbook: &OrderBook) -> MarketData {
    let n_levels = 8.min(orderbook.bids.len()).min(orderbook.asks.len());

    MarketData {
        bid_prices: orderbook.bids.iter().take(n_levels).map(|l| l.price).collect(),
        ask_prices: orderbook.asks.iter().take(n_levels).map(|l| l.price).collect(),
        bid_volumes: orderbook.bids.iter().take(n_levels).map(|l| l.quantity).collect(),
        ask_volumes: orderbook.asks.iter().take(n_levels).map(|l| l.quantity).collect(),
        timestamp: chrono::Utc::now(),
    }
}

/// Run demo with synthetic data
fn run_synthetic_demo(
    network: &mut SpikingNetwork,
    encoder: &RateEncoder,
    decoder: &TradingDecoder,
    strategy: &NeuromorphicStrategy,
) -> Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    info!("Running synthetic data demo (10 iterations)...");

    let base_price = 50000.0;

    for i in 0..10 {
        // Generate synthetic market data
        let price_change: f64 = rng.gen_range(-100.0..100.0);
        let current_price = base_price + price_change;

        let market_data = MarketData {
            bid_prices: (0..8).map(|j| current_price - 0.5 - j as f64 * 0.1).collect(),
            ask_prices: (0..8).map(|j| current_price + 0.5 + j as f64 * 0.1).collect(),
            bid_volumes: (0..8).map(|_| rng.gen_range(0.1..10.0)).collect(),
            ask_volumes: (0..8).map(|_| rng.gen_range(0.1..10.0)).collect(),
            timestamp: chrono::Utc::now(),
        };

        // Process through SNN
        let input_spikes = encoder.encode(&market_data);
        let output_spikes = network.step(&input_spikes, 1.0);
        let signal = decoder.decode(&output_spikes);

        // Log results
        let network_state = network.get_state();
        let valid = strategy.validate_signal(&signal, &network_state);

        info!(
            "Step {}: Price=${:.2}, Signal={:?}, Valid={}",
            i + 1,
            current_price,
            signal,
            valid
        );
    }

    Ok(())
}
