//! Simple Spiking Neural Network Example
//!
//! Demonstrates basic usage of the neuromorphic trading library.

use neuromorphic_trading::prelude::*;
use rand::Rng;

fn main() {
    println!("=== Simple SNN Example ===\n");

    // Create a small network
    let config = NetworkConfig {
        input_size: 16,
        hidden_sizes: vec![8],
        output_size: 3,
        tau_m: 20.0,
        threshold: 1.0,
        reset: 0.0,
        rest: 0.0,
    };

    let mut network = SpikingNetwork::new(config);
    println!("Created network with layers: {:?}", network.layer_sizes());
    println!("Total neurons: {}", network.total_neurons());
    println!("Total synapses: {}", network.total_synapses());

    // Create encoder and decoder
    let encoder = RateEncoder::new(EncoderConfig::default());
    let decoder = TradingDecoder::new(DecoderConfig::default());

    // Simulate some market data
    let mut rng = rand::thread_rng();

    println!("\n--- Running simulation ---\n");

    for step in 0..10 {
        // Generate synthetic market data
        let base_price = 50000.0 + rng.gen_range(-500.0..500.0);

        let market_data = MarketData {
            bid_prices: (0..8).map(|i| base_price - 0.5 - i as f64 * 0.1).collect(),
            ask_prices: (0..8).map(|i| base_price + 0.5 + i as f64 * 0.1).collect(),
            bid_volumes: (0..8).map(|_| rng.gen_range(0.1..10.0)).collect(),
            ask_volumes: (0..8).map(|_| rng.gen_range(0.1..10.0)).collect(),
            timestamp: chrono::Utc::now(),
        };

        // Encode market data to spikes
        let input_spikes = encoder.encode(&market_data);

        // Process through network
        let output_spikes = network.step(&input_spikes, 1.0);

        // Decode to trading signal
        let signal = decoder.decode(&output_spikes);

        // Print results
        println!(
            "Step {}: Price=${:.2}, Input spikes={}, Output spikes={}, Signal={:?}",
            step + 1,
            base_price,
            input_spikes.len(),
            output_spikes.len(),
            signal
        );
    }

    println!("\n--- Network state ---");
    let state = network.get_state();
    println!("Average membrane potential: {:.4}", state.avg_membrane_potential);
    println!("Active neurons: {}", state.active_neurons);
    println!("Current time: {:.2}ms", state.current_time);

    println!("\n=== Example Complete ===");
}
