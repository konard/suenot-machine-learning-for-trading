//! Basic Spiking Neural Network Example
//!
//! This example demonstrates the fundamentals of SNN:
//! - Creating neurons and networks
//! - Encoding data into spikes
//! - Processing through the network

use snn_trading::{
    neuron::{LIFNeuron, IzhikevichNeuron, Neuron, NeuronParams},
    network::SNNNetwork,
    encoding::{RateEncoder, DeltaEncoder, SpikeEncoder},
};

fn main() {
    println!("=== Spiking Neural Network Basics ===\n");

    // 1. Single Neuron Demo
    demo_single_neuron();

    // 2. Network Demo
    demo_network();

    // 3. Encoding Demo
    demo_encoding();

    println!("\n=== Demo Complete ===");
}

fn demo_single_neuron() {
    println!("--- 1. Single Neuron Dynamics ---\n");

    // Create a LIF neuron
    let mut lif = LIFNeuron::new();
    println!("Created LIF neuron with default parameters");
    println!("  Resting potential: {} mV", lif.params().v_rest);
    println!("  Threshold: {} mV", lif.params().v_thresh);
    println!("  Time constant: {} ms", lif.params().tau_m);

    // Simulate with constant input
    let dt = 0.1; // 0.1 ms timestep
    let input_current = 20.0; // Strong enough to cause spiking

    println!("\nSimulating with constant current = {} nA", input_current);

    let mut spike_times = Vec::new();
    for step in 0..1000 {
        let time = step as f64 * dt;
        if lif.step(input_current, dt) {
            spike_times.push(time);
        }
    }

    println!("Total spikes in 100ms: {}", spike_times.len());
    if spike_times.len() >= 2 {
        let avg_isi: f64 = spike_times.windows(2)
            .map(|w| w[1] - w[0])
            .sum::<f64>() / (spike_times.len() - 1) as f64;
        println!("Average inter-spike interval: {:.2} ms", avg_isi);
        println!("Firing rate: {:.1} Hz", 1000.0 / avg_isi);
    }

    // Demonstrate Izhikevich neuron
    println!("\n--- Izhikevich Neuron Types ---\n");

    let neuron_types = [
        ("Regular Spiking", IzhikevichNeuron::regular_spiking()),
        ("Fast Spiking", IzhikevichNeuron::fast_spiking()),
        ("Intrinsically Bursting", IzhikevichNeuron::intrinsically_bursting()),
        ("Chattering", IzhikevichNeuron::chattering()),
    ];

    for (name, mut neuron) in neuron_types {
        let mut spikes = 0;
        for _ in 0..1000 {
            if neuron.step(10.0, 1.0) {
                spikes += 1;
            }
        }
        println!("{}: {} spikes in 1000ms", name, spikes);
    }
}

fn demo_network() {
    println!("\n--- 2. Network Processing ---\n");

    // Create a simple 3-layer network
    let mut network = SNNNetwork::builder()
        .input_layer(10)
        .hidden_layer(20)
        .output_layer(2)
        .with_dt(1.0)
        .build();

    println!("Created network: 10 -> 20 -> 2");
    println!("  Input size: {}", network.input_size());
    println!("  Output size: {}", network.output_size());
    println!("  Number of layers: {}", network.num_layers());

    // Create input pattern
    let input: Vec<f64> = (0..10).map(|i| {
        if i < 5 { 50.0 } else { 0.0 }  // First half activated
    }).collect();

    println!("\nInput pattern: [50, 50, 50, 50, 50, 0, 0, 0, 0, 0]");

    // Run for 100 timesteps
    let mut output_spikes = vec![0usize; 2];

    for _ in 0..100 {
        let spikes = network.forward(&input);
        for (i, &spiked) in spikes.iter().enumerate() {
            if spiked {
                output_spikes[i] += 1;
            }
        }
    }

    println!("Output spike counts after 100ms:");
    println!("  Neuron 0: {} spikes", output_spikes[0]);
    println!("  Neuron 1: {} spikes", output_spikes[1]);

    // Winner-take-all decision
    let decision = if output_spikes[0] > output_spikes[1] {
        "Class 0 (BUY)"
    } else if output_spikes[1] > output_spikes[0] {
        "Class 1 (SELL)"
    } else {
        "Tie (HOLD)"
    };
    println!("Decision: {}", decision);
}

fn demo_encoding() {
    println!("\n--- 3. Spike Encoding Schemes ---\n");

    // Rate encoding
    println!("Rate Encoding:");
    let rate_encoder = RateEncoder::for_returns();

    let test_returns = [-0.05, -0.02, 0.0, 0.02, 0.05];
    for &ret in &test_returns {
        let rate = rate_encoder.encode(ret);
        println!("  Return {:.1}% -> Rate: {:.1} Hz", ret * 100.0, rate);
    }

    // Delta encoding
    println!("\nDelta Encoding:");
    let mut delta_encoder = DeltaEncoder::for_prices(1);

    let prices = [100.0, 100.5, 100.4, 101.0, 100.8, 102.0];
    delta_encoder.initialize(&[prices[0]]);

    println!("  Price sequence: {:?}", prices);
    print!("  Spikes:         ");

    for &price in &prices[1..] {
        let results = delta_encoder.process(&[price], 0.0);
        let spike_char = match &results[0] {
            Some((spike, _)) => {
                if spike.polarity == snn_trading::neuron::SpikePolarity::Positive {
                    "↑"
                } else {
                    "↓"
                }
            }
            None => "·",
        };
        print!("{} ", spike_char);
    }
    println!();

    // Demonstrate population coding
    println!("\nPopulation Encoding (conceptual):");
    let value = 0.7;
    let num_neurons = 5;
    println!("  Value: {}", value);
    print!("  Neurons: ");
    for i in 0..num_neurons {
        let preferred = (i as f64 + 0.5) / num_neurons as f64;
        let activation = (-(value - preferred).powi(2) / 0.1).exp();
        let bar_len = (activation * 10.0) as usize;
        print!("[{}{}] ", "█".repeat(bar_len), " ".repeat(10 - bar_len));
    }
    println!();
}
