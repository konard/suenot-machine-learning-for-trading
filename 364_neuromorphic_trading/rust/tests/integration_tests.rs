//! Integration Tests for Neuromorphic Trading Library

use neuromorphic_trading::prelude::*;

/// Test complete pipeline from market data to trading signal
#[test]
fn test_complete_pipeline() {
    // Create network
    let network_config = NetworkConfig {
        input_size: 32,
        hidden_sizes: vec![16],
        output_size: 3,
        ..Default::default()
    };
    let mut network = SpikingNetwork::new(network_config);

    // Create encoder and decoder
    let encoder = RateEncoder::new(EncoderConfig::default());
    let decoder = TradingDecoder::new(DecoderConfig::default());

    // Create market data
    let market_data = MarketData {
        bid_prices: vec![50000.0, 49999.0, 49998.0, 49997.0, 49996.0, 49995.0, 49994.0, 49993.0],
        ask_prices: vec![50001.0, 50002.0, 50003.0, 50004.0, 50005.0, 50006.0, 50007.0, 50008.0],
        bid_volumes: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ask_volumes: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        timestamp: chrono::Utc::now(),
    };

    // Encode
    let input_spikes = encoder.encode(&market_data);

    // Process
    let output_spikes = network.step(&input_spikes, 1.0);

    // Decode
    let signal = decoder.decode(&output_spikes);

    // Signal should be one of the valid types
    assert!(signal.is_buy() || signal.is_sell() || signal.is_hold());
}

/// Test LIF neuron dynamics
#[test]
fn test_lif_neuron_dynamics() {
    let mut neuron = LIFNeuron::new(0, LIFConfig {
        tau_m: 20.0,
        threshold: 1.0,
        reset: 0.0,
        rest: 0.0,
        refractory_period: 2.0,
    });

    // Apply constant input until spike
    let mut spike_time = None;
    for t in 0..100 {
        if let Some(spike) = neuron.step(0.1, 1.0) {
            spike_time = Some(spike.time);
            break;
        }
    }

    assert!(spike_time.is_some(), "Neuron should have spiked");

    // After spike, neuron should be at reset potential
    assert_eq!(neuron.membrane_potential(), 0.0);
}

/// Test Izhikevich neuron types
#[test]
fn test_izhikevich_neuron_types() {
    let types = vec![
        IzhikevichType::RegularSpiking,
        IzhikevichType::FastSpiking,
        IzhikevichType::Bursting,
    ];

    for neuron_type in types {
        let mut neuron = IzhikevichNeuron::from_type(0, neuron_type);

        // Apply strong input
        let mut spike_count = 0;
        for _ in 0..1000 {
            if neuron.step(15.0, 0.5).is_some() {
                spike_count += 1;
            }
        }

        assert!(spike_count > 0, "{:?} neuron should spike", neuron_type);
    }
}

/// Test synapse STDP learning
#[test]
fn test_synapse_stdp() {
    let mut synapse = Synapse::new(0, 1, SynapseConfig::excitatory(0.5));
    let initial_weight = synapse.weight();

    // Pre before post -> potentiation
    synapse.pre_spike(10.0);
    synapse.post_spike(15.0);
    synapse.apply_stdp(0.1, 0.1, 20.0, 20.0);

    assert!(synapse.weight() > initial_weight, "Weight should increase");
}

/// Test network reset
#[test]
fn test_network_reset() {
    let mut network = SpikingNetwork::new(NetworkConfig::default());

    // Run some steps
    for _ in 0..10 {
        network.step(&[], 1.0);
    }

    let state_before = network.get_state();
    assert!(state_before.current_time > 0.0);

    // Reset
    network.reset();

    let state_after = network.get_state();
    assert_eq!(state_after.current_time, 0.0);
}

/// Test rate encoder output size
#[test]
fn test_rate_encoder_output_size() {
    let encoder = RateEncoder::new(EncoderConfig {
        neurons_per_feature: 4,
        ..Default::default()
    });

    assert_eq!(encoder.output_size(), 4 * 8 * 4);  // 4 features * 8 levels * 4 neurons
}

/// Test temporal encoder timing
#[test]
fn test_temporal_encoder_timing() {
    let encoder = TemporalEncoder::new(EncoderConfig {
        time_window: 10.0,
        ..Default::default()
    });

    let high_data = MarketData {
        bid_prices: vec![90000.0],
        ask_prices: vec![],
        bid_volumes: vec![],
        ask_volumes: vec![],
        timestamp: chrono::Utc::now(),
    };

    let low_data = MarketData {
        bid_prices: vec![10000.0],
        ask_prices: vec![],
        bid_volumes: vec![],
        ask_volumes: vec![],
        timestamp: chrono::Utc::now(),
    };

    let high_spikes = encoder.encode(&high_data);
    let low_spikes = encoder.encode(&low_data);

    assert!(high_spikes[0].time < low_spikes[0].time, "Higher values should spike earlier");
}

/// Test trading signal validation
#[test]
fn test_trading_signal_validation() {
    let strategy = NeuromorphicStrategy::new(StrategyConfig {
        confidence_threshold: 0.6,
        spike_rate_threshold: 100.0,
        ..Default::default()
    });

    let normal_state = NetworkState {
        avg_membrane_potential: 0.5,
        avg_spike_rate: 50.0,
        spike_count: 10,
        active_neurons: 50,
        current_time: 100.0,
    };

    // High confidence should be valid
    let high_conf = TradingSignal::Buy { confidence: 0.8, urgency: 0.5 };
    assert!(strategy.validate_signal(&high_conf, &normal_state));

    // Low confidence should be invalid
    let low_conf = TradingSignal::Buy { confidence: 0.3, urgency: 0.5 };
    assert!(!strategy.validate_signal(&low_conf, &normal_state));
}

/// Test position tracking
#[test]
fn test_position_tracking() {
    let mut position = Position {
        size: 1.0,
        entry_price: 50000.0,
        ..Default::default()
    };

    // Price goes up
    position.update_pnl(51000.0);
    assert_eq!(position.unrealized_pnl, 1000.0);

    // Price goes down
    position.update_pnl(49000.0);
    assert_eq!(position.unrealized_pnl, -1000.0);
}

/// Test layer lateral inhibition
#[test]
fn test_layer_lateral_inhibition() {
    let mut layer = Layer::new(LayerConfig {
        size: 10,
        ..Default::default()
    });

    // Apply strong input to one neuron
    layer.apply_spike(0, 5.0);

    // Step once
    layer.step(1.0);

    // Apply lateral inhibition
    layer.apply_lateral_inhibition(0.5);

    // All neurons should have reduced input except the most active
    let states = layer.neuron_states();
    // The first neuron should have higher potential due to strong input
    assert!(states[0].membrane_potential >= states[1].membrane_potential);
}
