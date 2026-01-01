//! Live Trading Example (Simulation Mode)
//!
//! Demonstrates a complete neuromorphic trading pipeline with Bybit data.
//! NOTE: This example runs in simulation mode and does not execute real trades.

use neuromorphic_trading::prelude::*;
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Neuromorphic Trading System ===");
    println!("Mode: SIMULATION (no real trades)\n");

    // Configuration
    let symbol = "BTCUSDT";
    let iterations = 5;

    // Create components
    let bybit_config = BybitConfig {
        testnet: true,
        ..Default::default()
    };
    let client = BybitClient::new(bybit_config);

    let network_config = NetworkConfig {
        input_size: 128,
        hidden_sizes: vec![64, 32],
        output_size: 3,
        ..Default::default()
    };
    let mut network = SpikingNetwork::new(network_config);

    let encoder = RateEncoder::new(EncoderConfig {
        max_rate: 100.0,
        neurons_per_feature: 4,
        price_range: (0.0, 100000.0),
        volume_range: (0.0, 1000.0),
        time_window: 10.0,
    });

    let decoder = TradingDecoder::new(DecoderConfig {
        confidence_threshold: 0.5,
        time_window: 10.0,
        output_size: 3,
    });

    let strategy = NeuromorphicStrategy::new(StrategyConfig {
        confidence_threshold: 0.6,
        max_position_size: 0.01,
        spike_rate_threshold: 100.0,
    });

    println!("Network: {} neurons, {} synapses",
             network.total_neurons(), network.total_synapses());
    println!("Symbol: {}", symbol);
    println!("Iterations: {}\n", iterations);

    // Simulated position tracking
    let mut position = Position::default();
    let mut total_trades = 0;

    println!("--- Starting trading loop ---\n");

    for i in 0..iterations {
        println!("--- Iteration {} ---", i + 1);

        // Fetch market data
        match client.get_orderbook(symbol, 8).await {
            Ok(orderbook) => {
                let mid_price = orderbook.mid_price().unwrap_or(0.0);
                println!("Price: ${:.2}, Spread: {:.2} bps",
                         mid_price, orderbook.spread_bps().unwrap_or(0.0));

                // Convert to MarketData
                let market_data = MarketData {
                    bid_prices: orderbook.bids.iter().take(8).map(|l| l.price).collect(),
                    ask_prices: orderbook.asks.iter().take(8).map(|l| l.price).collect(),
                    bid_volumes: orderbook.bids.iter().take(8).map(|l| l.quantity).collect(),
                    ask_volumes: orderbook.asks.iter().take(8).map(|l| l.quantity).collect(),
                    timestamp: orderbook.timestamp,
                };

                // Encode to spikes
                let input_spikes = encoder.encode(&market_data);

                // Process through SNN
                let output_spikes = network.step(&input_spikes, 1.0);

                // Decode signal
                let signal = decoder.decode(&output_spikes);

                // Get network state
                let network_state = network.get_state();

                // Validate signal
                let is_valid = strategy.validate_signal(&signal, &network_state);

                println!("Signal: {:?}, Valid: {}", signal, is_valid);

                if is_valid {
                    // Check if we should close position
                    if strategy.should_close_position(&signal, &position) {
                        position.update_pnl(mid_price);
                        println!("CLOSE position: P&L = ${:.2}", position.unrealized_pnl);
                        position.realized_pnl += position.unrealized_pnl;
                        position = Position::default();
                        total_trades += 1;
                    }

                    // Check if we should open position
                    match signal {
                        TradingSignal::Buy { confidence, urgency } if position.is_flat() => {
                            let size = strategy.calculate_position_size(&signal, mid_price);
                            println!("OPEN LONG: size={:.4}, confidence={:.2}%, urgency={:.2}",
                                     size, confidence * 100.0, urgency);
                            position = Position {
                                size,
                                entry_price: mid_price,
                                ..Default::default()
                            };
                            total_trades += 1;
                        }
                        TradingSignal::Sell { confidence, urgency } if position.is_flat() => {
                            let size = strategy.calculate_position_size(&signal, mid_price);
                            println!("OPEN SHORT: size={:.4}, confidence={:.2}%, urgency={:.2}",
                                     size.abs(), confidence * 100.0, urgency);
                            position = Position {
                                size,
                                entry_price: mid_price,
                                ..Default::default()
                            };
                            total_trades += 1;
                        }
                        TradingSignal::Hold => {
                            println!("HOLD");
                        }
                        _ => {}
                    }
                }

                // Update position P&L
                if !position.is_flat() {
                    position.update_pnl(mid_price);
                    println!("Position: size={:.4}, entry=${:.2}, unrealized P&L=${:.2}",
                             position.size, position.entry_price, position.unrealized_pnl);
                }
            }
            Err(e) => {
                println!("Error fetching orderbook: {}", e);
            }
        }

        println!();

        // Wait before next iteration
        if i < iterations - 1 {
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }

    println!("--- Summary ---");
    println!("Total trades: {}", total_trades);
    println!("Realized P&L: ${:.2}", position.realized_pnl);
    if !position.is_flat() {
        println!("Open position: size={:.4}, unrealized P&L=${:.2}",
                 position.size, position.unrealized_pnl);
    }

    println!("\n=== Example Complete ===");

    Ok(())
}
