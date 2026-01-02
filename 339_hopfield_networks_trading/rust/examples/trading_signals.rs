//! Trading Signals Example
//!
//! This example demonstrates how to generate trading signals
//! using Hopfield Networks with real Bybit cryptocurrency data.
//!
//! Run with: cargo run --example trading_signals

use hopfield_trading::data::bybit::BybitClient;
use hopfield_trading::data::Candle;
use hopfield_trading::hopfield::modern::ModernHopfield;
use hopfield_trading::hopfield::HopfieldNetwork;
use hopfield_trading::trading::patterns::{
    create_labeled_patterns, PatternEncoder, PatternEncoderConfig, PatternLabel,
};
use hopfield_trading::trading::signals::{Signal, SignalEvaluator, SignalGenerator, SignalGeneratorConfig, SignalType};
use hopfield_trading::trading::MarketRegime;

fn print_signal(signal: &Signal, symbol: &str) {
    let direction = match signal.signal_type {
        SignalType::Long => "🟢 LONG",
        SignalType::Short => "🔴 SHORT",
        SignalType::Exit => "⚪ EXIT",
        SignalType::Hold => "🟡 HOLD",
        SignalType::None => "⚫ NONE",
    };

    println!("\n{} Signal for {}:", direction, symbol);
    println!("  Strength:     {:.2}", signal.strength);
    println!("  Confidence:   {:.1}%", signal.confidence * 100.0);
    println!("  Position Size: {:.2}%", signal.position_size * 100.0);

    if let Some(sl) = signal.stop_loss {
        println!("  Stop Loss:    {:.2}%", sl * 100.0);
    }
    if let Some(tp) = signal.take_profit {
        println!("  Take Profit:  {:.2}%", tp * 100.0);
    }

    if let Some(wr) = signal.metadata.historical_win_rate {
        println!("  Hist Win Rate: {:.1}%", wr * 100.0);
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("===========================================");
    println!("  Hopfield Network Trading Signals");
    println!("  Cryptocurrency Signal Generation");
    println!("===========================================\n");

    // Initialize Bybit client
    let client = BybitClient::new();

    // Configuration
    let symbol = "BTCUSDT";
    let timeframe = "60"; // 1 hour
    let window_size = 12; // 12-hour patterns
    let lookahead = 6; // 6-hour forward returns
    let return_threshold = 0.01; // 1% threshold for labeling

    println!("Configuration:");
    println!("  Symbol:    {}", symbol);
    println!("  Timeframe: {}h", timeframe);
    println!("  Window:    {} periods", window_size);
    println!("  Lookahead: {} periods", lookahead);
    println!("  Threshold: {:.1}%\n", return_threshold * 100.0);

    // Fetch historical data
    println!("Fetching data from Bybit...");
    let candles = client.get_klines(symbol, timeframe, 200).await?;
    println!("Fetched {} candles\n", candles.len());

    // Reverse candles (Bybit returns newest first)
    let candles: Vec<Candle> = candles.into_iter().rev().collect();

    // Configure pattern encoder
    let encoder_config = PatternEncoderConfig {
        window_size,
        include_volume: true,
        include_indicators: false,
        ..Default::default()
    };
    let encoder = PatternEncoder::new(encoder_config);

    // Create labeled patterns
    println!("Creating labeled patterns...");
    let labeled_patterns = create_labeled_patterns(&candles, &encoder, lookahead, return_threshold);
    println!("Created {} labeled patterns\n", labeled_patterns.len());

    if labeled_patterns.is_empty() {
        println!("Not enough data to create patterns!");
        return Ok(());
    }

    // Analyze label distribution
    let mut label_counts = std::collections::HashMap::new();
    for lp in &labeled_patterns {
        *label_counts.entry(lp.label).or_insert(0) += 1;
    }

    println!("Label distribution:");
    for (label, count) in &label_counts {
        let pct = *count as f64 / labeled_patterns.len() as f64 * 100.0;
        println!("  {:?}: {} ({:.1}%)", label, count, pct);
    }
    println!();

    // Split into train and test
    let split_idx = labeled_patterns.len() * 70 / 100;
    let (train, test) = labeled_patterns.split_at(split_idx);

    println!("Training patterns: {}", train.len());
    println!("Test patterns: {}\n", test.len());

    // Get pattern dimension
    let dimension = train[0].pattern.len();

    // Create and train Hopfield Network
    let mut network = ModernHopfield::with_beta(dimension, 5.0);
    let train_patterns: Vec<Vec<f64>> = train.iter().map(|lp| lp.pattern.clone()).collect();
    network.store_patterns(&train_patterns);

    println!("Hopfield Network trained with {} patterns\n", network.pattern_count());

    // Configure signal generator
    let signal_config = SignalGeneratorConfig {
        min_confidence: 0.55,
        min_strength: 0.15,
        default_stop_loss: 0.02,
        default_take_profit: 0.04,
        risk_per_trade: 0.02,
        use_regime_filter: false,
    };
    let mut generator = SignalGenerator::new(signal_config);
    generator.set_labeled_patterns(train.to_vec());

    // ===================================
    // Backtest on Test Set
    // ===================================
    println!("===========================================");
    println!("  Backtesting on Test Set");
    println!("===========================================\n");

    let mut evaluator = SignalEvaluator::new();
    let mut signals_generated = 0;
    let mut correct_direction = 0;

    for (i, test_pattern) in test.iter().enumerate() {
        let signal = generator.generate(&network, &test_pattern.pattern, None);

        // Calculate "actual return" from label
        let actual_return = test_pattern.label.to_value() * return_threshold * 2.0;

        if signal.signal_type.is_actionable() {
            signals_generated += 1;

            // Check if direction matches
            let predicted_dir = if signal.strength > 0.0 { 1.0 } else { -1.0 };
            let actual_dir = if actual_return > 0.0 { 1.0 } else { -1.0 };

            if predicted_dir == actual_dir {
                correct_direction += 1;
            }

            evaluator.evaluate(&signal, actual_return);

            // Show first 5 signals
            if signals_generated <= 5 {
                println!("Test {}:", i + 1);
                println!(
                    "  Signal: {:?} (str: {:.2}, conf: {:.1}%)",
                    signal.signal_type,
                    signal.strength,
                    signal.confidence * 100.0
                );
                println!("  Actual label: {:?}", test_pattern.label);
                println!(
                    "  Direction: {}",
                    if predicted_dir == actual_dir { "✓" } else { "✗" }
                );
                println!();
            }
        }
    }

    // Print backtest results
    println!("--- Backtest Results ---");
    println!("Total test patterns: {}", test.len());
    println!("Actionable signals: {}", signals_generated);
    println!("Direction accuracy: {:.1}%",
        if signals_generated > 0 { correct_direction as f64 / signals_generated as f64 * 100.0 } else { 0.0 });
    println!("Win rate: {:.1}%", evaluator.win_rate() * 100.0);
    println!("Profit factor: {:.2}", evaluator.profit_factor());
    println!("Total P&L: {:.4}", evaluator.total_pnl);
    println!();

    // ===================================
    // Live Signal Generation
    // ===================================
    println!("===========================================");
    println!("  Current Market Signal");
    println!("===========================================");

    // Get latest data for signal
    let latest_candles = client.get_klines(symbol, timeframe, 50).await?;
    let latest_candles: Vec<Candle> = latest_candles.into_iter().rev().collect();

    if latest_candles.len() >= window_size {
        // Encode current market state
        let current_window: Vec<Candle> = latest_candles
            .iter()
            .rev()
            .take(window_size)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        let current_pattern = encoder.encode_window(&current_window);

        // Detect market regime
        let regime = encoder.detect_regime(&current_window);
        println!("\nDetected regime: {:?}", regime);

        // Generate signal
        let signal = generator.generate(&network, &current_pattern, Some(regime));
        print_signal(&signal, symbol);

        // Show matched pattern info
        if let Some(idx) = signal.metadata.matched_pattern_idx {
            if let Some(matched) = train.get(idx) {
                println!("\nMatched pattern info:");
                println!("  Pattern index: {}", idx);
                println!("  Historical label: {:?}", matched.label);
                println!("  Pattern confidence: {:.1}%", matched.confidence * 100.0);
            }
        }

        // Get current price
        if let Some(latest) = latest_candles.last() {
            println!("\nCurrent price: ${:.2}", latest.close);

            if let Some(sl) = signal.stop_loss {
                let sl_price = latest.close * (1.0 - sl * signal.strength.signum());
                println!("Stop loss at:  ${:.2}", sl_price);
            }
            if let Some(tp) = signal.take_profit {
                let tp_price = latest.close * (1.0 + tp * signal.strength.signum());
                println!("Take profit at: ${:.2}", tp_price);
            }
        }
    }

    // ===================================
    // Multi-Symbol Scan
    // ===================================
    println!("\n===========================================");
    println!("  Multi-Symbol Signal Scan");
    println!("===========================================\n");

    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"];

    for sym in &symbols {
        match client.get_klines(sym, timeframe, 50).await {
            Ok(sym_candles) => {
                let sym_candles: Vec<Candle> = sym_candles.into_iter().rev().collect();

                if sym_candles.len() >= window_size {
                    let window: Vec<Candle> = sym_candles
                        .iter()
                        .rev()
                        .take(window_size)
                        .cloned()
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect();

                    let pattern = encoder.encode_window(&window);
                    let signal = generator.generate(&network, &pattern, None);

                    let emoji = match signal.signal_type {
                        SignalType::Long => "🟢",
                        SignalType::Short => "🔴",
                        SignalType::Exit => "⚪",
                        SignalType::Hold => "🟡",
                        SignalType::None => "⚫",
                    };

                    let price = sym_candles.last().map(|c| c.close).unwrap_or(0.0);

                    println!(
                        "{} {} ${:.2} | {:?} | Str: {:.2} | Conf: {:.0}%",
                        emoji,
                        sym,
                        price,
                        signal.signal_type,
                        signal.strength,
                        signal.confidence * 100.0
                    );
                }
            }
            Err(e) => {
                println!("⚠️  {} Error: {}", sym, e);
            }
        }
    }

    println!("\n===========================================");
    println!("  Signal Generation Complete");
    println!("===========================================");
    println!("\n⚠️  DISCLAIMER: This is for educational purposes only.");
    println!("    Do not use for actual trading without proper risk management.");

    Ok(())
}
