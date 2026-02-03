//! Trading example using Concept Bottleneck Model.
//!
//! Demonstrates using the CBM for trading decisions with real-time
//! data fetching from Bybit.

use concept_bottleneck_trading::api::bybit::{generate_sample_klines, BybitClient};
use concept_bottleneck_trading::models::cbm::{ConceptBottleneckTrader, TradingSignal};
use std::collections::HashMap;

fn main() {
    println!("=================================================");
    println!("  Concept Bottleneck Trading Example");
    println!("=================================================\n");

    // Try to fetch real data, fall back to sample data
    let klines = match fetch_real_data() {
        Some(data) => {
            println!("Using real market data from Bybit\n");
            data
        }
        None => {
            println!("Using sample data (API unavailable)\n");
            generate_sample_klines(200)
        }
    };

    println!("Data points: {}", klines.len());
    println!(
        "Price range: {:.2} - {:.2}",
        klines.iter().map(|k| k.low).fold(f64::INFINITY, f64::min),
        klines.iter().map(|k| k.high).fold(f64::NEG_INFINITY, f64::max)
    );
    println!(
        "Current price: {:.2}\n",
        klines.last().map(|k| k.close).unwrap_or(0.0)
    );

    // Create model
    let model = ConceptBottleneckTrader::new();

    // Standard prediction
    println!("=== STANDARD PREDICTION ===");
    if let Some(prediction) = model.predict_with_explanation(&klines) {
        print_prediction(&prediction);
    }

    // Demonstrate intervention
    println!("\n=== INTERVENTION DEMONSTRATION ===");
    demonstrate_intervention(&model, &klines);

    // Simple backtest
    println!("\n=== SIMPLE BACKTEST ===");
    run_simple_backtest(&model, &klines);

    println!("\n=================================================");
    println!("  Trading example complete!");
    println!("=================================================");
}

fn fetch_real_data() -> Option<Vec<concept_bottleneck_trading::api::bybit::Kline>> {
    let client = BybitClient::new();

    match client.fetch_klines_blocking("BTCUSDT", "60", 200) {
        Ok(klines) if !klines.is_empty() => Some(klines),
        _ => None,
    }
}

fn print_prediction(prediction: &concept_bottleneck_trading::models::cbm::Prediction) {
    println!("Signal: {:?}", prediction.signal);
    println!("Confidence: {:.2}%", prediction.confidence * 100.0);
    println!("\nKey concepts:");
    println!("  Trend: {:?} (strength: {:.2})",
             prediction.concepts.trend_direction,
             prediction.concepts.trend_strength);
    println!("  Volatility: {:?} (value: {:.2})",
             prediction.concepts.volatility_regime,
             prediction.concepts.volatility_value);
    println!("  Momentum: {:?} (value: {:.4})",
             prediction.concepts.momentum_signal,
             prediction.concepts.momentum_value);
    println!("  RSI: {:.2}", prediction.concepts.rsi);
}

fn demonstrate_intervention(
    model: &ConceptBottleneckTrader,
    klines: &[concept_bottleneck_trading::api::bybit::Kline],
) {
    // Original prediction
    let original = model.predict_with_explanation(klines).unwrap();
    println!("Original signal: {:?}", original.signal);

    // Intervention: Force bearish trend
    let mut overrides = HashMap::new();
    overrides.insert("trend_direction".to_string(), -1.0);
    overrides.insert("trend_strength".to_string(), 0.8);

    let intervened = model.predict_with_intervention(klines, &overrides).unwrap();
    println!("After forcing bearish trend: {:?}", intervened.signal);

    // Intervention: Force bullish momentum
    overrides.clear();
    overrides.insert("momentum_signal".to_string(), 1.0);
    overrides.insert("momentum_value".to_string(), 0.05);

    let intervened = model.predict_with_intervention(klines, &overrides).unwrap();
    println!("After forcing bullish momentum: {:?}", intervened.signal);

    // Intervention: High volatility warning
    overrides.clear();
    overrides.insert("volatility_regime".to_string(), 1.0);
    overrides.insert("volatility_value".to_string(), 2.5);

    let intervened = model.predict_with_intervention(klines, &overrides).unwrap();
    println!("After high volatility override: {:?}", intervened.signal);
}

fn run_simple_backtest(
    model: &ConceptBottleneckTrader,
    klines: &[concept_bottleneck_trading::api::bybit::Kline],
) {
    let warmup = 50;
    let mut position = 0; // 1 = long, -1 = short, 0 = flat
    let mut entry_price = 0.0;
    let mut total_pnl = 0.0;
    let mut trades = 0;
    let mut wins = 0;

    for i in warmup..klines.len() {
        let window = &klines[..=i];
        let current_price = klines[i].close;

        let (signal, _) = match model.predict(window) {
            Some(result) => result,
            None => continue,
        };

        // Simple trading logic
        match (position, signal) {
            (0, TradingSignal::Buy) => {
                position = 1;
                entry_price = current_price;
            }
            (0, TradingSignal::Sell) => {
                position = -1;
                entry_price = current_price;
            }
            (1, TradingSignal::Sell) | (1, TradingSignal::Hold) => {
                let pnl = (current_price - entry_price) / entry_price;
                total_pnl += pnl;
                trades += 1;
                if pnl > 0.0 {
                    wins += 1;
                }
                position = 0;
            }
            (-1, TradingSignal::Buy) | (-1, TradingSignal::Hold) => {
                let pnl = (entry_price - current_price) / entry_price;
                total_pnl += pnl;
                trades += 1;
                if pnl > 0.0 {
                    wins += 1;
                }
                position = 0;
            }
            _ => {}
        }
    }

    // Close any open position
    if position != 0 {
        let final_price = klines.last().unwrap().close;
        let pnl = if position == 1 {
            (final_price - entry_price) / entry_price
        } else {
            (entry_price - final_price) / entry_price
        };
        total_pnl += pnl;
        trades += 1;
        if pnl > 0.0 {
            wins += 1;
        }
    }

    println!("Total trades: {}", trades);
    println!("Win rate: {:.2}%", if trades > 0 { wins as f64 / trades as f64 * 100.0 } else { 0.0 });
    println!("Total PnL: {:.2}%", total_pnl * 100.0);
}
