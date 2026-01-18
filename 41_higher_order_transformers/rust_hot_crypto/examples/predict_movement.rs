//! Example: Price Movement Prediction
//!
//! Demonstrates using the HOT model to predict cryptocurrency price movements.

use anyhow::Result;
use hot_crypto::data::BybitClient;
use hot_crypto::model::{HOTPredictor, MovementClass};
use hot_crypto::strategy::{SignalGenerator, Signal};
use hot_crypto::get_crypto_universe;

#[tokio::main]
async fn main() -> Result<()> {
    println!("HOT Crypto - Price Movement Prediction");
    println!("=======================================\n");

    // Initialize components
    let client = BybitClient::new();
    let predictor = HOTPredictor::default_crypto();
    let signal_gen = SignalGenerator::new();

    // Get universe
    let universe = get_crypto_universe();

    println!("Fetching data for {} cryptocurrencies...\n", universe.len());

    // Fetch data
    let series_list = client.get_multi_klines(&universe, "D", Some(90)).await?;

    println!("Data fetched. Generating predictions...\n");

    // Generate predictions
    let predictions = predictor.predict_batch(&series_list);

    // Display results
    println!("{:-<70}", "");
    println!(" Price Movement Predictions");
    println!("{:-<70}", "");
    println!(
        "{:<12} {:>10} {:>12} {:>8} {:>8} {:>8}",
        "Symbol", "Prediction", "Confidence", "P(Down)", "P(Neut)", "P(Up)"
    );
    println!("{:-<70}", "");

    let mut bullish_count = 0;
    let mut bearish_count = 0;

    for pred in &predictions {
        let emoji = match pred.prediction {
            MovementClass::Up => {
                bullish_count += 1;
                "+"
            },
            MovementClass::Down => {
                bearish_count += 1;
                "-"
            },
            MovementClass::Neutral => "=",
        };

        println!(
            "{:<12} {:>9}{} {:>11.1}% {:>7.1}% {:>7.1}% {:>7.1}%",
            pred.symbol,
            pred.prediction.as_str(),
            emoji,
            pred.confidence * 100.0,
            pred.probabilities[0] * 100.0,
            pred.probabilities[1] * 100.0,
            pred.probabilities[2] * 100.0,
        );
    }

    println!("{:-<70}\n", "");

    // Market sentiment
    println!("Market Sentiment Analysis:");
    println!("  Bullish signals: {}", bullish_count);
    println!("  Bearish signals: {}", bearish_count);
    println!("  Neutral signals: {}", predictions.len() - bullish_count - bearish_count);

    let sentiment = if bullish_count > bearish_count * 2 {
        "STRONGLY BULLISH"
    } else if bullish_count > bearish_count {
        "MILDLY BULLISH"
    } else if bearish_count > bullish_count * 2 {
        "STRONGLY BEARISH"
    } else if bearish_count > bullish_count {
        "MILDLY BEARISH"
    } else {
        "NEUTRAL"
    };

    println!("  Overall sentiment: {}\n", sentiment);

    // Generate trading signals
    println!("{:-<70}", "");
    println!(" Trading Signals");
    println!("{:-<70}", "");

    let signals = signal_gen.generate_batch(&predictions);

    // Sort by confidence
    let mut sorted: Vec<_> = signals.iter()
        .filter(|s| s.signal != Signal::Hold)
        .collect();
    sorted.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    if sorted.is_empty() {
        println!("No actionable signals (all below confidence threshold)");
    } else {
        println!(
            "{:<12} {:>8} {:>12} {:>10}",
            "Symbol", "Signal", "Confidence", "Position"
        );
        println!("{:-<50}", "");

        for signal in sorted.iter().take(5) {
            println!(
                "{:<12} {:>8} {:>11.1}% {:>9.1}%",
                signal.symbol,
                signal.signal.as_str(),
                signal.confidence * 100.0,
                signal.position_size * 100.0,
            );
        }
    }

    println!("{:-<70}\n", "");

    // Top picks
    if let Some(best_buy) = signal_gen.strongest_buy(&signals) {
        println!("Top BUY:  {} (confidence: {:.1}%)",
                 best_buy.symbol, best_buy.confidence * 100.0);
    }

    if let Some(best_sell) = signal_gen.strongest_sell(&signals) {
        println!("Top SELL: {} (confidence: {:.1}%)",
                 best_sell.symbol, best_sell.confidence * 100.0);
    }

    println!("\nNote: These are model predictions, not financial advice!");
    println!("Always do your own research before trading.\n");

    Ok(())
}
