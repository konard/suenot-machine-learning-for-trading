//! # Trading Signals Example
//!
//! Demonstrates converting sentiment to actionable trading signals.

use llm_sentiment::{SentimentAnalyzer, SignalGenerator, NewsCollector};

fn main() {
    println!("=== Trading Signal Generation Demo ===\n");

    // Initialize components
    let analyzer = SentimentAnalyzer::new();
    let signal_gen = SignalGenerator::new()
        .with_thresholds(0.3, -0.3, 0.6)
        .with_min_confidence(0.5)
        .with_risk_params(0.5, 0.05, 0.10);

    // Example texts for different market conditions
    let bullish_texts = vec![
        "Bitcoin surges past resistance! Institutional FOMO is real.",
        "Massive whale accumulation detected. Bullish breakout imminent!",
        "ETF approval news sends crypto markets soaring!",
    ];

    let bearish_texts = vec![
        "Market crash warning! Bears in control as prices plunge.",
        "Regulatory crackdown fears grow. Massive sell-off continues.",
        "Exchange hack reported. FUD spreading across social media.",
    ];

    let neutral_texts = vec![
        "Bitcoin consolidates around support level.",
        "Trading volume remains low during weekend.",
        "Market awaits Fed decision on interest rates.",
    ];

    // Analyze bullish scenario
    println!("=== Bullish Market Scenario ===\n");
    let bullish_results: Vec<_> = bullish_texts.iter()
        .map(|t| analyzer.analyze(t))
        .collect();

    for (text, result) in bullish_texts.iter().zip(&bullish_results) {
        let signal = signal_gen.generate(result);
        println!("Text: \"{}\"", text);
        println!("  Signal: {} | Strength: {:?}",
            signal.signal.as_str(), signal.strength);
        println!("  Position Size: {:.1}% | SL: {:.1}% | TP: {:.1}%",
            signal.position_size * 100.0,
            signal.stop_loss * 100.0,
            signal.take_profit * 100.0);
        println!();
    }

    // Aggregated bullish signal
    let agg_bullish = signal_gen.generate_aggregated(&bullish_results);
    println!("Aggregated Bullish Signal: {}", agg_bullish.signal.as_str());
    println!("Reason: {}\n", agg_bullish.reason);

    // Analyze bearish scenario
    println!("=== Bearish Market Scenario ===\n");
    let bearish_results: Vec<_> = bearish_texts.iter()
        .map(|t| analyzer.analyze(t))
        .collect();

    for (text, result) in bearish_texts.iter().zip(&bearish_results) {
        let signal = signal_gen.generate(result);
        println!("Text: \"{}\"", text);
        println!("  Signal: {} | Strength: {:?}",
            signal.signal.as_str(), signal.strength);
        println!("  Position Size: {:.1}% | SL: {:.1}% | TP: {:.1}%",
            signal.position_size * 100.0,
            signal.stop_loss * 100.0,
            signal.take_profit * 100.0);
        println!();
    }

    // Aggregated bearish signal
    let agg_bearish = signal_gen.generate_aggregated(&bearish_results);
    println!("Aggregated Bearish Signal: {}", agg_bearish.signal.as_str());
    println!("Reason: {}\n", agg_bearish.reason);

    // Analyze neutral scenario
    println!("=== Neutral Market Scenario ===\n");
    let neutral_results: Vec<_> = neutral_texts.iter()
        .map(|t| analyzer.analyze(t))
        .collect();

    for (text, result) in neutral_texts.iter().zip(&neutral_results) {
        let signal = signal_gen.generate(result);
        println!("Text: \"{}\"", text);
        println!("  Signal: {} | Confidence: {:.0}%",
            signal.signal.as_str(), signal.confidence * 100.0);
        println!();
    }

    // Aggregated neutral signal
    let agg_neutral = signal_gen.generate_aggregated(&neutral_results);
    println!("Aggregated Neutral Signal: {}", agg_neutral.signal.as_str());
    println!("Reason: {}\n", agg_neutral.reason);

    // Mixed scenario with real news
    println!("=== Real-World Mixed Scenario ===\n");

    let news_collector = NewsCollector::new();
    let btc_news = news_collector.get_mock_crypto_news("BTC");

    let news_results: Vec<_> = btc_news.iter()
        .map(|n| analyzer.analyze(&n.full_text()))
        .collect();

    println!("Analyzing {} BTC news articles:\n", btc_news.len());

    for (news, result) in btc_news.iter().zip(&news_results) {
        let signal = signal_gen.generate(result);
        println!("[{}] {}", news.source.name().to_uppercase(), news.title);
        println!("    → {} (score: {:.2}, confidence: {:.0}%)",
            signal.signal.as_str(),
            result.score,
            result.confidence * 100.0);
    }

    let final_signal = signal_gen.generate_aggregated(&news_results);
    println!("\n=== Final Trading Decision ===\n");
    println!("Signal: {}", final_signal.signal.as_str());
    println!("Strength: {:?}", final_signal.strength);
    println!("Sentiment Score: {:.3}", final_signal.sentiment_score);
    println!("Confidence: {:.0}%", final_signal.confidence * 100.0);
    println!("Suggested Position: {:.1}%", final_signal.position_size * 100.0);
    println!("Stop Loss: {:.1}%", final_signal.stop_loss * 100.0);
    println!("Take Profit: {:.1}%", final_signal.take_profit * 100.0);
    println!("\nReason: {}", final_signal.reason);

    println!("\n=== Signal Generation Complete ===");
}
