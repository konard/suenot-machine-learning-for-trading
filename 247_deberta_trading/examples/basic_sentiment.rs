//! Basic sentiment analysis example.
//!
//! Demonstrates how to use the DeBERTa sentiment model
//! to classify financial headlines.

use deberta_trading::SentimentModel;

fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let model = SentimentModel::new();

    // Sample financial headlines
    let headlines = vec![
        "Apple revenue beats Q3 expectations by 15%",
        "Fed signals potential rate hike in September",
        "Tesla recalls 100,000 vehicles over safety issue",
        "Bitcoin surges past $100K on institutional demand",
        "Markets trade sideways ahead of earnings season",
        "NVIDIA reports record data center revenue growth",
        "Crypto exchange faces regulatory investigation",
    ];

    println!("=== DeBERTa Financial Sentiment Analysis ===\n");

    for headline in &headlines {
        let result = model.predict(headline);
        println!(
            "[{:>8}] (score: {:.4}) {}",
            result.label.to_string().to_uppercase(),
            result.score,
            headline
        );
    }

    // Generate a composite trading signal
    println!("\n=== Composite Trading Signal ===\n");

    let btc_headlines: Vec<&str> = vec![
        "Bitcoin surges on ETF approval news",
        "Institutional investors boost Bitcoin holdings",
        "Minor regulatory concerns for crypto market",
    ];

    let signal = model.get_trading_signal(&btc_headlines, 0.3);
    println!(
        "Signal: {:.4} (confidence: {:.4})",
        signal.signal, signal.confidence
    );
    println!(
        "Positive: {}, Negative: {}, Neutral: {}",
        signal.n_positive, signal.n_negative, signal.n_neutral
    );

    let action = if signal.signal > 0.0 {
        "BUY"
    } else if signal.signal < 0.0 {
        "SELL"
    } else {
        "HOLD"
    };
    println!("Recommended action: {}", action);
}
