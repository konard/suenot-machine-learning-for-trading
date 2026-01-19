//! # Crypto Sentiment Analysis Example
//!
//! Demonstrates cryptocurrency-specific sentiment analysis with slang handling.

use llm_sentiment::{
    SentimentAnalyzer, NewsCollector, SentimentAggregator,
    CryptoTerminology,
};
use llm_sentiment::sentiment::crypto::FearGreedLevel;

fn main() {
    println!("=== Crypto Sentiment Analysis Demo ===\n");

    // Initialize components
    let analyzer = SentimentAnalyzer::new();
    let crypto_terms = CryptoTerminology::new();
    let aggregator = SentimentAggregator::new();
    let news_collector = NewsCollector::new();

    // Demonstrate crypto terminology analysis
    println!("=== Crypto Slang Analysis ===\n");

    let crypto_texts = vec![
        "$BTC mooning! Diamond hands holding strong through the dip. WAGMI!",
        "Got rekt on leverage. Paper hands sold at the bottom. NGMI.",
        "DYOR before investing. This is not financial advice. DCA is the way.",
        "Massive rug pull! This was a scam from the start. FUD everywhere!",
        "Whale accumulation detected. Bullish signal! HODL!",
    ];

    for text in &crypto_texts {
        println!("Text: \"{}\"", text);

        let slang_result = crypto_terms.analyze(text);
        let sentiment_result = analyzer.analyze(text);

        println!("  Slang Score: {:.3}", slang_result.score);
        println!("  Terms Found: {:?}", slang_result.found_terms.iter()
            .map(|(t, _, _)| t.as_str())
            .collect::<Vec<_>>());
        println!("  Combined Sentiment: {:.3} ({})",
            sentiment_result.score,
            sentiment_result.level.as_str());
        println!();
    }

    // Analyze mock news
    println!("=== News Sentiment Aggregation ===\n");

    let news = news_collector.get_mock_crypto_news("BTC");
    println!("Analyzing {} news articles...\n", news.len());

    let mut results = Vec::new();
    let mut sources = Vec::new();

    for article in &news {
        let result = analyzer.analyze(&article.full_text());
        println!("[{}] \"{}\"", article.source.name(), article.title);
        println!("     Score: {:.3}, Confidence: {:.0}%",
            result.score, result.confidence * 100.0);

        sources.push(article.source);
        results.push(result);
    }

    // Aggregate sentiments
    println!("\n=== Aggregated Results ===\n");

    let aggregated = aggregator.aggregate(&results, &sources);

    println!("Aggregated Score: {:.3}", aggregated.score);
    println!("Text Count: {}", aggregated.text_count);
    println!("Source Count: {}", aggregated.source_count);
    println!("Average Confidence: {:.0}%", aggregated.confidence * 100.0);
    println!("Score Std Dev: {:.3}", aggregated.std_dev);

    println!("\nSource Breakdown:");
    for (source, (score, count)) in &aggregated.source_scores {
        println!("  {}: {:.3} ({} articles)", source, score, count);
    }

    // Fear & Greed comparison
    println!("\n=== Fear & Greed Index Comparison ===\n");

    let fg_score = FearGreedLevel::sentiment_to_fg(aggregated.score);
    let fg_level = FearGreedLevel::from_score(fg_score);

    println!("LLM Sentiment Score: {:.3}", aggregated.score);
    println!("Normalized to Fear & Greed (0-100): {:.0}", fg_score);
    println!("Fear & Greed Level: {}", fg_level.as_str());

    // Trading implication
    println!("\n=== Trading Implication ===\n");

    if aggregated.score > 0.3 && aggregated.confidence > 0.6 {
        println!("Signal: BULLISH - Consider LONG position");
    } else if aggregated.score < -0.3 && aggregated.confidence > 0.6 {
        println!("Signal: BEARISH - Consider SHORT or reduce exposure");
    } else {
        println!("Signal: NEUTRAL - No clear directional signal");
    }

    println!("\n=== Crypto Sentiment Analysis Complete ===");
}
