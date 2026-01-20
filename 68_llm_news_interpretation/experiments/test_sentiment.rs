// Quick test script to check sentiment analyzer output
use llm_news_interpretation::SentimentAnalyzer;

fn main() {
    let analyzer = SentimentAnalyzer::new();
    
    let bullish_text = "Bitcoin shows strong bullish momentum with prices surging to new all-time highs";
    let result = analyzer.analyze_rules(bullish_text);
    println!("Bullish text: {}", bullish_text);
    println!("Score: {}, Confidence: {}", result.score, result.confidence);
    println!("is_bullish: {}, is_bearish: {}", result.is_bullish(), result.is_bearish());
    println!();
    
    let bearish_text = "Major cryptocurrency exchange hacked, causing market panic and crash";
    let result = analyzer.analyze_rules(bearish_text);
    println!("Bearish text: {}", bearish_text);
    println!("Score: {}, Confidence: {}", result.score, result.confidence);
    println!("is_bullish: {}, is_bearish: {}", result.is_bullish(), result.is_bearish());
}
