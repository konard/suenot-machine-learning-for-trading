use aspect_sentiment_finance::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Aspect-Based Sentiment Analysis - Trading Example ===\n");

    // ── Step 1: Demonstrate Aspect Extraction ─────────────────────
    println!("[1] Aspect Extraction from Financial Text...\n");

    let extractor = AspectExtractor::new();
    let sample_texts = vec![
        "Apple reported record revenue of $120 billion driven by strong iPhone sales.",
        "Profit margins declined as supply chain costs increased significantly.",
        "The company achieved robust growth in market share despite competitive headwinds.",
        "Debt levels remain concerning but the protocol upgrade improved blockchain scalability.",
        "Institutional adoption surged with staking yield reaching record levels.",
    ];

    for text in &sample_texts {
        let aspects = extractor.extract(text);
        println!("  Text: \"{}\"", text);
        for a in &aspects {
            println!("    -> Aspect: {} (keyword: '{}', pos: {})", a.category, a.keyword, a.position);
        }
        println!();
    }

    // ── Step 2: Sentiment Scoring ─────────────────────────────────
    println!("[2] Sentiment Scoring per Aspect...\n");

    let analyzer = AspectSentimentAnalyzer::new();

    let test_texts = vec![
        "Revenue grew significantly this quarter, exceeding all expectations.",
        "Margins compressed sharply due to disappointing cost management.",
        "Revenue did not decline despite the challenging macro environment.",
        "The protocol upgrade beat expectations while regulatory compliance remains concerning.",
    ];

    for text in &test_texts {
        println!("  Text: \"{}\"", text);
        let results = analyzer.analyze(text);
        for r in &results {
            println!(
                "    -> {} [{}]: score={:+.3} ({})",
                r.keyword,
                r.category,
                r.score,
                r.polarity()
            );
        }
        println!();
    }

    // ── Step 3: Batch Analysis & Signal Generation ────────────────
    println!("[3] Batch Analysis & Trading Signal Generation...\n");

    let news_batch = vec![
        "Revenue exceeded expectations with record sales growth.",
        "Earnings per share beat estimates by a wide margin.",
        "Operating margins improved due to strong cost efficiency.",
        "The company reported robust revenue and raised guidance.",
    ];

    let aspect_scores = analyzer.analyze_batch(&news_batch.iter().map(|s| *s).collect::<Vec<&str>>());
    println!("  Aggregated aspect scores from {} articles:", news_batch.len());
    for (category, score) in &aspect_scores {
        println!("    {}: {:+.4}", category, score);
    }

    let mut signal_gen = TradingSignalGenerator::new(0.3);
    let signal = signal_gen.generate_signal(&aspect_scores);
    let action = TradingSignalGenerator::signal_to_action(signal, 0.1, -0.1);
    println!("\n  Trading signal: {:+.4} -> Action: {}", signal, action);

    // ── Step 4: Simulate Time Series of Signals ───────────────────
    println!("\n[4] Simulating Sentiment Signal Time Series...\n");

    let mut signal_gen = TradingSignalGenerator::new(0.2);

    let daily_news: Vec<Vec<&str>> = vec![
        vec!["Revenue grew strongly.", "Margins improved."],
        vec!["Revenue beat expectations.", "Growth accelerated."],
        vec!["Revenue was flat.", "Costs increased modestly."],
        vec!["Revenue declined slightly.", "Margins contracted."],
        vec!["Revenue missed estimates.", "Debt levels rose sharply."],
        vec!["Revenue recovered somewhat.", "Growth resumed modestly."],
    ];

    for (day, texts) in daily_news.iter().enumerate() {
        let scores = analyzer.analyze_batch(texts);
        let signal = signal_gen.generate_signal(&scores);
        let action = TradingSignalGenerator::signal_to_action(signal, 0.1, -0.1);
        println!("  Day {}: signal={:+.4} -> {}", day + 1, signal, action);
    }

    // ── Step 5: Train Aspect Sentiment Classifier ─────────────────
    println!("\n[5] Training Aspect Sentiment Classifier...\n");

    let training_data = generate_training_data(2000);
    let (train, test) = training_data.split_at(1600);

    let mut classifier = AspectSentimentClassifier::new(5, 0.01);
    println!(
        "  Accuracy before training: {:.2}%",
        classifier.accuracy(test) * 100.0
    );

    classifier.train(&train.to_vec(), 100);
    let acc = classifier.accuracy(test);
    println!("  Accuracy after training:  {:.2}%", acc * 100.0);

    // Test with specific feature vectors
    let pos_features = vec![4.0, 0.5, 0.0, 1.2, 0.5];
    let neg_features = vec![0.5, 4.0, 0.0, 1.2, 0.5];
    let neu_features = vec![2.0, 2.0, 0.0, 1.0, 0.5];

    let (pos_label, pos_conf) = classifier.predict(&pos_features);
    let (neg_label, neg_conf) = classifier.predict(&neg_features);
    let (neu_label, neu_conf) = classifier.predict(&neu_features);

    println!("\n  Predictions:");
    println!(
        "    High positive words: {} ({:.1}%), score={:+.4}",
        pos_label,
        pos_conf * 100.0,
        classifier.score(&pos_features)
    );
    println!(
        "    High negative words: {} ({:.1}%), score={:+.4}",
        neg_label,
        neg_conf * 100.0,
        classifier.score(&neg_features)
    );
    println!(
        "    Balanced words:      {} ({:.1}%), score={:+.4}",
        neu_label,
        neu_conf * 100.0,
        classifier.score(&neu_features)
    );

    // ── Step 6: Fetch Bybit Data for Backtesting Context ──────────
    println!("\n[6] Fetching BTCUSDT Data from Bybit V5 API...\n");

    let client = BybitClient::new();

    match client.get_klines("BTCUSDT", "D", 10).await {
        Ok(klines) => {
            println!("  Fetched {} daily kline bars", klines.len());
            for (i, k) in klines.iter().rev().take(5).enumerate() {
                let ret = (k.close - k.open) / k.open * 100.0;
                println!(
                    "  Bar {}: O={:.2} H={:.2} L={:.2} C={:.2} V={:.0} Ret={:+.2}%",
                    i, k.open, k.high, k.low, k.close, k.volume, ret
                );
            }
        }
        Err(e) => {
            println!("  Could not fetch klines: {}. Using synthetic scenario.", e);
            println!("  (This is expected in offline/CI environments.)");
        }
    }

    match client.get_ticker("BTCUSDT").await {
        Ok((price, volume)) => {
            println!("\n  BTCUSDT ticker: price={:.2}, 24h volume={:.0}", price, volume);
        }
        Err(e) => {
            println!("\n  Could not fetch ticker: {}", e);
        }
    }

    // ── Step 7: Full Pipeline Demo ────────────────────────────────
    println!("\n[7] Full Pipeline: Text -> Aspects -> Sentiment -> Signal...\n");

    let crypto_news = vec![
        "Bitcoin protocol upgrade significantly improved throughput and scalability.",
        "Major institutional adoption as three banks integrated Bitcoin payments.",
        "Regulatory concerns eased after SEC approved new compliance framework.",
        "Staking yield increased while supply dynamics shifted due to halving.",
    ];

    println!("  Analyzing {} crypto news articles...", crypto_news.len());
    let crypto_scores = analyzer.analyze_batch(&crypto_news);

    let mut crypto_signal_gen = TradingSignalGenerator::new(0.5);
    let crypto_signal = crypto_signal_gen.generate_signal(&crypto_scores);
    let crypto_action = TradingSignalGenerator::signal_to_action(crypto_signal, 0.1, -0.1);

    println!("\n  Crypto aspect sentiment:");
    for (category, score) in &crypto_scores {
        println!("    {}: {:+.4}", category, score);
    }
    println!(
        "\n  Final signal: {:+.4} -> Action: {}",
        crypto_signal, crypto_action
    );

    println!("\n=== Done ===");
    Ok(())
}
