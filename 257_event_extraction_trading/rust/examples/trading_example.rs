use event_extraction_trading::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Event Extraction Trading - Trading Example ===\n");

    // ── Step 1: Fetch live data from Bybit ──────────────────────────
    println!("[1] Fetching BTCUSDT data from Bybit V5 API...\n");

    let client = BybitClient::new();

    match client.get_klines("BTCUSDT", "1", 10).await {
        Ok(klines) => {
            println!("  Fetched {} kline bars", klines.len());
            if let Some(last) = klines.last() {
                println!(
                    "  Latest bar: O={:.2} H={:.2} L={:.2} C={:.2} V={:.2}",
                    last.open, last.high, last.low, last.close, last.volume
                );
            }
        }
        Err(e) => {
            println!("  Could not fetch klines: {}. Continuing with demo data.", e);
        }
    };

    match client.get_ticker("BTCUSDT").await {
        Ok(ticker) => {
            println!(
                "  Ticker: {} price={:.2} vol24h={:.2} high={:.2} low={:.2}",
                ticker.symbol, ticker.last_price, ticker.volume_24h,
                ticker.high_24h, ticker.low_24h
            );
        }
        Err(e) => {
            println!("  Could not fetch ticker: {}. Continuing with demo data.", e);
        }
    };

    // ── Step 2: Event Extraction ────────────────────────────────────
    println!("\n[2] Extracting events from sample news headlines...\n");

    let extractor = EventExtractor::new();
    let sample_news = generate_sample_news();

    for headline in &sample_news {
        let events = extractor.extract(headline);
        if events.is_empty() {
            println!("  [NO EVENT] {}", headline);
        } else {
            for event in &events {
                println!(
                    "  [{}] (conf: {:.2}) {}",
                    event.event_type, event.confidence, headline
                );
                if !event.entities.is_empty() {
                    println!("    Entities: {:?}", event.entities);
                }
            }
        }
    }

    // ── Step 3: Sentiment Analysis ──────────────────────────────────
    println!("\n[3] Sentiment scoring of sample headlines...\n");

    let scorer = SentimentScorer::new();

    for headline in &sample_news[..5] {
        let sentiment = scorer.score(headline);
        let label = if sentiment > 0.1 {
            "POSITIVE"
        } else if sentiment < -0.1 {
            "NEGATIVE"
        } else {
            "NEUTRAL"
        };
        println!("  [{:8}] (score: {:+.2}) {}", label, sentiment, headline);
    }

    // ── Step 4: Train Impact Predictor ──────────────────────────────
    println!("\n[4] Training impact prediction model...\n");

    let impact_data = generate_impact_training_data(1000);
    let (train_data, test_data) = impact_data.split_at(800);

    let mut predictor = ImpactPredictor::new(EventType::count() + 2, 0.001);
    let mae_before = predictor.mae(test_data);
    println!("  MAE before training: {:.4}", mae_before);

    predictor.train(&train_data.to_vec(), 100);
    let mae_after = predictor.mae(test_data);
    println!("  MAE after training:  {:.4}", mae_after);
    println!(
        "  Improvement: {:.1}%",
        (1.0 - mae_after / mae_before) * 100.0
    );

    // ── Step 5: Train Event Classifier ──────────────────────────────
    println!("\n[5] Training event classifier...\n");

    let clf_data = generate_classifier_training_data(1000);
    let (clf_train, clf_test) = clf_data.split_at(800);

    let mut classifier = EventClassifier::new(20, 0.01);
    let acc_before = classifier.accuracy(clf_test);
    println!("  Accuracy before training: {:.2}%", acc_before * 100.0);

    classifier.train(&clf_train.to_vec(), 50);
    let acc_after = classifier.accuracy(clf_test);
    println!("  Accuracy after training:  {:.2}%", acc_after * 100.0);

    // ── Step 6: Generate Trading Signals ─────────────────────────────
    println!("\n[6] Generating trading signals from news...\n");

    let generator = TradingSignalGenerator::new(0.5, 0.01);

    for headline in &sample_news {
        let signals = generator.process(headline);
        for signal in &signals {
            let emoji = match signal.action {
                TradingAction::Buy => ">>",
                TradingAction::Sell => "<<",
                TradingAction::Hold => "--",
            };
            println!(
                "  {} {} [{}] impact: {:+.2}% conf: {:.2}",
                emoji, signal.action, signal.event_type,
                signal.predicted_impact, signal.confidence
            );
            println!("    Reason: {}", signal.reason);
        }
    }

    // ── Step 7: Demo with custom events ──────────────────────────────
    println!("\n[7] Custom event extraction demo...\n");

    let custom_headlines = vec![
        "BREAKING: Bybit exchange suffers security breach, $150M in ETH drained from hot wallet",
        "Solana protocol upgrade v2 successfully deployed, transaction throughput increased by 400%",
        "Binance announces massive BNB token burn, destroying 2 million tokens worth $600M",
        "Federal Reserve not raising interest rates, markets rally on dovish guidance",
    ];

    for headline in &custom_headlines {
        println!("  Headline: {}", headline);
        let events = extractor.extract(headline);
        let sentiment = scorer.score(headline);
        println!(
            "    Events: {} detected, Sentiment: {:+.2}",
            events.len(),
            sentiment
        );
        for event in &events {
            println!(
                "    -> Type: {}, Trigger: '{}', Confidence: {:.2}",
                event.event_type, event.trigger_word, event.confidence
            );
            if !event.entities.is_empty() {
                println!("       Entities: {:?}", event.entities);
            }
        }
        let signals = generator.process(headline);
        for signal in &signals {
            println!(
                "    => Signal: {} (impact: {:+.2}%)",
                signal.action, signal.predicted_impact
            );
        }
        println!();
    }

    println!("=== Done ===");
    Ok(())
}
