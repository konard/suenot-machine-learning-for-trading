use roberta_trading::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== RoBERTa Trading - Sentiment Analysis Example ===\n");

    // ── Step 1: Initialize components ────────────────────────────────
    println!("[1] Initializing text feature extractor and classifier...\n");

    let extractor = TextFeatureExtractor::new();
    let mut classifier = SentimentClassifier::new(extractor.num_features(), 0.05);

    // ── Step 2: Train on sample financial headlines ───────────────────
    println!("[2] Training sentiment classifier on financial headlines...\n");

    let training_data = generate_training_data(&extractor);
    println!("  Training samples: {}", training_data.len());

    // Also add synthetic data for more robust training
    let synthetic_data = generate_synthetic_training_data(500, extractor.num_features());
    let mut all_data = training_data.clone();
    all_data.extend(synthetic_data);
    println!("  Total training samples (with synthetic): {}", all_data.len());

    let acc_before = classifier.accuracy(&training_data);
    println!("  Accuracy before training: {:.1}%", acc_before * 100.0);

    classifier.train(&all_data, 100);

    let acc_after = classifier.accuracy(&training_data);
    println!("  Accuracy after training:  {:.1}%", acc_after * 100.0);

    // ── Step 3: Analyze sample headlines ──────────────────────────────
    println!("\n[3] Analyzing sample financial headlines...\n");

    let test_headlines = vec![
        "Bitcoin surges past $100K as institutional buying intensifies",
        "Crypto exchange reports record profits and user growth",
        "Major security breach exposes millions of user accounts",
        "Regulatory crackdown sends market into sharp decline",
        "Federal Reserve announces scheduled policy review meeting",
        "Blockchain technology conference scheduled for next quarter",
        "DeFi protocol launches innovative yield farming strategy",
        "Stablecoin loses peg amid widespread market fears",
    ];

    let mut aggregator = SentimentAggregator::new(10);

    for headline in &test_headlines {
        let features = extractor.extract(headline);
        let pred = classifier.predict(&features);
        aggregator.add_score(pred.score());

        println!(
            "  {:60} -> {} ({:.1}%) score={:+.3}",
            &headline[..headline.len().min(60)],
            pred.sentiment,
            pred.confidence * 100.0,
            pred.score()
        );
    }

    // ── Step 4: Aggregate sentiment and generate signal ──────────────
    println!("\n[4] Aggregating sentiment...\n");

    println!("  Average sentiment:  {:+.4}", aggregator.average());
    println!("  Sentiment trend:    {:+.4}", aggregator.trend());
    println!("  Sentiment volatility: {:.4}", aggregator.volatility());

    let signal_gen = TradingSignalGenerator::new(0.1, -0.1);
    let signal = signal_gen.generate(&aggregator);
    let (trend_signal, strength) = signal_gen.generate_with_trend(&aggregator);

    println!("  Simple signal: {}", signal);
    println!(
        "  Trend-adjusted signal: {} (strength: {:.4})",
        trend_signal, strength
    );

    // ── Step 5: Fetch live market data from Bybit ────────────────────
    println!("\n[5] Fetching BTCUSDT data from Bybit V5 API...\n");

    let client = BybitClient::new();

    match client.get_ticker("BTCUSDT").await {
        Ok(ticker) => {
            let price: f64 = ticker.last_price.parse().unwrap_or(0.0);
            let change: f64 = ticker.price_24h_pcnt.parse().unwrap_or(0.0);
            println!("  BTCUSDT price: ${:.2}", price);
            println!("  24h change: {:+.2}%", change * 100.0);
            println!("  24h volume: {}", ticker.volume_24h);
        }
        Err(e) => {
            println!("  Could not fetch ticker: {}. Continuing with analysis.", e);
        }
    }

    let _klines = match client.get_klines("BTCUSDT", "60", 24).await {
        Ok(k) => {
            println!("  Fetched {} hourly klines", k.len());
            if let (Some(first), Some(last)) = (k.first(), k.last()) {
                let ret = (last.close - first.open) / first.open * 100.0;
                println!(
                    "  24h price range: {:.2} -> {:.2} ({:+.2}%)",
                    first.open, last.close, ret
                );
            }
            k
        }
        Err(e) => {
            println!("  Could not fetch klines: {}. Using simulated data.", e);
            Vec::new()
        }
    };

    // ── Step 6: Backtest sentiment signals against price data ────────
    println!("\n[6] Simulated backtest: sentiment vs price movement...\n");

    // Simulate a stream of headlines with timestamps
    let backtest_headlines = vec![
        ("Bitcoin rally gains momentum with strong buying", 0.02),
        ("Bullish sentiment drives crypto higher", 0.015),
        ("Minor pullback as traders take profits", -0.005),
        ("Concerns about regulation cause brief dip", -0.01),
        ("Recovery underway as buyers step back in", 0.008),
        ("Exchange announces new trading features, boost expected", 0.012),
        ("Market crashes on unexpected news", -0.03),
        ("Strong support level holds, bounce expected", 0.01),
    ];

    let mut backtest_agg = SentimentAggregator::new(5);
    let mut correct_signals = 0;
    let mut total_signals = 0;

    for (headline, actual_return) in &backtest_headlines {
        let features = extractor.extract(headline);
        let pred = classifier.predict(&features);
        backtest_agg.add_score(pred.score());

        let signal = signal_gen.generate(&backtest_agg);
        let signal_correct = match signal {
            Signal::Buy => *actual_return > 0.0,
            Signal::Sell => *actual_return < 0.0,
            Signal::Hold => true, // hold is always "correct" for simplicity
        };

        if signal != Signal::Hold {
            total_signals += 1;
            if signal_correct {
                correct_signals += 1;
            }
        }

        println!(
            "  {} | Sentiment: {:+.3} | Signal: {:4} | Return: {:+.3} | {}",
            &headline[..headline.len().min(50)],
            pred.score(),
            format!("{}", signal),
            actual_return,
            if signal == Signal::Hold {
                "---"
            } else if signal_correct {
                "HIT"
            } else {
                "MISS"
            }
        );
    }

    if total_signals > 0 {
        println!(
            "\n  Signal accuracy: {}/{} ({:.1}%)",
            correct_signals,
            total_signals,
            correct_signals as f64 / total_signals as f64 * 100.0
        );
    }

    println!("\n=== Done ===");
    Ok(())
}
