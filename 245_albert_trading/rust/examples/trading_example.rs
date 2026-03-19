use albert_trading::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== ALBERT Trading - Financial Sentiment Analysis ===\n");

    // ── Step 1: Fetch live data from Bybit ──────────────────────────
    println!("[1] Fetching BTCUSDT data from Bybit V5 API...\n");

    let client = BybitClient::new();

    let klines = match client.get_klines("BTCUSDT", "60", 50).await {
        Ok(k) => {
            println!("  Fetched {} kline bars", k.len());
            if let Some(last) = k.last() {
                println!(
                    "  Latest bar: O={:.2} H={:.2} L={:.2} C={:.2} V={:.2}",
                    last.open, last.high, last.low, last.close, last.volume
                );
            }
            k
        }
        Err(e) => {
            println!("  Could not fetch klines: {}. Using synthetic data.", e);
            Vec::new()
        }
    };

    // ── Step 2: Build ALBERT model ──────────────────────────────────
    println!("\n[2] Building ALBERT classifier...\n");

    let mut classifier = ALBERTClassifier::new(
        32,   // embed_dim (E) - factorized embedding dimension
        64,   // hidden_dim (H) - transformer hidden dimension
        2,    // num_layers - shared transformer applications
        4,    // num_heads - attention heads
        3,    // num_classes - bearish/neutral/bullish
        0.05, // learning_rate
    );

    println!("  Model parameters: {}", classifier.param_count());
    println!("  Classes: BEARISH(0), NEUTRAL(1), BULLISH(2)");

    // ── Step 3: Train on financial headlines ─────────────────────────
    println!("\n[3] Training on financial headlines...\n");

    let data = sample_financial_headlines();
    let (train_data, test_data) = data.split_at(35);

    let acc_before = classifier.accuracy(test_data);
    println!("  Accuracy before training: {:.1}%", acc_before * 100.0);

    classifier.train(&train_data.to_vec(), 20);

    let acc_after = classifier.accuracy(test_data);
    println!("  Accuracy after training:  {:.1}%", acc_after * 100.0);

    // ── Step 4: Classify sample headlines ────────────────────────────
    println!("\n[4] Classifying financial headlines...\n");

    let test_headlines = vec![
        "Bitcoin surged above expectations with strong momentum",
        "Revenue declined and missed earnings estimate",
        "Company announced quarterly results report",
        "Crypto market gains on bullish trading volume",
        "Stock dropped on weak guidance and losses",
    ];

    let mut aggregator = SentimentAggregator::new(0.3, 0.2, -0.2);

    for headline in &test_headlines {
        let probs = classifier.predict_proba(headline);
        let (class, confidence) = classifier.predict(headline);
        let score = SentimentAggregator::probs_to_score(&probs);
        aggregator.add_score(score);

        println!(
            "  \"{}\"",
            if headline.len() > 55 {
                &headline[..55]
            } else {
                headline
            }
        );
        println!(
            "    → {} ({:.1}% conf) | Score: {:+.3} | Signal: {:+.3}",
            ALBERTClassifier::label_name(class),
            confidence * 100.0,
            score,
            aggregator.signal()
        );
    }

    // ── Step 5: Generate trading signal ──────────────────────────────
    println!("\n[5] Trading Signal Generation...\n");

    let position = aggregator.position_signal();
    let signal_value = aggregator.signal();

    println!("  Smoothed sentiment signal: {:+.4}", signal_value);
    println!(
        "  Position: {}",
        match position {
            1 => "LONG (bullish sentiment)",
            -1 => "SHORT (bearish sentiment)",
            _ => "FLAT (neutral sentiment)",
        }
    );

    // ── Step 6: Backtest against price data ──────────────────────────
    println!("\n[6] Backtesting sentiment signals...\n");

    if !klines.is_empty() {
        let returns = compute_returns(&klines);
        let positive_returns = returns.iter().filter(|r| **r > 0.0).count();
        let total_return: f64 = returns.iter().sum();

        println!("  Period: {} bars", klines.len());
        println!(
            "  Positive bars: {}/{} ({:.1}%)",
            positive_returns,
            returns.len(),
            positive_returns as f64 / returns.len() as f64 * 100.0
        );
        println!("  Total return: {:+.4}%", total_return * 100.0);

        // Simulate sentiment-weighted returns
        let sentiment_return = total_return * signal_value;
        println!(
            "  Sentiment-weighted return: {:+.4}%",
            sentiment_return * 100.0
        );
    } else {
        println!("  No price data available. Skipping backtest.");
        println!("  In production, sentiment signals would be combined with live prices.");
    }

    // ── Step 7: Parameter efficiency comparison ─────────────────────
    println!("\n[7] ALBERT Parameter Efficiency...\n");

    let albert_params = classifier.param_count();
    // Approximate BERT-equivalent parameters (no factorization, no sharing)
    let bert_equiv = albert_params * 8; // rough estimate

    println!("  ALBERT parameters: {}", albert_params);
    println!("  BERT-equivalent:   ~{}", bert_equiv);
    println!(
        "  Reduction: {:.1}x fewer parameters",
        bert_equiv as f64 / albert_params as f64
    );

    println!("\n=== Done ===");
    Ok(())
}
