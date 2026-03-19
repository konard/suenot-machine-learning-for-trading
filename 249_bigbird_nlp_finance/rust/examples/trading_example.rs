use bigbird_nlp_finance::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== BigBird Financial NLP - Trading Example ===\n");

    // ── Step 1: Demonstrate BigBird attention patterns ────────────
    println!("[1] BigBird Sparse Attention Patterns\n");

    let config = BigBirdConfig::new(4096, 3, 3, 2, 64, 4);
    let attention = BigBirdAttention::new(config);

    // Show sparsity for different sequence lengths
    for &seq_len in &[32, 64, 128, 256] {
        let ratio = attention.sparsity_ratio(seq_len);
        let full_ops = seq_len * seq_len;
        let sparse_ops = (ratio * full_ops as f64) as usize;
        println!(
            "  Seq len={:>4}: sparsity={:.1}% | {}/{} attention pairs (vs full)",
            seq_len,
            ratio * 100.0,
            sparse_ops,
            full_ops
        );
    }

    // ── Step 2: Process sample financial documents ────────────────
    println!("\n[2] Processing Financial Documents\n");

    let processor = DocumentProcessor::new(4096);
    let texts = sample_financial_texts();

    for (text, expected_sentiment) in &texts {
        let doc = processor.process(text);
        println!("  Document: \"{}...\"", &text[..60.min(text.len())]);
        println!("    Tokens: {}", doc.tokens.len());
        println!("    Entities: {:?}", doc.financial_entities.len());
        println!(
            "    Entity types: {:?}",
            doc.financial_entities
                .iter()
                .map(|(_, name, etype)| format!("{}:{}", name, etype))
                .collect::<Vec<_>>()
        );
        println!("    Expected sentiment: {}", expected_sentiment);
        println!();
    }

    // ── Step 3: Extract features and classify ────────────────────
    println!("[3] Sentiment Classification\n");

    // Generate training data and train classifier
    let training_data = generate_sentiment_training_data(2000);
    let (train, test) = training_data.split_at(1600);

    let mut classifier = SentimentClassifier::new(6, 0.01);
    let acc_before = classifier.accuracy(test);
    println!("  Accuracy before training: {:.1}%", acc_before * 100.0);

    classifier.train(&train.to_vec(), 100);
    let acc_after = classifier.accuracy(test);
    println!("  Accuracy after training:  {:.1}%", acc_after * 100.0);

    // Classify sample documents
    println!("\n  Classifying sample documents:");
    for (text, expected) in &texts {
        let doc = processor.process(text);
        let features = extract_sentiment_features(&doc);
        let (class, confidence) = classifier.predict(&features);
        let label = SentimentClassifier::label(class);
        println!(
            "    Expected: {:>8} | Predicted: {:>8} ({:.1}%) | \"{}...\"",
            expected,
            label,
            confidence * 100.0,
            &text[..50.min(text.len())]
        );
    }

    // ── Step 4: BigBird forward pass demonstration ───────────────
    println!("\n[4] BigBird Attention Forward Pass\n");

    let small_config = BigBirdConfig::new(64, 2, 2, 1, 8, 1);
    let small_attn = BigBirdAttention::new(small_config);

    let seq_len = 16;
    let d = 8;
    let mut rng = rand::thread_rng();

    let queries = ndarray::Array2::from_shape_fn((seq_len, d), |_| {
        use rand::Rng;
        rng.gen_range(-1.0..1.0)
    });
    let keys = ndarray::Array2::from_shape_fn((seq_len, d), |_| {
        use rand::Rng;
        rng.gen_range(-1.0..1.0)
    });
    let values = ndarray::Array2::from_shape_fn((seq_len, d), |_| {
        use rand::Rng;
        rng.gen_range(-1.0..1.0)
    });

    let output = small_attn.forward(&queries, &keys, &values);
    println!(
        "  Input:  Q[{}x{}], K[{}x{}], V[{}x{}]",
        seq_len, d, seq_len, d, seq_len, d
    );
    println!("  Output: [{}x{}]", output.nrows(), output.ncols());
    println!(
        "  Output[0] = [{:.4}, {:.4}, {:.4}, ...]",
        output[[0, 0]],
        output[[0, 1]],
        output[[0, 2]]
    );

    // ── Step 5: Fetch live data from Bybit ───────────────────────
    println!("\n[5] Fetching BTCUSDT Data from Bybit V5 API\n");

    let client = BybitClient::new();

    let klines = match client.get_klines("BTCUSDT", "15", 20).await {
        Ok(k) => {
            println!("  Fetched {} kline bars", k.len());
            if let Some(last) = k.last() {
                println!(
                    "  Latest: O={:.2} H={:.2} L={:.2} C={:.2} V={:.2}",
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

    let _orderbook = match client.get_orderbook("BTCUSDT", 10).await {
        Ok((bids, asks)) => {
            println!(
                "  Order book: {} bid levels, {} ask levels",
                bids.len(),
                asks.len()
            );
            if let (Some(bb), Some(ba)) = (bids.first(), asks.first()) {
                println!("  Best bid: {:.2}, Best ask: {:.2}", bb.0, ba.0);
            }
            Some((bids, asks))
        }
        Err(e) => {
            println!("  Could not fetch orderbook: {}", e);
            None
        }
    };

    // ── Step 6: Combine NLP + Market Data for trading signal ─────
    println!("\n[6] Combined NLP + Market Data Trading Signal\n");

    // Simulate an earnings report sentiment
    let report = "The company reported strong revenue growth of 25% with earnings \
                  exceeding analyst expectations. Management expressed optimistic \
                  outlook for next quarter with improved margins.";
    let doc = processor.process(report);
    let features = extract_sentiment_features(&doc);
    let (sentiment_class, sentiment_conf) = classifier.predict(&features);
    let sentiment_label = SentimentClassifier::label(sentiment_class);

    println!("  NLP Signal:");
    println!("    Document sentiment: {} ({:.1}%)", sentiment_label, sentiment_conf * 100.0);
    println!(
        "    Features: pos={:.3}, neg={:.3}, ratio={:.2}, entities={}",
        features[0], features[1], features[2], doc.financial_entities.len()
    );

    // Market signal from price action
    let price_trend = if !klines.is_empty() && klines.len() >= 2 {
        let last = &klines[klines.len() - 1];
        let prev = &klines[klines.len() - 2];
        if last.close > prev.close {
            "bullish"
        } else {
            "bearish"
        }
    } else {
        "neutral (no data)"
    };

    println!("  Market Signal:");
    println!("    Price trend: {}", price_trend);

    // Combined signal
    let nlp_score = match sentiment_class {
        0 => 1.0,  // positive
        1 => 0.0,  // neutral
        2 => -1.0, // negative
        _ => 0.0,
    };
    let market_score = if price_trend == "bullish" {
        1.0
    } else if price_trend == "bearish" {
        -1.0
    } else {
        0.0
    };
    let combined = 0.6 * nlp_score + 0.4 * market_score;

    let action = if combined > 0.3 {
        "BUY"
    } else if combined < -0.3 {
        "SELL"
    } else {
        "HOLD"
    };

    println!("\n  Combined Signal:");
    println!("    NLP score:    {:+.2} (weight 60%)", nlp_score);
    println!("    Market score: {:+.2} (weight 40%)", market_score);
    println!("    Combined:     {:+.2}", combined);
    println!("    Action:       {}", action);

    println!("\n=== Done ===");
    Ok(())
}
