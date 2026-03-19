use longformer_nlp_finance::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Longformer NLP Finance - Trading Example ===\n");

    // ── Step 1: Fetch live data from Bybit ──────────────────────────
    println!("[1] Fetching BTCUSDT data from Bybit V5 API...\n");

    let client = BybitClient::new();

    let klines = match client.get_klines("BTCUSDT", "1", 50).await {
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

    let orderbook = match client.get_orderbook("BTCUSDT", 25).await {
        Ok((bids, asks)) => {
            println!(
                "  Order book: {} bid levels, {} ask levels",
                bids.len(),
                asks.len()
            );
            if let (Some(best_bid), Some(best_ask)) = (bids.first(), asks.first()) {
                println!(
                    "  Best bid: {:.2} ({:.4}), Best ask: {:.2} ({:.4})",
                    best_bid.0, best_bid.1, best_ask.0, best_ask.1
                );
            }
            Some((bids, asks))
        }
        Err(e) => {
            println!("  Could not fetch orderbook: {}. Using synthetic data.", e);
            None
        }
    };

    // ── Step 2: Demonstrate Sliding Window Attention ─────────────────
    println!("\n[2] Demonstrating Sliding Window Attention...\n");

    let seq_len = 20;
    let d_model = 8;
    let window_size = 3;

    let embeddings = generate_synthetic_embeddings(seq_len, d_model);
    let encoder = LongformerEncoder::new(window_size, d_model);

    // Set up global attention mask
    let mut global_mask = GlobalAttentionMask::new(seq_len);
    global_mask.set_cls_global(); // [CLS] token
    global_mask.set_positions_global(&[5, 10, 15]); // simulate section headers

    println!(
        "  Sequence length: {}, Model dim: {}, Window: {}",
        seq_len, d_model, window_size
    );
    println!(
        "  Global attention positions: {:?}",
        global_mask.global_indices()
    );

    let encoded = encoder.encode(&embeddings, &global_mask);
    println!("  Encoded {} positions", encoded.len());

    // Show a few encoded vectors
    for i in [0, 5, 10, 15, 19].iter() {
        if *i < encoded.len() {
            let norm: f64 = encoded[*i].iter().map(|x| x * x).sum::<f64>().sqrt();
            println!(
                "  Position {:>2} ({}): L2-norm = {:.4}",
                i,
                if global_mask.is_global(*i) {
                    "global"
                } else {
                    "local "
                },
                norm
            );
        }
    }

    // ── Step 3: Financial Sentiment Analysis ─────────────────────────
    println!("\n[3] Financial Sentiment Lexicon Analysis...\n");

    let lexicon = FinancialSentimentLexicon::new();

    let sample_texts = vec![
        "The company reported robust growth with profit margins that exceeded expectations. Strong momentum continues across all divisions with significant expansion into emerging markets.",
        "Ongoing litigation risk and restructuring costs led to a significant loss. Management warned of recession risk and uncertainty in key markets with potential impairment.",
        "Quarterly revenue was in line with estimates. The board approved the regular dividend payment and noted stable operational metrics across business segments.",
    ];

    let labels = ["Positive", "Negative", "Neutral"];

    for (i, text) in sample_texts.iter().enumerate() {
        let score = lexicon.score(text);
        let (pos, neg) = lexicon.word_counts(text);
        println!("  Text {}: \"{}...\"", i + 1, &text[..60.min(text.len())]);
        println!(
            "  Expected: {}, Score: {:+.4}, Positive words: {}, Negative words: {}",
            labels[i], score, pos, neg
        );
        println!();
    }

    // ── Step 4: Train Sentiment Classifier ───────────────────────────
    println!("[4] Training Sentiment Classifier...\n");

    let training_data = generate_training_data(2000);
    let (train, test) = training_data.split_at(1600);

    let mut classifier = SentimentClassifier::new(5, 0.05);
    println!(
        "  Accuracy before training: {:.2}%",
        classifier.accuracy(test) * 100.0
    );

    classifier.train(&train.to_vec(), 100);
    let acc = classifier.accuracy(test);
    println!("  Accuracy after training:  {:.2}%", acc * 100.0);
    println!("  Weights: {:?}", classifier.weights());
    println!("  Bias: {:.4}", classifier.bias());

    // ── Step 5: Analyze sample text and make prediction ──────────────
    println!("\n[5] Live Sentiment Prediction...\n");

    let extractor = TextFeatureExtractor::new();

    for (i, text) in sample_texts.iter().enumerate() {
        let features = extractor.extract(text);
        let (is_positive, confidence) = classifier.predict(&features);

        println!("  Text {}: \"{}...\"", i + 1, &text[..50.min(text.len())]);
        println!(
            "  Features: lexicon={:.4}, len_norm={:.4}, pos_ratio={:.4}, neg_ratio={:.4}, sent_norm={:.4}",
            features[0], features[1], features[2], features[3], features[4]
        );
        println!(
            "  Prediction: {} with {:.1}% confidence",
            if is_positive { "POSITIVE" } else { "NEGATIVE" },
            confidence * 100.0
        );
        println!();
    }

    // ── Step 6: Combine NLP signal with market data ──────────────────
    println!("[6] Combined NLP + Market Signal...\n");

    let market_trend = if !klines.is_empty() {
        let first_close = klines.first().map(|k| k.close).unwrap_or(50000.0);
        let last_close = klines.last().map(|k| k.close).unwrap_or(50000.0);
        let pct_change = (last_close - first_close) / first_close * 100.0;
        println!("  Market: {:.2} -> {:.2} ({:+.2}%)", first_close, last_close, pct_change);
        if pct_change > 0.0 { "BULLISH" } else { "BEARISH" }
    } else {
        println!("  Market: Using synthetic trend (no live data)");
        "NEUTRAL"
    };

    let spread = if let Some((ref bids, ref asks)) = orderbook {
        if let (Some(bb), Some(ba)) = (bids.first(), asks.first()) {
            let s = ba.0 - bb.0;
            println!("  Spread: {:.2}", s);
            s
        } else {
            1.0
        }
    } else {
        println!("  Spread: using default 1.0 (no live data)");
        1.0
    };

    // Use the first (positive) text as our "current news"
    let news_features = extractor.extract(sample_texts[0]);
    let (sentiment_positive, sentiment_conf) = classifier.predict(&news_features);

    println!(
        "\n  NLP Signal: {} ({:.1}% confidence)",
        if sentiment_positive { "POSITIVE" } else { "NEGATIVE" },
        sentiment_conf * 100.0
    );
    println!("  Market Trend: {}", market_trend);
    println!("  Spread: {:.2}", spread);

    let recommendation = match (sentiment_positive, market_trend) {
        (true, "BULLISH") => "STRONG BUY - Positive sentiment confirmed by market trend",
        (true, "BEARISH") => "HOLD - Positive sentiment but market trend is bearish (potential opportunity)",
        (true, _) => "LEAN BUY - Positive sentiment, neutral market",
        (false, "BEARISH") => "STRONG SELL - Negative sentiment confirmed by market trend",
        (false, "BULLISH") => "HOLD - Negative sentiment but market is bullish (caution advised)",
        (false, _) => "LEAN SELL - Negative sentiment, neutral market",
    };

    println!("\n  Recommendation: {}", recommendation);

    println!("\n=== Done ===");
    Ok(())
}
