use rand::Rng;
use temporal_reasoning_finance::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Temporal Reasoning in Finance - Trading Example ===\n");

    // ── Step 1: Extract temporal expressions from financial text ───
    println!("[1] Extracting temporal expressions from financial text...\n");

    let text = generate_sample_financial_text();
    println!("  Input text (first 200 chars): {}...\n", &text[..200]);

    let extractor = TemporalExpressionExtractor::new();
    let expressions = extractor.extract(text);

    println!("  Found {} temporal expressions:", expressions.len());
    for expr in &expressions {
        println!(
            "    - \"{}\": {:?} (confidence: {:.0}%)",
            expr.text,
            expr.expr_type,
            expr.confidence * 100.0
        );
    }

    // ── Step 2: Build an event timeline ───────────────────────────
    println!("\n[2] Building financial event timeline...\n");

    let mut timeline = EventTimeline::new();

    // Simulated events based on the text
    let base_ts: i64 = 1_700_000_000; // ~Nov 2023
    let events_data = vec![
        ("Q3 Earnings AAPL", EventType::Earnings, "AAPL", base_ts + 86400 * 30),
        ("FOMC September", EventType::FomcDecision, "FED", base_ts + 86400 * 60),
        ("BTC Halving", EventType::Halving, "BTCUSD", base_ts + 86400 * 120),
        ("SOL Token Unlock", EventType::TokenUnlock, "SOLUSDT", base_ts + 86400 * 90),
        ("T+1 Settlement", EventType::RegulatoryDeadline, "MARKET", base_ts + 86400 * 150),
        ("ETH Dencun Upgrade", EventType::ProtocolUpgrade, "ETHUSDT", base_ts + 86400 * 45),
    ];

    for (name, etype, entity, ts) in &events_data {
        timeline.insert(FinancialEvent {
            name: name.to_string(),
            event_type: *etype,
            entity: entity.to_string(),
            timestamp: *ts,
            duration_secs: Some(86400),
            impact_score: 0.8,
        });
    }

    println!("  Timeline ({} events):", timeline.len());
    for event in timeline.events() {
        println!(
            "    t={}: {} ({:?}) - {}",
            event.timestamp, event.name, event.event_type, event.entity
        );
    }

    // Detect gaps
    let gaps = timeline.detect_gaps(86400 * 20);
    println!("\n  Gaps > 20 days: {}", gaps.len());
    for (start, end) in &gaps {
        println!(
            "    Gap: {} to {} ({} days)",
            start,
            end,
            (end - start) / 86400
        );
    }

    // Compute Allen relations
    println!("\n  Allen relations between consecutive events:");
    for i in 0..timeline.len().saturating_sub(1) {
        if let Some(rel) = timeline.relation(i, i + 1) {
            println!(
                "    {} {:?} {}",
                timeline.events()[i].name,
                rel,
                timeline.events()[i + 1].name
            );
        }
    }

    // ── Step 3: Compute temporal features ─────────────────────────
    println!("\n[3] Computing temporal features...\n");

    let feature_extractor = TemporalFeatureExtractor::new(0.1, 86400.0);
    let current_time = base_ts + 86400 * 100;

    let features = feature_extractor.compute_features(
        text,
        timeline.events(),
        current_time,
    );

    println!("  Temporal Density:        {:.4}", features[0]);
    println!("  Average Recency:         {:.4}", features[1]);
    println!("  Forward-Looking Ratio:   {:.4}", features[2]);
    println!("  Temporal Coherence:      {:.4}", features[3]);
    println!("  Event Count:             {:.0}", features[4]);

    // ── Step 4: Fetch live data from Bybit ────────────────────────
    println!("\n[4] Fetching BTCUSDT data from Bybit V5 API...\n");

    let client = BybitClient::new();

    let klines = match client.get_klines("BTCUSDT", "60", 50).await {
        Ok(k) => {
            println!("  Fetched {} kline bars (1h)", k.len());
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

    match client.get_orderbook("BTCUSDT", 10).await {
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
        }
        Err(e) => {
            println!("  Could not fetch orderbook: {}", e);
        }
    }

    // ── Step 5: Train temporal classifier ─────────────────────────
    println!("\n[5] Training Temporal Classifier...\n");

    let training_data = generate_training_data(2000);
    let (train, test) = training_data.split_at(1600);

    let mut classifier = TemporalClassifier::new(5, 0.01);
    println!(
        "  Accuracy before training: {:.2}%",
        classifier.accuracy(test) * 100.0
    );

    classifier.train(&train.to_vec(), 100);
    let acc = classifier.accuracy(test);
    println!("  Accuracy after training:  {:.2}%", acc * 100.0);
    println!("  Weights: {:?}", classifier.weights());
    println!("  Bias: {:.4}", classifier.bias());

    // ── Step 6: Generate temporal trading signal ──────────────────
    println!("\n[6] Generating Temporal Trading Signal...\n");

    let mut strategy = TemporalTradingStrategy::new(0.55, 86400 * 3);
    strategy.train(&train.to_vec(), 100);

    let signal = strategy.generate_signal(text, timeline.events(), current_time);

    match signal {
        Some(sig) => {
            println!(
                "  Signal: {} with {:.1}% strength",
                if sig.direction > 0.0 { "BULLISH" } else { "BEARISH" },
                sig.strength * 100.0
            );
            println!("  Source event: {}", sig.source_event);
        }
        None => {
            println!("  No signal generated (below threshold)");
        }
    }

    // ── Step 7: Backtest with synthetic signals ──────────────────
    println!("\n[7] Backtesting temporal strategy...\n");

    // Generate synthetic signals for backtesting
    let mut rng = rand::thread_rng();
    let synthetic_signals: Vec<TemporalSignal> = if !klines.is_empty() {
        klines
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 5 == 0) // Signal every 5 bars
            .map(|(_, k)| {
                let feat_vec = vec![
                    rng.gen_range(0.0..0.3),
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.0..5.0),
                    rng.gen_range(0.5..1.0),
                    rng.gen_range(1.0..10.0),
                ];
                let (dir, conf) = classifier.predict(&feat_vec);
                TemporalSignal {
                    timestamp: k.timestamp as i64,
                    direction: if dir { 1.0 } else { -1.0 },
                    strength: conf,
                    source_event: "temporal_analysis".to_string(),
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    if !synthetic_signals.is_empty() && !klines.is_empty() {
        let result = strategy.backtest(&klines, &synthetic_signals);
        println!("  Total trades:    {}", result.total_trades);
        println!("  Winning trades:  {}", result.winning_trades);
        println!("  Losing trades:   {}", result.losing_trades);
        println!("  Win rate:        {:.1}%", result.win_rate * 100.0);
        println!("  Total PnL:       {:.4}", result.total_pnl);
        println!("  Max Drawdown:    {:.4}", result.max_drawdown);
        println!("  Sharpe Ratio:    {:.4}", result.sharpe_ratio);
    } else {
        println!("  Skipped backtest (no kline data available)");
    }

    println!("\n=== Done ===");
    Ok(())
}
