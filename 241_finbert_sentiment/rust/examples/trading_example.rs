use finbert_sentiment::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== FinBERT Sentiment Analysis - Trading Example ===\n");

    // ── Step 1: Analyze financial headlines ───────────────────────────
    println!("[1] Analyzing financial headline sentiment...\n");

    let analyzer = SentimentAnalyzer::new();
    let headlines = sample_financial_headlines();

    let mut aggregator = SentimentAggregator::new(0.9);

    for (i, headline) in headlines.iter().enumerate() {
        let result = analyzer.analyze(headline);
        aggregator.add(result.score, i as u64);
        let label = if result.score > 0.0 {
            "POSITIVE"
        } else if result.score < 0.0 {
            "NEGATIVE"
        } else {
            "NEUTRAL"
        };
        if i < 5 || i >= headlines.len() - 3 {
            println!(
                "  [{:>2}] {:>8} ({:+.3}) | \"{}\"",
                i, label, result.score,
                if headline.len() > 60 { &headline[..60] } else { headline }
            );
        } else if i == 5 {
            println!("  ... ({} more headlines) ...", headlines.len() - 8);
        }
    }

    println!("\n  Aggregated sentiment:");
    println!("    Average:  {:+.4}", aggregator.average());
    println!("    EWMA:     {:+.4}", aggregator.ewma());
    println!("    Momentum: {:+.4}", aggregator.momentum(5));
    println!("    Count:    {}", aggregator.count());

    // ── Step 2: Fetch live crypto data from Bybit ────────────────────
    println!("\n[2] Fetching BTCUSDT data from Bybit V5 API...\n");

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

    // ── Step 3: Train sentiment classifier ───────────────────────────
    println!("\n[3] Training Sentiment Classifier...\n");

    let training_data = generate_training_data(2000);
    let (train, test) = training_data.split_at(1600);

    let mut classifier = SentimentClassifier::new(5, 0.01);
    println!(
        "  Accuracy before training: {:.2}%",
        classifier.accuracy(test) * 100.0
    );

    classifier.train(&train.to_vec(), 100);
    let acc = classifier.accuracy(test);
    println!("  Accuracy after training:  {:.2}%", acc * 100.0);
    println!("  Weights: {:?}", classifier.weights());
    println!("  Bias: {:.4}", classifier.bias());

    // ── Step 4: Backtest sentiment strategy ──────────────────────────
    println!("\n[4] Backtesting sentiment trading strategy...\n");

    let mut strategy = SentimentStrategy::new(0.3, -0.3);

    // Use real prices from Bybit or synthetic
    let prices: Vec<(u64, f64)> = if !klines.is_empty() {
        klines
            .iter()
            .map(|k| (k.timestamp, k.close))
            .collect()
    } else {
        generate_synthetic_prices(50, 50000.0)
    };

    let sentiments = generate_synthetic_sentiments(&prices);

    println!("  {:>4} | {:>10} | {:>8} | {:>6} | {:>8}", "Bar", "Price", "Sentiment", "Signal", "Position");
    println!("  {}", "-".repeat(52));

    for (i, ((_, price), (_, sentiment))) in prices.iter().zip(sentiments.iter()).enumerate() {
        let signal = strategy.update(*sentiment, *price);
        let sig_str = match signal {
            Signal::Buy => "BUY",
            Signal::Sell => "SELL",
            Signal::Hold => "HOLD",
        };
        let pos_str = match strategy.position() as i32 {
            1 => "LONG",
            -1 => "SHORT",
            _ => "FLAT",
        };

        if i < 5 || i >= prices.len() - 3 || signal != Signal::Hold {
            println!(
                "  {:>4} | {:>10.2} | {:>+8.4} | {:>6} | {:>8}",
                i, price, sentiment, sig_str, pos_str
            );
        }
    }

    // ── Step 5: Performance summary ──────────────────────────────────
    println!("\n[5] Strategy Performance Summary...\n");

    println!("  Total trades:      {}", strategy.trades().len());
    println!("  Cumulative return: {:+.4}%", strategy.cumulative_return() * 100.0);
    println!("  Win rate:          {:.1}%", strategy.win_rate() * 100.0);
    println!("  Sharpe ratio:      {:.4}", strategy.sharpe_ratio());

    if !strategy.trades().is_empty() {
        println!("\n  Recent trades:");
        for (i, trade) in strategy.trades().iter().rev().take(5).enumerate() {
            let dir = if trade.direction > 0.0 { "LONG" } else { "SHORT" };
            println!(
                "    [{}] {} | Entry: {:.2} -> Exit: {:.2} | PnL: {:+.4}%",
                i, dir, trade.entry_price, trade.exit_price, trade.pnl * 100.0
            );
        }
    }

    // ── Step 6: Live sentiment prediction ────────────────────────────
    println!("\n[6] Live Sentiment Analysis...\n");

    let test_headlines = [
        "Bitcoin surges past resistance as institutional buyers accumulate",
        "Crypto market faces regulatory risk and potential sell pressure",
        "Ethereum development milestone reached ahead of schedule",
    ];

    for headline in &test_headlines {
        let result = analyzer.analyze(headline);
        let signal = strategy.signal(result.score);
        println!("  \"{}\"", headline);
        println!(
            "    Score: {:+.3} | Pos: {} | Neg: {} | Conf: {:.3} | Signal: {:?}\n",
            result.score, result.positive_count, result.negative_count,
            result.confidence, signal
        );
    }

    println!("=== Done ===");
    Ok(())
}
