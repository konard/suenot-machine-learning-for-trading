use xlnet_finance::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== XLNet Finance - Sentiment Trading Example ===\n");

    // ── Step 1: Train the sentiment scorer ─────────────────────────────
    println!("[1] Training Sentiment Scorer on financial texts...\n");

    let mut scorer = SentimentScorer::new(0.1);
    let training_news = generate_synthetic_news();
    let train_data: Vec<(&str, f64)> = training_news.iter().map(|&(t, l)| (t, l)).collect();

    scorer.train(&train_data, 200);

    // Demonstrate scoring on sample texts
    let sample_texts = vec![
        "Bitcoin surges past $100,000 on massive institutional buying",
        "Market crashes as regulatory crackdown causes panic selling",
        "Trading volume remains stable with neutral market sentiment",
        "Strong earnings beat expectations driving stock rally and growth",
        "Recession fears mount as economic indicators show decline",
    ];

    println!("  Sample sentiment scores:");
    for text in &sample_texts {
        let (label, score) = scorer.classify(text);
        println!("    [{:>7}] ({:+.4}) \"{}\"", label, score, text);
    }

    // ── Step 2: Fetch live data from Bybit ─────────────────────────────
    println!("\n[2] Fetching BTCUSDT data from Bybit V5 API...\n");

    let client = BybitClient::new();

    let klines = match client.get_klines("BTCUSDT", "60", 50).await {
        Ok(k) => {
            println!("  Fetched {} kline bars (1h interval)", k.len());
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

    // ── Step 3: Generate trading signals from sentiment ────────────────
    println!("\n[3] Generating Trading Signals...\n");

    let mut signal_gen = SignalGenerator::new(0.9, 2.0);

    // Simulate a stream of financial news throughout the day
    let live_news = vec![
        "Bitcoin shows strong momentum with bullish breakout above resistance",
        "Crypto exchange reports record trading volume and institutional accumulation",
        "Regulatory concerns emerge but market shows resilience",
        "Ethereum upgrade boosts network performance and innovation",
        "Analysts warn of potential correction after extended rally",
        "Positive earnings from major tech companies fuel risk-on sentiment",
        "Federal Reserve signals cautious approach to rate policy",
        "Crypto market shows strong recovery from morning selloff",
    ];

    println!("  Sentiment signals from news stream:");
    let mut signals = Vec::new();
    for text in &live_news {
        let sentiment = scorer.score(text);
        let signal = signal_gen.generate(sentiment);
        println!(
            "    Sentiment={:+.4} | Signal={:+.4} | Position={:+.4} | Action={}",
            sentiment, signal.signal, signal.position, signal.action
        );
        println!("      \"{}\"", text);
        signals.push(signal);
    }

    // ── Step 4: Backtest the strategy ──────────────────────────────────
    println!("\n[4] Backtesting Sentiment Strategy...\n");

    let mut backtester = Backtester::new(10000.0);

    // Use real prices if available, otherwise synthetic
    let prices = if !klines.is_empty() {
        klines.iter().map(|k| k.close).collect::<Vec<f64>>()
    } else {
        generate_synthetic_prices(50, 95000.0)
    };

    // Generate fresh signals for backtesting
    let mut bt_signal_gen = SignalGenerator::new(0.9, 2.0);
    let bt_news = generate_synthetic_news();
    let bt_train: Vec<(&str, f64)> = bt_news.iter().map(|&(t, l)| (t, l)).collect();

    // Run backtest: cycle through news for each price bar
    for (i, &price) in prices.iter().enumerate() {
        let news_idx = i % bt_train.len();
        let sentiment = scorer.score(bt_train[news_idx].0);
        let signal = bt_signal_gen.generate(sentiment);
        let equity = backtester.step(price, signal.position);

        if i % 10 == 0 || i == prices.len() - 1 {
            println!(
                "  Bar {:>3}: Price={:.2}, Position={:+.4}, Equity={:.2}",
                i, price, signal.position, equity
            );
        }
    }

    // ── Step 5: Performance metrics ────────────────────────────────────
    println!("\n[5] Performance Metrics:\n");

    let metrics = backtester.metrics();
    println!("  Total Return:    {:+.2}%", metrics.total_return * 100.0);
    println!("  Sharpe Ratio:    {:.4}", metrics.sharpe_ratio);
    println!("  Sortino Ratio:   {:.4}", metrics.sortino_ratio);
    println!("  Max Drawdown:    {:.2}%", metrics.max_drawdown * 100.0);
    println!("  Number of Trades: {}", metrics.num_trades);
    println!("  Equity Curve Points: {}", backtester.equity_curve().len());

    // ── Step 6: Batch analysis demonstration ───────────────────────────
    println!("\n[6] Batch Sentiment Analysis...\n");

    let batch_texts = vec![
        "BTCUSDT shows bullish divergence with strong accumulation",
        "ETHUSDT faces resistance at key level amid selling pressure",
        "Market consolidation continues with neutral volume profile",
    ];

    let scores = scorer.score_batch(&batch_texts);
    for (text, score) in batch_texts.iter().zip(scores.iter()) {
        let label = if *score > 0.2 {
            "BULLISH"
        } else if *score < -0.2 {
            "BEARISH"
        } else {
            "NEUTRAL"
        };
        println!("  [{:>7}] ({:+.4}) \"{}\"", label, score, text);
    }

    println!("\n=== Done ===");
    Ok(())
}
