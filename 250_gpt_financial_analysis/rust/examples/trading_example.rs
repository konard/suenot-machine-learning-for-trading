use gpt_financial_analysis::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== GPT Financial Analysis - Trading Example ===\n");

    // ── Step 1: Fetch live data from Bybit ──────────────────────────
    println!("[1] Fetching BTCUSDT data from Bybit V5 API...\n");

    let client = BybitClient::new();

    let klines = match client.get_klines("BTCUSDT", "15", 100).await {
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

    let _orderbook = match client.get_orderbook("BTCUSDT", 25).await {
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

    // Build price series from klines or synthetic data
    let prices: Vec<f64> = if !klines.is_empty() {
        klines.iter().rev().map(|k| k.close).collect()
    } else {
        generate_synthetic_prices(100, 50000.0, 0.01)
    };

    println!("  Price series length: {}", prices.len());
    if let (Some(&first), Some(&last)) = (prices.first(), prices.last()) {
        println!("  Price range: {:.2} -> {:.2}", first, last);
    }

    // ── Step 2: Initialize GPT Financial Analyzer ───────────────────
    println!("\n[2] Initializing GPT Financial Analyzer...\n");

    let analyzer = GptFinancialAnalyzer::new(32, 16, 64, 200);
    println!("  Model configuration:");
    println!("    Embedding dim: 32");
    println!("    Attention dim: 16");
    println!("    FF hidden dim: 64");
    println!("    Max seq length: 200");
    println!("    Vocab size: {}", analyzer.tokenizer().vocab_size());

    // ── Step 3: Analyze financial texts ──────────────────────────────
    println!("\n[3] Analyzing financial texts for sentiment...\n");

    let news_samples = generate_synthetic_news();

    let mut correct = 0;
    let mut total = 0;

    for (text, expected_label) in &news_samples {
        let (result, weights, words) = analyzer.analyze_with_attention(text);

        println!("  Text: \"{}...\"", &text[..text.len().min(60)]);
        println!(
            "  Predicted: {} ({:.1}%)  Expected: {}  {}",
            result.label,
            result.confidence * 100.0,
            expected_label,
            if result.label == *expected_label {
                "✓"
            } else {
                "✗"
            }
        );

        // Show top attention words for the last position
        if let Some(last_weights) = weights.last() {
            let mut word_weights: Vec<(&str, f64)> = words
                .iter()
                .zip(last_weights.iter())
                .map(|(w, &a)| (w.as_str(), a))
                .collect();
            word_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_3: Vec<String> = word_weights
                .iter()
                .take(3)
                .map(|(w, a)| format!("{}({:.2})", w, a))
                .collect();
            println!("  Top attention: {}", top_3.join(", "));
        }
        println!();

        if result.label == *expected_label {
            correct += 1;
        }
        total += 1;
    }

    println!(
        "  Sentiment accuracy: {}/{} ({:.1}%)\n",
        correct,
        total,
        correct as f64 / total as f64 * 100.0
    );

    // ── Step 4: Generate trading signals ─────────────────────────────
    println!("[4] Generating trading signals...\n");

    let signal_gen = TradingSignalGenerator::default_config();
    println!(
        "  Signal weights: sentiment={:.2}, momentum={:.2}, volatility={:.2}",
        0.40, 0.35, 0.25
    );
    println!("  Signal threshold: {:.2}\n", 0.15);

    // Simulate a trading session: analyze news and generate signals
    let sample_news = vec![
        "Strong revenue growth exceeded all expectations with bullish guidance",
        "Market consolidation continues with stable trading volume",
        "The company reported weak earnings with declining margin and negative outlook",
        "Acquisition announcement boosted positive sentiment and buy volume increased",
        "Cautious outlook from management as market conditions remain uncertain",
        "Profit margins expanded significantly beating analyst forecast targets",
        "Revenue decline accelerated with bearish sentiment and sell pressure",
        "Company maintained strong market position with steady growth trajectory",
    ];

    let mut signals = Vec::new();
    let window = prices.len() / sample_news.len();

    for (i, news) in sample_news.iter().enumerate() {
        let sentiment = analyzer.analyze(news);
        let price_start = (i * window).min(prices.len().saturating_sub(20));
        let price_slice = &prices[price_start..prices.len().min(price_start + 50)];

        let signal = signal_gen.generate_signal(&sentiment, price_slice, 10, 20);

        println!(
            "  Signal {}: {} (strength={:.2}) | sentiment={:.2}, momentum={:.2}, vol_adj={:.2}",
            i + 1,
            signal.direction,
            signal.strength,
            signal.sentiment_score,
            signal.momentum_score,
            signal.volatility_score,
        );
        println!("    News: \"{}...\"", &news[..news.len().min(55)]);
        println!("    Sentiment: {} ({:.1}%)\n", sentiment.label, sentiment.confidence * 100.0);

        signals.push(signal.direction.clone());
    }

    // ── Step 5: Backtest the strategy ────────────────────────────────
    println!("[5] Running backtest...\n");

    // Expand signals to match price data (each signal covers a window of bars)
    let mut full_signals = Vec::new();
    for signal in &signals {
        for _ in 0..window.max(1) {
            full_signals.push(signal.clone());
        }
    }
    // Trim or pad to match prices length
    full_signals.truncate(prices.len());
    while full_signals.len() < prices.len() {
        full_signals.push("HOLD".to_string());
    }

    let backtester = Backtester::new(100_000.0, 10_000.0);
    let bt_result = backtester.run(&full_signals, &prices);

    println!("  Backtest Results:");
    println!("  ─────────────────────────────────");
    println!("  Initial capital:  $100,000.00");
    println!("  Final equity:     ${:.2}", bt_result.final_equity);
    println!("  Total return:     {:.2}%", bt_result.total_return * 100.0);
    println!("  Number of trades: {}", bt_result.num_trades);
    println!("  Winning trades:   {}", bt_result.winning_trades);
    println!("  Losing trades:    {}", bt_result.losing_trades);
    if bt_result.num_trades > 0 {
        println!(
            "  Win rate:         {:.1}%",
            bt_result.winning_trades as f64 / bt_result.num_trades as f64 * 100.0
        );
    }
    println!("  Sharpe ratio:     {:.4}", bt_result.sharpe_ratio);
    println!("  Max drawdown:     {:.2}%", bt_result.max_drawdown * 100.0);

    // ── Step 6: Benchmark metrics ────────────────────────────────────
    println!("\n[6] Computing benchmark metrics...\n");

    // Buy and hold benchmark
    let buy_hold_return = if prices.len() >= 2 {
        (prices.last().unwrap() - prices.first().unwrap()) / prices.first().unwrap()
    } else {
        0.0
    };

    println!("  Buy & Hold return: {:.2}%", buy_hold_return * 100.0);
    println!("  Strategy return:   {:.2}%", bt_result.total_return * 100.0);
    println!(
        "  Alpha:             {:.2}%",
        (bt_result.total_return - buy_hold_return) * 100.0
    );

    // Price statistics
    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();
    println!("\n  Price statistics:");
    println!("    Mean return:  {:.4}%", mean(&returns) * 100.0);
    println!("    Volatility:   {:.4}%", std_dev(&returns) * 100.0);

    println!("\n=== Analysis Complete ===");

    Ok(())
}
