//! Backtesting example for news-based trading strategy
//!
//! This example demonstrates how to:
//! - Load historical news and price data
//! - Run a news-based trading strategy
//! - Calculate performance metrics
//!
//! Run with: cargo run --example backtest

use llm_news_interpretation::{
    NewsItem, NewsSource, SentimentAnalyzer, EventClassifier,
    NewsStrategy, StrategyConfig, Metrics, MetricsRecorder,
    OHLCV,
};
use chrono::{Duration, Utc};

fn main() {
    println!("=== LLM News Interpretation - Backtesting Example ===\n");

    // Generate simulated historical data
    let (historical_news, historical_prices) = generate_simulated_data();

    println!("Loaded {} news items and {} price candles\n",
             historical_news.len(), historical_prices.len());

    // Initialize components
    let sentiment_analyzer = SentimentAnalyzer::new();
    let event_classifier = EventClassifier::new();
    let metrics_recorder = MetricsRecorder::new()
        .with_label("strategy", "news_based")
        .with_label("symbol", "BTCUSDT");

    let config = StrategyConfig {
        symbols: vec!["BTCUSDT".to_string()],
        min_signal_strength: 0.25,
        min_confidence: 0.5,
        max_position_size: 0.1,
        trade_cooldown_secs: 3600, // 1 hour
        allow_shorts: true,
        max_positions: 1,
    };

    let mut strategy = NewsStrategy::new(config);

    // Simulated portfolio
    let initial_capital = 10000.0;
    let mut capital = initial_capital;
    let mut position_size = 0.0;
    let mut position_entry_price = 0.0;
    let mut trades: Vec<Trade> = Vec::new();

    println!("Starting backtest with ${:.2} initial capital\n", initial_capital);
    println!("{:-<70}", "");
    println!("{:<30} {:>12} {:>12} {:>12}", "Event", "Signal", "Price", "Capital");
    println!("{:-<70}", "");

    // Run backtest
    for (i, news) in historical_news.iter().enumerate() {
        let price_idx = i.min(historical_prices.len() - 1);
        let current_price = historical_prices[price_idx].close;

        // Analyze news
        let sentiment = sentiment_analyzer.analyze_rules(&format!("{} {}", news.title, news.content));
        let event = event_classifier.classify(&news.content);

        // Record for metrics
        metrics_recorder.record_news_processed(1);

        // Generate signal
        let signal_strength = sentiment.score.abs() * sentiment.confidence;
        let is_bullish = sentiment.score > 0.0;

        // Simple trading logic for backtesting
        if signal_strength > 0.25 {
            metrics_recorder.record_signal(sentiment.confidence);

            if position_size == 0.0 {
                // Open new position
                if is_bullish {
                    // Buy
                    let buy_size = capital * 0.1; // Use 10% of capital
                    position_size = buy_size / current_price;
                    position_entry_price = current_price;
                    capital -= buy_size;

                    println!("{:<30} {:>12} {:>12.2} {:>12.2}",
                             truncate(&news.title, 28),
                             format!("BUY {:.4}", position_size),
                             current_price,
                             capital + position_size * current_price);
                } else {
                    // Short (simplified)
                    let short_size = capital * 0.1;
                    position_size = -(short_size / current_price);
                    position_entry_price = current_price;

                    println!("{:<30} {:>12} {:>12.2} {:>12.2}",
                             truncate(&news.title, 28),
                             format!("SHORT {:.4}", position_size.abs()),
                             current_price,
                             capital);
                }
            } else {
                // Check if we should close position
                let should_close = (position_size > 0.0 && !is_bullish) ||
                                   (position_size < 0.0 && is_bullish);

                if should_close && signal_strength > 0.4 {
                    // Close position
                    let pnl = if position_size > 0.0 {
                        position_size * (current_price - position_entry_price)
                    } else {
                        position_size.abs() * (position_entry_price - current_price)
                    };

                    // For long: return position value (entry cost + pnl)
                    // For short: return collateral + pnl
                    capital += position_size.abs() * position_entry_price + pnl;

                    trades.push(Trade {
                        entry_price: position_entry_price,
                        exit_price: current_price,
                        size: position_size,
                        pnl,
                    });

                    metrics_recorder.record_trade(pnl);

                    let action = if position_size > 0.0 { "SELL" } else { "COVER" };
                    println!("{:<30} {:>12} {:>12.2} {:>12.2}",
                             truncate(&news.title, 28),
                             format!("{} (PnL: {:.2})", action, pnl),
                             current_price,
                             capital);

                    position_size = 0.0;
                }
            }
        }
    }

    // Close any remaining position at last price
    if position_size != 0.0 {
        let final_price = historical_prices.last().unwrap().close;
        let pnl = if position_size > 0.0 {
            position_size * (final_price - position_entry_price)
        } else {
            position_size.abs() * (position_entry_price - final_price)
        };

        // For long: return position value (entry cost + pnl)
        // For short: return collateral + pnl
        capital += position_size.abs() * position_entry_price + pnl;
        metrics_recorder.record_trade(pnl);

        trades.push(Trade {
            entry_price: position_entry_price,
            exit_price: final_price,
            size: position_size,
            pnl,
        });

        println!("{:<30} {:>12} {:>12.2} {:>12.2}",
                 "End of backtest",
                 format!("CLOSE (PnL: {:.2})", pnl),
                 final_price,
                 capital);
    }

    println!("{:-<70}", "");

    // Display results
    let metrics = metrics_recorder.snapshot();

    println!("\n=== Backtest Results ===\n");
    println!("Initial Capital:  ${:>12.2}", initial_capital);
    println!("Final Capital:    ${:>12.2}", capital);
    println!("Total Return:      {:>12.2}%", (capital / initial_capital - 1.0) * 100.0);
    println!();
    println!("Total Trades:      {:>12}", metrics.total_trades);
    println!("Winning Trades:    {:>12}", metrics.winning_trades);
    println!("Losing Trades:     {:>12}", metrics.losing_trades);
    println!("Win Rate:          {:>12.1}%", metrics.win_rate * 100.0);
    println!();
    println!("Total P&L:        ${:>12.2}", metrics.total_pnl);
    println!("Largest Win:      ${:>12.2}", metrics.largest_win);
    println!("Largest Loss:     ${:>12.2}", metrics.largest_loss);
    println!("Avg Win:          ${:>12.2}", metrics.average_win);
    println!("Avg Loss:         ${:>12.2}", metrics.average_loss);
    println!();
    println!("News Processed:    {:>12}", metrics.news_processed);
    println!("Signals Generated: {:>12}", metrics.signals_generated);
    println!("Avg Confidence:    {:>12.2}", metrics.avg_signal_confidence);

    println!("\n=== Backtest Complete ===");
}

/// Simulated trade record
struct Trade {
    entry_price: f64,
    exit_price: f64,
    size: f64,
    pnl: f64,
}

/// Generate simulated historical data for backtesting
fn generate_simulated_data() -> (Vec<NewsItem>, Vec<OHLCV>) {
    let mut news = Vec::new();
    let mut prices = Vec::new();
    let mut base_price = 45000.0;
    let now = Utc::now();

    // Simulated news events with expected market impact
    let events = vec![
        ("SEC delays Bitcoin ETF decision", -0.02, NewsSource::Bloomberg),
        ("Major exchange reports record trading volume", 0.03, NewsSource::CoinDesk),
        ("Whale moves 10,000 BTC to exchange", -0.04, NewsSource::Twitter),
        ("Large tech company adds Bitcoin to balance sheet", 0.05, NewsSource::Reuters),
        ("Regulatory concerns in Asian market", -0.02, NewsSource::CoinTelegraph),
        ("Network upgrade completes successfully", 0.03, NewsSource::CoinDesk),
        ("Security vulnerability discovered in DeFi protocol", -0.03, NewsSource::Twitter),
        ("Institutional adoption reaches new highs", 0.04, NewsSource::Bloomberg),
        ("Mining difficulty adjustment incoming", 0.01, NewsSource::CoinTelegraph),
        ("Major bank announces crypto custody service", 0.05, NewsSource::Reuters),
        ("Short-term holder selling pressure increases", -0.02, NewsSource::Twitter),
        ("ETF approval expected within weeks", 0.06, NewsSource::Bloomberg),
        ("Exchange faces regulatory scrutiny", -0.04, NewsSource::CoinDesk),
        ("Layer 2 solution launches mainnet", 0.03, NewsSource::CoinTelegraph),
        ("Whale accumulation pattern detected", 0.02, NewsSource::Twitter),
    ];

    for (i, (headline, price_impact, source)) in events.iter().enumerate() {
        let timestamp = now - Duration::hours((events.len() - i) as i64 * 4);

        // Create news item
        news.push(NewsItem {
            id: format!("news-{}", i),
            title: headline.to_string(),
            content: format!("{} This news is expected to have significant market impact.", headline),
            source: *source,
            url: None,
            published_at: timestamp,
            symbols: vec!["BTC".to_string()],
            relevance_score: 0.8,
        });

        // Update price based on news impact
        base_price *= 1.0 + price_impact + (rand_f64() - 0.5) * 0.01;

        // Create OHLCV candle
        let volatility = 0.005;
        let high = base_price * (1.0 + rand_f64() * volatility);
        let low = base_price * (1.0 - rand_f64() * volatility);
        let open = base_price + (rand_f64() - 0.5) * base_price * volatility;
        let close = base_price;

        prices.push(OHLCV::new(
            timestamp,
            open,
            high,
            low,
            close,
            rand_f64() * 1000.0 + 500.0,
        ));
    }

    (news, prices)
}

/// Simple deterministic random-like function for reproducibility
fn rand_f64() -> f64 {
    static mut SEED: u64 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED as f64 / u64::MAX as f64)
    }
}

/// Truncate string to max length
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
