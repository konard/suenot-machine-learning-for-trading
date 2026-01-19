//! # Backtesting Example
//!
//! Demonstrates backtesting a sentiment-based trading strategy.

use llm_sentiment::{
    SentimentAnalyzer, SignalGenerator, Backtester, BacktestConfig,
    Candle, TradingSignal,
};
use rand::Rng;

fn main() {
    println!("=== Sentiment Strategy Backtesting Demo ===\n");

    // Initialize components
    let analyzer = SentimentAnalyzer::new();
    let signal_gen = SignalGenerator::new()
        .with_thresholds(0.25, -0.25, 0.5)
        .with_min_confidence(0.55);

    let config = BacktestConfig {
        initial_capital: 10_000.0,
        fee_pct: 0.001,      // 0.1%
        slippage_pct: 0.0005, // 0.05%
        max_position_size: 0.5,
        use_stop_loss: true,
        use_take_profit: true,
    };

    let backtester = Backtester::new(config);

    // Generate synthetic price data (simulating BTC hourly data)
    println!("Generating synthetic market data...\n");
    let (candles, sentiment_texts) = generate_synthetic_data(168); // 7 days of hourly data

    println!("Generated {} candles (7 days of hourly data)", candles.len());
    println!("Initial price: ${:.2}", candles.first().unwrap().close);
    println!("Final price: ${:.2}", candles.last().unwrap().close);

    let buy_hold_return = (candles.last().unwrap().close / candles.first().unwrap().close - 1.0) * 100.0;
    println!("Buy & Hold Return: {:.2}%\n", buy_hold_return);

    // Analyze sentiment and generate signals
    println!("Analyzing sentiment and generating signals...\n");

    let results: Vec<_> = sentiment_texts.iter()
        .map(|t| analyzer.analyze(t))
        .collect();

    let signals: Vec<_> = results.iter()
        .map(|r| signal_gen.generate(r))
        .collect();

    // Count signal distribution
    let buy_signals = signals.iter()
        .filter(|s| matches!(s.signal, TradingSignal::Buy | TradingSignal::StrongBuy))
        .count();
    let sell_signals = signals.iter()
        .filter(|s| matches!(s.signal, TradingSignal::Sell | TradingSignal::StrongSell))
        .count();
    let hold_signals = signals.iter()
        .filter(|s| s.signal == TradingSignal::Hold)
        .count();

    println!("Signal Distribution:");
    println!("  Buy signals: {} ({:.1}%)", buy_signals, buy_signals as f64 / signals.len() as f64 * 100.0);
    println!("  Sell signals: {} ({:.1}%)", sell_signals, sell_signals as f64 / signals.len() as f64 * 100.0);
    println!("  Hold signals: {} ({:.1}%)", hold_signals, hold_signals as f64 / signals.len() as f64 * 100.0);
    println!();

    // Run backtest
    println!("Running backtest...\n");
    let result = backtester.run(&candles, &signals);

    // Print results
    println!("=== Backtest Results ===\n");
    println!("Performance Metrics:");
    println!("  Total Return: {:.2}%", result.total_return_pct);
    println!("  Annualized Return: {:.2}%", result.annualized_return_pct);
    println!("  Sharpe Ratio: {:.2}", result.sharpe_ratio);
    println!("  Max Drawdown: {:.2}%", result.max_drawdown_pct);
    println!();

    println!("Trade Statistics:");
    println!("  Total Trades: {}", result.total_trades);
    println!("  Winning Trades: {}", result.winning_trades);
    println!("  Losing Trades: {}", result.losing_trades);
    println!("  Win Rate: {:.1}%", result.win_rate * 100.0);
    println!("  Profit Factor: {:.2}", result.profit_factor);
    println!();

    println!("Average Trade Performance:");
    println!("  Avg Trade Return: {:.2}%", result.avg_trade_return);
    println!("  Avg Win: {:.2}%", result.avg_win);
    println!("  Avg Loss: {:.2}%", result.avg_loss);
    println!();

    println!("Capital:");
    println!("  Initial: ${:.2}", 10_000.0);
    println!("  Final: ${:.2}", result.final_capital);
    println!("  Profit/Loss: ${:.2}", result.final_capital - 10_000.0);
    println!();

    // Print recent trades
    if !result.trades.is_empty() {
        println!("=== Recent Trades ===\n");
        for (i, trade) in result.trades.iter().take(5).enumerate() {
            println!("Trade {}: {} @ ${:.2} → ${:.2}",
                i + 1,
                if trade.direction > 0 { "LONG" } else { "SHORT" },
                trade.entry_price,
                trade.exit_price);
            println!("  P/L: ${:.2} ({:.2}%)", trade.pnl, trade.return_pct);
            println!("  Exit Reason: {}", trade.exit_reason);
            println!();
        }
    }

    // Compare with Buy & Hold
    println!("=== Strategy vs Buy & Hold ===\n");
    println!("Sentiment Strategy: {:.2}%", result.total_return_pct);
    println!("Buy & Hold: {:.2}%", buy_hold_return);

    let outperformance = result.total_return_pct - buy_hold_return;
    if outperformance > 0.0 {
        println!("Strategy outperformed by: {:.2}%", outperformance);
    } else {
        println!("Strategy underperformed by: {:.2}%", -outperformance);
    }

    println!("\n=== Backtesting Complete ===");
}

/// Generate synthetic price data with correlated sentiment
fn generate_synthetic_data(num_candles: usize) -> (Vec<Candle>, Vec<String>) {
    let mut rng = rand::thread_rng();
    let mut candles = Vec::with_capacity(num_candles);
    let mut sentiments = Vec::with_capacity(num_candles);

    let mut price = 50_000.0; // Starting BTC price
    let base_timestamp = 1704067200000_i64; // Jan 1, 2024

    // Sentiment templates
    let bullish_texts = vec![
        "Bitcoin showing strong momentum! Bulls in control.",
        "Institutional buying pressure continues. Very bullish!",
        "Breaking resistance! New highs incoming. HODL!",
        "Massive whale accumulation detected. Bullish signal!",
        "Market sentiment extremely positive. Rally continues!",
    ];

    let bearish_texts = vec![
        "Market showing weakness. Bears taking control.",
        "Selling pressure increasing. Caution advised.",
        "Breaking support levels. More downside expected.",
        "FUD spreading on social media. Sentiment turning negative.",
        "Whale distributions detected. Bearish outlook.",
    ];

    let neutral_texts = vec![
        "Market consolidating in a range.",
        "Low volatility expected. Waiting for catalyst.",
        "Mixed signals from various indicators.",
        "Trading volume below average. Indecision prevails.",
        "No clear direction. Stay cautious.",
    ];

    for i in 0..num_candles {
        let timestamp = base_timestamp + (i as i64 * 3600 * 1000);

        // Generate price movement with some trend and noise
        let trend = (i as f64 / num_candles as f64 * std::f64::consts::PI * 2.0).sin() * 0.002;
        let noise: f64 = rng.gen_range(-0.02..0.02);
        let price_change = trend + noise;

        let open = price;
        price = price * (1.0 + price_change);
        let close = price;

        let high = open.max(close) * (1.0 + rng.gen_range(0.001..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.001..0.01));
        let volume = rng.gen_range(100.0..1000.0);

        candles.push(Candle {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            turnover: volume * close,
        });

        // Generate sentiment correlated with price movement
        let sentiment_text = if price_change > 0.01 {
            bullish_texts[rng.gen_range(0..bullish_texts.len())].to_string()
        } else if price_change < -0.01 {
            bearish_texts[rng.gen_range(0..bearish_texts.len())].to_string()
        } else {
            neutral_texts[rng.gen_range(0..neutral_texts.len())].to_string()
        };

        sentiments.push(sentiment_text);
    }

    (candles, sentiments)
}
