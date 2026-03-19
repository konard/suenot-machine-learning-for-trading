//! Backtest a DeBERTa sentiment strategy.
//!
//! Runs a full backtest using synthetic data and sample headlines,
//! then prints performance metrics.

use deberta_trading::SentimentBacktester;
use deberta_trading::data::bybit::generate_synthetic_klines;

fn main() {
    tracing_subscriber::fmt::init();

    println!("=== DeBERTa Sentiment Strategy Backtest ===\n");

    // Generate synthetic price data
    let klines = generate_synthetic_klines(200, "BTCUSDT");
    println!("Generated {} synthetic daily candles for BTCUSDT", klines.len());

    // Create sample headlines for various bars
    let mut headlines: Vec<Vec<&str>> = vec![vec![]; 200];

    // Simulate news events at various points
    headlines[10] = vec!["Bitcoin surges on massive institutional buying"];
    headlines[20] = vec!["Record inflows into crypto ETFs"];
    headlines[35] = vec!["Regulatory crackdown threatens crypto market"];
    headlines[50] = vec!["Bitcoin rally continues with strong volume growth"];
    headlines[65] = vec!["Crypto exchange reports security breach"];
    headlines[80] = vec!["Bitcoin exceeds analyst expectations", "Bullish momentum builds"];
    headlines[95] = vec!["Market selloff deepens amid recession fears"];
    headlines[110] = vec!["Institutional adoption of Bitcoin reaches record high"];
    headlines[125] = vec!["Fraud investigation into major crypto firm"];
    headlines[140] = vec!["Bitcoin network upgrade approved", "Strong growth in DeFi"];
    headlines[155] = vec!["Central bank policy change raises risk concerns"];
    headlines[170] = vec!["Bitcoin beats all-time high on broad rally"];
    headlines[185] = vec!["Crypto market crashes on regulatory news"];

    // Configure and run backtest
    let backtester = SentimentBacktester::new()
        .with_capital(100_000.0)
        .with_position_pct(0.1)
        .with_entry_threshold(0.3)
        .with_risk_params(0.05, 0.10);

    let results = backtester.run(&klines, &headlines);

    // Print results
    println!("\n=== Backtest Results ===\n");
    println!("Number of trades:  {}", results.n_trades);
    println!("Total return:      {:.2}%", results.total_return * 100.0);
    println!("Sharpe ratio:      {:.4}", results.sharpe_ratio);
    println!("Sortino ratio:     {:.4}", results.sortino_ratio);
    println!("Max drawdown:      {:.2}%", results.max_drawdown * 100.0);
    println!("Win rate:          {:.2}%", results.win_rate * 100.0);
    println!("Profit factor:     {:.4}", results.profit_factor);

    // Print individual trades
    if !results.trades.is_empty() {
        println!("\n=== Trade Log ===\n");
        for (i, trade) in results.trades.iter().enumerate() {
            let dir = if trade.direction > 0 { "LONG" } else { "SHORT" };
            println!(
                "Trade {}: {} | Entry: {:.2} (bar {}) → Exit: {:.2} (bar {}) | \
                 Return: {:.2}% | PnL: {:.2}",
                i + 1,
                dir,
                trade.entry_price,
                trade.entry_idx,
                trade.exit_price,
                trade.exit_idx,
                trade.return_pct * 100.0,
                trade.pnl,
            );
        }
    }

    println!("\n=== Backtest Complete ===");
}
