//! Backtesting example
//!
//! This example demonstrates how to backtest an EfficientNet-based trading strategy.

use efficientnet_trading::api::BybitClient;
use efficientnet_trading::backtest::{BacktestConfig, BacktestEngine};
use efficientnet_trading::data::Candle;
use efficientnet_trading::imaging::CandlestickRenderer;
use efficientnet_trading::model::ModelPredictor;
use efficientnet_trading::strategy::{Signal, SignalType};

const WINDOW_SIZE: usize = 50;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== EfficientNet Backtesting ===\n");

    // Fetch historical data
    println!("Fetching historical data from Bybit...");
    let client = BybitClient::new();
    let candles = client.fetch_klines("BTCUSDT", "15", 1000).await?;
    println!("  Fetched {} candles\n", candles.len());

    if candles.len() < WINDOW_SIZE + 100 {
        println!("Not enough data for backtesting");
        return Ok(());
    }

    // Initialize components
    let renderer = CandlestickRenderer::new(224, 224);
    let predictor = ModelPredictor::new(224);

    // Generate signals for all candles
    println!("Generating trading signals...");
    let signals = generate_signals(&candles, &renderer, &predictor)?;
    println!("  Generated {} signals\n", signals.len());

    // Count signal types
    let buy_count = signals.iter().filter(|s| s.signal_type == SignalType::Buy).count();
    let sell_count = signals.iter().filter(|s| s.signal_type == SignalType::Sell).count();
    let hold_count = signals.iter().filter(|s| s.signal_type == SignalType::Hold).count();

    println!("Signal Distribution:");
    println!("  Buy:  {} ({:.1}%)", buy_count, buy_count as f64 / signals.len() as f64 * 100.0);
    println!("  Sell: {} ({:.1}%)", sell_count, sell_count as f64 / signals.len() as f64 * 100.0);
    println!("  Hold: {} ({:.1}%)", hold_count, hold_count as f64 / signals.len() as f64 * 100.0);

    // Configure backtest
    let config = BacktestConfig {
        initial_capital: 10000.0,
        max_position_size: 1.0,
        commission_rate: 0.001,  // 0.1%
        slippage_rate: 0.0005,   // 0.05%
        risk_per_trade: 0.02,    // 2%
        stop_loss_pct: 0.02,     // 2%
        take_profit_pct: 0.04,   // 4%
    };

    println!("\nBacktest Configuration:");
    println!("  Initial Capital:  ${:.2}", config.initial_capital);
    println!("  Commission:       {:.2}%", config.commission_rate * 100.0);
    println!("  Slippage:         {:.2}%", config.slippage_rate * 100.0);
    println!("  Risk per Trade:   {:.1}%", config.risk_per_trade * 100.0);
    println!("  Stop Loss:        {:.1}%", config.stop_loss_pct * 100.0);
    println!("  Take Profit:      {:.1}%", config.take_profit_pct * 100.0);

    // Run backtest
    println!("\nRunning backtest...");
    let mut engine = BacktestEngine::new(config.clone());
    let test_candles = &candles[WINDOW_SIZE..];
    let result = engine.run(test_candles, &signals);

    // Display results
    println!("\n{}", "=".repeat(50));
    println!("BACKTEST RESULTS");
    println!("{}", "=".repeat(50));

    println!("\n--- Performance Metrics ---");
    println!("  Total Return:     {:+.2}%", result.metrics.total_return);
    println!("  Annual Return:    {:+.2}%", result.metrics.annual_return);
    println!("  Sharpe Ratio:     {:.2}", result.metrics.sharpe_ratio);
    println!("  Sortino Ratio:    {:.2}", result.metrics.sortino_ratio);
    println!("  Max Drawdown:     {:.2}%", result.metrics.max_drawdown);
    println!("  Calmar Ratio:     {:.2}", result.metrics.calmar_ratio);
    println!("  Volatility:       {:.2}%", result.metrics.volatility);

    println!("\n--- Trade Statistics ---");
    println!("  Total Trades:     {}", result.trade_stats.total_trades);
    println!("  Winning Trades:   {}", result.trade_stats.winning_trades);
    println!("  Losing Trades:    {}", result.trade_stats.losing_trades);
    println!("  Win Rate:         {:.1}%", result.trade_stats.win_rate);
    println!("  Avg Win:          ${:.2}", result.trade_stats.avg_win);
    println!("  Avg Loss:         ${:.2}", result.trade_stats.avg_loss);
    println!("  Largest Win:      ${:.2}", result.trade_stats.largest_win);
    println!("  Largest Loss:     ${:.2}", result.trade_stats.largest_loss);
    println!("  Profit Factor:    {:.2}", result.trade_stats.profit_factor);
    println!("  Avg Trade:        ${:.2}", result.trade_stats.avg_trade);

    // Display recent trades
    if !result.trades.is_empty() {
        println!("\n--- Recent Trades ---");
        println!("{:>10} {:>10} {:>12} {:>12} {:>10}",
            "Side", "Entry", "Exit", "PnL", "PnL %");
        println!("{}", "-".repeat(58));

        for trade in result.trades.iter().rev().take(10).rev() {
            let side = match trade.side {
                efficientnet_trading::strategy::PositionSide::Long => "LONG",
                efficientnet_trading::strategy::PositionSide::Short => "SHORT",
                _ => "FLAT",
            };

            println!("{:>10} {:>10.2} {:>12.2} {:>+10.2} {:>+9.2}%",
                side,
                trade.entry_price,
                trade.exit_price,
                trade.pnl,
                trade.pnl_percent
            );
        }
    }

    // Equity curve summary
    println!("\n--- Equity Curve ---");
    let equity = &result.equity_curve;
    if equity.len() > 10 {
        let step = equity.len() / 10;
        for i in (0..equity.len()).step_by(step).take(10) {
            let pct = (equity[i] / config.initial_capital - 1.0) * 100.0;
            let bar_len = ((pct + 50.0) / 5.0).clamp(0.0, 20.0) as usize;
            let bar: String = "█".repeat(bar_len);
            println!("  {:>5}: ${:>10.2} ({:+6.2}%) {}",
                i, equity[i], pct, bar);
        }
    }

    // Compare with buy & hold
    let first_price = test_candles.first().map(|c| c.close).unwrap_or(0.0);
    let last_price = test_candles.last().map(|c| c.close).unwrap_or(0.0);
    let buy_hold_return = (last_price / first_price - 1.0) * 100.0;

    println!("\n--- Comparison ---");
    println!("  Strategy Return:  {:+.2}%", result.metrics.total_return);
    println!("  Buy & Hold:       {:+.2}%", buy_hold_return);
    println!("  Outperformance:   {:+.2}%",
        result.metrics.total_return - buy_hold_return);

    println!("\nBacktest complete!");
    Ok(())
}

fn generate_signals(
    candles: &[Candle],
    renderer: &CandlestickRenderer,
    predictor: &ModelPredictor,
) -> anyhow::Result<Vec<Signal>> {
    let mut signals = Vec::new();

    for i in WINDOW_SIZE..candles.len() {
        let window: Vec<Candle> = candles[i - WINDOW_SIZE..i].to_vec();
        let current_candle = &candles[i];

        let image = renderer.render(&window);
        let prediction = predictor.predict(&image)?;

        let signal = Signal::new(
            prediction.signal,
            prediction.confidence,
            current_candle.timestamp,
            current_candle.close,
        );

        signals.push(signal);
    }

    Ok(signals)
}
