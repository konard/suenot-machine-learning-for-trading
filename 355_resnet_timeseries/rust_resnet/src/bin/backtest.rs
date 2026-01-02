//! Backtesting framework for ResNet trading strategy
//!
//! This binary runs a historical simulation of the trading strategy.

use anyhow::Result;
use rust_resnet::{
    api::Candle,
    data::{Dataset, StandardScaler},
    model::ResNet18,
    strategy::{RiskManager, TradingSignal, TradingStrategy, Trade, PortfolioState},
    utils::TradingMetrics,
};
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Load candles from CSV
fn load_csv(path: &str) -> Result<Vec<Candle>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut candles = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 { continue; }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 7 {
            candles.push(Candle::new(
                parts[0].parse().unwrap_or(0),
                parts[2].parse().unwrap_or(0.0),
                parts[3].parse().unwrap_or(0.0),
                parts[4].parse().unwrap_or(0.0),
                parts[5].parse().unwrap_or(0.0),
                parts[6].parse().unwrap_or(0.0),
                parts.get(7).and_then(|s| s.parse().ok()).unwrap_or(0.0),
            ));
        }
    }
    Ok(candles)
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("=== ResNet Backtesting Framework ===\n");

    let data_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/BTCUSDT_1_10000candles.csv".to_string());

    // Configuration
    let initial_capital = 100000.0f32;
    let sequence_length = 256;
    let forward_window = 12;
    let threshold = 0.002f32;

    println!("Configuration:");
    println!("  Data:            {}", data_path);
    println!("  Initial capital: ${:.2}", initial_capital);
    println!("  Sequence length: {}", sequence_length);
    println!("  Forward window:  {}", forward_window);
    println!("  Threshold:       {:.2}%\n", threshold * 100.0);

    // Load data
    println!("Loading data...");
    let candles = load_csv(&data_path)?;

    if candles.len() < sequence_length + forward_window + 100 {
        println!("Not enough data. Run 'cargo run --bin fetch_data' first.");
        return Ok(());
    }
    println!("Loaded {} candles\n", candles.len());

    // Create dataset
    let dataset = Dataset::from_candles(
        candles.clone(),
        sequence_length,
        forward_window,
        threshold,
    )?;

    // Use last 20% for backtesting
    let test_start = (dataset.len() as f32 * 0.8) as usize;

    // Create components
    let model = ResNet18::new(dataset.num_features, 3);
    let strategy = TradingStrategy::default();
    let risk_manager = RiskManager::default();
    let mut scaler = StandardScaler::new(dataset.num_features);

    // Fit scaler on training data
    let train_x = dataset.x.slice(ndarray::s![..test_start, .., ..]).to_owned();
    scaler.fit(&train_x);

    // Initialize portfolio
    let mut portfolio = PortfolioState::new(initial_capital);
    let mut trades: Vec<Trade> = Vec::new();
    let mut current_trade: Option<Trade> = None;
    let mut returns: Vec<f32> = Vec::new();

    println!("Running backtest...");
    println!("Test period: {} samples\n", dataset.len() - test_start);

    // Backtest loop
    for i in test_start..dataset.len() {
        let sample = dataset.get(i).unwrap();
        let candle_idx = sequence_length + i - 1;

        if candle_idx >= candles.len() {
            break;
        }

        let current_price = candles[candle_idx].close as f32;
        let current_time = candles[candle_idx].timestamp;

        // Prepare input
        let mut input = ndarray::Array3::zeros((1, dataset.num_features, sequence_length));
        for f in 0..dataset.num_features {
            for t in 0..sequence_length {
                input[[0, f, t]] = sample.features[[f, t]];
            }
        }
        let normalized = scaler.transform(&input);

        // Get prediction
        let probs = model.predict_proba(&normalized);
        let (signal, confidence) = strategy.generate_signal(&[
            probs[[0, 0]], probs[[0, 1]], probs[[0, 2]]
        ]);

        // Check existing position
        if let Some(ref mut trade) = current_trade {
            let (should_close, reason) = risk_manager.should_close(trade, current_price, current_time);

            // Also close if signal reverses
            let signal_reverse = match (&signal, trade.direction > 0.0) {
                (TradingSignal::Short, true) => true,
                (TradingSignal::Long, false) => true,
                _ => false,
            };

            if should_close || signal_reverse {
                trade.close(current_time, current_price);
                let pnl = trade.pnl.unwrap_or(0.0);
                portfolio.update(pnl);

                let trade_return = pnl / trade.size;
                returns.push(trade_return);

                trades.push(trade.clone());
                current_trade = None;
            }
        }

        // Open new position if no current trade
        if current_trade.is_none() && signal != TradingSignal::Neutral {
            let position_size = strategy.calculate_position_size(confidence, portfolio.value);
            let drawdown_adj = risk_manager.drawdown_adjustment(portfolio.current_drawdown);
            let adjusted_size = position_size * drawdown_adj;

            if adjusted_size > 0.0 {
                let direction = signal.direction();
                current_trade = Some(Trade::open(
                    current_time,
                    current_price,
                    direction,
                    adjusted_size,
                    signal,
                    confidence,
                ));
            }
        }
    }

    // Close any remaining position
    if let Some(mut trade) = current_trade {
        let last_candle = candles.last().unwrap();
        trade.close(last_candle.timestamp, last_candle.close as f32);
        if let Some(pnl) = trade.pnl {
            portfolio.update(pnl);
            returns.push(pnl / trade.size);
        }
        trades.push(trade);
    }

    // Calculate metrics
    println!("=== Backtest Results ===\n");

    let trading_metrics = TradingMetrics::from_minute_returns(returns);
    println!("{}", trading_metrics.summary());

    println!("\n=== Portfolio Summary ===");
    println!("Initial value:   ${:.2}", initial_capital);
    println!("Final value:     ${:.2}", portfolio.value);
    println!("Total return:    {:.2}%", portfolio.total_return() * 100.0);
    println!("Max drawdown:    {:.2}%", portfolio.max_drawdown * 100.0);
    println!("Total trades:    {}", trades.len());
    println!("Win rate:        {:.2}%", portfolio.win_rate() * 100.0);
    println!("Profit factor:   {:.2}", portfolio.profit_factor());

    // Trade analysis
    if !trades.is_empty() {
        let winning: Vec<_> = trades.iter().filter(|t| t.pnl.unwrap_or(0.0) > 0.0).collect();
        let losing: Vec<_> = trades.iter().filter(|t| t.pnl.unwrap_or(0.0) < 0.0).collect();

        let avg_win = if !winning.is_empty() {
            winning.iter().map(|t| t.pnl.unwrap_or(0.0)).sum::<f32>() / winning.len() as f32
        } else { 0.0 };

        let avg_loss = if !losing.is_empty() {
            losing.iter().map(|t| t.pnl.unwrap_or(0.0).abs()).sum::<f32>() / losing.len() as f32
        } else { 0.0 };

        println!("\n=== Trade Analysis ===");
        println!("Winning trades:  {}", winning.len());
        println!("Losing trades:   {}", losing.len());
        println!("Avg win:         ${:.2}", avg_win);
        println!("Avg loss:        ${:.2}", avg_loss);
        println!("Win/Loss ratio:  {:.2}", if avg_loss > 0.0 { avg_win / avg_loss } else { 0.0 });
    }

    println!("\nNote: Results are from randomly initialized model.");
    println!("Actual performance requires proper model training.");

    Ok(())
}
