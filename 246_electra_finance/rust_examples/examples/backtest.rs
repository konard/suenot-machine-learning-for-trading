//! Example: Backtest ELECTRA sentiment trading strategy
//!
//! This example demonstrates backtesting a sentiment-based trading
//! strategy using ELECTRA models with Bybit cryptocurrency data.

use electra_finance::api::bybit::{BybitClient, Interval, Kline};
use electra_finance::data::processor::DataProcessor;
use electra_finance::nlp::tokenizer::SimpleTokenizer;
use electra_finance::models::network::SentimentNetwork;
use rand::Rng;

/// Generate synthetic news headlines based on price movements
fn generate_news_for_kline(kline: &Kline) -> Vec<String> {
    let ret = kline.return_pct();
    let pct = (ret.abs() * 100.0).max(0.1);

    if ret > 0.005 {
        vec![
            format!("Asset rallied {:.1}% on strong trading volume", pct),
            format!("Bullish momentum drove prices up {:.1}% today", pct),
        ]
    } else if ret < -0.005 {
        vec![
            format!("Asset declined {:.1}% amid heavy selling pressure", pct),
            format!("Bearish sentiment drove prices down {:.1}% today", pct),
        ]
    } else {
        vec![
            format!("Asset traded flat with minimal price change of {:.1}%", pct),
        ]
    }
}

/// Backtesting engine for sentiment-based strategies
struct Backtester {
    initial_capital: f64,
    signal_threshold: f64,
    stop_loss_pct: f64,
    holding_period: usize,
}

/// Single trade record
#[derive(Debug)]
struct TradeRecord {
    entry_idx: usize,
    exit_idx: usize,
    direction: i32,
    entry_price: f64,
    exit_price: f64,
    pnl_pct: f64,
}

impl Backtester {
    fn new(initial_capital: f64) -> Self {
        Self {
            initial_capital,
            signal_threshold: 0.1,
            stop_loss_pct: 0.05,
            holding_period: 5,
        }
    }

    fn run(
        &self,
        klines: &[Kline],
        signals: &[(f64, f64)], // (signal, confidence)
    ) -> BacktestResult {
        let mut capital = self.initial_capital;
        let mut trades: Vec<TradeRecord> = Vec::new();
        let mut equity_curve: Vec<f64> = Vec::with_capacity(klines.len());

        let mut position: Option<(usize, i32, f64)> = None; // (entry_idx, direction, entry_price)
        let mut days_held = 0usize;

        for i in 0..klines.len() {
            // Check exit conditions
            if let Some((entry_idx, direction, entry_price)) = position {
                days_held += 1;
                let current_price = klines[i].close;
                let pnl_pct = if direction == 1 {
                    (current_price - entry_price) / entry_price
                } else {
                    (entry_price - current_price) / entry_price
                };

                let should_exit = pnl_pct <= -self.stop_loss_pct
                    || days_held >= self.holding_period;

                if should_exit {
                    capital *= 1.0 + pnl_pct;
                    trades.push(TradeRecord {
                        entry_idx,
                        exit_idx: i,
                        direction,
                        entry_price,
                        exit_price: current_price,
                        pnl_pct,
                    });
                    position = None;
                    days_held = 0;
                }
            }

            // Check entry conditions
            if position.is_none() && i < signals.len() {
                let (signal, confidence) = signals[i];
                if signal.abs() > self.signal_threshold && confidence > 0.3 {
                    let direction = if signal > 0.0 { 1 } else { -1 };
                    position = Some((i, direction, klines[i].close));
                    days_held = 0;
                }
            }

            equity_curve.push(capital);
        }

        // Compute metrics
        let total_return = (capital - self.initial_capital) / self.initial_capital;
        let winning = trades.iter().filter(|t| t.pnl_pct > 0.0).count();
        let win_rate = if trades.is_empty() { 0.0 } else { winning as f64 / trades.len() as f64 };

        let gross_profit: f64 = trades.iter().filter(|t| t.pnl_pct > 0.0).map(|t| t.pnl_pct).sum();
        let gross_loss: f64 = trades.iter().filter(|t| t.pnl_pct <= 0.0).map(|t| t.pnl_pct.abs()).sum();
        let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else { f64::INFINITY };

        // Compute daily returns for Sharpe ratio
        let daily_returns: Vec<f64> = equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let sharpe = compute_sharpe(&daily_returns);
        let sortino = compute_sortino(&daily_returns);
        let max_dd = compute_max_drawdown(&equity_curve);

        BacktestResult {
            total_return,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown: max_dd,
            win_rate,
            profit_factor,
            num_trades: trades.len(),
            final_capital: capital,
        }
    }
}

#[derive(Debug)]
struct BacktestResult {
    total_return: f64,
    sharpe_ratio: f64,
    sortino_ratio: f64,
    max_drawdown: f64,
    win_rate: f64,
    profit_factor: f64,
    num_trades: usize,
    final_capital: f64,
}

fn compute_sharpe(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    if std == 0.0 { 0.0 } else { (252.0_f64).sqrt() * mean / std }
}

fn compute_sortino(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
    if downside.is_empty() {
        return 0.0;
    }
    let down_var = downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64;
    let down_std = down_var.sqrt();
    if down_std == 0.0 { 0.0 } else { (252.0_f64).sqrt() * mean / down_std }
}

fn compute_max_drawdown(equity: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0;

    for &val in equity {
        if val > peak {
            peak = val;
        }
        let dd = (val - peak) / peak;
        if dd < max_dd {
            max_dd = dd;
        }
    }

    max_dd
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("=== ELECTRA Finance: Backtesting ===\n");

    // Create components
    let tokenizer = SimpleTokenizer::new(64);
    let model = SentimentNetwork::new(tokenizer.vocab_size(), 64, 128, 3);

    // Try fetching real data, fall back to synthetic
    let klines = match BybitClient::new().get_klines("BTCUSDT", Interval::Day1, 200, None, None) {
        Ok(k) if !k.is_empty() => {
            println!("Using real Bybit data: {} klines", k.len());
            k
        }
        _ => {
            println!("Generating synthetic data for demo...");
            generate_synthetic_klines(200)
        }
    };

    println!("Data points: {}\n", klines.len());

    // Generate sentiment signals for each kline
    let mut signals: Vec<(f64, f64)> = Vec::with_capacity(klines.len());

    for kline in &klines {
        let news = generate_news_for_kline(kline);
        let (batch_ids, batch_masks) = tokenizer.tokenize_batch(&news);
        let results = model.predict_batch(&batch_ids, &batch_masks);
        let (signal, confidence) = model.aggregate_signal(&results, 0.3);
        signals.push((signal, confidence));
    }

    // Run backtests with different configurations
    println!("--- Strategy 1: Default Parameters ---");
    let bt1 = Backtester::new(100_000.0);
    let result1 = bt1.run(&klines, &signals);
    print_result(&result1);

    println!("\n--- Strategy 2: Tight Stop-Loss ---");
    let mut bt2 = Backtester::new(100_000.0);
    bt2.stop_loss_pct = 0.02;
    bt2.holding_period = 3;
    let result2 = bt2.run(&klines, &signals);
    print_result(&result2);

    println!("\n--- Strategy 3: High Conviction Only ---");
    let mut bt3 = Backtester::new(100_000.0);
    bt3.signal_threshold = 0.3;
    bt3.holding_period = 7;
    let result3 = bt3.run(&klines, &signals);
    print_result(&result3);

    // Comparison
    println!("\n=== Strategy Comparison ===");
    println!("{:<25} {:>12} {:>10} {:>10} {:>10} {:>8}",
        "Strategy", "Return", "Sharpe", "Max DD", "Win Rate", "Trades");
    println!("{:-<85}", "");
    print_row("Default", &result1);
    print_row("Tight Stop-Loss", &result2);
    print_row("High Conviction", &result3);

    println!("\n=== Backtesting complete ===");

    Ok(())
}

fn print_result(result: &BacktestResult) {
    println!("  Total Return:   {:+.2}%", result.total_return * 100.0);
    println!("  Sharpe Ratio:   {:.2}", result.sharpe_ratio);
    println!("  Sortino Ratio:  {:.2}", result.sortino_ratio);
    println!("  Max Drawdown:   {:.2}%", result.max_drawdown * 100.0);
    println!("  Win Rate:       {:.1}%", result.win_rate * 100.0);
    println!("  Profit Factor:  {:.2}", result.profit_factor);
    println!("  Num Trades:     {}", result.num_trades);
    println!("  Final Capital:  ${:.2}", result.final_capital);
}

fn print_row(name: &str, result: &BacktestResult) {
    println!("{:<25} {:>+11.2}% {:>10.2} {:>+9.2}% {:>9.1}% {:>8}",
        name,
        result.total_return * 100.0,
        result.sharpe_ratio,
        result.max_drawdown * 100.0,
        result.win_rate * 100.0,
        result.num_trades
    );
}

/// Generate synthetic kline data for testing
fn generate_synthetic_klines(n: usize) -> Vec<Kline> {
    let mut rng = rand::thread_rng();
    let mut klines = Vec::with_capacity(n);
    let mut price = 40000.0; // Starting BTC price

    let start_ts = 1700000000000i64; // Nov 2023

    for i in 0..n {
        let change: f64 = rng.gen_range(-0.03..0.03);
        let open = price;
        let close = open * (1.0 + change);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..10000.0);

        klines.push(Kline {
            timestamp: start_ts + (i as i64 * 86400000),
            open,
            high,
            low,
            close,
            volume,
            turnover: volume * close,
        });

        price = close;
    }

    klines
}
