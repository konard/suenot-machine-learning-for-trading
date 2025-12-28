//! Пример: Бэктест стратегии парной торговли
//!
//! Тестирует стратегию парной торговли на исторических данных.
//!
//! Использование:
//! ```
//! cargo run --bin pairs_trading -- --file1 data/BTCUSDT_1h.csv --file2 data/ETHUSDT_1h.csv
//! ```

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use crypto_time_series::api::load_candles;
use crypto_time_series::trading::{
    PairsTradingStrategy, PairsTradingParams,
    run_backtest, BacktestParams,
    WalkForwardAnalysis,
    engle_granger_test,
};
use crypto_time_series::TimeSeries;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Backtest pairs trading strategy")]
struct Args {
    /// First asset CSV file
    #[arg(long)]
    file1: PathBuf,

    /// Second asset CSV file
    #[arg(long)]
    file2: PathBuf,

    /// Entry Z-score threshold
    #[arg(long, default_value = "2.0")]
    entry: f64,

    /// Exit Z-score threshold
    #[arg(long, default_value = "0.5")]
    exit: f64,

    /// Stop-loss Z-score threshold
    #[arg(long, default_value = "4.0")]
    stop_loss: f64,

    /// Lookback period for Z-score
    #[arg(long, default_value = "20")]
    lookback: usize,

    /// Initial capital
    #[arg(long, default_value = "10000")]
    capital: f64,

    /// Commission rate (e.g., 0.001 = 0.1%)
    #[arg(long, default_value = "0.001")]
    commission: f64,

    /// Run walk-forward analysis
    #[arg(long)]
    walk_forward: bool,

    /// In-sample period for walk-forward
    #[arg(long, default_value = "500")]
    in_sample: usize,

    /// Out-of-sample period for walk-forward
    #[arg(long, default_value = "100")]
    out_sample: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "=".repeat(60).blue());
    println!("{}", "Pairs Trading Backtest".bold().blue());
    println!("{}", "=".repeat(60).blue());

    // Загружаем данные
    let candles1 = load_candles(&args.file1)?;
    let candles2 = load_candles(&args.file2)?;

    let name1 = args
        .file1
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Asset1");
    let name2 = args
        .file2
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Asset2");

    println!(
        "\n{}: {} candles",
        name1.cyan(),
        candles1.len()
    );
    println!(
        "{}: {} candles\n",
        name2.cyan(),
        candles2.len()
    );

    let ts1 = TimeSeries::from_candles(name1, &candles1);
    let ts2 = TimeSeries::from_candles(name2, &candles2);

    // Выравниваем данные
    let min_len = ts1.data.len().min(ts2.data.len());
    let prices1: Vec<f64> = ts1.data[..min_len].to_vec();
    let prices2: Vec<f64> = ts2.data[..min_len].to_vec();

    if min_len < 200 {
        anyhow::bail!("Not enough data for backtesting (need at least 200 observations)");
    }

    // Вычисляем hedge ratio
    println!("{}", "Cointegration Analysis".bold());
    println!("{}", "-".repeat(40));

    let coint = engle_granger_test(&prices1, &prices2)
        .ok_or_else(|| anyhow::anyhow!("Cointegration test failed"))?;

    println!("Hedge ratio: {:.4}", coint.hedge_ratio);
    println!("Is cointegrated: {}", if coint.is_cointegrated {
        "Yes".green()
    } else {
        "No".red()
    });

    if let Some(hl) = coint.half_life {
        println!("Half-life: {:.1} periods", hl);
    }

    // Настраиваем стратегию
    println!("\n{}", "Strategy Parameters".bold());
    println!("{}", "-".repeat(40));

    let strategy_params = PairsTradingParams {
        entry_threshold: args.entry,
        exit_threshold: args.exit,
        stop_loss_threshold: args.stop_loss,
        lookback_period: args.lookback,
        min_holding_period: 1,
        dynamic_hedge: true,
        hedge_recalc_period: 60,
    };

    println!("Entry threshold: ±{:.1}σ", args.entry);
    println!("Exit threshold: ±{:.1}σ", args.exit);
    println!("Stop-loss: ±{:.1}σ", args.stop_loss);
    println!("Lookback period: {}", args.lookback);

    let backtest_params = BacktestParams {
        initial_capital: args.capital,
        commission: args.commission,
        slippage: 0.0005,
        position_size: 1.0,
    };

    println!("\nInitial capital: ${:.0}", args.capital);
    println!("Commission: {:.2}%", args.commission * 100.0);

    // Запускаем бэктест
    println!("\n{}", "Running Backtest".bold());
    println!("{}", "-".repeat(40));

    let mut strategy = PairsTradingStrategy::new(
        name1,
        name2,
        coint.hedge_ratio,
        strategy_params.clone(),
    );

    let result = run_backtest(&mut strategy, &prices1, &prices2, &backtest_params);

    // Результаты
    println!("\n{}", "Backtest Results".bold());
    println!("{}", "=".repeat(40));
    println!("{}", result.display());

    // Equity curve summary
    if !result.equity_curve.is_empty() {
        let final_equity = *result.equity_curve.last().unwrap();
        let total_return = (final_equity - args.capital) / args.capital * 100.0;

        println!("\n{}", "Equity Summary".bold());
        println!("{}", "-".repeat(40));
        println!("Starting capital: ${:.2}", args.capital);
        println!("Final equity: ${:.2}", final_equity);
        println!(
            "Total return: {}",
            if total_return >= 0.0 {
                format!("+{:.2}%", total_return).green()
            } else {
                format!("{:.2}%", total_return).red()
            }
        );
    }

    // Показываем последние сделки
    if !result.trades.is_empty() {
        println!("\n{}", "Recent Trades".bold());
        println!("{}", "-".repeat(40));

        let start = result.trades.len().saturating_sub(10);
        for trade in &result.trades[start..] {
            let pnl_str = if trade.pnl >= 0.0 {
                format!("+{:.4}", trade.pnl).green()
            } else {
                format!("{:.4}", trade.pnl).red()
            };
            println!(
                "  {:?} {}->{}: PnL={} ({})",
                trade.position,
                trade.entry_time,
                trade.exit_time,
                pnl_str,
                trade.exit_reason.as_str()
            );
        }
    }

    // Walk-forward анализ
    if args.walk_forward {
        println!("\n{}", "Walk-Forward Analysis".bold());
        println!("{}", "=".repeat(40));

        let mut wfa = WalkForwardAnalysis::new(args.in_sample, args.out_sample);
        wfa.run(&prices1, &prices2, &strategy_params, &backtest_params);

        println!("{}", wfa.display());

        let efficiency = wfa.efficiency_ratio();
        let efficiency_status = if efficiency > 0.5 {
            "ROBUST".green()
        } else if efficiency > 0.2 {
            "MODERATE".yellow()
        } else {
            "WEAK".red()
        };

        println!("\nStrategy robustness: {} ({:.2})", efficiency_status, efficiency);
    }

    // Рекомендации
    println!("\n{}", "Recommendations".bold());
    println!("{}", "-".repeat(40));

    if !coint.is_cointegrated {
        println!("  {} Pair is not cointegrated - high divergence risk", "⚠".yellow());
    }

    if result.stats.win_rate < 0.4 {
        println!("  {} Low win rate - consider adjusting entry/exit thresholds", "⚠".yellow());
    }

    if result.stats.profit_factor < 1.0 {
        println!("  {} Strategy is not profitable - review parameters", "⚠".red());
    }

    if result.max_drawdown > 0.2 {
        println!("  {} High drawdown - consider position sizing", "⚠".yellow());
    }

    if result.sharpe_ratio > 1.0 {
        println!("  {} Good risk-adjusted returns", "✓".green());
    }

    if coint.half_life.map(|hl| hl < 20.0).unwrap_or(false) {
        println!("  {} Short half-life suggests good mean-reversion", "✓".green());
    }

    // Оптимальные торговые сигналы
    println!("\n{}", "Current Trading Signals".bold());
    println!("{}", "-".repeat(40));

    let zscore = crypto_time_series::trading::spread_zscore(
        &crypto_time_series::trading::compute_spread(&prices1, &prices2, coint.hedge_ratio),
        args.lookback,
    );

    let current_z = *zscore.last().unwrap_or(&0.0);
    println!("Current Z-score: {:.4}", current_z);

    if current_z > args.entry {
        println!(
            "{} Short spread: Sell {} @ {:.2}, Buy {} @ {:.2}",
            "SIGNAL:".red().bold(),
            name1,
            prices1.last().unwrap_or(&0.0),
            name2,
            prices2.last().unwrap_or(&0.0)
        );
        println!(
            "  Hedge ratio: {:.4} (${:.2} of {} per ${} of {})",
            coint.hedge_ratio,
            coint.hedge_ratio * 100.0,
            name2,
            100,
            name1
        );
    } else if current_z < -args.entry {
        println!(
            "{} Long spread: Buy {} @ {:.2}, Sell {} @ {:.2}",
            "SIGNAL:".green().bold(),
            name1,
            prices1.last().unwrap_or(&0.0),
            name2,
            prices2.last().unwrap_or(&0.0)
        );
        println!(
            "  Hedge ratio: {:.4} (${:.2} of {} per ${} of {})",
            coint.hedge_ratio,
            coint.hedge_ratio * 100.0,
            name2,
            100,
            name1
        );
    } else {
        println!("{} No entry signal (Z-score within ±{:.1}σ)", "WAIT:".yellow(), args.entry);
    }

    println!("\n{}", "Done!".green().bold());
    Ok(())
}
