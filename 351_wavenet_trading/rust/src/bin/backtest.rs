//! Бэктестинг торговой стратегии на основе WaveNet
//!
//! Использование:
//! ```
//! cargo run --bin backtest -- --data ./data/BTCUSDT_1h.csv
//! ```

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use std::path::PathBuf;

use wavenet_trading::api::load_candles;
use wavenet_trading::analysis::FeatureBuilder;
use wavenet_trading::models::{WaveNet, WaveNetConfig};
use wavenet_trading::trading::{Backtester, BacktestConfig, SignalStats};
use wavenet_trading::Signal;

#[derive(Parser, Debug)]
#[command(author, version, about = "Backtest WaveNet trading strategy")]
struct Args {
    /// Path to CSV data file
    #[arg(short, long, default_value = "./data/BTCUSDT_1h.csv")]
    data: PathBuf,

    /// Window size for WaveNet
    #[arg(short, long, default_value = "100")]
    window_size: usize,

    /// Signal threshold
    #[arg(short, long, default_value = "0.001")]
    threshold: f64,

    /// Initial capital
    #[arg(long, default_value = "10000")]
    capital: f64,

    /// Commission per trade (fraction)
    #[arg(long, default_value = "0.001")]
    commission: f64,

    /// Position size (fraction of capital)
    #[arg(long, default_value = "1.0")]
    position_size: f64,

    /// Output results to CSV
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "=".repeat(60).blue());
    println!("{}", "WaveNet Trading - Backtesting Engine".bold().blue());
    println!("{}", "=".repeat(60).blue());

    // Загружаем данные
    println!("\n{} Loading data...", "📂".cyan());

    let candles = if args.data.exists() {
        load_candles(&args.data)?
    } else {
        println!("{} Data file not found: {}", "⚠".yellow(), args.data.display());
        println!("  Using synthetic demo data...\n");

        use chrono::Utc;
        use wavenet_trading::Candle;

        // Создаём более реалистичные синтетические данные
        let mut price = 50000.0;
        let mut rng_seed = 42u64;

        (0..1000)
            .map(|i| {
                // Простой PRNG для воспроизводимости
                rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let random = (rng_seed as f64 / u64::MAX as f64) - 0.5;

                // Случайное блуждание с трендом и волатильностью
                let trend = (i as f64 * 0.01).sin() * 0.001;
                let volatility = 0.002;
                let return_pct = trend + random * volatility;

                price *= 1.0 + return_pct;

                let high = price * (1.0 + random.abs() * 0.005);
                let low = price * (1.0 - random.abs() * 0.005);

                Candle {
                    timestamp: Utc::now(),
                    open: price * (1.0 - return_pct * 0.3),
                    high,
                    low,
                    close: price,
                    volume: 100.0 + random.abs() * 50.0,
                }
            })
            .collect()
    };

    println!("  {} {} candles loaded", "✓".green(), candles.len());
    println!("  ├─ First price: ${:.2}", candles.first().unwrap().close);
    println!("  └─ Last price:  ${:.2}", candles.last().unwrap().close);

    if candles.len() < args.window_size + 10 {
        println!("{} Not enough data for backtesting", "✗".red());
        return Ok(());
    }

    // Подготавливаем признаки
    println!("\n{} Preparing features...", "🔧".cyan());
    let builder = FeatureBuilder::new(candles.clone());
    let mut features = builder.build_all();

    features.fill_nan(0.0);
    features.normalize();

    println!("  {} {} features prepared", "✓".green(), features.num_features());

    // Создаём модель
    println!("\n{} Initializing WaveNet model...", "🏗".cyan());

    let config = WaveNetConfig {
        input_channels: features.num_features(),
        residual_channels: 32,
        skip_channels: 32,
        output_channels: 1,
        kernel_size: 2,
        num_blocks: 8,
        num_stacks: 1,
    };

    let model = WaveNet::new(config);
    println!("  Receptive field: {}", model.config.receptive_field());

    // Генерируем предсказания для всего датасета
    println!("\n{} Generating predictions...", "🎯".cyan());

    let input_data = features.as_wavenet_input();
    let mut predictions = Vec::new();

    for i in args.window_size..candles.len() {
        let window: Vec<Vec<f64>> = input_data
            .iter()
            .map(|ch| ch[i - args.window_size..i].to_vec())
            .collect();

        let pred = model.predict(&window);
        predictions.push(pred);
    }

    println!("  {} {} predictions generated", "✓".green(), predictions.len());

    // Преобразуем предсказания в сигналы
    let signals: Vec<Signal> = predictions
        .iter()
        .map(|&p| {
            if p > args.threshold {
                Signal::Buy
            } else if p < -args.threshold {
                Signal::Sell
            } else {
                Signal::Hold
            }
        })
        .collect();

    // Статистика сигналов
    let signal_stats = SignalStats::from_signals(&signals);
    println!("\n{} Signal distribution:", "📊".cyan());
    println!("  ├─ Buy:  {} ({:.1}%)", signal_stats.buy_count,
        100.0 * signal_stats.buy_count as f64 / signal_stats.total as f64);
    println!("  ├─ Sell: {} ({:.1}%)", signal_stats.sell_count,
        100.0 * signal_stats.sell_count as f64 / signal_stats.total as f64);
    println!("  ├─ Hold: {} ({:.1}%)", signal_stats.hold_count,
        100.0 * signal_stats.hold_count as f64 / signal_stats.total as f64);
    println!("  └─ Transitions: {}", signal_stats.transitions);

    // Запускаем бэктест
    println!("\n{} Running backtest...", "⚡".cyan());

    let backtest_config = BacktestConfig {
        initial_capital: args.capital,
        commission: args.commission,
        slippage: 0.0005,
        position_size: args.position_size,
    };

    let backtester = Backtester::new(backtest_config);
    let test_candles = &candles[args.window_size..];
    let result = backtester.run(test_candles, &signals);

    // Выводим результаты
    println!("\n{}", "=".repeat(60).green());
    result.print_summary();
    println!("{}", "=".repeat(60).green());

    // Сравнение с Buy & Hold
    println!("\n{} Comparison with Buy & Hold:", "📈".cyan());

    let first_price = test_candles.first().unwrap().close;
    let last_price = test_candles.last().unwrap().close;
    let buy_hold_return = (last_price - first_price) / first_price;
    let strategy_return = result.metrics.total_return;

    let buy_hold_final = args.capital * (1.0 + buy_hold_return);
    let strategy_final = result.equity_curve.last().copied().unwrap_or(args.capital);

    println!("  Strategy vs Buy & Hold:");
    println!("  ├─ Strategy return:   {:.2}% (${:.2})",
        strategy_return * 100.0, strategy_final);
    println!("  ├─ Buy & Hold return: {:.2}% (${:.2})",
        buy_hold_return * 100.0, buy_hold_final);

    let outperformance = strategy_return - buy_hold_return;
    let outperf_str = if outperformance >= 0.0 {
        format!("+{:.2}%", outperformance * 100.0).green()
    } else {
        format!("{:.2}%", outperformance * 100.0).red()
    };
    println!("  └─ Outperformance:    {}", outperf_str);

    // Risk-adjusted metrics
    println!("\n{} Risk-Adjusted Performance:", "📊".cyan());
    println!("  ├─ Sharpe Ratio:  {:.3}", result.metrics.sharpe_ratio);
    println!("  ├─ Sortino Ratio: {:.3}", result.metrics.sortino_ratio);
    println!("  └─ Max Drawdown:  {:.2}%", result.metrics.max_drawdown * 100.0);

    // Equity curve summary
    println!("\n{} Equity Curve Summary:", "📈".cyan());
    let eq = &result.equity_curve;
    let peak = eq.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let trough = eq.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("  ├─ Starting: ${:.2}", eq.first().unwrap_or(&args.capital));
    println!("  ├─ Peak:     ${:.2}", peak);
    println!("  ├─ Trough:   ${:.2}", trough);
    println!("  └─ Final:    ${:.2}", eq.last().unwrap_or(&args.capital));

    // Сохраняем результаты
    if let Some(output_path) = &args.output {
        let csv_content = result.to_csv();
        std::fs::write(output_path, csv_content)?;
        println!("\n{} Results saved to: {}", "💾".green(), output_path.display());
    }

    // Disclaimer
    println!("\n{}", "=".repeat(60).yellow());
    println!("{} DISCLAIMER:", "⚠".yellow().bold());
    println!("  This is a DEMO backtest with randomly initialized model.");
    println!("  Results do NOT represent actual trading performance.");
    println!("  DO NOT use for real trading decisions.");
    println!("{}", "=".repeat(60).yellow());

    println!("\n{}", "Backtest complete!".green().bold());

    Ok(())
}
