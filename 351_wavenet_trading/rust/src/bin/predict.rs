//! Генерация прогнозов с помощью WaveNet
//!
//! Использование:
//! ```
//! cargo run --bin predict -- --symbol BTCUSDT
//! ```

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use std::path::PathBuf;

use wavenet_trading::api::{BybitClient, Interval, load_candles};
use wavenet_trading::analysis::FeatureBuilder;
use wavenet_trading::models::{SimpleWaveNet, WaveNet, WaveNetConfig};
use wavenet_trading::Signal;

#[derive(Parser, Debug)]
#[command(author, version, about = "Generate predictions using WaveNet model")]
struct Args {
    /// Trading symbol
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Path to historical data CSV (optional, will fetch if not provided)
    #[arg(short, long)]
    data: Option<PathBuf>,

    /// Window size for prediction
    #[arg(short, long, default_value = "100")]
    window_size: usize,

    /// Fetch live data from Bybit
    #[arg(long, default_value = "false")]
    live: bool,

    /// Signal threshold
    #[arg(long, default_value = "0.001")]
    threshold: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "=".repeat(60).blue());
    println!("{}", "WaveNet Trading - Prediction Engine".bold().blue());
    println!("{}", "=".repeat(60).blue());

    // Получаем данные
    let candles = if let Some(data_path) = &args.data {
        println!("\n{} Loading data from: {}", "📂".cyan(), data_path.display());
        load_candles(data_path)?
    } else if args.live {
        println!("\n{} Fetching live data from Bybit...", "🌐".cyan());
        let client = BybitClient::new();
        client.get_klines(&args.symbol, Interval::Hour1, 200).await?
    } else {
        println!("\n{} Using demo data (use --data or --live for real data)", "📊".cyan());

        // Генерируем демо-данные
        use chrono::Utc;
        use wavenet_trading::Candle;

        (0..200)
            .map(|i| {
                let t = i as f64 * 0.1;
                let price = 50000.0 + t.sin() * 1000.0 + (t * 2.5).cos() * 500.0;
                Candle {
                    timestamp: Utc::now(),
                    open: price - 10.0,
                    high: price + 50.0,
                    low: price - 50.0,
                    close: price,
                    volume: 100.0,
                }
            })
            .collect()
    };

    println!("  {} {} candles available", "✓".green(), candles.len());

    if candles.len() < args.window_size {
        println!("{} Not enough data. Need at least {} candles.",
            "✗".red(), args.window_size);
        return Ok(());
    }

    // Текущая цена
    let current_price = candles.last().unwrap().close;
    println!("\n{} Current {} price: ${:.2}", "💰".cyan(), args.symbol, current_price);

    // Подготавливаем признаки
    println!("\n{} Building features...", "🔧".cyan());
    let builder = FeatureBuilder::new(candles.clone());
    let mut features = builder.build_all();

    println!("  {} {} features extracted", "✓".green(), features.num_features());

    // Нормализуем
    features.fill_nan(0.0);
    features.normalize();

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
    println!("  Receptive field: {} timesteps", model.config.receptive_field());

    // Также создаём простую модель
    let simple_model = SimpleWaveNet::new(features.num_features(), 16, 5);

    // Получаем последнее окно для предсказания
    let input_data = features.as_wavenet_input();
    let window: Vec<Vec<f64>> = input_data
        .iter()
        .map(|ch| ch[ch.len() - args.window_size..].to_vec())
        .collect();

    // Генерируем предсказания
    println!("\n{} Generating predictions...", "🎯".cyan());

    let wavenet_pred = model.predict(&window);
    let simple_pred = simple_model.forward(&window);

    println!("\n  {} WaveNet Predictions:", "📈".green().bold());
    println!("  ├─ WaveNet:       {:.6} ({:.4}%)", wavenet_pred, wavenet_pred * 100.0);
    println!("  └─ Simple WaveNet: {:.6} ({:.4}%)", simple_pred, simple_pred * 100.0);

    // Генерируем сигналы
    let avg_pred = (wavenet_pred + simple_pred) / 2.0;

    let signal = if avg_pred > args.threshold {
        Signal::Buy
    } else if avg_pred < -args.threshold {
        Signal::Sell
    } else {
        Signal::Hold
    };

    println!("\n{}", "=".repeat(60).yellow());
    println!("{} TRADING SIGNAL: {}", "🚦".yellow(),
        match signal {
            Signal::Buy => "BUY".green().bold(),
            Signal::Sell => "SELL".red().bold(),
            Signal::Hold => "HOLD".yellow().bold(),
        }
    );
    println!("{}", "=".repeat(60).yellow());

    // Дополнительная информация
    println!("\n{} Analysis:", "📊".cyan());

    // Прогнозируемая цена
    let predicted_price = current_price * (1.0 + avg_pred);
    let price_change = predicted_price - current_price;
    let price_change_str = if price_change >= 0.0 {
        format!("+${:.2}", price_change).green()
    } else {
        format!("-${:.2}", price_change.abs()).red()
    };

    println!("  ├─ Current price:   ${:.2}", current_price);
    println!("  ├─ Predicted price: ${:.2} ({})", predicted_price, price_change_str);
    println!("  ├─ Confidence:      {:.1}%", (avg_pred.abs() / args.threshold * 100.0).min(100.0));
    println!("  └─ Threshold:       ±{:.2}%", args.threshold * 100.0);

    // Предупреждения
    println!("\n{} Important Notes:", "⚠".yellow());
    println!("  • This is a DEMO prediction without real training");
    println!("  • Model weights are randomly initialized");
    println!("  • DO NOT use for real trading decisions");
    println!("  • Past performance does not guarantee future results");

    // Исторические предсказания
    println!("\n{} Recent predictions history:", "📜".cyan());

    let num_history = 10.min(candles.len() - args.window_size);
    println!("  {:^10} {:^12} {:^10} {:^10}", "Index", "Predicted", "Actual", "Correct");
    println!("  {}", "-".repeat(44));

    for i in 0..num_history {
        let start = candles.len() - args.window_size - num_history + i;
        let end = start + args.window_size;

        let hist_window: Vec<Vec<f64>> = input_data
            .iter()
            .map(|ch| ch[start..end].to_vec())
            .collect();

        let pred = model.predict(&hist_window);

        // Actual return (if available)
        let actual = if end < candles.len() {
            let actual_ret = (candles[end].close - candles[end - 1].close) / candles[end - 1].close;
            let correct = (pred.signum() == actual_ret.signum()) || (pred.abs() < 0.0001 && actual_ret.abs() < 0.0001);
            let mark = if correct { "✓".green() } else { "✗".red() };
            format!("{:>9.4}%  {}", actual_ret * 100.0, mark)
        } else {
            "    N/A".to_string()
        };

        println!("  {:^10} {:>9.4}%   {}", start, pred * 100.0, actual);
    }

    println!("\n{}", "Prediction complete!".green().bold());

    Ok(())
}
