//! Демонстрация обучения WaveNet модели
//!
//! Использование:
//! ```
//! cargo run --bin train_wavenet -- --data ./data/BTCUSDT_1h.csv
//! ```

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use std::path::PathBuf;

use wavenet_trading::api::load_candles;
use wavenet_trading::analysis::{FeatureBuilder, create_windows};
use wavenet_trading::models::{WaveNet, WaveNetConfig, SimpleWaveNet};

#[derive(Parser, Debug)]
#[command(author, version, about = "Train WaveNet model for cryptocurrency prediction")]
struct Args {
    /// Path to CSV data file
    #[arg(short, long, default_value = "./data/BTCUSDT_1h.csv")]
    data: PathBuf,

    /// Window size for WaveNet input
    #[arg(short, long, default_value = "100")]
    window_size: usize,

    /// Number of training epochs (demo)
    #[arg(short, long, default_value = "10")]
    epochs: usize,

    /// Learning rate
    #[arg(short, long, default_value = "0.001")]
    learning_rate: f64,

    /// Number of hidden channels
    #[arg(long, default_value = "32")]
    hidden_channels: usize,

    /// Number of WaveNet blocks
    #[arg(long, default_value = "8")]
    num_blocks: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "=".repeat(60).blue());
    println!("{}", "WaveNet Trading - Model Training Demo".bold().blue());
    println!("{}", "=".repeat(60).blue());

    // Загружаем данные
    println!("\n{} Loading data from: {}", "📂".cyan(), args.data.display());

    let candles = if args.data.exists() {
        load_candles(&args.data)?
    } else {
        println!("{} Data file not found. Run 'fetch_data' first.", "⚠".yellow());
        println!("  Using synthetic demo data...\n");

        // Создаём синтетические данные для демонстрации
        use chrono::Utc;
        use wavenet_trading::Candle;

        (0..1000)
            .map(|i| {
                let price = 50000.0 + (i as f64 * 0.1).sin() * 1000.0 + (i as f64 * 0.01).cos() * 500.0;
                Candle {
                    timestamp: Utc::now(),
                    open: price - 10.0,
                    high: price + 50.0,
                    low: price - 50.0,
                    close: price,
                    volume: 100.0 + (i as f64 * 0.5).sin().abs() * 50.0,
                }
            })
            .collect()
    };

    println!("  {} {} candles loaded", "✓".green(), candles.len());

    // Подготавливаем признаки
    println!("\n{} Building features...", "🔧".cyan());
    let builder = FeatureBuilder::new(candles.clone());
    let mut features = builder.build_all();
    let target = builder.build_target(1); // Прогноз на 1 шаг вперёд

    println!("  {} {} features created", "✓".green(), features.num_features());

    // Нормализуем признаки
    features.fill_nan(0.0);
    features.normalize();

    // Создаём окна для обучения
    println!("\n{} Creating training windows...", "🔧".cyan());
    let (x_windows, y_targets) = create_windows(
        features.as_wavenet_input(),
        &target,
        args.window_size,
    );

    println!("  {} {} training samples", "✓".green(), x_windows.len());

    if x_windows.is_empty() {
        println!("{} Not enough data for training. Need at least {} candles.",
            "✗".red(), args.window_size + 1);
        return Ok(());
    }

    // Разделяем на train/test
    let train_size = (x_windows.len() as f64 * 0.8) as usize;
    let (train_x, test_x) = x_windows.split_at(train_size);
    let (train_y, test_y) = y_targets.split_at(train_size);

    println!("  ├─ Training samples: {}", train_x.len());
    println!("  └─ Test samples: {}", test_x.len());

    // Создаём модель WaveNet
    println!("\n{} Creating WaveNet model...", "🏗".cyan());

    let config = WaveNetConfig {
        input_channels: features.num_features(),
        residual_channels: args.hidden_channels,
        skip_channels: args.hidden_channels,
        output_channels: 1,
        kernel_size: 2,
        num_blocks: args.num_blocks,
        num_stacks: 1,
    };

    let model = WaveNet::new(config.clone());
    model.summary();

    // Также создаём упрощённую модель для сравнения
    println!("\n{} Creating Simple WaveNet for comparison...", "🏗".cyan());
    let simple_model = SimpleWaveNet::new(features.num_features(), 16, 5);
    println!("  Receptive field: {}", simple_model.receptive_field());

    // Демонстрация forward pass
    println!("\n{} Running forward pass demo...", "🎯".cyan());

    if let Some(sample) = train_x.first() {
        let prediction = model.predict(sample);
        println!("  Sample prediction: {:.6}", prediction);

        let simple_pred = simple_model.forward(sample);
        println!("  Simple model prediction: {:.6}", simple_pred);
    }

    // Демонстрация "обучения" (псевдо-обучение для демо)
    println!("\n{} Training simulation...", "📈".cyan());
    println!("  (Note: This is a forward-pass demo. Full training requires");
    println!("   gradient computation which is beyond this demo scope.)\n");

    for epoch in 1..=args.epochs {
        // Симулируем обучение
        let mut total_loss = 0.0;

        for (sample, &target_val) in train_x.iter().zip(train_y.iter()).take(100) {
            let pred = model.predict(sample);
            let loss = (pred - target_val).powi(2);
            total_loss += loss;
        }

        let avg_loss = total_loss / 100.0;
        let progress = "█".repeat(epoch * 50 / args.epochs);
        let remaining = "░".repeat(50 - epoch * 50 / args.epochs);

        print!("\r  Epoch {}/{}: [{}{}] Loss: {:.6}",
            epoch, args.epochs, progress.green(), remaining, avg_loss);

        std::io::Write::flush(&mut std::io::stdout())?;
    }
    println!();

    // Оценка на тестовых данных
    println!("\n{} Evaluating on test set...", "📊".cyan());

    let mut predictions = Vec::new();
    let mut actuals = Vec::new();

    for (sample, &target_val) in test_x.iter().zip(test_y.iter()) {
        let pred = model.predict(sample);
        predictions.push(pred);
        actuals.push(target_val);
    }

    // Вычисляем метрики
    let mse: f64 = predictions.iter()
        .zip(actuals.iter())
        .map(|(p, a)| (p - a).powi(2))
        .sum::<f64>() / predictions.len() as f64;

    let mae: f64 = predictions.iter()
        .zip(actuals.iter())
        .map(|(p, a)| (p - a).abs())
        .sum::<f64>() / predictions.len() as f64;

    // Directional accuracy
    let correct_direction: usize = predictions.iter()
        .zip(actuals.iter())
        .filter(|(p, a)| (p.signum() == a.signum()) || (p.abs() < 0.0001 && a.abs() < 0.0001))
        .count();

    let direction_accuracy = correct_direction as f64 / predictions.len() as f64;

    println!("  ├─ MSE:  {:.8}", mse);
    println!("  ├─ MAE:  {:.8}", mae);
    println!("  ├─ RMSE: {:.8}", mse.sqrt());
    println!("  └─ Direction Accuracy: {:.1}%", direction_accuracy * 100.0);

    // Советы по улучшению
    println!("\n{} Tips for improvement:", "💡".cyan());
    println!("  • Use a proper deep learning framework (PyTorch, TensorFlow)");
    println!("  • Implement backpropagation for actual training");
    println!("  • Use more training data (months/years)");
    println!("  • Tune hyperparameters (learning rate, channels, blocks)");
    println!("  • Add regularization (dropout, weight decay)");
    println!("  • Consider ensemble methods");

    println!("\n{}", "Demo completed!".green().bold());

    Ok(())
}
