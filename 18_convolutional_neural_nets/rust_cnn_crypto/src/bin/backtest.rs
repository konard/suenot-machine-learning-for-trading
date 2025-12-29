//! Бэктестинг торговой стратегии
//!
//! Пример использования:
//! ```bash
//! cargo run --release --bin backtest -- --data data/BTCUSDT_15_30d.csv
//! ```

use anyhow::Result;
use burn_ndarray::{NdArray, NdArrayDevice};
use cnn_crypto_trading::{
    bybit::Kline,
    data::{DataProcessor, Dataset, ProcessorConfig},
    model::{CnnConfig, CnnModel},
    trading::{Backtest, Signal, StrategyConfig},
};
use burn::tensor::Tensor;
use std::path::Path;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

type Backend = NdArray<f32>;

fn main() -> Result<()> {
    // Настройка логирования
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Параметры
    let data_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "data/BTCUSDT_15_30d.csv".to_string());

    let initial_capital: f64 = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10000.0);

    info!("Loading data from {}", data_path);

    // Загружаем данные
    let klines = load_klines_from_csv(&data_path)?;
    info!("Loaded {} klines", klines.len());

    if klines.len() < 200 {
        anyhow::bail!("Not enough data for backtesting");
    }

    // Создаём процессор
    let processor_config = ProcessorConfig {
        window_size: 60,
        prediction_horizon: 4,
        classification_threshold: 0.3,
        use_log_returns: true,
        normalize_window: true,
    };

    let processor = DataProcessor::with_config(processor_config.clone());

    // Создаём образцы
    info!("Creating samples...");
    let samples = processor.create_normalized_samples(&klines);
    info!("Created {} samples", samples.len());

    // Создаём модель (в реальном приложении загружаем обученную)
    let cnn_config = CnnConfig::default();
    let device = NdArrayDevice::default();
    let model: CnnModel<Backend> = CnnModel::new(&device, &cnn_config);

    // Генерируем сигналы для каждого образца
    info!("Generating signals...");
    let mut signals = Vec::new();

    for sample in &samples {
        // Преобразуем в тензор
        let input_data: Vec<f32> = sample.features.iter().cloned().collect();
        let input = Tensor::<Backend, 3>::from_data(
            burn::tensor::TensorData::new(
                input_data,
                [1, sample.num_channels(), sample.window_size()],
            ),
            &device,
        );

        // Получаем предсказание
        let logits = model.forward(input);
        let probs = burn::tensor::activation::softmax(logits, 1);
        let probs_data: Vec<f32> = probs.into_data().to_vec().unwrap();

        // Находим соответствующую цену
        let price = klines
            .iter()
            .find(|k| k.timestamp == sample.timestamp)
            .map(|k| k.close)
            .unwrap_or(0.0);

        if price > 0.0 {
            let signal = Signal::new(
                sample.timestamp,
                [
                    probs_data[0] as f64,
                    probs_data[1] as f64,
                    probs_data[2] as f64,
                ],
                price,
                0.5,
            );
            signals.push(signal);
        }
    }

    info!("Generated {} signals", signals.len());

    // Конфигурация стратегии
    let strategy_config = StrategyConfig {
        min_signal_strength: 0.6,
        probability_threshold: 0.5,
        position_size: 0.2,
        stop_loss_pct: 2.0,
        take_profit_pct: 4.0,
        max_positions: 1,
        commission_rate: 0.001,
        min_trade_interval: 3600000, // 1 час
    };

    // Запускаем бэктест
    info!("Running backtest...");
    let backtest = Backtest::new("BTCUSDT", initial_capital, strategy_config);
    let result = backtest.run(&klines, &signals);

    // Выводим результаты
    println!("\n{}", result);

    // Сохраняем equity curve
    let output_path = Path::new("backtest_results.csv");
    backtest.export_equity_curve(&result, output_path.to_str().unwrap())?;
    info!("Equity curve saved to {:?}", output_path);

    // Визуализация в консоли
    println!("\n=== Equity Curve (sampled) ===");
    let step = result.equity_curve.len() / 20;
    for (i, (ts, equity)) in result.equity_curve.iter().enumerate() {
        if i % step.max(1) == 0 {
            let datetime = chrono::DateTime::from_timestamp_millis(*ts)
                .map(|dt| dt.format("%m-%d %H:%M").to_string())
                .unwrap_or_default();
            let bar_len = ((equity / initial_capital - 0.8) * 50.0).max(0.0).min(50.0) as usize;
            println!("{}: {:>10.2} |{}", datetime, equity, "#".repeat(bar_len));
        }
    }

    // Предупреждение
    println!("\nNOTE: This backtest uses an UNTRAINED model for demonstration.");
    println!("      For meaningful results, first train the model with:");
    println!("      cargo run --release --bin train");

    Ok(())
}

/// Загрузка свечей из CSV
fn load_klines_from_csv(path: &str) -> Result<Vec<Kline>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut klines = Vec::new();

    for result in reader.records() {
        let record = result?;

        let kline = Kline {
            timestamp: record[0].parse()?,
            open: record[2].parse()?,
            high: record[3].parse()?,
            low: record[4].parse()?,
            close: record[5].parse()?,
            volume: record[6].parse()?,
            turnover: record[7].parse()?,
        };

        klines.push(kline);
    }

    Ok(klines)
}
