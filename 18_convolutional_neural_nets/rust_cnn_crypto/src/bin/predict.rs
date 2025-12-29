//! Генерация предсказаний с обученной моделью
//!
//! Пример использования:
//! ```bash
//! cargo run --release --bin predict -- --symbol BTCUSDT
//! ```

use anyhow::Result;
use burn_ndarray::{NdArray, NdArrayDevice};
use cnn_crypto_trading::{
    bybit::{BybitClient, Kline, KlineInterval},
    data::{DataProcessor, ProcessorConfig, Sample},
    model::{CnnConfig, CnnModel},
    trading::{Signal, SignalDirection},
};
use burn::tensor::Tensor;
use chrono::Utc;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

type Backend = NdArray<f32>;

#[tokio::main]
async fn main() -> Result<()> {
    // Настройка логирования
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Параметры
    let symbol = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "BTCUSDT".to_string());

    info!("Generating prediction for {}", symbol);

    // Загружаем последние данные
    let client = BybitClient::new();
    let klines = client
        .get_klines(&symbol, KlineInterval::Min15, Some(100), None, None)
        .await?;

    info!("Fetched {} klines", klines.len());

    if klines.len() < 60 {
        anyhow::bail!("Not enough data for prediction (need at least 60 klines)");
    }

    // Создаём процессор
    let processor_config = ProcessorConfig {
        window_size: 60,
        prediction_horizon: 4,
        classification_threshold: 0.3,
        use_log_returns: true,
        normalize_window: true,
    };

    let processor = DataProcessor::with_config(processor_config);

    // Создаём признаки для последнего окна
    let features = processor.klines_to_features(&klines);
    let n = features.shape()[1];

    if n < 60 {
        anyhow::bail!("Not enough features");
    }

    // Берём последние 60 точек
    let window_start = n - 60;
    let window_features = features.slice(ndarray::s![.., window_start..n]).to_owned();

    // Нормализуем
    let mut sample = Sample::new(
        window_features.mapv(|x| x as f32),
        klines.last().unwrap().timestamp,
    );
    processor.normalize_sample(&mut sample);

    // Создаём модель (в реальном приложении загружаем обученную модель)
    let cnn_config = CnnConfig::default();
    let device = NdArrayDevice::default();
    let model: CnnModel<Backend> = CnnModel::new(&device, &cnn_config);

    // Преобразуем в тензор [1, channels, seq]
    let input_data: Vec<f32> = sample.features.iter().cloned().collect();
    let input = Tensor::<Backend, 3>::from_data(
        burn::tensor::TensorData::new(input_data, [1, 10, 60]),
        &device,
    );

    // Получаем предсказание
    let logits = model.forward(input);
    let probs = burn::tensor::activation::softmax(logits.clone(), 1);

    let probs_data: Vec<f32> = probs.into_data().to_vec().unwrap();
    let prediction = logits.argmax(1);
    let pred_class: i32 = prediction.into_data().to_vec::<i32>().unwrap()[0];

    // Создаём сигнал
    let current_price = klines.last().unwrap().close;
    let probabilities = [
        probs_data[0] as f64,
        probs_data[1] as f64,
        probs_data[2] as f64,
    ];

    let signal = Signal::new(
        Utc::now().timestamp_millis(),
        probabilities,
        current_price,
        0.5,
    );

    // Выводим результат
    println!("\n=== Prediction for {} ===", symbol);
    println!("Current time: {}", Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
    println!("Current price: ${:.2}", current_price);
    println!();
    println!("Probabilities:");
    println!("  Down:    {:.2}%", probabilities[0] * 100.0);
    println!("  Neutral: {:.2}%", probabilities[1] * 100.0);
    println!("  Up:      {:.2}%", probabilities[2] * 100.0);
    println!();
    println!("Predicted class: {} ({})",
        pred_class,
        match pred_class {
            0 => "DOWN",
            1 => "NEUTRAL",
            2 => "UP",
            _ => "UNKNOWN",
        }
    );
    println!();
    println!("Signal:");
    println!("  Direction: {:?}", signal.direction);
    println!("  Strength: {:.2}%", signal.strength * 100.0);

    if signal.direction != SignalDirection::Neutral {
        println!();
        println!("RECOMMENDATION: {:?} with {:.0}% confidence",
            signal.direction,
            signal.strength * 100.0
        );
    } else {
        println!();
        println!("RECOMMENDATION: Hold / No action");
    }

    // Предупреждение
    println!();
    println!("NOTE: This is a demonstration with an untrained model.");
    println!("      For real trading, train the model first using:");
    println!("      cargo run --release --bin train");

    Ok(())
}
