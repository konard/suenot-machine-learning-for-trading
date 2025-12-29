//! Обучение CNN модели
//!
//! Пример использования:
//! ```bash
//! cargo run --release --bin train -- --data data/BTCUSDT_15_30d.csv
//! ```

use anyhow::Result;
use burn_ndarray::{NdArray, NdArrayDevice};
use cnn_crypto_trading::{
    bybit::Kline,
    data::{DataProcessor, Dataset, ProcessorConfig},
    model::{CnnConfig, CnnModel, TrainingConfig, train_model},
};
use std::path::Path;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

type Backend = NdArray<f32>;
type AutodiffBackend = burn::backend::Autodiff<Backend>;

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

    info!("Loading data from {}", data_path);

    // Загружаем данные
    let klines = load_klines_from_csv(&data_path)?;
    info!("Loaded {} klines", klines.len());

    if klines.len() < 200 {
        anyhow::bail!("Not enough data for training (need at least 200 klines)");
    }

    // Создаём процессор данных
    let processor_config = ProcessorConfig {
        window_size: 60,
        prediction_horizon: 4,
        classification_threshold: 0.3,
        use_log_returns: true,
        normalize_window: true,
    };

    let processor = DataProcessor::with_config(processor_config);

    // Создаём образцы
    info!("Creating samples...");
    let samples = processor.create_normalized_samples(&klines);
    info!("Created {} samples", samples.len());

    if samples.is_empty() {
        anyhow::bail!("No samples created from the data");
    }

    // Выводим информацию о первом образце
    let first = &samples[0];
    info!(
        "Sample shape: channels={}, window={}",
        first.num_channels(),
        first.window_size()
    );

    // Разделяем на train/test
    let (train_samples, test_samples) = DataProcessor::train_test_split(samples, 0.2);
    info!(
        "Train samples: {}, Test samples: {}",
        train_samples.len(),
        test_samples.len()
    );

    // Создаём датасеты
    let batch_size = 32;
    let mut train_dataset = Dataset::new(train_samples, batch_size);
    let mut test_dataset = Dataset::new(test_samples, batch_size).without_shuffle();

    // Выводим распределение классов
    let train_dist = train_dataset.class_distribution();
    let test_dist = test_dataset.class_distribution();
    info!(
        "Train class distribution: Down={}, Neutral={}, Up={}",
        train_dist[0], train_dist[1], train_dist[2]
    );
    info!(
        "Test class distribution: Down={}, Neutral={}, Up={}",
        test_dist[0], test_dist[1], test_dist[2]
    );

    // Конфигурация модели
    let cnn_config = CnnConfig {
        in_channels: 10,
        input_size: 60,
        num_classes: 3,
        conv1_filters: 32,
        conv2_filters: 64,
        conv3_filters: 128,
        kernel_size: 3,
        pool_size: 2,
        fc_size: 64,
        dropout: 0.3,
    };

    // Конфигурация обучения
    let training_config = TrainingConfig {
        num_epochs: 50,
        batch_size,
        learning_rate: 0.001,
        lr_decay: 0.95,
        patience: 10,
        min_delta: 0.001,
        validation_split: 0.2,
        use_class_weights: true,
        checkpoint_path: Some("model_checkpoint".to_string()),
        log_interval: 10,
    };

    info!("Creating model...");
    let device = NdArrayDevice::default();
    let model: CnnModel<AutodiffBackend> = CnnModel::new(&device, &cnn_config);

    info!("Starting training...");
    let (trained_model, result) = train_model(
        model,
        &mut train_dataset,
        &mut test_dataset,
        &training_config,
        &device,
    );

    // Выводим результаты
    println!("\n=== Training Results ===");
    println!("Best epoch: {}", result.best_epoch);
    println!("Best validation accuracy: {:.4}", result.best_accuracy);
    println!("Final train loss: {:.4}", result.train_losses.last().unwrap_or(&0.0));
    println!("Final val loss: {:.4}", result.val_losses.last().unwrap_or(&0.0));

    // Сохраняем историю обучения
    let history_path = Path::new("training_history.csv");
    let mut writer = csv::Writer::from_path(history_path)?;
    writer.write_record(&["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])?;

    for i in 0..result.train_losses.len() {
        writer.write_record(&[
            (i + 1).to_string(),
            result.train_losses[i].to_string(),
            result.val_losses[i].to_string(),
            result.train_accuracies[i].to_string(),
            result.val_accuracies[i].to_string(),
        ])?;
    }
    writer.flush()?;

    info!("Training history saved to {:?}", history_path);

    Ok(())
}

/// Загрузка свечей из CSV файла
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
