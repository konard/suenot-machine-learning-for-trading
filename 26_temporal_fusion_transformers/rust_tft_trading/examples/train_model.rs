//! Пример обучения TFT модели
//!
//! Демонстрирует полный пайплайн: загрузка данных, подготовка, обучение.
//!
//! Запуск:
//! ```bash
//! cargo run --example train_model
//! ```

use tft_trading::{
    DataLoader, DataLoaderConfig, TFTConfig, TFTModel, Trainer, TrainingConfig,
};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Инициализация логирования
    env_logger::init();

    println!("=== TFT Trading: Обучение модели ===\n");

    // 1. Конфигурация загрузчика данных
    let data_config = DataLoaderConfig {
        symbol: "BTCUSDT".to_string(),
        interval: "1h".to_string(),
        num_candles: 1000, // 1000 часовых свечей (~42 дня)
        encoder_length: 168, // 7 дней
        prediction_length: 24, // 24 часа
        use_testnet: false,
        request_delay_ms: 100,
    };

    println!("Конфигурация данных:");
    println!("  Символ: {}", data_config.symbol);
    println!("  Интервал: {}", data_config.interval);
    println!("  Количество свечей: {}", data_config.num_candles);
    println!("  Encoder length: {} часов", data_config.encoder_length);
    println!("  Prediction length: {} часов", data_config.prediction_length);
    println!();

    // 2. Загружаем данные
    println!("Загрузка данных с Bybit...");
    let loader = DataLoader::with_config(data_config.clone());
    let dataset = loader.load_from_bybit().await?;

    println!("Создан dataset:");
    println!("  Количество samples: {}", dataset.len());
    println!("  Encoder features: {}", dataset.encoder_feature_names.len());
    println!("  Decoder features: {}", dataset.decoder_feature_names.len());
    println!();

    // 3. Разбиваем на train/val/test
    let (train_data, val_data, test_data) = dataset.train_val_test_split(0.7, 0.15);

    println!("Разбиение данных:");
    println!("  Train: {} samples (70%)", train_data.len());
    println!("  Validation: {} samples (15%)", val_data.len());
    println!("  Test: {} samples (15%)", test_data.len());
    println!();

    // 4. Конфигурация модели
    let model_config = TFTConfig {
        hidden_size: 32,
        num_attention_heads: 4,
        dropout: 0.1,
        encoder_length: data_config.encoder_length,
        prediction_length: data_config.prediction_length,
        num_encoder_features: dataset.encoder_feature_names.len(),
        num_decoder_features: dataset.decoder_feature_names.len(),
        quantiles: vec![0.1, 0.5, 0.9],
        ..Default::default()
    };

    println!("Конфигурация модели:");
    println!("  Hidden size: {}", model_config.hidden_size);
    println!("  Attention heads: {}", model_config.num_attention_heads);
    println!("  Dropout: {}", model_config.dropout);
    println!("  Quantiles: {:?}", model_config.quantiles);
    println!();

    // 5. Создаем модель
    println!("Создание TFT модели...");
    let mut model = TFTModel::new(model_config);
    println!("  Количество параметров: {}", model.num_parameters());
    println!();

    // 6. Конфигурация обучения
    let training_config = TrainingConfig {
        learning_rate: 0.001,
        batch_size: 32,
        max_epochs: 10, // Небольшое количество для демонстрации
        patience: 5,
        log_every: 1,
        validate_every: 1,
        ..Default::default()
    };

    println!("Конфигурация обучения:");
    println!("  Learning rate: {}", training_config.learning_rate);
    println!("  Batch size: {}", training_config.batch_size);
    println!("  Max epochs: {}", training_config.max_epochs);
    println!("  Early stopping patience: {}", training_config.patience);
    println!();

    // 7. Обучаем модель
    println!("Начинаем обучение...\n");
    let mut trainer = Trainer::new(training_config);
    let history = trainer.train(&mut model, &train_data, &val_data);

    // 8. Оценка на тестовых данных
    println!("\n=== Оценка на тестовых данных ===");

    if !test_data.samples.is_empty() {
        // Делаем прогнозы
        let predictions = trainer.predict(&mut model, &test_data);

        // Вычисляем метрики
        let mut total_loss = 0.0;
        let mut coverage_sum = 0.0;

        for (pred, sample) in predictions.iter().zip(test_data.samples.iter()) {
            let coverage = pred.coverage(&sample.target);
            coverage_sum += coverage;

            // Простой quantile loss
            let median = pred.median();
            let mae: f64 = sample
                .target
                .iter()
                .zip(median.iter())
                .map(|(t, p)| (t - p).abs())
                .sum::<f64>()
                / sample.target.len() as f64;
            total_loss += mae;
        }

        let avg_loss = total_loss / predictions.len() as f64;
        let avg_coverage = coverage_sum / predictions.len() as f64;

        println!("Результаты на тестовой выборке:");
        println!("  Средний MAE: {:.6}", avg_loss);
        println!("  Средний coverage (80% интервал): {:.2}%", avg_coverage * 100.0);

        // Показываем пример прогноза
        if let (Some(pred), Some(sample)) = (predictions.first(), test_data.samples.first()) {
            println!("\nПример прогноза (первый sample):");
            println!("  Target (первые 5): {:?}", &sample.target.as_slice().unwrap()[..5.min(sample.target.len())]);
            println!("  Median (первые 5): {:?}", &pred.median().as_slice().unwrap()[..5.min(pred.median().len())]);
            println!(
                "  Interval width (avg): {:.4}",
                pred.interval_width().mean().unwrap_or(0.0)
            );
        }
    }

    // 9. Интерпретируемость
    println!("\n=== Интерпретируемость модели ===");

    if let Some(sample) = train_data.samples.first() {
        let _ = model.forward(sample);

        if let Some(encoder_importance) = model.get_encoder_importance() {
            println!("\nВажность encoder признаков (топ-5):");
            let mut indexed: Vec<_> = encoder_importance.iter().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

            for (i, (idx, weight)) in indexed.iter().take(5).enumerate() {
                let name = dataset
                    .encoder_feature_names
                    .get(*idx)
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");
                println!("  {}. {} - {:.4}", i + 1, name, weight);
            }
        }
    }

    println!("\n=== Готово! ===");

    Ok(())
}
