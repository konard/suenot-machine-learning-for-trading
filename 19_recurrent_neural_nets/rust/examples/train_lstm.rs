//! Пример: Обучение LSTM модели
//!
//! Этот пример демонстрирует:
//! - Загрузку данных из CSV
//! - Подготовку последовательностей
//! - Обучение LSTM модели
//! - Сохранение обученной модели
//!
//! Запуск: cargo run --example train_lstm --release

use crypto_rnn::data::BybitClient;
use crypto_rnn::model::{LSTMConfig, LSTM};
use crypto_rnn::preprocessing::DataProcessor;
use crypto_rnn::utils::{load_candles_csv, mse, rmse, mae, r2_score};
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("=== Обучение LSTM модели для прогнозирования криптовалют ===\n");

    // Параметры
    let symbol = env::var("SYMBOL").unwrap_or_else(|_| "BTCUSDT".to_string());
    let interval = env::var("INTERVAL").unwrap_or_else(|_| "1h".to_string());
    let sequence_length: usize = env::var("SEQ_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let epochs: usize = env::var("EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let hidden_size: usize = env::var("HIDDEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let learning_rate: f64 = env::var("LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.001);

    println!("Параметры:");
    println!("  Символ: {}", symbol);
    println!("  Интервал: {}", interval);
    println!("  Длина последовательности: {}", sequence_length);
    println!("  Размер скрытого слоя: {}", hidden_size);
    println!("  Скорость обучения: {}", learning_rate);
    println!("  Эпохи: {}", epochs);
    println!();

    // Пытаемся загрузить данные из файла, иначе скачиваем
    let data_file = format!("data/{}_{}.csv", symbol.to_lowercase(), interval);
    let candles = if std::path::Path::new(&data_file).exists() {
        println!("Загружаем данные из {}...", data_file);
        load_candles_csv(&data_file)?
    } else {
        println!("Скачиваем данные с Bybit...");
        let client = BybitClient::new();
        client.get_klines(&symbol, &interval, 1000).await?
    };

    println!("Загружено {} свечей\n", candles.len());

    // Подготавливаем данные
    println!("Подготовка данных...");
    let mut processor = DataProcessor::new(sequence_length, 1);
    let (x, y) = processor.prepare_sequences(&candles)?;

    println!("  X shape: [{}, {}, {}]", x.shape()[0], x.shape()[1], x.shape()[2]);
    println!("  y shape: [{}, {}]", y.shape()[0], y.shape()[1]);

    // Разделяем на train/test
    let (x_train, x_test, y_train, y_test) = processor.train_test_split(&x, &y, 0.8);

    println!(
        "  Train samples: {}, Test samples: {}",
        x_train.shape()[0],
        x_test.shape()[0]
    );
    println!();

    // Создаём модель
    let input_size = x.shape()[2];
    let config = LSTMConfig::new(input_size, hidden_size, 1)
        .with_learning_rate(learning_rate)
        .with_batch_size(32);

    println!("Создаём LSTM модель...");
    println!("  Входной размер: {}", input_size);
    println!("  Скрытый размер: {}", hidden_size);
    println!("  Выходной размер: 1");
    println!();

    let mut lstm = LSTM::from_config(config);

    // Обучаем
    println!("Начинаем обучение...\n");
    lstm.train(&x_train, &y_train, epochs, learning_rate)?;

    // Оцениваем
    println!("\n=== Оценка модели ===\n");

    let train_pred = lstm.predict(&x_train);
    let test_pred = lstm.predict(&x_test);

    println!("Метрики на обучающей выборке:");
    println!("  MSE:  {:.6}", mse(&y_train, &train_pred));
    println!("  RMSE: {:.6}", rmse(&y_train, &train_pred));
    println!("  MAE:  {:.6}", mae(&y_train, &train_pred));
    println!("  R²:   {:.4}", r2_score(&y_train, &train_pred));

    println!("\nМетрики на тестовой выборке:");
    println!("  MSE:  {:.6}", mse(&y_test, &test_pred));
    println!("  RMSE: {:.6}", rmse(&y_test, &test_pred));
    println!("  MAE:  {:.6}", mae(&y_test, &test_pred));
    println!("  R²:   {:.4}", r2_score(&y_test, &test_pred));

    // Показываем последние предсказания
    println!("\nПоследние 5 предсказаний (тестовая выборка):");
    println!("{:>12} {:>12} {:>12}", "Факт", "Прогноз", "Ошибка");
    println!("{}", "-".repeat(40));

    let n = y_test.shape()[0];
    for i in (n.saturating_sub(5))..n {
        let actual = y_test[[i, 0]];
        let predicted = test_pred[[i, 0]];
        let error = actual - predicted;
        println!("{:>12.4} {:>12.4} {:>12.4}", actual, predicted, error);
    }

    // Сохраняем модель
    std::fs::create_dir_all("models")?;
    let model_path = format!("models/lstm_{}_{}.bin", symbol.to_lowercase(), interval);
    lstm.save(&model_path)?;
    println!("\nМодель сохранена в: {}", model_path);

    // Сохраняем процессор
    let processor_path = format!("models/processor_{}_{}.bin", symbol.to_lowercase(), interval);
    processor.save(&processor_path)?;
    println!("Процессор сохранён в: {}", processor_path);

    // Показываем историю потерь
    if !lstm.loss_history.is_empty() {
        println!("\nИстория потерь:");
        let first_loss = lstm.loss_history.first().unwrap();
        let last_loss = lstm.loss_history.last().unwrap();
        let improvement = (first_loss - last_loss) / first_loss * 100.0;
        println!(
            "  Начальная потеря: {:.6}, Конечная: {:.6}, Улучшение: {:.2}%",
            first_loss, last_loss, improvement
        );
    }

    println!("\nОбучение завершено!");

    Ok(())
}
