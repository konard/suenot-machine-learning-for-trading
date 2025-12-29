//! Пример: Прогнозирование с обученной моделью
//!
//! Этот пример демонстрирует:
//! - Загрузку обученной модели
//! - Получение свежих данных
//! - Генерацию прогноза
//!
//! Запуск: cargo run --example predict

use crypto_rnn::data::BybitClient;
use crypto_rnn::model::LSTM;
use crypto_rnn::preprocessing::DataProcessor;
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("=== Прогнозирование с обученной LSTM моделью ===\n");

    // Параметры
    let symbol = env::var("SYMBOL").unwrap_or_else(|_| "BTCUSDT".to_string());
    let interval = env::var("INTERVAL").unwrap_or_else(|_| "1h".to_string());

    // Пути к файлам
    let model_path = format!("models/lstm_{}_{}.bin", symbol.to_lowercase(), interval);
    let processor_path = format!("models/processor_{}_{}.bin", symbol.to_lowercase(), interval);

    // Проверяем наличие модели
    if !std::path::Path::new(&model_path).exists() {
        println!("Модель не найдена: {}", model_path);
        println!("Сначала обучите модель: cargo run --example train_lstm --release");
        return Ok(());
    }

    // Загружаем модель и процессор
    println!("Загружаем модель из {}...", model_path);
    let mut lstm = LSTM::load(&model_path)?;
    let processor = DataProcessor::load(&processor_path)?;

    println!("  Длина последовательности: {}", processor.sequence_length);
    println!();

    // Получаем свежие данные
    println!("Получаем актуальные данные с Bybit...");
    let client = BybitClient::new();
    let candles = client
        .get_klines(&symbol, &interval, (processor.sequence_length + 10) as u32)
        .await?;

    println!("Получено {} свечей\n", candles.len());

    // Показываем текущую цену
    if let Some(last) = candles.last() {
        println!("Текущая ситуация:");
        println!("  Время: {}", last.datetime());
        println!("  Цена: ${:.2}", last.close);
        println!("  Изменение: {:.2}%", last.price_change_pct());
        println!();
    }

    // Подготавливаем данные для прогноза
    let x = processor.prepare_single(&candles)?;

    // Делаем прогноз
    let prediction = lstm.predict(&x);
    let predicted_normalized = prediction[[0, 0]];

    // Преобразуем обратно в реальную цену
    let predicted_price = processor.inverse_transform_price(predicted_normalized);

    // Вычисляем ожидаемое изменение
    let current_price = candles.last().unwrap().close;
    let change = predicted_price - current_price;
    let change_pct = change / current_price * 100.0;

    println!("=== ПРОГНОЗ ===\n");
    println!("Текущая цена:    ${:.2}", current_price);
    println!("Прогноз:         ${:.2}", predicted_price);
    println!("Изменение:       ${:.2} ({:+.2}%)", change, change_pct);
    println!();

    // Интерпретация
    let signal = if change_pct > 1.0 {
        "СИЛЬНЫЙ РОСТ 📈"
    } else if change_pct > 0.3 {
        "Умеренный рост ↗️"
    } else if change_pct > -0.3 {
        "Боковое движение ↔️"
    } else if change_pct > -1.0 {
        "Умеренное падение ↘️"
    } else {
        "СИЛЬНОЕ ПАДЕНИЕ 📉"
    };

    println!("Сигнал: {}", signal);
    println!();

    // Показываем последние свечи для контекста
    println!("Последние 5 свечей:");
    println!(
        "{:<20} {:>12} {:>12} {:>10}",
        "Время", "Close", "Volume", "Изм.%"
    );
    println!("{}", "-".repeat(60));

    for candle in candles.iter().rev().take(5).rev() {
        println!(
            "{:<20} {:>12.2} {:>12.0} {:>10.2}%",
            candle.datetime().format("%Y-%m-%d %H:%M"),
            candle.close,
            candle.volume,
            candle.price_change_pct()
        );
    }

    println!();
    println!("Отказ от ответственности: Это учебный пример.");
    println!("НЕ используйте для реальной торговли!");

    Ok(())
}
