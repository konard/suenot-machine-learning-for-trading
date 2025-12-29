//! Пример прогнозирования с TFT
//!
//! Демонстрирует использование обученной модели для прогнозирования.
//!
//! Запуск:
//! ```bash
//! cargo run --example predict
//! ```

use tft_trading::{
    BybitClient, DataLoader, DataLoaderConfig, TFTConfig, TFTModel,
    FeatureExtractor, TimeSeriesDataset, TimeSeriesDatasetConfig,
};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    println!("=== TFT Trading: Прогнозирование ===\n");

    // 1. Загружаем последние данные
    let client = BybitClient::new();
    let symbol = "BTCUSDT";

    println!("Загрузка последних данных для {}...", symbol);
    let klines = client.get_klines(symbol, "1h", 200).await?;
    println!("Загружено {} свечей\n", klines.len());

    // Последняя цена
    let current_price = klines.last().map(|k| k.close).unwrap_or(0.0);
    println!("Текущая цена: ${:.2}\n", current_price);

    // 2. Подготавливаем признаки
    let extractor = FeatureExtractor::default();
    let mut features = extractor.extract(&klines);
    features.normalize();

    println!("Извлечено {} признаков", features.num_features());
    println!("Признаки: {:?}\n", &features.names[..5]);

    // 3. Создаем dataset для прогноза
    let encoder_length = 168; // 7 дней
    let prediction_length = 24; // 24 часа

    let dataset_config = TimeSeriesDatasetConfig {
        encoder_length,
        prediction_length,
        target_idx: 4, // returns
        known_future_indices: vec![20, 21, 22, 23],
        static_indices: vec![],
        step: 1,
    };

    let ts_dataset = TimeSeriesDataset::new(features.clone(), dataset_config);

    // Берем последний sample для прогноза
    let last_sample = ts_dataset.get_sample(ts_dataset.num_samples() - 1)
        .ok_or("No samples available")?;

    println!("Подготовлен sample для прогноза:");
    println!("  Encoder input shape: {:?}", last_sample.encoder_input.shape());
    println!("  Decoder input shape: {:?}", last_sample.decoder_input.shape());
    println!();

    // 4. Создаем модель
    let model_config = TFTConfig {
        hidden_size: 32,
        num_attention_heads: 4,
        encoder_length,
        prediction_length,
        num_encoder_features: features.num_features(),
        num_decoder_features: 4,
        quantiles: vec![0.1, 0.25, 0.5, 0.75, 0.9],
        ..Default::default()
    };

    let mut model = TFTModel::new(model_config);
    println!("Модель создана ({} параметров)\n", model.num_parameters());

    // 5. Делаем прогноз
    println!("Генерация прогноза на {} часов вперед...\n", prediction_length);
    let prediction = model.forward(&last_sample);

    // 6. Выводим результаты
    let q10 = prediction.lower();
    let q50 = prediction.median();
    let q90 = prediction.upper();

    println!("=== Прогноз доходности ===\n");
    println!("{:>6} {:>12} {:>12} {:>12} {:>12}",
        "Hour", "Q10 (%)", "Q50 (%)", "Q90 (%)", "Width (%)");
    println!("{}", "-".repeat(60));

    for i in 0..prediction_length.min(12) {
        println!(
            "{:>6} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
            i + 1,
            q10[i] * 100.0,
            q50[i] * 100.0,
            q90[i] * 100.0,
            (q90[i] - q10[i]) * 100.0
        );
    }

    if prediction_length > 12 {
        println!("... (еще {} часов)", prediction_length - 12);
    }

    // 7. Конвертируем в цены
    println!("\n=== Прогноз цены ===\n");
    println!("{:>6} {:>14} {:>14} {:>14}",
        "Hour", "Lower ($)", "Expected ($)", "Upper ($)");
    println!("{}", "-".repeat(56));

    let mut expected_price = current_price;
    for i in 0..prediction_length.min(12) {
        let cumulative_return: f64 = q50.iter().take(i + 1).sum();
        let lower_return: f64 = q10.iter().take(i + 1).sum();
        let upper_return: f64 = q90.iter().take(i + 1).sum();

        let expected = current_price * (1.0 + cumulative_return);
        let lower = current_price * (1.0 + lower_return);
        let upper = current_price * (1.0 + upper_return);

        println!(
            "{:>6} {:>14.2} {:>14.2} {:>14.2}",
            i + 1, lower, expected, upper
        );
    }

    // 8. Интерпретация
    println!("\n=== Интерпретация ===\n");

    // Важность признаков
    if let Some(importance) = model.get_encoder_importance() {
        println!("Топ-5 важных признаков:");
        let mut indexed: Vec<_> = importance.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        for (i, (idx, weight)) in indexed.iter().take(5).enumerate() {
            let name = features.names.get(*idx).map(|s| s.as_str()).unwrap_or("?");
            println!("  {}. {} ({:.2}%)", i + 1, name, weight * 100.0);
        }
    }

    // 9. Торговый сигнал
    println!("\n=== Торговый сигнал ===\n");

    let avg_return = q50.mean().unwrap_or(0.0);
    let confidence_width = (q90[0] - q10[0]).abs();

    let signal = if q10[0] > 0.005 {
        "STRONG LONG"
    } else if q90[0] < -0.005 {
        "STRONG SHORT"
    } else if avg_return > 0.002 && confidence_width < 0.02 {
        "LONG"
    } else if avg_return < -0.002 && confidence_width < 0.02 {
        "SHORT"
    } else {
        "HOLD"
    };

    println!("Сигнал: {}", signal);
    println!("Ожидаемая доходность (24h): {:.2}%", avg_return * 100.0);
    println!("Уверенность: {:.2}%", (1.0 - confidence_width) * 100.0);

    println!("\n=== Готово! ===");

    Ok(())
}
