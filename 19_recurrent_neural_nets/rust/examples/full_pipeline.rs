//! Пример: Полный пайплайн от данных до прогноза
//!
//! Этот пример демонстрирует весь процесс:
//! 1. Получение данных с Bybit
//! 2. Предобработка и создание признаков
//! 3. Обучение LSTM модели
//! 4. Оценка и прогнозирование
//!
//! Запуск: cargo run --example full_pipeline --release

use crypto_rnn::data::BybitClient;
use crypto_rnn::model::{LSTMConfig, LSTM, GRU};
use crypto_rnn::preprocessing::{DataProcessor, FeatureExtractor};
use crypto_rnn::utils::{mse, rmse, mae, r2_score, save_candles_csv};
use ndarray::Array1;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Crypto RNN - Полный пайплайн машинного обучения         ║");
    println!("║  Прогнозирование криптовалют с помощью LSTM/GRU          ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // =====================
    // ЭТАП 1: Сбор данных
    // =====================
    println!("📊 ЭТАП 1: Сбор данных\n");

    let client = BybitClient::new();

    // Получаем данные для нескольких криптовалют
    let symbols = vec!["BTCUSDT", "ETHUSDT"];
    let interval = "1h";
    let limit = 500;

    for symbol in &symbols {
        println!("Загружаем {} ({})...", symbol, interval);
        match client.get_klines(symbol, interval, limit).await {
            Ok(candles) => {
                println!("  ✓ Получено {} свечей", candles.len());

                if let Some(last) = candles.last() {
                    println!("  ✓ Последняя цена: ${:.2}", last.close);
                }

                // Сохраняем данные
                std::fs::create_dir_all("data")?;
                let path = format!("data/{}_{}.csv", symbol.to_lowercase(), interval);
                save_candles_csv(&candles, &path)?;
            }
            Err(e) => {
                println!("  ✗ Ошибка: {}", e);
            }
        }
    }

    // Работаем с BTC для обучения
    println!("\n📈 Загружаем BTCUSDT для обучения...");
    let candles = client.get_klines("BTCUSDT", interval, limit).await?;
    println!("Получено {} свечей\n", candles.len());

    // =====================
    // ЭТАП 2: Анализ данных
    // =====================
    println!("🔍 ЭТАП 2: Анализ данных\n");

    // Извлекаем признаки
    let returns = FeatureExtractor::extract_returns(&candles);

    // Статистика
    let mean_return: f64 = returns.sum() / returns.len() as f64;
    let volatility: f64 = {
        let variance: f64 = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>()
            / returns.len() as f64;
        variance.sqrt() * (252.0_f64).sqrt() // Годовая волатильность
    };

    // Тренд
    let first_close = candles.first().map(|c| c.close).unwrap_or(0.0);
    let last_close = candles.last().map(|c| c.close).unwrap_or(0.0);
    let total_return = (last_close - first_close) / first_close * 100.0;

    println!("Статистика:");
    println!("  Средняя доходность:  {:.4}%", mean_return * 100.0);
    println!("  Годовая волатильность: {:.2}%", volatility * 100.0);
    println!("  Общая доходность:    {:.2}%", total_return);
    println!("  Диапазон цен:        ${:.2} - ${:.2}",
        candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
        candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max)
    );
    println!();

    // =====================
    // ЭТАП 3: Подготовка данных
    // =====================
    println!("⚙️ ЭТАП 3: Подготовка данных\n");

    let sequence_length = 24; // 24 часа назад
    let forecast_horizon = 1;  // Прогноз на 1 час вперёд

    let mut processor = DataProcessor::new(sequence_length, forecast_horizon);
    let (x, y) = processor.prepare_sequences(&candles)?;

    println!("Параметры:");
    println!("  Длина последовательности: {} часов", sequence_length);
    println!("  Горизонт прогноза:        {} час", forecast_horizon);
    println!("  Размер входа:             {} признаков", x.shape()[2]);
    println!();

    // Разделяем данные
    let (x_train, x_test, y_train, y_test) = processor.train_test_split(&x, &y, 0.8);

    println!("Разделение данных:");
    println!("  Обучающая выборка: {} примеров", x_train.shape()[0]);
    println!("  Тестовая выборка:  {} примеров", x_test.shape()[0]);
    println!();

    // =====================
    // ЭТАП 4: Обучение моделей
    // =====================
    println!("🧠 ЭТАП 4: Обучение моделей\n");

    let input_size = x.shape()[2];
    let hidden_size = 32;
    let epochs = 30;
    let learning_rate = 0.001;

    // Обучаем LSTM
    println!("Обучаем LSTM модель...");
    let lstm_config = LSTMConfig::new(input_size, hidden_size, 1)
        .with_learning_rate(learning_rate)
        .with_batch_size(16);

    let mut lstm = LSTM::from_config(lstm_config);
    lstm.train(&x_train, &y_train, epochs, learning_rate)?;

    // Обучаем GRU
    println!("\nОбучаем GRU модель...");
    let gru_config = LSTMConfig::new(input_size, hidden_size, 1)
        .with_learning_rate(learning_rate)
        .with_batch_size(16);

    let mut gru = GRU::from_config(gru_config);
    gru.train(&x_train, &y_train, epochs, learning_rate)?;

    // =====================
    // ЭТАП 5: Оценка моделей
    // =====================
    println!("\n📊 ЭТАП 5: Оценка моделей\n");

    // Предсказания
    let lstm_train_pred = lstm.predict(&x_train);
    let lstm_test_pred = lstm.predict(&x_test);
    let gru_train_pred = gru.predict(&x_train);
    let gru_test_pred = gru.predict(&x_test);

    println!("┌─────────────────────────────────────────────────────┐");
    println!("│              Сравнение моделей                      │");
    println!("├─────────────────────────────────────────────────────┤");
    println!("│ Модель │ Выборка │    MSE    │   RMSE   │    R²    │");
    println!("├────────┼─────────┼───────────┼──────────┼──────────┤");
    println!(
        "│ LSTM   │ Train   │ {:>9.6} │ {:>8.6} │ {:>8.4} │",
        mse(&y_train, &lstm_train_pred),
        rmse(&y_train, &lstm_train_pred),
        r2_score(&y_train, &lstm_train_pred)
    );
    println!(
        "│ LSTM   │ Test    │ {:>9.6} │ {:>8.6} │ {:>8.4} │",
        mse(&y_test, &lstm_test_pred),
        rmse(&y_test, &lstm_test_pred),
        r2_score(&y_test, &lstm_test_pred)
    );
    println!(
        "│ GRU    │ Train   │ {:>9.6} │ {:>8.6} │ {:>8.4} │",
        mse(&y_train, &gru_train_pred),
        rmse(&y_train, &gru_train_pred),
        r2_score(&y_train, &gru_train_pred)
    );
    println!(
        "│ GRU    │ Test    │ {:>9.6} │ {:>8.6} │ {:>8.4} │",
        mse(&y_test, &gru_test_pred),
        rmse(&y_test, &gru_test_pred),
        r2_score(&y_test, &gru_test_pred)
    );
    println!("└─────────────────────────────────────────────────────┘");
    println!();

    // =====================
    // ЭТАП 6: Прогнозирование
    // =====================
    println!("🔮 ЭТАП 6: Прогнозирование\n");

    // Берём последние данные для прогноза
    let latest = processor.prepare_single(&candles)?;
    let lstm_pred = lstm.predict(&latest);
    let gru_pred = gru.predict(&latest);

    let current_price = candles.last().unwrap().close;
    let lstm_price = processor.inverse_transform_price(lstm_pred[[0, 0]]);
    let gru_price = processor.inverse_transform_price(gru_pred[[0, 0]]);

    // Ансамбль (среднее)
    let ensemble_price = (lstm_price + gru_price) / 2.0;

    println!("Текущая цена: ${:.2}", current_price);
    println!();
    println!("Прогнозы на следующий час:");
    println!("  LSTM:     ${:.2} ({:+.2}%)",
        lstm_price,
        (lstm_price - current_price) / current_price * 100.0
    );
    println!("  GRU:      ${:.2} ({:+.2}%)",
        gru_price,
        (gru_price - current_price) / current_price * 100.0
    );
    println!("  Ансамбль: ${:.2} ({:+.2}%)",
        ensemble_price,
        (ensemble_price - current_price) / current_price * 100.0
    );
    println!();

    // Сохраняем модели
    std::fs::create_dir_all("models")?;
    lstm.save("models/lstm_btcusdt_1h.bin")?;
    gru.save("models/gru_btcusdt_1h.bin")?;
    processor.save("models/processor_btcusdt_1h.bin")?;
    println!("✓ Модели сохранены в папке models/");

    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Пайплайн успешно завершён!                              ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("⚠️  Отказ от ответственности:");
    println!("   Это учебный пример. НЕ используйте для реальной торговли!");
    println!("   Криптовалютный рынок высоко волатилен и рискован.");

    Ok(())
}
