//! Пример бэктестинга торговой стратегии
//!
//! Демонстрирует бэктестинг стратегии на основе прогнозов TFT.
//!
//! Запуск:
//! ```bash
//! cargo run --example backtest
//! ```

use tft_trading::{
    DataLoader, DataLoaderConfig, TFTConfig, TFTModel, Trainer, TrainingConfig,
    BacktestConfig, Backtester, SignalGenerator,
};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    println!("=== TFT Trading: Бэктестинг стратегии ===\n");

    // 1. Загружаем данные
    let data_config = DataLoaderConfig {
        symbol: "BTCUSDT".to_string(),
        interval: "1h".to_string(),
        num_candles: 2000,
        encoder_length: 168,
        prediction_length: 24,
        use_testnet: false,
        request_delay_ms: 100,
    };

    println!("Загрузка данных...");
    let loader = DataLoader::with_config(data_config.clone());
    let dataset = loader.load_from_bybit().await?;

    println!("  Загружено {} samples\n", dataset.len());

    // 2. Разбиваем данные
    let (train_data, val_data, test_data) = dataset.train_val_test_split(0.6, 0.2);

    println!("Разбиение данных:");
    println!("  Train: {} (60%)", train_data.len());
    println!("  Validation: {} (20%)", val_data.len());
    println!("  Test: {} (20%)", test_data.len());
    println!();

    // 3. Создаем и обучаем модель
    let model_config = TFTConfig {
        hidden_size: 32,
        num_attention_heads: 4,
        encoder_length: data_config.encoder_length,
        prediction_length: data_config.prediction_length,
        num_encoder_features: dataset.encoder_feature_names.len(),
        num_decoder_features: dataset.decoder_feature_names.len(),
        quantiles: vec![0.1, 0.5, 0.9],
        ..Default::default()
    };

    let mut model = TFTModel::new(model_config);

    let training_config = TrainingConfig {
        max_epochs: 5, // Быстрое обучение для демонстрации
        batch_size: 32,
        patience: 3,
        log_every: 1,
        ..Default::default()
    };

    println!("Обучение модели ({} эпох)...\n", training_config.max_epochs);
    let mut trainer = Trainer::new(training_config);
    trainer.train(&mut model, &train_data, &val_data);

    // 4. Генерируем прогнозы на тестовых данных
    println!("\nГенерация прогнозов на тестовых данных...");
    let predictions = trainer.predict(&mut model, &test_data);
    println!("  Сгенерировано {} прогнозов\n", predictions.len());

    // Собираем цены из test data (используем последнюю свечу encoder как текущую цену)
    // В реальном сценарии цены берутся из исходных данных
    let prices: Vec<f64> = (0..predictions.len())
        .map(|i| 50000.0 * (1.0 + 0.001 * i as f64)) // Синтетические цены для демо
        .collect();

    let timestamps: Vec<i64> = test_data.samples.iter()
        .map(|s| s.timestamp_prediction)
        .collect();

    // 5. Конфигурация бэктеста
    let backtest_config = BacktestConfig {
        initial_capital: 10000.0,
        commission: 0.001, // 0.1%
        slippage: 0.0005,  // 0.05%
        use_margin: false,
        max_leverage: 1.0,
        risk_free_rate: 0.02,
    };

    println!("Конфигурация бэктеста:");
    println!("  Начальный капитал: ${:.2}", backtest_config.initial_capital);
    println!("  Комиссия: {:.2}%", backtest_config.commission * 100.0);
    println!("  Slippage: {:.2}%", backtest_config.slippage * 100.0);
    println!();

    // 6. Конфигурация генератора сигналов
    let signal_generator = SignalGenerator {
        confidence_threshold: 0.03,  // 3% ширина интервала
        direction_threshold: 0.005,  // 0.5% минимальный прогноз
        max_position_size: 1.0,
        base_position_size: 0.5,
    };

    println!("Конфигурация сигналов:");
    println!("  Порог уверенности: {:.1}%", signal_generator.confidence_threshold * 100.0);
    println!("  Порог направления: {:.2}%", signal_generator.direction_threshold * 100.0);
    println!("  Базовый размер позиции: {:.0}%", signal_generator.base_position_size * 100.0);
    println!();

    // 7. Запускаем бэктест
    println!("Запуск бэктеста...\n");
    let mut backtester = Backtester::new(backtest_config.clone(), signal_generator);
    let result = backtester.run(&predictions, &prices, &timestamps);

    // 8. Выводим результаты
    result.print_summary();

    // 9. Дополнительный анализ
    println!("\n=== Дополнительный анализ ===");

    // Сравнение с Buy & Hold
    let buy_hold_return = if prices.len() > 1 {
        (prices.last().unwrap() - prices.first().unwrap()) / prices.first().unwrap()
    } else {
        0.0
    };

    println!("\nСравнение со стратегией Buy & Hold:");
    println!("  TFT Strategy Return: {:.2}%", result.total_return * 100.0);
    println!("  Buy & Hold Return: {:.2}%", buy_hold_return * 100.0);
    println!(
        "  Outperformance: {:.2}%",
        (result.total_return - buy_hold_return) * 100.0
    );

    // Анализ drawdown
    if !result.equity_curve.is_empty() {
        let peak_to_trough: f64 = result.max_drawdown * result.equity_curve[0];
        println!("\nАнализ просадки:");
        println!("  Max Drawdown: {:.2}%", result.max_drawdown * 100.0);
        println!("  Peak to Trough: ${:.2}", peak_to_trough);
    }

    // Анализ сделок
    if result.num_trades > 0 {
        println!("\nАнализ сделок:");
        println!("  Всего сделок: {}", result.num_trades);
        println!("  Выигрышных: {:.0}", result.win_rate * result.num_trades as f64);
        println!("  Проигрышных: {:.0}", (1.0 - result.win_rate) * result.num_trades as f64);
        println!("  Win Rate: {:.2}%", result.win_rate * 100.0);
        println!("  Profit Factor: {:.2}", result.profit_factor);
        println!("  Avg Trade Return: {:.4}%", result.avg_trade_return * 100.0);
    }

    // 10. Risk-adjusted метрики
    println!("\nRisk-adjusted метрики:");
    println!("  Sharpe Ratio: {:.2}", result.sharpe_ratio);
    println!("  Sortino Ratio: {:.2}", result.sortino_ratio);
    println!("  Calmar Ratio: {:.2}", result.calmar_ratio);

    // Интерпретация Sharpe
    let sharpe_interpretation = if result.sharpe_ratio > 2.0 {
        "Отлично"
    } else if result.sharpe_ratio > 1.0 {
        "Хорошо"
    } else if result.sharpe_ratio > 0.5 {
        "Приемлемо"
    } else if result.sharpe_ratio > 0.0 {
        "Слабо"
    } else {
        "Плохо"
    };
    println!("  Оценка Sharpe: {}", sharpe_interpretation);

    println!("\n=== Готово! ===");

    Ok(())
}
