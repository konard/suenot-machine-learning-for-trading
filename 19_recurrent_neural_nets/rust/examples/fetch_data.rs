//! Пример: Получение данных с биржи Bybit
//!
//! Этот пример демонстрирует:
//! - Подключение к API Bybit
//! - Получение исторических свечей
//! - Сохранение данных в CSV
//!
//! Запуск: cargo run --example fetch_data

use crypto_rnn::data::BybitClient;
use crypto_rnn::utils::save_candles_csv;
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Инициализируем логирование
    env_logger::init();

    println!("=== Получение данных с Bybit ===\n");

    // Параметры из командной строки или значения по умолчанию
    let args: Vec<String> = env::args().collect();
    let symbol = args.get(1).map(|s| s.as_str()).unwrap_or("BTCUSDT");
    let interval = args.get(2).map(|s| s.as_str()).unwrap_or("1h");
    let limit: u32 = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    println!("Символ: {}", symbol);
    println!("Интервал: {}", interval);
    println!("Количество свечей: {}", limit);
    println!();

    // Создаём клиент
    let client = BybitClient::new();

    // Получаем данные
    println!("Загружаем данные...");
    let candles = client.get_klines(symbol, interval, limit).await?;

    println!("Получено {} свечей\n", candles.len());

    // Показываем статистику
    if let (Some(first), Some(last)) = (candles.first(), candles.last()) {
        println!("Период: {} - {}", first.datetime(), last.datetime());
        println!(
            "Диапазон цен: {:.2} - {:.2}",
            candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min),
            candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max)
        );
        println!("Последняя цена: {:.2}", last.close);

        // Вычисляем изменение за период
        let change_pct = (last.close - first.open) / first.open * 100.0;
        println!("Изменение за период: {:.2}%", change_pct);
    }

    // Показываем последние 5 свечей
    println!("\nПоследние 5 свечей:");
    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12} {:>15}",
        "Время", "Open", "High", "Low", "Close", "Volume"
    );
    println!("{}", "-".repeat(85));

    for candle in candles.iter().rev().take(5).rev() {
        println!(
            "{:<20} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
            candle.datetime().format("%Y-%m-%d %H:%M"),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume
        );
    }

    // Сохраняем в CSV
    let filename = format!("data/{}_{}.csv", symbol.to_lowercase(), interval);
    std::fs::create_dir_all("data")?;
    save_candles_csv(&candles, &filename)?;
    println!("\nДанные сохранены в: {}", filename);

    Ok(())
}
