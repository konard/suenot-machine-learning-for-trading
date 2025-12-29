//! Пример: Получение данных с Bybit
//!
//! Демонстрирует использование Bybit API клиента
//! для получения свечных данных криптовалют.
//!
//! Запуск: cargo run --example fetch_bybit_data

use anyhow::Result;
use chrono::{Duration, Utc};
use rust_sentiment_momentum::api::BybitClient;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Получение данных с Bybit ===\n");

    // Создаём клиент
    let client = BybitClient::new();

    // Список символов для анализа
    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    for symbol in &symbols {
        println!("--- {} ---", symbol);

        // Получаем последние 50 часовых свечей
        match client.fetch_klines(symbol, "60", None, None, Some(50)).await {
            Ok(klines) => {
                if let Some(last) = klines.last() {
                    println!("  Последняя цена: ${:.2}", last.close);
                    println!("  Объём: {:.2}", last.volume);
                    println!("  Время: {}", last.datetime().format("%Y-%m-%d %H:%M"));

                    // Расчёт изменения за период
                    if let Some(first) = klines.first() {
                        let change = (last.close - first.close) / first.close * 100.0;
                        println!(
                            "  Изменение за {} свечей: {:+.2}%",
                            klines.len(),
                            change
                        );
                    }
                }

                // Средний объём
                let avg_volume: f64 = klines.iter().map(|k| k.volume).sum::<f64>() / klines.len() as f64;
                println!("  Средний объём: {:.2}", avg_volume);
            }
            Err(e) => {
                println!("  Ошибка: {}", e);
            }
        }

        println!();
    }

    // Пример получения исторических данных
    println!("=== Исторические данные BTC ===\n");

    let end_time = Utc::now();
    let start_time = end_time - Duration::days(7);

    match client
        .fetch_historical_klines(
            "BTCUSDT",
            "240", // 4-часовые свечи
            start_time.timestamp_millis(),
            end_time.timestamp_millis(),
        )
        .await
    {
        Ok(data) => {
            println!("Получено {} свечей за 7 дней", data.len());

            if !data.is_empty() {
                let closes = data.close_prices();
                let min_price = closes.iter().copied().fold(f64::INFINITY, f64::min);
                let max_price = closes.iter().copied().fold(f64::NEG_INFINITY, f64::max);

                println!("Мин. цена: ${:.2}", min_price);
                println!("Макс. цена: ${:.2}", max_price);
                println!("Диапазон: ${:.2}", max_price - min_price);

                // Волатильность
                let returns = data.returns();
                if !returns.is_empty() {
                    let volatility: f64 = returns
                        .iter()
                        .map(|r| r * r)
                        .sum::<f64>()
                        .sqrt() / returns.len() as f64
                        * 100.0;
                    println!("Волатильность (std returns): {:.2}%", volatility * returns.len() as f64);
                }
            }
        }
        Err(e) => {
            println!("Ошибка: {}", e);
        }
    }

    println!("\n=== Получение списка символов ===\n");

    match client.get_symbols("linear").await {
        Ok(symbols) => {
            println!("Найдено {} торговых пар", symbols.len());
            println!("Первые 10:");
            for symbol in symbols.iter().take(10) {
                println!("  - {}", symbol);
            }
        }
        Err(e) => {
            println!("Ошибка: {}", e);
        }
    }

    Ok(())
}
