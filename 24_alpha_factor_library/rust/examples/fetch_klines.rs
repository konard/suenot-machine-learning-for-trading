//! Пример: Получение свечей с биржи Bybit
//!
//! Демонстрирует:
//! - Подключение к API Bybit
//! - Получение OHLCV данных
//! - Вывод базовой статистики

use alpha_factors::{BybitClient, api::Interval};
use alpha_factors::data::kline::KlineVec;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Инициализация логирования
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("=== Получение свечей с Bybit ===\n");

    // Создаём клиент API
    let client = BybitClient::new();

    // Список популярных криптовалют
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    for symbol in symbols {
        println!("--- {} ---", symbol);

        // Получаем последние 100 часовых свечей
        match client.get_klines_with_interval(symbol, Interval::Hour1, 100).await {
            Ok(klines) => {
                println!("Получено {} свечей", klines.len());

                if let (Some(first), Some(last)) = (klines.first(), klines.last()) {
                    println!(
                        "Период: {} - {}",
                        chrono::DateTime::from_timestamp_millis(first.timestamp)
                            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                            .unwrap_or_default(),
                        chrono::DateTime::from_timestamp_millis(last.timestamp)
                            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                            .unwrap_or_default()
                    );

                    // Статистика
                    let closes = klines.closes();
                    let volumes = klines.volumes();

                    let min_price = closes.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_price = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let avg_volume: f64 = volumes.iter().sum::<f64>() / volumes.len() as f64;

                    println!("Цена: мин={:.2}, макс={:.2}, текущая={:.2}",
                        min_price, max_price, last.close);
                    println!("Средний объём: {:.2}", avg_volume);

                    // Изменение за период
                    let change = ((last.close - first.close) / first.close) * 100.0;
                    println!("Изменение за период: {:.2}%", change);
                }
            }
            Err(e) => {
                eprintln!("Ошибка получения данных: {}", e);
            }
        }

        println!();
    }

    // Получаем тикер для BTC
    println!("=== Текущий тикер BTCUSDT ===\n");

    match client.get_ticker("BTCUSDT").await {
        Ok(ticker) => {
            println!("Последняя цена: ${:.2}", ticker.last_price);
            println!("Bid: ${:.2} x {:.4}", ticker.bid_price, ticker.bid_size);
            println!("Ask: ${:.2} x {:.4}", ticker.ask_price, ticker.ask_size);
            println!("Спред: ${:.2} ({:.4}%)", ticker.spread(), ticker.spread_percent());
            println!("24ч изменение: {:.2}%", ticker.price_change_percent_24h);
            println!("24ч объём: {:.2} BTC", ticker.volume_24h);
        }
        Err(e) => {
            eprintln!("Ошибка получения тикера: {}", e);
        }
    }

    Ok(())
}
