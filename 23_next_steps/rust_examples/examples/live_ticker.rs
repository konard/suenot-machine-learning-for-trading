//! Пример подписки на live данные через WebSocket
//!
//! Запуск: cargo run --example live_ticker

use anyhow::Result;
use ml4t_bybit::client::{BybitWebSocket, SubscriptionType};
use std::time::Duration;
use tokio::time::timeout;

#[tokio::main]
async fn main() -> Result<()> {
    // Инициализируем логирование
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("╔════════════════════════════════════════════╗");
    println!("║       Live Ticker WebSocket Demo           ║");
    println!("╚════════════════════════════════════════════╝");
    println!();
    println!("Подключение к Bybit WebSocket...");
    println!("(Программа будет работать 30 секунд)");
    println!();

    // Создаём WebSocket клиент
    let ws = BybitWebSocket::new();

    // Подписываемся на несколько тикеров
    let subscriptions = vec![
        SubscriptionType::Ticker("BTCUSDT".to_string()),
        SubscriptionType::Ticker("ETHUSDT".to_string()),
        SubscriptionType::Ticker("SOLUSDT".to_string()),
    ];

    let mut receiver = ws.subscribe(subscriptions).await?;

    println!("✅ Подключено! Ожидание данных...");
    println!();
    println!("┌────────────┬─────────────────┬─────────────────┬─────────────────┬────────────────┐");
    println!("│   Symbol   │   Last Price    │    24h High     │    24h Low      │   24h Volume   │");
    println!("├────────────┼─────────────────┼─────────────────┼─────────────────┼────────────────┤");

    // Читаем сообщения в течение 30 секунд
    let duration = Duration::from_secs(30);
    let start = std::time::Instant::now();

    loop {
        if start.elapsed() > duration {
            break;
        }

        match timeout(Duration::from_secs(5), receiver.recv()).await {
            Ok(Some(msg)) => {
                // Парсим данные тикера
                if let Ok(ticker) = serde_json::from_value::<TickerData>(msg.data) {
                    // Очищаем строку и выводим обновление
                    print!("\r");
                    println!(
                        "│ {:>10} │ {:>15} │ {:>15} │ {:>15} │ {:>14} │",
                        ticker.symbol,
                        format!("${}", ticker.last_price),
                        format!("${}", ticker.high_price_24h),
                        format!("${}", ticker.low_price_24h),
                        format_volume(&ticker.volume_24h)
                    );
                }
            }
            Ok(None) => {
                println!("WebSocket закрыт");
                break;
            }
            Err(_) => {
                // Таймаут - продолжаем
            }
        }
    }

    println!("└────────────┴─────────────────┴─────────────────┴─────────────────┴────────────────┘");
    println!();
    println!("⏱️  Время работы: {:?}", start.elapsed());
    println!("✅ Демонстрация завершена!");

    Ok(())
}

#[derive(Debug, serde::Deserialize)]
struct TickerData {
    symbol: String,
    #[serde(rename = "lastPrice")]
    last_price: String,
    #[serde(rename = "highPrice24h")]
    high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    low_price_24h: String,
    #[serde(rename = "volume24h")]
    volume_24h: String,
}

fn format_volume(vol: &str) -> String {
    if let Ok(v) = vol.parse::<f64>() {
        if v >= 1_000_000.0 {
            format!("{:.2}M", v / 1_000_000.0)
        } else if v >= 1_000.0 {
            format!("{:.2}K", v / 1_000.0)
        } else {
            format!("{:.2}", v)
        }
    } else {
        vol.to_string()
    }
}
