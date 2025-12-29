//! Пример получения исторических свечей с Bybit
//!
//! Запуск: cargo run --example fetch_klines

use anyhow::Result;
use ml4t_bybit::client::BybitClient;
use ml4t_bybit::data::Interval;

#[tokio::main]
async fn main() -> Result<()> {
    // Инициализируем логирование
    tracing_subscriber::fmt::init();

    println!("╔════════════════════════════════════════════╗");
    println!("║     Bybit Klines (Candlestick) Fetcher     ║");
    println!("╚════════════════════════════════════════════╝");
    println!();

    // Создаём клиент (используем mainnet для публичных данных)
    let client = BybitClient::new();

    // Символы для получения данных
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    for symbol in symbols {
        println!("📊 Fetching data for {}", symbol);
        println!("─────────────────────────────────────────────");

        // Получаем текущую цену
        match client.get_ticker(symbol).await {
            Ok(ticker) => {
                println!("  Current Price: ${}", ticker.last_price);
                println!("  24h High:      ${}", ticker.high_price_24h);
                println!("  24h Low:       ${}", ticker.low_price_24h);
                println!("  24h Volume:    {}", ticker.volume_24h);
                println!("  24h Change:    {}%", ticker.price_24h_pcnt);
            }
            Err(e) => {
                println!("  Error fetching ticker: {}", e);
            }
        }

        println!();

        // Получаем последние 20 часовых свечей
        match client.get_klines(symbol, Interval::H1, Some(20)).await {
            Ok(klines) => {
                println!("  Last {} hourly candles:", klines.len());
                println!("  ┌─────────────────────┬───────────┬───────────┬───────────┬───────────┐");
                println!("  │       Time          │   Open    │   High    │   Low     │   Close   │");
                println!("  ├─────────────────────┼───────────┼───────────┼───────────┼───────────┤");

                for kline in klines.iter().take(10) {
                    // Конвертируем timestamp в читаемую дату
                    let datetime = chrono::DateTime::from_timestamp_millis(kline.timestamp as i64)
                        .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                        .unwrap_or_else(|| "Unknown".to_string());

                    println!(
                        "  │ {} │ {:>9.2} │ {:>9.2} │ {:>9.2} │ {:>9.2} │",
                        datetime, kline.open, kline.high, kline.low, kline.close
                    );
                }

                if klines.len() > 10 {
                    println!("  │        ...          │    ...    │    ...    │    ...    │    ...    │");
                }

                println!("  └─────────────────────┴───────────┴───────────┴───────────┴───────────┘");

                // Статистика
                if !klines.is_empty() {
                    let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
                    let avg = closes.iter().sum::<f64>() / closes.len() as f64;
                    let min = closes.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                    println!();
                    println!("  Statistics:");
                    println!("  • Average close: ${:.2}", avg);
                    println!("  • Min close:     ${:.2}", min);
                    println!("  • Max close:     ${:.2}", max);
                    println!("  • Range:         ${:.2}", max - min);
                }
            }
            Err(e) => {
                println!("  Error fetching klines: {}", e);
            }
        }

        println!();
        println!();
    }

    // Демонстрация разных интервалов
    println!("═════════════════════════════════════════════");
    println!("Демонстрация различных интервалов для BTCUSDT");
    println!("═════════════════════════════════════════════");
    println!();

    let intervals = vec![
        (Interval::M1, "1 минута"),
        (Interval::M5, "5 минут"),
        (Interval::M15, "15 минут"),
        (Interval::H1, "1 час"),
        (Interval::H4, "4 часа"),
        (Interval::D1, "1 день"),
    ];

    for (interval, name) in intervals {
        match client.get_klines("BTCUSDT", interval, Some(5)).await {
            Ok(klines) => {
                if let Some(last) = klines.last() {
                    println!(
                        "  {} ({}): Open ${:.2}, Close ${:.2}, Volume {:.2}",
                        name, interval, last.open, last.close, last.volume
                    );
                }
            }
            Err(e) => {
                println!("  {}: Error - {}", name, e);
            }
        }
    }

    println!();
    println!("✅ Done!");

    Ok(())
}
