//! Пример загрузки данных с Bybit
//!
//! Демонстрирует работу с Bybit API для получения криптовалютных данных.
//!
//! Запуск:
//! ```bash
//! cargo run --example fetch_data
//! ```

use tft_trading::{BybitClient, Kline};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Инициализация логирования
    env_logger::init();

    println!("=== TFT Trading: Загрузка данных с Bybit ===\n");

    // Создаем клиент Bybit
    let client = BybitClient::new();

    // 1. Получаем список топовых символов по объему
    println!("Получаем топ-5 торговых пар по объему...");
    let top_symbols = client.get_top_symbols_by_volume(5).await?;
    println!("Топ символы: {:?}\n", top_symbols);

    // 2. Загружаем часовые свечи для Bitcoin
    let symbol = "BTCUSDT";
    let interval = "1h";
    let limit = 100;

    println!("Загружаем {} свечей {} ({})...", limit, symbol, interval);
    let klines = client.get_klines(symbol, interval, limit).await?;
    println!("Загружено {} свечей\n", klines.len());

    // Выводим первые 5 свечей
    println!("Первые 5 свечей:");
    for (i, kline) in klines.iter().take(5).enumerate() {
        println!(
            "  {}: {} | O: {:.2} H: {:.2} L: {:.2} C: {:.2} | Vol: {:.0} | Return: {:.2}%",
            i + 1,
            kline.datetime().format("%Y-%m-%d %H:%M"),
            kline.open,
            kline.high,
            kline.low,
            kline.close,
            kline.volume,
            kline.returns_percent()
        );
    }
    println!();

    // 3. Статистика по данным
    let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let returns: Vec<f64> = klines.iter().map(|k| k.returns()).collect();

    let avg_price = closes.iter().sum::<f64>() / closes.len() as f64;
    let min_price = closes.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_price = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let volatility = {
        let variance = returns.iter().map(|r| (r - avg_return).powi(2)).sum::<f64>()
            / returns.len() as f64;
        variance.sqrt() * 100.0
    };

    println!("Статистика {}:", symbol);
    println!("  Средняя цена: ${:.2}", avg_price);
    println!("  Диапазон: ${:.2} - ${:.2}", min_price, max_price);
    println!("  Средняя доходность: {:.4}%", avg_return * 100.0);
    println!("  Волатильность (std): {:.2}%", volatility);
    println!();

    // 4. Bullish/Bearish анализ
    let bullish_count = klines.iter().filter(|k| k.is_bullish()).count();
    let bearish_count = klines.iter().filter(|k| k.is_bearish()).count();

    println!("Анализ свечей:");
    println!(
        "  Бычьи: {} ({:.1}%)",
        bullish_count,
        bullish_count as f64 / klines.len() as f64 * 100.0
    );
    println!(
        "  Медвежьи: {} ({:.1}%)",
        bearish_count,
        bearish_count as f64 / klines.len() as f64 * 100.0
    );
    println!();

    // 5. Получаем стакан заявок
    println!("Получаем стакан заявок {}...", symbol);
    let orderbook = client.get_orderbook(symbol, 10).await?;

    println!("Order Book:");
    println!(
        "  Best Bid: ${:.2} | Best Ask: ${:.2}",
        orderbook.best_bid().unwrap_or(0.0),
        orderbook.best_ask().unwrap_or(0.0)
    );
    println!("  Spread: ${:.2} ({:.4}%)",
        orderbook.spread().unwrap_or(0.0),
        orderbook.spread_percent().unwrap_or(0.0)
    );
    println!("  Imbalance (10 levels): {:.2}", orderbook.imbalance(10));
    println!();

    // 6. Получаем тикер
    println!("Получаем тикер {}...", symbol);
    let ticker = client.get_ticker(symbol).await?;

    println!("Ticker:");
    println!("  Последняя цена: ${:.2}", ticker.last_price);
    println!("  Изменение 24h: {:.2}%", ticker.price_24h_pcnt * 100.0);
    println!("  High 24h: ${:.2}", ticker.high_price_24h);
    println!("  Low 24h: ${:.2}", ticker.low_price_24h);
    println!("  Объем 24h: {:.0} {}", ticker.volume_24h, symbol.replace("USDT", ""));
    println!();

    // 7. Загружаем данные для нескольких символов
    println!("Загружаем данные для нескольких символов...");
    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];

    for symbol in &symbols {
        match client.get_klines(symbol, "1h", 24).await {
            Ok(klines) => {
                let returns: f64 = klines.iter().map(|k| k.returns()).sum();
                println!(
                    "  {}: {} свечей, суммарная доходность: {:.2}%",
                    symbol,
                    klines.len(),
                    returns * 100.0
                );
            }
            Err(e) => {
                println!("  {}: ошибка - {}", symbol, e);
            }
        }
    }

    println!("\n=== Готово! ===");

    Ok(())
}
