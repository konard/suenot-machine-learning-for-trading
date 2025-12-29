//! Пример: Загрузка цен с Bybit
//!
//! Запуск: cargo run --example fetch_prices

use chrono::{Duration, Utc};
use momentum_crypto::data::{get_momentum_universe, BybitClient};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Инициализация логирования
    tracing_subscriber::fmt::init();

    println!("=== Загрузка цен с Bybit ===\n");

    // Создаём клиент
    let client = BybitClient::new();

    // Получаем вселенную активов
    let universe = get_momentum_universe();
    println!("Вселенная активов: {:?}\n", universe);

    // Загружаем дневные свечи для Bitcoin
    println!("Загрузка BTCUSDT (дневные свечи)...");
    let btc_daily = client
        .get_klines("BTCUSDT", "D", None, None, Some(30))
        .await?;

    println!("Загружено {} свечей", btc_daily.len());

    if let Some(last) = btc_daily.last() {
        println!("Последняя свеча:");
        println!("  Дата: {}", last.timestamp);
        println!("  Open: ${:.2}", last.open);
        println!("  High: ${:.2}", last.high);
        println!("  Low: ${:.2}", last.low);
        println!("  Close: ${:.2}", last.close);
        println!("  Volume: {:.2}", last.volume);
    }

    // Загружаем часовые свечи для Ethereum
    println!("\nЗагрузка ETHUSDT (часовые свечи)...");
    let eth_hourly = client
        .get_klines("ETHUSDT", "60", None, None, Some(24))
        .await?;

    println!("Загружено {} свечей", eth_hourly.len());

    // Рассчитываем простую статистику
    let closes = eth_hourly.closes();
    if !closes.is_empty() {
        let min = closes.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg = closes.iter().sum::<f64>() / closes.len() as f64;

        println!("\nСтатистика за последние 24 часа:");
        println!("  Min: ${:.2}", min);
        println!("  Max: ${:.2}", max);
        println!("  Avg: ${:.2}", avg);
    }

    // Загружаем данные за определённый период
    println!("\nЗагрузка SOL за последние 7 дней...");
    let end = Utc::now();
    let start = end - Duration::days(7);

    let sol_week = client
        .get_all_klines("SOLUSDT", "D", start, end)
        .await?;

    println!("Загружено {} свечей", sol_week.len());

    // Выводим доходности
    let returns = sol_week.returns();
    if !returns.is_empty() {
        println!("\nДневные доходности SOL:");
        for (i, ret) in returns.iter().enumerate() {
            let sign = if *ret >= 0.0 { "+" } else { "" };
            println!("  День {}: {}{:.2}%", i + 1, sign, ret * 100.0);
        }

        let total_return: f64 = returns.iter().map(|r| 1.0 + r).product::<f64>() - 1.0;
        println!("\nОбщая доходность за неделю: {:.2}%", total_return * 100.0);
    }

    // Получаем текущие тикеры
    println!("\n=== Текущие цены ===\n");
    let symbols: Vec<&str> = universe.iter().take(5).copied().collect();
    let tickers = client.get_tickers(&symbols).await?;

    println!("{:<12} {:>12} {:>10}", "Symbol", "Price", "24h Change");
    println!("{}", "-".repeat(36));

    for (symbol, ticker) in tickers {
        let price: f64 = ticker.last_price.parse().unwrap_or(0.0);
        let change: f64 = ticker.price_24h_pcnt.parse().unwrap_or(0.0);
        let change_str = format!("{:+.2}%", change * 100.0);
        println!("{:<12} {:>12.2} {:>10}", symbol, price, change_str);
    }

    println!("\nГотово!");

    Ok(())
}
