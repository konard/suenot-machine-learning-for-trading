//! # Пример: Получение данных с Bybit
//!
//! Получение реальных рыночных данных с биржи Bybit.
//!
//! ## Запуск
//!
//! ```bash
//! cargo run --example bybit_live
//! ```
//!
//! Примечание: Требуется подключение к интернету.

use options_greeks_ml::{
    api::bybit::BybitClient,
    greeks::BlackScholes,
    volatility::RealizedVolatility,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Bybit Live Data Example ===\n");

    let client = BybitClient::new_public();

    // 1. Получаем текущую цену BTC
    println!("Fetching BTCUSDT ticker...\n");

    match client.get_ticker("BTCUSDT").await {
        Ok(ticker) => {
            let price: f64 = ticker.last_price.parse().unwrap_or(0.0);
            let high: f64 = ticker.high_price_24h.parse().unwrap_or(0.0);
            let low: f64 = ticker.low_price_24h.parse().unwrap_or(0.0);
            let volume: f64 = ticker.volume_24h.parse().unwrap_or(0.0);

            println!("BTCUSDT Ticker:");
            println!("  Last Price:  ${:.2}", price);
            println!("  24h High:    ${:.2}", high);
            println!("  24h Low:     ${:.2}", low);
            println!("  24h Volume:  {:.2} BTC", volume);
            println!("  24h Change:  {}%", ticker.price_24h_pcnt);
            println!();

            // Расчёт дневного диапазона
            let daily_range = (high - low) / price * 100.0;
            println!("  Daily Range: {:.2}%", daily_range);
        }
        Err(e) => {
            eprintln!("Failed to fetch ticker: {}", e);
        }
    }
    println!();

    // 2. Получаем исторические свечи для расчёта волатильности
    println!("Fetching historical candles...\n");

    match client.get_klines("BTCUSDT", "D", 30).await {
        Ok(klines) => {
            println!("Received {} daily candles", klines.len());

            // Конвертируем в цены закрытия
            let closes: Vec<f64> = klines
                .iter()
                .map(|k| k.close.parse().unwrap_or(0.0))
                .collect();

            // Показываем последние несколько
            println!("\nRecent daily closes:");
            for (i, kline) in klines.iter().take(5).enumerate() {
                let candle = kline.to_candle();
                println!(
                    "  {}: O ${:.0} H ${:.0} L ${:.0} C ${:.0}",
                    candle.timestamp.format("%Y-%m-%d"),
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close
                );
            }
            println!();

            // Расчёт реализованной волатильности
            let rv = RealizedVolatility::crypto();

            if let Some(rv_7d) = rv.calculate(&closes, Some(7)) {
                println!("7-day Realized Volatility:  {:.1}%", rv_7d * 100.0);
            }
            if let Some(rv_14d) = rv.calculate(&closes, Some(14)) {
                println!("14-day Realized Volatility: {:.1}%", rv_14d * 100.0);
            }
            if let Some(rv_30d) = rv.calculate(&closes, None) {
                println!("30-day Realized Volatility: {:.1}%", rv_30d * 100.0);
            }
        }
        Err(e) => {
            eprintln!("Failed to fetch candles: {}", e);
        }
    }
    println!();

    // 3. Получаем часовые свечи для более детального анализа
    println!("Fetching hourly candles for intraday volatility...\n");

    match client.get_klines("BTCUSDT", "60", 168).await {
        // 7 дней * 24 часа
        Ok(klines) => {
            let closes: Vec<f64> = klines
                .iter()
                .map(|k| k.close.parse().unwrap_or(0.0))
                .collect();

            let rv = RealizedVolatility::crypto();
            if let Some(rv_hourly) = rv.calculate(&closes, None) {
                // Конвертируем часовую волатильность в годовую
                // Для крипты: sqrt(365 * 24) = sqrt(8760)
                let hourly_to_annual = (365.0 * 24.0_f64).sqrt() / 365.0_f64.sqrt();
                let adjusted_rv = rv_hourly * hourly_to_annual;

                println!("Intraday Volatility (168 hours):");
                println!("  Raw hourly vol:    {:.2}%", rv_hourly * 100.0);
                println!("  Annualized:        {:.1}%", adjusted_rv * 100.0);
            }
        }
        Err(e) => {
            eprintln!("Failed to fetch hourly candles: {}", e);
        }
    }
    println!();

    // 4. Пробуем получить опционные данные
    println!("Fetching BTC options data...\n");

    match client.get_options_chain("BTC").await {
        Ok(options) => {
            if options.is_empty() {
                println!("No options data available (might need different API access)");
            } else {
                println!("Found {} BTC options", options.len());
                println!("\nSample options:");

                for opt in options.iter().take(10) {
                    let iv = opt.iv() * 100.0;
                    let delta = opt.delta_f64();

                    println!(
                        "  {} | IV: {:.1}% | Delta: {:.3} | Bid: {} Ask: {}",
                        opt.symbol, iv, delta, opt.bid_price, opt.ask_price
                    );
                }

                // Анализ IV surface
                println!("\nIV Analysis:");

                let ivs: Vec<f64> = options.iter().map(|o| o.iv()).collect();
                if !ivs.is_empty() {
                    let avg_iv = ivs.iter().sum::<f64>() / ivs.len() as f64;
                    let min_iv = ivs.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_iv = ivs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                    println!("  Average IV: {:.1}%", avg_iv * 100.0);
                    println!("  Min IV:     {:.1}%", min_iv * 100.0);
                    println!("  Max IV:     {:.1}%", max_iv * 100.0);
                    println!("  IV Range:   {:.1}%", (max_iv - min_iv) * 100.0);
                }
            }
        }
        Err(e) => {
            println!("Options data not available: {}", e);
            println!("(This is normal - Bybit options API may require authentication)");
        }
    }
    println!();

    // 5. Теоретические цены опционов на основе рыночных данных
    println!("=== Theoretical Option Prices ===\n");

    if let Ok(ticker) = client.get_ticker("BTCUSDT").await {
        let spot: f64 = ticker.last_price.parse().unwrap_or(42000.0);

        println!("Based on current spot ${:.2}:\n", spot);

        // Используем примерную волатильность
        let iv = 0.55; // 55%

        for days in [7, 14, 30] {
            let bs = BlackScholes::crypto(spot, spot, days as f64, iv);

            let call = bs.call_price();
            let put = bs.put_price();
            let straddle = bs.straddle_price();
            let greeks = bs.straddle_greeks();

            println!("{}-day ATM options (IV=55%):", days);
            println!("  Call:     ${:.2}", call);
            println!("  Put:      ${:.2}", put);
            println!("  Straddle: ${:.2}", straddle);
            println!("  Vega:     ${:.2}/1%", greeks.vega);
            println!("  Theta:    ${:.2}/day", greeks.theta);
            println!();
        }
    }

    println!("Done!");
    Ok(())
}
