//! # Options Greeks ML - CLI
//!
//! Утилита командной строки для торговли волатильностью опционов.

use options_greeks_ml::{
    api::bybit::BybitClient,
    greeks::{BlackScholes, OptionType},
    strategy::{DeltaHedger, StraddleStrategy},
    volatility::{RealizedVolatility, VolatilityPredictor, VolatilityRiskPremium, VolatilityFeatures},
};

use std::env;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Настройка логирования
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Options Greeks ML v{}", options_greeks_ml::VERSION);
    info!("Delta-Neutral Volatility Trading for Crypto");
    println!();

    // Проверяем аргументы командной строки
    let args: Vec<String> = env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("demo") => run_demo().await?,
        Some("greeks") => run_greeks_demo()?,
        Some("volatility") => run_volatility_demo()?,
        Some("strategy") => run_strategy_demo()?,
        Some("live") => run_live_data().await?,
        _ => print_usage(),
    }

    Ok(())
}

fn print_usage() {
    println!("Usage: options-trader <command>");
    println!();
    println!("Commands:");
    println!("  demo        Run full demonstration");
    println!("  greeks      Demonstrate Greeks calculation");
    println!("  volatility  Demonstrate volatility prediction");
    println!("  strategy    Demonstrate straddle strategy");
    println!("  live        Fetch live data from Bybit");
    println!();
    println!("Examples:");
    println!("  options-trader demo");
    println!("  options-trader greeks");
}

async fn run_demo() -> anyhow::Result<()> {
    info!("Running full demonstration...");
    println!();

    // 1. Демонстрация расчёта греков
    println!("=== 1. Greeks Calculation ===");
    run_greeks_demo()?;
    println!();

    // 2. Демонстрация предсказания волатильности
    println!("=== 2. Volatility Prediction ===");
    run_volatility_demo()?;
    println!();

    // 3. Демонстрация стратегии
    println!("=== 3. Trading Strategy ===");
    run_strategy_demo()?;
    println!();

    info!("Demo completed successfully!");
    Ok(())
}

fn run_greeks_demo() -> anyhow::Result<()> {
    println!("Calculating Greeks for BTC ATM options...");
    println!();

    // Параметры
    let spot = 42000.0;
    let strike = 42000.0;
    let days = 7.0;
    let iv = 0.55; // 55% IV

    // Создаём модель
    let bs = BlackScholes::crypto(spot, strike, days, iv);

    // Цены опционов
    let call_price = bs.call_price();
    let put_price = bs.put_price();
    let straddle_price = bs.straddle_price();

    println!("Parameters:");
    println!("  Spot:      ${:.2}", spot);
    println!("  Strike:    ${:.2}", strike);
    println!("  Days:      {:.0}", days);
    println!("  IV:        {:.1}%", iv * 100.0);
    println!();

    println!("Prices:");
    println!("  Call:      ${:.2}", call_price);
    println!("  Put:       ${:.2}", put_price);
    println!("  Straddle:  ${:.2}", straddle_price);
    println!();

    // Греки
    let call_greeks = bs.call_greeks();
    let put_greeks = bs.put_greeks();
    let straddle_greeks = bs.straddle_greeks();

    println!("Call Greeks:");
    println!("  Delta:  {:.4}", call_greeks.delta);
    println!("  Gamma:  {:.6}", call_greeks.gamma);
    println!("  Theta:  ${:.2}/day", call_greeks.theta);
    println!("  Vega:   ${:.2}/1%", call_greeks.vega);
    println!();

    println!("Put Greeks:");
    println!("  Delta:  {:.4}", put_greeks.delta);
    println!("  Gamma:  {:.6}", put_greeks.gamma);
    println!("  Theta:  ${:.2}/day", put_greeks.theta);
    println!("  Vega:   ${:.2}/1%", put_greeks.vega);
    println!();

    println!("Straddle Greeks:");
    println!("  Delta:  {:.4} (should be ~0 for ATM)", straddle_greeks.delta);
    println!("  Gamma:  {:.6}", straddle_greeks.gamma);
    println!("  Theta:  ${:.2}/day", straddle_greeks.theta);
    println!("  Vega:   ${:.2}/1%", straddle_greeks.vega);

    Ok(())
}

fn run_volatility_demo() -> anyhow::Result<()> {
    println!("Simulating volatility prediction...");
    println!();

    // Генерируем тестовые данные (цены BTC за 100 дней)
    let mut prices: Vec<f64> = Vec::new();
    let mut price = 40000.0;

    use rand::Rng;
    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        // Симулируем дневные движения
        let daily_return: f64 = rng.gen_range(-0.03..0.03);
        price *= 1.0 + daily_return;
        prices.push(price);
    }

    // Расчёт реализованной волатильности
    let rv_calc = RealizedVolatility::crypto();

    let rv_5d = rv_calc.calculate(&prices, Some(5)).unwrap_or(0.0);
    let rv_10d = rv_calc.calculate(&prices, Some(10)).unwrap_or(0.0);
    let rv_20d = rv_calc.calculate(&prices, Some(20)).unwrap_or(0.0);

    println!("Historical Realized Volatility:");
    println!("  5-day RV:   {:.1}%", rv_5d * 100.0);
    println!("  10-day RV:  {:.1}%", rv_10d * 100.0);
    println!("  20-day RV:  {:.1}%", rv_20d * 100.0);
    println!();

    // Предсказание волатильности
    let predictor = VolatilityPredictor::default_weights(7);
    let current_iv = 0.55; // Текущая IV = 55%

    let features = VolatilityFeatures::from_prices(&prices, None, Some(current_iv));
    let predicted_rv = predictor.predict(&features);

    println!("Volatility Prediction (7-day horizon):");
    println!("  Current IV:     {:.1}%", current_iv * 100.0);
    println!("  Predicted RV:   {:.1}%", predicted_rv * 100.0);
    println!();

    // VRP анализ
    let vrp = VolatilityRiskPremium::default_crypto();
    let signal = vrp.trading_signal(current_iv, predicted_rv, None);

    println!("Trading Signal:");
    println!("  Action:     {}", signal.action);
    println!("  Edge:       {:.2}%", signal.edge * 100.0);
    println!("  Confidence: {:.1}%", signal.confidence * 100.0);
    println!("  Reason:     {}", signal.reason);

    Ok(())
}

fn run_strategy_demo() -> anyhow::Result<()> {
    use chrono::{Duration, Utc};
    use options_greeks_ml::models::{OptionContract, OptionPosition, Portfolio};

    println!("Simulating straddle strategy with delta hedging...");
    println!();

    // Создаём портфель
    let mut portfolio = Portfolio::new(100000.0); // $100k начальный капитал

    // Текущая цена BTC
    let spot = 42000.0;
    let expiry = Utc::now() + Duration::days(7);

    // Симулируем short straddle (IV переоценена)
    println!("Opening short straddle:");
    println!("  Spot:   ${:.2}", spot);
    println!("  Strike: ${:.2}", spot);
    println!("  IV:     55%");
    println!();

    let call = OptionContract::new("BTC", spot, expiry, OptionType::Call, 800.0, 0.55);
    let put = OptionContract::new("BTC", spot, expiry, OptionType::Put, 750.0, 0.53);

    // Продаём страддл (short = отрицательное количество)
    let bs = BlackScholes::crypto(spot, spot, 7.0, 0.54);

    let call_pos = OptionPosition::new(call, -1.0, 800.0).with_greeks(bs.call_greeks());
    let put_pos = OptionPosition::new(put, -1.0, 750.0).with_greeks(bs.put_greeks());

    portfolio.add_option(call_pos);
    portfolio.add_option(put_pos);
    portfolio.underlying_spot = spot;

    // Получили премию
    let premium = 800.0 + 750.0;
    portfolio.cash += premium;

    println!("Premium collected: ${:.2}", premium);
    println!();

    // Показываем греки
    let greeks = portfolio.total_greeks();
    println!("Portfolio Greeks:");
    println!("  Delta: {:.4}", greeks.delta);
    println!("  Gamma: {:.6}", greeks.gamma);
    println!("  Theta: ${:.2}/day", greeks.theta);
    println!("  Vega:  ${:.2}/1%", greeks.vega);
    println!();

    // Дельта-хеджирование
    let mut hedger = DeltaHedger::default_crypto();

    println!("Delta hedging simulation:");
    println!("  Initial delta: {:.4}", portfolio.total_delta());

    // Симулируем движения цены
    let price_moves = vec![
        (42500.0, "Day 1: Price up"),
        (42200.0, "Day 2: Price down"),
        (43000.0, "Day 3: Big move up"),
        (42100.0, "Day 4: Drop"),
        (42300.0, "Day 5: Recovery"),
    ];

    for (new_price, description) in price_moves {
        println!();
        println!("  {} - ${:.2}", description, new_price);

        // Обновляем спот
        portfolio.underlying_spot = new_price;

        // Упрощённо: обновляем дельту пропорционально движению
        let bs_new = BlackScholes::crypto(new_price, spot, 6.0, 0.54); // Уменьшаем время
        let new_delta = -(bs_new.call_greeks().delta + bs_new.put_greeks().delta);

        // Выполняем хедж
        if let Some(trade) = hedger.execute_hedge(&mut portfolio, new_price) {
            println!("    Hedge: {} {:.4} BTC @ ${:.2}",
                if trade.quantity > 0.0 { "BUY" } else { "SELL" },
                trade.quantity.abs(),
                trade.price
            );
            println!("    Delta after: {:.4}", portfolio.total_delta());
        } else {
            println!("    No hedge needed");
        }
    }

    println!();
    println!("Final Results:");
    println!("  Number of hedges:     {}", hedger.num_hedges());
    println!("  Transaction costs:    ${:.2}", hedger.total_costs());
    println!("  Options P&L:          ${:.2}", portfolio.options_pnl());
    println!("  Underlying P&L:       ${:.2}", portfolio.underlying_pnl());
    println!("  Total P&L:            ${:.2}", portfolio.total_pnl());

    Ok(())
}

async fn run_live_data() -> anyhow::Result<()> {
    info!("Fetching live data from Bybit...");
    println!();

    let client = BybitClient::new_public();

    // Получаем тикер BTC
    println!("Fetching BTCUSDT ticker...");
    match client.get_ticker("BTCUSDT").await {
        Ok(ticker) => {
            println!("BTCUSDT:");
            println!("  Last price: ${}", ticker.last_price);
            println!("  24h change: {}%", ticker.price_24h_pcnt);
            println!("  24h high:   ${}", ticker.high_price_24h);
            println!("  24h low:    ${}", ticker.low_price_24h);
            println!("  24h volume: {}", ticker.volume_24h);
        }
        Err(e) => {
            warn!("Failed to fetch ticker: {}", e);
        }
    }
    println!();

    // Получаем свечи
    println!("Fetching 1h candles...");
    match client.get_klines("BTCUSDT", "60", 24).await {
        Ok(klines) => {
            println!("Last 24 hourly candles:");
            for kline in klines.iter().take(5) {
                let candle = kline.to_candle();
                println!(
                    "  {} | O: ${:.2} H: ${:.2} L: ${:.2} C: ${:.2}",
                    candle.timestamp.format("%Y-%m-%d %H:%M"),
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close
                );
            }
            println!("  ... ({} more candles)", klines.len().saturating_sub(5));

            // Расчёт RV
            let closes: Vec<f64> = klines.iter().map(|k| k.to_candle().close).collect();
            let rv = RealizedVolatility::crypto();
            if let Some(vol) = rv.calculate(&closes, None) {
                println!();
                println!("24h Realized Volatility: {:.1}%", vol * 100.0);
            }
        }
        Err(e) => {
            warn!("Failed to fetch klines: {}", e);
        }
    }
    println!();

    // Пробуем получить опционы
    println!("Fetching BTC options...");
    match client.get_options_chain("BTC").await {
        Ok(options) => {
            println!("Found {} BTC options", options.len());

            // Показываем несколько
            for opt in options.iter().take(5) {
                println!(
                    "  {} | IV: {:.1}% | Delta: {} | Bid: {} Ask: {}",
                    opt.symbol,
                    opt.iv() * 100.0,
                    opt.delta,
                    opt.bid_price,
                    opt.ask_price
                );
            }
        }
        Err(e) => {
            warn!("Failed to fetch options: {}", e);
            println!("Note: Options data may require different API endpoints or permissions");
        }
    }

    info!("Live data fetch completed");
    Ok(())
}
