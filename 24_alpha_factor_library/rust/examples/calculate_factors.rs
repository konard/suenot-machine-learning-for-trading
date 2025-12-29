//! Пример: Расчёт технических индикаторов
//!
//! Демонстрирует:
//! - Получение данных с Bybit
//! - Расчёт различных индикаторов
//! - Анализ результатов

use alpha_factors::{
    BybitClient,
    api::Interval,
    factors::{
        self,
        FactorCalculator,
        Signal,
    },
};
use alpha_factors::data::kline::KlineVec;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("=== Расчёт технических индикаторов ===\n");

    // Получаем данные
    let client = BybitClient::new();
    let symbol = "BTCUSDT";

    println!("Получаем данные для {}...", symbol);

    let klines = client
        .get_klines_with_interval(symbol, Interval::Hour1, 200)
        .await?;

    println!("Получено {} свечей\n", klines.len());

    // Извлекаем OHLCV данные
    let closes = klines.closes();
    let highs = klines.highs();
    let lows = klines.lows();
    let volumes = klines.volumes();

    let current_price = *closes.last().unwrap();

    // === Трендовые индикаторы ===
    println!("--- Трендовые индикаторы ---");

    let sma_20 = factors::sma(&closes, 20);
    let sma_50 = factors::sma(&closes, 50);
    let ema_12 = factors::ema(&closes, 12);
    let ema_26 = factors::ema(&closes, 26);

    println!("SMA(20): {:.2}", sma_20.last().unwrap());
    println!("SMA(50): {:.2}", sma_50.last().unwrap());
    println!("EMA(12): {:.2}", ema_12.last().unwrap());
    println!("EMA(26): {:.2}", ema_26.last().unwrap());

    // Тренд
    let trend = if sma_20.last() > sma_50.last() {
        "Бычий (SMA20 > SMA50)"
    } else {
        "Медвежий (SMA20 < SMA50)"
    };
    println!("Тренд: {}", trend);

    // MACD
    let macd = factors::macd(&closes, 12, 26, 9);
    println!("\nMACD:");
    println!("  Линия: {:.4}", macd.macd_line.last().unwrap());
    println!("  Сигнал: {:.4}", macd.signal_line.last().unwrap());
    println!("  Гистограмма: {:.4}", macd.histogram.last().unwrap());

    let macd_signal = if *macd.histogram.last().unwrap() > 0.0 {
        "Покупка (гистограмма > 0)"
    } else {
        "Продажа (гистограмма < 0)"
    };
    println!("  Сигнал MACD: {}", macd_signal);

    // Bollinger Bands
    println!("\nПолосы Боллинджера (20, 2):");
    let bb = factors::bollinger_bands(&closes, 20, 2.0);
    println!("  Верхняя: {:.2}", bb.upper.last().unwrap());
    println!("  Средняя: {:.2}", bb.middle.last().unwrap());
    println!("  Нижняя: {:.2}", bb.lower.last().unwrap());
    println!("  %B: {:.2}", bb.percent_b.last().unwrap());

    let bb_position = if current_price > *bb.upper.last().unwrap() {
        "Выше верхней полосы (перекуплен)"
    } else if current_price < *bb.lower.last().unwrap() {
        "Ниже нижней полосы (перепродан)"
    } else {
        "Внутри полос"
    };
    println!("  Позиция: {}", bb_position);

    // === Индикаторы моментума ===
    println!("\n--- Индикаторы моментума ---");

    let rsi = factors::rsi(&closes, 14);
    let rsi_value = *rsi.last().unwrap();
    println!("RSI(14): {:.2}", rsi_value);

    let rsi_signal = if rsi_value < 30.0 {
        "Перепродан (покупать)"
    } else if rsi_value > 70.0 {
        "Перекуплен (продавать)"
    } else {
        "Нейтрально"
    };
    println!("  Сигнал RSI: {}", rsi_signal);

    let stoch = factors::stochastic(&highs, &lows, &closes, 14, 3);
    println!("\nСтохастик (14, 3):");
    println!("  %K: {:.2}", stoch.k.last().unwrap());
    println!("  %D: {:.2}", stoch.d.last().unwrap());

    let roc = factors::roc(&closes, 10);
    println!("\nROC(10): {:.2}%", roc.last().unwrap());

    // === Индикаторы объёма ===
    println!("\n--- Индикаторы объёма ---");

    let obv = factors::obv(&closes, &volumes);
    println!("OBV: {:.0}", obv.last().unwrap());

    let vwap = factors::vwap(&highs, &lows, &closes, &volumes);
    println!("VWAP: {:.2}", vwap.last().unwrap());

    let vwap_diff = current_price - vwap.last().unwrap();
    let vwap_signal = if vwap_diff > 0.0 {
        format!("Цена выше VWAP на ${:.2}", vwap_diff)
    } else {
        format!("Цена ниже VWAP на ${:.2}", -vwap_diff)
    };
    println!("  {}", vwap_signal);

    let mfi = factors::mfi(&highs, &lows, &closes, &volumes, 14);
    let mfi_value = *mfi.last().unwrap();
    println!("MFI(14): {:.2}", mfi_value);

    // === Индикаторы волатильности ===
    println!("\n--- Индикаторы волатильности ---");

    let atr = factors::atr(&highs, &lows, &closes, 14);
    let atr_value = *atr.last().unwrap();
    println!("ATR(14): {:.2}", atr_value);
    println!("  ATR/Цена: {:.2}%", (atr_value / current_price) * 100.0);

    let hv = factors::historical_volatility(&closes, 20, 365.0 * 24.0); // Часовые данные
    println!("Историческая волатильность (20): {:.2}%", hv.last().unwrap() * 100.0);

    // === Альфа-факторы ===
    println!("\n--- Альфа-факторы ---");

    let opens = klines.opens();

    let alpha_003 = factors::alpha_003(&opens, &volumes, 10);
    println!("Alpha #003 (open-volume corr): {:.4}", alpha_003.last().unwrap());

    let alpha_012 = factors::alpha_012(&closes, &volumes);
    println!("Alpha #012 (vol-price signal): {:.4}", alpha_012.last().unwrap());

    let mom_factor = factors::momentum_factor(&closes, 20);
    println!("Momentum Factor (20): {:.4}", mom_factor.last().unwrap());

    let mr_factor = factors::mean_reversion_factor(&closes, 20);
    println!("Mean Reversion Factor (20): {:.4}", mr_factor.last().unwrap());

    let vol_spike = factors::volume_spike_factor(&volumes, 20);
    println!("Volume Spike Factor (20): {:.4}", vol_spike.last().unwrap());

    // === Комплексный анализ ===
    println!("\n=== Комплексный анализ ===");
    println!("Текущая цена: ${:.2}", current_price);

    // Используем FactorCalculator
    let calc = FactorCalculator::from_klines(&klines);
    let factor_set = calc.calculate_all();

    if let Some(snapshot) = factor_set.last_values() {
        let signal = snapshot.generate_signal(current_price);

        let signal_text = match signal {
            Signal::StrongBuy => "СИЛЬНАЯ ПОКУПКА",
            Signal::Buy => "ПОКУПКА",
            Signal::Neutral => "ДЕРЖАТЬ",
            Signal::Sell => "ПРОДАЖА",
            Signal::StrongSell => "СИЛЬНАЯ ПРОДАЖА",
        };

        println!("\nОбщий сигнал: {} (score: {})", signal_text, signal.to_score());

        println!("\nОбоснование:");
        println!("  - RSI: {:.1} ({})",
            snapshot.rsi_14,
            if snapshot.rsi_14 < 30.0 { "перепродан" }
            else if snapshot.rsi_14 > 70.0 { "перекуплен" }
            else { "нейтрально" }
        );
        println!("  - MACD гистограмма: {:.4} ({})",
            snapshot.macd_histogram,
            if snapshot.macd_histogram > 0.0 { "бычий" } else { "медвежий" }
        );
        println!("  - Цена vs BB: {} ({})",
            if current_price > snapshot.bb_upper { "выше" }
            else if current_price < snapshot.bb_lower { "ниже" }
            else { "внутри" },
            if current_price > snapshot.bb_upper { "перекуплен" }
            else if current_price < snapshot.bb_lower { "перепродан" }
            else { "нейтрально" }
        );
        println!("  - Тренд: SMA20={:.2} vs SMA50={:.2} ({})",
            snapshot.sma_20,
            snapshot.sma_50,
            if snapshot.sma_20 > snapshot.sma_50 { "бычий" } else { "медвежий" }
        );
    }

    Ok(())
}
