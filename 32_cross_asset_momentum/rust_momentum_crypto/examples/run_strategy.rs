//! Пример: Запуск стратегии
//!
//! Запуск: cargo run --example run_strategy

use chrono::Utc;
use momentum_crypto::{
    data::{get_momentum_universe, BybitClient, PriceSeries},
    momentum::DualMomentumConfig,
    strategy::{SignalGenerator, WeightCalculator, WeightConfig},
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Запуск стратегии ===\n");

    let client = BybitClient::new();
    let universe = get_momentum_universe();

    // Загружаем данные
    println!("Загрузка данных...");
    let mut price_data: HashMap<String, PriceSeries> = HashMap::new();

    for symbol in &universe[..8] {
        match client.get_klines(symbol, "D", None, None, Some(60)).await {
            Ok(series) => {
                println!("  {} - {} свечей", symbol, series.len());
                price_data.insert(symbol.to_string(), series);
            }
            Err(e) => {
                println!("  {} - ошибка: {}", symbol, e);
            }
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }

    if price_data.is_empty() {
        anyhow::bail!("Не удалось загрузить данные");
    }

    // Создаём генератор сигналов
    let momentum_config = DualMomentumConfig {
        ts_lookback: 30,
        cs_lookback: 30,
        top_n: 3,
        risk_free_rate: 0.0,
        skip_period: 1,
        equal_weight: true,
    };

    let signal_generator = SignalGenerator::new(momentum_config);

    // Генерируем сигналы
    println!("\n--- Сигналы ---\n");
    let timestamp = Utc::now();
    let signals = signal_generator.generate(&price_data, timestamp)?;

    println!("{:<12} {:>10}", "Symbol", "Signal");
    println!("{}", "-".repeat(24));

    for (symbol, signal) in &signals.signals {
        println!("{:<12} {:>10?}", symbol, signal);
    }

    // Рассчитываем веса с учётом волатильности
    println!("\n--- Веса портфеля (Risk Parity) ---\n");

    let weight_config = WeightConfig {
        target_volatility: 0.30,
        volatility_lookback: 30,
        max_weight: 0.40,
        min_weight: 0.05,
        use_risk_parity: true,
    };

    let weight_calc = WeightCalculator::new(weight_config);

    // Собираем символы с сигналом Long
    let long_symbols: Vec<String> = signals.long_symbols().into_iter().cloned().collect();

    if long_symbols.is_empty() {
        println!("Нет активов для покупки - 100% в кеше");
    } else {
        // Рассчитываем волатильность каждого актива
        println!("Волатильность активов:");
        for symbol in &long_symbols {
            if let Some(series) = price_data.get(symbol) {
                if let Some(vol) = weight_calc.calculate_volatility(series) {
                    println!("  {}: {:.1}% годовых", symbol, vol * 100.0);
                }
            }
        }

        // Inverse volatility weights
        let inv_vol_weights = weight_calc.inverse_volatility_weights(&price_data, &long_symbols)?;

        println!("\nInverse Volatility Weights:");
        let mut total_weight = 0.0;
        for (symbol, weight) in &inv_vol_weights {
            println!("  {}: {:.1}%", symbol, weight * 100.0);
            total_weight += weight;
        }
        println!("  Total: {:.1}%", total_weight * 100.0);

        // Volatility targeted weights
        let vol_target_weights =
            weight_calc.volatility_targeted_weights(&price_data, &long_symbols)?;

        println!("\nVolatility Targeted Weights:");
        let mut total_weight = 0.0;
        for (symbol, weight) in &vol_target_weights {
            println!("  {}: {:.1}%", symbol, weight * 100.0);
            total_weight += weight;
        }
        let cash = (1.0 - total_weight).max(0.0);
        if cash > 0.01 {
            println!("  CASH: {:.1}%", cash * 100.0);
        }
    }

    // Создаём портфель
    println!("\n--- Итоговый портфель ---\n");

    let portfolio = weight_calc.create_portfolio(&price_data, &signals, timestamp)?;

    if portfolio.weights.is_empty() {
        println!("Портфель: 100% CASH");
    } else {
        println!("Распределение:");
        for (symbol, weight) in &portfolio.weights {
            println!("  {}: {:.1}%", symbol, weight * 100.0);
        }

        let total_invested: f64 = portfolio.weights.values().sum();
        let cash = 1.0 - total_invested;
        if cash > 0.01 {
            println!("  CASH: {:.1}%", cash * 100.0);
        }

        // Расчёт в долларах для примера
        let capital = 10000.0;
        println!("\nДля капитала ${:.0}:", capital);
        for (symbol, weight) in &portfolio.weights {
            let amount = capital * weight;
            println!("  {}: ${:.2}", symbol, amount);
        }
        if cash > 0.01 {
            println!("  CASH: ${:.2}", capital * cash);
        }
    }

    println!("\nГотово!");

    Ok(())
}
