//! Пример: Бэктестинг стратегии
//!
//! Запуск: cargo run --example backtest

use chrono::{Duration, Utc};
use momentum_crypto::{
    backtest::{BacktestConfig, BacktestEngine, calculate_all_metrics},
    data::{get_momentum_universe, BybitClient, PriceSeries},
    momentum::DualMomentumConfig,
    strategy::WeightConfig,
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Бэктестинг стратегии ===\n");

    let client = BybitClient::new();
    let universe = get_momentum_universe();

    // Параметры бэктеста
    let days_back = 90;
    let initial_capital = 10000.0;

    let end_date = Utc::now();
    let start_date = end_date - Duration::days(days_back + 60); // +60 для lookback

    // Загружаем исторические данные
    println!("Загрузка данных за {} дней...", days_back + 60);
    let mut price_data: HashMap<String, PriceSeries> = HashMap::new();

    for symbol in &universe[..6] {
        print!("  {}... ", symbol);
        match client
            .get_all_klines(symbol, "D", start_date, end_date)
            .await
        {
            Ok(series) => {
                println!("{} свечей", series.len());
                price_data.insert(symbol.to_string(), series);
            }
            Err(e) => {
                println!("ошибка: {}", e);
            }
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
    }

    if price_data.is_empty() {
        anyhow::bail!("Не удалось загрузить данные");
    }

    println!("\nЗапуск бэктеста...\n");

    // Конфигурация бэктеста
    let backtest_config = BacktestConfig {
        initial_capital,
        commission: 0.001,  // 0.1%
        slippage: 0.0005,   // 0.05%
        rebalance_period: 7, // Еженедельно
        rebalance_threshold: 0.05,
        allow_fractional: true,
    };

    // Конфигурация моментума
    let momentum_config = DualMomentumConfig {
        ts_lookback: 14,
        cs_lookback: 14,
        top_n: 3,
        risk_free_rate: 0.0,
        skip_period: 1,
        equal_weight: true,
    };

    // Конфигурация весов
    let weight_config = WeightConfig {
        target_volatility: 0.30,
        volatility_lookback: 30,
        max_weight: 0.40,
        min_weight: 0.05,
        use_risk_parity: true,
    };

    // Создаём движок бэктеста
    let engine = BacktestEngine::new(backtest_config, momentum_config, weight_config);

    // Запускаем бэктест
    let backtest_start = start_date + Duration::days(30);
    let result = engine.run(&price_data, backtest_start, end_date)?;

    // Выводим результаты
    println!("╔════════════════════════════════════════╗");
    println!("║         РЕЗУЛЬТАТЫ БЭКТЕСТА            ║");
    println!("╠════════════════════════════════════════╣");
    println!("║ Период:           {:>6} дней          ║", days_back);
    println!("╠════════════════════════════════════════╣");
    println!("║ КАПИТАЛ                                ║");
    println!("║   Начальный:      ${:>12.2}       ║", result.initial_capital);
    println!("║   Конечный:       ${:>12.2}       ║", result.final_capital);
    println!("╠════════════════════════════════════════╣");
    println!("║ ДОХОДНОСТЬ                             ║");
    println!("║   Total Return:   {:>12.2}%       ║", result.total_return * 100.0);
    println!("║   CAGR:           {:>12.2}%       ║", result.cagr * 100.0);
    println!("╠════════════════════════════════════════╣");
    println!("║ РИСК                                   ║");
    println!("║   Volatility:     {:>12.2}%       ║", result.volatility * 100.0);
    println!("║   Max Drawdown:   {:>12.2}%       ║", result.max_drawdown * 100.0);
    println!("╠════════════════════════════════════════╣");
    println!("║ RISK-ADJUSTED                          ║");
    println!("║   Sharpe Ratio:   {:>12.2}        ║", result.sharpe_ratio);
    println!("║   Sortino Ratio:  {:>12.2}        ║", result.sortino_ratio);
    println!("║   Calmar Ratio:   {:>12.2}        ║", result.calmar_ratio);
    println!("╠════════════════════════════════════════╣");
    println!("║ АКТИВНОСТЬ                             ║");
    println!("║   Сделок:         {:>12}        ║", result.num_trades);
    println!("║   Комиссии:       ${:>11.2}        ║", result.total_commission);
    println!("║   Turnover:       {:>11.2}x        ║", result.turnover);
    println!("╚════════════════════════════════════════╝");

    // Дополнительные метрики
    if !result.portfolio_history.is_empty() {
        println!("\n--- Дополнительные метрики ---");
        let metrics = calculate_all_metrics(&result.portfolio_history, 365.0);
        println!("{}", metrics);
    }

    // История стоимости портфеля
    if result.portfolio_history.len() > 5 {
        println!("\n--- Динамика портфеля ---");
        println!("{:<20} {:>12} {:>10}", "Дата", "Стоимость", "Cash %");
        println!("{}", "-".repeat(44));

        let step = result.portfolio_history.len() / 5;
        for (i, snapshot) in result.portfolio_history.iter().enumerate() {
            if i % step == 0 || i == result.portfolio_history.len() - 1 {
                println!(
                    "{:<20} ${:>11.2} {:>9.1}%",
                    snapshot.timestamp.format("%Y-%m-%d"),
                    snapshot.value,
                    snapshot.cash_ratio * 100.0
                );
            }
        }
    }

    // Список последних сделок
    if !result.trades.is_empty() {
        println!("\n--- Последние сделки ---");
        println!(
            "{:<20} {:<12} {:>6} {:>10} {:>10}",
            "Дата", "Symbol", "Type", "Qty", "Price"
        );
        println!("{}", "-".repeat(60));

        for trade in result.trades.iter().rev().take(10) {
            let trade_type = if trade.is_buy { "BUY" } else { "SELL" };
            println!(
                "{:<20} {:<12} {:>6} {:>10.4} {:>10.2}",
                trade.timestamp.format("%Y-%m-%d %H:%M"),
                trade.symbol,
                trade_type,
                trade.quantity,
                trade.price
            );
        }
    }

    // Сравнение с Buy & Hold
    println!("\n--- Сравнение с Buy & Hold BTC ---");

    if let Some(btc_series) = price_data.get("BTCUSDT") {
        let btc_closes = btc_series.closes();
        if btc_closes.len() > 30 {
            let btc_start = btc_closes[30];
            let btc_end = btc_closes[btc_closes.len() - 1];
            let btc_return = (btc_end - btc_start) / btc_start;

            println!("  Strategy Return:  {:+.2}%", result.total_return * 100.0);
            println!("  BTC Buy&Hold:     {:+.2}%", btc_return * 100.0);

            let diff = result.total_return - btc_return;
            let label = if diff >= 0.0 { "Alpha" } else { "Underperformance" };
            println!("  {}: {:+.2}%", label, diff * 100.0);
        }
    }

    println!("\nГотово!");

    Ok(())
}
