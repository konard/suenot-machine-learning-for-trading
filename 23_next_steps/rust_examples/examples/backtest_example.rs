//! Пример бэктестинга стратегии
//!
//! Запуск: cargo run --example backtest_example

use anyhow::Result;
use ml4t_bybit::client::BybitClient;
use ml4t_bybit::data::Interval;
use ml4t_bybit::backtest::{BacktestConfig, BacktestEngine};
use ml4t_bybit::strategies::{SmaCrossStrategy, RsiStrategy, Strategy};

#[tokio::main]
async fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════╗");
    println!("║         Strategy Backtest Example          ║");
    println!("╚════════════════════════════════════════════╝");
    println!();

    let client = BybitClient::new();

    // Получаем исторические данные для бэктеста
    println!("📥 Загрузка исторических данных BTCUSDT...");
    let klines = client
        .get_klines("BTCUSDT", Interval::H1, Some(500))
        .await?;

    println!("   Загружено {} свечей", klines.len());

    if let (Some(first), Some(last)) = (klines.first(), klines.last()) {
        let start_date = chrono::DateTime::from_timestamp_millis(first.timestamp as i64)
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_default();
        let end_date = chrono::DateTime::from_timestamp_millis(last.timestamp as i64)
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_default();

        println!("   Период: {} - {}", start_date, end_date);
        println!("   Начальная цена: ${:.2}", first.close);
        println!("   Конечная цена:  ${:.2}", last.close);

        let buy_hold_return = (last.close - first.close) / first.close * 100.0;
        println!("   Buy & Hold:     {:.2}%", buy_hold_return);
    }

    println!();

    // Конфигурация бэктеста
    let config = BacktestConfig {
        initial_capital: 10000.0,
        commission: 0.001,   // 0.1%
        slippage: 0.0005,    // 0.05%
        position_size: 1.0,  // 100% капитала
        long_only: true,     // Только лонг (спот)
        stop_loss: None,
        take_profit: None,
    };

    let engine = BacktestEngine::new(config.clone());

    // Тестируем разные стратегии
    println!("═══════════════════════════════════════════════");
    println!("          Сравнение стратегий                  ");
    println!("═══════════════════════════════════════════════");
    println!();

    // 1. SMA Crossover (10/20)
    println!("📊 Стратегия 1: SMA Crossover (10/20)");
    println!("─────────────────────────────────────────────");
    let sma_strategy = SmaCrossStrategy::new(10, 20);
    let sma_result = engine.run(&sma_strategy, &klines);
    sma_result.print_report();
    println!();

    // 2. SMA Crossover (20/50)
    println!("📊 Стратегия 2: SMA Crossover (20/50)");
    println!("─────────────────────────────────────────────");
    let sma_strategy_2 = SmaCrossStrategy::new(20, 50);
    let sma_result_2 = engine.run(&sma_strategy_2, &klines);
    sma_result_2.print_report();
    println!();

    // 3. RSI Strategy
    println!("📊 Стратегия 3: RSI (14) 30/70");
    println!("─────────────────────────────────────────────");
    let rsi_strategy = RsiStrategy::standard();
    let rsi_result = engine.run(&rsi_strategy, &klines);
    rsi_result.print_report();
    println!();

    // 4. RSI с более агрессивными уровнями
    println!("📊 Стратегия 4: RSI (14) 20/80 (агрессивная)");
    println!("─────────────────────────────────────────────");
    let rsi_aggressive = RsiStrategy::aggressive();
    let rsi_agg_result = engine.run(&rsi_aggressive, &klines);
    rsi_agg_result.print_report();
    println!();

    // Сравнительная таблица
    println!("═══════════════════════════════════════════════");
    println!("            Сравнительная таблица              ");
    println!("═══════════════════════════════════════════════");
    println!();

    println!("┌─────────────────────┬──────────┬────────┬──────────┬────────┬─────────┐");
    println!("│      Strategy       │  Return  │ Trades │ Win Rate │ Sharpe │ Max DD  │");
    println!("├─────────────────────┼──────────┼────────┼──────────┼────────┼─────────┤");

    let strategies_results = vec![
        ("SMA (10/20)", &sma_result),
        ("SMA (20/50)", &sma_result_2),
        ("RSI (30/70)", &rsi_result),
        ("RSI (20/80)", &rsi_agg_result),
    ];

    for (name, result) in &strategies_results {
        let metrics = result.performance_metrics();
        println!(
            "│ {:>19} │ {:>7.2}% │ {:>6} │ {:>7.1}% │ {:>6.2} │ {:>6.2}% │",
            name,
            metrics.total_return,
            metrics.total_trades,
            metrics.win_rate,
            metrics.sharpe_ratio,
            metrics.max_drawdown
        );
    }

    println!("└─────────────────────┴──────────┴────────┴──────────┴────────┴─────────┘");
    println!();

    // Находим лучшую стратегию
    let best = strategies_results
        .iter()
        .max_by(|a, b| {
            a.1.total_return()
                .partial_cmp(&b.1.total_return())
                .unwrap()
        });

    if let Some((name, result)) = best {
        println!("🏆 Лучшая стратегия: {} с доходностью {:.2}%",
            name, result.total_return_percent());
    }

    println!();

    // Анализ кривой эквити лучшей стратегии
    println!("═══════════════════════════════════════════════");
    println!("     Анализ кривой эквити (SMA 10/20)          ");
    println!("═══════════════════════════════════════════════");
    println!();

    // Показываем мини-график кривой эквити
    let equity = &sma_result.equity_curve;
    if equity.len() > 10 {
        let step = equity.len() / 20;
        let sampled: Vec<f64> = equity.iter().step_by(step.max(1)).cloned().collect();

        let min_eq = sampled.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_eq = sampled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_eq - min_eq;

        if range > 0.0 {
            println!("  ${:.0} ┤", max_eq);

            for row in (0..5).rev() {
                let threshold = min_eq + range * (row as f64 + 0.5) / 5.0;
                let mut line = String::new();

                for &eq in &sampled {
                    if eq >= threshold {
                        line.push('█');
                    } else {
                        line.push(' ');
                    }
                }

                let value = min_eq + range * (row as f64 + 0.5) / 5.0;
                println!("  ${:.0} ┤{}", value, line);
            }

            println!("  ${:.0} ┴{}", min_eq, "─".repeat(sampled.len()));
            println!("        Start{:^width$}End", "", width = sampled.len() - 8);
        }
    }

    println!();
    println!("⚠️  ВАЖНО:");
    println!("   • Прошлые результаты не гарантируют будущую прибыль");
    println!("   • Бэктест не учитывает все рыночные условия");
    println!("   • Всегда тестируйте на бумажной торговле перед реальной");
    println!();

    Ok(())
}
