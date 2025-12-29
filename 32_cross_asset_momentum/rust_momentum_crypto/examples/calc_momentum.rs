//! Пример: Расчёт моментума
//!
//! Запуск: cargo run --example calc_momentum

use momentum_crypto::{
    data::{get_momentum_universe, BybitClient, PriceSeries},
    momentum::{
        simple_momentum, rolling_momentum,
        TimeSeriesMomentum, TimeSeriesMomentumConfig,
        CrossSectionalMomentum, CrossSectionalMomentumConfig,
        DualMomentum, DualMomentumConfig,
    },
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Расчёт моментума ===\n");

    let client = BybitClient::new();
    let universe = get_momentum_universe();

    // Загружаем данные для нескольких активов
    println!("Загрузка данных...");
    let mut price_data: HashMap<String, PriceSeries> = HashMap::new();

    for symbol in &universe[..6] {
        match client.get_klines(symbol, "D", None, None, Some(60)).await {
            Ok(series) => {
                println!("  {} - {} свечей", symbol, series.len());
                price_data.insert(symbol.to_string(), series);
            }
            Err(e) => {
                println!("  {} - ошибка: {}", symbol, e);
            }
        }
        // Rate limiting
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }

    if price_data.is_empty() {
        anyhow::bail!("Не удалось загрузить данные");
    }

    println!("\n--- 1. Простой моментум ---\n");

    for (symbol, series) in &price_data {
        let closes = series.closes();

        // Моментум за 7 дней
        if let Some(mom_7d) = simple_momentum(&closes, 7) {
            println!("{}: 7d = {:+.2}%", symbol, mom_7d * 100.0);
        }

        // Моментум за 30 дней
        if let Some(mom_30d) = simple_momentum(&closes, 30) {
            println!("{}: 30d = {:+.2}%", symbol, mom_30d * 100.0);
        }

        println!();
    }

    println!("--- 2. Time-Series Momentum ---\n");

    let ts_config = TimeSeriesMomentumConfig {
        lookback: 30,
        skip_period: 1,
        threshold: 0.0,
        risk_free_rate: 0.0,
    };
    let ts_momentum = TimeSeriesMomentum::new(ts_config);

    for (symbol, series) in &price_data {
        if let Ok(Some(momentum)) = ts_momentum.current_momentum(series) {
            let signal = ts_momentum.signal(momentum);
            println!(
                "{}: momentum = {:+.2}%, signal = {:?}",
                symbol,
                momentum * 100.0,
                signal
            );
        }
    }

    println!("\n--- 3. Cross-Sectional Momentum ---\n");

    let cs_config = CrossSectionalMomentumConfig {
        lookback: 30,
        top_n: 3,
        bottom_n: 0,
        use_percentile: false,
        long_percentile: 0.66,
        short_percentile: 0.33,
    };
    let cs_momentum = CrossSectionalMomentum::new(cs_config);

    let ranked = cs_momentum.rank_assets(&price_data)?;

    println!("{:<12} {:>10} {:>6} {:>10}", "Symbol", "Momentum", "Rank", "Percentile");
    println!("{}", "-".repeat(42));

    for asset in &ranked {
        println!(
            "{:<12} {:>9.2}% {:>6} {:>9.0}%",
            asset.symbol,
            asset.momentum * 100.0,
            asset.rank,
            asset.percentile * 100.0
        );
    }

    println!("\n--- 4. Dual Momentum ---\n");

    let dual_config = DualMomentumConfig {
        ts_lookback: 30,
        cs_lookback: 30,
        top_n: 3,
        risk_free_rate: 0.0,
        skip_period: 1,
        equal_weight: true,
    };
    let dual_momentum = DualMomentum::new(dual_config);

    let analysis = dual_momentum.analyze(&price_data)?;

    println!(
        "{:<12} {:>10} {:>8} {:>8} {:>10}",
        "Symbol", "Momentum", "TS Pass", "CS Rank", "Weight"
    );
    println!("{}", "-".repeat(52));

    for result in &analysis {
        let ts_pass = if result.ts_passed { "YES" } else { "NO" };
        let rank_str = if result.cs_rank == usize::MAX {
            "-".to_string()
        } else {
            result.cs_rank.to_string()
        };
        let weight_str = if result.weight > 0.0 {
            format!("{:.1}%", result.weight * 100.0)
        } else {
            "-".to_string()
        };

        let marker = if result.selected { " <--" } else { "" };

        println!(
            "{:<12} {:>9.2}% {:>8} {:>8} {:>10}{}",
            result.symbol,
            result.ts_momentum * 100.0,
            ts_pass,
            rank_str,
            weight_str,
            marker
        );
    }

    // Выводим итоговые рекомендации
    let selected: Vec<_> = analysis.iter().filter(|a| a.selected).collect();

    println!("\n=== Рекомендации ===\n");

    if selected.is_empty() {
        println!("Все активы имеют отрицательный моментум.");
        println!("Рекомендация: оставаться в кеше (USDT/USDC)");
    } else {
        println!("Портфель:");
        for result in &selected {
            println!("  {} - {:.1}%", result.symbol, result.weight * 100.0);
        }

        let cash_weight = 1.0 - selected.iter().map(|r| r.weight).sum::<f64>();
        if cash_weight > 0.01 {
            println!("  CASH - {:.1}%", cash_weight * 100.0);
        }
    }

    println!("\nГотово!");

    Ok(())
}
