//! CLI для Momentum Crypto
//!
//! Запуск: cargo run -- <command>

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::{Parser, Subcommand};
use momentum_crypto::{
    backtest::{BacktestConfig, BacktestEngine},
    data::{get_momentum_universe, BybitClient, PriceSeries},
    momentum::{DualMomentum, DualMomentumConfig},
    strategy::WeightConfig,
    utils::StrategyConfig,
};
use std::collections::HashMap;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "momentum-crypto")]
#[command(about = "Cross-asset momentum strategy for cryptocurrencies")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Получить текущие цены
    Prices {
        /// Символы для проверки (через запятую)
        #[arg(short, long, default_value = "BTCUSDT,ETHUSDT,SOLUSDT")]
        symbols: String,
    },

    /// Рассчитать моментум для всех активов
    Momentum {
        /// Период lookback (дни)
        #[arg(short, long, default_value = "30")]
        lookback: usize,
        /// Количество топ активов для отображения
        #[arg(short, long, default_value = "5")]
        top: usize,
    },

    /// Сгенерировать сигналы
    Signals {
        /// Количество активов для покупки
        #[arg(short, long, default_value = "3")]
        top_n: usize,
    },

    /// Запустить бэктест
    Backtest {
        /// Количество дней для бэктеста
        #[arg(short, long, default_value = "90")]
        days: i64,
        /// Начальный капитал
        #[arg(short, long, default_value = "10000")]
        capital: f64,
    },

    /// Создать файл конфигурации
    Config {
        /// Путь для сохранения
        #[arg(short, long, default_value = "config.json")]
        output: String,
        /// Тип конфигурации (default, aggressive, conservative)
        #[arg(short, long, default_value = "default")]
        preset: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Инициализация логирования
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let cli = Cli::parse();

    match cli.command {
        Commands::Prices { symbols } => {
            cmd_prices(&symbols).await?;
        }
        Commands::Momentum { lookback, top } => {
            cmd_momentum(lookback, top).await?;
        }
        Commands::Signals { top_n } => {
            cmd_signals(top_n).await?;
        }
        Commands::Backtest { days, capital } => {
            cmd_backtest(days, capital).await?;
        }
        Commands::Config { output, preset } => {
            cmd_config(&output, &preset)?;
        }
    }

    Ok(())
}

async fn cmd_prices(symbols_str: &str) -> Result<()> {
    let client = BybitClient::new();
    let symbols: Vec<&str> = symbols_str.split(',').collect();

    info!("Получение цен для {} символов...", symbols.len());

    let tickers = client.get_tickers(&symbols).await?;

    println!("\n{:<12} {:>12} {:>10}", "Symbol", "Price", "24h %");
    println!("{}", "-".repeat(36));

    for (symbol, ticker) in tickers {
        let price: f64 = ticker.last_price.parse().unwrap_or(0.0);
        let change: f64 = ticker.price_24h_pcnt.parse().unwrap_or(0.0);
        let change_pct = change * 100.0;

        let change_str = if change_pct >= 0.0 {
            format!("+{:.2}%", change_pct)
        } else {
            format!("{:.2}%", change_pct)
        };

        println!("{:<12} {:>12.2} {:>10}", symbol, price, change_str);
    }

    Ok(())
}

async fn cmd_momentum(lookback: usize, top: usize) -> Result<()> {
    let client = BybitClient::new();
    let universe = get_momentum_universe();

    info!("Загрузка данных для {} активов...", universe.len());

    let mut price_data: HashMap<String, PriceSeries> = HashMap::new();

    for symbol in universe {
        info!("Загрузка {}...", symbol);
        match client.get_klines(symbol, "D", None, None, Some(lookback as u32 + 10)).await {
            Ok(series) => {
                if series.len() > lookback {
                    price_data.insert(symbol.to_string(), series);
                }
            }
            Err(e) => {
                info!("Ошибка загрузки {}: {}", symbol, e);
            }
        }
        // Rate limiting
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }

    // Рассчитываем моментум
    let mut momentum_values: Vec<(String, f64)> = Vec::new();

    for (symbol, series) in &price_data {
        let closes = series.closes();
        if closes.len() > lookback {
            let current = closes[closes.len() - 1];
            let past = closes[closes.len() - 1 - lookback];
            let momentum = (current - past) / past;
            momentum_values.push((symbol.clone(), momentum));
        }
    }

    // Сортируем по убыванию
    momentum_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nМоментум за {} дней (топ {}):", lookback, top);
    println!("{:<12} {:>12} {:>10}", "Symbol", "Momentum", "Signal");
    println!("{}", "-".repeat(36));

    for (i, (symbol, momentum)) in momentum_values.iter().take(top).enumerate() {
        let signal = if *momentum > 0.0 { "LONG" } else { "CASH" };
        let rank_indicator = if i < 3 { "*" } else { "" };

        println!(
            "{:<12} {:>11.2}% {:>10}{}",
            symbol,
            momentum * 100.0,
            signal,
            rank_indicator
        );
    }

    println!("\n* = выбрано для портфеля");

    Ok(())
}

async fn cmd_signals(top_n: usize) -> Result<()> {
    let client = BybitClient::new();
    let universe = get_momentum_universe();

    info!("Загрузка данных...");

    let mut price_data: HashMap<String, PriceSeries> = HashMap::new();

    for symbol in universe {
        match client.get_klines(symbol, "D", None, None, Some(60)).await {
            Ok(series) => {
                price_data.insert(symbol.to_string(), series);
            }
            Err(_) => {}
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }

    let config = DualMomentumConfig {
        ts_lookback: 30,
        cs_lookback: 30,
        top_n,
        risk_free_rate: 0.0,
        skip_period: 1,
        equal_weight: true,
    };

    let strategy = DualMomentum::new(config);
    let analysis = strategy.analyze(&price_data)?;

    println!("\nDual Momentum Signals (top_n = {}):", top_n);
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

        let indicator = if result.selected { " <--" } else { "" };

        println!(
            "{:<12} {:>9.2}% {:>8} {:>8} {:>10}{}",
            result.symbol,
            result.ts_momentum * 100.0,
            ts_pass,
            rank_str,
            weight_str,
            indicator
        );
    }

    let selected: Vec<_> = analysis.iter().filter(|a| a.selected).collect();
    if !selected.is_empty() {
        println!("\nВыбранные активы:");
        for result in selected {
            println!("  {} (вес: {:.1}%)", result.symbol, result.weight * 100.0);
        }
    } else {
        println!("\nВсе активы имеют отрицательный моментум - CASH");
    }

    Ok(())
}

async fn cmd_backtest(days: i64, capital: f64) -> Result<()> {
    let client = BybitClient::new();
    let universe = get_momentum_universe();

    info!("Загрузка данных для бэктеста ({} дней)...", days);

    let end_date = Utc::now();
    let start_date = end_date - Duration::days(days + 60); // +60 для lookback

    let mut price_data: HashMap<String, PriceSeries> = HashMap::new();

    for symbol in &universe[..6] {
        // Берём только 6 активов для примера
        info!("Загрузка {}...", symbol);
        match client
            .get_all_klines(symbol, "D", start_date, end_date)
            .await
        {
            Ok(series) => {
                price_data.insert(symbol.to_string(), series);
            }
            Err(e) => {
                info!("Ошибка загрузки {}: {}", symbol, e);
            }
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
    }

    if price_data.is_empty() {
        anyhow::bail!("Не удалось загрузить данные");
    }

    info!("Запуск бэктеста...");

    let backtest_config = BacktestConfig {
        initial_capital: capital,
        commission: 0.001,
        slippage: 0.0005,
        rebalance_period: 7,
        rebalance_threshold: 0.05,
        allow_fractional: true,
    };

    let momentum_config = DualMomentumConfig {
        ts_lookback: 14,
        cs_lookback: 14,
        top_n: 3,
        risk_free_rate: 0.0,
        skip_period: 1,
        equal_weight: true,
    };

    let weight_config = WeightConfig::default();

    let engine = BacktestEngine::new(backtest_config, momentum_config, weight_config);

    let backtest_start = start_date + Duration::days(30);
    let result = engine.run(&price_data, backtest_start, end_date)?;

    println!("\n========== BACKTEST RESULTS ==========");
    println!("Period: {} days", days);
    println!("Initial Capital: ${:.2}", result.initial_capital);
    println!("Final Capital: ${:.2}", result.final_capital);
    println!();
    println!("Total Return: {:.2}%", result.total_return * 100.0);
    println!("CAGR: {:.2}%", result.cagr * 100.0);
    println!("Volatility: {:.2}%", result.volatility * 100.0);
    println!();
    println!("Sharpe Ratio: {:.2}", result.sharpe_ratio);
    println!("Sortino Ratio: {:.2}", result.sortino_ratio);
    println!("Max Drawdown: {:.2}%", result.max_drawdown * 100.0);
    println!("Calmar Ratio: {:.2}", result.calmar_ratio);
    println!();
    println!("Number of Trades: {}", result.num_trades);
    println!("Total Commission: ${:.2}", result.total_commission);
    println!("Turnover: {:.2}x", result.turnover);
    println!("=======================================\n");

    Ok(())
}

fn cmd_config(output: &str, preset: &str) -> Result<()> {
    let config = match preset {
        "aggressive" => StrategyConfig::aggressive(),
        "conservative" => StrategyConfig::conservative(),
        _ => StrategyConfig::default(),
    };

    config.to_file(output)?;
    info!("Конфигурация сохранена в {}", output);

    println!("Создан файл конфигурации: {}", output);
    println!("Пресет: {}", preset);
    println!("Вселенная активов: {:?}", config.universe);

    Ok(())
}
