//! # Fetch Data Example
//!
//! Пример загрузки данных с биржи Bybit и сохранения в CSV.
//!
//! ```bash
//! cargo run --example fetch_data -- --symbol BTCUSDT --interval 1h --limit 1000
//! ```

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::Parser;
use crypto_autoencoders::{utils, BybitClient, DataProcessor, NormalizationMethod};
use std::path::PathBuf;

/// Загрузка криптовалютных данных с Bybit
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Торговая пара (например, BTCUSDT, ETHUSDT)
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    /// Интервал свечей (1m, 5m, 15m, 1h, 4h, 1d)
    #[arg(short, long, default_value = "1h")]
    interval: String,

    /// Количество свечей для загрузки
    #[arg(short, long, default_value_t = 1000)]
    limit: u32,

    /// Папка для сохранения данных
    #[arg(short, long, default_value = "data")]
    output_dir: PathBuf,

    /// Извлечь признаки из данных
    #[arg(long, default_value_t = true)]
    extract_features: bool,

    /// Загрузить несколько символов
    #[arg(long)]
    multi_symbol: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    // Создаем папку для данных
    std::fs::create_dir_all(&args.output_dir)?;

    let client = BybitClient::new();

    if args.multi_symbol {
        // Загружаем данные для нескольких популярных пар
        let symbols = vec![
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "DOTUSDT",
        ];

        println!("Загрузка данных для {} символов...", symbols.len());

        for symbol in symbols {
            match fetch_and_save(&client, symbol, &args).await {
                Ok(_) => println!("  {} - OK", symbol),
                Err(e) => println!("  {} - ОШИБКА: {}", symbol, e),
            }
        }
    } else {
        fetch_and_save(&client, &args.symbol, &args).await?;
    }

    println!("\nГотово! Данные сохранены в {:?}", args.output_dir);

    Ok(())
}

async fn fetch_and_save(client: &BybitClient, symbol: &str, args: &Args) -> Result<()> {
    println!("Загрузка {} ({}, {} свечей)...", symbol, args.interval, args.limit);

    // Загружаем свечи
    let klines = client.get_klines(symbol, &args.interval, args.limit).await?;

    if klines.is_empty() {
        println!("  Нет данных для {}", symbol);
        return Ok(());
    }

    println!(
        "  Загружено {} свечей с {} по {}",
        klines.len(),
        utils::format_timestamp(klines.first().unwrap().open_time),
        utils::format_timestamp(klines.last().unwrap().open_time)
    );

    // Показываем базовую статистику
    let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();

    let price_min = closes.iter().copied().fold(f64::INFINITY, f64::min);
    let price_max = closes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let price_avg = closes.iter().sum::<f64>() / closes.len() as f64;
    let volume_avg = volumes.iter().sum::<f64>() / volumes.len() as f64;

    println!("  Цена: min={:.2}, max={:.2}, avg={:.2}", price_min, price_max, price_avg);
    println!("  Средний объем: {}", utils::format_number(volume_avg));

    // Сохраняем сырые данные
    let klines_path = args.output_dir.join(format!("{}_{}_klines.csv", symbol.to_lowercase(), args.interval));
    utils::save_klines(&klines, &klines_path)?;
    println!("  Свечи сохранены в {:?}", klines_path);

    // Извлекаем и сохраняем признаки
    if args.extract_features {
        let mut processor = DataProcessor::new()
            .with_lookback(20)
            .with_normalization(NormalizationMethod::MinMax);

        let features = processor.extract_features(&klines);

        if features.nrows() > 0 {
            let normalized = processor.normalize(&features);

            let features_path = args.output_dir.join(format!(
                "{}_{}_features.csv",
                symbol.to_lowercase(),
                args.interval
            ));
            utils::save_features(&normalized, &features_path)?;
            println!(
                "  Признаки ({} samples x {} features) сохранены в {:?}",
                normalized.nrows(),
                normalized.ncols(),
                features_path
            );
        }
    }

    // Загружаем текущий тикер
    match client.get_ticker(symbol).await {
        Ok(ticker) => {
            println!("  Текущая цена: {} (24h: {:.2}%)",
                utils::format_number(ticker.last_price),
                ticker.price_24h_pcnt * 100.0
            );
        }
        Err(e) => {
            println!("  Не удалось получить тикер: {}", e);
        }
    }

    Ok(())
}
