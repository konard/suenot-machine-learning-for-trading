//! Пример: Анализ стационарности временного ряда
//!
//! Загружает данные из CSV и проводит тесты на стационарность.
//!
//! Использование:
//! ```
//! cargo run --bin analyze_stationarity -- --file data/BTCUSDT_1h.csv
//! ```

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use crypto_time_series::api::load_candles;
use crypto_time_series::analysis::{
    adf_test, kpss_test, rolling_stationarity_check,
    acf, pacf, ljung_box_test, plot_acf_text,
    DescriptiveStats,
};
use crypto_time_series::TimeSeries;
use std::path::PathBuf;
use tabled::{Table, Tabled};

#[derive(Parser, Debug)]
#[command(author, version, about = "Analyze time series stationarity")]
struct Args {
    /// Input CSV file with candle data
    #[arg(short, long)]
    file: PathBuf,

    /// Maximum lag for ACF/PACF
    #[arg(short, long, default_value = "20")]
    max_lag: usize,

    /// Use returns instead of prices
    #[arg(short, long)]
    returns: bool,

    /// Use log returns
    #[arg(long)]
    log_returns: bool,
}

#[derive(Tabled)]
struct TestResultRow {
    #[tabled(rename = "Test")]
    test: String,
    #[tabled(rename = "Statistic")]
    statistic: String,
    #[tabled(rename = "P-Value")]
    p_value: String,
    #[tabled(rename = "Result")]
    result: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "=".repeat(60).blue());
    println!("{}", "Stationarity Analysis".bold().blue());
    println!("{}", "=".repeat(60).blue());

    // Загружаем данные
    let candles = load_candles(&args.file)?;
    println!(
        "\n{} {} candles from {}",
        "Loaded:".green(),
        candles.len(),
        args.file.display()
    );

    let ts = TimeSeries::from_candles("data", &candles);

    // Выбираем данные для анализа
    let (data, data_name) = if args.log_returns {
        (ts.log_returns(), "Log Returns")
    } else if args.returns {
        (ts.returns(), "Returns")
    } else {
        (ts.data.clone(), "Prices")
    };

    println!("{} {}\n", "Analyzing:".cyan(), data_name);

    if data.len() < 50 {
        anyhow::bail!("Not enough data points for analysis");
    }

    // Описательная статистика
    println!("{}", "Descriptive Statistics".bold());
    println!("{}", "-".repeat(40));
    let stats = DescriptiveStats::new(&data);
    println!("{}\n", stats.display());

    // Тесты на стационарность
    println!("{}", "Stationarity Tests".bold());
    println!("{}", "-".repeat(40));

    let adf = adf_test(&data, None);
    let kpss_c = kpss_test(&data, false);
    let kpss_t = kpss_test(&data, true);

    let test_results = vec![
        TestResultRow {
            test: "ADF (H0: unit root)".to_string(),
            statistic: format!("{:.4}", adf.statistic),
            p_value: format!("{:.4}", adf.p_value),
            result: if adf.is_significant {
                "Stationary".green().to_string()
            } else {
                "Non-stationary".red().to_string()
            },
        },
        TestResultRow {
            test: "KPSS-c (H0: stationary)".to_string(),
            statistic: format!("{:.4}", kpss_c.statistic),
            p_value: format!("{:.4}", kpss_c.p_value),
            result: if !kpss_c.is_significant {
                "Stationary".green().to_string()
            } else {
                "Non-stationary".red().to_string()
            },
        },
        TestResultRow {
            test: "KPSS-t (H0: trend stat.)".to_string(),
            statistic: format!("{:.4}", kpss_t.statistic),
            p_value: format!("{:.4}", kpss_t.p_value),
            result: if !kpss_t.is_significant {
                "Stationary".green().to_string()
            } else {
                "Non-stationary".red().to_string()
            },
        },
    ];

    let table = Table::new(test_results).to_string();
    println!("{}\n", table);

    // Вывод интерпретации
    println!("{}", "Interpretation:".bold());
    if adf.is_significant && !kpss_c.is_significant {
        println!("  {} The series appears to be stationary.", "✓".green());
    } else if !adf.is_significant && kpss_c.is_significant {
        println!("  {} The series is non-stationary (has unit root).", "✗".red());
        println!("  {} Consider differencing the data.", "→".yellow());
    } else {
        println!("  {} Results are inconclusive.", "?".yellow());
        println!("  {} ADF and KPSS give different conclusions.", "→".yellow());
    }

    // Скользящая проверка стационарности
    println!("\n{}", "Rolling Stationarity Check".bold());
    println!("{}", "-".repeat(40));

    let window = data.len() / 10;
    let rolling = rolling_stationarity_check(&data, window.max(20));

    println!("  Window size: {}", window);
    println!(
        "  Mean stability: {} (variation: {:.4})",
        if rolling.is_stable_mean {
            "Stable".green()
        } else {
            "Unstable".red()
        },
        rolling.mean_variation
    );
    println!(
        "  Variance stability: {} (variation: {:.4})",
        if rolling.is_stable_variance {
            "Stable".green()
        } else {
            "Unstable".red()
        },
        rolling.variance_variation
    );

    // ACF и PACF
    println!("\n{}", "Autocorrelation Analysis".bold());
    println!("{}", "-".repeat(40));

    let acf_values = acf(&data, args.max_lag);
    let pacf_values = pacf(&data, args.max_lag);

    println!("{}", plot_acf_text(&acf_values, "ACF", 30));
    println!("{}", plot_acf_text(&pacf_values, "PACF", 30));

    // Тест Льюнг-Бокса
    let lb = ljung_box_test(&data, args.max_lag);
    println!(
        "\nLjung-Box test (lag={}): Q={:.2}, p-value={:.4}",
        args.max_lag, lb.statistic, lb.p_value
    );
    if lb.is_significant {
        println!(
            "  {} Significant autocorrelation detected",
            "→".yellow()
        );
    } else {
        println!(
            "  {} No significant autocorrelation (white noise)",
            "✓".green()
        );
    }

    // Рекомендации
    println!("\n{}", "Recommendations".bold());
    println!("{}", "-".repeat(40));

    if !adf.is_significant {
        println!("  1. Difference the series to achieve stationarity");
        println!("  2. Consider using ARIMA with d=1");
    }

    let significant_acf: Vec<_> = acf_values
        .iter()
        .skip(1)
        .enumerate()
        .filter(|(_, &v)| v.abs() > 1.96 / (data.len() as f64).sqrt())
        .collect();

    let significant_pacf: Vec<_> = pacf_values
        .iter()
        .skip(1)
        .enumerate()
        .filter(|(_, &v)| v.abs() > 1.96 / (data.len() as f64).sqrt())
        .collect();

    if !significant_pacf.is_empty() {
        let max_lag = significant_pacf.iter().map(|(i, _)| i + 1).max().unwrap_or(1);
        println!("  3. Consider AR({}) model based on PACF", max_lag);
    }

    if !significant_acf.is_empty() {
        let max_lag = significant_acf.iter().map(|(i, _)| i + 1).max().unwrap_or(1);
        println!("  4. Consider MA({}) model based on ACF", max_lag);
    }

    println!("\n{}", "Done!".green().bold());
    Ok(())
}
