//! Пример: Анализ и прогнозирование волатильности (GARCH)
//!
//! Строит GARCH модель для прогнозирования волатильности.
//!
//! Использование:
//! ```
//! cargo run --bin volatility_analysis -- --file data/BTCUSDT_1h.csv
//! ```

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use crypto_time_series::api::load_candles;
use crypto_time_series::models::{GarchModel, GarchParams, arch_test};
use crypto_time_series::analysis::DescriptiveStats;
use crypto_time_series::TimeSeries;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "GARCH volatility analysis")]
struct Args {
    /// Input CSV file with candle data
    #[arg(short, long)]
    file: PathBuf,

    /// GARCH order p
    #[arg(short, long, default_value = "1")]
    p: usize,

    /// ARCH order q
    #[arg(short, long, default_value = "1")]
    q: usize,

    /// Forecast horizon
    #[arg(short = 'H', long, default_value = "10")]
    horizon: usize,

    /// Use percentage returns
    #[arg(long)]
    pct_returns: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "=".repeat(60).blue());
    println!("{}", "GARCH Volatility Analysis".bold().blue());
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
    let returns: Vec<f64> = if args.pct_returns {
        ts.returns().iter().map(|r| r * 100.0).collect()
    } else {
        ts.returns()
    };

    if returns.len() < 100 {
        anyhow::bail!("Not enough data for GARCH modeling (need at least 100 returns)");
    }

    let unit = if args.pct_returns { "%" } else { "decimal" };
    println!(
        "{} {} returns ({})\n",
        "Analyzing:".cyan(),
        returns.len(),
        unit
    );

    // Статистика доходностей
    println!("{}", "Return Statistics".bold());
    println!("{}", "-".repeat(40));
    let stats = DescriptiveStats::new(&returns);
    println!("{}\n", stats.display());

    // Проверка на ARCH эффекты
    println!("{}", "ARCH Effects Test".bold());
    println!("{}", "-".repeat(40));

    for lags in [5, 10, 15] {
        let arch = arch_test(&returns, lags);
        let status = if arch.is_significant {
            "ARCH effects detected".yellow()
        } else {
            "No ARCH effects".green()
        };
        println!(
            "  Lag {}: statistic={:.2}, p-value={:.4} - {}",
            lags, arch.statistic, arch.p_value, status
        );
    }

    // Строим GARCH модель
    println!("\n{}", format!("Fitting GARCH({},{})", args.p, args.q).bold());
    println!("{}", "-".repeat(40));

    let params = GarchParams::new(args.p, args.q);
    let model = GarchModel::fit(&returns, params)
        .ok_or_else(|| anyhow::anyhow!("Failed to fit GARCH model"))?;

    println!("{}", model.summary());

    // Анализ волатильности
    println!("\n{}", "Volatility Analysis".bold());
    println!("{}", "-".repeat(40));

    let conditional_vol: Vec<f64> = model.conditional_var.iter().map(|v| v.sqrt()).collect();
    let vol_stats = DescriptiveStats::new(&conditional_vol);

    println!("Conditional Volatility Stats:");
    println!("  Mean: {:.4}", vol_stats.mean);
    println!("  Std: {:.4}", vol_stats.std);
    println!("  Min: {:.4}", vol_stats.min);
    println!("  Max: {:.4}", vol_stats.max);

    // Текущий уровень волатильности
    let current_vol = *conditional_vol.last().unwrap_or(&0.0);
    let vol_percentile = conditional_vol
        .iter()
        .filter(|&&v| v <= current_vol)
        .count() as f64
        / conditional_vol.len() as f64
        * 100.0;

    println!("\nCurrent volatility: {:.4} ({:.0}th percentile)", current_vol, vol_percentile);

    let vol_regime = if vol_percentile > 80.0 {
        "HIGH".red()
    } else if vol_percentile > 50.0 {
        "MEDIUM".yellow()
    } else {
        "LOW".green()
    };
    println!("Volatility regime: {}", vol_regime);

    // Прогноз волатильности
    println!("\n{} (horizon = {})", "Volatility Forecast".bold(), args.horizon);
    println!("{}", "-".repeat(40));

    let vol_forecast = model.forecast_volatility(&returns, args.horizon);

    println!("{:>4} {:>12}", "Step", "Volatility");
    println!("{}", "-".repeat(20));

    for (i, &vol) in vol_forecast.iter().enumerate() {
        let direction = if i > 0 && vol > vol_forecast[i - 1] {
            "↑".red()
        } else if i > 0 && vol < vol_forecast[i - 1] {
            "↓".green()
        } else {
            " ".normal()
        };
        println!("{:>4} {:>12.4} {}", i + 1, vol, direction);
    }

    // Долгосрочная волатильность
    if model.is_stable() {
        let long_run_vol = (model.omega / (1.0 - model.persistence())).sqrt();
        println!("\nLong-run volatility: {:.4}", long_run_vol);

        if let Some(hl) = model.half_life() {
            println!("Half-life: {:.1} periods", hl);
            println!("(Time for volatility to revert halfway to long-run level)");
        }
    }

    // VaR оценка
    println!("\n{}", "Value at Risk Estimates".bold());
    println!("{}", "-".repeat(40));

    let z_95 = 1.645;
    let z_99 = 2.326;

    let var_95_1d = current_vol * z_95;
    let var_99_1d = current_vol * z_99;

    println!("Based on current volatility ({:.4}):", current_vol);
    println!("  1-period VaR (95%): {:.4}{}", var_95_1d, unit);
    println!("  1-period VaR (99%): {:.4}{}", var_99_1d, unit);

    if !vol_forecast.is_empty() {
        let avg_forecast_vol: f64 = vol_forecast.iter().sum::<f64>() / vol_forecast.len() as f64;
        let var_95_forecast = avg_forecast_vol * z_95 * (args.horizon as f64).sqrt();
        let var_99_forecast = avg_forecast_vol * z_99 * (args.horizon as f64).sqrt();

        println!("\nBased on {}-period forecast:", args.horizon);
        println!("  {}-period VaR (95%): {:.4}{}", args.horizon, var_95_forecast, unit);
        println!("  {}-period VaR (99%): {:.4}{}", args.horizon, var_99_forecast, unit);
    }

    // Volatility clustering visualization (text)
    println!("\n{}", "Recent Volatility Pattern".bold());
    println!("{}", "-".repeat(40));

    let recent = conditional_vol.len().saturating_sub(20);
    let max_vol = conditional_vol[recent..].iter().cloned().fold(0.0, f64::max);

    for (i, &vol) in conditional_vol[recent..].iter().enumerate() {
        let bar_len = ((vol / max_vol) * 30.0) as usize;
        let bar = "#".repeat(bar_len);
        println!("{:>3}: {:>8.4} |{}", recent + i, vol, bar);
    }

    println!("\n{}", "Done!".green().bold());
    Ok(())
}
