//! Пример: Тест на коинтеграцию пары криптовалют
//!
//! Проверяет коинтеграцию между двумя криптовалютами.
//!
//! Использование:
//! ```
//! cargo run --bin cointegration_test -- --file1 data/BTCUSDT_1h.csv --file2 data/ETHUSDT_1h.csv
//! ```

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use crypto_time_series::api::load_candles;
use crypto_time_series::trading::{
    engle_granger_test, compute_spread, spread_zscore, analyze_pair, compute_spread_bands,
};
use crypto_time_series::analysis::{adf_test, correlation, DescriptiveStats};
use crypto_time_series::TimeSeries;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Test cointegration between crypto pairs")]
struct Args {
    /// First asset CSV file
    #[arg(long)]
    file1: PathBuf,

    /// Second asset CSV file
    #[arg(long)]
    file2: PathBuf,

    /// Z-score lookback period
    #[arg(long, default_value = "20")]
    lookback: usize,

    /// Number of standard deviations for bands
    #[arg(long, default_value = "2.0")]
    num_std: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("{}", "=".repeat(60).blue());
    println!("{}", "Cointegration Analysis".bold().blue());
    println!("{}", "=".repeat(60).blue());

    // Загружаем данные
    let candles1 = load_candles(&args.file1)?;
    let candles2 = load_candles(&args.file2)?;

    let name1 = args
        .file1
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Asset1");
    let name2 = args
        .file2
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Asset2");

    println!(
        "\n{}: {} candles",
        name1.cyan(),
        candles1.len()
    );
    println!(
        "{}: {} candles\n",
        name2.cyan(),
        candles2.len()
    );

    let ts1 = TimeSeries::from_candles(name1, &candles1);
    let ts2 = TimeSeries::from_candles(name2, &candles2);

    // Выравниваем данные по длине
    let min_len = ts1.data.len().min(ts2.data.len());
    let prices1 = &ts1.data[..min_len];
    let prices2 = &ts2.data[..min_len];

    if min_len < 100 {
        anyhow::bail!("Not enough overlapping data (need at least 100 observations)");
    }

    println!(
        "{} {} observations",
        "Using:".green(),
        min_len
    );

    // Корреляционный анализ
    println!("\n{}", "Correlation Analysis".bold());
    println!("{}", "-".repeat(40));

    let price_corr = correlation(prices1, prices2);
    let returns1 = ts1.returns();
    let returns2 = ts2.returns();
    let return_corr = correlation(&returns1[..min_len - 1], &returns2[..min_len - 1]);

    println!("Price correlation: {:.4}", price_corr);
    println!("Return correlation: {:.4}", return_corr);

    let corr_interpretation = if price_corr.abs() > 0.9 {
        "Very high".green()
    } else if price_corr.abs() > 0.7 {
        "High".yellow()
    } else if price_corr.abs() > 0.5 {
        "Moderate".yellow()
    } else {
        "Low".red()
    };
    println!("Interpretation: {}", corr_interpretation);

    // Стационарность отдельных рядов
    println!("\n{}", "Stationarity of Individual Series".bold());
    println!("{}", "-".repeat(40));

    let adf1 = adf_test(prices1, None);
    let adf2 = adf_test(prices2, None);

    println!(
        "{}: ADF={:.3}, p={:.4} - {}",
        name1,
        adf1.statistic,
        adf1.p_value,
        if adf1.is_significant {
            "Stationary".green()
        } else {
            "Unit root".yellow()
        }
    );
    println!(
        "{}: ADF={:.3}, p={:.4} - {}",
        name2,
        adf2.statistic,
        adf2.p_value,
        if adf2.is_significant {
            "Stationary".green()
        } else {
            "Unit root".yellow()
        }
    );

    // Тест на коинтеграцию
    println!("\n{}", "Engle-Granger Cointegration Test".bold());
    println!("{}", "-".repeat(40));

    let coint_result = engle_granger_test(prices1, prices2)
        .ok_or_else(|| anyhow::anyhow!("Cointegration test failed"))?;

    println!("ADF statistic: {:.4}", coint_result.test_statistic);
    println!("P-value: {:.4}", coint_result.p_value);
    println!("Hedge ratio: {:.4}", coint_result.hedge_ratio);

    let coint_status = if coint_result.is_cointegrated {
        "COINTEGRATED".green().bold()
    } else {
        "NOT COINTEGRATED".red().bold()
    };
    println!("\nResult: {}", coint_status);

    if coint_result.is_cointegrated {
        println!(
            "\n{} The pair shows a long-term equilibrium relationship.",
            "✓".green()
        );
        println!(
            "  Formula: {} = {:.4} * {} + spread",
            name1, coint_result.hedge_ratio, name2
        );

        if let Some(hl) = coint_result.half_life {
            println!("  Half-life: {:.1} periods", hl);
            println!("  (Expected time to mean-revert halfway)");
        }
    } else {
        println!(
            "\n{} The pair does not show a stable equilibrium relationship.",
            "✗".red()
        );
        println!("  Consider other pairs or shorter time horizons.");
    }

    // Анализ спреда
    println!("\n{}", "Spread Analysis".bold());
    println!("{}", "-".repeat(40));

    let spread = compute_spread(prices1, prices2, coint_result.hedge_ratio);
    let spread_stats = DescriptiveStats::new(&spread);

    println!("Spread Statistics:");
    println!("  Mean: {:.4}", spread_stats.mean);
    println!("  Std: {:.4}", spread_stats.std);
    println!("  Min: {:.4}", spread_stats.min);
    println!("  Max: {:.4}", spread_stats.max);
    println!("  Skewness: {:.4}", spread_stats.skewness);
    println!("  Kurtosis: {:.4}", spread_stats.kurtosis);

    // Проверяем стационарность спреда
    let spread_adf = adf_test(&spread, None);
    println!(
        "\nSpread stationarity: ADF={:.3}, p={:.4} - {}",
        spread_adf.statistic,
        spread_adf.p_value,
        if spread_adf.is_significant {
            "Stationary".green()
        } else {
            "Non-stationary".red()
        }
    );

    // Z-score анализ
    println!("\n{}", "Z-Score Analysis".bold());
    println!("{}", "-".repeat(40));

    let zscore = spread_zscore(&spread, args.lookback);
    let current_zscore = *zscore.last().unwrap_or(&0.0);

    println!("Lookback period: {}", args.lookback);
    println!("Current Z-score: {:.4}", current_zscore);

    let signal = if current_zscore > 2.0 {
        format!("SELL SPREAD ({}={:.2}, {}={:.2})",
            name1, prices1.last().unwrap_or(&0.0),
            name2, prices2.last().unwrap_or(&0.0)).red()
    } else if current_zscore < -2.0 {
        format!("BUY SPREAD ({}={:.2}, {}={:.2})",
            name1, prices1.last().unwrap_or(&0.0),
            name2, prices2.last().unwrap_or(&0.0)).green()
    } else {
        "NEUTRAL".yellow().to_string()
    };

    println!("Signal: {}", signal);

    // Bollinger Bands для спреда
    let bands = compute_spread_bands(prices1, prices2, coint_result.hedge_ratio, args.lookback, args.num_std);

    let current_spread = *spread.last().unwrap_or(&0.0);
    let upper = bands.upper.last().unwrap_or(&f64::NAN);
    let middle = bands.middle.last().unwrap_or(&f64::NAN);
    let lower = bands.lower.last().unwrap_or(&f64::NAN);

    println!("\nBollinger Bands ({:.1}σ):", args.num_std);
    println!("  Upper: {:.4}", upper);
    println!("  Middle: {:.4}", middle);
    println!("  Lower: {:.4}", lower);
    println!("  Current: {:.4}", current_spread);

    // Визуализация последних Z-score
    println!("\n{}", "Recent Z-Score Pattern".bold());
    println!("{}", "-".repeat(40));

    let recent_start = zscore.len().saturating_sub(20);
    for (i, &z) in zscore[recent_start..].iter().enumerate() {
        let bar_len = (z.abs() * 10.0).min(30.0) as usize;
        let bar = if z > 0.0 {
            format!("{:>30}|{}", "", "#".repeat(bar_len)).red()
        } else {
            format!(
                "{}{}|",
                " ".repeat(30 - bar_len),
                "#".repeat(bar_len)
            ).green()
        };
        println!("{:>3}: {:>6.2} {}", recent_start + i, z, bar);
    }

    // Trading statistics
    if let Some(pair_analysis) = analyze_pair(name1, name2, prices1, prices2) {
        println!("\n{}", "Trading Analysis".bold());
        println!("{}", "-".repeat(40));
        println!("Zero crossings: {}", pair_analysis.num_crossings);
        println!(
            "Avg periods between crossings: {:.1}",
            pair_analysis.avg_time_between_crossings
        );
    }

    println!("\n{}", "Done!".green().bold());
    Ok(())
}
