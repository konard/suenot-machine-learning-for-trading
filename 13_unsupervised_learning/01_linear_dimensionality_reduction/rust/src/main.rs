//! PCA Crypto - Main entry point
//!
//! This CLI tool demonstrates Principal Component Analysis
//! applied to cryptocurrency market data from Bybit.

use anyhow::Result;
use clap::{Parser, Subcommand};
use pca_crypto::{
    api::{BybitClient, Timeframe},
    data::{MarketData, Returns},
    pca::{PCAAnalysis, RiskFactorAnalysis},
    portfolio::{EigenportfolioSet, PortfolioMetrics},
    utils::{print_variance_plot, SummaryStats},
};

#[derive(Parser)]
#[command(name = "pca-crypto")]
#[command(about = "PCA analysis for cryptocurrency trading", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch market data from Bybit
    Fetch {
        /// Number of top symbols by turnover
        #[arg(short, long, default_value = "10")]
        n_symbols: usize,

        /// Number of candles to fetch
        #[arg(short, long, default_value = "100")]
        limit: u32,

        /// Output file path
        #[arg(short, long, default_value = "data/prices.csv")]
        output: String,
    },

    /// Run PCA analysis on market data
    Analyze {
        /// Input data file (CSV)
        #[arg(short, long, default_value = "data/prices.csv")]
        input: String,

        /// Number of principal components
        #[arg(short, long)]
        n_components: Option<usize>,

        /// Target explained variance ratio
        #[arg(short, long)]
        variance_target: Option<f64>,
    },

    /// Create and analyze eigenportfolios
    Portfolio {
        /// Input data file (CSV)
        #[arg(short, long, default_value = "data/prices.csv")]
        input: String,

        /// Number of portfolios to create
        #[arg(short, long, default_value = "4")]
        n_portfolios: usize,
    },

    /// Analyze risk factors
    RiskFactors {
        /// Input data file (CSV)
        #[arg(short, long, default_value = "data/prices.csv")]
        input: String,

        /// Number of factors
        #[arg(short, long, default_value = "5")]
        n_factors: usize,
    },

    /// Demonstrate curse of dimensionality
    CurseDimensionality {
        /// Number of points
        #[arg(short, long, default_value = "100")]
        n_points: usize,

        /// Maximum dimensions
        #[arg(short, long, default_value = "100")]
        max_dims: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Fetch {
            n_symbols,
            limit,
            output,
        } => {
            fetch_data(n_symbols, limit, &output)?;
        }
        Commands::Analyze {
            input,
            n_components,
            variance_target,
        } => {
            analyze_pca(&input, n_components, variance_target)?;
        }
        Commands::Portfolio { input, n_portfolios } => {
            create_portfolios(&input, n_portfolios)?;
        }
        Commands::RiskFactors { input, n_factors } => {
            analyze_risk_factors(&input, n_factors)?;
        }
        Commands::CurseDimensionality { n_points, max_dims } => {
            demonstrate_curse(n_points, max_dims)?;
        }
    }

    Ok(())
}

fn fetch_data(n_symbols: usize, limit: u32, output: &str) -> Result<()> {
    println!("Fetching market data from Bybit...");
    println!("  Symbols: {} (top by turnover)", n_symbols);
    println!("  Candles: {}", limit);

    let client = BybitClient::new();

    // Get top symbols
    let symbols = client.get_top_symbols_by_turnover("linear", n_symbols)?;
    println!("\nTop symbols: {:?}", symbols);

    // Fetch data
    let market_data = MarketData::fetch_from_bybit(&client, &symbols, "linear", Timeframe::Day1, limit)?;

    println!(
        "\nFetched {} periods for {} symbols",
        market_data.n_periods(),
        market_data.n_symbols()
    );

    // Ensure output directory exists
    if let Some(parent) = std::path::Path::new(output).parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Save to CSV
    market_data.to_csv(output)?;
    println!("Data saved to: {}", output);

    Ok(())
}

fn analyze_pca(input: &str, n_components: Option<usize>, variance_target: Option<f64>) -> Result<()> {
    println!("Loading data from: {}", input);

    let market_data = MarketData::from_csv(input)?;
    println!(
        "Loaded {} periods for {} symbols",
        market_data.n_periods(),
        market_data.n_symbols()
    );

    // Calculate returns
    let returns = Returns::from_market_data(&market_data);
    println!("Calculated {} return periods", returns.n_periods());

    // Fit PCA
    let pca = if let Some(target) = variance_target {
        println!("\nFitting PCA with variance target: {:.1}%", target * 100.0);
        PCAAnalysis::fit_with_variance_threshold(&returns, target)
    } else {
        println!("\nFitting PCA with {} components", n_components.unwrap_or(returns.n_symbols()));
        PCAAnalysis::fit(&returns, n_components)
    };

    // Print results
    pca.summary();
    print_variance_plot(&pca.explained_variance_ratio, 15);

    // Find elbow
    let elbow = pca.find_elbow();
    println!("\nSuggested number of components (elbow): {}", elbow);

    // Reconstruction error
    let error = pca.reconstruction_error(&returns.returns);
    println!("Reconstruction RMSE: {:.6}", error);

    Ok(())
}

fn create_portfolios(input: &str, n_portfolios: usize) -> Result<()> {
    println!("Loading data from: {}", input);

    let market_data = MarketData::from_csv(input)?;
    let returns = Returns::from_market_data(&market_data);

    println!(
        "Creating {} eigenportfolios from {} assets",
        n_portfolios,
        returns.n_symbols()
    );

    // Create eigenportfolio set
    let portfolio_set = EigenportfolioSet::from_returns(&returns, n_portfolios);
    portfolio_set.summary();

    // Compare performance (assuming daily data, 365 days/year for crypto)
    portfolio_set.compare_performance(&returns, 365.0);

    // Detailed metrics for first portfolio
    if let Some(first_portfolio) = portfolio_set.portfolios.first() {
        let port_returns = first_portfolio.calculate_returns(&returns);
        let metrics = PortfolioMetrics::from_returns(&port_returns, 365.0);

        println!("\n{} - Detailed Metrics:", first_portfolio.name);
        metrics.summary();
    }

    Ok(())
}

fn analyze_risk_factors(input: &str, n_factors: usize) -> Result<()> {
    println!("Loading data from: {}", input);

    let market_data = MarketData::from_csv(input)?;
    let returns = Returns::from_market_data(&market_data);

    println!(
        "Analyzing {} risk factors from {} assets",
        n_factors,
        returns.n_symbols()
    );

    // Create risk factor analysis
    let analysis = RiskFactorAnalysis::from_returns(&returns, Some(n_factors));
    analysis.summary();

    // Factor statistics
    println!("\nFactor Return Statistics:");
    for i in 0..n_factors.min(analysis.pca.n_components) {
        if let Some(factor_returns) = analysis.get_factor_returns(i) {
            println!("\n--- Factor {} ---", i + 1);
            let stats = SummaryStats::from_data(&factor_returns);
            stats.print();
        }
    }

    Ok(())
}

fn demonstrate_curse(n_points: usize, max_dims: usize) -> Result<()> {
    use ndarray::Array2;
    use pca_crypto::data::{mean_pairwise_distance, min_pairwise_distances};
    use rand::Rng;

    println!("Demonstrating the Curse of Dimensionality");
    println!("=========================================");
    println!("Points: {}", n_points);
    println!("Max dimensions: {}", max_dims);
    println!();

    let mut rng = rand::thread_rng();
    let dims = [1, 2, 5, 10, 25, 50, 100].iter().filter(|&&d| d <= max_dims);

    println!("{:>6} {:>15} {:>15}", "Dims", "Mean Distance", "Min Distance");
    println!("{:-<40}", "");

    for &d in dims {
        // Generate random points in [0, 1]^d
        let mut points = Array2::zeros((n_points, d));
        for i in 0..n_points {
            for j in 0..d {
                points[[i, j]] = rng.gen::<f64>();
            }
        }

        let mean_dist = mean_pairwise_distance(&points);
        let min_dists = min_pairwise_distances(&points);
        let avg_min_dist = min_dists.mean().unwrap_or(0.0);

        println!("{:>6} {:>15.4} {:>15.4}", d, mean_dist, avg_min_dist);
    }

    println!();
    println!("Observation: As dimensions increase, distances grow,");
    println!("making it harder to distinguish between 'near' and 'far' points.");

    Ok(())
}
