//! Eigenportfolio implementation

use crate::data::Returns;
use crate::pca::PCAAnalysis;
use ndarray::{Array1, Array2, Axis};

/// Eigenportfolio constructed from PCA components
#[derive(Debug, Clone)]
pub struct Eigenportfolio {
    /// Portfolio name/identifier
    pub name: String,
    /// Asset symbols
    pub symbols: Vec<String>,
    /// Portfolio weights (normalized)
    pub weights: Array1<f64>,
    /// Original PCA component index
    pub component_idx: usize,
    /// Explained variance ratio of this component
    pub explained_variance_ratio: f64,
}

impl Eigenportfolio {
    /// Create an eigenportfolio from a PCA component
    pub fn from_pca_component(pca: &PCAAnalysis, component_idx: usize) -> Option<Self> {
        if component_idx >= pca.n_components {
            return None;
        }

        let raw_weights = pca.components.column(component_idx).to_owned();

        // Normalize weights to sum to 1
        let weight_sum = raw_weights.sum();
        let weights = if weight_sum.abs() > 1e-10 {
            &raw_weights / weight_sum
        } else {
            raw_weights.clone()
        };

        Some(Self {
            name: format!("Eigenportfolio {}", component_idx + 1),
            symbols: pca.feature_names.clone(),
            weights,
            component_idx,
            explained_variance_ratio: pca.explained_variance_ratio[component_idx],
        })
    }

    /// Create multiple eigenportfolios from PCA
    pub fn from_pca(pca: &PCAAnalysis, n_portfolios: usize) -> Vec<Self> {
        (0..n_portfolios.min(pca.n_components))
            .filter_map(|i| Self::from_pca_component(pca, i))
            .collect()
    }

    /// Get top N assets by weight (absolute value)
    pub fn top_holdings(&self, n: usize) -> Vec<(&String, f64)> {
        let mut holdings: Vec<(&String, f64)> = self
            .symbols
            .iter()
            .zip(self.weights.iter())
            .map(|(s, &w)| (s, w))
            .collect();

        holdings.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        holdings.into_iter().take(n).collect()
    }

    /// Calculate portfolio returns
    pub fn calculate_returns(&self, returns: &Returns) -> Array1<f64> {
        let n_periods = returns.n_periods();
        let mut portfolio_returns = Array1::zeros(n_periods);

        for (i, symbol) in self.symbols.iter().enumerate() {
            if let Some(asset_returns) = returns.get_symbol_returns(symbol) {
                for t in 0..n_periods {
                    portfolio_returns[t] += self.weights[i] * asset_returns[t];
                }
            }
        }

        portfolio_returns
    }

    /// Calculate cumulative returns
    pub fn cumulative_returns(&self, returns: &Returns) -> Array1<f64> {
        let portfolio_returns = self.calculate_returns(returns);
        let n = portfolio_returns.len();
        let mut cumulative = Array1::zeros(n);

        let mut cum = 1.0;
        for i in 0..n {
            cum *= 1.0 + portfolio_returns[i];
            cumulative[i] = cum - 1.0;
        }

        cumulative
    }

    /// Calculate portfolio volatility (annualized)
    pub fn volatility(&self, returns: &Returns, periods_per_year: f64) -> f64 {
        let portfolio_returns = self.calculate_returns(returns);
        let std = portfolio_returns.std(0.0);
        std * periods_per_year.sqrt()
    }

    /// Calculate Sharpe ratio
    pub fn sharpe_ratio(&self, returns: &Returns, risk_free_rate: f64, periods_per_year: f64) -> f64 {
        let portfolio_returns = self.calculate_returns(returns);
        let mean_return = portfolio_returns.mean().unwrap_or(0.0);
        let std = portfolio_returns.std(0.0);

        let annual_return = mean_return * periods_per_year;
        let annual_std = std * periods_per_year.sqrt();

        if annual_std > 1e-10 {
            (annual_return - risk_free_rate) / annual_std
        } else {
            0.0
        }
    }

    /// Print portfolio summary
    pub fn summary(&self) {
        println!("\n=== {} ===", self.name);
        println!("Component index: {}", self.component_idx);
        println!(
            "Explained variance: {:.2}%",
            self.explained_variance_ratio * 100.0
        );
        println!("\nTop holdings:");
        for (symbol, weight) in self.top_holdings(10) {
            let direction = if *weight > 0.0 { "LONG" } else { "SHORT" };
            println!("  {:>10}: {:>8.2}% ({})", symbol, weight * 100.0, direction);
        }
    }
}

/// Manager for multiple eigenportfolios
#[derive(Debug, Clone)]
pub struct EigenportfolioSet {
    /// Individual portfolios
    pub portfolios: Vec<Eigenportfolio>,
    /// Source PCA analysis
    pub pca: PCAAnalysis,
}

impl EigenportfolioSet {
    /// Create a set of eigenportfolios from returns data
    pub fn from_returns(returns: &Returns, n_portfolios: usize) -> Self {
        let pca = PCAAnalysis::fit(returns, Some(n_portfolios.max(returns.n_symbols())));
        let portfolios = Eigenportfolio::from_pca(&pca, n_portfolios);

        Self { portfolios, pca }
    }

    /// Get equal-weight market portfolio for comparison
    pub fn market_portfolio(&self) -> Eigenportfolio {
        let n = self.pca.feature_names.len();
        let weights = Array1::from_elem(n, 1.0 / n as f64);

        Eigenportfolio {
            name: "Market (Equal Weight)".to_string(),
            symbols: self.pca.feature_names.clone(),
            weights,
            component_idx: usize::MAX,
            explained_variance_ratio: 1.0,
        }
    }

    /// Compare portfolio performances
    pub fn compare_performance(&self, returns: &Returns, periods_per_year: f64) {
        println!("\n=== Portfolio Performance Comparison ===");
        println!(
            "{:<25} {:>12} {:>12} {:>12}",
            "Portfolio", "Return (%)", "Volatility (%)", "Sharpe"
        );
        println!("{:-<65}", "");

        // Market portfolio
        let market = self.market_portfolio();
        let market_returns = market.calculate_returns(returns);
        let market_vol = market.volatility(returns, periods_per_year);
        let market_sharpe = market.sharpe_ratio(returns, 0.0, periods_per_year);
        let market_cum = market_returns.sum() * 100.0;

        println!(
            "{:<25} {:>11.2}% {:>11.2}% {:>12.2}",
            market.name, market_cum, market_vol * 100.0, market_sharpe
        );

        // Eigenportfolios
        for portfolio in &self.portfolios {
            let port_returns = portfolio.calculate_returns(returns);
            let vol = portfolio.volatility(returns, periods_per_year);
            let sharpe = portfolio.sharpe_ratio(returns, 0.0, periods_per_year);
            let cum = port_returns.sum() * 100.0;

            println!(
                "{:<25} {:>11.2}% {:>11.2}% {:>12.2}",
                portfolio.name, cum, vol * 100.0, sharpe
            );
        }
    }

    /// Get all portfolio weights as a matrix
    pub fn weights_matrix(&self) -> Array2<f64> {
        let n_assets = self.pca.feature_names.len();
        let n_portfolios = self.portfolios.len();
        let mut weights = Array2::zeros((n_portfolios, n_assets));

        for (i, portfolio) in self.portfolios.iter().enumerate() {
            for j in 0..n_assets {
                weights[[i, j]] = portfolio.weights[j];
            }
        }

        weights
    }

    /// Print summary of all portfolios
    pub fn summary(&self) {
        println!("\n=== Eigenportfolio Set Summary ===");
        println!("Number of portfolios: {}", self.portfolios.len());
        println!("Number of assets: {}", self.pca.feature_names.len());

        println!("\nCumulative Explained Variance:");
        let mut cum_var = 0.0;
        for (i, portfolio) in self.portfolios.iter().enumerate() {
            cum_var += portfolio.explained_variance_ratio;
            println!(
                "  PC{}: {:.2}% (cumulative: {:.2}%)",
                i + 1,
                portfolio.explained_variance_ratio * 100.0,
                cum_var * 100.0
            );
        }

        for portfolio in &self.portfolios {
            portfolio.summary();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::MarketData;
    use ndarray::array;

    fn create_test_returns() -> Returns {
        let symbols = vec!["BTC".to_string(), "ETH".to_string(), "XRP".to_string()];
        let timestamps = vec![1, 2, 3, 4, 5];
        let returns = array![
            [0.01, 0.02, 0.015],
            [-0.01, -0.015, -0.01],
            [0.02, 0.025, 0.02],
            [0.005, 0.01, 0.008],
            [-0.005, -0.008, -0.006]
        ];

        Returns::new(symbols, timestamps, returns)
    }

    #[test]
    fn test_eigenportfolio_creation() {
        let returns = create_test_returns();
        let pca = PCAAnalysis::fit(&returns, Some(2));
        let portfolio = Eigenportfolio::from_pca_component(&pca, 0).unwrap();

        assert_eq!(portfolio.symbols.len(), 3);
        assert!((portfolio.weights.sum() - 1.0).abs() < 1e-10 || portfolio.weights.sum().abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_returns() {
        let returns = create_test_returns();
        let pca = PCAAnalysis::fit(&returns, Some(2));
        let portfolio = Eigenportfolio::from_pca_component(&pca, 0).unwrap();

        let port_returns = portfolio.calculate_returns(&returns);
        assert_eq!(port_returns.len(), 5);
    }

    #[test]
    fn test_eigenportfolio_set() {
        let returns = create_test_returns();
        let set = EigenportfolioSet::from_returns(&returns, 2);

        assert_eq!(set.portfolios.len(), 2);
    }
}
