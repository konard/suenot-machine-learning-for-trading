//! Returns calculation and analysis

use super::MarketData;
use ndarray::{s, Array1, Array2, Axis};

/// Container for returns data
#[derive(Debug, Clone)]
pub struct Returns {
    /// Symbol names
    pub symbols: Vec<String>,
    /// Timestamps (one less than prices due to returns calculation)
    pub timestamps: Vec<i64>,
    /// Returns matrix (rows = timestamps, cols = symbols)
    pub returns: Array2<f64>,
}

impl Returns {
    /// Create new Returns from data
    pub fn new(symbols: Vec<String>, timestamps: Vec<i64>, returns: Array2<f64>) -> Self {
        Self {
            symbols,
            timestamps,
            returns,
        }
    }

    /// Calculate simple returns from market data
    pub fn from_market_data(data: &MarketData) -> Self {
        let prices = &data.close_prices;
        let n_periods = data.n_periods();
        let n_symbols = data.n_symbols();

        if n_periods < 2 {
            return Self {
                symbols: data.symbols.clone(),
                timestamps: vec![],
                returns: Array2::zeros((0, n_symbols)),
            };
        }

        // Calculate returns: (P_t - P_{t-1}) / P_{t-1}
        let prices_current = prices.slice(s![1.., ..]);
        let prices_prev = prices.slice(s![..n_periods - 1, ..]);

        let returns = (&prices_current - &prices_prev) / &prices_prev;

        Self {
            symbols: data.symbols.clone(),
            timestamps: data.timestamps[1..].to_vec(),
            returns,
        }
    }

    /// Calculate log returns from market data
    pub fn log_returns_from_market_data(data: &MarketData) -> Self {
        let prices = &data.close_prices;
        let n_periods = data.n_periods();
        let n_symbols = data.n_symbols();

        if n_periods < 2 {
            return Self {
                symbols: data.symbols.clone(),
                timestamps: vec![],
                returns: Array2::zeros((0, n_symbols)),
            };
        }

        // Calculate log returns: ln(P_t / P_{t-1})
        let prices_current = prices.slice(s![1.., ..]);
        let prices_prev = prices.slice(s![..n_periods - 1, ..]);

        let returns = (&prices_current / &prices_prev).mapv(f64::ln);

        Self {
            symbols: data.symbols.clone(),
            timestamps: data.timestamps[1..].to_vec(),
            returns,
        }
    }

    /// Number of symbols
    pub fn n_symbols(&self) -> usize {
        self.symbols.len()
    }

    /// Number of return periods
    pub fn n_periods(&self) -> usize {
        self.timestamps.len()
    }

    /// Get returns for a specific symbol
    pub fn get_symbol_returns(&self, symbol: &str) -> Option<Array1<f64>> {
        let idx = self.symbols.iter().position(|s| s == symbol)?;
        Some(self.returns.column(idx).to_owned())
    }

    /// Calculate mean returns for each symbol
    pub fn mean_returns(&self) -> Array1<f64> {
        self.returns.mean_axis(Axis(0)).unwrap()
    }

    /// Calculate standard deviation of returns for each symbol
    pub fn std_returns(&self) -> Array1<f64> {
        self.returns.std_axis(Axis(0), 0.0)
    }

    /// Calculate covariance matrix
    pub fn covariance_matrix(&self) -> Array2<f64> {
        let n = self.n_periods() as f64;
        let mean = self.mean_returns();

        // Center the data
        let centered = &self.returns - &mean;

        // Covariance = (X^T * X) / (n - 1)
        let cov = centered.t().dot(&centered) / (n - 1.0);
        cov
    }

    /// Calculate correlation matrix
    pub fn correlation_matrix(&self) -> Array2<f64> {
        let cov = self.covariance_matrix();
        let std = self.std_returns();
        let n = self.n_symbols();

        let mut corr = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if std[i] > 1e-10 && std[j] > 1e-10 {
                    corr[[i, j]] = cov[[i, j]] / (std[i] * std[j]);
                } else {
                    corr[[i, j]] = if i == j { 1.0 } else { 0.0 };
                }
            }
        }
        corr
    }

    /// Winsorize returns at given quantiles
    pub fn winsorize(&mut self, lower_quantile: f64, upper_quantile: f64) {
        for j in 0..self.n_symbols() {
            let mut col: Vec<f64> = self.returns.column(j).to_vec();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let lower_idx = ((col.len() as f64) * lower_quantile) as usize;
            let upper_idx = ((col.len() as f64) * upper_quantile) as usize;

            let lower_bound = col[lower_idx.min(col.len() - 1)];
            let upper_bound = col[upper_idx.min(col.len() - 1)];

            for i in 0..self.n_periods() {
                let val = self.returns[[i, j]];
                if val < lower_bound {
                    self.returns[[i, j]] = lower_bound;
                } else if val > upper_bound {
                    self.returns[[i, j]] = upper_bound;
                }
            }
        }
    }

    /// Standardize returns (zero mean, unit variance)
    pub fn standardize(&self) -> Returns {
        let mean = self.mean_returns();
        let std = self.std_returns();

        let mut standardized = self.returns.clone();
        for j in 0..self.n_symbols() {
            if std[j] > 1e-10 {
                for i in 0..self.n_periods() {
                    standardized[[i, j]] = (standardized[[i, j]] - mean[j]) / std[j];
                }
            }
        }

        Returns {
            symbols: self.symbols.clone(),
            timestamps: self.timestamps.clone(),
            returns: standardized,
        }
    }

    /// Center returns (zero mean)
    pub fn center(&self) -> Returns {
        let mean = self.mean_returns();
        let centered = &self.returns - &mean;

        Returns {
            symbols: self.symbols.clone(),
            timestamps: self.timestamps.clone(),
            returns: centered,
        }
    }

    /// Remove symbols with too many missing or zero values
    pub fn filter_valid_symbols(&self, min_valid_ratio: f64) -> Returns {
        let mut valid_indices = Vec::new();
        let mut valid_symbols = Vec::new();

        for (j, symbol) in self.symbols.iter().enumerate() {
            let col = self.returns.column(j);
            let valid_count = col.iter().filter(|&&v| v.is_finite() && v != 0.0).count();
            let valid_ratio = valid_count as f64 / self.n_periods() as f64;

            if valid_ratio >= min_valid_ratio {
                valid_indices.push(j);
                valid_symbols.push(symbol.clone());
            }
        }

        let mut filtered_returns = Array2::zeros((self.n_periods(), valid_indices.len()));
        for (new_j, &old_j) in valid_indices.iter().enumerate() {
            for i in 0..self.n_periods() {
                filtered_returns[[i, new_j]] = self.returns[[i, old_j]];
            }
        }

        Returns {
            symbols: valid_symbols,
            timestamps: self.timestamps.clone(),
            returns: filtered_returns,
        }
    }

    /// Calculate cumulative returns
    pub fn cumulative_returns(&self) -> Array2<f64> {
        let mut cumulative = Array2::ones((self.n_periods(), self.n_symbols()));

        for j in 0..self.n_symbols() {
            let mut cum = 1.0;
            for i in 0..self.n_periods() {
                cum *= 1.0 + self.returns[[i, j]];
                cumulative[[i, j]] = cum - 1.0; // Return as percentage
            }
        }

        cumulative
    }

    /// Calculate annualized Sharpe ratio (assuming 365 periods per year for crypto)
    pub fn sharpe_ratio(&self, risk_free_rate: f64, periods_per_year: f64) -> Array1<f64> {
        let mean = self.mean_returns();
        let std = self.std_returns();

        let annual_return = &mean * periods_per_year;
        let annual_std = &std * periods_per_year.sqrt();

        (&annual_return - risk_free_rate) / &annual_std
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_returns_calculation() {
        let symbols = vec!["BTC".to_string(), "ETH".to_string()];
        let timestamps = vec![1000, 2000, 3000];
        let prices = array![[100.0, 10.0], [110.0, 11.0], [105.0, 12.0]];
        let volumes = Array2::ones((3, 2));

        let data = MarketData::new(symbols, timestamps, prices, volumes);
        let returns = Returns::from_market_data(&data);

        assert_eq!(returns.n_periods(), 2);
        assert_eq!(returns.n_symbols(), 2);

        // BTC: (110-100)/100 = 0.1, (105-110)/110 = -0.0455
        let btc_returns = returns.get_symbol_returns("BTC").unwrap();
        assert!((btc_returns[0] - 0.1).abs() < 1e-10);
        assert!((btc_returns[1] - (-5.0 / 110.0)).abs() < 1e-10);
    }

    #[test]
    fn test_covariance_matrix() {
        let returns = Returns::new(
            vec!["A".to_string(), "B".to_string()],
            vec![1, 2, 3],
            array![[0.1, 0.2], [0.2, 0.1], [0.15, 0.15]],
        );

        let cov = returns.covariance_matrix();
        assert_eq!(cov.shape(), &[2, 2]);

        // Covariance matrix should be symmetric
        assert!((cov[[0, 1]] - cov[[1, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_standardize() {
        let returns = Returns::new(
            vec!["A".to_string()],
            vec![1, 2, 3, 4, 5],
            array![[0.1], [0.2], [0.15], [0.05], [0.1]],
        );

        let standardized = returns.standardize();
        let mean = standardized.mean_returns();
        let std = standardized.std_returns();

        // Mean should be ~0, std should be ~1
        assert!(mean[0].abs() < 1e-10);
        assert!((std[0] - 1.0).abs() < 0.1); // Sample std might not be exactly 1
    }
}
