//! Directed information flow network based on Transfer Entropy.

use crate::entropy::estimator::{TransferEntropyEstimator, TEMethod};

/// A directed information flow network computed from pairwise Transfer Entropy.
pub struct TENetwork {
    /// Asset/symbol names.
    pub symbols: Vec<String>,
    /// TE estimator instance.
    estimator: TransferEntropyEstimator,
    /// TE matrix: te_matrix[i][j] = TE(symbol_i → symbol_j).
    pub te_matrix: Vec<Vec<f64>>,
}

impl TENetwork {
    /// Create a new TE network.
    ///
    /// # Arguments
    /// * `symbols` - Asset names.
    /// * `history_length` - TE history parameter.
    pub fn new(symbols: Vec<String>, history_length: usize) -> Self {
        let n = symbols.len();
        Self {
            symbols,
            estimator: TransferEntropyEstimator::new(
                TEMethod::Binning { n_bins: 8 },
                history_length,
            ),
            te_matrix: vec![vec![0.0; n]; n],
        }
    }

    /// Create a network with a custom estimator.
    pub fn with_estimator(symbols: Vec<String>, estimator: TransferEntropyEstimator) -> Self {
        let n = symbols.len();
        Self {
            symbols,
            estimator,
            te_matrix: vec![vec![0.0; n]; n],
        }
    }

    /// Compute pairwise TE for all asset pairs.
    ///
    /// # Arguments
    /// * `returns` - Slice of return series, one per asset. Each inner slice
    ///   must have the same length.
    pub fn compute_all_pairs(&mut self, returns: &[&[f64]]) {
        let n = self.symbols.len();
        assert_eq!(returns.len(), n, "Must provide returns for each symbol");

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    self.te_matrix[i][j] = self.estimator.compute_te(returns[i], returns[j]);
                }
            }
        }
    }

    /// Get net TE (outgoing - incoming) for each asset.
    pub fn net_te(&self) -> Vec<f64> {
        let n = self.symbols.len();
        let mut net = vec![0.0; n];
        for i in 0..n {
            let outgoing: f64 = self.te_matrix[i].iter().sum();
            let incoming: f64 = self.te_matrix.iter().map(|row| row[i]).sum();
            net[i] = outgoing - incoming;
        }
        net
    }

    /// Identify leader and follower assets based on net TE.
    ///
    /// Returns (leaders, followers) as vectors of symbol names.
    pub fn identify_leaders_followers(&self) -> (Vec<String>, Vec<String>) {
        let net = self.net_te();
        let mut indexed: Vec<(usize, f64)> = net.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mid = self.symbols.len() / 2;
        let leaders: Vec<String> = indexed[..mid]
            .iter()
            .map(|(i, _)| self.symbols[*i].clone())
            .collect();
        let followers: Vec<String> = indexed[mid..]
            .iter()
            .map(|(i, _)| self.symbols[*i].clone())
            .collect();

        (leaders, followers)
    }

    /// Get the strongest leader (highest net outgoing TE).
    pub fn strongest_leader(&self) -> Option<(String, f64)> {
        let net = self.net_te();
        net.iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, v)| (self.symbols[i].clone(), v))
    }

    /// Get the strongest follower (lowest net TE, i.e., highest incoming).
    pub fn strongest_follower(&self) -> Option<(String, f64)> {
        let net = self.net_te();
        net.iter()
            .copied()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, v)| (self.symbols[i].clone(), v))
    }

    /// Print the TE matrix to stdout.
    pub fn print_matrix(&self) {
        let n = self.symbols.len();
        // Header
        print!("{:>10}", "");
        for sym in &self.symbols {
            print!("{:>10}", sym);
        }
        println!();

        for i in 0..n {
            print!("{:>10}", self.symbols[i]);
            for j in 0..n {
                if i == j {
                    print!("{:>10}", "---");
                } else {
                    print!("{:>10.4}", self.te_matrix[i][j]);
                }
            }
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_network_leader_follower() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 300;

        // Leader series
        let leader: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();

        // Follower with lag-1 dependency on leader
        let mut follower = vec![0.0; n];
        for i in 1..n {
            follower[i] = 0.5 * leader[i - 1] + 0.5 * (rng.gen::<f64>() - 0.5);
        }

        let symbols = vec!["Leader".to_string(), "Follower".to_string()];
        let mut network = TENetwork::new(symbols, 2);
        network.compute_all_pairs(&[&leader, &follower]);

        // Leader should have positive net TE
        let net = network.net_te();
        assert!(
            net[0] > net[1],
            "Leader net TE ({}) should exceed follower net TE ({})",
            net[0],
            net[1]
        );

        let (leaders, followers) = network.identify_leaders_followers();
        assert_eq!(leaders[0], "Leader");
        assert_eq!(followers[0], "Follower");
    }
}
