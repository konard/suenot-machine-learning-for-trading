//! TE-based trading strategy implementation.

use crate::trading::signals::{TradingSignal, generate_signal};
use crate::entropy::estimator::{TransferEntropyEstimator, TEMethod};

/// Transfer Entropy based leader-follower trading strategy.
pub struct TEStrategy {
    /// Minimum return threshold to trigger a signal.
    pub return_threshold: f64,
    /// Minimum TE threshold to consider a pair valid.
    pub te_threshold: f64,
    /// Lookback window for rolling TE computation.
    pub te_lookback: usize,
    /// TE estimator.
    estimator: TransferEntropyEstimator,
}

impl TEStrategy {
    /// Create a new TE strategy.
    ///
    /// # Arguments
    /// * `return_threshold` - Minimum leader return to trigger signal.
    /// * `te_threshold` - Minimum TE for valid leader-follower pair.
    /// * `te_lookback` - Rolling window size for TE estimation.
    pub fn new(return_threshold: f64, te_threshold: f64, te_lookback: usize) -> Self {
        Self {
            return_threshold,
            te_threshold,
            te_lookback,
            estimator: TransferEntropyEstimator::new(
                TEMethod::Binning { n_bins: 8 },
                3,
            ),
        }
    }

    /// Generate signals for the entire time series.
    ///
    /// # Arguments
    /// * `leader_returns` - Return series of the leader asset.
    /// * `follower_returns` - Return series of the follower asset.
    ///
    /// # Returns
    /// Vector of trading signals, one per time step.
    pub fn generate_signals(
        &self,
        leader_returns: &[f64],
        follower_returns: &[f64],
    ) -> Vec<TradingSignal> {
        let n = leader_returns.len();
        let mut signals = vec![TradingSignal::Neutral; n];

        for t in self.te_lookback..n {
            // Compute rolling TE
            let leader_window = &leader_returns[t - self.te_lookback..t];
            let follower_window = &follower_returns[t - self.te_lookback..t];

            let te = self.estimator.compute_te(leader_window, follower_window);
            let leader_ret = leader_returns[t - 1]; // Previous period return

            signals[t] = generate_signal(
                leader_ret,
                te,
                self.return_threshold,
                self.te_threshold,
            );
        }

        signals
    }
}
