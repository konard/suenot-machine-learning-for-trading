//! Signal generation utilities.

use crate::graph::MarketGraph;
use ndarray::Array2;
use std::collections::HashMap;

/// Signal processor for post-processing MPNN outputs.
pub struct SignalProcessor {
    /// Smoothing factor for exponential moving average
    ema_alpha: f64,
    /// Previous signals for smoothing
    prev_signals: HashMap<String, f64>,
    /// Z-score lookback
    zscore_lookback: usize,
    /// Historical scores for z-score calculation
    historical_scores: HashMap<String, Vec<f64>>,
}

impl SignalProcessor {
    /// Create a new signal processor.
    pub fn new(ema_alpha: f64, zscore_lookback: usize) -> Self {
        Self {
            ema_alpha,
            prev_signals: HashMap::new(),
            zscore_lookback,
            historical_scores: HashMap::new(),
        }
    }

    /// Process raw MPNN output into smoothed signals.
    pub fn process(
        &mut self,
        graph: &MarketGraph,
        raw_output: &Array2<f64>,
    ) -> HashMap<String, f64> {
        let mut signals = HashMap::new();

        for (i, node) in graph.nodes.iter().enumerate() {
            let row = raw_output.row(i);
            let raw_score: f64 = row.iter().sum::<f64>() / row.len() as f64;

            // Apply EMA smoothing
            let smoothed = if let Some(&prev) = self.prev_signals.get(&node.symbol) {
                self.ema_alpha * raw_score + (1.0 - self.ema_alpha) * prev
            } else {
                raw_score
            };

            // Update history
            let history = self
                .historical_scores
                .entry(node.symbol.clone())
                .or_insert_with(Vec::new);
            history.push(smoothed);
            if history.len() > self.zscore_lookback {
                history.remove(0);
            }

            // Compute z-score normalized signal
            let final_signal = if history.len() >= 2 {
                let mean: f64 = history.iter().sum::<f64>() / history.len() as f64;
                let variance: f64 = history.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / history.len() as f64;
                let std = variance.sqrt().max(1e-6);
                ((smoothed - mean) / std).tanh()
            } else {
                smoothed.tanh()
            };

            self.prev_signals.insert(node.symbol.clone(), smoothed);
            signals.insert(node.symbol.clone(), final_signal);
        }

        signals
    }

    /// Reset processor state.
    pub fn reset(&mut self) {
        self.prev_signals.clear();
        self.historical_scores.clear();
    }
}

/// Combine multiple signals with weights.
pub struct SignalCombiner {
    /// Weights for different signal sources
    weights: HashMap<String, f64>,
}

impl SignalCombiner {
    /// Create a new signal combiner.
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
        }
    }

    /// Add a signal source with weight.
    pub fn add_source(mut self, name: impl Into<String>, weight: f64) -> Self {
        self.weights.insert(name.into(), weight);
        self
    }

    /// Combine signals from multiple sources.
    pub fn combine(
        &self,
        signals: &HashMap<String, HashMap<String, f64>>,
    ) -> HashMap<String, f64> {
        let mut combined = HashMap::new();
        let mut total_weight = 0.0;

        for (source_name, source_signals) in signals {
            let weight = self.weights.get(source_name).copied().unwrap_or(1.0);
            total_weight += weight;

            for (symbol, &signal) in source_signals {
                *combined.entry(symbol.clone()).or_insert(0.0) += signal * weight;
            }
        }

        // Normalize by total weight
        if total_weight > 0.0 {
            for signal in combined.values_mut() {
                *signal /= total_weight;
            }
        }

        combined
    }
}

impl Default for SignalCombiner {
    fn default() -> Self {
        Self::new()
    }
}

/// Signal decay for reducing stale signals.
pub struct SignalDecay {
    /// Decay factor per period
    decay_rate: f64,
    /// Current signals with ages
    signals: HashMap<String, (f64, usize)>,
}

impl SignalDecay {
    /// Create a new signal decay handler.
    pub fn new(decay_rate: f64) -> Self {
        Self {
            decay_rate,
            signals: HashMap::new(),
        }
    }

    /// Update with new signals.
    pub fn update(&mut self, new_signals: &HashMap<String, f64>) {
        // Age existing signals
        for (signal, age) in self.signals.values_mut() {
            *signal *= self.decay_rate;
            *age += 1;
        }

        // Add/update with new signals
        for (symbol, &signal) in new_signals {
            self.signals.insert(symbol.clone(), (signal, 0));
        }
    }

    /// Get current decayed signals.
    pub fn get_signals(&self) -> HashMap<String, f64> {
        self.signals
            .iter()
            .map(|(k, (v, _))| (k.clone(), *v))
            .collect()
    }

    /// Get signal age for a symbol.
    pub fn get_age(&self, symbol: &str) -> Option<usize> {
        self.signals.get(symbol).map(|(_, age)| *age)
    }

    /// Remove old signals.
    pub fn prune(&mut self, max_age: usize) {
        self.signals.retain(|_, (_, age)| *age <= max_age);
    }
}

/// Leading indicator detector.
pub struct LeadingIndicator {
    /// Lag correlations
    lag_correlations: HashMap<(String, String), Vec<f64>>,
    /// Max lag to check
    max_lag: usize,
}

impl LeadingIndicator {
    /// Create a new leading indicator detector.
    pub fn new(max_lag: usize) -> Self {
        Self {
            lag_correlations: HashMap::new(),
            max_lag,
        }
    }

    /// Update with new returns.
    pub fn update(&mut self, returns: &HashMap<String, Vec<f64>>) {
        let symbols: Vec<&String> = returns.keys().collect();

        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                let r_i = &returns[symbols[i]];
                let r_j = &returns[symbols[j]];

                if r_i.len() < self.max_lag + 10 || r_j.len() < self.max_lag + 10 {
                    continue;
                }

                let mut correlations = Vec::with_capacity(2 * self.max_lag + 1);

                for lag in -(self.max_lag as i32)..=(self.max_lag as i32) {
                    let corr = if lag >= 0 {
                        let lag = lag as usize;
                        pearson_correlation(
                            &r_i[..r_i.len() - lag],
                            &r_j[lag..],
                        )
                    } else {
                        let lag = (-lag) as usize;
                        pearson_correlation(
                            &r_i[lag..],
                            &r_j[..r_j.len() - lag],
                        )
                    };
                    correlations.push(corr);
                }

                self.lag_correlations.insert(
                    (symbols[i].clone(), symbols[j].clone()),
                    correlations,
                );
            }
        }
    }

    /// Get leading symbols for a target symbol.
    pub fn get_leaders(&self, target: &str, min_lead: usize) -> Vec<(String, i32, f64)> {
        let mut leaders = Vec::new();

        for ((sym_a, sym_b), correlations) in &self.lag_correlations {
            let mid = self.max_lag;

            // Check if sym_a leads sym_b (target)
            if sym_b == target {
                for lag in min_lead..=self.max_lag {
                    let idx = mid - lag;
                    if correlations[idx].abs() > 0.5 {
                        leaders.push((sym_a.clone(), lag as i32, correlations[idx]));
                    }
                }
            }

            // Check if sym_b leads sym_a (target)
            if sym_a == target {
                for lag in min_lead..=self.max_lag {
                    let idx = mid + lag;
                    if correlations[idx].abs() > 0.5 {
                        leaders.push((sym_b.clone(), lag as i32, correlations[idx]));
                    }
                }
            }
        }

        // Sort by absolute correlation
        leaders.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap_or(std::cmp::Ordering::Equal));
        leaders
    }
}

/// Compute Pearson correlation.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n < 2.0 {
        return 0.0;
    }

    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len().min(y.len()) {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x <= 0.0 || var_y <= 0.0 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Risk-adjusted signal generator.
pub struct RiskAdjustedSignals {
    /// Volatility scaling factor
    vol_scaling: f64,
    /// Maximum signal magnitude
    max_signal: f64,
}

impl RiskAdjustedSignals {
    /// Create a new risk-adjusted signal generator.
    pub fn new(vol_scaling: f64, max_signal: f64) -> Self {
        Self {
            vol_scaling,
            max_signal,
        }
    }

    /// Adjust signals by volatility.
    pub fn adjust(
        &self,
        signals: &HashMap<String, f64>,
        volatilities: &HashMap<String, f64>,
    ) -> HashMap<String, f64> {
        let mut adjusted = HashMap::new();

        // Compute average volatility
        let avg_vol: f64 = volatilities.values().sum::<f64>() / volatilities.len().max(1) as f64;

        for (symbol, &signal) in signals {
            let vol = volatilities.get(symbol).copied().unwrap_or(avg_vol);
            let vol_ratio = avg_vol / vol.max(1e-6);

            // Scale signal by inverse volatility
            let scaled = signal * vol_ratio.powf(self.vol_scaling);

            // Clip to max signal
            let clipped = scaled.max(-self.max_signal).min(self.max_signal);

            adjusted.insert(symbol.clone(), clipped);
        }

        adjusted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_signal_processor() {
        let mut processor = SignalProcessor::new(0.3, 10);
        let mut graph = crate::graph::MarketGraph::new();

        graph.add_node("BTCUSDT", array![0.1, 0.2, 0.3, 0.4]);
        graph.add_node("ETHUSDT", array![0.2, 0.3, 0.4, 0.5]);

        let output = array![[0.5, 0.3], [0.2, 0.4]];
        let signals = processor.process(&graph, &output);

        assert!(signals.contains_key("BTCUSDT"));
        assert!(signals.contains_key("ETHUSDT"));
    }

    #[test]
    fn test_signal_combiner() {
        let combiner = SignalCombiner::new()
            .add_source("mpnn", 0.7)
            .add_source("momentum", 0.3);

        let mut sources = HashMap::new();

        let mut mpnn_signals = HashMap::new();
        mpnn_signals.insert("BTCUSDT".to_string(), 0.5);
        sources.insert("mpnn".to_string(), mpnn_signals);

        let mut momentum_signals = HashMap::new();
        momentum_signals.insert("BTCUSDT".to_string(), 0.8);
        sources.insert("momentum".to_string(), momentum_signals);

        let combined = combiner.combine(&sources);

        // (0.5 * 0.7 + 0.8 * 0.3) / 1.0 = 0.59
        let expected = (0.5 * 0.7 + 0.8 * 0.3) / 1.0;
        assert!((combined["BTCUSDT"] - expected).abs() < 0.01);
    }

    #[test]
    fn test_signal_decay() {
        let mut decay = SignalDecay::new(0.9);

        let mut signals = HashMap::new();
        signals.insert("BTCUSDT".to_string(), 1.0);

        decay.update(&signals);
        assert_eq!(decay.get_signals()["BTCUSDT"], 1.0);

        // Empty update to age
        decay.update(&HashMap::new());
        assert!((decay.get_signals()["BTCUSDT"] - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_risk_adjusted() {
        let adjuster = RiskAdjustedSignals::new(0.5, 1.0);

        let mut signals = HashMap::new();
        signals.insert("BTCUSDT".to_string(), 0.5);
        signals.insert("ETHUSDT".to_string(), 0.5);

        let mut vols = HashMap::new();
        vols.insert("BTCUSDT".to_string(), 0.02);
        vols.insert("ETHUSDT".to_string(), 0.04);

        let adjusted = adjuster.adjust(&signals, &vols);

        // Lower vol asset should have higher adjusted signal
        assert!(adjusted["BTCUSDT"] > adjusted["ETHUSDT"]);
    }
}
