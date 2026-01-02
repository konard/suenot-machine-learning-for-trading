//! Hierarchical graph pooling trading strategy

use crate::graph::MarketGraph;
use crate::pooling::{HierarchicalPooling, HierarchicalResult};
use ndarray::Array1;
use std::collections::{HashMap, VecDeque};

/// Market regime detected from cluster dynamics
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Regime {
    /// Normal market conditions - diverse cluster behavior
    Normal,

    /// Correlation breakdown - all assets moving together (panic)
    CorrelationBreakdown,

    /// Cluster formation - clusters becoming more distinct
    ClusterFormation,
}

impl std::fmt::Display for Regime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Regime::Normal => write!(f, "Normal"),
            Regime::CorrelationBreakdown => write!(f, "CorrelationBreakdown"),
            Regime::ClusterFormation => write!(f, "ClusterFormation"),
        }
    }
}

/// Trading signal for an asset
#[derive(Debug, Clone)]
pub struct Signal {
    /// Asset symbol
    pub symbol: String,

    /// Position size (-1 to 1, negative = short)
    pub position: f64,

    /// Confidence in the signal (0 to 1)
    pub confidence: f64,

    /// Cluster the asset belongs to
    pub cluster: usize,

    /// Relative momentum vs cluster
    pub relative_momentum: f64,

    /// Current regime
    pub regime: Regime,
}

/// Strategy configuration
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Number of clusters at level 1
    pub n_clusters_l1: usize,

    /// Number of super-clusters at level 2
    pub n_clusters_l2: usize,

    /// Weight for model predictions
    pub prediction_weight: f64,

    /// Weight for relative momentum
    pub momentum_weight: f64,

    /// Regime change sensitivity
    pub regime_sensitivity: f64,

    /// History length for regime detection
    pub history_length: usize,

    /// Maximum position size per asset
    pub max_position: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            n_clusters_l1: 10,
            n_clusters_l2: 3,
            prediction_weight: 0.5,
            momentum_weight: 0.5,
            regime_sensitivity: 0.2,
            history_length: 20,
            max_position: 0.1,
        }
    }
}

/// Graph pooling trading strategy
pub struct GraphPoolingStrategy {
    /// Configuration
    config: StrategyConfig,

    /// Hierarchical pooling model
    pooling: HierarchicalPooling,

    /// Entropy history for regime detection
    entropy_history: VecDeque<f64>,

    /// Current detected regime
    current_regime: Regime,

    /// Last cluster assignments
    last_assignments: Option<Vec<usize>>,
}

impl GraphPoolingStrategy {
    /// Create a new strategy with default configuration
    pub fn new() -> Self {
        Self::with_config(StrategyConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: StrategyConfig) -> Self {
        let pooling = HierarchicalPooling::new(config.n_clusters_l1, config.n_clusters_l2);

        Self {
            config,
            pooling,
            entropy_history: VecDeque::new(),
            current_regime: Regime::Normal,
            last_assignments: None,
        }
    }

    /// Generate trading signals from market graph
    pub fn generate_signals(&mut self, graph: &MarketGraph) -> anyhow::Result<Vec<Signal>> {
        // Run hierarchical pooling
        let pool_result = self.pooling.forward(graph)?;

        // Detect regime
        let regime = self.detect_regime(&pool_result);
        self.current_regime = regime;

        // Compute cluster-relative momentum
        let relative_momentum = self.compute_relative_momentum(graph, &pool_result);

        // Generate positions based on regime
        let positions = self.compute_positions(&relative_momentum, &pool_result, regime);

        // Build signals
        let cluster_assignments = pool_result.level1.hard_assignments();
        let signals: Vec<Signal> = graph
            .symbols
            .iter()
            .enumerate()
            .map(|(i, symbol)| {
                let position = positions[i].clamp(-self.config.max_position, self.config.max_position);
                let confidence = self.compute_confidence(&pool_result, i);

                Signal {
                    symbol: symbol.clone(),
                    position,
                    confidence,
                    cluster: cluster_assignments[i],
                    relative_momentum: relative_momentum[i],
                    regime,
                }
            })
            .collect();

        // Update state
        self.last_assignments = Some(cluster_assignments);

        Ok(signals)
    }

    /// Detect market regime from cluster dynamics
    fn detect_regime(&mut self, result: &HierarchicalResult) -> Regime {
        // Compute cluster entropy (measure of assignment uncertainty)
        let entropy = result.level1.entropy();

        // Update history
        self.entropy_history.push_back(entropy);
        if self.entropy_history.len() > self.config.history_length {
            self.entropy_history.pop_front();
        }

        // Need enough history
        if self.entropy_history.len() < self.config.history_length / 2 {
            return Regime::Normal;
        }

        // Compare recent to historical
        let history_vec: Vec<f64> = self.entropy_history.iter().copied().collect();
        let split = history_vec.len() / 2;
        let recent: f64 = history_vec[split..].iter().sum::<f64>() / (history_vec.len() - split) as f64;
        let historical: f64 = history_vec[..split].iter().sum::<f64>() / split as f64;

        let change_ratio = recent / historical.max(1e-10);

        if change_ratio > 1.0 + self.config.regime_sensitivity {
            // Entropy increasing -> clusters becoming less distinct -> correlation breakdown
            Regime::CorrelationBreakdown
        } else if change_ratio < 1.0 - self.config.regime_sensitivity {
            // Entropy decreasing -> clusters becoming more distinct
            Regime::ClusterFormation
        } else {
            Regime::Normal
        }
    }

    /// Compute relative momentum for each asset vs its cluster
    fn compute_relative_momentum(
        &self,
        graph: &MarketGraph,
        result: &HierarchicalResult,
    ) -> Vec<f64> {
        let n_assets = graph.n_assets();
        let cluster_assignments = result.level1.hard_assignments();
        let soft_assignments = &result.level1.cluster_assignments;

        // Get latest returns from graph features (first feature is return_1)
        let latest_returns: Vec<f64> = (0..n_assets)
            .map(|i| graph.features[[i, 0]])
            .collect();

        // Compute weighted cluster returns
        let n_clusters = soft_assignments.ncols();
        let mut cluster_returns = vec![0.0; n_clusters];
        let mut cluster_weights = vec![0.0; n_clusters];

        for i in 0..n_assets {
            for c in 0..n_clusters {
                let weight = soft_assignments[[i, c]];
                cluster_returns[c] += weight * latest_returns[i];
                cluster_weights[c] += weight;
            }
        }

        for c in 0..n_clusters {
            if cluster_weights[c] > 0.0 {
                cluster_returns[c] /= cluster_weights[c];
            }
        }

        // Compute relative momentum
        (0..n_assets)
            .map(|i| {
                let cluster = cluster_assignments[i];
                latest_returns[i] - cluster_returns[cluster]
            })
            .collect()
    }

    /// Compute positions based on regime and momentum
    fn compute_positions(
        &self,
        relative_momentum: &[f64],
        result: &HierarchicalResult,
        regime: Regime,
    ) -> Vec<f64> {
        let n_assets = relative_momentum.len();
        let cluster_assignments = result.level1.hard_assignments();
        let soft_assignments = &result.level1.cluster_assignments;

        let mut positions = vec![0.0; n_assets];

        match regime {
            Regime::CorrelationBreakdown => {
                // Risk-off: reduce positions, favor uncorrelated assets
                let cluster_sizes = result.level1.cluster_sizes();
                let smallest_cluster = cluster_sizes
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                for i in 0..n_assets {
                    if cluster_assignments[i] == smallest_cluster {
                        // Small position in the least correlated cluster
                        positions[i] = 0.3 * relative_momentum[i].signum() * 0.5;
                    }
                }
            }

            Regime::ClusterFormation => {
                // Momentum within clusters: go with cluster leaders
                let n_clusters = soft_assignments.ncols();

                for c in 0..n_clusters {
                    // Find the leader in each cluster (highest positive rel momentum)
                    let cluster_members: Vec<usize> = (0..n_assets)
                        .filter(|&i| cluster_assignments[i] == c)
                        .collect();

                    if let Some(&leader) = cluster_members
                        .iter()
                        .max_by(|&&a, &&b| {
                            relative_momentum[a]
                                .partial_cmp(&relative_momentum[b])
                                .unwrap()
                        })
                    {
                        if relative_momentum[leader] > 0.0 {
                            positions[leader] = self.config.max_position;
                        }
                    }
                }
            }

            Regime::Normal => {
                // Mean reversion on relative momentum + trend following
                for i in 0..n_assets {
                    // Contrarian on relative momentum
                    let contrarian = -relative_momentum[i] * self.config.momentum_weight;

                    // Trend following on absolute momentum (use features)
                    // Second feature is momentum_5
                    let trend = result.level1.pooled_features[[cluster_assignments[i], 1]]
                        * self.config.prediction_weight;

                    positions[i] = contrarian + trend;
                }
            }
        }

        // Normalize positions
        let total_abs: f64 = positions.iter().map(|p| p.abs()).sum();
        if total_abs > 0.0 {
            for p in &mut positions {
                *p /= total_abs;
            }
        }

        positions
    }

    /// Compute confidence for a signal
    fn compute_confidence(&self, result: &HierarchicalResult, asset_idx: usize) -> f64 {
        // Confidence based on cluster assignment certainty
        let soft_assignment = result.level1.cluster_assignments.row(asset_idx);
        let max_prob = soft_assignment.iter().cloned().fold(0.0, f64::max);

        // Higher max probability = higher confidence
        max_prob
    }

    /// Get current regime
    pub fn current_regime(&self) -> Regime {
        self.current_regime
    }

    /// Get entropy history
    pub fn entropy_history(&self) -> Vec<f64> {
        self.entropy_history.iter().copied().collect()
    }

    /// Summary of current state
    pub fn summary(&self) -> String {
        format!(
            "GraphPoolingStrategy:\n\
             - Regime: {}\n\
             - History length: {}\n\
             - Config: {:?}",
            self.current_regime,
            self.entropy_history.len(),
            self.config
        )
    }
}

impl Default for GraphPoolingStrategy {
    fn default() -> Self {
        Self::new()
    }
}

/// Backtest result
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Total return
    pub total_return: f64,

    /// Annualized return (assuming 252 trading days)
    pub annualized_return: f64,

    /// Volatility (annualized)
    pub volatility: f64,

    /// Sharpe ratio (assuming 0 risk-free rate)
    pub sharpe_ratio: f64,

    /// Maximum drawdown
    pub max_drawdown: f64,

    /// Number of trades
    pub n_trades: usize,

    /// Average turnover
    pub avg_turnover: f64,

    /// Regime counts
    pub regime_counts: HashMap<String, usize>,
}

impl BacktestResult {
    /// Create from returns series
    pub fn from_returns(returns: &[f64], turnovers: &[f64], regimes: &[Regime]) -> Self {
        let n = returns.len();
        if n == 0 {
            return Self::empty();
        }

        // Total return (compounded)
        let mut cumulative = 1.0;
        let mut peak = 1.0;
        let mut max_dd = 0.0;

        for &r in returns {
            cumulative *= 1.0 + r;
            if cumulative > peak {
                peak = cumulative;
            }
            let dd = (peak - cumulative) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        let total_return = cumulative - 1.0;
        let annualized_return = (cumulative.powf(252.0 / n as f64)) - 1.0;

        // Volatility
        let mean_return = returns.iter().sum::<f64>() / n as f64;
        let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / n as f64;
        let daily_vol = variance.sqrt();
        let volatility = daily_vol * (252.0_f64).sqrt();

        // Sharpe
        let sharpe = if volatility > 0.0 {
            annualized_return / volatility
        } else {
            0.0
        };

        // Turnover
        let avg_turnover = if turnovers.is_empty() {
            0.0
        } else {
            turnovers.iter().sum::<f64>() / turnovers.len() as f64
        };

        // Regime counts
        let mut regime_counts = HashMap::new();
        for regime in regimes {
            *regime_counts.entry(regime.to_string()).or_insert(0) += 1;
        }

        Self {
            total_return,
            annualized_return,
            volatility,
            sharpe_ratio: sharpe,
            max_drawdown: max_dd,
            n_trades: turnovers.iter().filter(|&&t| t > 0.0).count(),
            avg_turnover,
            regime_counts,
        }
    }

    fn empty() -> Self {
        Self {
            total_return: 0.0,
            annualized_return: 0.0,
            volatility: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            n_trades: 0,
            avg_turnover: 0.0,
            regime_counts: HashMap::new(),
        }
    }

    /// Display summary
    pub fn summary(&self) -> String {
        format!(
            "Backtest Results:\n\
             - Total Return: {:.2}%\n\
             - Annualized Return: {:.2}%\n\
             - Volatility: {:.2}%\n\
             - Sharpe Ratio: {:.2}\n\
             - Max Drawdown: {:.2}%\n\
             - Number of Trades: {}\n\
             - Average Turnover: {:.4}\n\
             - Regimes: {:?}",
            self.total_return * 100.0,
            self.annualized_return * 100.0,
            self.volatility * 100.0,
            self.sharpe_ratio,
            self.max_drawdown * 100.0,
            self.n_trades,
            self.avg_turnover,
            self.regime_counts
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Kline;

    fn create_test_graph() -> MarketGraph {
        let mut klines = std::collections::HashMap::new();
        for i in 0..10 {
            let symbol = format!("ASSET{}", i);
            let data: Vec<Kline> = (0..50)
                .map(|t| Kline {
                    symbol: symbol.clone(),
                    open_time: t * 1000,
                    open: 100.0 + (t as f64 + i as f64).sin(),
                    high: 101.0 + (t as f64 + i as f64).sin(),
                    low: 99.0 + (t as f64 + i as f64).sin(),
                    close: 100.5 + (t as f64 + i as f64).sin(),
                    volume: 1000.0,
                    turnover: 100000.0,
                })
                .collect();
            klines.insert(symbol, data);
        }
        MarketGraph::from_klines(&klines).unwrap()
    }

    #[test]
    fn test_strategy_signals() {
        let graph = create_test_graph();
        let mut strategy = GraphPoolingStrategy::new();
        let signals = strategy.generate_signals(&graph).unwrap();

        assert_eq!(signals.len(), 10);
        for signal in &signals {
            assert!(signal.position >= -1.0 && signal.position <= 1.0);
            assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
        }
    }

    #[test]
    fn test_regime_detection() {
        let graph = create_test_graph();
        let mut strategy = GraphPoolingStrategy::new();

        // Run multiple times to build history
        for _ in 0..25 {
            let _ = strategy.generate_signals(&graph);
        }

        // Should have detected some regime
        let regime = strategy.current_regime();
        assert!(regime == Regime::Normal ||
                regime == Regime::CorrelationBreakdown ||
                regime == Regime::ClusterFormation);
    }

    #[test]
    fn test_backtest_result() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015];
        let turnovers = vec![0.1, 0.05, 0.15, 0.08, 0.12];
        let regimes = vec![
            Regime::Normal,
            Regime::Normal,
            Regime::ClusterFormation,
            Regime::Normal,
            Regime::Normal,
        ];

        let result = BacktestResult::from_returns(&returns, &turnovers, &regimes);

        assert!(result.total_return > 0.0);
        assert!(result.volatility > 0.0);
        assert_eq!(result.n_trades, 5);
    }
}
