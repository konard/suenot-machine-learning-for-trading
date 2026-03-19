//! Attention Rollout Trading Library
//!
//! This library provides attention rollout computation for interpretable
//! trading models implemented in Rust for high performance.

use ndarray::{Array1, Array2, Array3, Axis};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during attention rollout computation
#[derive(Error, Debug)]
pub enum AttentionRolloutError {
    #[error("Empty attention list provided")]
    EmptyAttentionList,

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Invalid layer index: {0}")]
    InvalidLayerIndex(usize),

    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Method for fusing multiple attention heads
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeadFusion {
    Mean,
    Max,
    Min,
}

/// Configuration for attention rollout computation
#[derive(Debug, Clone)]
pub struct AttentionRolloutConfig {
    pub head_fusion: HeadFusion,
    pub discard_ratio: f64,
    pub add_residual: bool,
    pub residual_weight: f64,
}

impl Default for AttentionRolloutConfig {
    fn default() -> Self {
        Self {
            head_fusion: HeadFusion::Mean,
            discard_ratio: 0.0,
            add_residual: true,
            residual_weight: 0.5,
        }
    }
}

/// Attention Rollout computation engine
pub struct AttentionRollout {
    config: AttentionRolloutConfig,
}

impl AttentionRollout {
    /// Create new AttentionRollout with default configuration
    pub fn new() -> Self {
        Self {
            config: AttentionRolloutConfig::default(),
        }
    }

    /// Create AttentionRollout with custom configuration
    pub fn with_config(config: AttentionRolloutConfig) -> Self {
        Self { config }
    }

    /// Fuse multiple attention heads into single matrix
    fn fuse_heads(&self, attention: &Array3<f64>) -> Array2<f64> {
        match self.config.head_fusion {
            HeadFusion::Mean => attention.mean_axis(Axis(0)).unwrap(),
            HeadFusion::Max => {
                let shape = (attention.shape()[1], attention.shape()[2]);
                let mut result = Array2::zeros(shape);
                for i in 0..shape.0 {
                    for j in 0..shape.1 {
                        let mut max_val = f64::NEG_INFINITY;
                        for h in 0..attention.shape()[0] {
                            max_val = max_val.max(attention[[h, i, j]]);
                        }
                        result[[i, j]] = max_val;
                    }
                }
                result
            }
            HeadFusion::Min => {
                let shape = (attention.shape()[1], attention.shape()[2]);
                let mut result = Array2::zeros(shape);
                for i in 0..shape.0 {
                    for j in 0..shape.1 {
                        let mut min_val = f64::INFINITY;
                        for h in 0..attention.shape()[0] {
                            min_val = min_val.min(attention[[h, i, j]]);
                        }
                        result[[i, j]] = min_val;
                    }
                }
                result
            }
        }
    }

    /// Discard lowest attention weights
    fn discard_low_attention(&self, attention: &mut Array2<f64>) {
        if self.config.discard_ratio <= 0.0 {
            return;
        }

        let mut values: Vec<f64> = attention.iter().cloned().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let threshold_idx = (values.len() as f64 * self.config.discard_ratio) as usize;
        let threshold = values[threshold_idx.min(values.len() - 1)];

        attention.mapv_inplace(|x| if x > threshold { x } else { 0.0 });

        // Renormalize rows
        for mut row in attention.rows_mut() {
            let sum: f64 = row.sum();
            if sum > 0.0 {
                row.mapv_inplace(|x| x / sum);
            }
        }
    }

    /// Compute attention rollout from list of attention matrices
    ///
    /// # Arguments
    /// * `attentions` - List of attention matrices (n_heads, seq_len, seq_len)
    /// * `start_layer` - Layer to start rollout computation
    ///
    /// # Returns
    /// Rollout matrix of shape (seq_len, seq_len)
    pub fn compute_rollout(
        &self,
        attentions: &[Array3<f64>],
        start_layer: usize,
    ) -> Result<Array2<f64>, AttentionRolloutError> {
        if attentions.is_empty() {
            return Err(AttentionRolloutError::EmptyAttentionList);
        }

        if start_layer >= attentions.len() {
            return Err(AttentionRolloutError::InvalidLayerIndex(start_layer));
        }

        let seq_len = attentions[0].shape()[1];

        // Initialize rollout with identity matrix
        let mut rollout = Array2::eye(seq_len);

        for attention in attentions.iter().skip(start_layer) {
            // Fuse attention heads
            let mut fused = self.fuse_heads(attention);

            // Discard low attention weights
            self.discard_low_attention(&mut fused);

            // Add residual connection
            if self.config.add_residual {
                let identity: Array2<f64> = Array2::eye(seq_len);
                let w = self.config.residual_weight;
                fused = &fused * (1.0 - w) + &identity * w;
            }

            // Accumulate rollout: R = A @ R
            rollout = fused.dot(&rollout);
        }

        // Normalize rows
        for mut row in rollout.rows_mut() {
            let sum: f64 = row.sum();
            if sum > 0.0 {
                row.mapv_inplace(|x| x / sum);
            }
        }

        Ok(rollout)
    }

    /// Get attribution scores for input positions
    ///
    /// # Arguments
    /// * `attentions` - List of attention matrices
    /// * `output_position` - Position to get attribution for
    pub fn get_input_attribution(
        &self,
        attentions: &[Array3<f64>],
        output_position: usize,
    ) -> Result<Array1<f64>, AttentionRolloutError> {
        let rollout = self.compute_rollout(attentions, 0)?;
        Ok(rollout.row(output_position).to_owned())
    }
}

impl Default for AttentionRollout {
    fn default() -> Self {
        Self::new()
    }
}

/// Trading-specific attention analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingAttentionAnalysis {
    pub temporal_importance: Vec<f64>,
    pub regime: AttentionRegime,
    pub concentration_score: f64,
    pub recent_bias: f64,
}

/// Detected attention regime
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AttentionRegime {
    Momentum,
    MeanReversion,
    Mixed,
}

/// Trading-specific attention rollout analyzer
pub struct TradingAttentionAnalyzer {
    rollout: AttentionRollout,
    momentum_threshold: f64,
}

impl TradingAttentionAnalyzer {
    pub fn new() -> Self {
        Self {
            rollout: AttentionRollout::new(),
            momentum_threshold: 0.6,
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.momentum_threshold = threshold;
        self
    }

    /// Analyze attention patterns for trading insights
    pub fn analyze(
        &self,
        attentions: &[Array3<f64>],
    ) -> Result<TradingAttentionAnalysis, AttentionRolloutError> {
        let seq_len = attentions[0].shape()[1];
        let attribution = self.rollout.get_input_attribution(attentions, seq_len - 1)?;

        // Calculate recent bias
        let recent_window = seq_len / 4;
        let recent_bias: f64 = attribution
            .slice(ndarray::s![seq_len - recent_window..])
            .sum();

        // Determine regime
        let regime = if recent_bias > self.momentum_threshold {
            AttentionRegime::Momentum
        } else if recent_bias < 1.0 - self.momentum_threshold {
            AttentionRegime::MeanReversion
        } else {
            AttentionRegime::Mixed
        };

        // Calculate concentration (entropy)
        let concentration_score = self.calculate_entropy(&attribution);

        Ok(TradingAttentionAnalysis {
            temporal_importance: attribution.to_vec(),
            regime,
            concentration_score,
            recent_bias,
        })
    }

    fn calculate_entropy(&self, probs: &Array1<f64>) -> f64 {
        -probs
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }
}

impl Default for TradingAttentionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Bybit market data fetcher
pub mod data {
    use super::*;
    use reqwest::blocking::Client;

    #[derive(Debug, Deserialize)]
    struct BybitResponse {
        #[serde(rename = "retCode")]
        ret_code: i32,
        #[serde(rename = "retMsg")]
        ret_msg: String,
        result: BybitResult,
    }

    #[derive(Debug, Deserialize)]
    struct BybitResult {
        list: Vec<Vec<String>>,
    }

    /// OHLCV candle data
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Candle {
        pub timestamp: i64,
        pub open: f64,
        pub high: f64,
        pub low: f64,
        pub close: f64,
        pub volume: f64,
    }

    /// Fetch candle data from Bybit
    pub fn fetch_bybit_candles(
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>, anyhow::Error> {
        let client = Client::new();
        let url = format!(
            "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            symbol, interval, limit
        );

        let response: BybitResponse = client.get(&url).send()?.json()?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let candles: Vec<Candle> = response
            .result
            .list
            .into_iter()
            .map(|row| Candle {
                timestamp: row[0].parse().unwrap_or(0),
                open: row[1].parse().unwrap_or(0.0),
                high: row[2].parse().unwrap_or(0.0),
                low: row[3].parse().unwrap_or(0.0),
                close: row[4].parse().unwrap_or(0.0),
                volume: row[5].parse().unwrap_or(0.0),
            })
            .collect();

        Ok(candles)
    }
}

/// Backtesting utilities
pub mod backtest {
    /// Backtest result metrics
    #[derive(Debug, Clone)]
    pub struct BacktestResult {
        pub total_return: f64,
        pub sharpe_ratio: f64,
        pub sortino_ratio: f64,
        pub max_drawdown: f64,
        pub win_rate: f64,
        pub n_trades: usize,
    }

    /// Calculate Sharpe ratio
    pub fn calculate_sharpe(returns: &[f64], risk_free_rate: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let std = variance.sqrt();

        if std == 0.0 {
            return 0.0;
        }

        let daily_rf = risk_free_rate / 252.0;
        (252.0_f64).sqrt() * (mean - daily_rf) / std
    }

    /// Calculate Sortino ratio
    pub fn calculate_sortino(returns: &[f64], risk_free_rate: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;

        let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

        if downside.is_empty() {
            return if mean > 0.0 { f64::INFINITY } else { 0.0 };
        }

        let downside_var: f64 = downside.iter().map(|r| r.powi(2)).sum::<f64>()
            / downside.len() as f64;
        let downside_std = downside_var.sqrt();

        if downside_std == 0.0 {
            return 0.0;
        }

        let daily_rf = risk_free_rate / 252.0;
        (252.0_f64).sqrt() * (mean - daily_rf) / downside_std
    }

    /// Calculate maximum drawdown
    pub fn calculate_max_drawdown(equity_curve: &[f64]) -> f64 {
        if equity_curve.is_empty() {
            return 0.0;
        }

        let mut max_dd = 0.0;
        let mut peak = equity_curve[0];

        for &value in equity_curve.iter() {
            if value > peak {
                peak = value;
            }
            let dd = (peak - value) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        max_dd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_rollout_basic() {
        let rollout = AttentionRollout::new();

        // Create simple attention matrices
        let attention1 = Array3::from_shape_fn((2, 4, 4), |(_, i, j)| {
            if i == j { 0.5 } else { 0.5 / 3.0 }
        });
        let attention2 = attention1.clone();

        let attentions = vec![attention1, attention2];
        let result = rollout.compute_rollout(&attentions, 0).unwrap();

        assert_eq!(result.shape(), &[4, 4]);

        // Check row normalization
        for row in result.rows() {
            let sum: f64 = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_head_fusion() {
        let rollout = AttentionRollout::with_config(AttentionRolloutConfig {
            head_fusion: HeadFusion::Max,
            ..Default::default()
        });

        let attention = Array3::from_shape_fn((2, 3, 3), |(h, _, _)| {
            if h == 0 { 0.3 } else { 0.7 }
        });

        let fused = rollout.fuse_heads(&attention);
        assert!((fused[[0, 0]] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_trading_regime_detection() {
        let analyzer = TradingAttentionAnalyzer::new();

        // Create attention that focuses on recent data (momentum)
        let mut attention = Array3::zeros((1, 10, 10));
        for i in 0..10 {
            attention[[0, i, 9]] = 0.8;
            for j in 0..9 {
                attention[[0, i, j]] = 0.2 / 9.0;
            }
        }

        let result = analyzer.analyze(&[attention]).unwrap();
        assert_eq!(result.regime, AttentionRegime::Momentum);
    }

    #[test]
    fn test_backtest_metrics() {
        let returns = vec![0.01, -0.02, 0.015, 0.005, -0.01];
        let sharpe = backtest::calculate_sharpe(&returns, 0.02);
        assert!(sharpe.is_finite());

        let equity = vec![100.0, 101.0, 99.0, 100.5, 100.75, 99.75];
        let max_dd = backtest::calculate_max_drawdown(&equity);
        assert!(max_dd > 0.0);
        assert!(max_dd < 1.0);
    }
}
