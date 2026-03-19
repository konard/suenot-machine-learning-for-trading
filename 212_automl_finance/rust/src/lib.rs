//! AutoML Finance - Automated Machine Learning Pipeline for Trading
//!
//! This library implements a complete AutoML system for financial time series,
//! including automated feature selection, model zoo, hyperparameter optimization,
//! walk-forward cross-validation, and ensemble construction.

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Data Structures
// ============================================================================

/// OHLCV candle from exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub open_time: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit kline API response
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// A feature matrix with named columns
#[derive(Debug, Clone)]
pub struct FeatureMatrix {
    pub data: Array2<f64>,
    pub feature_names: Vec<String>,
    pub target: Array1<f64>,
}

/// Model hyperparameter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperParams {
    pub params: HashMap<String, f64>,
}

impl HyperParams {
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }

    pub fn set(&mut self, key: &str, value: f64) -> &mut Self {
        self.params.insert(key.to_string(), value);
        self
    }

    pub fn get(&self, key: &str, default: f64) -> f64 {
        self.params.get(key).copied().unwrap_or(default)
    }
}

impl Default for HyperParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of evaluating a pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub model_type: ModelType,
    pub hyperparams: HyperParams,
    pub selected_features: Vec<usize>,
    pub mean_score: f64,
    pub std_score: f64,
    pub fold_scores: Vec<f64>,
}

// ============================================================================
// Bybit API Client
// ============================================================================

/// Fetch historical klines from Bybit
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let response: BybitResponse = client.get(&url).send()?.json()?;

    if response.ret_code != 0 {
        return Err(anyhow!("Bybit API error: {}", response.ret_msg));
    }

    let mut candles: Vec<Candle> = response
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(Candle {
                    open_time: row[0].parse().ok()?,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns newest first, reverse to chronological order
    candles.reverse();
    Ok(candles)
}

// ============================================================================
// Feature Engineering
// ============================================================================

/// Generate technical features from OHLCV candles
pub fn generate_features(candles: &[Candle], forecast_horizon: usize) -> Result<FeatureMatrix> {
    let n = candles.len();
    if n < 60 + forecast_horizon {
        return Err(anyhow!(
            "Need at least {} candles, got {}",
            60 + forecast_horizon,
            n
        ));
    }

    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

    let mut feature_names = Vec::new();
    let mut feature_columns: Vec<Vec<f64>> = Vec::new();

    // Simple Moving Averages (ratio to current price)
    for window in [5, 10, 20, 50] {
        let sma = sma(&closes, window);
        let ratio: Vec<f64> = closes
            .iter()
            .zip(sma.iter())
            .map(|(c, s)| if *s != 0.0 { c / s - 1.0 } else { 0.0 })
            .collect();
        feature_names.push(format!("sma_ratio_{}", window));
        feature_columns.push(ratio);
    }

    // RSI
    for period in [7, 14, 21] {
        let rsi = rsi(&closes, period);
        feature_names.push(format!("rsi_{}", period));
        feature_columns.push(rsi);
    }

    // Volatility (rolling std of returns)
    let returns: Vec<f64> = (0..n)
        .map(|i| {
            if i == 0 {
                0.0
            } else {
                (closes[i] / closes[i - 1]).ln()
            }
        })
        .collect();

    for window in [5, 10, 20] {
        let vol = rolling_std(&returns, window);
        feature_names.push(format!("volatility_{}", window));
        feature_columns.push(vol);
    }

    // High-Low range ratio
    let hl_ratio: Vec<f64> = highs
        .iter()
        .zip(lows.iter())
        .zip(closes.iter())
        .map(|((h, l), c)| if *c != 0.0 { (h - l) / c } else { 0.0 })
        .collect();
    feature_names.push("hl_range_ratio".to_string());
    feature_columns.push(hl_ratio);

    // Volume change
    let vol_change: Vec<f64> = (0..n)
        .map(|i| {
            if i == 0 || volumes[i - 1] == 0.0 {
                0.0
            } else {
                volumes[i] / volumes[i - 1] - 1.0
            }
        })
        .collect();
    feature_names.push("volume_change".to_string());
    feature_columns.push(vol_change);

    // Volume SMA ratio
    let vol_sma = sma(&volumes, 20);
    let vol_ratio: Vec<f64> = volumes
        .iter()
        .zip(vol_sma.iter())
        .map(|(v, s)| if *s != 0.0 { v / s - 1.0 } else { 0.0 })
        .collect();
    feature_names.push("volume_sma_ratio".to_string());
    feature_columns.push(vol_ratio);

    // Return momentum
    for window in [1, 3, 5, 10] {
        let mom: Vec<f64> = (0..n)
            .map(|i| {
                if i < window || closes[i - window] == 0.0 {
                    0.0
                } else {
                    closes[i] / closes[i - window] - 1.0
                }
            })
            .collect();
        feature_names.push(format!("momentum_{}", window));
        feature_columns.push(mom);
    }

    // Target: future return over forecast_horizon
    let target: Vec<f64> = (0..n)
        .map(|i| {
            if i + forecast_horizon < n && closes[i] != 0.0 {
                closes[i + forecast_horizon] / closes[i] - 1.0
            } else {
                0.0
            }
        })
        .collect();

    // Trim to valid range (skip first 50 for indicator warmup, skip last forecast_horizon)
    let start = 50;
    let end = n - forecast_horizon;
    if end <= start {
        return Err(anyhow!("Not enough data after trimming"));
    }

    let valid_len = end - start;
    let num_features = feature_columns.len();

    let mut data = Array2::<f64>::zeros((valid_len, num_features));
    for (col_idx, col) in feature_columns.iter().enumerate() {
        for row_idx in 0..valid_len {
            data[[row_idx, col_idx]] = col[start + row_idx];
        }
    }

    let target_arr = Array1::from_vec(target[start..end].to_vec());

    Ok(FeatureMatrix {
        data,
        feature_names,
        target: target_arr,
    })
}

// Technical indicator helpers

fn sma(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];
    let mut sum = 0.0;
    for i in 0..n {
        sum += data[i];
        if i >= window {
            sum -= data[i - window];
        }
        if i >= window - 1 {
            result[i] = sum / window as f64;
        } else {
            result[i] = data[i]; // fallback for warmup period
        }
    }
    result
}

fn rsi(closes: &[f64], period: usize) -> Vec<f64> {
    let n = closes.len();
    let mut result = vec![50.0; n];

    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    for i in 1..n {
        let change = closes[i] - closes[i - 1];
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { -change } else { 0.0 };

        if i <= period {
            avg_gain += gain / period as f64;
            avg_loss += loss / period as f64;
        } else {
            avg_gain = (avg_gain * (period - 1) as f64 + gain) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + loss) / period as f64;
        }

        if i >= period {
            if avg_loss == 0.0 {
                result[i] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                result[i] = 100.0 - 100.0 / (1.0 + rs);
            }
        }
    }
    result
}

fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];
    for i in window - 1..n {
        let slice = &data[i + 1 - window..=i];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let var: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
        result[i] = var.sqrt();
    }
    result
}

// ============================================================================
// Feature Selection
// ============================================================================

/// Automated feature selection using mutual information and correlation filtering
pub struct FeatureSelector {
    pub mi_threshold: f64,
    pub correlation_threshold: f64,
}

impl FeatureSelector {
    pub fn new(mi_threshold: f64, correlation_threshold: f64) -> Self {
        Self {
            mi_threshold,
            correlation_threshold,
        }
    }

    /// Select features based on mutual information with target and inter-feature correlation
    pub fn select(&self, fm: &FeatureMatrix) -> Vec<usize> {
        let n_features = fm.data.ncols();
        let n_samples = fm.data.nrows();

        // Step 1: Compute mutual information (approximated via correlation magnitude)
        let mut mi_scores: Vec<(usize, f64)> = Vec::new();
        for j in 0..n_features {
            let col: Vec<f64> = fm.data.column(j).to_vec();
            let target: Vec<f64> = fm.target.to_vec();
            let mi = mutual_information_approx(&col, &target, n_samples);
            mi_scores.push((j, mi));
        }

        // Filter by MI threshold
        let mut candidates: Vec<(usize, f64)> = mi_scores
            .into_iter()
            .filter(|(_, mi)| *mi > self.mi_threshold)
            .collect();

        // Sort by MI descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Step 2: Correlation filtering - greedily add features with low pairwise correlation
        let mut selected: Vec<usize> = Vec::new();
        for (idx, _mi) in &candidates {
            let col: Vec<f64> = fm.data.column(*idx).to_vec();
            let mut too_correlated = false;
            for &sel_idx in &selected {
                let sel_col: Vec<f64> = fm.data.column(sel_idx).to_vec();
                let corr = pearson_correlation(
                    &col,
                    &sel_col,
                );
                if corr.abs() > self.correlation_threshold {
                    too_correlated = true;
                    break;
                }
            }
            if !too_correlated {
                selected.push(*idx);
            }
        }

        selected
    }
}

/// Approximate mutual information using binned entropy estimation
fn mutual_information_approx(x: &[f64], y: &[f64], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let bins = 10usize;

    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let x_range = if (x_max - x_min).abs() < 1e-12 { 1.0 } else { x_max - x_min };
    let y_range = if (y_max - y_min).abs() < 1e-12 { 1.0 } else { y_max - y_min };

    let mut joint = vec![vec![0u32; bins]; bins];
    let mut x_marginal = vec![0u32; bins];
    let mut y_marginal = vec![0u32; bins];

    for i in 0..n {
        let xi = ((x[i] - x_min) / x_range * (bins - 1) as f64).round() as usize;
        let yi = ((y[i] - y_min) / y_range * (bins - 1) as f64).round() as usize;
        let xi = xi.min(bins - 1);
        let yi = yi.min(bins - 1);
        joint[xi][yi] += 1;
        x_marginal[xi] += 1;
        y_marginal[yi] += 1;
    }

    let nf = n as f64;
    let mut mi = 0.0;
    for xi in 0..bins {
        for yi in 0..bins {
            if joint[xi][yi] > 0 && x_marginal[xi] > 0 && y_marginal[yi] > 0 {
                let p_xy = joint[xi][yi] as f64 / nf;
                let p_x = x_marginal[xi] as f64 / nf;
                let p_y = y_marginal[yi] as f64 / nf;
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }
    }
    mi.max(0.0)
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        cov / denom
    }
}

// ============================================================================
// Model Zoo
// ============================================================================

/// Supported model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    Linear,
    DecisionTree,
    NeuralNet,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Linear => write!(f, "Linear"),
            ModelType::DecisionTree => write!(f, "DecisionTree"),
            ModelType::NeuralNet => write!(f, "NeuralNet"),
        }
    }
}

/// Trait for all models in the zoo
pub trait Model: std::fmt::Debug {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>);
    fn predict(&self, x: &Array2<f64>) -> Array1<f64>;
    fn model_type(&self) -> ModelType;
}

// --- Linear Model (Ridge Regression) ---

#[derive(Debug, Clone)]
pub struct LinearModel {
    pub alpha: f64,
    weights: Array1<f64>,
    bias: f64,
}

impl LinearModel {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            weights: Array1::zeros(0),
            bias: 0.0,
        }
    }

    pub fn from_params(params: &HyperParams) -> Self {
        Self::new(params.get("alpha", 1.0))
    }
}

impl Model for LinearModel {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let n = x.nrows() as f64;
        let p = x.ncols();

        // Gradient descent for ridge regression
        self.weights = Array1::zeros(p);
        self.bias = y.mean().unwrap_or(0.0);
        let lr = 0.01;
        let epochs = 100;

        for _ in 0..epochs {
            let pred = x.dot(&self.weights) + self.bias;
            let error = &pred - y;

            let grad_w = x.t().dot(&error) / n + &self.weights * (2.0 * self.alpha);
            let grad_b = error.sum() / n;

            self.weights = &self.weights - &(&grad_w * lr);
            self.bias -= grad_b * lr;
        }
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        x.dot(&self.weights) + self.bias
    }

    fn model_type(&self) -> ModelType {
        ModelType::Linear
    }
}

// --- Decision Tree Model (simplified stump-based) ---

#[derive(Debug, Clone)]
struct TreeNode {
    feature_idx: usize,
    threshold: f64,
    left_value: f64,
    right_value: f64,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

#[derive(Debug, Clone)]
pub struct DecisionTreeModel {
    pub max_depth: usize,
    root: Option<TreeNode>,
}

impl DecisionTreeModel {
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            root: None,
        }
    }

    pub fn from_params(params: &HyperParams) -> Self {
        Self::new(params.get("max_depth", 3.0) as usize)
    }

    fn build_tree(
        x: &Array2<f64>,
        y: &Array1<f64>,
        indices: &[usize],
        depth: usize,
        max_depth: usize,
    ) -> Option<TreeNode> {
        if indices.len() < 4 || depth >= max_depth {
            let mean_val = indices.iter().map(|&i| y[i]).sum::<f64>()
                / indices.len().max(1) as f64;
            return Some(TreeNode {
                feature_idx: 0,
                threshold: 0.0,
                left_value: mean_val,
                right_value: mean_val,
                left: None,
                right: None,
            });
        }

        let n_features = x.ncols();
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_reduction = f64::NEG_INFINITY;
        let mut best_left_indices = Vec::new();
        let mut best_right_indices = Vec::new();

        let total_var = variance_of(y, indices);

        let mut rng = rand::thread_rng();
        let features_to_try = (n_features as f64).sqrt().ceil() as usize;

        for _ in 0..features_to_try {
            let feat = rng.gen_range(0..n_features);
            // Try median as threshold
            let mut vals: Vec<f64> = indices.iter().map(|&i| x[[i, feat]]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let threshold = vals[vals.len() / 2];

            let (left_idx, right_idx): (Vec<usize>, Vec<usize>) =
                indices.iter().partition(|&&i| x[[i, feat]] <= threshold);

            if left_idx.is_empty() || right_idx.is_empty() {
                continue;
            }

            let left_var = variance_of(y, &left_idx);
            let right_var = variance_of(y, &right_idx);
            let weighted_var = (left_idx.len() as f64 * left_var
                + right_idx.len() as f64 * right_var)
                / indices.len() as f64;
            let reduction = total_var - weighted_var;

            if reduction > best_reduction {
                best_reduction = reduction;
                best_feature = feat;
                best_threshold = threshold;
                best_left_indices = left_idx;
                best_right_indices = right_idx;
            }
        }

        if best_left_indices.is_empty() || best_right_indices.is_empty() {
            let mean_val =
                indices.iter().map(|&i| y[i]).sum::<f64>() / indices.len().max(1) as f64;
            return Some(TreeNode {
                feature_idx: 0,
                threshold: 0.0,
                left_value: mean_val,
                right_value: mean_val,
                left: None,
                right: None,
            });
        }

        let left_val =
            best_left_indices.iter().map(|&i| y[i]).sum::<f64>() / best_left_indices.len() as f64;
        let right_val = best_right_indices.iter().map(|&i| y[i]).sum::<f64>()
            / best_right_indices.len() as f64;

        Some(TreeNode {
            feature_idx: best_feature,
            threshold: best_threshold,
            left_value: left_val,
            right_value: right_val,
            left: Self::build_tree(x, y, &best_left_indices, depth + 1, max_depth).map(Box::new),
            right: Self::build_tree(x, y, &best_right_indices, depth + 1, max_depth).map(Box::new),
        })
    }

    fn predict_one(node: &TreeNode, row: &[f64]) -> f64 {
        if row[node.feature_idx] <= node.threshold {
            match &node.left {
                Some(child) => Self::predict_one(child, row),
                None => node.left_value,
            }
        } else {
            match &node.right {
                Some(child) => Self::predict_one(child, row),
                None => node.right_value,
            }
        }
    }
}

fn variance_of(y: &Array1<f64>, indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let n = indices.len() as f64;
    let mean = indices.iter().map(|&i| y[i]).sum::<f64>() / n;
    indices.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f64>() / n
}

impl Model for DecisionTreeModel {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let indices: Vec<usize> = (0..x.nrows()).collect();
        self.root = Self::build_tree(x, y, &indices, 0, self.max_depth);
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let n = x.nrows();
        let mut preds = Array1::zeros(n);
        if let Some(ref root) = self.root {
            for i in 0..n {
                let row: Vec<f64> = x.row(i).to_vec();
                preds[i] = Self::predict_one(root, &row);
            }
        }
        preds
    }

    fn model_type(&self) -> ModelType {
        ModelType::DecisionTree
    }
}

// --- Neural Network Model (two-layer perceptron) ---

#[derive(Debug, Clone)]
pub struct NeuralNetModel {
    pub hidden_size: usize,
    pub learning_rate: f64,
    pub epochs: usize,
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array1<f64>,
    b2: f64,
}

impl NeuralNetModel {
    pub fn new(hidden_size: usize, learning_rate: f64, epochs: usize) -> Self {
        Self {
            hidden_size,
            learning_rate,
            epochs,
            w1: Array2::zeros((0, 0)),
            b1: Array1::zeros(0),
            w2: Array1::zeros(0),
            b2: 0.0,
        }
    }

    pub fn from_params(params: &HyperParams) -> Self {
        Self::new(
            params.get("hidden_size", 16.0) as usize,
            params.get("learning_rate", 0.001),
            params.get("epochs", 50.0) as usize,
        )
    }

    fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    fn relu_deriv(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}

impl Model for NeuralNetModel {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let n = x.nrows();
        let p = x.ncols();
        let h = self.hidden_size;

        // Xavier initialization
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / p as f64).sqrt();
        let scale2 = (2.0 / h as f64).sqrt();

        self.w1 = Array2::from_shape_fn((p, h), |_| rng.gen_range(-scale1..scale1));
        self.b1 = Array1::zeros(h);
        self.w2 = Array1::from_shape_fn(h, |_| rng.gen_range(-scale2..scale2));
        self.b2 = 0.0;

        for _ in 0..self.epochs {
            // Forward pass
            let z1 = x.dot(&self.w1) + &self.b1;
            let a1 = z1.mapv(Self::relu);
            let pred = a1.dot(&self.w2) + self.b2;
            let error = &pred - y;

            let nf = n as f64;

            // Backward pass
            // d_loss/d_pred = error / n
            let d_pred = &error / nf;

            // d_loss/d_w2 = a1^T * d_pred
            let d_w2 = a1.t().dot(&d_pred);
            let d_b2 = d_pred.sum();

            // d_loss/d_a1 = d_pred * w2^T
            let d_a1 = Array2::from_shape_fn((n, h), |(_i, j)| d_pred[_i] * self.w2[j]);
            let d_z1 = &d_a1 * &z1.mapv(Self::relu_deriv);

            let d_w1 = x.t().dot(&d_z1);
            let d_b1 = d_z1.sum_axis(Axis(0));

            // Update
            self.w2 = &self.w2 - &(&d_w2 * self.learning_rate);
            self.b2 -= d_b2 * self.learning_rate;
            self.w1 = &self.w1 - &(&d_w1 * self.learning_rate);
            self.b1 = &self.b1 - &(&d_b1 * self.learning_rate);
        }
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let z1 = x.dot(&self.w1) + &self.b1;
        let a1 = z1.mapv(Self::relu);
        a1.dot(&self.w2) + self.b2
    }

    fn model_type(&self) -> ModelType {
        ModelType::NeuralNet
    }
}

/// Create a model from type and hyperparameters
pub fn create_model(model_type: ModelType, params: &HyperParams) -> Box<dyn Model> {
    match model_type {
        ModelType::Linear => Box::new(LinearModel::from_params(params)),
        ModelType::DecisionTree => Box::new(DecisionTreeModel::from_params(params)),
        ModelType::NeuralNet => Box::new(NeuralNetModel::from_params(params)),
    }
}

// ============================================================================
// Walk-Forward Cross-Validation
// ============================================================================

/// Walk-forward time-series cross-validation
pub struct WalkForwardCV {
    pub n_splits: usize,
    pub train_ratio: f64,
    pub embargo: usize,
}

impl WalkForwardCV {
    pub fn new(n_splits: usize, train_ratio: f64, embargo: usize) -> Self {
        Self {
            n_splits,
            train_ratio,
            embargo,
        }
    }

    /// Generate train/validation index pairs respecting temporal order
    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut splits = Vec::new();
        let val_size = n_samples / (self.n_splits + 1);
        if val_size == 0 {
            return splits;
        }

        for i in 0..self.n_splits {
            let val_start = (i + 1) * val_size;
            let val_end = (val_start + val_size).min(n_samples);
            let train_start = 0;
            let train_end = if val_start > self.embargo {
                val_start - self.embargo
            } else {
                0
            };

            if train_end <= train_start || val_end <= val_start {
                continue;
            }

            let train_indices: Vec<usize> = (train_start..train_end).collect();
            let val_indices: Vec<usize> = (val_start..val_end).collect();
            splits.push((train_indices, val_indices));
        }

        splits
    }

    /// Evaluate a model using walk-forward CV, returns (mean_mse, std_mse, fold_scores)
    pub fn evaluate(
        &self,
        fm: &FeatureMatrix,
        selected_features: &[usize],
        model_type: ModelType,
        params: &HyperParams,
    ) -> (f64, f64, Vec<f64>) {
        let splits = self.split(fm.data.nrows());
        let mut scores = Vec::new();

        // Select feature columns
        let x_selected = select_columns(&fm.data, selected_features);

        for (train_idx, val_idx) in &splits {
            let x_train = select_rows(&x_selected, train_idx);
            let y_train = Array1::from_vec(train_idx.iter().map(|&i| fm.target[i]).collect());
            let x_val = select_rows(&x_selected, val_idx);
            let y_val = Array1::from_vec(val_idx.iter().map(|&i| fm.target[i]).collect());

            let mut model = create_model(model_type, params);
            model.fit(&x_train, &y_train);
            let pred = model.predict(&x_val);

            // MSE (negated so higher is better)
            let mse = (&pred - &y_val).mapv(|x| x.powi(2)).mean().unwrap_or(f64::MAX);
            scores.push(-mse);
        }

        if scores.is_empty() {
            return (f64::NEG_INFINITY, 0.0, scores);
        }

        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let std = (scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64)
            .sqrt();
        (mean, std, scores)
    }
}

fn select_columns(data: &Array2<f64>, cols: &[usize]) -> Array2<f64> {
    let n = data.nrows();
    let p = cols.len();
    let mut result = Array2::zeros((n, p));
    for (new_j, &old_j) in cols.iter().enumerate() {
        for i in 0..n {
            result[[i, new_j]] = data[[i, old_j]];
        }
    }
    result
}

fn select_rows(data: &Array2<f64>, rows: &[usize]) -> Array2<f64> {
    let n = rows.len();
    let p = data.ncols();
    let mut result = Array2::zeros((n, p));
    for (new_i, &old_i) in rows.iter().enumerate() {
        for j in 0..p {
            result[[new_i, j]] = data[[old_i, j]];
        }
    }
    result
}

// ============================================================================
// Hyperparameter Optimization Engine
// ============================================================================

/// HPO search strategy
#[derive(Debug, Clone, Copy)]
pub enum SearchStrategy {
    Random,
    BayesianInspired,
}

/// Hyperparameter optimization engine
pub struct HPOEngine {
    pub strategy: SearchStrategy,
    pub n_iterations: usize,
    history: Vec<(HyperParams, f64)>,
}

impl HPOEngine {
    pub fn new(strategy: SearchStrategy, n_iterations: usize) -> Self {
        Self {
            strategy,
            n_iterations,
            history: Vec::new(),
        }
    }

    /// Sample hyperparameters for a given model type
    pub fn sample_params(&self, model_type: ModelType, rng: &mut impl Rng) -> HyperParams {
        let mut params = HyperParams::new();

        match self.strategy {
            SearchStrategy::Random => {
                self.sample_random(model_type, &mut params, rng);
            }
            SearchStrategy::BayesianInspired => {
                if self.history.len() < 3 {
                    self.sample_random(model_type, &mut params, rng);
                } else {
                    self.sample_near_best(model_type, &mut params, rng);
                }
            }
        }

        params
    }

    fn sample_random(&self, model_type: ModelType, params: &mut HyperParams, rng: &mut impl Rng) {
        match model_type {
            ModelType::Linear => {
                params.set("alpha", 10f64.powf(rng.gen_range(-4.0..2.0)));
            }
            ModelType::DecisionTree => {
                params.set("max_depth", rng.gen_range(2.0_f64..8.0).floor());
            }
            ModelType::NeuralNet => {
                let hidden_sizes = [8.0, 16.0, 32.0, 64.0];
                params.set("hidden_size", hidden_sizes[rng.gen_range(0..hidden_sizes.len())]);
                params.set("learning_rate", 10f64.powf(rng.gen_range(-4.0..-1.0)));
                params.set("epochs", rng.gen_range(30.0_f64..100.0).floor());
            }
        }
    }

    fn sample_near_best(
        &self,
        model_type: ModelType,
        params: &mut HyperParams,
        rng: &mut impl Rng,
    ) {
        // Find best configuration in history
        let best = self
            .history
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((best_params, _)) = best {
            // Perturb best params with noise
            match model_type {
                ModelType::Linear => {
                    let base = best_params.get("alpha", 1.0).ln();
                    let perturbed = (base + rng.gen_range(-1.0..1.0)).exp();
                    params.set("alpha", perturbed.clamp(1e-4, 100.0));
                }
                ModelType::DecisionTree => {
                    let base = best_params.get("max_depth", 3.0);
                    let perturbed = base + rng.gen_range(-1.0..1.0);
                    params.set("max_depth", perturbed.clamp(2.0, 7.0).floor());
                }
                ModelType::NeuralNet => {
                    let base_h = best_params.get("hidden_size", 16.0);
                    let base_lr = best_params.get("learning_rate", 0.001).ln();
                    let base_ep = best_params.get("epochs", 50.0);

                    let hidden_sizes = [8.0, 16.0, 32.0, 64.0];
                    let h_idx = hidden_sizes
                        .iter()
                        .position(|&h| (h - base_h).abs() < 1.0)
                        .unwrap_or(1);
                    let new_idx = (h_idx as i32 + rng.gen_range(-1..=1))
                        .clamp(0, hidden_sizes.len() as i32 - 1)
                        as usize;
                    params.set("hidden_size", hidden_sizes[new_idx]);
                    params.set(
                        "learning_rate",
                        (base_lr + rng.gen_range(-0.5..0.5)).exp().clamp(1e-4, 0.1),
                    );
                    params.set(
                        "epochs",
                        (base_ep + rng.gen_range(-10.0..10.0)).clamp(30.0, 100.0).floor(),
                    );
                }
            }
        } else {
            self.sample_random(model_type, params, rng);
        }
    }

    /// Record a result for Bayesian-inspired search
    pub fn record(&mut self, params: HyperParams, score: f64) {
        self.history.push((params, score));
    }
}

// ============================================================================
// Ensemble Builder
// ============================================================================

/// Build weighted ensemble from top models
pub struct EnsembleBuilder {
    pub top_k: usize,
}

impl EnsembleBuilder {
    pub fn new(top_k: usize) -> Self {
        Self { top_k }
    }

    /// Select top-k pipeline results and compute ensemble weights
    pub fn build(&self, results: &[PipelineResult]) -> Vec<(usize, f64)> {
        let mut sorted: Vec<(usize, f64)> = results
            .iter()
            .enumerate()
            .map(|(i, r)| (i, r.mean_score))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top: Vec<(usize, f64)> = sorted.into_iter().take(self.top_k).collect();

        // Normalize weights (shift scores to be positive)
        let min_score = top
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::INFINITY, f64::min);
        let shifted: Vec<f64> = top.iter().map(|(_, s)| s - min_score + 1e-8).collect();
        let total: f64 = shifted.iter().sum();

        top.iter()
            .zip(shifted.iter())
            .map(|((idx, _), w)| (*idx, w / total))
            .collect()
    }

    /// Generate ensemble predictions
    pub fn predict_ensemble(
        &self,
        fm: &FeatureMatrix,
        results: &[PipelineResult],
        weights: &[(usize, f64)],
    ) -> Array1<f64> {
        let n = fm.data.nrows();
        let mut ensemble_pred = Array1::zeros(n);

        for &(idx, weight) in weights {
            let result = &results[idx];
            let x_selected = select_columns(&fm.data, &result.selected_features);
            let mut model = create_model(result.model_type, &result.hyperparams);

            // Fit on all data for final prediction
            let y = fm.target.clone();
            model.fit(&x_selected, &y);
            let pred = model.predict(&x_selected);
            ensemble_pred = ensemble_pred + &pred * weight;
        }

        ensemble_pred
    }
}

// ============================================================================
// AutoML Pipeline Orchestrator
// ============================================================================

/// Main AutoML pipeline that ties everything together
pub struct AutoMLPipeline {
    pub feature_selector: FeatureSelector,
    pub cv: WalkForwardCV,
    pub hpo_engine: HPOEngine,
    pub ensemble_builder: EnsembleBuilder,
    pub model_types: Vec<ModelType>,
}

impl AutoMLPipeline {
    pub fn new(
        mi_threshold: f64,
        correlation_threshold: f64,
        n_cv_splits: usize,
        embargo: usize,
        search_strategy: SearchStrategy,
        hpo_iterations: usize,
        ensemble_top_k: usize,
    ) -> Self {
        Self {
            feature_selector: FeatureSelector::new(mi_threshold, correlation_threshold),
            cv: WalkForwardCV::new(n_cv_splits, 0.7, embargo),
            hpo_engine: HPOEngine::new(search_strategy, hpo_iterations),
            ensemble_builder: EnsembleBuilder::new(ensemble_top_k),
            model_types: vec![ModelType::Linear, ModelType::DecisionTree, ModelType::NeuralNet],
        }
    }

    /// Run the complete AutoML pipeline
    pub fn run(&mut self, fm: &FeatureMatrix) -> Result<(Vec<PipelineResult>, Vec<(usize, f64)>)> {
        // Step 1: Feature selection
        let selected_features = self.feature_selector.select(fm);
        if selected_features.is_empty() {
            return Err(anyhow!("No features selected — try lowering MI threshold"));
        }

        println!(
            "Selected {} features: {:?}",
            selected_features.len(),
            selected_features
                .iter()
                .map(|&i| &fm.feature_names[i])
                .collect::<Vec<_>>()
        );

        // Step 2: Model search with HPO
        let mut all_results = Vec::new();
        let mut rng = rand::thread_rng();

        for &model_type in &self.model_types {
            println!("\nSearching {} configurations...", model_type);
            for iter in 0..self.hpo_engine.n_iterations {
                let params = self.hpo_engine.sample_params(model_type, &mut rng);
                let (mean_score, std_score, fold_scores) =
                    self.cv.evaluate(fm, &selected_features, model_type, &params);

                self.hpo_engine.record(params.clone(), mean_score);

                all_results.push(PipelineResult {
                    model_type,
                    hyperparams: params,
                    selected_features: selected_features.clone(),
                    mean_score,
                    std_score,
                    fold_scores,
                });

                if (iter + 1) % 5 == 0 {
                    println!(
                        "  Iteration {}/{}: best so far = {:.6}",
                        iter + 1,
                        self.hpo_engine.n_iterations,
                        all_results
                            .iter()
                            .filter(|r| r.model_type == model_type)
                            .map(|r| r.mean_score)
                            .fold(f64::NEG_INFINITY, f64::max)
                    );
                }
            }
        }

        // Step 3: Build ensemble from top models
        let ensemble_weights = self.ensemble_builder.build(&all_results);

        println!("\nEnsemble composition:");
        for &(idx, weight) in &ensemble_weights {
            let r = &all_results[idx];
            println!(
                "  {} (score={:.6}, weight={:.3})",
                r.model_type, r.mean_score, weight
            );
        }

        Ok((all_results, ensemble_weights))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_synthetic_data(n: usize, p: usize) -> FeatureMatrix {
        let mut rng = rand::thread_rng();
        let data = Array2::from_shape_fn((n, p), |_| rng.gen_range(-1.0..1.0));
        // target = sum of first 2 features + noise
        let target = Array1::from_shape_fn(n, |i| {
            data[[i, 0]] * 0.5 + data[[i, 1]] * 0.3 + rng.gen_range(-0.1..0.1)
        });
        FeatureMatrix {
            data,
            feature_names: (0..p).map(|i| format!("feat_{}", i)).collect(),
            target,
        }
    }

    #[test]
    fn test_feature_selector() {
        let fm = make_synthetic_data(200, 5);
        let selector = FeatureSelector::new(0.0, 0.95);
        let selected = selector.select(&fm);
        assert!(!selected.is_empty(), "Should select at least one feature");
    }

    #[test]
    fn test_linear_model() {
        let fm = make_synthetic_data(100, 3);
        let mut model = LinearModel::new(0.01);
        model.fit(&fm.data, &fm.target);
        let preds = model.predict(&fm.data);
        assert_eq!(preds.len(), 100);

        // Predictions should have some correlation with target
        let corr = pearson_correlation(preds.as_slice().unwrap(), fm.target.as_slice().unwrap());
        assert!(corr > 0.3, "Linear model should learn some signal, got r={}", corr);
    }

    #[test]
    fn test_decision_tree_model() {
        let fm = make_synthetic_data(100, 3);
        let mut model = DecisionTreeModel::new(4);
        model.fit(&fm.data, &fm.target);
        let preds = model.predict(&fm.data);
        assert_eq!(preds.len(), 100);
    }

    #[test]
    fn test_neural_net_model() {
        let fm = make_synthetic_data(100, 3);
        let mut model = NeuralNetModel::new(16, 0.01, 50);
        model.fit(&fm.data, &fm.target);
        let preds = model.predict(&fm.data);
        assert_eq!(preds.len(), 100);
    }

    #[test]
    fn test_walk_forward_cv() {
        let cv = WalkForwardCV::new(3, 0.7, 2);
        let splits = cv.split(100);
        assert!(!splits.is_empty(), "Should produce at least one split");

        // Verify temporal ordering
        for (train, val) in &splits {
            let train_max = *train.last().unwrap();
            let val_min = *val.first().unwrap();
            assert!(
                val_min > train_max,
                "Validation must come after training (with embargo)"
            );
        }
    }

    #[test]
    fn test_walk_forward_cv_evaluate() {
        let fm = make_synthetic_data(200, 5);
        let cv = WalkForwardCV::new(3, 0.7, 1);
        let features: Vec<usize> = (0..5).collect();
        let params = HyperParams::new();
        let (mean, _std, scores) = cv.evaluate(&fm, &features, ModelType::Linear, &params);
        assert!(!scores.is_empty(), "Should produce scores");
        assert!(mean.is_finite(), "Mean score should be finite");
    }

    #[test]
    fn test_hpo_random_search() {
        let mut hpo = HPOEngine::new(SearchStrategy::Random, 5);
        let mut rng = rand::thread_rng();

        for model_type in [ModelType::Linear, ModelType::DecisionTree, ModelType::NeuralNet] {
            let params = hpo.sample_params(model_type, &mut rng);
            assert!(!params.params.is_empty(), "Should sample some parameters");
            hpo.record(params, -0.01);
        }
    }

    #[test]
    fn test_hpo_bayesian_inspired() {
        let mut hpo = HPOEngine::new(SearchStrategy::BayesianInspired, 5);
        let mut rng = rand::thread_rng();

        // Seed history
        for _ in 0..5 {
            let mut p = HyperParams::new();
            p.set("alpha", rng.gen_range(0.01..10.0));
            hpo.record(p, rng.gen_range(-1.0..0.0));
        }

        let params = hpo.sample_params(ModelType::Linear, &mut rng);
        assert!(params.params.contains_key("alpha"));
    }

    #[test]
    fn test_ensemble_builder() {
        let results = vec![
            PipelineResult {
                model_type: ModelType::Linear,
                hyperparams: HyperParams::new(),
                selected_features: vec![0, 1],
                mean_score: -0.01,
                std_score: 0.001,
                fold_scores: vec![-0.01, -0.012],
            },
            PipelineResult {
                model_type: ModelType::DecisionTree,
                hyperparams: HyperParams::new(),
                selected_features: vec![0, 1],
                mean_score: -0.02,
                std_score: 0.002,
                fold_scores: vec![-0.02, -0.022],
            },
            PipelineResult {
                model_type: ModelType::NeuralNet,
                hyperparams: HyperParams::new(),
                selected_features: vec![0, 1],
                mean_score: -0.015,
                std_score: 0.001,
                fold_scores: vec![-0.015, -0.016],
            },
        ];

        let builder = EnsembleBuilder::new(2);
        let weights = builder.build(&results);
        assert_eq!(weights.len(), 2, "Should select top-2 models");

        let total_weight: f64 = weights.iter().map(|(_, w)| w).sum();
        assert!(
            (total_weight - 1.0).abs() < 1e-6,
            "Weights should sum to 1.0"
        );
    }

    #[test]
    fn test_automl_pipeline_synthetic() {
        let fm = make_synthetic_data(300, 8);
        let mut pipeline = AutoMLPipeline::new(
            0.0,  // low MI threshold for synthetic data
            0.95, // correlation threshold
            3,    // CV splits
            2,    // embargo
            SearchStrategy::Random,
            3,    // HPO iterations (small for test speed)
            2,    // ensemble top-k
        );

        let result = pipeline.run(&fm);
        assert!(result.is_ok(), "Pipeline should complete without error");

        let (results, weights) = result.unwrap();
        assert!(!results.is_empty(), "Should have evaluated some configurations");
        assert!(!weights.is_empty(), "Should have ensemble weights");
    }

    #[test]
    fn test_mutual_information() {
        // Identical arrays should have high MI
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = x.clone();
        let mi = mutual_information_approx(&x, &y, x.len());
        assert!(mi > 0.0, "Identical arrays should have positive MI");
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = pearson_correlation(&x, &y);
        assert!(
            (corr - 1.0).abs() < 1e-10,
            "Perfect linear relationship should give r=1.0"
        );
    }

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&data, 3);
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyperparams() {
        let mut hp = HyperParams::new();
        hp.set("alpha", 0.5);
        assert!((hp.get("alpha", 1.0) - 0.5).abs() < 1e-10);
        assert!((hp.get("beta", 2.0) - 2.0).abs() < 1e-10);
    }
}
