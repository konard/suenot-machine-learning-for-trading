//! Risk metric calculations using normalizing flows.

use crate::flows::traits::NormalizingFlow;
use std::collections::HashMap;

/// Compute Value at Risk (VaR) using flow samples.
///
/// VaR_alpha is the threshold such that P(X < VaR_alpha) = alpha.
///
/// # Arguments
/// * `flow` - Trained normalizing flow model
/// * `alpha` - Confidence level (e.g., 0.05 for 95% VaR)
/// * `n_samples` - Number of Monte Carlo samples
///
/// # Returns
/// VaR value
pub fn compute_var<F: NormalizingFlow>(flow: &F, alpha: f64, n_samples: usize) -> f64 {
    let mut samples = flow.sample_flat(n_samples);
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = ((alpha * n_samples as f64) as usize).max(1) - 1;
    samples[idx]
}

/// Compute VaR at multiple confidence levels.
///
/// # Arguments
/// * `flow` - Trained normalizing flow model
/// * `alpha_levels` - List of alpha values
/// * `n_samples` - Number of Monte Carlo samples
///
/// # Returns
/// HashMap mapping confidence level names to VaR values
pub fn compute_var_multiple<F: NormalizingFlow>(
    flow: &F,
    alpha_levels: &[f64],
    n_samples: usize,
) -> HashMap<String, f64> {
    let mut samples = flow.sample_flat(n_samples);
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut results = HashMap::new();

    for &alpha in alpha_levels {
        let idx = ((alpha * n_samples as f64) as usize).max(1) - 1;
        let var = samples[idx];
        let confidence = ((1.0 - alpha) * 100.0) as i32;
        results.insert(format!("VaR_{}", confidence), var);
    }

    results
}

/// Compute Conditional Value at Risk (CVaR / Expected Shortfall).
///
/// CVaR_alpha = E[X | X <= VaR_alpha]
///
/// # Arguments
/// * `flow` - Trained normalizing flow model
/// * `alpha` - Confidence level
/// * `n_samples` - Number of Monte Carlo samples
///
/// # Returns
/// (VaR, CVaR) tuple
pub fn compute_cvar<F: NormalizingFlow>(flow: &F, alpha: f64, n_samples: usize) -> (f64, f64) {
    let mut samples = flow.sample_flat(n_samples);
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = ((alpha * n_samples as f64) as usize).max(1) - 1;
    let var = samples[idx];

    // CVaR: average of samples below VaR
    let tail_samples: Vec<f64> = samples.iter().take(idx + 1).copied().collect();
    let cvar = if tail_samples.is_empty() {
        var
    } else {
        tail_samples.iter().sum::<f64>() / tail_samples.len() as f64
    };

    (var, cvar)
}

/// Compute tail probability P(X < threshold).
///
/// # Arguments
/// * `flow` - Trained normalizing flow model
/// * `threshold` - Return threshold
/// * `n_samples` - Number of Monte Carlo samples
///
/// # Returns
/// Tail probability
pub fn compute_tail_probability<F: NormalizingFlow>(
    flow: &F,
    threshold: f64,
    n_samples: usize,
) -> f64 {
    let samples = flow.sample_flat(n_samples);
    let count = samples.iter().filter(|&&x| x < threshold).count();
    count as f64 / n_samples as f64
}

/// Comprehensive risk metrics from flow.
#[derive(Debug, Clone)]
pub struct RiskMetrics {
    pub mean: f64,
    pub std: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub var_90: f64,
    pub var_95: f64,
    pub var_99: f64,
    pub cvar_90: f64,
    pub cvar_95: f64,
    pub cvar_99: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
}

/// Compute comprehensive risk metrics.
///
/// # Arguments
/// * `flow` - Trained normalizing flow model
/// * `n_samples` - Number of Monte Carlo samples
///
/// # Returns
/// RiskMetrics struct with all computed metrics
pub fn compute_risk_metrics<F: NormalizingFlow>(flow: &F, n_samples: usize) -> RiskMetrics {
    let samples = flow.sample_flat(n_samples);

    // Basic statistics
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;

    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    // Higher moments
    let m3 = samples.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
    let m4 = samples.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;

    let skewness = m3 / std.powi(3);
    let kurtosis = m4 / std.powi(4) - 3.0; // Excess kurtosis

    // VaR and CVaR
    let mut sorted = samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx_90 = (0.10 * n) as usize;
    let idx_95 = (0.05 * n) as usize;
    let idx_99 = (0.01 * n) as usize;
    let idx_50 = (0.50 * n) as usize;

    let var_90 = sorted[idx_90];
    let var_95 = sorted[idx_95];
    let var_99 = sorted[idx_99];
    let median = sorted[idx_50];

    let cvar_90 = sorted[..idx_90 + 1].iter().sum::<f64>() / (idx_90 + 1) as f64;
    let cvar_95 = sorted[..idx_95 + 1].iter().sum::<f64>() / (idx_95 + 1) as f64;
    let cvar_99 = sorted[..idx_99 + 1].iter().sum::<f64>() / (idx_99 + 1) as f64;

    RiskMetrics {
        mean,
        std,
        skewness,
        kurtosis,
        var_90,
        var_95,
        var_99,
        cvar_90,
        cvar_95,
        cvar_99,
        min: sorted[0],
        max: sorted[sorted.len() - 1],
        median,
    }
}

/// Compare flow distribution to historical data.
#[derive(Debug, Clone)]
pub struct DistributionComparison {
    pub mean_diff: f64,
    pub std_diff: f64,
    pub ks_statistic: f64,
}

/// Compare two distributions.
///
/// # Arguments
/// * `historical` - Historical return data
/// * `synthetic` - Synthetic samples from flow
///
/// # Returns
/// Comparison metrics
pub fn compare_distributions(historical: &[f64], synthetic: &[f64]) -> DistributionComparison {
    let hist_mean = historical.iter().sum::<f64>() / historical.len() as f64;
    let synth_mean = synthetic.iter().sum::<f64>() / synthetic.len() as f64;

    let hist_var = historical.iter().map(|x| (x - hist_mean).powi(2)).sum::<f64>()
        / historical.len() as f64;
    let synth_var =
        synthetic.iter().map(|x| (x - synth_mean).powi(2)).sum::<f64>() / synthetic.len() as f64;

    let hist_std = hist_var.sqrt();
    let synth_std = synth_var.sqrt();

    // Kolmogorov-Smirnov statistic (two-sample)
    let mut hist_sorted = historical.to_vec();
    let mut synth_sorted = synthetic.to_vec();
    hist_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    synth_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n1 = hist_sorted.len() as f64;
    let n2 = synth_sorted.len() as f64;

    let mut ks_stat = 0.0f64;

    // Simplified KS calculation
    for (i, &x) in hist_sorted.iter().enumerate() {
        let cdf1 = (i + 1) as f64 / n1;
        let cdf2 = synth_sorted.iter().filter(|&&s| s <= x).count() as f64 / n2;
        ks_stat = ks_stat.max((cdf1 - cdf2).abs());
    }

    DistributionComparison {
        mean_diff: synth_mean - hist_mean,
        std_diff: synth_std - hist_std,
        ks_statistic: ks_stat,
    }
}

/// Stress test results
#[derive(Debug, Clone)]
pub struct StressTestResult {
    pub scenario: String,
    pub base_var_95: f64,
    pub stressed_var_95: f64,
    pub base_var_99: f64,
    pub stressed_var_99: f64,
    pub max_loss_base: f64,
    pub max_loss_stressed: f64,
}

/// Perform stress test on flow samples.
///
/// # Arguments
/// * `flow` - Trained normalizing flow
/// * `scenario` - Stress scenario type ("2x_vol", "crash")
/// * `n_samples` - Number of samples
///
/// # Returns
/// Stress test results
pub fn stress_test<F: NormalizingFlow>(
    flow: &F,
    scenario: &str,
    n_samples: usize,
) -> StressTestResult {
    let base_samples = flow.sample_flat(n_samples);

    // Apply stress transformation
    let stressed_samples: Vec<f64> = match scenario {
        "2x_vol" => {
            let mean = base_samples.iter().sum::<f64>() / base_samples.len() as f64;
            base_samples
                .iter()
                .map(|&x| (x - mean) * 2.0 + mean)
                .collect()
        }
        "crash" => base_samples.iter().map(|&x| x - 0.05).collect(),
        _ => base_samples.clone(),
    };

    // Compute metrics
    let mut base_sorted = base_samples;
    let mut stressed_sorted = stressed_samples;
    base_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    stressed_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = base_sorted.len();
    let idx_95 = (0.05 * n as f64) as usize;
    let idx_99 = (0.01 * n as f64) as usize;

    StressTestResult {
        scenario: scenario.to_string(),
        base_var_95: base_sorted[idx_95],
        stressed_var_95: stressed_sorted[idx_95],
        base_var_99: base_sorted[idx_99],
        stressed_var_99: stressed_sorted[idx_99],
        max_loss_base: base_sorted[0],
        max_loss_stressed: stressed_sorted[0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flows::config::FlowConfig;
    use crate::flows::realnvp::RealNVP;

    #[test]
    fn test_var_calculation() {
        let config = FlowConfig::new(1);
        let flow = RealNVP::new(config);

        let var = compute_var(&flow, 0.05, 10000);
        // VaR should be negative (loss)
        assert!(var < 0.5); // Rough sanity check
    }

    #[test]
    fn test_cvar_calculation() {
        let config = FlowConfig::new(1);
        let flow = RealNVP::new(config);

        let (var, cvar) = compute_cvar(&flow, 0.05, 10000);
        // CVaR should be less than or equal to VaR
        assert!(cvar <= var + 1e-6);
    }

    #[test]
    fn test_risk_metrics() {
        let config = FlowConfig::new(1);
        let flow = RealNVP::new(config);

        let metrics = compute_risk_metrics(&flow, 10000);
        assert!(metrics.std > 0.0);
        assert!(metrics.var_99 <= metrics.var_95);
        assert!(metrics.var_95 <= metrics.var_90);
    }
}
