//! Performance and prediction metrics

use serde::{Deserialize, Serialize};

/// Trading performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annual_return: f64,
    /// Monthly returns
    pub monthly_returns: Vec<f64>,
    /// Volatility (annualized)
    pub volatility: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Value at Risk (95%)
    pub value_at_risk: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Average trade return
    pub avg_trade: f64,
    /// Number of trades
    pub num_trades: usize,
}

impl PerformanceMetrics {
    /// Calculate metrics from returns
    pub fn from_returns(returns: &[f64], risk_free_rate: f64) -> Self {
        let n = returns.len();
        if n == 0 {
            return Self::default();
        }

        // Total return
        let total_return = returns.iter().fold(1.0, |acc, r| acc * (1.0 + r)) - 1.0;

        // Mean return (annualized)
        let mean_return = returns.iter().sum::<f64>() / n as f64;
        let annual_return = (1.0 + mean_return).powf(252.0) - 1.0;

        // Volatility
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / n as f64;
        let volatility = variance.sqrt() * (252_f64).sqrt();

        // Maximum drawdown
        let max_drawdown = Self::calculate_max_drawdown(returns);

        // VaR (5th percentile)
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let var_index = (n as f64 * 0.05) as usize;
        let value_at_risk = -sorted_returns.get(var_index).unwrap_or(&0.0);

        // Sharpe ratio
        let excess_return = annual_return - risk_free_rate;
        let sharpe_ratio = if volatility > 0.0 { excess_return / volatility } else { 0.0 };

        // Sortino ratio
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_variance = if !downside_returns.is_empty() {
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64
        } else {
            0.0
        };
        let downside_deviation = downside_variance.sqrt() * (252_f64).sqrt();
        let sortino_ratio = if downside_deviation > 0.0 { excess_return / downside_deviation } else { 0.0 };

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 0.0 { annual_return / max_drawdown } else { 0.0 };

        // Win rate
        let positive_returns = returns.iter().filter(|&&r| r > 0.0).count();
        let win_rate = positive_returns as f64 / n as f64;

        // Profit factor
        let gross_profit: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_loss: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
        let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else { f64::INFINITY };

        // Average trade
        let avg_trade = mean_return;

        Self {
            total_return,
            annual_return,
            monthly_returns: Vec::new(),
            volatility,
            max_drawdown,
            value_at_risk,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            win_rate,
            profit_factor,
            avg_trade,
            num_trades: n,
        }
    }

    fn calculate_max_drawdown(returns: &[f64]) -> f64 {
        let mut peak = 1.0;
        let mut max_dd = 0.0;
        let mut equity = 1.0;

        for &ret in returns {
            equity *= 1.0 + ret;
            peak = peak.max(equity);
            let dd = (peak - equity) / peak;
            max_dd = max_dd.max(dd);
        }

        max_dd
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("=== Performance Metrics ===");
        println!("Total Return:     {:.2}%", self.total_return * 100.0);
        println!("Annual Return:    {:.2}%", self.annual_return * 100.0);
        println!("Volatility:       {:.2}%", self.volatility * 100.0);
        println!("Max Drawdown:     {:.2}%", self.max_drawdown * 100.0);
        println!("VaR (95%):        {:.2}%", self.value_at_risk * 100.0);
        println!("Sharpe Ratio:     {:.3}", self.sharpe_ratio);
        println!("Sortino Ratio:    {:.3}", self.sortino_ratio);
        println!("Calmar Ratio:     {:.3}", self.calmar_ratio);
        println!("Win Rate:         {:.2}%", self.win_rate * 100.0);
        println!("Profit Factor:    {:.2}", self.profit_factor);
    }
}

/// Prediction quality metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictionMetrics {
    /// Mean Squared Error
    pub mse: f64,
    /// Mean Absolute Error
    pub mae: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Directional accuracy (% correct direction)
    pub directional_accuracy: f64,
    /// R-squared (coefficient of determination)
    pub r_squared: f64,
    /// Mean Absolute Percentage Error
    pub mape: f64,
    /// Symmetric MAPE
    pub smape: f64,
}

impl PredictionMetrics {
    /// Calculate metrics from predictions and actuals
    pub fn calculate(predictions: &[f64], actuals: &[f64]) -> Self {
        let n = predictions.len().min(actuals.len());
        if n == 0 {
            return Self::default();
        }

        // MSE and MAE
        let mut se_sum = 0.0;
        let mut ae_sum = 0.0;
        let mut ape_sum = 0.0;
        let mut smape_sum = 0.0;

        for i in 0..n {
            let error = predictions[i] - actuals[i];
            se_sum += error * error;
            ae_sum += error.abs();

            if actuals[i].abs() > 1e-10 {
                ape_sum += (error / actuals[i]).abs();
            }

            let denom = predictions[i].abs() + actuals[i].abs();
            if denom > 1e-10 {
                smape_sum += 2.0 * error.abs() / denom;
            }
        }

        let mse = se_sum / n as f64;
        let mae = ae_sum / n as f64;
        let rmse = mse.sqrt();
        let mape = ape_sum / n as f64;
        let smape = smape_sum / n as f64;

        // Directional accuracy
        let mut correct_direction = 0;
        for i in 1..n {
            let pred_direction = predictions[i] - predictions[i - 1];
            let actual_direction = actuals[i] - actuals[i - 1];
            if pred_direction * actual_direction > 0.0 {
                correct_direction += 1;
            }
        }
        let directional_accuracy = if n > 1 {
            correct_direction as f64 / (n - 1) as f64
        } else {
            0.0
        };

        // R-squared
        let actual_mean: f64 = actuals.iter().sum::<f64>() / n as f64;
        let ss_tot: f64 = actuals.iter().map(|a| (a - actual_mean).powi(2)).sum();
        let ss_res: f64 = (0..n).map(|i| (actuals[i] - predictions[i]).powi(2)).sum();
        let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };

        Self {
            mse,
            mae,
            rmse,
            directional_accuracy,
            r_squared,
            mape,
            smape,
        }
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("=== Prediction Metrics ===");
        println!("MSE:                  {:.6}", self.mse);
        println!("MAE:                  {:.6}", self.mae);
        println!("RMSE:                 {:.6}", self.rmse);
        println!("Directional Accuracy: {:.2}%", self.directional_accuracy * 100.0);
        println!("R-squared:            {:.4}", self.r_squared);
        println!("MAPE:                 {:.2}%", self.mape * 100.0);
        println!("SMAPE:                {:.2}%", self.smape * 100.0);
    }
}

/// Calculate Information Coefficient (IC)
pub fn information_coefficient(predictions: &[f64], actuals: &[f64]) -> f64 {
    let n = predictions.len().min(actuals.len());
    if n < 2 {
        return 0.0;
    }

    let pred_mean: f64 = predictions.iter().take(n).sum::<f64>() / n as f64;
    let actual_mean: f64 = actuals.iter().take(n).sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut pred_var = 0.0;
    let mut actual_var = 0.0;

    for i in 0..n {
        let pred_diff = predictions[i] - pred_mean;
        let actual_diff = actuals[i] - actual_mean;
        cov += pred_diff * actual_diff;
        pred_var += pred_diff * pred_diff;
        actual_var += actual_diff * actual_diff;
    }

    let denom = (pred_var * actual_var).sqrt();
    if denom > 0.0 {
        cov / denom
    } else {
        0.0
    }
}

/// Calculate hit rate (directional accuracy)
pub fn hit_rate(predictions: &[f64], actuals: &[f64]) -> f64 {
    let n = predictions.len().min(actuals.len());
    if n < 2 {
        return 0.0;
    }

    let mut hits = 0;
    for i in 1..n {
        let pred_dir = (predictions[i] - predictions[i - 1]).signum();
        let actual_dir = (actuals[i] - actuals[i - 1]).signum();
        if pred_dir == actual_dir {
            hits += 1;
        }
    }

    hits as f64 / (n - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics() {
        let returns = vec![0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.01];
        let metrics = PerformanceMetrics::from_returns(&returns, 0.0);

        assert!(metrics.total_return > 0.0);
        assert!(metrics.win_rate > 0.0 && metrics.win_rate < 1.0);
    }

    #[test]
    fn test_prediction_metrics() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let actuals = vec![1.1, 1.9, 3.1, 3.9, 5.1];

        let metrics = PredictionMetrics::calculate(&predictions, &actuals);

        assert!(metrics.mse < 0.1);
        assert!(metrics.r_squared > 0.9);
    }

    #[test]
    fn test_information_coefficient() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let actuals = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let ic = information_coefficient(&predictions, &actuals);
        assert!((ic - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_max_drawdown() {
        let returns = vec![0.1, -0.2, 0.05, -0.1, 0.15];
        let metrics = PerformanceMetrics::from_returns(&returns, 0.0);

        assert!(metrics.max_drawdown > 0.0);
        assert!(metrics.max_drawdown < 1.0);
    }
}
