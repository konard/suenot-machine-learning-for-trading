//! Performance metrics for evaluation

use serde::{Deserialize, Serialize};

/// Metrics calculator for model and trading evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    /// True labels
    pub y_true: Vec<u8>,
    /// Predicted labels
    pub y_pred: Vec<u8>,
    /// Number of classes
    pub num_classes: usize,
}

impl Metrics {
    /// Create new metrics from predictions
    pub fn new(y_true: Vec<u8>, y_pred: Vec<u8>, num_classes: usize) -> Self {
        Self {
            y_true,
            y_pred,
            num_classes,
        }
    }

    /// Calculate accuracy
    pub fn accuracy(&self) -> f32 {
        if self.y_true.is_empty() {
            return 0.0;
        }

        let correct = self
            .y_true
            .iter()
            .zip(&self.y_pred)
            .filter(|(t, p)| t == p)
            .count();

        correct as f32 / self.y_true.len() as f32
    }

    /// Calculate confusion matrix
    /// Returns matrix[true_label][pred_label]
    pub fn confusion_matrix(&self) -> Vec<Vec<usize>> {
        let mut matrix = vec![vec![0usize; self.num_classes]; self.num_classes];

        for (&t, &p) in self.y_true.iter().zip(&self.y_pred) {
            let t = t as usize;
            let p = p as usize;
            if t < self.num_classes && p < self.num_classes {
                matrix[t][p] += 1;
            }
        }

        matrix
    }

    /// Calculate precision for each class
    pub fn precision(&self) -> Vec<f32> {
        let cm = self.confusion_matrix();
        let mut precision = vec![0.0f32; self.num_classes];

        for c in 0..self.num_classes {
            let tp = cm[c][c] as f32;
            let fp: f32 = (0..self.num_classes)
                .filter(|&i| i != c)
                .map(|i| cm[i][c] as f32)
                .sum();

            precision[c] = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        }

        precision
    }

    /// Calculate recall for each class
    pub fn recall(&self) -> Vec<f32> {
        let cm = self.confusion_matrix();
        let mut recall = vec![0.0f32; self.num_classes];

        for c in 0..self.num_classes {
            let tp = cm[c][c] as f32;
            let fn_: f32 = (0..self.num_classes)
                .filter(|&i| i != c)
                .map(|i| cm[c][i] as f32)
                .sum();

            recall[c] = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        }

        recall
    }

    /// Calculate F1 score for each class
    pub fn f1_score(&self) -> Vec<f32> {
        let precision = self.precision();
        let recall = self.recall();

        precision
            .iter()
            .zip(&recall)
            .map(|(&p, &r)| {
                if p + r > 0.0 {
                    2.0 * p * r / (p + r)
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Calculate macro-averaged F1 score
    pub fn macro_f1(&self) -> f32 {
        let f1 = self.f1_score();
        f1.iter().sum::<f32>() / self.num_classes as f32
    }

    /// Calculate weighted F1 score
    pub fn weighted_f1(&self) -> f32 {
        let f1 = self.f1_score();
        let mut class_counts = vec![0usize; self.num_classes];

        for &t in &self.y_true {
            let t = t as usize;
            if t < self.num_classes {
                class_counts[t] += 1;
            }
        }

        let total = class_counts.iter().sum::<usize>() as f32;
        if total == 0.0 {
            return 0.0;
        }

        f1.iter()
            .zip(&class_counts)
            .map(|(&f, &c)| f * c as f32)
            .sum::<f32>()
            / total
    }

    /// Print classification report
    pub fn classification_report(&self) -> String {
        let precision = self.precision();
        let recall = self.recall();
        let f1 = self.f1_score();

        let mut class_counts = vec![0usize; self.num_classes];
        for &t in &self.y_true {
            let t = t as usize;
            if t < self.num_classes {
                class_counts[t] += 1;
            }
        }

        let class_names = ["Down", "Neutral", "Up"];

        let mut report = String::new();
        report.push_str(&format!(
            "{:>10} {:>10} {:>10} {:>10} {:>10}\n",
            "", "precision", "recall", "f1-score", "support"
        ));
        report.push_str(&format!("{}\n", "-".repeat(52)));

        for c in 0..self.num_classes {
            let name = if c < class_names.len() {
                class_names[c]
            } else {
                "Unknown"
            };
            report.push_str(&format!(
                "{:>10} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
                name, precision[c], recall[c], f1[c], class_counts[c]
            ));
        }

        report.push_str(&format!("{}\n", "-".repeat(52)));
        report.push_str(&format!(
            "{:>10} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
            "accuracy",
            "",
            "",
            self.accuracy(),
            self.y_true.len()
        ));
        report.push_str(&format!(
            "{:>10} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
            "macro avg",
            precision.iter().sum::<f32>() / self.num_classes as f32,
            recall.iter().sum::<f32>() / self.num_classes as f32,
            self.macro_f1(),
            self.y_true.len()
        ));

        report
    }
}

/// Trading performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingMetrics {
    /// List of returns
    pub returns: Vec<f32>,
    /// Risk-free rate (annualized)
    pub risk_free_rate: f32,
    /// Trading periods per year
    pub periods_per_year: f32,
}

impl TradingMetrics {
    /// Create new trading metrics
    pub fn new(returns: Vec<f32>, risk_free_rate: f32, periods_per_year: f32) -> Self {
        Self {
            returns,
            risk_free_rate,
            periods_per_year,
        }
    }

    /// Create for minute-level crypto trading
    pub fn from_minute_returns(returns: Vec<f32>) -> Self {
        Self::new(returns, 0.0, 525600.0) // 60 * 24 * 365
    }

    /// Calculate mean return
    pub fn mean_return(&self) -> f32 {
        if self.returns.is_empty() {
            return 0.0;
        }
        self.returns.iter().sum::<f32>() / self.returns.len() as f32
    }

    /// Calculate return standard deviation
    pub fn std_return(&self) -> f32 {
        if self.returns.len() < 2 {
            return 0.0;
        }

        let mean = self.mean_return();
        let variance = self
            .returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f32>()
            / (self.returns.len() - 1) as f32;

        variance.sqrt()
    }

    /// Calculate Sharpe ratio
    pub fn sharpe_ratio(&self) -> f32 {
        let mean = self.mean_return();
        let std = self.std_return();

        if std == 0.0 {
            return 0.0;
        }

        let rf_per_period = self.risk_free_rate / self.periods_per_year;
        let excess_return = mean - rf_per_period;

        (excess_return / std) * self.periods_per_year.sqrt()
    }

    /// Calculate Sortino ratio (uses downside deviation)
    pub fn sortino_ratio(&self) -> f32 {
        let mean = self.mean_return();
        let rf_per_period = self.risk_free_rate / self.periods_per_year;

        // Downside deviation
        let negative_returns: Vec<f32> = self
            .returns
            .iter()
            .filter(|&&r| r < rf_per_period)
            .map(|&r| (r - rf_per_period).powi(2))
            .collect();

        if negative_returns.is_empty() {
            return f32::INFINITY;
        }

        let downside_variance =
            negative_returns.iter().sum::<f32>() / negative_returns.len() as f32;
        let downside_std = downside_variance.sqrt();

        if downside_std == 0.0 {
            return f32::INFINITY;
        }

        let excess_return = mean - rf_per_period;
        (excess_return / downside_std) * self.periods_per_year.sqrt()
    }

    /// Calculate maximum drawdown
    pub fn max_drawdown(&self) -> f32 {
        if self.returns.is_empty() {
            return 0.0;
        }

        let mut cumulative = Vec::with_capacity(self.returns.len());
        let mut running = 1.0f32;

        for &r in &self.returns {
            running *= 1.0 + r;
            cumulative.push(running);
        }

        let mut max_dd = 0.0f32;
        let mut peak = cumulative[0];

        for &val in &cumulative {
            if val > peak {
                peak = val;
            }
            let dd = (peak - val) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        max_dd
    }

    /// Calculate Calmar ratio (annual return / max drawdown)
    pub fn calmar_ratio(&self) -> f32 {
        let max_dd = self.max_drawdown();
        if max_dd == 0.0 {
            return f32::INFINITY;
        }

        let annualized_return = self.mean_return() * self.periods_per_year;
        annualized_return / max_dd
    }

    /// Calculate cumulative return
    pub fn cumulative_return(&self) -> f32 {
        let mut result = 1.0f32;
        for &r in &self.returns {
            result *= 1.0 + r;
        }
        result - 1.0
    }

    /// Calculate annualized return
    pub fn annualized_return(&self) -> f32 {
        let cum_return = self.cumulative_return();
        let years = self.returns.len() as f32 / self.periods_per_year;

        if years == 0.0 {
            return 0.0;
        }

        (1.0 + cum_return).powf(1.0 / years) - 1.0
    }

    /// Calculate win rate
    pub fn win_rate(&self) -> f32 {
        if self.returns.is_empty() {
            return 0.0;
        }

        let wins = self.returns.iter().filter(|&&r| r > 0.0).count();
        wins as f32 / self.returns.len() as f32
    }

    /// Calculate profit factor
    pub fn profit_factor(&self) -> f32 {
        let gross_profit: f32 = self.returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_loss: f32 = self.returns.iter().filter(|&&r| r < 0.0).map(|r| -r).sum();

        if gross_loss == 0.0 {
            return if gross_profit > 0.0 {
                f32::INFINITY
            } else {
                0.0
            };
        }

        gross_profit / gross_loss
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        format!(
            "Trading Performance Summary\n\
             ============================\n\
             Cumulative Return: {:.2}%\n\
             Annualized Return: {:.2}%\n\
             Sharpe Ratio:      {:.4}\n\
             Sortino Ratio:     {:.4}\n\
             Max Drawdown:      {:.2}%\n\
             Calmar Ratio:      {:.4}\n\
             Win Rate:          {:.2}%\n\
             Profit Factor:     {:.4}\n\
             Num Trades:        {}",
            self.cumulative_return() * 100.0,
            self.annualized_return() * 100.0,
            self.sharpe_ratio(),
            self.sortino_ratio(),
            self.max_drawdown() * 100.0,
            self.calmar_ratio(),
            self.win_rate() * 100.0,
            self.profit_factor(),
            self.returns.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy() {
        let metrics = Metrics::new(
            vec![0, 1, 2, 0, 1, 2],
            vec![0, 1, 2, 0, 2, 2], // 5 correct, 1 wrong
            3,
        );

        let acc = metrics.accuracy();
        assert!((acc - 5.0 / 6.0).abs() < 0.001);
    }

    #[test]
    fn test_confusion_matrix() {
        let metrics = Metrics::new(vec![0, 0, 1, 1, 2, 2], vec![0, 1, 1, 0, 2, 2], 3);

        let cm = metrics.confusion_matrix();
        assert_eq!(cm[0][0], 1); // True 0, Pred 0
        assert_eq!(cm[0][1], 1); // True 0, Pred 1
        assert_eq!(cm[1][0], 1); // True 1, Pred 0
        assert_eq!(cm[1][1], 1); // True 1, Pred 1
        assert_eq!(cm[2][2], 2); // True 2, Pred 2
    }

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005, -0.005];
        let metrics = TradingMetrics::new(returns, 0.0, 252.0);

        let sharpe = metrics.sharpe_ratio();
        assert!(sharpe > 0.0); // Should be positive with positive returns
    }

    #[test]
    fn test_max_drawdown() {
        let returns = vec![0.1, -0.15, 0.05, -0.1]; // Creates a drawdown
        let metrics = TradingMetrics::new(returns, 0.0, 252.0);

        let max_dd = metrics.max_drawdown();
        assert!(max_dd > 0.0);
        assert!(max_dd < 1.0);
    }

    #[test]
    fn test_win_rate() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, -0.005, 0.01];
        let metrics = TradingMetrics::new(returns, 0.0, 252.0);

        let wr = metrics.win_rate();
        assert!((wr - 4.0 / 6.0).abs() < 0.001);
    }
}
