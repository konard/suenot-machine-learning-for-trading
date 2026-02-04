//! Performance metrics for backtesting

/// Performance metrics from backtest
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total return (final / initial - 1)
    pub total_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized)
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate (fraction of profitable trades)
    pub win_rate: f64,
    /// Number of trades
    pub num_trades: usize,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Final capital
    pub final_capital: f64,
    /// Initial capital
    pub initial_capital: f64,
}

impl PerformanceMetrics {
    /// Calculate metrics from equity curve
    pub fn from_equity_curve(equity: &[f64], num_trades: usize, win_rate: f64) -> Self {
        if equity.is_empty() {
            return Self::default();
        }

        let initial = equity[0];
        let final_capital = *equity.last().unwrap();
        let total_return = (final_capital / initial) - 1.0;

        // Calculate returns
        let returns: Vec<f64> = equity
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Sharpe ratio (assuming hourly data, annualize)
        let sharpe_ratio = if returns.is_empty() {
            0.0
        } else {
            let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / returns.len() as f64;
            let std = variance.sqrt();

            if std > 1e-10 {
                (mean_return / std) * (252.0 * 24.0_f64).sqrt()
            } else {
                0.0
            }
        };

        // Sortino ratio (only downside volatility)
        let sortino_ratio = {
            let mean_return: f64 = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
            let negative_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

            if negative_returns.is_empty() {
                sharpe_ratio // No downside risk
            } else {
                let downside_variance: f64 = negative_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                    / negative_returns.len() as f64;
                let downside_std = downside_variance.sqrt();

                if downside_std > 1e-10 {
                    (mean_return / downside_std) * (252.0 * 24.0_f64).sqrt()
                } else {
                    0.0
                }
            }
        };

        // Maximum drawdown
        let max_drawdown = {
            let mut peak = equity[0];
            let mut max_dd = 0.0;

            for &value in equity.iter() {
                if value > peak {
                    peak = value;
                }
                let drawdown = (peak - value) / peak;
                if drawdown > max_dd {
                    max_dd = drawdown;
                }
            }
            max_dd
        };

        // Profit factor
        let profit_factor = {
            let gross_profit: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
            let gross_loss: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();

            if gross_loss > 1e-10 {
                gross_profit / gross_loss
            } else if gross_profit > 0.0 {
                f64::INFINITY
            } else {
                1.0
            }
        };

        Self {
            total_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            num_trades,
            profit_factor,
            final_capital,
            initial_capital: initial,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            num_trades: 0,
            profit_factor: 1.0,
            final_capital: 0.0,
            initial_capital: 0.0,
        }
    }
}

impl std::fmt::Display for PerformanceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Performance Metrics")?;
        writeln!(f, "===================")?;
        writeln!(f, "Total Return:    {:>10.2}%", self.total_return * 100.0)?;
        writeln!(f, "Sharpe Ratio:    {:>10.2}", self.sharpe_ratio)?;
        writeln!(f, "Sortino Ratio:   {:>10.2}", self.sortino_ratio)?;
        writeln!(f, "Max Drawdown:    {:>10.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "Win Rate:        {:>10.2}%", self.win_rate * 100.0)?;
        writeln!(f, "Num Trades:      {:>10}", self.num_trades)?;
        writeln!(f, "Profit Factor:   {:>10.2}", self.profit_factor)?;
        writeln!(f, "Final Capital:   ${:>10.2}", self.final_capital)?;
        Ok(())
    }
}

/// Coverage metrics for conformal prediction
#[derive(Debug, Clone)]
pub struct CoverageMetrics {
    /// Empirical coverage rate
    pub coverage: f64,
    /// Target coverage (1 - alpha)
    pub target_coverage: f64,
    /// Coverage gap
    pub coverage_gap: f64,
    /// Mean interval width
    pub mean_width: f64,
    /// Median interval width
    pub median_width: f64,
}

impl CoverageMetrics {
    /// Calculate coverage metrics
    pub fn calculate(
        y_true: &[f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        target_alpha: f64,
    ) -> Self {
        let n = y_true.len();
        if n == 0 {
            return Self::default();
        }

        let mut covered_count = 0;
        let mut widths = Vec::with_capacity(n);

        for i in 0..n {
            let covered = y_true[i] >= lower_bounds[i] && y_true[i] <= upper_bounds[i];
            if covered {
                covered_count += 1;
            }
            widths.push(upper_bounds[i] - lower_bounds[i]);
        }

        let coverage = covered_count as f64 / n as f64;
        let target_coverage = 1.0 - target_alpha;
        let coverage_gap = (coverage - target_coverage).abs();
        let mean_width = widths.iter().sum::<f64>() / n as f64;

        // Median width
        widths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_width = if n % 2 == 0 {
            (widths[n / 2 - 1] + widths[n / 2]) / 2.0
        } else {
            widths[n / 2]
        };

        Self {
            coverage,
            target_coverage,
            coverage_gap,
            mean_width,
            median_width,
        }
    }
}

impl Default for CoverageMetrics {
    fn default() -> Self {
        Self {
            coverage: 0.0,
            target_coverage: 0.9,
            coverage_gap: 0.9,
            mean_width: 0.0,
            median_width: 0.0,
        }
    }
}

impl std::fmt::Display for CoverageMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Coverage Metrics")?;
        writeln!(f, "================")?;
        writeln!(f, "Coverage:        {:>10.2}%", self.coverage * 100.0)?;
        writeln!(f, "Target:          {:>10.2}%", self.target_coverage * 100.0)?;
        writeln!(f, "Gap:             {:>10.2}%", self.coverage_gap * 100.0)?;
        writeln!(f, "Mean Width:      {:>10.6}", self.mean_width)?;
        writeln!(f, "Median Width:    {:>10.6}", self.median_width)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_calculation() {
        let equity: Vec<f64> = (0..100)
            .map(|i| 10000.0 * (1.0 + 0.001 * i as f64))
            .collect();

        let metrics = PerformanceMetrics::from_equity_curve(&equity, 50, 0.6);

        assert!(metrics.total_return > 0.0);
        assert!(metrics.max_drawdown >= 0.0);
        assert!(metrics.max_drawdown <= 1.0);
    }

    #[test]
    fn test_coverage_metrics() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let lower = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let upper = vec![1.5, 2.5, 3.5, 4.5, 5.5];

        let metrics = CoverageMetrics::calculate(&y_true, &lower, &upper, 0.1);

        assert!((metrics.coverage - 1.0).abs() < 1e-10);
        assert!((metrics.mean_width - 1.0).abs() < 1e-10);
    }
}
