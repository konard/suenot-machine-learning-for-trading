//! Statistical utility functions

use ndarray::{Array1, Array2, Axis};

/// Calculate mean of an array
pub fn mean(data: &Array1<f64>) -> f64 {
    data.mean().unwrap_or(0.0)
}

/// Calculate variance of an array
pub fn variance(data: &Array1<f64>, ddof: usize) -> f64 {
    let n = data.len();
    if n <= ddof {
        return 0.0;
    }

    let mean = mean(data);
    let sum_sq: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();
    sum_sq / (n - ddof) as f64
}

/// Calculate standard deviation of an array
pub fn std_dev(data: &Array1<f64>, ddof: usize) -> f64 {
    variance(data, ddof).sqrt()
}

/// Calculate skewness
pub fn skewness(data: &Array1<f64>) -> f64 {
    let n = data.len() as f64;
    if n < 3.0 {
        return 0.0;
    }

    let m = mean(data);
    let std = std_dev(data, 1);

    if std < 1e-10 {
        return 0.0;
    }

    let sum_cubed: f64 = data.iter().map(|&x| ((x - m) / std).powi(3)).sum();
    (n / ((n - 1.0) * (n - 2.0))) * sum_cubed
}

/// Calculate excess kurtosis
pub fn kurtosis(data: &Array1<f64>) -> f64 {
    let n = data.len() as f64;
    if n < 4.0 {
        return 0.0;
    }

    let m = mean(data);
    let std = std_dev(data, 1);

    if std < 1e-10 {
        return 0.0;
    }

    let sum_fourth: f64 = data.iter().map(|&x| ((x - m) / std).powi(4)).sum();
    let term1 = (n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0)) * sum_fourth;
    let term2 = (3.0 * (n - 1.0).powi(2)) / ((n - 2.0) * (n - 3.0));

    term1 - term2
}

/// Calculate percentile
pub fn percentile(data: &Array1<f64>, p: f64) -> f64 {
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if sorted.is_empty() {
        return f64::NAN;
    }

    let idx = (p / 100.0 * (sorted.len() - 1) as f64) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Calculate median
pub fn median(data: &Array1<f64>) -> f64 {
    percentile(data, 50.0)
}

/// Calculate interquartile range
pub fn iqr(data: &Array1<f64>) -> f64 {
    percentile(data, 75.0) - percentile(data, 25.0)
}

/// Calculate correlation between two arrays
pub fn correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let mean_x = mean(x);
    let mean_y = mean(y);

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x > 1e-10 && var_y > 1e-10 {
        cov / (var_x.sqrt() * var_y.sqrt())
    } else {
        0.0
    }
}

/// Calculate autocorrelation at given lag
pub fn autocorrelation(data: &Array1<f64>, lag: usize) -> f64 {
    if lag >= data.len() {
        return 0.0;
    }

    let n = data.len() - lag;
    let x = data.slice(ndarray::s![..n]).to_owned();
    let y = data.slice(ndarray::s![lag..]).to_owned();

    correlation(&x, &y)
}

/// Ljung-Box test statistic for autocorrelation
pub fn ljung_box(data: &Array1<f64>, lags: usize) -> f64 {
    let n = data.len() as f64;
    let mut q = 0.0;

    for k in 1..=lags {
        let rho = autocorrelation(data, k);
        q += (rho * rho) / (n - k as f64);
    }

    n * (n + 2.0) * q
}

/// Jarque-Bera test statistic for normality
pub fn jarque_bera(data: &Array1<f64>) -> f64 {
    let n = data.len() as f64;
    let s = skewness(data);
    let k = kurtosis(data);

    (n / 6.0) * (s.powi(2) + (k.powi(2) / 4.0))
}

/// Summary statistics for a data series
#[derive(Debug, Clone)]
pub struct SummaryStats {
    pub count: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub q25: f64,
    pub median: f64,
    pub q75: f64,
    pub max: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

impl SummaryStats {
    /// Calculate summary statistics
    pub fn from_data(data: &Array1<f64>) -> Self {
        let mut sorted: Vec<f64> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Self {
            count: data.len(),
            mean: mean(data),
            std: std_dev(data, 1),
            min: *sorted.first().unwrap_or(&f64::NAN),
            q25: percentile(data, 25.0),
            median: percentile(data, 50.0),
            q75: percentile(data, 75.0),
            max: *sorted.last().unwrap_or(&f64::NAN),
            skewness: skewness(data),
            kurtosis: kurtosis(data),
        }
    }

    /// Print summary
    pub fn print(&self) {
        println!("Count:    {}", self.count);
        println!("Mean:     {:.6}", self.mean);
        println!("Std:      {:.6}", self.std);
        println!("Min:      {:.6}", self.min);
        println!("25%:      {:.6}", self.q25);
        println!("50%:      {:.6}", self.median);
        println!("75%:      {:.6}", self.q75);
        println!("Max:      {:.6}", self.max);
        println!("Skewness: {:.4}", self.skewness);
        println!("Kurtosis: {:.4}", self.kurtosis);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mean() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&data) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_std_dev() {
        let data = array![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = std_dev(&data, 0);
        assert!((std - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_percentile() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&data, 0.0), 1.0);
        assert_eq!(percentile(&data, 50.0), 3.0);
        assert_eq!(percentile(&data, 100.0), 5.0);
    }

    #[test]
    fn test_correlation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((correlation(&x, &y) - 1.0).abs() < 1e-10);

        let y_neg = array![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((correlation(&x, &y_neg) - (-1.0)).abs() < 1e-10);
    }
}
