//! Statistical utility functions.

/// Calculate mean of a slice
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Calculate variance of a slice
pub fn variance(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64
}

/// Calculate standard deviation of a slice
pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Calculate skewness of a slice
pub fn skewness(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-10 {
        return 0.0;
    }
    let n = data.len() as f64;
    let m3 = data.iter().map(|x| (x - m).powi(3)).sum::<f64>() / n;
    m3 / s.powi(3)
}

/// Calculate excess kurtosis of a slice
pub fn kurtosis(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-10 {
        return 0.0;
    }
    let n = data.len() as f64;
    let m4 = data.iter().map(|x| (x - m).powi(4)).sum::<f64>() / n;
    m4 / s.powi(4) - 3.0
}

/// Calculate percentile of a slice
pub fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = (p * sorted.len() as f64) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Calculate multiple percentiles at once
pub fn percentiles(data: &[f64], ps: &[f64]) -> Vec<f64> {
    if data.is_empty() {
        return vec![0.0; ps.len()];
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    ps.iter()
        .map(|&p| {
            let idx = (p * sorted.len() as f64) as usize;
            sorted[idx.min(sorted.len() - 1)]
        })
        .collect()
}

/// Calculate median
pub fn median(data: &[f64]) -> f64 {
    percentile(data, 0.5)
}

/// Calculate inter-quartile range
pub fn iqr(data: &[f64]) -> f64 {
    let q1 = percentile(data, 0.25);
    let q3 = percentile(data, 0.75);
    q3 - q1
}

/// Descriptive statistics summary
#[derive(Debug, Clone)]
pub struct DescriptiveStats {
    pub count: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

/// Calculate descriptive statistics
pub fn describe(data: &[f64]) -> DescriptiveStats {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = data.len();
    let q1_idx = (0.25 * n as f64) as usize;
    let q2_idx = (0.50 * n as f64) as usize;
    let q3_idx = (0.75 * n as f64) as usize;

    DescriptiveStats {
        count: n,
        mean: mean(data),
        std: std_dev(data),
        min: if n > 0 { sorted[0] } else { 0.0 },
        max: if n > 0 { sorted[n - 1] } else { 0.0 },
        q1: if n > 0 { sorted[q1_idx] } else { 0.0 },
        median: if n > 0 { sorted[q2_idx] } else { 0.0 },
        q3: if n > 0 { sorted[q3_idx] } else { 0.0 },
        skewness: skewness(data),
        kurtosis: kurtosis(data),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&data) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_std_dev() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = std_dev(&data);
        assert!((std - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p50 = percentile(&data, 0.5);
        assert!((p50 - 5.0).abs() < 1.0);
    }
}
