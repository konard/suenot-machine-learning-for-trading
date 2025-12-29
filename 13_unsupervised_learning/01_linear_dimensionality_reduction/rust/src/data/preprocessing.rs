//! Data preprocessing utilities

use ndarray::{Array1, Array2, Axis};

/// Standardize data (zero mean, unit variance)
pub fn standardize(data: &Array2<f64>) -> Array2<f64> {
    let mean = data.mean_axis(Axis(0)).unwrap();
    let std = data.std_axis(Axis(0), 0.0);

    let mut result = data.clone();
    let (n_rows, n_cols) = data.dim();

    for j in 0..n_cols {
        if std[j] > 1e-10 {
            for i in 0..n_rows {
                result[[i, j]] = (result[[i, j]] - mean[j]) / std[j];
            }
        }
    }

    result
}

/// Center data (zero mean)
pub fn center(data: &Array2<f64>) -> Array2<f64> {
    let mean = data.mean_axis(Axis(0)).unwrap();
    data - &mean
}

/// Calculate quantile of a vector
pub fn quantile(data: &[f64], q: f64) -> f64 {
    let mut sorted: Vec<f64> = data.iter().filter(|x| x.is_finite()).copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if sorted.is_empty() {
        return f64::NAN;
    }

    let idx = ((sorted.len() - 1) as f64 * q) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Winsorize data at given quantiles
pub fn winsorize(data: &mut Array2<f64>, lower_q: f64, upper_q: f64) {
    let (n_rows, n_cols) = data.dim();

    for j in 0..n_cols {
        let col: Vec<f64> = data.column(j).to_vec();
        let lower_bound = quantile(&col, lower_q);
        let upper_bound = quantile(&col, upper_q);

        for i in 0..n_rows {
            let val = data[[i, j]];
            if val.is_finite() {
                if val < lower_bound {
                    data[[i, j]] = lower_bound;
                } else if val > upper_bound {
                    data[[i, j]] = upper_bound;
                }
            }
        }
    }
}

/// Fill missing values with column mean
pub fn fill_na_with_mean(data: &mut Array2<f64>) {
    let (n_rows, n_cols) = data.dim();

    for j in 0..n_cols {
        let col: Vec<f64> = data.column(j).to_vec();
        let valid_values: Vec<f64> = col.iter().filter(|x| x.is_finite()).copied().collect();

        if valid_values.is_empty() {
            continue;
        }

        let mean: f64 = valid_values.iter().sum::<f64>() / valid_values.len() as f64;

        for i in 0..n_rows {
            if !data[[i, j]].is_finite() {
                data[[i, j]] = mean;
            }
        }
    }
}

/// Fill missing values with row mean
pub fn fill_na_with_row_mean(data: &mut Array2<f64>) {
    let (n_rows, n_cols) = data.dim();

    for i in 0..n_rows {
        let row: Vec<f64> = data.row(i).to_vec();
        let valid_values: Vec<f64> = row.iter().filter(|x| x.is_finite()).copied().collect();

        if valid_values.is_empty() {
            continue;
        }

        let mean: f64 = valid_values.iter().sum::<f64>() / valid_values.len() as f64;

        for j in 0..n_cols {
            if !data[[i, j]].is_finite() {
                data[[i, j]] = mean;
            }
        }
    }
}

/// Calculate rolling window statistics
pub fn rolling_mean(data: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::zeros(n);

    for i in 0..n {
        let start = if i >= window { i - window + 1 } else { 0 };
        let window_data = data.slice(ndarray::s![start..=i]);
        result[i] = window_data.mean().unwrap_or(f64::NAN);
    }

    result
}

/// Calculate rolling standard deviation
pub fn rolling_std(data: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = data.len();
    let mut result = Array1::zeros(n);

    for i in 0..n {
        let start = if i >= window { i - window + 1 } else { 0 };
        let window_data = data.slice(ndarray::s![start..=i]);
        result[i] = window_data.std(0.0);
    }

    result
}

/// Calculate Euclidean distance between two vectors
pub fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    (a - b).mapv(|x| x * x).sum().sqrt()
}

/// Calculate pairwise distances between all rows
pub fn pairwise_distances(data: &Array2<f64>) -> Array2<f64> {
    let n = data.nrows();
    let mut distances = Array2::zeros((n, n));

    for i in 0..n {
        for j in i..n {
            let dist = euclidean_distance(&data.row(i).to_owned(), &data.row(j).to_owned());
            distances[[i, j]] = dist;
            distances[[j, i]] = dist;
        }
    }

    distances
}

/// Calculate mean pairwise distance
pub fn mean_pairwise_distance(data: &Array2<f64>) -> f64 {
    let distances = pairwise_distances(data);
    let n = data.nrows();

    if n <= 1 {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut count = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            sum += distances[[i, j]];
            count += 1;
        }
    }

    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

/// Calculate minimum pairwise distance for each point
pub fn min_pairwise_distances(data: &Array2<f64>) -> Array1<f64> {
    let distances = pairwise_distances(data);
    let n = data.nrows();
    let mut min_distances = Array1::zeros(n);

    for i in 0..n {
        let mut min_dist = f64::INFINITY;
        for j in 0..n {
            if i != j && distances[[i, j]] < min_dist {
                min_dist = distances[[i, j]];
            }
        }
        min_distances[i] = if min_dist.is_infinite() { 0.0 } else { min_dist };
    }

    min_distances
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_standardize() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let std_data = standardize(&data);

        // Mean should be ~0 for each column
        let mean = std_data.mean_axis(Axis(0)).unwrap();
        assert!(mean[0].abs() < 1e-10);
        assert!(mean[1].abs() < 1e-10);
    }

    #[test]
    fn test_quantile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile(&data, 0.0), 1.0);
        assert_eq!(quantile(&data, 1.0), 5.0);
        assert_eq!(quantile(&data, 0.5), 3.0);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_distances() {
        let data = array![[0.0, 0.0], [3.0, 4.0]];
        let distances = pairwise_distances(&data);

        assert!((distances[[0, 1]] - 5.0).abs() < 1e-10);
        assert!((distances[[1, 0]] - 5.0).abs() < 1e-10);
        assert_eq!(distances[[0, 0]], 0.0);
        assert_eq!(distances[[1, 1]], 0.0);
    }
}
