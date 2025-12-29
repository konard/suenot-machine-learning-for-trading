//! Метрики для оценки качества модели

use ndarray::{Array1, Array2};

/// Mean Squared Error (среднеквадратичная ошибка)
pub fn mse(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
    let diff = y_true - y_pred;
    let squared = diff.mapv(|x| x * x);
    squared.mean().unwrap_or(0.0)
}

/// Root Mean Squared Error (корень из MSE)
pub fn rmse(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
    mse(y_true, y_pred).sqrt()
}

/// Mean Absolute Error (средняя абсолютная ошибка)
pub fn mae(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
    let diff = y_true - y_pred;
    let abs_diff = diff.mapv(|x| x.abs());
    abs_diff.mean().unwrap_or(0.0)
}

/// Mean Absolute Percentage Error (средняя абсолютная процентная ошибка)
pub fn mape(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
    let n = y_true.len() as f64;
    let mut sum = 0.0;

    for (t, p) in y_true.iter().zip(y_pred.iter()) {
        if *t != 0.0 {
            sum += ((t - p) / t).abs();
        }
    }

    (sum / n) * 100.0
}

/// Accuracy для классификации (процент правильных предсказаний)
pub fn accuracy(y_true: &Array2<f64>, y_pred: &Array2<f64>, threshold: f64) -> f64 {
    let n = y_true.len() as f64;
    let mut correct = 0.0;

    for (t, p) in y_true.iter().zip(y_pred.iter()) {
        let pred_class = if *p >= threshold { 1.0 } else { 0.0 };
        if (t - &pred_class).abs() < 1e-10 {
            correct += 1.0;
        }
    }

    correct / n
}

/// R² score (коэффициент детерминации)
pub fn r2_score(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
    let mean = y_true.mean().unwrap_or(0.0);

    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();

    let ss_tot: f64 = y_true.iter().map(|t| (t - mean).powi(2)).sum();

    if ss_tot == 0.0 {
        0.0
    } else {
        1.0 - (ss_res / ss_tot)
    }
}

/// Directional accuracy (процент правильно предсказанных направлений)
pub fn directional_accuracy(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    if y_true.len() < 2 {
        return 0.0;
    }

    let n = y_true.len() - 1;
    let mut correct = 0;

    for i in 1..y_true.len() {
        let true_direction = y_true[i] > y_true[i - 1];
        let pred_direction = y_pred[i] > y_pred[i - 1];

        if true_direction == pred_direction {
            correct += 1;
        }
    }

    correct as f64 / n as f64
}

/// Sharpe Ratio для оценки стратегии
pub fn sharpe_ratio(returns: &Array1<f64>, risk_free_rate: f64) -> f64 {
    let mean_return = returns.mean().unwrap_or(0.0);
    let std_return = std_dev(returns);

    if std_return == 0.0 {
        0.0
    } else {
        (mean_return - risk_free_rate) / std_return
    }
}

/// Стандартное отклонение
fn std_dev(arr: &Array1<f64>) -> f64 {
    let n = arr.len() as f64;
    let mean = arr.sum() / n;
    let variance: f64 = arr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
}

/// Maximum Drawdown (максимальная просадка)
pub fn max_drawdown(equity_curve: &Array1<f64>) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mse() {
        let y_true = array![[1.0], [2.0], [3.0]];
        let y_pred = array![[1.1], [2.0], [2.9]];

        let error = mse(&y_true, &y_pred);
        assert!((error - 0.006666666666666667).abs() < 1e-10);
    }

    #[test]
    fn test_accuracy() {
        let y_true = array![[1.0], [0.0], [1.0], [1.0]];
        let y_pred = array![[0.8], [0.2], [0.6], [0.4]];

        let acc = accuracy(&y_true, &y_pred, 0.5);
        assert_eq!(acc, 0.75);
    }

    #[test]
    fn test_r2_score() {
        let y_true = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y_pred = array![[1.1], [2.1], [2.9], [4.0], [5.1]];

        let r2 = r2_score(&y_true, &y_pred);
        assert!(r2 > 0.95); // Должен быть близок к 1
    }

    #[test]
    fn test_max_drawdown() {
        let equity = array![100.0, 110.0, 105.0, 120.0, 90.0, 100.0];
        let dd = max_drawdown(&equity);
        assert!((dd - 0.25).abs() < 1e-10); // (120-90)/120 = 0.25
    }
}
