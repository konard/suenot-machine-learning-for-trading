//! Loss Functions
//!
//! Quantile Loss и другие функции потерь для TFT.

use ndarray::{Array1, Array2};

/// Результат квантильного прогноза
#[derive(Debug, Clone)]
pub struct QuantilePrediction {
    /// Квантили
    pub quantiles: Vec<f64>,

    /// Предсказанные значения для каждого квантиля
    /// Shape: (prediction_length, num_quantiles)
    pub values: Array2<f64>,
}

impl QuantilePrediction {
    /// Создает новый прогноз
    pub fn new(quantiles: Vec<f64>, values: Array2<f64>) -> Self {
        Self { quantiles, values }
    }

    /// Возвращает медианный прогноз (q50)
    pub fn median(&self) -> Array1<f64> {
        // Ищем квантиль 0.5
        if let Some(idx) = self.quantiles.iter().position(|&q| (q - 0.5).abs() < 1e-6) {
            self.values.column(idx).to_owned()
        } else {
            // Если нет точно 0.5, берем средний квантиль
            let mid_idx = self.quantiles.len() / 2;
            self.values.column(mid_idx).to_owned()
        }
    }

    /// Возвращает нижний квантиль (например, q10)
    pub fn lower(&self) -> Array1<f64> {
        self.values.column(0).to_owned()
    }

    /// Возвращает верхний квантиль (например, q90)
    pub fn upper(&self) -> Array1<f64> {
        self.values.column(self.quantiles.len() - 1).to_owned()
    }

    /// Возвращает ширину prediction interval
    pub fn interval_width(&self) -> Array1<f64> {
        &self.upper() - &self.lower()
    }

    /// Проверяет, попадает ли фактическое значение в интервал
    pub fn coverage(&self, actual: &Array1<f64>) -> f64 {
        let lower = self.lower();
        let upper = self.upper();

        let mut covered = 0;
        for i in 0..actual.len() {
            if actual[i] >= lower[i] && actual[i] <= upper[i] {
                covered += 1;
            }
        }

        covered as f64 / actual.len() as f64
    }
}

/// Quantile Loss Function
///
/// L_q(y, ŷ) = max(q(y - ŷ), (q-1)(y - ŷ))
#[derive(Debug, Clone)]
pub struct QuantileLoss {
    /// Квантили для которых считаем loss
    pub quantiles: Vec<f64>,
}

impl Default for QuantileLoss {
    fn default() -> Self {
        Self {
            quantiles: vec![0.1, 0.5, 0.9],
        }
    }
}

impl QuantileLoss {
    /// Создает с дефолтными квантилями [0.1, 0.5, 0.9]
    pub fn new() -> Self {
        Self::default()
    }

    /// Создает с кастомными квантилями
    pub fn with_quantiles(quantiles: Vec<f64>) -> Self {
        Self { quantiles }
    }

    /// Вычисляет quantile loss для одного квантиля
    fn pinball_loss(y_true: f64, y_pred: f64, quantile: f64) -> f64 {
        let error = y_true - y_pred;
        if error >= 0.0 {
            quantile * error
        } else {
            (quantile - 1.0) * error
        }
    }

    /// Вычисляет loss для одного sample
    pub fn loss_single(
        &self,
        y_true: &Array1<f64>,
        predictions: &Array2<f64>,
    ) -> f64 {
        let mut total_loss = 0.0;
        let n = y_true.len();

        for (q_idx, &quantile) in self.quantiles.iter().enumerate() {
            for i in 0..n {
                total_loss += Self::pinball_loss(y_true[i], predictions[[i, q_idx]], quantile);
            }
        }

        total_loss / (n * self.quantiles.len()) as f64
    }

    /// Вычисляет средний loss для batch
    pub fn loss_batch(
        &self,
        y_true: &[Array1<f64>],
        predictions: &[Array2<f64>],
    ) -> f64 {
        if y_true.is_empty() {
            return 0.0;
        }

        let total: f64 = y_true
            .iter()
            .zip(predictions.iter())
            .map(|(y, p)| self.loss_single(y, p))
            .sum();

        total / y_true.len() as f64
    }

    /// Вычисляет loss для каждого квантиля отдельно
    pub fn loss_per_quantile(
        &self,
        y_true: &Array1<f64>,
        predictions: &Array2<f64>,
    ) -> Vec<f64> {
        let n = y_true.len();

        self.quantiles
            .iter()
            .enumerate()
            .map(|(q_idx, &quantile)| {
                let mut loss = 0.0;
                for i in 0..n {
                    loss += Self::pinball_loss(y_true[i], predictions[[i, q_idx]], quantile);
                }
                loss / n as f64
            })
            .collect()
    }

    /// Возвращает количество квантилей
    pub fn num_quantiles(&self) -> usize {
        self.quantiles.len()
    }
}

/// Метрики для оценки качества прогнозов
#[derive(Debug, Clone)]
pub struct ForecastMetrics {
    /// Mean Absolute Error
    pub mae: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Percentage Error
    pub mape: f64,
    /// Quantile Loss
    pub quantile_loss: f64,
    /// Coverage (процент попаданий в интервал)
    pub coverage: f64,
    /// Средняя ширина интервала
    pub interval_width: f64,
}

impl ForecastMetrics {
    /// Вычисляет все метрики
    pub fn compute(
        y_true: &Array1<f64>,
        prediction: &QuantilePrediction,
    ) -> Self {
        let median = prediction.median();
        let n = y_true.len() as f64;

        // MAE
        let mae = y_true
            .iter()
            .zip(median.iter())
            .map(|(t, p)| (t - p).abs())
            .sum::<f64>()
            / n;

        // RMSE
        let mse = y_true
            .iter()
            .zip(median.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>()
            / n;
        let rmse = mse.sqrt();

        // MAPE
        let mape = y_true
            .iter()
            .zip(median.iter())
            .filter_map(|(t, p)| {
                if t.abs() > 1e-10 {
                    Some(((t - p) / t).abs())
                } else {
                    None
                }
            })
            .sum::<f64>()
            / n
            * 100.0;

        // Quantile loss
        let loss_fn = QuantileLoss::with_quantiles(prediction.quantiles.clone());
        let quantile_loss = loss_fn.loss_single(y_true, &prediction.values);

        // Coverage
        let coverage = prediction.coverage(y_true);

        // Interval width
        let interval_width = prediction.interval_width().mean().unwrap_or(0.0);

        Self {
            mae,
            rmse,
            mape,
            quantile_loss,
            coverage,
            interval_width,
        }
    }

    /// Выводит метрики в читаемом формате
    pub fn to_string(&self) -> String {
        format!(
            "MAE: {:.4}, RMSE: {:.4}, MAPE: {:.2}%, QL: {:.4}, Coverage: {:.2}%, Width: {:.4}",
            self.mae,
            self.rmse,
            self.mape,
            self.quantile_loss,
            self.coverage * 100.0,
            self.interval_width
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinball_loss() {
        // Когда prediction ниже true: q * error
        assert!((QuantileLoss::pinball_loss(10.0, 8.0, 0.5) - 1.0).abs() < 1e-10);

        // Когда prediction выше true: (q-1) * error
        assert!((QuantileLoss::pinball_loss(8.0, 10.0, 0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_loss() {
        let loss_fn = QuantileLoss::new();

        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let predictions = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.5, 1.0, 1.5, // prediction for y=1
                1.5, 2.0, 2.5, // prediction for y=2
                2.5, 3.0, 3.5, // prediction for y=3
            ],
        )
        .unwrap();

        let loss = loss_fn.loss_single(&y_true, &predictions);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_quantile_prediction() {
        let quantiles = vec![0.1, 0.5, 0.9];
        let values = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.5, 1.0, 1.5, 1.5, 2.0, 2.5, 2.5, 3.0, 3.5,
            ],
        )
        .unwrap();

        let prediction = QuantilePrediction::new(quantiles, values);

        let median = prediction.median();
        assert_eq!(median.len(), 3);
        assert!((median[0] - 1.0).abs() < 1e-10);

        let lower = prediction.lower();
        assert!((lower[0] - 0.5).abs() < 1e-10);

        let upper = prediction.upper();
        assert!((upper[0] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_coverage() {
        let quantiles = vec![0.1, 0.5, 0.9];
        let values = Array2::from_shape_vec(
            (3, 3),
            vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        )
        .unwrap();

        let prediction = QuantilePrediction::new(quantiles, values);

        // Все значения попадают в интервал
        let actual = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let coverage = prediction.coverage(&actual);
        assert!((coverage - 1.0).abs() < 1e-10);

        // Ни одно значение не попадает
        let actual = Array1::from_vec(vec![5.0, 5.0, 5.0]);
        let coverage = prediction.coverage(&actual);
        assert!((coverage - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_forecast_metrics() {
        let quantiles = vec![0.1, 0.5, 0.9];
        let values = Array2::from_shape_vec(
            (3, 3),
            vec![0.5, 1.0, 1.5, 1.5, 2.0, 2.5, 2.5, 3.0, 3.5],
        )
        .unwrap();

        let prediction = QuantilePrediction::new(quantiles, values);
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let metrics = ForecastMetrics::compute(&y_true, &prediction);

        assert!(metrics.mae >= 0.0);
        assert!(metrics.rmse >= 0.0);
        assert!(metrics.coverage >= 0.0 && metrics.coverage <= 1.0);
    }
}
