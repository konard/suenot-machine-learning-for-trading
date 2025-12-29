//! # Предсказание волатильности
//!
//! Простая модель предсказания будущей реализованной волатильности
//! на основе исторических данных.

use super::{RealizedVolatility, VolatilityConfig};

/// Признаки для предсказания волатильности
#[derive(Debug, Clone)]
pub struct VolatilityFeatures {
    /// Историческая RV на разных окнах
    pub rv_5d: Option<f64>,
    pub rv_10d: Option<f64>,
    pub rv_20d: Option<f64>,
    pub rv_60d: Option<f64>,

    /// Волатильность волатильности
    pub vol_of_vol: Option<f64>,

    /// Доходность за период
    pub return_5d: Option<f64>,
    pub return_20d: Option<f64>,

    /// Текущая IV (если доступна)
    pub current_iv: Option<f64>,

    /// Отношение объёмов
    pub volume_ratio: Option<f64>,
}

impl VolatilityFeatures {
    /// Создать пустой набор признаков
    pub fn empty() -> Self {
        Self {
            rv_5d: None,
            rv_10d: None,
            rv_20d: None,
            rv_60d: None,
            vol_of_vol: None,
            return_5d: None,
            return_20d: None,
            current_iv: None,
            volume_ratio: None,
        }
    }

    /// Создать признаки из ценовых данных
    pub fn from_prices(prices: &[f64], volumes: Option<&[f64]>, current_iv: Option<f64>) -> Self {
        let rv = RealizedVolatility::new(VolatilityConfig::crypto());

        let rv_5d = rv.calculate(prices, Some(5));
        let rv_10d = rv.calculate(prices, Some(10));
        let rv_20d = rv.calculate(prices, Some(20));
        let rv_60d = rv.calculate(prices, Some(60));
        let vol_of_vol = rv.vol_of_vol(prices, 20, 20);

        // Доходности
        let return_5d = if prices.len() >= 6 {
            Some((prices.last().unwrap() / prices[prices.len() - 6] - 1.0))
        } else {
            None
        };

        let return_20d = if prices.len() >= 21 {
            Some((prices.last().unwrap() / prices[prices.len() - 21] - 1.0))
        } else {
            None
        };

        // Отношение объёмов
        let volume_ratio = volumes.and_then(|v| {
            if v.len() >= 20 {
                let recent = v.last()?;
                let avg: f64 = v[v.len() - 20..].iter().sum::<f64>() / 20.0;
                Some(recent / avg)
            } else {
                None
            }
        });

        Self {
            rv_5d,
            rv_10d,
            rv_20d,
            rv_60d,
            vol_of_vol,
            return_5d,
            return_20d,
            current_iv,
            volume_ratio,
        }
    }

    /// Преобразовать в вектор для ML модели
    pub fn to_vector(&self) -> Vec<f64> {
        vec![
            self.rv_5d.unwrap_or(0.0),
            self.rv_10d.unwrap_or(0.0),
            self.rv_20d.unwrap_or(0.0),
            self.rv_60d.unwrap_or(0.0),
            self.vol_of_vol.unwrap_or(0.0),
            self.return_5d.unwrap_or(0.0),
            self.return_20d.unwrap_or(0.0),
            self.current_iv.unwrap_or(0.0),
            self.volume_ratio.unwrap_or(1.0),
        ]
    }
}

/// Предсказатель волатильности
#[derive(Debug, Clone)]
pub struct VolatilityPredictor {
    /// Веса для линейной модели
    weights: Vec<f64>,
    /// Смещение
    bias: f64,
    /// Горизонт предсказания (дни)
    horizon: usize,
}

impl VolatilityPredictor {
    /// Создать предсказатель с весами по умолчанию
    ///
    /// Веса подобраны эмпирически:
    /// - Краткосрочная RV имеет больший вес
    /// - IV является хорошим предиктором
    /// - Vol of vol сигнализирует о возможных изменениях
    pub fn default_weights(horizon: usize) -> Self {
        Self {
            weights: vec![
                0.15, // rv_5d
                0.20, // rv_10d
                0.25, // rv_20d
                0.10, // rv_60d
                0.05, // vol_of_vol
                0.02, // return_5d
                0.03, // return_20d
                0.15, // current_iv
                0.05, // volume_ratio
            ],
            bias: 0.0,
            horizon,
        }
    }

    /// Создать предсказатель с кастомными весами
    pub fn with_weights(weights: Vec<f64>, bias: f64, horizon: usize) -> Self {
        Self {
            weights,
            bias,
            horizon,
        }
    }

    /// Предсказать будущую RV
    pub fn predict(&self, features: &VolatilityFeatures) -> f64 {
        let x = features.to_vector();

        let prediction: f64 = x
            .iter()
            .zip(self.weights.iter())
            .map(|(xi, wi)| xi * wi)
            .sum::<f64>()
            + self.bias;

        // Ограничиваем разумными значениями
        prediction.max(0.05).min(3.0)
    }

    /// Обновить веса на основе ошибки (простой gradient descent)
    pub fn update(&mut self, features: &VolatilityFeatures, actual: f64, learning_rate: f64) {
        let prediction = self.predict(features);
        let error = actual - prediction;

        let x = features.to_vector();
        for (i, xi) in x.iter().enumerate() {
            if i < self.weights.len() {
                self.weights[i] += learning_rate * error * xi;
            }
        }
        self.bias += learning_rate * error;
    }

    /// Получить текущие веса
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Горизонт предсказания
    pub fn horizon(&self) -> usize {
        self.horizon
    }
}

/// GARCH-подобная модель для предсказания волатильности
#[derive(Debug, Clone)]
pub struct GarchPredictor {
    /// Константа
    omega: f64,
    /// Коэффициент для лага волатильности
    alpha: f64,
    /// Коэффициент для лага дисперсии
    beta: f64,
    /// Текущая оценка дисперсии
    current_variance: f64,
}

impl GarchPredictor {
    /// Создать GARCH(1,1) модель со стандартными параметрами
    pub fn new() -> Self {
        Self {
            omega: 0.000001,
            alpha: 0.10,
            beta: 0.85,
            current_variance: 0.0004, // ~20% годовая волатильность
        }
    }

    /// Создать с кастомными параметрами
    pub fn with_params(omega: f64, alpha: f64, beta: f64) -> Self {
        Self {
            omega,
            alpha,
            beta,
            current_variance: omega / (1.0 - alpha - beta),
        }
    }

    /// Обновить модель новым доходом
    pub fn update(&mut self, return_value: f64) {
        // GARCH(1,1): σ²_t = ω + α*r²_{t-1} + β*σ²_{t-1}
        self.current_variance =
            self.omega + self.alpha * return_value.powi(2) + self.beta * self.current_variance;
    }

    /// Предсказать волатильность на N дней вперёд
    pub fn forecast(&self, days: usize) -> f64 {
        // Long-run variance
        let long_run = self.omega / (1.0 - self.alpha - self.beta);

        // Multi-step forecast
        let persistence = self.alpha + self.beta;
        let forecast_var =
            long_run + persistence.powi(days as i32) * (self.current_variance - long_run);

        // Аннуализируем
        (forecast_var * 365.0).sqrt()
    }

    /// Текущая оценка волатильности
    pub fn current_volatility(&self) -> f64 {
        (self.current_variance * 365.0).sqrt()
    }

    /// Долгосрочная волатильность
    pub fn long_run_volatility(&self) -> f64 {
        let long_run_var = self.omega / (1.0 - self.alpha - self.beta);
        (long_run_var * 365.0).sqrt()
    }
}

impl Default for GarchPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_features_from_prices() {
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0)
            .collect();

        let features = VolatilityFeatures::from_prices(&prices, None, Some(0.30));

        assert!(features.rv_5d.is_some());
        assert!(features.rv_20d.is_some());
        assert!(features.current_iv == Some(0.30));
    }

    #[test]
    fn test_predictor() {
        let predictor = VolatilityPredictor::default_weights(7);

        let features = VolatilityFeatures {
            rv_5d: Some(0.30),
            rv_10d: Some(0.28),
            rv_20d: Some(0.25),
            rv_60d: Some(0.22),
            vol_of_vol: Some(0.05),
            return_5d: Some(0.02),
            return_20d: Some(0.05),
            current_iv: Some(0.32),
            volume_ratio: Some(1.2),
        };

        let prediction = predictor.predict(&features);

        // Предсказание должно быть разумным
        assert!(prediction > 0.1 && prediction < 1.0, "Prediction: {}", prediction);
    }

    #[test]
    fn test_garch() {
        let mut garch = GarchPredictor::new();

        // Симулируем несколько доходностей
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.005];

        for r in returns {
            garch.update(r);
        }

        let forecast = garch.forecast(7);
        assert!(forecast > 0.0, "GARCH forecast should be positive");

        let current = garch.current_volatility();
        assert!(current > 0.0, "Current volatility should be positive");
    }
}
