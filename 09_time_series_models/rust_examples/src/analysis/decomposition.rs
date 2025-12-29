//! Декомпозиция временных рядов

use crate::analysis::statistics::{mean, moving_average};

/// Результат декомпозиции временного ряда
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
    pub period: usize,
    pub model: DecompositionModel,
}

/// Тип модели декомпозиции
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecompositionModel {
    Additive,      // Y = T + S + R
    Multiplicative, // Y = T * S * R
}

/// Классическая декомпозиция временного ряда
///
/// Разделяет ряд на компоненты: тренд, сезонность, остаток
pub fn decompose(
    data: &[f64],
    period: usize,
    model: DecompositionModel,
) -> Option<DecompositionResult> {
    let n = data.len();
    if n < period * 2 {
        return None;
    }

    // 1. Оценка тренда с помощью скользящего среднего
    let trend = if period % 2 == 0 {
        // Для чётного периода используем центрированное MA
        let ma1 = moving_average(data, period);
        let ma2 = moving_average(&ma1, 2);

        // Центрируем
        let offset = period / 2;
        let mut trend = vec![f64::NAN; n];
        for (i, &val) in ma2.iter().enumerate() {
            let idx = i + offset;
            if idx < n {
                trend[idx] = val;
            }
        }
        trend
    } else {
        // Для нечётного периода просто MA
        let ma = moving_average(data, period);
        let offset = period / 2;
        let mut trend = vec![f64::NAN; n];
        for (i, &val) in ma.iter().enumerate() {
            let idx = i + offset;
            if idx < n {
                trend[idx] = val;
            }
        }
        trend
    };

    // 2. Удаляем тренд для получения сезонной + остаточной компоненты
    let detrended: Vec<f64> = data
        .iter()
        .zip(trend.iter())
        .map(|(&y, &t)| {
            if t.is_nan() {
                f64::NAN
            } else {
                match model {
                    DecompositionModel::Additive => y - t,
                    DecompositionModel::Multiplicative => {
                        if t != 0.0 {
                            y / t
                        } else {
                            f64::NAN
                        }
                    }
                }
            }
        })
        .collect();

    // 3. Оценка сезонной компоненты
    let mut seasonal_indices = vec![0.0; period];
    let mut counts = vec![0usize; period];

    for (i, &val) in detrended.iter().enumerate() {
        if !val.is_nan() {
            let season_idx = i % period;
            seasonal_indices[season_idx] += val;
            counts[season_idx] += 1;
        }
    }

    // Усредняем сезонные индексы
    for i in 0..period {
        if counts[i] > 0 {
            seasonal_indices[i] /= counts[i] as f64;
        }
    }

    // Нормализуем сезонную компоненту
    match model {
        DecompositionModel::Additive => {
            let avg = mean(&seasonal_indices);
            for s in &mut seasonal_indices {
                *s -= avg;
            }
        }
        DecompositionModel::Multiplicative => {
            let avg = mean(&seasonal_indices);
            if avg != 0.0 {
                for s in &mut seasonal_indices {
                    *s /= avg;
                }
            }
        }
    }

    // 4. Расширяем сезонную компоненту на весь ряд
    let seasonal: Vec<f64> = (0..n).map(|i| seasonal_indices[i % period]).collect();

    // 5. Вычисляем остаток
    let residual: Vec<f64> = data
        .iter()
        .zip(trend.iter())
        .zip(seasonal.iter())
        .map(|((&y, &t), &s)| {
            if t.is_nan() {
                f64::NAN
            } else {
                match model {
                    DecompositionModel::Additive => y - t - s,
                    DecompositionModel::Multiplicative => {
                        if t != 0.0 && s != 0.0 {
                            y / (t * s)
                        } else {
                            f64::NAN
                        }
                    }
                }
            }
        })
        .collect();

    Some(DecompositionResult {
        trend,
        seasonal,
        residual,
        period,
        model,
    })
}

/// Определение силы тренда
///
/// Возвращает значение от 0 до 1, где 1 = сильный тренд
pub fn trend_strength(decomposition: &DecompositionResult) -> f64 {
    let valid_residuals: Vec<f64> = decomposition
        .residual
        .iter()
        .filter(|x| !x.is_nan())
        .cloned()
        .collect();

    let detrended: Vec<f64> = decomposition
        .residual
        .iter()
        .zip(decomposition.seasonal.iter())
        .filter(|(r, _)| !r.is_nan())
        .map(|(r, s)| {
            match decomposition.model {
                DecompositionModel::Additive => r + s,
                DecompositionModel::Multiplicative => r * s,
            }
        })
        .collect();

    let var_residual: f64 = variance(&valid_residuals);
    let var_detrended: f64 = variance(&detrended);

    if var_detrended == 0.0 {
        return 0.0;
    }

    (1.0 - var_residual / var_detrended).max(0.0)
}

/// Определение силы сезонности
///
/// Возвращает значение от 0 до 1, где 1 = сильная сезонность
pub fn seasonality_strength(decomposition: &DecompositionResult) -> f64 {
    let valid_residuals: Vec<f64> = decomposition
        .residual
        .iter()
        .filter(|x| !x.is_nan())
        .cloned()
        .collect();

    let deseasoned: Vec<f64> = decomposition
        .residual
        .iter()
        .zip(decomposition.trend.iter())
        .filter(|(r, t)| !r.is_nan() && !t.is_nan())
        .map(|(r, t)| {
            match decomposition.model {
                DecompositionModel::Additive => r + t,
                DecompositionModel::Multiplicative => r * t,
            }
        })
        .collect();

    let var_residual: f64 = variance(&valid_residuals);
    let var_deseasoned: f64 = variance(&deseasoned);

    if var_deseasoned == 0.0 {
        return 0.0;
    }

    (1.0 - var_residual / var_deseasoned).max(0.0)
}

fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_additive() {
        // Создаём ряд с известными компонентами
        let period = 12;
        let n = 48;

        let data: Vec<f64> = (0..n)
            .map(|i| {
                let trend = 100.0 + i as f64 * 0.5; // Линейный тренд
                let seasonal = 10.0 * (2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64).sin();
                trend + seasonal
            })
            .collect();

        let result = decompose(&data, period, DecompositionModel::Additive);
        assert!(result.is_some());

        let result = result.unwrap();
        assert_eq!(result.period, period);
        assert_eq!(result.trend.len(), n);
        assert_eq!(result.seasonal.len(), n);
        assert_eq!(result.residual.len(), n);
    }
}
