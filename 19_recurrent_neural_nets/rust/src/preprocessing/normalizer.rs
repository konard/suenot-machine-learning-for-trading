//! Нормализаторы данных для подготовки к обучению

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Трейт для нормализаторов данных
pub trait Normalizer: Send + Sync {
    /// Обучает нормализатор на данных
    fn fit(&mut self, data: &Array2<f64>);

    /// Преобразует данные
    fn transform(&self, data: &Array2<f64>) -> Array2<f64>;

    /// Обучает и преобразует данные за один вызов
    fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }

    /// Обратное преобразование
    fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64>;

    /// Преобразует одномерный массив (для предсказаний)
    fn transform_1d(&self, data: &Array1<f64>) -> Array1<f64>;

    /// Обратное преобразование одномерного массива
    fn inverse_transform_1d(&self, data: &Array1<f64>) -> Array1<f64>;
}

/// Min-Max нормализация (приводит значения к диапазону [0, 1])
///
/// Формула: x_norm = (x - min) / (max - min)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinMaxNormalizer {
    /// Минимальные значения по каждому признаку
    min_vals: Option<Array1<f64>>,
    /// Максимальные значения по каждому признаку
    max_vals: Option<Array1<f64>>,
    /// Диапазон (max - min)
    range: Option<Array1<f64>>,
}

impl Default for MinMaxNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MinMaxNormalizer {
    /// Создаёт новый Min-Max нормализатор
    pub fn new() -> Self {
        Self {
            min_vals: None,
            max_vals: None,
            range: None,
        }
    }

    /// Проверяет, обучен ли нормализатор
    pub fn is_fitted(&self) -> bool {
        self.min_vals.is_some()
    }

    /// Возвращает минимальные значения
    pub fn min_vals(&self) -> Option<&Array1<f64>> {
        self.min_vals.as_ref()
    }

    /// Возвращает максимальные значения
    pub fn max_vals(&self) -> Option<&Array1<f64>> {
        self.max_vals.as_ref()
    }
}

impl Normalizer for MinMaxNormalizer {
    fn fit(&mut self, data: &Array2<f64>) {
        // Находим min и max по каждому столбцу (признаку)
        let min_vals = data
            .axis_iter(Axis(1))
            .map(|col| col.iter().cloned().fold(f64::INFINITY, f64::min))
            .collect::<Array1<f64>>();

        let max_vals = data
            .axis_iter(Axis(1))
            .map(|col| col.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            .collect::<Array1<f64>>();

        let range = &max_vals - &min_vals;
        // Защита от деления на ноль
        let range = range.mapv(|x| if x == 0.0 { 1.0 } else { x });

        self.min_vals = Some(min_vals);
        self.max_vals = Some(max_vals);
        self.range = Some(range);
    }

    fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let min_vals = self.min_vals.as_ref().expect("Нормализатор не обучен");
        let range = self.range.as_ref().expect("Нормализатор не обучен");

        let mut result = data.clone();

        for (i, mut col) in result.axis_iter_mut(Axis(1)).enumerate() {
            col.mapv_inplace(|x| (x - min_vals[i]) / range[i]);
        }

        result
    }

    fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let min_vals = self.min_vals.as_ref().expect("Нормализатор не обучен");
        let range = self.range.as_ref().expect("Нормализатор не обучен");

        let mut result = data.clone();

        for (i, mut col) in result.axis_iter_mut(Axis(1)).enumerate() {
            col.mapv_inplace(|x| x * range[i] + min_vals[i]);
        }

        result
    }

    fn transform_1d(&self, data: &Array1<f64>) -> Array1<f64> {
        let min_vals = self.min_vals.as_ref().expect("Нормализатор не обучен");
        let range = self.range.as_ref().expect("Нормализатор не обучен");

        data.iter()
            .enumerate()
            .map(|(i, &x)| (x - min_vals[i % min_vals.len()]) / range[i % range.len()])
            .collect()
    }

    fn inverse_transform_1d(&self, data: &Array1<f64>) -> Array1<f64> {
        let min_vals = self.min_vals.as_ref().expect("Нормализатор не обучен");
        let range = self.range.as_ref().expect("Нормализатор не обучен");

        data.iter()
            .enumerate()
            .map(|(i, &x)| x * range[i % range.len()] + min_vals[i % min_vals.len()])
            .collect()
    }
}

/// Стандартная нормализация (Z-score)
///
/// Формула: x_norm = (x - mean) / std
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardNormalizer {
    /// Средние значения по каждому признаку
    means: Option<Array1<f64>>,
    /// Стандартные отклонения по каждому признаку
    stds: Option<Array1<f64>>,
}

impl Default for StandardNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl StandardNormalizer {
    /// Создаёт новый стандартный нормализатор
    pub fn new() -> Self {
        Self {
            means: None,
            stds: None,
        }
    }

    /// Проверяет, обучен ли нормализатор
    pub fn is_fitted(&self) -> bool {
        self.means.is_some()
    }
}

impl Normalizer for StandardNormalizer {
    fn fit(&mut self, data: &Array2<f64>) {
        let n = data.nrows() as f64;

        // Вычисляем среднее по каждому столбцу
        let means = data
            .axis_iter(Axis(1))
            .map(|col| col.sum() / n)
            .collect::<Array1<f64>>();

        // Вычисляем стандартное отклонение
        let stds = data
            .axis_iter(Axis(1))
            .enumerate()
            .map(|(i, col)| {
                let mean = means[i];
                let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
                let std = variance.sqrt();
                // Защита от деления на ноль
                if std == 0.0 {
                    1.0
                } else {
                    std
                }
            })
            .collect::<Array1<f64>>();

        self.means = Some(means);
        self.stds = Some(stds);
    }

    fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let means = self.means.as_ref().expect("Нормализатор не обучен");
        let stds = self.stds.as_ref().expect("Нормализатор не обучен");

        let mut result = data.clone();

        for (i, mut col) in result.axis_iter_mut(Axis(1)).enumerate() {
            col.mapv_inplace(|x| (x - means[i]) / stds[i]);
        }

        result
    }

    fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let means = self.means.as_ref().expect("Нормализатор не обучен");
        let stds = self.stds.as_ref().expect("Нормализатор не обучен");

        let mut result = data.clone();

        for (i, mut col) in result.axis_iter_mut(Axis(1)).enumerate() {
            col.mapv_inplace(|x| x * stds[i] + means[i]);
        }

        result
    }

    fn transform_1d(&self, data: &Array1<f64>) -> Array1<f64> {
        let means = self.means.as_ref().expect("Нормализатор не обучен");
        let stds = self.stds.as_ref().expect("Нормализатор не обучен");

        data.iter()
            .enumerate()
            .map(|(i, &x)| (x - means[i % means.len()]) / stds[i % stds.len()])
            .collect()
    }

    fn inverse_transform_1d(&self, data: &Array1<f64>) -> Array1<f64> {
        let means = self.means.as_ref().expect("Нормализатор не обучен");
        let stds = self.stds.as_ref().expect("Нормализатор не обучен");

        data.iter()
            .enumerate()
            .map(|(i, &x)| x * stds[i % stds.len()] + means[i % means.len()])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_minmax_normalizer() {
        let data = array![[0.0, 100.0], [50.0, 200.0], [100.0, 300.0]];

        let mut normalizer = MinMaxNormalizer::new();
        let normalized = normalizer.fit_transform(&data);

        // Проверяем, что значения в диапазоне [0, 1]
        assert!((normalized[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((normalized[[2, 0]] - 1.0).abs() < 1e-10);
        assert!((normalized[[1, 0]] - 0.5).abs() < 1e-10);

        // Проверяем обратное преобразование
        let restored = normalizer.inverse_transform(&normalized);
        assert!((restored[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((restored[[2, 0]] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_standard_normalizer() {
        let data = array![[1.0], [2.0], [3.0], [4.0], [5.0]];

        let mut normalizer = StandardNormalizer::new();
        let normalized = normalizer.fit_transform(&data);

        // Среднее должно быть примерно 0
        let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert!(mean.abs() < 1e-10);

        // Проверяем обратное преобразование
        let restored = normalizer.inverse_transform(&normalized);
        assert!((restored[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((restored[[4, 0]] - 5.0).abs() < 1e-10);
    }
}
