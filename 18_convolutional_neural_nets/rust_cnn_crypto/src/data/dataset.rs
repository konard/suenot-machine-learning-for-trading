//! Dataset для батчевой загрузки данных

use super::sample::{Label, Sample};
use ndarray::{Array2, Array3};
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Dataset для обучения CNN
#[derive(Debug, Clone)]
pub struct Dataset {
    samples: Vec<Sample>,
    batch_size: usize,
    shuffle: bool,
    current_index: usize,
    indices: Vec<usize>,
}

/// Батч данных для обучения
#[derive(Debug, Clone)]
pub struct Batch {
    /// Входные данные [batch_size, channels, window_size]
    pub features: Array3<f32>,
    /// Метки классов [batch_size]
    pub labels: Vec<usize>,
    /// Временные метки
    pub timestamps: Vec<i64>,
}

impl Dataset {
    /// Создание нового датасета
    pub fn new(samples: Vec<Sample>, batch_size: usize) -> Self {
        let indices: Vec<usize> = (0..samples.len()).collect();
        Self {
            samples,
            batch_size,
            shuffle: true,
            current_index: 0,
            indices,
        }
    }

    /// Отключение перемешивания
    pub fn without_shuffle(mut self) -> Self {
        self.shuffle = false;
        self
    }

    /// Количество образцов
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Проверка на пустоту
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Количество батчей
    pub fn num_batches(&self) -> usize {
        (self.samples.len() + self.batch_size - 1) / self.batch_size
    }

    /// Размер батча
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Сброс итератора
    pub fn reset(&mut self) {
        self.current_index = 0;
        if self.shuffle {
            let mut rng = thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    /// Получение следующего батча
    pub fn next_batch(&mut self) -> Option<Batch> {
        if self.current_index >= self.samples.len() {
            return None;
        }

        let end_idx = (self.current_index + self.batch_size).min(self.samples.len());
        let batch_indices = &self.indices[self.current_index..end_idx];
        let actual_batch_size = batch_indices.len();

        if actual_batch_size == 0 {
            return None;
        }

        // Определяем размеры из первого образца
        let first_sample = &self.samples[batch_indices[0]];
        let (num_channels, window_size) = first_sample.shape();

        // Создаём тензоры для батча
        let mut features = Array3::zeros((actual_batch_size, num_channels, window_size));
        let mut labels = Vec::with_capacity(actual_batch_size);
        let mut timestamps = Vec::with_capacity(actual_batch_size);

        for (batch_idx, &sample_idx) in batch_indices.iter().enumerate() {
            let sample = &self.samples[sample_idx];

            // Копируем признаки
            for c in 0..num_channels {
                for w in 0..window_size {
                    features[[batch_idx, c, w]] = sample.features[[c, w]];
                }
            }

            // Добавляем метку
            labels.push(sample.label.map(|l| l.as_usize()).unwrap_or(1));
            timestamps.push(sample.timestamp);
        }

        self.current_index = end_idx;

        Some(Batch {
            features,
            labels,
            timestamps,
        })
    }

    /// Получение всех данных как один батч
    pub fn as_single_batch(&self) -> Batch {
        if self.samples.is_empty() {
            return Batch {
                features: Array3::zeros((0, 0, 0)),
                labels: Vec::new(),
                timestamps: Vec::new(),
            };
        }

        let first_sample = &self.samples[0];
        let (num_channels, window_size) = first_sample.shape();

        let mut features = Array3::zeros((self.samples.len(), num_channels, window_size));
        let mut labels = Vec::with_capacity(self.samples.len());
        let mut timestamps = Vec::with_capacity(self.samples.len());

        for (batch_idx, sample) in self.samples.iter().enumerate() {
            for c in 0..num_channels {
                for w in 0..window_size {
                    features[[batch_idx, c, w]] = sample.features[[c, w]];
                }
            }
            labels.push(sample.label.map(|l| l.as_usize()).unwrap_or(1));
            timestamps.push(sample.timestamp);
        }

        Batch {
            features,
            labels,
            timestamps,
        }
    }

    /// Распределение классов в датасете
    pub fn class_distribution(&self) -> [usize; 3] {
        let mut counts = [0usize; 3];
        for sample in &self.samples {
            if let Some(label) = sample.label {
                counts[label.as_usize()] += 1;
            }
        }
        counts
    }

    /// Веса классов для балансировки
    pub fn class_weights(&self) -> [f32; 3] {
        let dist = self.class_distribution();
        let total: usize = dist.iter().sum();
        let num_classes = 3.0;

        dist.map(|count| {
            if count > 0 {
                (total as f32) / (num_classes * count as f32)
            } else {
                1.0
            }
        })
    }

    /// Получение образцов
    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    /// Фильтрация по меткам
    pub fn filter_by_labels(&self, labels: &[Label]) -> Self {
        let filtered: Vec<Sample> = self
            .samples
            .iter()
            .filter(|s| s.label.map(|l| labels.contains(&l)).unwrap_or(false))
            .cloned()
            .collect();

        Self::new(filtered, self.batch_size)
    }
}

impl Iterator for Dataset {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_test_samples(n: usize) -> Vec<Sample> {
        (0..n)
            .map(|i| {
                let features = Array2::from_elem((10, 60), i as f32 * 0.1);
                Sample::new(features, i as i64 * 1000).with_label(Label::from(i % 3))
            })
            .collect()
    }

    #[test]
    fn test_dataset_iteration() {
        let samples = create_test_samples(100);
        let mut dataset = Dataset::new(samples, 32);

        let mut total_samples = 0;
        while let Some(batch) = dataset.next_batch() {
            total_samples += batch.labels.len();
        }

        assert_eq!(total_samples, 100);
    }

    #[test]
    fn test_class_distribution() {
        let samples = create_test_samples(99);
        let dataset = Dataset::new(samples, 32);
        let dist = dataset.class_distribution();

        assert_eq!(dist[0], 33);
        assert_eq!(dist[1], 33);
        assert_eq!(dist[2], 33);
    }
}
