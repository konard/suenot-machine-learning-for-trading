//! Конфигурация CNN модели

use serde::{Deserialize, Serialize};

/// Конфигурация архитектуры CNN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CnnConfig {
    /// Количество входных каналов (признаков)
    pub in_channels: usize,
    /// Размер входного окна
    pub input_size: usize,
    /// Количество выходных классов
    pub num_classes: usize,
    /// Количество фильтров в первом сверточном слое
    pub conv1_filters: usize,
    /// Количество фильтров во втором сверточном слое
    pub conv2_filters: usize,
    /// Количество фильтров в третьем сверточном слое
    pub conv3_filters: usize,
    /// Размер ядра свёртки
    pub kernel_size: usize,
    /// Размер пулинга
    pub pool_size: usize,
    /// Размер полносвязного слоя
    pub fc_size: usize,
    /// Dropout rate
    pub dropout: f64,
}

impl Default for CnnConfig {
    fn default() -> Self {
        Self {
            in_channels: 10,
            input_size: 60,
            num_classes: 3,
            conv1_filters: 32,
            conv2_filters: 64,
            conv3_filters: 128,
            kernel_size: 3,
            pool_size: 2,
            fc_size: 64,
            dropout: 0.3,
        }
    }
}

impl CnnConfig {
    /// Создание конфигурации для быстрой модели (меньше параметров)
    pub fn fast() -> Self {
        Self {
            conv1_filters: 16,
            conv2_filters: 32,
            conv3_filters: 64,
            fc_size: 32,
            ..Default::default()
        }
    }

    /// Создание конфигурации для большой модели (больше параметров)
    pub fn large() -> Self {
        Self {
            conv1_filters: 64,
            conv2_filters: 128,
            conv3_filters: 256,
            fc_size: 128,
            ..Default::default()
        }
    }

    /// Вычисление размера после сверточных слоёв
    pub fn compute_conv_output_size(&self) -> usize {
        let mut size = self.input_size;

        // После каждого conv + pool слоя
        // Conv1: size - kernel_size + 1
        size = size - self.kernel_size + 1;
        size = size / self.pool_size;

        // Conv2
        size = size - self.kernel_size + 1;
        size = size / self.pool_size;

        // Conv3 (без пулинга)
        size = size - self.kernel_size + 1;

        size * self.conv3_filters
    }

    /// Валидация конфигурации
    pub fn validate(&self) -> Result<(), String> {
        if self.in_channels == 0 {
            return Err("in_channels must be > 0".to_string());
        }
        if self.input_size < 10 {
            return Err("input_size must be >= 10".to_string());
        }
        if self.num_classes < 2 {
            return Err("num_classes must be >= 2".to_string());
        }
        if self.kernel_size < 2 {
            return Err("kernel_size must be >= 2".to_string());
        }
        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err("dropout must be in [0, 1]".to_string());
        }
        Ok(())
    }
}

/// Конфигурация обучения
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Количество эпох
    pub num_epochs: usize,
    /// Размер батча
    pub batch_size: usize,
    /// Скорость обучения
    pub learning_rate: f64,
    /// Decay скорости обучения
    pub lr_decay: f64,
    /// Терпение для early stopping
    pub patience: usize,
    /// Минимальное улучшение для early stopping
    pub min_delta: f64,
    /// Соотношение валидационной выборки
    pub validation_split: f64,
    /// Использовать веса классов
    pub use_class_weights: bool,
    /// Путь для сохранения модели
    pub checkpoint_path: Option<String>,
    /// Логировать каждые N батчей
    pub log_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 50,
            batch_size: 32,
            learning_rate: 0.001,
            lr_decay: 0.95,
            patience: 10,
            min_delta: 0.001,
            validation_split: 0.2,
            use_class_weights: true,
            checkpoint_path: Some("model_checkpoint".to_string()),
            log_interval: 10,
        }
    }
}

impl TrainingConfig {
    /// Быстрое обучение (для тестирования)
    pub fn quick() -> Self {
        Self {
            num_epochs: 5,
            batch_size: 64,
            patience: 3,
            ..Default::default()
        }
    }

    /// Тщательное обучение
    pub fn thorough() -> Self {
        Self {
            num_epochs: 100,
            batch_size: 16,
            patience: 20,
            learning_rate: 0.0005,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CnnConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_conv_output_size() {
        let config = CnnConfig::default();
        let output_size = config.compute_conv_output_size();
        assert!(output_size > 0);
    }
}
