//! Архитектура CNN модели
//!
//! Реализация 1D CNN для анализа временных рядов криптовалют.

use super::config::CnnConfig;
use burn::{
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        pool::{MaxPool1d, MaxPool1dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    tensor::{backend::Backend, Tensor},
};

/// 1D CNN модель для прогнозирования направления цены
#[derive(Module, Debug)]
pub struct CnnModel<B: Backend> {
    /// Первый сверточный слой
    conv1: Conv1d<B>,
    /// Второй сверточный слой
    conv2: Conv1d<B>,
    /// Третий сверточный слой
    conv3: Conv1d<B>,
    /// Первый пулинг
    pool1: MaxPool1d,
    /// Второй пулинг
    pool2: MaxPool1d,
    /// Полносвязный слой 1
    fc1: Linear<B>,
    /// Полносвязный слой 2 (выходной)
    fc2: Linear<B>,
    /// Dropout
    dropout: Dropout,
    /// Функция активации
    activation: Relu,
    /// Конфигурация
    #[module(skip)]
    config: CnnConfig,
}

impl<B: Backend> CnnModel<B> {
    /// Создание новой модели
    pub fn new(device: &B::Device, config: &CnnConfig) -> Self {
        // Вычисляем размер после сверточных слоёв
        let conv_output_size = Self::compute_flatten_size(config);

        // Сверточные слои
        let conv1 = Conv1dConfig::new(config.in_channels, config.conv1_filters, config.kernel_size)
            .with_padding(burn::nn::PaddingConfig1d::Valid)
            .init(device);

        let conv2 = Conv1dConfig::new(
            config.conv1_filters,
            config.conv2_filters,
            config.kernel_size,
        )
        .with_padding(burn::nn::PaddingConfig1d::Valid)
        .init(device);

        let conv3 = Conv1dConfig::new(
            config.conv2_filters,
            config.conv3_filters,
            config.kernel_size,
        )
        .with_padding(burn::nn::PaddingConfig1d::Valid)
        .init(device);

        // Пулинг
        let pool1 = MaxPool1dConfig::new(config.pool_size).init();
        let pool2 = MaxPool1dConfig::new(config.pool_size).init();

        // Полносвязные слои
        let fc1 = LinearConfig::new(conv_output_size, config.fc_size).init(device);
        let fc2 = LinearConfig::new(config.fc_size, config.num_classes).init(device);

        // Dropout
        let dropout = DropoutConfig::new(config.dropout).init();

        Self {
            conv1,
            conv2,
            conv3,
            pool1,
            pool2,
            fc1,
            fc2,
            dropout,
            activation: Relu::new(),
            config: config.clone(),
        }
    }

    /// Вычисление размера после flatten
    fn compute_flatten_size(config: &CnnConfig) -> usize {
        let mut size = config.input_size;

        // Conv1 + Pool1
        size = size - config.kernel_size + 1;
        size = size / config.pool_size;

        // Conv2 + Pool2
        size = size - config.kernel_size + 1;
        size = size / config.pool_size;

        // Conv3 (без пулинга)
        size = size - config.kernel_size + 1;

        size * config.conv3_filters
    }

    /// Прямой проход через сеть
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        // x: [batch_size, channels, seq_len]

        // Сверточные слои с активацией и пулингом
        let x = self.conv1.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool1.forward(x);

        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool2.forward(x);

        let x = self.conv3.forward(x);
        let x = self.activation.forward(x);

        // Flatten
        let batch_size = x.dims()[0];
        let x = x.flatten(1, 2);

        // Полносвязные слои
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        let x = self.fc2.forward(x);

        x
    }

    /// Предсказание класса
    pub fn predict(&self, x: Tensor<B, 3>) -> Tensor<B, 1, burn::tensor::Int> {
        let logits = self.forward(x);
        logits.argmax(1)
    }

    /// Предсказание вероятностей
    pub fn predict_proba(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let logits = self.forward(x);
        burn::tensor::activation::softmax(logits, 1)
    }

    /// Получение конфигурации
    pub fn config(&self) -> &CnnConfig {
        &self.config
    }
}

/// Вспомогательный модуль для создания модели
pub struct CnnModelBuilder {
    config: CnnConfig,
}

impl CnnModelBuilder {
    /// Создание билдера с конфигурацией по умолчанию
    pub fn new() -> Self {
        Self {
            config: CnnConfig::default(),
        }
    }

    /// Установка количества входных каналов
    pub fn in_channels(mut self, channels: usize) -> Self {
        self.config.in_channels = channels;
        self
    }

    /// Установка размера входного окна
    pub fn input_size(mut self, size: usize) -> Self {
        self.config.input_size = size;
        self
    }

    /// Установка количества классов
    pub fn num_classes(mut self, classes: usize) -> Self {
        self.config.num_classes = classes;
        self
    }

    /// Установка количества фильтров
    pub fn filters(mut self, conv1: usize, conv2: usize, conv3: usize) -> Self {
        self.config.conv1_filters = conv1;
        self.config.conv2_filters = conv2;
        self.config.conv3_filters = conv3;
        self
    }

    /// Установка dropout
    pub fn dropout(mut self, rate: f64) -> Self {
        self.config.dropout = rate;
        self
    }

    /// Построение модели
    pub fn build<B: Backend>(self, device: &B::Device) -> CnnModel<B> {
        CnnModel::new(device, &self.config)
    }
}

impl Default for CnnModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_model_creation() {
        let device = Default::default();
        let config = CnnConfig::default();
        let _model: CnnModel<TestBackend> = CnnModel::new(&device, &config);
    }

    #[test]
    fn test_forward_pass() {
        let device = Default::default();
        let config = CnnConfig::default();
        let model: CnnModel<TestBackend> = CnnModel::new(&device, &config);

        // Создаём тестовый тензор [batch=2, channels=10, seq=60]
        let input = Tensor::<TestBackend, 3>::zeros([2, 10, 60], &device);
        let output = model.forward(input);

        assert_eq!(output.dims(), [2, 3]); // [batch_size, num_classes]
    }

    #[test]
    fn test_builder() {
        let device = Default::default();
        let model: CnnModel<TestBackend> = CnnModelBuilder::new()
            .in_channels(5)
            .input_size(30)
            .num_classes(2)
            .build(&device);

        assert_eq!(model.config().in_channels, 5);
        assert_eq!(model.config().input_size, 30);
        assert_eq!(model.config().num_classes, 2);
    }
}
