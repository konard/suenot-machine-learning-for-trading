//! Простая нейронная сеть для DQN

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Функция активации
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    ReLU,
    Tanh,
    Sigmoid,
    Linear,
}

impl Activation {
    /// Применить функцию активации
    pub fn forward(&self, x: f64) -> f64 {
        match self {
            Self::ReLU => x.max(0.0),
            Self::Tanh => x.tanh(),
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::Linear => x,
        }
    }

    /// Производная функции активации
    pub fn backward(&self, x: f64) -> f64 {
        match self {
            Self::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Self::Tanh => 1.0 - x.tanh().powi(2),
            Self::Sigmoid => {
                let s = self.forward(x);
                s * (1.0 - s)
            }
            Self::Linear => 1.0,
        }
    }
}

/// Слой нейронной сети
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    /// Веса (input_size x output_size)
    pub weights: Array2<f64>,
    /// Смещения
    pub biases: Array1<f64>,
    /// Функция активации
    pub activation: Activation,
    /// Кэш для обратного прохода
    #[serde(skip)]
    input_cache: Option<Array1<f64>>,
    #[serde(skip)]
    output_cache: Option<Array1<f64>>,
}

impl Layer {
    /// Создать новый слой
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier инициализация
        let std = (2.0 / (input_size + output_size) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weights = Array2::from_shape_fn((input_size, output_size), |_| {
            normal.sample(&mut rng)
        });

        let biases = Array1::zeros(output_size);

        Self {
            weights,
            biases,
            activation,
            input_cache: None,
            output_cache: None,
        }
    }

    /// Прямой проход
    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        self.input_cache = Some(input.clone());

        // z = Wx + b
        let z = input.dot(&self.weights) + &self.biases;

        // a = activation(z)
        let output = z.mapv(|x| self.activation.forward(x));
        self.output_cache = Some(output.clone());

        output
    }

    /// Обратный проход
    pub fn backward(&mut self, grad_output: &Array1<f64>, learning_rate: f64) -> Array1<f64> {
        let input = self.input_cache.as_ref().expect("Forward pass not called");
        let output = self.output_cache.as_ref().expect("Forward pass not called");

        // Градиент через активацию
        let grad_z: Array1<f64> = grad_output.iter()
            .zip(output.iter())
            .map(|(&go, &o)| go * self.activation.backward(o))
            .collect();

        // Градиент весов
        let grad_weights = input.view()
            .insert_axis(Axis(1))
            .dot(&grad_z.view().insert_axis(Axis(0)));

        // Градиент смещений
        let grad_biases = grad_z.clone();

        // Градиент для предыдущего слоя
        let grad_input = self.weights.dot(&grad_z);

        // Обновляем веса
        self.weights = &self.weights - &(grad_weights * learning_rate);
        self.biases = &self.biases - &(grad_biases * learning_rate);

        grad_input
    }

    /// Выходной размер слоя
    pub fn output_size(&self) -> usize {
        self.biases.len()
    }
}

/// Полносвязная нейронная сеть
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    /// Слои сети
    layers: Vec<Layer>,
    /// Скорость обучения
    learning_rate: f64,
}

impl NeuralNetwork {
    /// Создать новую сеть
    pub fn new(layer_sizes: &[usize], learning_rate: f64) -> Self {
        let mut layers = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let activation = if i == layer_sizes.len() - 2 {
                Activation::Linear // Выходной слой
            } else {
                Activation::ReLU // Скрытые слои
            };

            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1], activation));
        }

        Self {
            layers,
            learning_rate,
        }
    }

    /// Прямой проход
    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    /// Прямой проход без изменения состояния
    pub fn predict(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut output = input.clone();
        for layer in &self.layers {
            let z = output.dot(&layer.weights) + &layer.biases;
            output = z.mapv(|x| layer.activation.forward(x));
        }
        output
    }

    /// Обратный проход с обновлением весов
    pub fn backward(&mut self, loss_gradient: &Array1<f64>) {
        let mut grad = loss_gradient.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad, self.learning_rate);
        }
    }

    /// Обучить на одном примере (MSE loss)
    pub fn train_single(&mut self, input: &Array1<f64>, target: &Array1<f64>) -> f64 {
        let output = self.forward(input);

        // MSE loss и градиент
        let diff = &output - target;
        let loss = diff.iter().map(|x| x.powi(2)).sum::<f64>() / diff.len() as f64;

        // Градиент MSE: 2 * (output - target) / n
        let grad: Array1<f64> = diff.mapv(|x| 2.0 * x / diff.len() as f64);

        self.backward(&grad);

        loss
    }

    /// Обучить на батче
    pub fn train_batch(
        &mut self,
        inputs: &[Array1<f64>],
        targets: &[Array1<f64>],
    ) -> f64 {
        let mut total_loss = 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            total_loss += self.train_single(input, target);
        }

        total_loss / inputs.len() as f64
    }

    /// Копировать веса из другой сети
    pub fn copy_weights_from(&mut self, other: &NeuralNetwork) {
        for (self_layer, other_layer) in self.layers.iter_mut().zip(other.layers.iter()) {
            self_layer.weights.assign(&other_layer.weights);
            self_layer.biases.assign(&other_layer.biases);
        }
    }

    /// Soft update весов (для target network)
    pub fn soft_update(&mut self, source: &NeuralNetwork, tau: f64) {
        for (self_layer, source_layer) in self.layers.iter_mut().zip(source.layers.iter()) {
            self_layer.weights = &self_layer.weights * (1.0 - tau) + &source_layer.weights * tau;
            self_layer.biases = &self_layer.biases * (1.0 - tau) + &source_layer.biases * tau;
        }
    }

    /// Установить скорость обучения
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    /// Получить количество параметров
    pub fn num_parameters(&self) -> usize {
        self.layers.iter()
            .map(|l| l.weights.len() + l.biases.len())
            .sum()
    }

    /// Сохранить в JSON
    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Загрузить из JSON
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let net = NeuralNetwork::new(&[10, 64, 32, 11], 0.001);
        assert_eq!(net.layers.len(), 3);
    }

    #[test]
    fn test_forward_pass() {
        let mut net = NeuralNetwork::new(&[4, 8, 2], 0.001);
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let output = net.forward(&input);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_training() {
        let mut net = NeuralNetwork::new(&[2, 4, 1], 0.01);

        // XOR-подобная задача
        let inputs = vec![
            Array1::from_vec(vec![0.0, 0.0]),
            Array1::from_vec(vec![0.0, 1.0]),
            Array1::from_vec(vec![1.0, 0.0]),
            Array1::from_vec(vec![1.0, 1.0]),
        ];
        let targets = vec![
            Array1::from_vec(vec![0.0]),
            Array1::from_vec(vec![1.0]),
            Array1::from_vec(vec![1.0]),
            Array1::from_vec(vec![0.0]),
        ];

        let initial_loss = net.train_batch(&inputs, &targets);

        // Обучаем несколько эпох
        for _ in 0..100 {
            net.train_batch(&inputs, &targets);
        }

        let final_loss = net.train_batch(&inputs, &targets);
        assert!(final_loss < initial_loss);
    }

    #[test]
    fn test_soft_update() {
        let mut target = NeuralNetwork::new(&[4, 8, 4], 0.001);
        let source = NeuralNetwork::new(&[4, 8, 4], 0.001);

        let old_weight = target.layers[0].weights[[0, 0]];
        target.soft_update(&source, 0.1);
        let new_weight = target.layers[0].weights[[0, 0]];

        // Вес должен измениться, но не полностью
        assert!((old_weight - new_weight).abs() > 0.0);
    }
}
