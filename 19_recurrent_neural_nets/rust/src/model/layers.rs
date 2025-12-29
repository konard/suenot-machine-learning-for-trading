//! Базовые слои нейронной сети

use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Функция активации
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    /// Сигмоида: 1 / (1 + exp(-x))
    Sigmoid,
    /// Гиперболический тангенс
    Tanh,
    /// ReLU: max(0, x)
    ReLU,
    /// Leaky ReLU: max(alpha * x, x)
    LeakyReLU(f64),
    /// Линейная (без активации)
    Linear,
    /// Softmax
    Softmax,
}

impl Activation {
    /// Применяет функцию активации
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::ReLU => x.max(0.0),
            Activation::LeakyReLU(alpha) => {
                if x > 0.0 {
                    x
                } else {
                    alpha * x
                }
            }
            Activation::Linear => x,
            Activation::Softmax => x, // Softmax применяется к массиву
        }
    }

    /// Применяет функцию активации к массиву
    pub fn apply_array(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            Activation::Softmax => {
                let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_x: Array1<f64> = x.mapv(|v| (v - max_val).exp());
                let sum: f64 = exp_x.sum();
                exp_x / sum
            }
            _ => x.mapv(|v| self.apply(v)),
        }
    }

    /// Вычисляет производную функции активации
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            }
            Activation::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::LeakyReLU(alpha) => {
                if x > 0.0 {
                    1.0
                } else {
                    *alpha
                }
            }
            Activation::Linear => 1.0,
            Activation::Softmax => 1.0, // Упрощённо для backprop
        }
    }
}

/// Полносвязный (Dense) слой
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dense {
    /// Входной размер
    pub input_size: usize,
    /// Выходной размер
    pub output_size: usize,
    /// Веса [output_size, input_size]
    pub weights: Array2<f64>,
    /// Смещения [output_size]
    pub biases: Array1<f64>,
    /// Функция активации
    pub activation: Activation,
    /// Градиенты весов для обновления
    pub weight_gradients: Array2<f64>,
    /// Градиенты смещений для обновления
    pub bias_gradients: Array1<f64>,
    /// Последний вход (для backprop)
    last_input: Option<Array1<f64>>,
    /// Последний выход до активации (для backprop)
    last_pre_activation: Option<Array1<f64>>,
}

impl Dense {
    /// Создаёт новый полносвязный слой
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        // Инициализация Xavier/Glorot
        let limit = (6.0 / (input_size + output_size) as f64).sqrt();
        let weights = Array2::random((output_size, input_size), Uniform::new(-limit, limit));
        let biases = Array1::zeros(output_size);

        Self {
            input_size,
            output_size,
            weights,
            biases,
            activation,
            weight_gradients: Array2::zeros((output_size, input_size)),
            bias_gradients: Array1::zeros(output_size),
            last_input: None,
            last_pre_activation: None,
        }
    }

    /// Прямой проход
    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        self.last_input = Some(input.clone());

        // z = W * x + b
        let z = self.weights.dot(input) + &self.biases;
        self.last_pre_activation = Some(z.clone());

        // a = activation(z)
        self.activation.apply_array(&z)
    }

    /// Обратный проход
    ///
    /// # Аргументы
    ///
    /// * `output_gradient` - Градиент от следующего слоя
    ///
    /// # Возвращает
    ///
    /// Градиент для предыдущего слоя
    pub fn backward(&mut self, output_gradient: &Array1<f64>) -> Array1<f64> {
        let pre_activation = self.last_pre_activation.as_ref().expect("forward не вызван");
        let input = self.last_input.as_ref().expect("forward не вызван");

        // Градиент активации
        let activation_grad: Array1<f64> = pre_activation
            .iter()
            .map(|&x| self.activation.derivative(x))
            .collect();

        // delta = output_gradient * activation'(z)
        let delta = output_gradient * &activation_grad;

        // Градиенты весов: dW = delta * input^T
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                self.weight_gradients[[i, j]] += delta[i] * input[j];
            }
        }

        // Градиенты смещений: db = delta
        self.bias_gradients = self.bias_gradients.clone() + &delta;

        // Градиент для предыдущего слоя: dx = W^T * delta
        self.weights.t().dot(&delta)
    }

    /// Обновляет веса с использованием градиентного спуска
    pub fn update_weights(&mut self, learning_rate: f64) {
        self.weights = &self.weights - &(learning_rate * &self.weight_gradients);
        self.biases = &self.biases - &(learning_rate * &self.bias_gradients);

        // Обнуляем градиенты
        self.weight_gradients.fill(0.0);
        self.bias_gradients.fill(0.0);
    }

    /// Обновляет веса с momentum
    pub fn update_weights_momentum(
        &mut self,
        learning_rate: f64,
        momentum: f64,
        velocity_w: &mut Array2<f64>,
        velocity_b: &mut Array1<f64>,
    ) {
        // v = momentum * v - lr * grad
        *velocity_w = momentum * velocity_w.clone() - learning_rate * &self.weight_gradients;
        *velocity_b = momentum * velocity_b.clone() - learning_rate * &self.bias_gradients;

        // w = w + v
        self.weights = &self.weights + velocity_w;
        self.biases = &self.biases + velocity_b;

        self.weight_gradients.fill(0.0);
        self.bias_gradients.fill(0.0);
    }
}

/// Dropout слой для регуляризации
#[derive(Debug, Clone)]
pub struct Dropout {
    /// Вероятность "выключения" нейрона
    pub rate: f64,
    /// Маска dropout (для backprop)
    mask: Option<Array1<f64>>,
    /// Режим обучения/инференса
    pub training: bool,
}

impl Dropout {
    /// Создаёт новый Dropout слой
    pub fn new(rate: f64) -> Self {
        Self {
            rate,
            mask: None,
            training: true,
        }
    }

    /// Прямой проход
    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        if !self.training || self.rate == 0.0 {
            return input.clone();
        }

        let mut rng = rand::thread_rng();
        let mask: Array1<f64> = Array1::from_iter(
            (0..input.len()).map(|_| {
                if rng.gen::<f64>() > self.rate {
                    1.0 / (1.0 - self.rate) // Inverted dropout
                } else {
                    0.0
                }
            }),
        );

        self.mask = Some(mask.clone());
        input * &mask
    }

    /// Обратный проход
    pub fn backward(&self, output_gradient: &Array1<f64>) -> Array1<f64> {
        if !self.training || self.rate == 0.0 {
            return output_gradient.clone();
        }

        let mask = self.mask.as_ref().expect("forward не вызван");
        output_gradient * mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sigmoid_activation() {
        let sigmoid = Activation::Sigmoid;
        assert!((sigmoid.apply(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid.apply(10.0) > 0.99);
        assert!(sigmoid.apply(-10.0) < 0.01);
    }

    #[test]
    fn test_relu_activation() {
        let relu = Activation::ReLU;
        assert_eq!(relu.apply(5.0), 5.0);
        assert_eq!(relu.apply(-5.0), 0.0);
        assert_eq!(relu.apply(0.0), 0.0);
    }

    #[test]
    fn test_softmax() {
        let softmax = Activation::Softmax;
        let input = array![1.0, 2.0, 3.0];
        let output = softmax.apply_array(&input);

        // Сумма должна быть 1
        assert!((output.sum() - 1.0).abs() < 1e-10);
        // Все значения положительные
        assert!(output.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_dense_forward() {
        let mut dense = Dense::new(3, 2, Activation::Linear);
        let input = array![1.0, 2.0, 3.0];
        let output = dense.forward(&input);

        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_dropout() {
        let mut dropout = Dropout::new(0.5);
        dropout.training = true;

        let input = array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let output = dropout.forward(&input);

        // Некоторые значения должны быть 0 (выключены)
        let zeros = output.iter().filter(|&&x| x == 0.0).count();
        assert!(zeros > 0 && zeros < 10);
    }
}
