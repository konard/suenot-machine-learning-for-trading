//! # Autoencoder Models
//!
//! Модуль с реализациями различных автоэнкодеров:
//! - Vanilla Autoencoder (простой автоэнкодер)
//! - Deep Autoencoder (глубокий автоэнкодер)
//! - Denoising Autoencoder (шумоподавляющий)
//! - Variational Autoencoder (вариационный)
//! - Conditional Autoencoder (условный)

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Функции активации
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Activation {
    /// ReLU: max(0, x)
    ReLU,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Tanh,
    /// Leaky ReLU: max(0.01*x, x)
    LeakyReLU,
    /// Линейная: x
    Linear,
}

impl Activation {
    /// Применяет функцию активации
    pub fn forward(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            Activation::Linear => x,
        }
    }

    /// Вычисляет производную для обратного распространения
    pub fn backward(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Sigmoid => {
                let s = self.forward(x);
                s * (1.0 - s)
            }
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::LeakyReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            Activation::Linear => 1.0,
        }
    }
}

/// Полносвязный слой нейронной сети
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLayer {
    /// Матрица весов (input_size x output_size)
    pub weights: Array2<f64>,
    /// Вектор смещений
    pub biases: Array1<f64>,
    /// Функция активации
    pub activation: Activation,
    /// Последние входы (для обратного распространения)
    #[serde(skip)]
    last_input: Option<Array1<f64>>,
    /// Последние выходы до активации
    #[serde(skip)]
    last_z: Option<Array1<f64>>,
}

impl DenseLayer {
    /// Создает новый слой с инициализацией Xavier
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();
        let normal = Normal::new(0.0, scale).unwrap();
        let mut rng = rand::thread_rng();

        let weights = Array2::from_shape_fn((input_size, output_size), |_| normal.sample(&mut rng));
        let biases = Array1::zeros(output_size);

        Self {
            weights,
            biases,
            activation,
            last_input: None,
            last_z: None,
        }
    }

    /// Прямой проход через слой
    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        self.last_input = Some(input.clone());

        // z = W^T * x + b
        let z = input.dot(&self.weights) + &self.biases;
        self.last_z = Some(z.clone());

        // a = activation(z)
        z.mapv(|x| self.activation.forward(x))
    }

    /// Обратный проход для обучения
    pub fn backward(&mut self, grad_output: &Array1<f64>, learning_rate: f64) -> Array1<f64> {
        let z = self.last_z.as_ref().expect("Forward pass required");
        let input = self.last_input.as_ref().expect("Forward pass required");

        // Градиент через активацию
        let activation_grad = z.mapv(|x| self.activation.backward(x));
        let delta = grad_output * &activation_grad;

        // Градиент по весам: dL/dW = x^T * delta
        let grad_weights = outer_product(input, &delta);

        // Градиент по входу: dL/dx = W * delta
        let grad_input = self.weights.dot(&delta);

        // Обновление весов и смещений
        self.weights = &self.weights - learning_rate * &grad_weights;
        self.biases = &self.biases - learning_rate * &delta;

        grad_input
    }
}

/// Внешнее произведение двух векторов
fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let rows = a.len();
    let cols = b.len();
    Array2::from_shape_fn((rows, cols), |(i, j)| a[i] * b[j])
}

/// Базовый автоэнкодер
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Autoencoder {
    /// Слои энкодера
    encoder_layers: Vec<DenseLayer>,
    /// Слои декодера
    decoder_layers: Vec<DenseLayer>,
    /// Размер входа
    input_size: usize,
    /// Размер скрытого представления
    latent_size: usize,
    /// История потерь при обучении
    #[serde(skip)]
    loss_history: Vec<f64>,
}

impl Autoencoder {
    /// Создает новый автоэнкодер
    ///
    /// # Аргументы
    /// * `input_size` - Размер входного вектора
    /// * `latent_size` - Размер скрытого представления (bottleneck)
    pub fn new(input_size: usize, latent_size: usize) -> Self {
        let encoder_layers = vec![DenseLayer::new(input_size, latent_size, Activation::ReLU)];

        let decoder_layers = vec![DenseLayer::new(latent_size, input_size, Activation::Sigmoid)];

        Self {
            encoder_layers,
            decoder_layers,
            input_size,
            latent_size,
            loss_history: Vec::new(),
        }
    }

    /// Создает глубокий автоэнкодер с несколькими слоями
    pub fn deep(input_size: usize, hidden_sizes: &[usize], latent_size: usize) -> Self {
        let mut encoder_layers = Vec::new();
        let mut decoder_layers = Vec::new();

        // Encoder layers
        let mut prev_size = input_size;
        for &hidden_size in hidden_sizes {
            encoder_layers.push(DenseLayer::new(prev_size, hidden_size, Activation::ReLU));
            prev_size = hidden_size;
        }
        encoder_layers.push(DenseLayer::new(prev_size, latent_size, Activation::ReLU));

        // Decoder layers (симметрично)
        prev_size = latent_size;
        for &hidden_size in hidden_sizes.iter().rev() {
            decoder_layers.push(DenseLayer::new(prev_size, hidden_size, Activation::ReLU));
            prev_size = hidden_size;
        }
        decoder_layers.push(DenseLayer::new(prev_size, input_size, Activation::Sigmoid));

        Self {
            encoder_layers,
            decoder_layers,
            input_size,
            latent_size,
            loss_history: Vec::new(),
        }
    }

    /// Кодирует входные данные в скрытое представление
    pub fn encode(&mut self, input: &Array1<f64>) -> Array1<f64> {
        let mut x = input.clone();
        for layer in &mut self.encoder_layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Декодирует скрытое представление обратно
    pub fn decode(&mut self, latent: &Array1<f64>) -> Array1<f64> {
        let mut x = latent.clone();
        for layer in &mut self.decoder_layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Прямой проход: encode + decode
    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        let latent = self.encode(input);
        self.decode(&latent)
    }

    /// Вычисляет MSE loss
    pub fn mse_loss(&self, predicted: &Array1<f64>, target: &Array1<f64>) -> f64 {
        let diff = predicted - target;
        diff.mapv(|x| x.powi(2)).sum() / diff.len() as f64
    }

    /// Обучает автоэнкодер на одном примере
    fn train_step(&mut self, input: &Array1<f64>, learning_rate: f64) -> f64 {
        // Forward pass
        let output = self.forward(input);

        // Compute loss
        let loss = self.mse_loss(&output, input);

        // Compute gradient of loss
        let mut grad = (&output - input) * 2.0 / input.len() as f64;

        // Backward pass through decoder
        for layer in self.decoder_layers.iter_mut().rev() {
            grad = layer.backward(&grad, learning_rate);
        }

        // Backward pass through encoder
        for layer in self.encoder_layers.iter_mut().rev() {
            grad = layer.backward(&grad, learning_rate);
        }

        loss
    }

    /// Обучает автоэнкодер на датасете
    ///
    /// # Аргументы
    /// * `data` - Матрица данных (samples x features)
    /// * `epochs` - Количество эпох
    /// * `learning_rate` - Скорость обучения
    pub fn fit(&mut self, data: &Array2<f64>, epochs: usize, learning_rate: f64) {
        self.loss_history.clear();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for sample in data.outer_iter() {
                let sample_vec = sample.to_owned();
                let loss = self.train_step(&sample_vec, learning_rate);
                epoch_loss += loss;
            }

            epoch_loss /= data.nrows() as f64;
            self.loss_history.push(epoch_loss);

            if epoch % 10 == 0 {
                log::info!("Epoch {}/{}: Loss = {:.6}", epoch + 1, epochs, epoch_loss);
            }
        }
    }

    /// Кодирует весь датасет
    pub fn transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        let rows = data.nrows();
        let mut encoded = Array2::zeros((rows, self.latent_size));

        for (i, sample) in data.outer_iter().enumerate() {
            let sample_vec = sample.to_owned();
            let latent = self.encode(&sample_vec);
            encoded.row_mut(i).assign(&latent);
        }

        encoded
    }

    /// Восстанавливает данные из скрытого представления
    pub fn inverse_transform(&mut self, encoded: &Array2<f64>) -> Array2<f64> {
        let rows = encoded.nrows();
        let mut decoded = Array2::zeros((rows, self.input_size));

        for (i, latent) in encoded.outer_iter().enumerate() {
            let latent_vec = latent.to_owned();
            let output = self.decode(&latent_vec);
            decoded.row_mut(i).assign(&output);
        }

        decoded
    }

    /// Возвращает историю потерь
    pub fn loss_history(&self) -> &[f64] {
        &self.loss_history
    }

    /// Вычисляет ошибку реконструкции для датасета
    pub fn reconstruction_error(&mut self, data: &Array2<f64>) -> f64 {
        let mut total_loss = 0.0;

        for sample in data.outer_iter() {
            let sample_vec = sample.to_owned();
            let output = self.forward(&sample_vec);
            total_loss += self.mse_loss(&output, &sample_vec);
        }

        total_loss / data.nrows() as f64
    }
}

/// Шумоподавляющий автоэнкодер (Denoising Autoencoder)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenoisingAutoencoder {
    /// Базовый автоэнкодер
    autoencoder: Autoencoder,
    /// Стандартное отклонение шума
    noise_std: f64,
}

impl DenoisingAutoencoder {
    /// Создает новый шумоподавляющий автоэнкодер
    pub fn new(input_size: usize, latent_size: usize, noise_std: f64) -> Self {
        Self {
            autoencoder: Autoencoder::new(input_size, latent_size),
            noise_std,
        }
    }

    /// Создает глубокий шумоподавляющий автоэнкодер
    pub fn deep(input_size: usize, hidden_sizes: &[usize], latent_size: usize, noise_std: f64) -> Self {
        Self {
            autoencoder: Autoencoder::deep(input_size, hidden_sizes, latent_size),
            noise_std,
        }
    }

    /// Добавляет шум к входным данным
    fn add_noise(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, self.noise_std).unwrap();
        input.mapv(|x| x + normal.sample(&mut rng))
    }

    /// Обучает на зашумленных данных
    pub fn fit(&mut self, data: &Array2<f64>, epochs: usize, learning_rate: f64) {
        self.autoencoder.loss_history.clear();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for sample in data.outer_iter() {
                let clean = sample.to_owned();
                let noisy = self.add_noise(&clean);

                // Forward с зашумленным входом
                let output = self.autoencoder.forward(&noisy);

                // Loss относительно чистых данных
                let loss = self.autoencoder.mse_loss(&output, &clean);
                epoch_loss += loss;

                // Backward pass
                let mut grad = (&output - &clean) * 2.0 / clean.len() as f64;

                for layer in self.autoencoder.decoder_layers.iter_mut().rev() {
                    grad = layer.backward(&grad, learning_rate);
                }
                for layer in self.autoencoder.encoder_layers.iter_mut().rev() {
                    grad = layer.backward(&grad, learning_rate);
                }
            }

            epoch_loss /= data.nrows() as f64;
            self.autoencoder.loss_history.push(epoch_loss);

            if epoch % 10 == 0 {
                log::info!("Epoch {}/{}: Loss = {:.6}", epoch + 1, epochs, epoch_loss);
            }
        }
    }

    /// Очищает зашумленные данные
    pub fn denoise(&mut self, noisy_data: &Array2<f64>) -> Array2<f64> {
        self.autoencoder.inverse_transform(&self.autoencoder.transform(noisy_data))
    }

    /// Возвращает ссылку на базовый автоэнкодер
    pub fn autoencoder(&mut self) -> &mut Autoencoder {
        &mut self.autoencoder
    }
}

/// Вариационный автоэнкодер (VAE)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariationalAutoencoder {
    /// Слои энкодера (до mu и log_var)
    encoder_layers: Vec<DenseLayer>,
    /// Слой для среднего (mu)
    mu_layer: DenseLayer,
    /// Слой для log variance
    log_var_layer: DenseLayer,
    /// Слои декодера
    decoder_layers: Vec<DenseLayer>,
    /// Размер входа
    input_size: usize,
    /// Размер скрытого пространства
    latent_size: usize,
    /// Вес KL-divergence
    kl_weight: f64,
    /// История потерь
    #[serde(skip)]
    loss_history: Vec<(f64, f64)>, // (reconstruction_loss, kl_loss)
}

impl VariationalAutoencoder {
    /// Создает новый VAE
    pub fn new(input_size: usize, hidden_size: usize, latent_size: usize) -> Self {
        let encoder_layers = vec![DenseLayer::new(input_size, hidden_size, Activation::ReLU)];

        let mu_layer = DenseLayer::new(hidden_size, latent_size, Activation::Linear);
        let log_var_layer = DenseLayer::new(hidden_size, latent_size, Activation::Linear);

        let decoder_layers = vec![
            DenseLayer::new(latent_size, hidden_size, Activation::ReLU),
            DenseLayer::new(hidden_size, input_size, Activation::Sigmoid),
        ];

        Self {
            encoder_layers,
            mu_layer,
            log_var_layer,
            decoder_layers,
            input_size,
            latent_size,
            kl_weight: 1.0,
            loss_history: Vec::new(),
        }
    }

    /// Устанавливает вес KL-divergence
    pub fn with_kl_weight(mut self, weight: f64) -> Self {
        self.kl_weight = weight;
        self
    }

    /// Reparameterization trick: z = mu + eps * exp(0.5 * log_var)
    fn reparameterize(&self, mu: &Array1<f64>, log_var: &Array1<f64>) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let std = log_var.mapv(|x| (0.5 * x).exp());
        let eps = Array1::from_shape_fn(mu.len(), |_| normal.sample(&mut rng));

        mu + &(std * eps)
    }

    /// Кодирует вход в параметры распределения
    pub fn encode(&mut self, input: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let mut x = input.clone();
        for layer in &mut self.encoder_layers {
            x = layer.forward(&x);
        }

        let mu = self.mu_layer.forward(&x);
        let log_var = self.log_var_layer.forward(&x);

        (mu, log_var)
    }

    /// Декодирует скрытое представление
    pub fn decode(&mut self, z: &Array1<f64>) -> Array1<f64> {
        let mut x = z.clone();
        for layer in &mut self.decoder_layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Прямой проход
    pub fn forward(&mut self, input: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let (mu, log_var) = self.encode(input);
        let z = self.reparameterize(&mu, &log_var);
        let output = self.decode(&z);
        (output, mu, log_var)
    }

    /// Вычисляет KL-divergence loss
    fn kl_divergence_loss(&self, mu: &Array1<f64>, log_var: &Array1<f64>) -> f64 {
        // KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        let kl = mu.mapv(|x| x.powi(2)) + log_var.mapv(|x| x.exp()) - log_var - 1.0;
        0.5 * kl.sum()
    }

    /// Обучает VAE
    pub fn fit(&mut self, data: &Array2<f64>, epochs: usize, learning_rate: f64) {
        self.loss_history.clear();

        for epoch in 0..epochs {
            let mut epoch_recon_loss = 0.0;
            let mut epoch_kl_loss = 0.0;

            for sample in data.outer_iter() {
                let input = sample.to_owned();
                let (output, mu, log_var) = self.forward(&input);

                // Losses
                let recon_loss = (&output - &input).mapv(|x| x.powi(2)).sum();
                let kl_loss = self.kl_divergence_loss(&mu, &log_var);

                epoch_recon_loss += recon_loss;
                epoch_kl_loss += kl_loss;

                // Backward pass (упрощенный)
                let grad = (&output - &input) * 2.0 / input.len() as f64;

                let mut g = grad.clone();
                for layer in self.decoder_layers.iter_mut().rev() {
                    g = layer.backward(&g, learning_rate);
                }

                // Обновляем энкодер (упрощенно)
                for layer in self.encoder_layers.iter_mut().rev() {
                    g = layer.backward(&g, learning_rate);
                }
            }

            let n = data.nrows() as f64;
            self.loss_history
                .push((epoch_recon_loss / n, epoch_kl_loss / n));

            if epoch % 10 == 0 {
                log::info!(
                    "Epoch {}/{}: Recon = {:.4}, KL = {:.4}",
                    epoch + 1,
                    epochs,
                    epoch_recon_loss / n,
                    epoch_kl_loss / n
                );
            }
        }
    }

    /// Генерирует новые данные из случайного z
    pub fn generate(&mut self, n_samples: usize) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut samples = Array2::zeros((n_samples, self.input_size));

        for i in 0..n_samples {
            let z = Array1::from_shape_fn(self.latent_size, |_| normal.sample(&mut rng));
            let output = self.decode(&z);
            samples.row_mut(i).assign(&output);
        }

        samples
    }

    /// Возвращает историю потерь
    pub fn loss_history(&self) -> &[(f64, f64)] {
        &self.loss_history
    }
}

/// Условный автоэнкодер (Conditional Autoencoder)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalAutoencoder {
    /// Слои энкодера (принимает input + conditions)
    encoder_layers: Vec<DenseLayer>,
    /// Слои декодера (принимает latent + conditions)
    decoder_layers: Vec<DenseLayer>,
    /// Размер входа
    input_size: usize,
    /// Размер условий
    condition_size: usize,
    /// Размер скрытого представления
    latent_size: usize,
    /// История потерь
    #[serde(skip)]
    loss_history: Vec<f64>,
}

impl ConditionalAutoencoder {
    /// Создает условный автоэнкодер
    ///
    /// # Аргументы
    /// * `input_size` - Размер входных данных
    /// * `condition_size` - Размер вектора условий
    /// * `hidden_size` - Размер скрытых слоев
    /// * `latent_size` - Размер латентного представления
    pub fn new(
        input_size: usize,
        condition_size: usize,
        hidden_size: usize,
        latent_size: usize,
    ) -> Self {
        // Encoder: [input, condition] -> hidden -> latent
        let encoder_layers = vec![
            DenseLayer::new(input_size + condition_size, hidden_size, Activation::ReLU),
            DenseLayer::new(hidden_size, latent_size, Activation::ReLU),
        ];

        // Decoder: [latent, condition] -> hidden -> output
        let decoder_layers = vec![
            DenseLayer::new(latent_size + condition_size, hidden_size, Activation::ReLU),
            DenseLayer::new(hidden_size, input_size, Activation::Sigmoid),
        ];

        Self {
            encoder_layers,
            decoder_layers,
            input_size,
            condition_size,
            latent_size,
            loss_history: Vec::new(),
        }
    }

    /// Объединяет входные данные с условиями
    fn concat(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(a.len() + b.len());
        for (i, &val) in a.iter().enumerate() {
            result[i] = val;
        }
        for (i, &val) in b.iter().enumerate() {
            result[a.len() + i] = val;
        }
        result
    }

    /// Кодирует вход с учетом условий
    pub fn encode(&mut self, input: &Array1<f64>, condition: &Array1<f64>) -> Array1<f64> {
        let mut x = Self::concat(input, condition);
        for layer in &mut self.encoder_layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Декодирует с учетом условий
    pub fn decode(&mut self, latent: &Array1<f64>, condition: &Array1<f64>) -> Array1<f64> {
        let mut x = Self::concat(latent, condition);
        for layer in &mut self.decoder_layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Прямой проход
    pub fn forward(&mut self, input: &Array1<f64>, condition: &Array1<f64>) -> Array1<f64> {
        let latent = self.encode(input, condition);
        self.decode(&latent, condition)
    }

    /// Обучает условный автоэнкодер
    ///
    /// # Аргументы
    /// * `data` - Входные данные (samples x input_features)
    /// * `conditions` - Условия (samples x condition_features)
    /// * `epochs` - Количество эпох
    /// * `learning_rate` - Скорость обучения
    pub fn fit(
        &mut self,
        data: &Array2<f64>,
        conditions: &Array2<f64>,
        epochs: usize,
        learning_rate: f64,
    ) {
        assert_eq!(data.nrows(), conditions.nrows());
        self.loss_history.clear();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for (sample, cond) in data.outer_iter().zip(conditions.outer_iter()) {
                let input = sample.to_owned();
                let condition = cond.to_owned();

                // Forward
                let output = self.forward(&input, &condition);

                // Loss
                let diff = &output - &input;
                let loss = diff.mapv(|x| x.powi(2)).sum() / input.len() as f64;
                epoch_loss += loss;

                // Backward through decoder
                let mut grad = diff * 2.0 / input.len() as f64;
                for layer in self.decoder_layers.iter_mut().rev() {
                    grad = layer.backward(&grad, learning_rate);
                }

                // Отсекаем градиент по условиям
                let grad_latent = Array1::from_iter(grad.iter().take(self.latent_size).copied());

                // Backward through encoder
                let mut grad = grad_latent;
                for layer in self.encoder_layers.iter_mut().rev() {
                    grad = layer.backward(&grad, learning_rate);
                }
            }

            epoch_loss /= data.nrows() as f64;
            self.loss_history.push(epoch_loss);

            if epoch % 10 == 0 {
                log::info!("Epoch {}/{}: Loss = {:.6}", epoch + 1, epochs, epoch_loss);
            }
        }
    }

    /// Трансформирует данные в скрытое пространство
    pub fn transform(&mut self, data: &Array2<f64>, conditions: &Array2<f64>) -> Array2<f64> {
        let rows = data.nrows();
        let mut encoded = Array2::zeros((rows, self.latent_size));

        for (i, (sample, cond)) in data.outer_iter().zip(conditions.outer_iter()).enumerate() {
            let latent = self.encode(&sample.to_owned(), &cond.to_owned());
            encoded.row_mut(i).assign(&latent);
        }

        encoded
    }

    /// Возвращает историю потерь
    pub fn loss_history(&self) -> &[f64] {
        &self.loss_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_functions() {
        assert_eq!(Activation::ReLU.forward(-1.0), 0.0);
        assert_eq!(Activation::ReLU.forward(1.0), 1.0);

        assert!((Activation::Sigmoid.forward(0.0) - 0.5).abs() < 1e-10);
        assert!((Activation::Tanh.forward(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_autoencoder_forward() {
        let mut ae = Autoencoder::new(10, 4);
        let input = Array1::from_vec(vec![0.5; 10]);
        let output = ae.forward(&input);
        assert_eq!(output.len(), 10);
    }

    #[test]
    fn test_deep_autoencoder() {
        let mut ae = Autoencoder::deep(16, &[8], 4);
        let data = Array2::from_shape_fn((100, 16), |_| rand::random::<f64>());
        ae.fit(&data, 10, 0.01);
        assert!(!ae.loss_history().is_empty());
    }

    #[test]
    fn test_conditional_autoencoder() {
        let mut cae = ConditionalAutoencoder::new(10, 3, 8, 4);
        let input = Array1::from_vec(vec![0.5; 10]);
        let condition = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let output = cae.forward(&input, &condition);
        assert_eq!(output.len(), 10);
    }
}
