//! Функции активации для WaveNet

/// Сигмоида
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Производная сигмоиды
#[inline]
pub fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

/// Гиперболический тангенс
#[inline]
pub fn tanh_activation(x: f64) -> f64 {
    x.tanh()
}

/// Производная tanh
#[inline]
pub fn tanh_derivative(x: f64) -> f64 {
    let t = x.tanh();
    1.0 - t * t
}

/// ReLU
#[inline]
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Производная ReLU
#[inline]
pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

/// Leaky ReLU
#[inline]
pub fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { x } else { alpha * x }
}

/// Вентильная активация (Gated Activation)
/// Используется в WaveNet: tanh(x) * sigmoid(x)
#[inline]
pub fn gated_activation(filter_val: f64, gate_val: f64) -> f64 {
    filter_val.tanh() * sigmoid(gate_val)
}

/// Применить вентильную активацию к векторам
pub fn gated_activation_vec(filter: &[f64], gate: &[f64]) -> Vec<f64> {
    filter.iter()
        .zip(gate.iter())
        .map(|(f, g)| gated_activation(*f, *g))
        .collect()
}

/// Softmax для вектора
pub fn softmax(x: &[f64]) -> Vec<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = x.iter().map(|v| (v - max_val).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();
    exp_vals.iter().map(|v| v / sum).collect()
}

/// GLU (Gated Linear Unit)
pub fn glu(x: &[f64]) -> Vec<f64> {
    let half = x.len() / 2;
    let (a, b) = x.split_at(half);
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| ai * sigmoid(*bi))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_gated_activation() {
        let result = gated_activation(1.0, 1.0);
        assert!(result > 0.0 && result < 1.0);
    }

    #[test]
    fn test_softmax() {
        let x = vec![1.0, 2.0, 3.0];
        let result = softmax(&x);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
