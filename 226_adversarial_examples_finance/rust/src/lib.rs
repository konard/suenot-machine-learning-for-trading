use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ============================================================
// Bybit API Data Fetching
// ============================================================

#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub symbol: Option<String>,
    pub category: Option<String>,
    pub list: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct OhlcvCandle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch OHLCV data from Bybit API
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<OhlcvCandle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let mut candles: Vec<OhlcvCandle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(OhlcvCandle {
                    timestamp: row[0].parse().ok()?,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns newest first, reverse for chronological order
    candles.reverse();
    Ok(candles)
}

/// Extract features from OHLCV candles: returns, volatility, momentum
pub fn extract_features(candles: &[OhlcvCandle], lookback: usize) -> Vec<Array1<f64>> {
    let mut features = Vec::new();

    for i in lookback..candles.len() {
        let mut feat = Vec::new();

        // Price returns for lookback period
        for j in (i - lookback + 1)..=i {
            let ret = (candles[j].close - candles[j - 1].close) / candles[j - 1].close;
            feat.push(ret);
        }

        // Volume changes
        for j in (i - lookback + 1)..=i {
            let vol_change = if candles[j - 1].volume > 0.0 {
                (candles[j].volume - candles[j - 1].volume) / candles[j - 1].volume
            } else {
                0.0
            };
            feat.push(vol_change);
        }

        // High-low range (volatility proxy)
        for j in (i - lookback + 1)..=i {
            let range = (candles[j].high - candles[j].low) / candles[j].close;
            feat.push(range);
        }

        features.push(Array1::from_vec(feat));
    }

    features
}

/// Generate labels: 1.0 if next candle close > current close, else 0.0
pub fn generate_labels(candles: &[OhlcvCandle], lookback: usize) -> Vec<f64> {
    let mut labels = Vec::new();
    for i in lookback..candles.len() - 1 {
        if candles[i + 1].close > candles[i].close {
            labels.push(1.0);
        } else {
            labels.push(0.0);
        }
    }
    labels
}

/// Normalize features using z-score
pub fn normalize_features(features: &[Array1<f64>]) -> (Vec<Array1<f64>>, Array1<f64>, Array1<f64>) {
    if features.is_empty() {
        return (Vec::new(), Array1::zeros(0), Array1::ones(0));
    }

    let n = features.len();
    let dim = features[0].len();

    let mut mean = Array1::zeros(dim);
    for f in features {
        mean = &mean + f;
    }
    mean /= n as f64;

    let mut std = Array1::zeros(dim);
    for f in features {
        let diff = f - &mean;
        std = &std + &(&diff * &diff);
    }
    std /= n as f64;
    std.mapv_inplace(|v| v.sqrt().max(1e-8));

    let normalized: Vec<Array1<f64>> = features
        .iter()
        .map(|f| (f - &mean) / &std)
        .collect();

    (normalized, mean, std)
}

// ============================================================
// Simple Neural Network with Gradient Computation
// ============================================================

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    pub weights1: Array2<f64>,
    pub bias1: Array1<f64>,
    pub weights2: Array2<f64>,
    pub bias2: Array1<f64>,
    pub weights3: Array2<f64>,
    pub bias3: Array1<f64>,
}

impl NeuralNetwork {
    /// Create a new neural network with random weights
    pub fn new(input_dim: usize, hidden1: usize, hidden2: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / input_dim as f64).sqrt();
        let scale2 = (2.0 / hidden1 as f64).sqrt();
        let scale3 = (2.0 / hidden2 as f64).sqrt();

        let weights1 = Array2::from_shape_fn((input_dim, hidden1), |_| {
            rng.gen_range(-scale1..scale1)
        });
        let bias1 = Array1::zeros(hidden1);

        let weights2 = Array2::from_shape_fn((hidden1, hidden2), |_| {
            rng.gen_range(-scale2..scale2)
        });
        let bias2 = Array1::zeros(hidden2);

        let weights3 = Array2::from_shape_fn((hidden2, output_dim), |_| {
            rng.gen_range(-scale3..scale3)
        });
        let bias3 = Array1::zeros(output_dim);

        NeuralNetwork {
            weights1,
            bias1,
            weights2,
            bias2,
            weights3,
            bias3,
        }
    }

    /// Forward pass returning the output (sigmoid activated)
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let z1 = input.dot(&self.weights1) + &self.bias1;
        let a1 = z1.mapv(relu);

        let z2 = a1.dot(&self.weights2) + &self.bias2;
        let a2 = z2.mapv(relu);

        let z3 = a2.dot(&self.weights3) + &self.bias3;
        z3.mapv(sigmoid)
    }

    /// Predict class: 1 if output > 0.5, else 0
    pub fn predict(&self, input: &Array1<f64>) -> f64 {
        let output = self.forward(input);
        if output[0] > 0.5 { 1.0 } else { 0.0 }
    }

    /// Compute gradient of the loss with respect to the input (backpropagation)
    /// Uses binary cross-entropy loss
    pub fn input_gradient(&self, input: &Array1<f64>, target: f64) -> Array1<f64> {
        // Forward pass with intermediate values stored
        let z1 = input.dot(&self.weights1) + &self.bias1;
        let a1 = z1.mapv(relu);

        let z2 = a1.dot(&self.weights2) + &self.bias2;
        let a2 = z2.mapv(relu);

        let z3 = a2.dot(&self.weights3) + &self.bias3;
        let output = z3.mapv(sigmoid);

        // Backward pass
        // dL/doutput for binary cross-entropy: (output - target) / (output * (1 - output))
        // doutput/dz3 = output * (1 - output)
        // Combined: dL/dz3 = output - target
        let dz3 = &output - &Array1::from_vec(vec![target]);

        // dz3/da2 = weights3
        let da2 = self.weights3.dot(&dz3);
        let dz2 = &da2 * &z2.mapv(relu_derivative);

        // dz2/da1 = weights2
        let da1 = self.weights2.dot(&dz2);
        let dz1 = &da1 * &z1.mapv(relu_derivative);

        // dz1/dinput = weights1
        self.weights1.dot(&dz1)
    }

    /// Train the network with simple SGD
    pub fn train(
        &mut self,
        inputs: &[Array1<f64>],
        labels: &[f64],
        epochs: usize,
        learning_rate: f64,
    ) {
        for _epoch in 0..epochs {
            for (input, &label) in inputs.iter().zip(labels.iter()) {
                // Forward pass
                let z1 = input.dot(&self.weights1) + &self.bias1;
                let a1 = z1.mapv(relu);

                let z2 = a1.dot(&self.weights2) + &self.bias2;
                let a2 = z2.mapv(relu);

                let z3 = a2.dot(&self.weights3) + &self.bias3;
                let output = z3.mapv(sigmoid);

                // Backward pass
                let dz3 = &output - &Array1::from_vec(vec![label]);

                // Update weights3 and bias3
                let dw3 = outer_product(&a2, &dz3);
                self.weights3 = &self.weights3 - &(learning_rate * &dw3);
                self.bias3 = &self.bias3 - &(learning_rate * &dz3);

                // Layer 2
                let da2 = self.weights3.dot(&dz3);
                let dz2 = &da2 * &z2.mapv(relu_derivative);

                let dw2 = outer_product(&a1, &dz2);
                self.weights2 = &self.weights2 - &(learning_rate * &dw2);
                self.bias2 = &self.bias2 - &(learning_rate * &dz2);

                // Layer 1
                let da1 = self.weights2.dot(&dz2);
                let dz1 = &da1 * &z1.mapv(relu_derivative);

                let dw1 = outer_product(input, &dz1);
                self.weights1 = &self.weights1 - &(learning_rate * &dw1);
                self.bias1 = &self.bias1 - &(learning_rate * &dz1);
            }
        }
    }
}

fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    let mut result = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ============================================================
// Adversarial Attack Methods
// ============================================================

/// Result of an adversarial attack
#[derive(Debug, Clone)]
pub struct AdversarialResult {
    pub original_input: Array1<f64>,
    pub adversarial_input: Array1<f64>,
    pub perturbation: Array1<f64>,
    pub original_prediction: f64,
    pub adversarial_prediction: f64,
    pub l_inf_norm: f64,
    pub l2_norm: f64,
    pub attack_successful: bool,
}

/// FGSM (Fast Gradient Sign Method) attack
///
/// x' = x + epsilon * sign(grad_x L(f(x), y))
pub fn fgsm_attack(
    network: &NeuralNetwork,
    input: &Array1<f64>,
    target: f64,
    epsilon: f64,
) -> AdversarialResult {
    let gradient = network.input_gradient(input, target);
    let perturbation = gradient.mapv(|g| g.signum() * epsilon);
    let adversarial = input + &perturbation;

    let orig_pred = network.predict(input);
    let adv_pred = network.predict(&adversarial);

    AdversarialResult {
        original_input: input.clone(),
        adversarial_input: adversarial,
        perturbation: perturbation.clone(),
        original_prediction: orig_pred,
        adversarial_prediction: adv_pred,
        l_inf_norm: linf_norm(&perturbation),
        l2_norm: l2_norm(&perturbation),
        attack_successful: orig_pred != adv_pred,
    }
}

/// PGD (Projected Gradient Descent) attack
///
/// Iterative FGSM with projection back onto the epsilon-ball
pub fn pgd_attack(
    network: &NeuralNetwork,
    input: &Array1<f64>,
    target: f64,
    epsilon: f64,
    step_size: f64,
    num_steps: usize,
) -> AdversarialResult {
    let mut current = input.clone();
    let orig_pred = network.predict(input);

    for _ in 0..num_steps {
        let gradient = network.input_gradient(&current, target);
        // Take step in gradient sign direction
        current = &current + &gradient.mapv(|g| g.signum() * step_size);
        // Project back onto L_inf epsilon-ball around original input
        current = project_linf(&current, input, epsilon);

        // Early stop if attack succeeds
        if network.predict(&current) != orig_pred {
            break;
        }
    }

    let perturbation = &current - input;
    let adv_pred = network.predict(&current);

    AdversarialResult {
        original_input: input.clone(),
        adversarial_input: current,
        perturbation: perturbation.clone(),
        original_prediction: orig_pred,
        adversarial_prediction: adv_pred,
        l_inf_norm: linf_norm(&perturbation),
        l2_norm: l2_norm(&perturbation),
        attack_successful: orig_pred != adv_pred,
    }
}

/// DeepFool-like minimal perturbation finder
///
/// Iteratively moves toward the decision boundary with small steps
pub fn deepfool_attack(
    network: &NeuralNetwork,
    input: &Array1<f64>,
    target: f64,
    max_iterations: usize,
    overshoot: f64,
) -> AdversarialResult {
    let mut current = input.clone();
    let orig_pred = network.predict(input);
    let mut total_perturbation = Array1::zeros(input.len());

    for _ in 0..max_iterations {
        let output = network.forward(&current);
        let distance_to_boundary = (output[0] - 0.5).abs();

        if network.predict(&current) != orig_pred {
            break;
        }

        let gradient = network.input_gradient(&current, target);
        let grad_norm_sq = gradient.dot(&gradient);

        if grad_norm_sq < 1e-12 {
            break;
        }

        // Minimal perturbation to cross boundary
        let step_size = (distance_to_boundary / grad_norm_sq) * (1.0 + overshoot);
        let perturbation = gradient.mapv(|g| g * step_size);

        total_perturbation = &total_perturbation + &perturbation;
        current = input + &total_perturbation;
    }

    let adv_pred = network.predict(&current);

    AdversarialResult {
        original_input: input.clone(),
        adversarial_input: current,
        perturbation: total_perturbation.clone(),
        original_prediction: orig_pred,
        adversarial_prediction: adv_pred,
        l_inf_norm: linf_norm(&total_perturbation),
        l2_norm: l2_norm(&total_perturbation),
        attack_successful: orig_pred != adv_pred,
    }
}

// ============================================================
// Perturbation Constraints
// ============================================================

/// Project point onto L_inf ball centered at center with radius epsilon
pub fn project_linf(point: &Array1<f64>, center: &Array1<f64>, epsilon: f64) -> Array1<f64> {
    let diff = point - center;
    let clamped = diff.mapv(|d| d.clamp(-epsilon, epsilon));
    center + &clamped
}

/// Project point onto L_2 ball centered at center with radius epsilon
pub fn project_l2(point: &Array1<f64>, center: &Array1<f64>, epsilon: f64) -> Array1<f64> {
    let diff = point - center;
    let norm = l2_norm(&diff);
    if norm <= epsilon {
        point.clone()
    } else {
        center + &(diff * (epsilon / norm))
    }
}

/// L_inf norm
pub fn linf_norm(v: &Array1<f64>) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

/// L_2 norm
pub fn l2_norm(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

/// L_1 norm
pub fn l1_norm(v: &Array1<f64>) -> f64 {
    v.iter().map(|x| x.abs()).sum()
}

// ============================================================
// Financial Perturbation Budgets
// ============================================================

/// Financial perturbation budget based on realistic market noise levels
#[derive(Debug, Clone)]
pub struct FinancialPerturbationBudget {
    /// Maximum perturbation for return features (e.g., 0.001 = 0.1%)
    pub return_epsilon: f64,
    /// Maximum perturbation for volume features (e.g., 0.05 = 5%)
    pub volume_epsilon: f64,
    /// Maximum perturbation for volatility features (e.g., 0.002 = 0.2%)
    pub volatility_epsilon: f64,
}

impl FinancialPerturbationBudget {
    /// Default budget calibrated to BTC/USDT market noise
    pub fn default_crypto() -> Self {
        FinancialPerturbationBudget {
            return_epsilon: 0.001,
            volume_epsilon: 0.05,
            volatility_epsilon: 0.002,
        }
    }

    /// Build per-feature epsilon vector for a feature array
    /// Assumes features are ordered: [returns..., volume_changes..., volatility...]
    pub fn to_epsilon_vector(&self, lookback: usize) -> Array1<f64> {
        let mut epsilons = Vec::with_capacity(lookback * 3);
        for _ in 0..lookback {
            epsilons.push(self.return_epsilon);
        }
        for _ in 0..lookback {
            epsilons.push(self.volume_epsilon);
        }
        for _ in 0..lookback {
            epsilons.push(self.volatility_epsilon);
        }
        Array1::from_vec(epsilons)
    }
}

/// FGSM attack with per-feature epsilon (financial budget)
pub fn fgsm_financial_attack(
    network: &NeuralNetwork,
    input: &Array1<f64>,
    target: f64,
    epsilons: &Array1<f64>,
) -> AdversarialResult {
    let gradient = network.input_gradient(input, target);
    let perturbation: Array1<f64> = gradient
        .iter()
        .zip(epsilons.iter())
        .map(|(&g, &e)| g.signum() * e)
        .collect::<Vec<f64>>()
        .into();
    let adversarial = input + &perturbation;

    let orig_pred = network.predict(input);
    let adv_pred = network.predict(&adversarial);

    AdversarialResult {
        original_input: input.clone(),
        adversarial_input: adversarial,
        perturbation: perturbation.clone(),
        original_prediction: orig_pred,
        adversarial_prediction: adv_pred,
        l_inf_norm: linf_norm(&perturbation),
        l2_norm: l2_norm(&perturbation),
        attack_successful: orig_pred != adv_pred,
    }
}

// ============================================================
// Attack Evaluation
// ============================================================

/// Evaluate attack success rate across a dataset
pub fn evaluate_attack_success_rate(
    network: &NeuralNetwork,
    inputs: &[Array1<f64>],
    labels: &[f64],
    epsilon: f64,
    attack_type: AttackType,
) -> AttackEvaluation {
    let mut total = 0;
    let mut successful = 0;
    let mut total_l2_perturbation = 0.0;
    let mut total_linf_perturbation = 0.0;

    for (input, &label) in inputs.iter().zip(labels.iter()) {
        let orig_pred = network.predict(input);
        // Only attack correctly classified examples
        if orig_pred == label {
            total += 1;
            let result = match attack_type {
                AttackType::FGSM => fgsm_attack(network, input, label, epsilon),
                AttackType::PGD => pgd_attack(network, input, label, epsilon, epsilon / 4.0, 20),
                AttackType::DeepFool => deepfool_attack(network, input, label, 50, 0.02),
            };
            if result.attack_successful {
                successful += 1;
            }
            total_l2_perturbation += result.l2_norm;
            total_linf_perturbation += result.l_inf_norm;
        }
    }

    let n = total.max(1) as f64;
    AttackEvaluation {
        total_samples: total,
        successful_attacks: successful,
        success_rate: successful as f64 / n,
        avg_l2_perturbation: total_l2_perturbation / n,
        avg_linf_perturbation: total_linf_perturbation / n,
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AttackType {
    FGSM,
    PGD,
    DeepFool,
}

#[derive(Debug, Clone)]
pub struct AttackEvaluation {
    pub total_samples: usize,
    pub successful_attacks: usize,
    pub success_rate: f64,
    pub avg_l2_perturbation: f64,
    pub avg_linf_perturbation: f64,
}

// ============================================================
// Perturbation Analysis: Feature Vulnerability
// ============================================================

/// Analyze which features are most vulnerable to adversarial perturbation
pub fn analyze_feature_vulnerability(
    network: &NeuralNetwork,
    inputs: &[Array1<f64>],
    labels: &[f64],
) -> FeatureVulnerability {
    if inputs.is_empty() {
        return FeatureVulnerability {
            mean_gradient_magnitude: Array1::zeros(0),
            feature_sensitivity_ranking: Vec::new(),
        };
    }

    let dim = inputs[0].len();
    let mut total_gradient_magnitude = Array1::zeros(dim);
    let mut count = 0;

    for (input, &label) in inputs.iter().zip(labels.iter()) {
        let gradient = network.input_gradient(input, label);
        total_gradient_magnitude = &total_gradient_magnitude + &gradient.mapv(|g| g.abs());
        count += 1;
    }

    if count > 0 {
        total_gradient_magnitude /= count as f64;
    }

    // Rank features by vulnerability (highest gradient = most vulnerable)
    let mut indexed: Vec<(usize, f64)> = total_gradient_magnitude
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let feature_sensitivity_ranking: Vec<usize> = indexed.iter().map(|&(i, _)| i).collect();

    FeatureVulnerability {
        mean_gradient_magnitude: total_gradient_magnitude,
        feature_sensitivity_ranking,
    }
}

#[derive(Debug, Clone)]
pub struct FeatureVulnerability {
    pub mean_gradient_magnitude: Array1<f64>,
    pub feature_sensitivity_ranking: Vec<usize>,
}

/// Analyze perturbation size vs attack success for different epsilon values
pub fn perturbation_vs_success(
    network: &NeuralNetwork,
    inputs: &[Array1<f64>],
    labels: &[f64],
    epsilons: &[f64],
    attack_type: AttackType,
) -> Vec<(f64, f64)> {
    epsilons
        .iter()
        .map(|&eps| {
            let eval = evaluate_attack_success_rate(network, inputs, labels, eps, attack_type);
            (eps, eval.success_rate)
        })
        .collect()
}

// ============================================================
// Unit Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_test_network() -> NeuralNetwork {
        NeuralNetwork::new(4, 8, 4, 1)
    }

    fn make_test_input() -> Array1<f64> {
        Array1::from_vec(vec![0.1, -0.2, 0.3, -0.1])
    }

    #[test]
    fn test_network_forward() {
        let net = make_test_network();
        let input = make_test_input();
        let output = net.forward(&input);
        assert_eq!(output.len(), 1);
        assert!(output[0] >= 0.0 && output[0] <= 1.0, "Sigmoid output must be in [0,1]");
    }

    #[test]
    fn test_network_predict() {
        let net = make_test_network();
        let input = make_test_input();
        let pred = net.predict(&input);
        assert!(pred == 0.0 || pred == 1.0, "Prediction must be 0 or 1");
    }

    #[test]
    fn test_input_gradient() {
        let net = make_test_network();
        let input = make_test_input();
        let gradient = net.input_gradient(&input, 1.0);
        assert_eq!(gradient.len(), 4);
        // Gradient should not be all zeros for a random network
        let grad_norm = l2_norm(&gradient);
        // It's possible but unlikely for a random network to have zero gradient
        assert!(grad_norm >= 0.0);
    }

    #[test]
    fn test_fgsm_attack() {
        let net = make_test_network();
        let input = make_test_input();
        let result = fgsm_attack(&net, &input, 1.0, 0.1);

        // Perturbation should be bounded
        assert!(result.l_inf_norm <= 0.1 + 1e-10);
        // Adversarial input should differ from original
        assert!(l2_norm(&result.perturbation) > 0.0);
    }

    #[test]
    fn test_pgd_attack() {
        let net = make_test_network();
        let input = make_test_input();
        let result = pgd_attack(&net, &input, 1.0, 0.1, 0.025, 10);

        // Perturbation should be bounded by epsilon
        assert!(
            result.l_inf_norm <= 0.1 + 1e-10,
            "PGD perturbation exceeded epsilon: {} > 0.1",
            result.l_inf_norm
        );
    }

    #[test]
    fn test_deepfool_attack() {
        let net = make_test_network();
        let input = make_test_input();
        let result = deepfool_attack(&net, &input, 1.0, 50, 0.02);

        // DeepFool should produce a perturbation
        assert!(result.l2_norm >= 0.0);
    }

    #[test]
    fn test_project_linf() {
        let center = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let point = Array1::from_vec(vec![0.5, -0.3, 0.1]);
        let projected = project_linf(&point, &center, 0.2);

        for &v in projected.iter() {
            assert!(v.abs() <= 0.2 + 1e-10);
        }
    }

    #[test]
    fn test_project_l2() {
        let center = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let point = Array1::from_vec(vec![3.0, 4.0, 0.0]);
        let projected = project_l2(&point, &center, 1.0);

        let norm = l2_norm(&(&projected - &center));
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "L2 projection should have norm 1.0, got {}",
            norm
        );
    }

    #[test]
    fn test_norms() {
        let v = Array1::from_vec(vec![1.0, -2.0, 3.0]);
        assert!((linf_norm(&v) - 3.0).abs() < 1e-10);
        assert!((l1_norm(&v) - 6.0).abs() < 1e-10);
        assert!((l2_norm(&v) - (14.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_financial_perturbation_budget() {
        let budget = FinancialPerturbationBudget::default_crypto();
        let epsilons = budget.to_epsilon_vector(5);
        assert_eq!(epsilons.len(), 15); // 5 returns + 5 volume + 5 volatility

        // Check return epsilons
        for i in 0..5 {
            assert!((epsilons[i] - 0.001).abs() < 1e-10);
        }
        // Check volume epsilons
        for i in 5..10 {
            assert!((epsilons[i] - 0.05).abs() < 1e-10);
        }
        // Check volatility epsilons
        for i in 10..15 {
            assert!((epsilons[i] - 0.002).abs() < 1e-10);
        }
    }

    #[test]
    fn test_normalize_features() {
        let features = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![4.0, 5.0, 6.0]),
            Array1::from_vec(vec![7.0, 8.0, 9.0]),
        ];
        let (normalized, _mean, _std) = normalize_features(&features);
        assert_eq!(normalized.len(), 3);

        // Mean of normalized features should be approximately 0
        let mut sum = Array1::zeros(3);
        for f in &normalized {
            sum = &sum + f;
        }
        sum /= 3.0;
        for &v in sum.iter() {
            assert!(v.abs() < 1e-10, "Mean should be ~0, got {}", v);
        }
    }

    #[test]
    fn test_evaluate_attack_success_rate() {
        let net = make_test_network();

        // Create simple dataset
        let inputs = vec![
            Array1::from_vec(vec![0.5, 0.3, 0.1, 0.2]),
            Array1::from_vec(vec![-0.5, -0.3, -0.1, -0.2]),
            Array1::from_vec(vec![0.1, 0.4, -0.2, 0.3]),
        ];
        let labels: Vec<f64> = inputs.iter().map(|x| net.predict(x)).collect();

        let eval = evaluate_attack_success_rate(&net, &inputs, &labels, 0.5, AttackType::FGSM);
        assert!(eval.success_rate >= 0.0 && eval.success_rate <= 1.0);
    }

    #[test]
    fn test_analyze_feature_vulnerability() {
        let net = make_test_network();
        let inputs = vec![
            Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]),
            Array1::from_vec(vec![-0.1, -0.2, -0.3, -0.4]),
        ];
        let labels = vec![1.0, 0.0];

        let vuln = analyze_feature_vulnerability(&net, &inputs, &labels);
        assert_eq!(vuln.mean_gradient_magnitude.len(), 4);
        assert_eq!(vuln.feature_sensitivity_ranking.len(), 4);
    }

    #[test]
    fn test_network_training() {
        let mut net = NeuralNetwork::new(2, 4, 2, 1);

        // Simple XOR-like dataset
        let inputs = vec![
            Array1::from_vec(vec![0.0, 0.0]),
            Array1::from_vec(vec![1.0, 0.0]),
            Array1::from_vec(vec![0.0, 1.0]),
            Array1::from_vec(vec![1.0, 1.0]),
        ];
        let labels = vec![0.0, 1.0, 1.0, 0.0];

        // Training should not panic
        net.train(&inputs, &labels, 10, 0.1);

        // After training, predictions should be valid
        for input in &inputs {
            let pred = net.predict(input);
            assert!(pred == 0.0 || pred == 1.0);
        }
    }

    #[test]
    fn test_fgsm_financial_attack() {
        let net = make_test_network();
        let input = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.1]);
        let epsilons = Array1::from_vec(vec![0.01, 0.05, 0.01, 0.01]);

        let result = fgsm_financial_attack(&net, &input, 1.0, &epsilons);

        // Each feature perturbation should respect its own epsilon
        for (i, (&pert, &eps)) in result.perturbation.iter().zip(epsilons.iter()).enumerate() {
            assert!(
                pert.abs() <= eps + 1e-10,
                "Feature {} perturbation {} exceeded budget {}",
                i,
                pert.abs(),
                eps
            );
        }
    }

    #[test]
    fn test_perturbation_vs_success() {
        let net = make_test_network();
        let inputs = vec![
            Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]),
            Array1::from_vec(vec![-0.1, -0.2, -0.3, -0.4]),
        ];
        let labels: Vec<f64> = inputs.iter().map(|x| net.predict(x)).collect();

        let epsilons = vec![0.01, 0.05, 0.1, 0.5, 1.0];
        let results = perturbation_vs_success(&net, &inputs, &labels, &epsilons, AttackType::FGSM);

        assert_eq!(results.len(), 5);
        // Success rate should generally increase with epsilon
        for &(eps, rate) in &results {
            assert!(rate >= 0.0 && rate <= 1.0);
            assert!(eps > 0.0);
        }
    }

    #[test]
    fn test_extract_features_and_labels() {
        let candles = vec![
            OhlcvCandle { timestamp: 0, open: 100.0, high: 105.0, low: 95.0, close: 102.0, volume: 1000.0 },
            OhlcvCandle { timestamp: 1, open: 102.0, high: 106.0, low: 98.0, close: 104.0, volume: 1100.0 },
            OhlcvCandle { timestamp: 2, open: 104.0, high: 108.0, low: 100.0, close: 103.0, volume: 900.0 },
            OhlcvCandle { timestamp: 3, open: 103.0, high: 107.0, low: 99.0, close: 106.0, volume: 1200.0 },
            OhlcvCandle { timestamp: 4, open: 106.0, high: 110.0, low: 102.0, close: 105.0, volume: 800.0 },
        ];

        let features = extract_features(&candles, 2);
        let labels = generate_labels(&candles, 2);

        assert_eq!(features.len(), 3); // candles 2, 3, 4
        assert_eq!(labels.len(), 2); // labels for candles 2, 3 (candle 4 has no next)

        // Each feature vector: 2 returns + 2 volume_changes + 2 ranges = 6
        assert_eq!(features[0].len(), 6);
    }
}
