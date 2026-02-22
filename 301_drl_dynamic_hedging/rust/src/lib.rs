use ndarray::{Array1, Array2, Axis};

/// Activation Functions matching PyTorch
fn silu(x: f32) -> f32 {
    x * (1.0 / (1.0 + (-x).exp()))
}

fn tanh(x: f32) -> f32 {
    x.tanh()
}

/// A highly optimized Rust implementation of the PyTorch PPO Actor-Mean Network.
/// This executes the forward pass dynamically in microseconds without needing a Python runtime.
pub struct PpoActorInference {
    // layer 0: State(4) -> Hidden(64)
    l0_weights: Array2<f32>, // Shape: (64, 4)
    l0_bias: Array1<f32>,    // Shape: (64,)

    // layer 2: Hidden(64) -> Hidden(64)
    l2_weights: Array2<f32>, // Shape: (64, 64)
    l2_bias: Array1<f32>,    // Shape: (64,)

    // layer 4: Hidden(64) -> Action(1)
    l4_weights: Array2<f32>, // Shape: (1, 64)
    l4_bias: Array1<f32>,    // Shape: (1,)
}

impl PpoActorInference {
    /// Initializes the network using explicit weight matrices (normally loaded from .pth via safetensors)
    pub fn new(
        l0_w: Vec<f32>, l0_b: Vec<f32>,
        l2_w: Vec<f32>, l2_b: Vec<f32>,
        l4_w: Vec<f32>, l4_b: Vec<f32>,
    ) -> Self {
        PpoActorInference {
            l0_weights: Array2::from_shape_vec((64, 4), l0_w).expect("Mismatched weights L0"),
            l0_bias: Array1::from_vec(l0_b),
            
            l2_weights: Array2::from_shape_vec((64, 64), l2_w).expect("Mismatched weights L2"),
            l2_bias: Array1::from_vec(l2_b),
            
            l4_weights: Array2::from_shape_vec((1, 64), l4_w).expect("Mismatched weights L4"),
            l4_bias: Array1::from_vec(l4_b),
        }
    }

    /// Primary Inference method.
    /// Takes the 4-dimensional market state and returns the optimal continuous hedge ratio [-1.0, 1.0].
    pub fn predict_hedge_ratio(&self, state: &[f32; 4]) -> f32 {
        let x = Array1::from_vec(state.to_vec());

        // Forward Pass Layer 1: Linear + SiLU
        let mut h1 = self.l0_weights.dot(&x) + &self.l0_bias;
        h1.mapv_inplace(silu);

        // Forward Pass Layer 2: Linear + SiLU
        let mut h2 = self.l2_weights.dot(&h1) + &self.l2_bias;
        h2.mapv_inplace(silu);

        // Forward Pass Layer 3: Linear + Tanh
        let mut out = self.l4_weights.dot(&h2) + &self.l4_bias;
        out.mapv_inplace(tanh);

        // The final shape is (1,), so we return the single f32
        out[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppo_inference_forward_pass_bounds() {
        // Create mock weights replicating the structure of the PyTorch PPO Actor
        let l0_w = vec![0.1; 64 * 4];
        let l0_b = vec![0.0; 64];
        let l2_w = vec![0.05; 64 * 64];
        let l2_b = vec![0.0; 64];
        let l4_w = vec![0.1; 1 * 64];
        let l4_b = vec![0.0; 1];

        let agent = PpoActorInference::new(l0_w, l0_b, l2_w, l2_b, l4_w, l4_b);

        // Mock State: [Price_Norm(1.0), LogReturn(0.0), CurrentHedge(0.0), TimeLeft(0.5)]
        let state = [1.0, 0.0, 0.0, 0.5];
        let hedge_action = agent.predict_hedge_ratio(&state);

        println!("Calculated Target Hedge Ratio: {}", hedge_action);
        
        // Assert the hedge ratio is mathematically bounded by the Tanh activation [-1.0, 1.0]
        assert!(hedge_action >= -1.0 && hedge_action <= 1.0);
    }
}
