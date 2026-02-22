use ndarray::{Array1, Array2, Axis, ArrayView2};

/// A simplified GRU implementation for inference.
pub struct GruLayer {
    pub wr: Array2<f32>, // (3 * hidden_dim, input_dim)
    pub wh: Array2<f32>, // (3 * hidden_dim, hidden_dim)
    pub br: Array1<f32>, // (3 * hidden_dim)
    pub bh: Array1<f32>, // (3 * hidden_dim)
    pub hidden_dim: usize,
}

impl GruLayer {
    pub fn step(&self, input: &Array1<f32>, h_prev: &Array1<f32>) -> Array1<f32> {
        let dim = self.hidden_dim;

        // Gates: Reset (r), Update (z), New Candidate (n)
        // PyTorch GRU layout for gates is (z, r, n)
        
        let i_proj = self.wr.dot(input) + &self.br;
        let h_proj = self.wh.dot(h_prev) + &self.bh;

        // Extract z, r, n components
        let iz = i_proj.slice(ndarray::s![0..dim]);
        let ir = i_proj.slice(ndarray::s![dim..2*dim]);
        let in_gate = i_proj.slice(ndarray::s![2*dim..3*dim]);

        let hz = h_proj.slice(ndarray::s![0..dim]);
        let hr = h_proj.slice(ndarray::s![dim..2*dim]);
        let hn = h_proj.slice(ndarray::s![2*dim..3*dim]);

        // r = sigmoid(ir + hr)
        let r = (ir.to_owned() + &hr).mapv(sigmoid);
        // z = sigmoid(iz + hz)
        let z = (iz.to_owned() + &hz).mapv(sigmoid);
        // n = tanh(in + r * hn)
        let n = (in_gate.to_owned() + &(r * &hn)).mapv(|x| x.tanh());

        // h' = (1 - z) * n + z * h_prev
        let h_next = (Array1::from_elem(dim, 1.0) - &z) * n + &z * h_prev;

        h_next
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// CPC Inference Engine: 1D-CNN Encoder + GRU Layer.
pub struct CpcInference {
    pub gru: GruLayer,
    // Note: We skip the full CNN implementation for the walkthrough brevity 
    // and assume latent vectors are provided or CNN is simplified as a linear projection.
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gru_step_dimensions() {
        let hidden_dim = 4;
        let input_dim = 2;
        
        // Mock weights for a simple diagonal pass
        let gru = GruLayer {
            wr: Array2::zeros((3 * hidden_dim, input_dim)),
            wh: Array2::zeros((3 * hidden_dim, hidden_dim)),
            br: Array1::zeros(3 * hidden_dim),
            bh: Array1::zeros(3 * hidden_dim),
            hidden_dim,
        };

        let x = array![0.5, -0.5];
        let h = array![0.0, 0.0, 0.0, 0.0];

        let h_next = gru.step(&x, &h);
        
        assert_eq!(h_next.len(), hidden_dim);
        println!("GRU Step output: {:?}", h_next);
    }
}
