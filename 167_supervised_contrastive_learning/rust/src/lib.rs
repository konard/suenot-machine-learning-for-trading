use ndarray::{Array1, Array2, Axis};

pub struct ProjectionHead {
    pub w1: Array2<f32>, // (feature_dim, feature_dim)
    pub b1: Array1<f32>,
    pub w2: Array2<f32>, // (projection_dim, feature_dim)
    pub b2: Array1<f32>,
}

impl ProjectionHead {
    /// Projects features to the contrastive space and applies L2 normalization.
    pub fn project(&self, features: &Array2<f32>) -> Array2<f32> {
        let mut projections = Array2::zeros((features.nrows(), self.w2.nrows()));

        for (i, row) in features.axis_iter(Axis(0)).enumerate() {
            // Layer 1: Linear + ReLU
            let h1 = (self.w1.dot(&row) + &self.b1).mapv(|x| x.max(0.0));
            
            // Layer 2: Linear
            let mut z = self.w2.dot(&h1) + &self.b2;
            
            // L2 Normalization
            let norm = z.dot(&z).sqrt();
            if norm > 1e-9 {
                z.mapv_inplace(|x| x / norm);
            }
            
            projections.row_mut(i).assign(&z);
        }

        projections
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_projection_normalization() {
        let f_dim = 4;
        let p_dim = 2;
        
        // Simple scaling projection
        let head = ProjectionHead {
            w1: Array2::eye(f_dim),
            b1: Array1::zeros(f_dim),
            w2: array![[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
            b2: Array1::zeros(p_dim),
        };

        let features = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ];

        let z = head.project(&features);
        
        // Check L2 norm is 1.0
        for row in z.axis_iter(Axis(0)) {
            let norm = row.dot(&row).sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
        
        println!("Rust Projection output: {:?}", z);
    }
}
