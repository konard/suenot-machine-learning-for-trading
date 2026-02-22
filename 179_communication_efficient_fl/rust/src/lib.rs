use ndarray::Array1;
use rayon::prelude::*;
use bitvec::prelude::*;

pub struct CompressionEngine;

impl CompressionEngine {
    /// Performs 1-bit quantization (SignSGD) and packs result into a bit-vector.
    /// This reduces 32-bit floats to single bits for transmission.
    pub fn sign_pack(tensor: &Array1<f32>) -> (BitVec<u8, Msb0>, f32) {
        let length = tensor.len();
        let magnitude = tensor.mapv(|x| x.abs()).mean().unwrap_or(0.0);
        
        let mut bits = BitVec::<u8, Msb0>::repeat(false, length);
        
        let tensor_slice = tensor.as_slice().unwrap();
        
        // Parallel bit setting (Note: BitVec mutation is complex in parallel,
        // so we use a standard loop or a safer parallel approach)
        for (i, &val) in tensor_slice.iter().enumerate() {
            if val >= 0.0 {
                bits.set(i, true);
            }
        }

        (bits, magnitude)
    }

    /// Unpacks a bit-vector back into a 32-bit float tensor using the magnitude scale.
    pub fn sign_unpack(bits: &BitVec<u8, Msb0>, magnitude: f32) -> Array1<f32> {
        let length = bits.len();
        let mut decompressed = Array1::zeros(length);
        let dec_slice = decompressed.as_slice_mut().unwrap();

        dec_slice.par_iter_mut().enumerate().for_each(|(i, val)| {
            let bit = bits.get(i).map(|b| *b).unwrap_or(false);
            *val = if bit { magnitude } else { -magnitude };
        });

        decompressed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sign_compression_integrity() {
        let original = array![-1.5, 0.5, -0.1, 2.0];
        // Mean magnitude: (1.5 + 0.5 + 0.1 + 2.0) / 4 = 4.1 / 4 = 1.025
        
        let (packed, magnitude) = CompressionEngine::sign_pack(&original);
        
        assert_eq!(packed.len(), 4);
        assert!((magnitude - 1.025).abs() < 1e-6);
        
        let unpacked = CompressionEngine::sign_unpack(&packed, magnitude);
        
        // Signs should match: -1, +1, -1, +1
        assert!(unpacked[0] < 0.0);
        assert!(unpacked[1] > 0.0);
        assert!(unpacked[2] < 0.0);
        assert!(unpacked[3] > 0.0);
        
        // Magnitudes should be equal to the mean
        assert!((unpacked[0].abs() - 1.025).abs() < 1e-6);
    }
}
