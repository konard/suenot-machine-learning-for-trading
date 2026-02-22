use ndarray::Array1;
use rayon::prelude::*;
use bitvec::prelude::*;

pub struct TGCEngine;

impl TGCEngine {
    /// Ternary Quantization: maps floats to {-1, 0, +1}.
    pub fn ternary_quantize(tensor: &Array1<f32>, k: f32) -> (Array1<i8>, f32) {
        let std = tensor.std(0.0);
        let gamma = k * std;
        
        let mut quantized = Array1::zeros(tensor.len());
        let mut sum_abs = 0.0;
        let mut count = 0;

        let tensor_slice = tensor.as_slice().unwrap();
        let quant_slice = quantized.as_slice_mut().unwrap();

        for (i, &val) in tensor_slice.iter().enumerate() {
            if val > gamma {
                quant_slice[i] = 1;
                sum_abs += val.abs();
                count += 1;
            } else if val < -gamma {
                quant_slice[i] = -1;
                sum_abs += val.abs();
                count += 1;
            } else {
                quant_slice[i] = 0;
            }
        }

        let scale = if count > 0 { sum_abs / (count as f32) } else { 1.0 };
        (quantized, scale)
    }

    /// Packs ternary values (-1, 0, 1) into a 2-bit per element format.
    /// Mapping: 0 -> 00, 1 -> 01, -1 -> 10
    pub fn pack_2bit(ternary: &Array1<i8>) -> BitVec<u8, Msb0> {
        let mut bits = BitVec::<u8, Msb0>::repeat(false, ternary.len() * 2);
        
        for (i, &val) in ternary.iter().enumerate() {
            let base = i * 2;
            match val {
                1 => bits.set(base + 1, true),   // 01
                -1 => bits.set(base, true),      // 10
                _ => {}                          // 00
            }
        }
        bits
    }

    /// Unpacks 2-bit representation back to floats using the scale.
    pub fn unpack_2bit(bits: &BitVec<u8, Msb0>, scale: f32) -> Array1<f32> {
        let num_elements = bits.len() / 2;
        let mut unpacked = Array1::zeros(num_elements);
        let up_slice = unpacked.as_slice_mut().unwrap();

        up_slice.par_iter_mut().enumerate().for_each(|(i, val)| {
            let base = i * 2;
            let b0 = bits.get(base).map(|b| *b).unwrap_or(false);
            let b1 = bits.get(base + 1).map(|b| *b).unwrap_or(false);
            
            *val = match (b0, b1) {
                (false, true) => scale,   // 01 -> +1
                (true, false) => -scale,  // 10 -> -1
                _ => 0.0,                 // 00 -> 0
            };
        });

        unpacked
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_ternary_compression_cycle() {
        let original = array![-1.0, 0.0, 0.5, 2.0];
        // std is approx 1.1, with k=0.5 gamma=0.55
        // Expected ternary: [-1, 0, 0, 1]
        
        let (ternary, scale) = TGCEngine::ternary_quantize(&original, 0.5);
        assert_eq!(ternary[0], -1);
        assert_eq!(ternary[1], 0);
        assert_eq!(ternary[2], 0);
        assert_eq!(ternary[3], 1);
        
        let packed = TGCEngine::pack_2bit(&ternary);
        assert_eq!(packed.len(), 8);
        
        let unpacked = TGCEngine::unpack_2bit(&packed, scale);
        assert!(unpacked[0] < 0.0);
        assert_eq!(unpacked[1], 0.0);
        assert_eq!(unpacked[2], 0.0);
        assert!(unpacked[3] > 0.0);
    }
}
