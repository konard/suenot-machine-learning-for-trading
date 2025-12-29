//! Loss functions for GAN training
//!
//! Implements Binary Cross Entropy losses for generator and discriminator.

use tch::Tensor;

/// Generator loss: -log(D(G(z)))
///
/// The generator wants the discriminator to output 1 (real) for fake samples.
/// We use the negative log likelihood with ones as targets.
///
/// # Arguments
///
/// * `fake_output` - Discriminator output on generated samples (logits)
///
/// # Returns
///
/// Scalar loss tensor
pub fn generator_loss(fake_output: &Tensor) -> Tensor {
    // BCEWithLogitsLoss: targets are ones (we want D to classify fake as real)
    let targets = Tensor::ones_like(fake_output);
    fake_output.binary_cross_entropy_with_logits::<Tensor>(
        &targets,
        None,
        None,
        tch::Reduction::Mean,
    )
}

/// Discriminator loss: -log(D(x)) - log(1-D(G(z)))
///
/// The discriminator wants to output 1 for real samples and 0 for fake samples.
///
/// # Arguments
///
/// * `real_output` - Discriminator output on real samples (logits)
/// * `fake_output` - Discriminator output on generated samples (logits)
///
/// # Returns
///
/// Scalar loss tensor
pub fn discriminator_loss(real_output: &Tensor, fake_output: &Tensor) -> Tensor {
    // Loss on real samples (target = 1)
    let real_targets = Tensor::ones_like(real_output);
    let real_loss = real_output.binary_cross_entropy_with_logits::<Tensor>(
        &real_targets,
        None,
        None,
        tch::Reduction::Mean,
    );

    // Loss on fake samples (target = 0)
    let fake_targets = Tensor::zeros_like(fake_output);
    let fake_loss = fake_output.binary_cross_entropy_with_logits::<Tensor>(
        &fake_targets,
        None,
        None,
        tch::Reduction::Mean,
    );

    // Total discriminator loss
    real_loss + fake_loss
}

/// Wasserstein loss for generator (WGAN variant)
///
/// Generator loss: -E[D(G(z))]
pub fn generator_loss_wasserstein(fake_output: &Tensor) -> Tensor {
    -fake_output.mean(tch::Kind::Float)
}

/// Wasserstein loss for discriminator (WGAN variant)
///
/// Discriminator loss: E[D(G(z))] - E[D(x)]
pub fn discriminator_loss_wasserstein(real_output: &Tensor, fake_output: &Tensor) -> Tensor {
    fake_output.mean(tch::Kind::Float) - real_output.mean(tch::Kind::Float)
}

/// Label smoothing for improved training stability
///
/// Instead of using hard 1s and 0s, use smoothed labels:
/// - Real: 0.9 instead of 1.0
/// - Fake: 0.1 instead of 0.0
///
/// # Arguments
///
/// * `real_output` - Discriminator output on real samples
/// * `fake_output` - Discriminator output on fake samples
/// * `smooth_real` - Smoothed label for real (e.g., 0.9)
/// * `smooth_fake` - Smoothed label for fake (e.g., 0.1)
pub fn discriminator_loss_smoothed(
    real_output: &Tensor,
    fake_output: &Tensor,
    smooth_real: f64,
    smooth_fake: f64,
) -> Tensor {
    let real_targets = Tensor::full_like(real_output, smooth_real);
    let real_loss = real_output.binary_cross_entropy_with_logits::<Tensor>(
        &real_targets,
        None,
        None,
        tch::Reduction::Mean,
    );

    let fake_targets = Tensor::full_like(fake_output, smooth_fake);
    let fake_loss = fake_output.binary_cross_entropy_with_logits::<Tensor>(
        &fake_targets,
        None,
        None,
        tch::Reduction::Mean,
    );

    real_loss + fake_loss
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_generator_loss() {
        let fake_output = Tensor::randn([4, 1], (tch::Kind::Float, Device::Cpu));
        let loss = generator_loss(&fake_output);

        assert_eq!(loss.size(), vec![]);
        assert!(loss.double_value(&[]) > 0.0);
    }

    #[test]
    fn test_discriminator_loss() {
        let real_output = Tensor::randn([4, 1], (tch::Kind::Float, Device::Cpu));
        let fake_output = Tensor::randn([4, 1], (tch::Kind::Float, Device::Cpu));
        let loss = discriminator_loss(&real_output, &fake_output);

        assert_eq!(loss.size(), vec![]);
        assert!(loss.double_value(&[]) > 0.0);
    }

    #[test]
    fn test_perfect_discriminator() {
        // Perfect discriminator: high confidence on real, low on fake
        let real_output = Tensor::full(&[4, 1], 10.0, (tch::Kind::Float, Device::Cpu));
        let fake_output = Tensor::full(&[4, 1], -10.0, (tch::Kind::Float, Device::Cpu));
        let loss = discriminator_loss(&real_output, &fake_output);

        // Loss should be very small for perfect discriminator
        assert!(loss.double_value(&[]) < 0.1);
    }
}
