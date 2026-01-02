//! Softmax and probability smoothing utilities.
//!
//! Contains temperature-scaled softmax and Laplace smoothing functions
//! used by the gating network.

use crate::error::{EmbeddingError, EmbeddingResult};
use rand_distr::{Distribution, Normal};

/// Add Gaussian noise to logits for training exploration.
///
/// # Arguments
///
/// * `logits` - Mutable slice of logits to modify
/// * `noise_std` - Standard deviation of the Gaussian noise
pub fn add_gaussian_noise(logits: &mut [f32], noise_std: f32) {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0f64, noise_std as f64).unwrap();

    for logit in logits.iter_mut() {
        *logit += normal.sample(&mut rng) as f32;
    }
}

/// Temperature-scaled softmax.
///
/// Computes: softmax(logits / temperature)
///
/// Uses numerically stable implementation with max subtraction.
///
/// # Arguments
///
/// * `logits` - Input logits [batch_size * num_experts]
/// * `batch_size` - Number of samples in the batch
/// * `num_experts` - Number of experts
/// * `temperature` - Temperature for scaling (lower = sharper)
///
/// # Returns
///
/// Probabilities [batch_size * num_experts], each row sums to 1.0.
///
/// # Errors
///
/// Returns `EmbeddingError::FusionError` if softmax computation results in NaN.
pub fn softmax_with_temperature(
    logits: &[f32],
    batch_size: usize,
    num_experts: usize,
    temperature: f32,
) -> EmbeddingResult<Vec<f32>> {
    let mut probs = vec![0.0f32; batch_size * num_experts];

    for b in 0..batch_size {
        let offset = b * num_experts;
        let sample_logits = &logits[offset..offset + num_experts];

        // Apply temperature scaling
        let scaled: Vec<f32> = sample_logits
            .iter()
            .map(|&x| x / temperature)
            .collect();

        // Numerical stability: subtract max
        let max_logit = scaled
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(scaled - max)
        let exp_values: Vec<f32> = scaled.iter().map(|&x| (x - max_logit).exp()).collect();

        // Sum of exponentials
        let sum_exp: f32 = exp_values.iter().sum();

        // Avoid division by zero (shouldn't happen with valid inputs)
        if sum_exp == 0.0 || sum_exp.is_nan() {
            return Err(EmbeddingError::FusionError {
                message: format!(
                    "Softmax sum is invalid ({}). Check for NaN/Inf in logits.",
                    sum_exp
                ),
            });
        }

        // Normalize to probabilities
        for (i, exp_val) in exp_values.iter().enumerate() {
            probs[offset + i] = exp_val / sum_exp;
        }
    }

    Ok(probs)
}

/// Apply Laplace smoothing to probabilities.
///
/// Formula: (p + alpha) / (1 + alpha * K)
///
/// where K = num_experts
///
/// This prevents any expert from having exactly zero probability,
/// which improves gradient flow during training.
///
/// # Arguments
///
/// * `probs` - Mutable slice of probabilities to smooth
/// * `batch_size` - Number of samples in the batch
/// * `num_experts` - Number of experts
/// * `alpha` - Smoothing parameter (typically 0.01)
pub fn apply_laplace_smoothing(
    probs: &mut [f32],
    batch_size: usize,
    num_experts: usize,
    alpha: f32,
) {
    let k = num_experts as f32;
    let denominator = 1.0 + alpha * k;

    for b in 0..batch_size {
        let offset = b * num_experts;
        for i in 0..num_experts {
            probs[offset + i] = (probs[offset + i] + alpha) / denominator;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax_with_temperature(&logits, 1, 4, 1.0).unwrap();

        // Should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Higher logit should have higher probability
        assert!(probs[3] > probs[2]);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_temperature_effect() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];

        let probs_sharp = softmax_with_temperature(&logits, 1, 4, 0.5).unwrap();
        let probs_flat = softmax_with_temperature(&logits, 1, 4, 2.0).unwrap();

        // Sharp temperature should have higher max
        let max_sharp = probs_sharp.iter().copied().fold(0.0f32, f32::max);
        let max_flat = probs_flat.iter().copied().fold(0.0f32, f32::max);

        assert!(max_sharp > max_flat);
    }

    #[test]
    fn test_laplace_smoothing() {
        let mut probs = vec![0.9, 0.1, 0.0, 0.0];
        apply_laplace_smoothing(&mut probs, 1, 4, 0.01);

        // All should be > 0 now
        assert!(probs.iter().all(|&p| p > 0.0));

        // Should still approximately sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
