//! Random direction sampling and temperature-based selection.
//!
//! Implements direction sampling for random walks in the Poincare ball,
//! with softmax temperature control for exploration vs. exploitation.

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use super::config::PoincareBallConfig;
use super::math::norm_64;
use super::mobius::geodesic_distance;

/// Generate a random direction vector on the 64D unit sphere.
///
/// Uses the Gaussian method: sample from N(0,1) for each component,
/// then normalize to unit length.
///
/// # Arguments
/// * `rng` - Random number generator
///
/// # Returns
/// Unit vector in R^64
///
/// # Panics
/// Panics if RNG produces degenerate all-zero samples.
pub fn random_direction<R: Rng>(rng: &mut R) -> [f32; 64] {
    let normal = StandardNormal;
    let mut direction = [0.0f32; 64];

    for x in direction.iter_mut() {
        *x = normal.sample(rng);
    }

    // Normalize
    let norm = norm_64(&direction);
    if norm < 1e-10 {
        panic!(
            "[POINCARE_WALK] Degenerate random direction at {}:{}: norm = {:e}",
            file!(), line!(), norm
        );
    }

    for x in direction.iter_mut() {
        *x /= norm;
    }

    direction
}

/// Compute softmax with temperature.
///
/// P(i) = exp(score_i / T) / sum_j(exp(score_j / T))
///
/// Constitution: temperature = 2.0 (line 393)
///
/// # Arguments
/// * `scores` - Raw scores
/// * `temperature` - Temperature parameter (higher = more uniform)
///
/// # Returns
/// Probability distribution over scores
///
/// # Panics
/// Panics if scores is empty or temperature is invalid.
pub fn softmax_temperature(scores: &[f32], temperature: f32) -> Vec<f32> {
    if scores.is_empty() {
        panic!(
            "[POINCARE_WALK] Empty scores array at {}:{}",
            file!(), line!()
        );
    }

    if temperature <= 0.0 {
        panic!(
            "[POINCARE_WALK] Invalid temperature at {}:{}: expected > 0, got {:.6}",
            file!(), line!(), temperature
        );
    }

    // Scale by temperature
    let scaled: Vec<f32> = scores.iter().map(|&s| s / temperature).collect();

    // Find max for numerical stability (log-sum-exp trick)
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max)
    let exp_scores: Vec<f32> = scaled.iter().map(|&s| (s - max).exp()).collect();

    // Normalize
    let sum: f32 = exp_scores.iter().sum();
    if sum < 1e-10 {
        panic!(
            "[POINCARE_WALK] Softmax sum underflow at {}:{}: sum = {:e}",
            file!(), line!(), sum
        );
    }

    exp_scores.iter().map(|&e| e / sum).collect()
}

/// Sample multiple random directions and select one via softmax with temperature.
///
/// Higher temperature (> 1.0) makes selection more uniform (exploratory).
/// Lower temperature (< 1.0) makes selection more greedy toward high scores.
///
/// Constitution: temperature = 2.0 (line 393)
///
/// # Arguments
/// * `rng` - Random number generator
/// * `n_samples` - Number of directions to sample (must be > 0)
/// * `scores` - Optional scores for each direction (if None, uniform selection)
/// * `temperature` - Softmax temperature
///
/// # Returns
/// Selected direction vector
///
/// # Panics
/// Panics if n_samples is 0 or scores length doesn't match n_samples.
pub fn sample_direction_with_temperature<R: Rng>(
    rng: &mut R,
    n_samples: usize,
    scores: Option<&[f32]>,
    temperature: f32,
) -> [f32; 64] {
    if n_samples == 0 {
        panic!(
            "[POINCARE_WALK] n_samples must be > 0 at {}:{}",
            file!(), line!()
        );
    }

    // Validate scores length if provided
    if let Some(s) = scores {
        if s.len() != n_samples {
            panic!(
                "[POINCARE_WALK] Scores length mismatch at {}:{}: expected {}, got {}",
                file!(), line!(), n_samples, s.len()
            );
        }
    }

    // Generate candidate directions
    let candidates: Vec<[f32; 64]> = (0..n_samples)
        .map(|_| random_direction(rng))
        .collect();

    // Use provided scores or uniform
    let scores_vec: Vec<f32> = match scores {
        Some(s) => s.to_vec(),
        None => vec![1.0; n_samples],
    };

    // Apply softmax with temperature
    let probs = softmax_temperature(&scores_vec, temperature);

    // Sample from distribution
    let mut cumulative = 0.0;
    let threshold: f32 = rng.gen();

    for (i, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if threshold < cumulative {
            return candidates[i];
        }
    }

    // Return last candidate (floating point edge case)
    candidates[n_samples - 1]
}

/// Scale a direction vector by step size, respecting Poincare geometry.
///
/// In hyperbolic space, movement near the boundary requires smaller
/// Euclidean steps to achieve the same geodesic distance.
///
/// # Arguments
/// * `direction` - Unit direction vector
/// * `step_size` - Desired step size in Euclidean terms
/// * `current_norm` - Current position's norm
/// * `config` - Ball configuration
///
/// # Returns
/// Scaled velocity vector safe for Mobius addition
///
/// # Panics
/// Panics if direction is not unit length or current_norm >= max_norm.
pub fn scale_direction(
    direction: &[f32; 64],
    step_size: f32,
    current_norm: f32,
    config: &PoincareBallConfig,
) -> [f32; 64] {
    // Validate direction is unit length
    let dir_norm = norm_64(direction);
    if (dir_norm - 1.0).abs() > 1e-4 {
        panic!(
            "[POINCARE_WALK] Direction not unit length at {}:{}: norm = {:.6}",
            file!(), line!(), dir_norm
        );
    }

    // Validate current position is inside ball
    if current_norm >= config.max_norm {
        panic!(
            "[POINCARE_WALK] Current position outside ball at {}:{}: norm = {:.6}",
            file!(), line!(), current_norm
        );
    }

    // Near the boundary, we need smaller steps
    // Factor: (1 - ||p||²) / 2 scales appropriately
    let boundary_factor = ((1.0 - current_norm * current_norm) / 2.0).max(config.epsilon);
    let effective_step = step_size * boundary_factor;

    let mut result = *direction;
    for x in result.iter_mut() {
        *x *= effective_step;
    }

    result
}

/// Check if a point is far from all reference points (blind spot detection).
///
/// Constitution: semantic_leap >= 0.7 (line 394)
///
/// # Arguments
/// * `point` - Point to check
/// * `reference_points` - Set of reference points (visited nodes)
/// * `min_distance` - Minimum geodesic distance to be "far" (0.7 per constitution)
/// * `config` - Ball configuration
///
/// # Returns
/// True if point is far from all reference points (potential blind spot)
pub fn is_far_from_all(
    point: &[f32; 64],
    reference_points: &[[f32; 64]],
    min_distance: f32,
    config: &PoincareBallConfig,
) -> bool {
    for ref_point in reference_points {
        let dist = geodesic_distance(point, ref_point, config);
        if dist < min_distance {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Deterministic RNG for reproducible tests
    fn make_rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(42)
    }

    /// Create a point at given norm along first axis
    fn point_at_norm(norm: f32) -> [f32; 64] {
        let mut p = [0.0f32; 64];
        p[0] = norm;
        p
    }

    #[test]
    fn test_random_direction_unit_length() {
        let mut rng = make_rng();

        for _ in 0..20 {
            let dir = random_direction(&mut rng);
            let norm = norm_64(&dir);

            assert!((norm - 1.0).abs() < 1e-5,
                "direction should be unit length, got {}", norm);
        }
    }

    #[test]
    fn test_random_direction_reproducible() {
        let mut rng1 = make_rng();
        let mut rng2 = make_rng();

        let dir1 = random_direction(&mut rng1);
        let dir2 = random_direction(&mut rng2);

        assert_eq!(dir1, dir2, "same seed should produce same direction");
    }

    #[test]
    fn test_softmax_temperature_sums_to_one() {
        let scores = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax_temperature(&scores, 2.0); // Constitution temperature

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax should sum to 1, got {}", sum);
    }

    #[test]
    fn test_softmax_temperature_uniform_input() {
        let scores = vec![1.0, 1.0, 1.0, 1.0];
        let probs = softmax_temperature(&scores, 2.0);

        for p in &probs {
            assert!((*p - 0.25).abs() < 0.01, "uniform scores should give uniform probs");
        }
    }

    #[test]
    fn test_softmax_high_temp_more_uniform() {
        let scores = vec![1.0, 2.0, 3.0];

        let probs_high = softmax_temperature(&scores, 10.0);
        let probs_low = softmax_temperature(&scores, 0.1);

        let range_high = probs_high.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - probs_high.iter().cloned().fold(f32::INFINITY, f32::min);
        let range_low = probs_low.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - probs_low.iter().cloned().fold(f32::INFINITY, f32::min);

        assert!(range_high < range_low,
            "high temp range {} should be < low temp range {}", range_high, range_low);
    }

    #[test]
    #[should_panic(expected = "[POINCARE_WALK] Empty scores array")]
    fn test_softmax_rejects_empty() {
        let scores: Vec<f32> = vec![];
        softmax_temperature(&scores, 2.0);
    }

    #[test]
    #[should_panic(expected = "[POINCARE_WALK] Invalid temperature")]
    fn test_softmax_rejects_zero_temp() {
        let scores = vec![1.0, 2.0];
        softmax_temperature(&scores, 0.0);
    }

    #[test]
    fn test_sample_direction_returns_unit() {
        let mut rng = make_rng();
        let dir = sample_direction_with_temperature(&mut rng, 5, None, 2.0);
        let norm = norm_64(&dir);

        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    #[should_panic(expected = "[POINCARE_WALK] n_samples must be > 0")]
    fn test_sample_direction_rejects_zero_samples() {
        let mut rng = make_rng();
        sample_direction_with_temperature(&mut rng, 0, None, 2.0);
    }

    #[test]
    #[should_panic(expected = "[POINCARE_WALK] Scores length mismatch")]
    fn test_sample_direction_rejects_wrong_scores_length() {
        let mut rng = make_rng();
        let scores = vec![1.0, 2.0]; // 2 scores
        sample_direction_with_temperature(&mut rng, 5, Some(&scores), 2.0); // 5 samples
    }

    #[test]
    fn test_scale_direction_smaller_near_boundary() {
        let config = PoincareBallConfig::default();
        let dir = {
            let mut d = [0.0f32; 64];
            d[0] = 1.0;
            d
        };

        let scaled_origin = scale_direction(&dir, 0.1, 0.0, &config);
        let scaled_boundary = scale_direction(&dir, 0.1, 0.9, &config);

        let norm_origin = norm_64(&scaled_origin);
        let norm_boundary = norm_64(&scaled_boundary);

        assert!(norm_origin > norm_boundary,
            "step near origin ({}) should be larger than near boundary ({})",
            norm_origin, norm_boundary);
    }

    #[test]
    #[should_panic(expected = "[POINCARE_WALK] Direction not unit length")]
    fn test_scale_direction_rejects_non_unit() {
        let config = PoincareBallConfig::default();
        let dir = point_at_norm(0.5); // Not unit length
        scale_direction(&dir, 0.1, 0.0, &config);
    }

    #[test]
    fn test_is_far_from_all_empty_refs() {
        let config = PoincareBallConfig::default();
        let point = point_at_norm(0.5);
        let refs: Vec<[f32; 64]> = vec![];

        assert!(is_far_from_all(&point, &refs, 0.7, &config),
            "should be far from empty reference set");
    }

    #[test]
    fn test_is_far_from_all_close_ref() {
        let config = PoincareBallConfig::default();
        let point = point_at_norm(0.5);
        let ref_point = point_at_norm(0.51); // Very close
        let refs = vec![ref_point];

        assert!(!is_far_from_all(&point, &refs, 0.7, &config),
            "should NOT be far from close reference");
    }

    #[test]
    fn test_is_far_from_all_semantic_leap_threshold() {
        // Constitution: semantic_leap >= 0.7
        let config = PoincareBallConfig::default();
        let origin = [0.0f32; 64];

        // Create a reference point that's exactly at semantic_leap distance
        let mut ref_point = [0.0f32; 64];
        ref_point[0] = 0.6; // Creates meaningful distance from origin
        let refs = vec![ref_point];

        let dist = geodesic_distance(&origin, &ref_point, &config);

        // Point should be classified based on 0.7 threshold
        let result = is_far_from_all(&origin, &refs, 0.7, &config);
        assert_eq!(result, dist >= 0.7,
            "is_far_from_all result {} inconsistent with distance {} vs threshold 0.7",
            result, dist);
    }
}
