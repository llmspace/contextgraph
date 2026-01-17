//! Core math utilities for Poincare ball operations.
//!
//! Provides fundamental vector operations and ball validation.

use super::config::PoincareBallConfig;

/// Compute squared Euclidean norm of a 64D vector.
#[inline]
pub fn norm_squared_64(v: &[f32; 64]) -> f32 {
    v.iter().map(|&x| x * x).sum()
}

/// Compute Euclidean norm of a 64D vector.
#[inline]
pub fn norm_64(v: &[f32; 64]) -> f32 {
    norm_squared_64(v).sqrt()
}

/// Compute inner product of two 64D vectors.
#[inline]
pub fn inner_product_64(a: &[f32; 64], b: &[f32; 64]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Validate that a point is strictly inside the Poincare ball.
///
/// # Panics
/// Panics with detailed error if point is outside the ball.
#[inline]
pub fn validate_in_ball(point: &[f32; 64], config: &PoincareBallConfig, context: &str) {
    let norm = norm_64(point);
    if norm >= config.max_norm {
        panic!(
            "[POINCARE_WALK] Point outside ball at {}:{} ({}): norm = {:.6}, max = {:.6}",
            file!(),
            line!(),
            context,
            norm,
            config.max_norm
        );
    }
}

/// Project a point to stay strictly inside the Poincare ball.
///
/// If norm >= max_norm, rescales the point to have norm = max_norm - epsilon.
///
/// # Arguments
/// * `point` - Point to project (modified in place)
/// * `config` - Ball configuration with max_norm
///
/// # Returns
/// Whether projection was needed
pub fn project_to_ball(point: &mut [f32; 64], config: &PoincareBallConfig) -> bool {
    let norm = norm_64(point);

    if norm >= config.max_norm {
        let target_norm = config.max_norm - config.epsilon;
        let scale = target_norm / norm.max(config.epsilon);

        for x in point.iter_mut() {
            *x *= scale;
        }
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a point at given norm along first axis
    fn point_at_norm(norm: f32) -> [f32; 64] {
        let mut p = [0.0f32; 64];
        p[0] = norm;
        p
    }

    #[test]
    fn test_norm_squared_64_known_value() {
        // 64 elements of 0.125, squared = 0.015625, sum = 1.0
        let v = [0.125f32; 64];
        let expected = 64.0 * 0.015625; // 1.0
        let actual = norm_squared_64(&v);
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn test_norm_64_known_value() {
        let v = [0.125f32; 64];
        let expected = 1.0f32; // sqrt(1.0)
        let actual = norm_64(&v);
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn test_inner_product_64_orthogonal() {
        let mut a = [0.0f32; 64];
        let mut b = [0.0f32; 64];
        a[0] = 1.0;
        b[1] = 1.0;

        let result = inner_product_64(&a, &b);
        assert!(
            result.abs() < 1e-10,
            "orthogonal vectors should have 0 inner product"
        );
    }

    #[test]
    fn test_inner_product_64_parallel() {
        let v = [0.125f32; 64];
        let result = inner_product_64(&v, &v);
        let expected = norm_squared_64(&v);
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_project_to_ball_inside_unchanged() {
        let config = PoincareBallConfig::default();
        let mut point = point_at_norm(0.5);
        let original = point;

        let projected = project_to_ball(&mut point, &config);

        assert!(!projected, "should not need projection");
        assert_eq!(point, original, "point should be unchanged");
    }

    #[test]
    fn test_project_to_ball_outside_projected() {
        let config = PoincareBallConfig::default();
        let mut point = point_at_norm(1.5); // Outside ball

        let projected = project_to_ball(&mut point, &config);
        let new_norm = norm_64(&point);

        assert!(projected, "should need projection");
        assert!(
            new_norm < config.max_norm,
            "projected norm {} should be < max_norm {}",
            new_norm,
            config.max_norm
        );
    }

    #[test]
    fn test_project_to_ball_boundary_projected() {
        let config = PoincareBallConfig::default();
        let mut point = point_at_norm(0.99999); // At boundary

        let projected = project_to_ball(&mut point, &config);
        let new_norm = norm_64(&point);

        assert!(projected, "should need projection at boundary");
        assert!(new_norm < config.max_norm - config.epsilon / 2.0);
    }
}
