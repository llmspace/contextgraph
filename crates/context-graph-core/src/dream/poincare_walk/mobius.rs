//! Mobius addition and geodesic operations in Poincare ball.
//!
//! Implements the core hyperbolic geometry operations for movement
//! and distance calculation in the Poincare ball model.

use super::config::PoincareBallConfig;
use super::math::{inner_product_64, norm_64, norm_squared_64, project_to_ball, validate_in_ball};

/// Mobius addition in the Poincare ball model.
///
/// Computes p ⊕ v in hyperbolic space using the Mobius addition formula:
///
/// ```text
/// p ⊕ v = ((1 + 2c<p,v> + c||v||²)p + (1 - c||p||²)v) / (1 + 2c<p,v> + c²||p||²||v||²)
/// ```
///
/// where c = |curvature| (typically 1.0).
///
/// # Arguments
/// * `p` - Current position in Poincare ball
/// * `v` - Velocity/displacement vector
/// * `config` - Ball configuration
///
/// # Returns
/// New position after Mobius addition, projected to stay in ball
///
/// # Panics
/// Panics if input point p is outside the ball.
pub fn mobius_add(
    p: &[f32; 64],
    v: &[f32; 64],
    config: &PoincareBallConfig,
) -> [f32; 64] {
    // Fail fast: validate input
    validate_in_ball(p, config, "mobius_add input p");

    let c = config.curvature.abs(); // typically 1.0
    let p_sq = norm_squared_64(p);
    let v_sq = norm_squared_64(v);
    let pv = inner_product_64(p, v);

    // Denominator: 1 + 2c<p,v> + c²||p||²||v||²
    let denom = 1.0 + 2.0 * c * pv + c * c * p_sq * v_sq;

    // Fail fast on degenerate case
    if denom.abs() < config.epsilon {
        panic!(
            "[POINCARE_WALK] Degenerate Mobius addition at {}:{}: denom = {:e}",
            file!(), line!(), denom
        );
    }

    // Numerator coefficients
    let coeff_p = 1.0 + 2.0 * c * pv + c * v_sq;
    let coeff_v = 1.0 - c * p_sq;

    // Compute result
    let mut result = [0.0f32; 64];
    for i in 0..64 {
        result[i] = (coeff_p * p[i] + coeff_v * v[i]) / denom;
    }

    // Project to ensure we stay in the ball
    project_to_ball(&mut result, config);

    result
}

/// Compute geodesic distance in the Poincare ball model.
///
/// Uses the formula:
/// ```text
/// d(p, q) = (1/√c) * acosh(1 + 2c||p - q||² / ((1 - c||p||²)(1 - c||q||²)))
/// ```
///
/// # Arguments
/// * `p` - First point
/// * `q` - Second point
/// * `config` - Ball configuration
///
/// # Returns
/// Geodesic distance (always >= 0)
///
/// # Panics
/// Panics if either input point is outside the ball.
pub fn geodesic_distance(
    p: &[f32; 64],
    q: &[f32; 64],
    config: &PoincareBallConfig,
) -> f32 {
    // Fail fast: validate inputs
    validate_in_ball(p, config, "geodesic_distance input p");
    validate_in_ball(q, config, "geodesic_distance input q");

    let c = config.curvature.abs();
    let p_sq = norm_squared_64(p);
    let q_sq = norm_squared_64(q);

    // ||p - q||²
    let diff_sq: f32 = p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi - qi).powi(2))
        .sum();

    // Denominators with epsilon guard
    let denom_p = (1.0 - c * p_sq).max(config.epsilon);
    let denom_q = (1.0 - c * q_sq).max(config.epsilon);

    // Argument to acosh
    let arg = 1.0 + 2.0 * c * diff_sq / (denom_p * denom_q);

    // acosh(x) = ln(x + sqrt(x² - 1)) for x >= 1
    if arg <= 1.0 {
        return 0.0;
    }

    let sqrt_c = c.sqrt();
    (arg + (arg * arg - 1.0).sqrt()).ln() / sqrt_c
}

/// Compute the Riemannian gradient direction from p toward q.
///
/// This gives the direction to move in Poincare ball to approach q.
///
/// # Arguments
/// * `p` - Current position
/// * `q` - Target position
/// * `config` - Ball configuration
///
/// # Returns
/// Direction vector (not normalized)
pub fn direction_toward(
    p: &[f32; 64],
    q: &[f32; 64],
    config: &PoincareBallConfig,
) -> [f32; 64] {
    // Use -p ⊕ q to get the direction
    let neg_p = {
        let mut neg = *p;
        for x in neg.iter_mut() {
            *x = -*x;
        }
        neg
    };

    mobius_add(&neg_p, q, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use crate::dream::poincare_walk::sampling::random_direction;

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
    fn test_mobius_add_origin_returns_v() {
        let config = PoincareBallConfig::default();
        let origin = [0.0f32; 64];
        let v = point_at_norm(0.5);

        let result = mobius_add(&origin, &v, &config);

        // Adding v to origin should give approximately v
        assert!((result[0] - 0.5).abs() < 1e-5);
        for i in 1..64 {
            assert!(result[i].abs() < 1e-6);
        }
    }

    #[test]
    fn test_mobius_add_stays_in_ball() {
        let config = PoincareBallConfig::default();
        let mut rng = make_rng();

        for _ in 0..20 {
            let mut p = random_direction(&mut rng);
            for x in p.iter_mut() {
                *x *= 0.5;
            }

            let mut v = random_direction(&mut rng);
            for x in v.iter_mut() {
                *x *= 0.3;
            }

            let result = mobius_add(&p, &v, &config);
            let norm = norm_64(&result);

            assert!(norm < config.max_norm,
                "result norm {} should be < max_norm {}", norm, config.max_norm);
        }
    }

    #[test]
    #[should_panic(expected = "[POINCARE_WALK] Point outside ball")]
    fn test_mobius_add_rejects_outside_ball() {
        let config = PoincareBallConfig::default();
        let p = point_at_norm(1.5); // Outside ball
        let v = point_at_norm(0.1);

        mobius_add(&p, &v, &config);
    }

    #[test]
    fn test_geodesic_distance_same_point_zero() {
        let config = PoincareBallConfig::default();
        let p = point_at_norm(0.5);

        let dist = geodesic_distance(&p, &p, &config);

        assert!(dist.abs() < 1e-6, "distance to self should be 0, got {}", dist);
    }

    #[test]
    fn test_geodesic_distance_symmetric() {
        let config = PoincareBallConfig::default();
        let mut rng = make_rng();

        for _ in 0..10 {
            let mut p = random_direction(&mut rng);
            for x in p.iter_mut() { *x *= 0.3; }

            let mut q = random_direction(&mut rng);
            for x in q.iter_mut() { *x *= 0.4; }

            let d1 = geodesic_distance(&p, &q, &config);
            let d2 = geodesic_distance(&q, &p, &config);

            assert!((d1 - d2).abs() < 1e-5,
                "distance should be symmetric: {} vs {}", d1, d2);
        }
    }

    #[test]
    fn test_geodesic_distance_triangle_inequality() {
        let config = PoincareBallConfig::default();
        let p = point_at_norm(0.3);
        let mut q = [0.0f32; 64];
        q[1] = 0.4;
        let mut r = [0.0f32; 64];
        r[2] = 0.5;

        let d_pq = geodesic_distance(&p, &q, &config);
        let d_qr = geodesic_distance(&q, &r, &config);
        let d_pr = geodesic_distance(&p, &r, &config);

        assert!(d_pr <= d_pq + d_qr + 1e-5,
            "triangle inequality violated: {} > {} + {}", d_pr, d_pq, d_qr);
    }

    #[test]
    #[should_panic(expected = "[POINCARE_WALK] Point outside ball")]
    fn test_geodesic_distance_rejects_outside_ball() {
        let config = PoincareBallConfig::default();
        let p = point_at_norm(1.5);
        let q = point_at_norm(0.5);

        geodesic_distance(&p, &q, &config);
    }

    #[test]
    fn test_direction_toward_reduces_distance() {
        let config = PoincareBallConfig::default();
        let p = point_at_norm(0.3);
        let mut q = [0.0f32; 64];
        q[1] = 0.4;

        let dir = direction_toward(&p, &q, &config);

        // Taking a step in this direction should reduce distance
        let mut scaled_dir = dir;
        for x in scaled_dir.iter_mut() { *x *= 0.01; }
        let p_new = mobius_add(&p, &scaled_dir, &config);

        let dist_before = geodesic_distance(&p, &q, &config);
        let dist_after = geodesic_distance(&p_new, &q, &config);

        assert!(dist_after < dist_before,
            "moving toward q should reduce distance: {} -> {}", dist_before, dist_after);
    }
}
