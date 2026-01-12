//! Distance metrics for clustering.
//!
//! Provides Euclidean distance functions for purpose vectors.

use crate::index::config::PURPOSE_VECTOR_DIM;

/// Compute squared Euclidean distance between two vectors.
///
/// Uses squared distance to avoid sqrt for comparison.
#[inline]
pub fn euclidean_distance_squared(
    a: &[f32; PURPOSE_VECTOR_DIM],
    b: &[f32; PURPOSE_VECTOR_DIM],
) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Compute Euclidean distance between two vectors.
#[inline]
pub fn euclidean_distance(a: &[f32; PURPOSE_VECTOR_DIM], b: &[f32; PURPOSE_VECTOR_DIM]) -> f32 {
    euclidean_distance_squared(a, b).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance_squared() {
        let a = [0.0; PURPOSE_VECTOR_DIM];
        let b = [1.0; PURPOSE_VECTOR_DIM];

        let dist_sq = euclidean_distance_squared(&a, &b);

        // Distance should be 13 (sum of 13 ones squared)
        assert!((dist_sq - 13.0).abs() < f32::EPSILON);

        println!("[VERIFIED] euclidean_distance_squared computes correctly");
    }

    #[test]
    fn test_euclidean_distance() {
        let a = [0.0; PURPOSE_VECTOR_DIM];
        let b = [1.0; PURPOSE_VECTOR_DIM];

        let dist = euclidean_distance(&a, &b);

        // Distance should be sqrt(13)
        let expected = (PURPOSE_VECTOR_DIM as f32).sqrt();
        assert!((dist - expected).abs() < 1e-6);

        println!("[VERIFIED] euclidean_distance computes correctly");
    }

    #[test]
    fn test_euclidean_distance_same_point() {
        let a = [0.5; PURPOSE_VECTOR_DIM];

        let dist = euclidean_distance(&a, &a);

        assert!(dist.abs() < f32::EPSILON);

        println!("[VERIFIED] euclidean_distance returns 0 for same point");
    }
}
