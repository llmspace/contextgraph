//! Mathematical helper functions for vector operations.
//!
//! These functions are optimized for the Hopfield network cache lookups
//! and must be fast enough to meet the <100us latency budget.

// ============================================================
// Helper Functions
// ============================================================

/// Compute dot product of two vectors.
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0f32;

    // Unroll loop for better performance
    let mut i = 0;
    while i + 4 <= len {
        sum += a[i] * b[i];
        sum += a[i + 1] * b[i + 1];
        sum += a[i + 2] * b[i + 2];
        sum += a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

/// Compute L2 norm of a vector.
#[inline]
pub fn vector_norm(v: &[f32]) -> f32 {
    dot_product_f32(v, v).sqrt()
}

/// Normalize a vector in place.
#[inline]
pub fn normalize_vector(v: &mut [f32]) {
    let norm = vector_norm(v);
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product_f32(&a, &b);
        assert!((result - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_vector_norm() {
        let v = vec![3.0, 4.0];
        let norm = vector_norm(&v);
        assert!((norm - 5.0).abs() < 1e-6); // sqrt(9 + 16) = 5
    }

    #[test]
    fn test_normalize_vector() {
        let mut v = vec![3.0, 4.0];
        normalize_vector(&mut v);
        let norm = vector_norm(&v);
        assert!((norm - 1.0).abs() < 1e-6); // Should be unit vector
    }

    #[test]
    fn test_zero_vector_normalize() {
        let mut v = vec![0.0; 10];
        normalize_vector(&mut v); // Should not panic
                                  // Zero vector stays zero (no division by zero)
        assert!(v.iter().all(|&x| x.abs() < 1e-9));
    }
}
