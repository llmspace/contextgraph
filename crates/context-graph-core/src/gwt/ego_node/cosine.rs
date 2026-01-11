//! Cosine similarity computation for 13D purpose vectors
//!
//! Provides the cosine similarity function used in identity continuity calculation.

use super::types::COSINE_EPSILON;

/// Compute cosine similarity between two 13-dimensional purpose vectors.
///
/// # Algorithm
/// cos(v1, v2) = (v1 . v2) / (||v1|| x ||v2||)
///
/// # Arguments
/// * `v1` - First 13D purpose vector
/// * `v2` - Second 13D purpose vector
///
/// # Returns
/// * Cosine similarity in range [-1, 1]
/// * Returns 0.0 if either vector has near-zero magnitude (below COSINE_EPSILON)
///
/// # Constitution Reference
/// Used for computing cos(PV_t, PV_{t-1}) in IC = cos(PV_t, PV_{t-1}) x r(t)
pub fn cosine_similarity_13d(v1: &[f32; 13], v2: &[f32; 13]) -> f32 {
    // Compute dot product: v1 . v2 = sum(v1_i x v2_i)
    let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();

    // Compute magnitudes: ||v|| = sqrt(sum(v_i^2))
    let magnitude_v1: f32 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let magnitude_v2: f32 = v2.iter().map(|a| a * a).sum::<f32>().sqrt();

    // Handle near-zero magnitude vectors (prevents division by zero)
    // Per spec: return 0.0 for degenerate cases
    if magnitude_v1 < COSINE_EPSILON || magnitude_v2 < COSINE_EPSILON {
        return 0.0;
    }

    // Compute cosine similarity and clamp to valid range [-1, 1]
    // Clamping handles floating point errors that could produce values like 1.0000001
    let similarity = dot_product / (magnitude_v1 * magnitude_v2);
    similarity.clamp(-1.0, 1.0)
}
