//! Positional encoding computation for the Temporal-Positional model.

use chrono::{DateTime, Utc};

use super::constants::TEMPORAL_POSITIONAL_DIMENSION;

/// Compute the transformer-style positional encoding for a given timestamp.
///
/// The embedding uses the standard formula from "Attention Is All You Need":
/// - PE(pos, 2i) = sin(pos / base^(2i/d_model))
/// - PE(pos, 2i+1) = cos(pos / base^(2i/d_model))
///
/// # Arguments
/// * `timestamp` - The timestamp to encode
/// * `base` - Base frequency for positional encoding
/// * `d_model` - Model dimension (always 512)
///
/// # Returns
/// A 512-dimensional L2-normalized vector
pub fn compute_positional_encoding(
    timestamp: DateTime<Utc>,
    base: f32,
    d_model: usize,
) -> Vec<f32> {
    let pos = timestamp.timestamp() as f64; // Position is Unix timestamp
    let d_model_f64 = d_model as f64;
    let base_f64 = base as f64;

    let mut vector = Vec::with_capacity(TEMPORAL_POSITIONAL_DIMENSION);

    // Transformer PE formula:
    // PE(pos, 2i) = sin(pos / base^(2i/d_model))
    // PE(pos, 2i+1) = cos(pos / base^(2i/d_model))
    for i in 0..(d_model / 2) {
        let i_f64 = i as f64;
        let exponent = 2.0 * i_f64 / d_model_f64;
        let div_term = base_f64.powf(exponent);

        let angle = pos / div_term;
        let sin_val = angle.sin() as f32;
        let cos_val = angle.cos() as f32;

        vector.push(sin_val);
        vector.push(cos_val);
    }

    // L2 normalize
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for v in &mut vector {
            *v /= norm;
        }
    }

    vector
}
