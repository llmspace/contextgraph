//! Tensor loading utilities for BERT model weights.
//!
//! Provides helper functions for loading and validating tensors from VarBuilder.

use candle_core::Tensor;
use candle_nn::VarBuilder;

use super::error::ModelLoadError;

/// Get a tensor from VarBuilder with shape validation.
pub fn get_tensor(
    vb: &VarBuilder,
    name: &str,
    expected_shape: &[usize],
    model_path: &str,
) -> Result<Tensor, ModelLoadError> {
    let tensor = vb.get(expected_shape, name).map_err(|e| {
        // Check if it's a shape error or missing tensor
        let err_str = e.to_string();
        if err_str.contains("shape") || err_str.contains("Shape") {
            ModelLoadError::ShapeMismatch {
                weight_name: name.to_string(),
                expected: expected_shape.to_vec(),
                actual: vec![], // We don't have access to actual shape in error
            }
        } else {
            ModelLoadError::WeightNotFound {
                weight_name: name.to_string(),
                model_path: model_path.to_string(),
            }
        }
    })?;

    // Verify shape matches exactly
    let actual_shape: Vec<usize> = tensor.dims().to_vec();
    if actual_shape != expected_shape {
        return Err(ModelLoadError::ShapeMismatch {
            weight_name: name.to_string(),
            expected: expected_shape.to_vec(),
            actual: actual_shape,
        });
    }

    Ok(tensor)
}
