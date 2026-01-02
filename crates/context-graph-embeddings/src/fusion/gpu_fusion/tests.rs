//! Tests for GPU basic components (LayerNorm, Linear, Activation).

#[cfg(test)]
#[cfg(feature = "candle")]
mod tests {
    use candle_core::{Device, Tensor};

    use crate::types::dimensions::TOTAL_CONCATENATED;

    use crate::fusion::gpu_fusion::{GpuActivation, GpuLayerNorm, GpuLinear};

    // Helper to get test device - GPU-only architecture
    fn test_device() -> Device {
        match Device::new_cuda(0) {
            Ok(device) => device,
            Err(e) => {
                panic!(
                    "GPU test device initialization failed: {}. \
                     This crate requires a CUDA GPU for all tests.",
                    e
                );
            }
        }
    }

    // =========================================================================
    // GPU LAYER NORM TESTS
    // =========================================================================

    #[test]
    fn test_gpu_layernorm_creation() {
        let device = test_device();
        let norm = GpuLayerNorm::new(1024, &device).unwrap();
        assert_eq!(norm.dim(), 1024);
    }

    #[test]
    fn test_gpu_layernorm_zero_dim_fails() {
        let device = test_device();
        let result = GpuLayerNorm::new(0, &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_layernorm_forward() {
        let device = test_device();
        let norm = GpuLayerNorm::new(4, &device).unwrap();

        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let output = norm.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 4]);

        // Mean should be ~0
        let mean: f32 = output.mean_all().unwrap().to_vec0().unwrap();
        assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);
    }

    #[test]
    fn test_gpu_layernorm_real_dimensions() {
        let device = test_device();
        let norm = GpuLayerNorm::new(TOTAL_CONCATENATED, &device).unwrap();
        assert_eq!(norm.dim(), 8320);

        let input_data: Vec<f32> = (0..4 * 8320).map(|i| (i as f32) * 0.001).collect();
        let input = Tensor::from_slice(&input_data, (4, 8320), &device).unwrap();
        let output = norm.forward(&input).unwrap();

        assert_eq!(output.dims(), &[4, 8320]);
    }

    // =========================================================================
    // GPU LINEAR TESTS
    // =========================================================================

    #[test]
    fn test_gpu_linear_creation() {
        let device = test_device();
        let linear = GpuLinear::new(8320, 8, &device).unwrap();
        assert_eq!(linear.in_features(), 8320);
        assert_eq!(linear.out_features(), 8);
    }

    #[test]
    fn test_gpu_linear_forward() {
        let device = test_device();
        let linear = GpuLinear::new(4, 2, &device).unwrap();

        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let output = linear.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 2]);
    }

    #[test]
    fn test_gpu_linear_real_gating_projection() {
        let device = test_device();
        let linear = GpuLinear::new(8320, 8, &device).unwrap();

        let input_data: Vec<f32> = (0..8320).map(|i| (i as f32) * 0.0001).collect();
        let input = Tensor::from_slice(&input_data, (1, 8320), &device).unwrap();
        let output = linear.forward(&input).unwrap();

        let dims = output.dims();
        assert!(dims.contains(&8), "Output should have 8 expert logits, got {:?}", dims);
    }

    // =========================================================================
    // GPU ACTIVATION TESTS
    // =========================================================================

    #[test]
    fn test_gpu_activation_gelu() {
        let device = test_device();
        let input = Tensor::from_slice(&[1.0f32, -1.0, 0.0], (1, 3), &device).unwrap();

        let act = GpuActivation::Gelu;
        let output = act.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 3]);
    }

    #[test]
    fn test_gpu_activation_relu() {
        let device = test_device();
        let input = Tensor::from_slice(&[1.0f32, -1.0, 0.0], (1, 3), &device).unwrap();

        let act = GpuActivation::Relu;
        let output = act.forward(&input).unwrap();

        let values: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        assert!(values[0] > 0.0);
        assert!(values[1] == 0.0);
        assert!(values[2] == 0.0);
    }

    #[test]
    fn test_gpu_activation_silu() {
        let device = test_device();
        let input = Tensor::from_slice(&[1.0f32, -1.0, 0.0], (1, 3), &device).unwrap();

        let act = GpuActivation::Silu;
        let output = act.forward(&input).unwrap();

        let values: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        assert!(values[0] > 0.5);
        assert!(values[1] < 0.0);
    }
}
