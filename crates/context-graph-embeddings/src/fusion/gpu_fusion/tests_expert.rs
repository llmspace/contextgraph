//! Tests for GPU Expert and Gating Network components.

#[cfg(test)]
#[cfg(feature = "candle")]
mod tests {
    use candle_core::{Device, Tensor};

    use crate::config::FusionConfig;
    use crate::types::dimensions::{FUSED_OUTPUT, TOTAL_CONCATENATED};

    use crate::fusion::gpu_fusion::{GpuExpert, GpuGatingNetwork};

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
    // GPU EXPERT TESTS
    // =========================================================================

    #[test]
    fn test_gpu_expert_creation() {
        let device = test_device();
        let expert = GpuExpert::new(
            0,
            TOTAL_CONCATENATED,
            4096,
            FUSED_OUTPUT,
            &device,
        )
        .unwrap();

        assert_eq!(expert.input_dim(), 8320);
        assert_eq!(expert.output_dim(), 1536);
    }

    #[test]
    fn test_gpu_expert_forward_real_dims() {
        let device = test_device();
        let expert = GpuExpert::new(
            0,
            TOTAL_CONCATENATED,
            4096,
            FUSED_OUTPUT,
            &device,
        )
        .unwrap();

        let input_data: Vec<f32> = (0..8320).map(|i| (i as f32) * 0.0001).collect();
        let input = Tensor::from_slice(&input_data, (1, 8320), &device).unwrap();
        let output = expert.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 1536], "Expert output should be [1, 1536]");
    }

    #[test]
    fn test_gpu_expert_forward_batch() {
        let device = test_device();
        let expert = GpuExpert::new(
            0,
            TOTAL_CONCATENATED,
            4096,
            FUSED_OUTPUT,
            &device,
        )
        .unwrap();

        let input_data: Vec<f32> = (0..8 * 8320).map(|i| (i as f32) * 0.00001).collect();
        let input = Tensor::from_slice(&input_data, (8, 8320), &device).unwrap();
        let output = expert.forward(&input).unwrap();

        assert_eq!(output.dims(), &[8, 1536], "Expert output should be [8, 1536]");
    }

    #[test]
    fn test_gpu_expert_parameter_count() {
        let device = test_device();
        let expert = GpuExpert::new(0, 8320, 4096, 1536, &device).unwrap();

        let params = expert.parameter_count();
        assert_eq!(params, 40_375_808, "Expert should have ~40M parameters");
    }

    // =========================================================================
    // GPU GATING NETWORK TESTS
    // =========================================================================

    #[test]
    fn test_gpu_gating_network_creation() {
        let device = test_device();
        let config = FusionConfig::default();
        let gating = GpuGatingNetwork::new(&config, &device).unwrap();

        assert_eq!(gating.input_dim(), 8320);
        assert_eq!(gating.num_experts(), 8);
    }

    #[test]
    fn test_gpu_gating_forward_real_data() {
        let device = test_device();
        let config = FusionConfig::default();
        let gating = GpuGatingNetwork::new(&config, &device).unwrap();

        let input_data: Vec<f32> = (0..8320).map(|i| (i as f32) * 0.0001).collect();
        let input = Tensor::from_slice(&input_data, (1, 8320), &device).unwrap();
        let probs = gating.forward(&input).unwrap();

        let dims = probs.dims();
        assert!(dims.contains(&8), "Gating should output 8 expert probs, got {:?}", dims);

        let probs_vec: Vec<f32> = probs.flatten_all().unwrap().to_vec1().unwrap();
        let sum: f32 = probs_vec.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Probabilities should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_gpu_gating_topk_selection() {
        let device = test_device();
        let config = FusionConfig::default();
        let gating = GpuGatingNetwork::new(&config, &device).unwrap();

        let input_data: Vec<f32> = (0..4 * 8320).map(|i| (i as f32) * 0.00001).collect();
        let input = Tensor::from_slice(&input_data, (4, 8320), &device).unwrap();

        let result: (Tensor, Tensor) = gating.forward_topk(&input, 4).unwrap();
        let indices = result.0;
        let weights = result.1;

        let indices_flat = indices.flatten_all().unwrap();
        let indices_vec: Vec<u32> = indices_flat.to_vec1::<u32>().unwrap();
        let weights_flat = weights.flatten_all().unwrap();
        let weights_vec: Vec<f32> = weights_flat.to_vec1::<f32>().unwrap();

        assert_eq!(indices_vec.len(), 16, "Should have 4 samples * 4 experts = 16 indices");
        assert_eq!(weights_vec.len(), 16, "Should have 16 weights");

        for idx in &indices_vec {
            assert!(*idx < 8, "Expert index should be < 8, got {}", idx);
        }

        for sample in 0..4 {
            let sample_weights: f32 = (0..4)
                .map(|k| weights_vec[sample * 4 + k])
                .sum();
            assert!(
                (sample_weights - 1.0).abs() < 0.01,
                "Sample {} weights should sum to 1, got {}",
                sample, sample_weights
            );
        }
    }
}
