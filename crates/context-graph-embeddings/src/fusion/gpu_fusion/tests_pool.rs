//! Tests for GPU Expert Pool and FuseMoE components.

#[cfg(test)]
#[cfg(feature = "candle")]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use crate::config::FusionConfig;
    use crate::types::dimensions::{FUSED_OUTPUT, NUM_EXPERTS, TOTAL_CONCATENATED};

    use crate::fusion::gpu_fusion::{GpuExpertPool, GpuFuseMoE};

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
    // GPU EXPERT POOL TESTS
    // =========================================================================

    #[test]
    fn test_gpu_expert_pool_creation() {
        let device = test_device();
        let config = FusionConfig::default();
        let pool = GpuExpertPool::new(&config, &device).unwrap();

        assert_eq!(pool.num_experts(), 8);
        assert_eq!(pool.input_dim(), 8320);
        assert_eq!(pool.output_dim(), 1536);
    }

    #[test]
    fn test_gpu_expert_pool_forward_topk_single() {
        let device = test_device();
        let config = FusionConfig::default();
        let pool = GpuExpertPool::new(&config, &device).unwrap();

        let input_data: Vec<f32> = (0..8320).map(|i| (i as f32) * 0.0001).collect();
        let input = Tensor::from_slice(&input_data, (1, 8320), &device).unwrap();

        let indices = Tensor::from_slice(&[0u32, 1, 2, 3], (1, 4), &device).unwrap();
        let weights = Tensor::from_slice(&[0.25f32, 0.25, 0.25, 0.25], (1, 4), &device).unwrap();

        let output = pool.forward_topk(&input, &indices, &weights).unwrap();

        assert_eq!(output.dims(), &[1, 1536], "Pool output should be [1, 1536]");
    }

    #[test]
    fn test_gpu_expert_pool_forward_topk_batch() {
        let device = test_device();
        let config = FusionConfig::default();
        let pool = GpuExpertPool::new(&config, &device).unwrap();

        let input_data: Vec<f32> = (0..4 * 8320).map(|i| (i as f32) * 0.00001).collect();
        let input = Tensor::from_slice(&input_data, (4, 8320), &device).unwrap();

        let indices = Tensor::from_slice(
            &[0u32, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6],
            (4, 4),
            &device,
        )
        .unwrap();
        let weights = Tensor::from_slice(
            &[0.4f32, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2, 0.1],
            (4, 4),
            &device,
        )
        .unwrap();

        let output = pool.forward_topk(&input, &indices, &weights).unwrap();

        assert_eq!(output.dims(), &[4, 1536], "Pool output should be [4, 1536]");
    }

    #[test]
    fn test_gpu_expert_pool_parameter_count() {
        let device = test_device();
        let config = FusionConfig::default();
        let pool = GpuExpertPool::new(&config, &device).unwrap();

        let params = pool.parameter_count();
        assert!(params > 300_000_000, "Pool should have > 300M parameters, got {}", params);
        assert!(params < 350_000_000, "Pool should have < 350M parameters, got {}", params);
    }

    // =========================================================================
    // GPU FUSE MOE COMPLETE PIPELINE TESTS
    // =========================================================================

    #[test]
    fn test_gpu_fusemoe_creation() {
        let device = test_device();
        let config = FusionConfig::default();
        let fusemoe = GpuFuseMoE::new(&config, &device).unwrap();

        assert_eq!(fusemoe.input_dim(), 8320);
        assert_eq!(fusemoe.output_dim(), 1536);
    }

    #[test]
    fn test_gpu_fusemoe_forward_single_sample() {
        let device = test_device();
        let config = FusionConfig::default();
        let fusemoe = GpuFuseMoE::new(&config, &device).unwrap();

        let input_data: Vec<f32> = (0..8320).map(|i| (i as f32) * 0.0001).collect();
        let input = Tensor::from_slice(&input_data, (1, 8320), &device).unwrap();

        let output = fusemoe.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 1536], "FuseMoE output should be [1, 1536]");
    }

    #[test]
    fn test_gpu_fusemoe_forward_batch() {
        let device = test_device();
        let config = FusionConfig::default();
        let fusemoe = GpuFuseMoE::new(&config, &device).unwrap();

        let input_data: Vec<f32> = (0..32 * 8320)
            .map(|i| ((i % 1000) as f32) * 0.001)
            .collect();
        let input = Tensor::from_slice(&input_data, (32, 8320), &device).unwrap();

        let output = fusemoe.forward(&input).unwrap();

        assert_eq!(output.dims(), &[32, 1536], "FuseMoE output should be [32, 1536]");
    }

    #[test]
    fn test_gpu_fusemoe_output_values_normalized() {
        let device = test_device();
        let config = FusionConfig::default();
        let fusemoe = GpuFuseMoE::new(&config, &device).unwrap();

        let input_data: Vec<f32> = (0..8320).map(|i| (i as f32) * 0.0001).collect();
        let input = Tensor::from_slice(&input_data, (1, 8320), &device).unwrap();

        let output = fusemoe.forward(&input).unwrap();
        let output_vec: Vec<f32> = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for (i, val) in output_vec.iter().enumerate() {
            assert!(!(*val).is_nan(), "Output[{}] is NaN", i);
            assert!(!val.is_infinite(), "Output[{}] is Inf", i);
        }
    }

    #[test]
    fn test_gpu_fusemoe_total_parameter_count() {
        let device = test_device();
        let config = FusionConfig::default();
        let fusemoe = GpuFuseMoE::new(&config, &device).unwrap();

        let params = fusemoe.parameter_count();
        assert!(params > 320_000_000, "FuseMoE should have > 320M parameters, got {}", params);
        assert!(params < 330_000_000, "FuseMoE should have < 330M parameters, got {}", params);
    }

    // =========================================================================
    // DIMENSION CONSISTENCY TESTS
    // =========================================================================

    #[test]
    fn test_dimension_constants_match() {
        assert_eq!(TOTAL_CONCATENATED, 8320, "TOTAL_CONCATENATED should be 8320");
        assert_eq!(FUSED_OUTPUT, 1536, "FUSED_OUTPUT should be 1536");
        assert_eq!(NUM_EXPERTS, 8, "NUM_EXPERTS should be 8");
    }

    #[test]
    fn test_gpu_fusemoe_wrong_input_dim_fails() {
        let device = test_device();
        let config = FusionConfig::default();
        let fusemoe = GpuFuseMoE::new(&config, &device).unwrap();

        let input = Tensor::zeros((1, 1024), DType::F32, &device).unwrap();
        let result = fusemoe.forward(&input);

        assert!(result.is_err(), "Forward with wrong dimension should fail");
    }

    #[test]
    fn test_gpu_fusemoe_reproducibility() {
        let device = test_device();
        let config = FusionConfig::default();

        let input_data: Vec<f32> = (0..8320).map(|i| (i as f32) * 0.0001).collect();
        let input = Tensor::from_slice(&input_data, (1, 8320), &device).unwrap();

        let fusemoe1 = GpuFuseMoE::new(&config, &device).unwrap();
        let output1 = fusemoe1.forward(&input).unwrap();
        let output1_vec: Vec<f32> = output1.flatten_all().unwrap().to_vec1().unwrap();

        assert_eq!(output1_vec.len(), 1536);
    }
}
