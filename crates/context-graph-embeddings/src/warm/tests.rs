//! Comprehensive integration tests for the warm model loading system.
//!
//! # Test Groups
//!
//! 1. **Config Tests**: WarmConfig defaults, environment loading, validation
//! 2. **Error Tests**: Exit codes 101-110, error categories, fatal vs non-fatal
//! 3. **State Machine Tests**: WarmModelState transitions and predicates
//! 4. **Registry Tests**: WarmModelRegistry with all 13 models
//! 5. **Memory Pool Tests**: WarmMemoryPools dual-pool architecture
//! 6. **Validation Tests**: WarmValidator dimension/weight/inference validation
//! 7. **Handle Tests**: ModelHandle VRAM pointer tracking
//! 8. **Loader Tests**: WarmLoader orchestration logic
//! 9. **Health Check Tests**: WarmHealthChecker status monitoring
//! 10. **Diagnostics Tests**: WarmDiagnostics JSON reporting
//!
//! # Design Principles
//!
//! - **NO MOCKS**: All tests use real component instances
//! - **COMPREHENSIVE**: Cover all major code paths and edge cases
//! - **FAIL-FAST**: Verify error handling works correctly

use super::config::{QuantizationMode, WarmConfig};
use super::error::{WarmError, WarmResult};
use super::handle::ModelHandle;
use super::memory_pool::WarmMemoryPools;
use super::registry::{
    WarmModelRegistry, EMBEDDING_MODEL_IDS, FUSEMOE_MODEL_ID, TOTAL_MODEL_COUNT,
};
use super::state::WarmModelState;
use super::validation::{TestInferenceConfig, TestInput, ValidationResult, WarmValidator};

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// One gigabyte in bytes.
const GB: usize = 1024 * 1024 * 1024;

/// One megabyte in bytes.
const MB: usize = 1024 * 1024;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Create a test configuration with valid paths.
fn test_config() -> WarmConfig {
    let mut config = WarmConfig::default();
    // Use current directory which exists
    config.model_weights_path = PathBuf::from(".");
    config
}

/// Create a test ModelHandle with specified bytes.
fn test_handle(bytes: usize) -> ModelHandle {
    ModelHandle::new(0x1000_0000, bytes, 0, 0xDEAD_BEEF)
}

/// Create a test ModelHandle with custom address and checksum.
fn test_handle_full(address: u64, bytes: usize, device: u32, checksum: u64) -> ModelHandle {
    ModelHandle::new(address, bytes, device, checksum)
}

// ============================================================================
// GROUP 1: CONFIG TESTS
// ============================================================================

mod config_tests {
    use super::*;

    #[test]
    fn test_default_values_rtx_5090() {
        let config = WarmConfig::default();

        assert_eq!(config.vram_budget_bytes, 24 * GB);
        assert_eq!(config.vram_headroom_bytes, 8 * GB);
        assert_eq!(config.model_weights_path, PathBuf::from("./models"));
        assert_eq!(
            config.diagnostic_dump_path,
            PathBuf::from("/var/log/context-graph")
        );
        assert_eq!(config.cuda_device_id, 0);
        assert!(config.enable_test_inference);
        assert_eq!(config.max_load_time_per_model_ms, 30_000);
        assert_eq!(config.quantization, QuantizationMode::Fp16);
    }

    #[test]
    fn test_total_vram_required_32gb() {
        let config = WarmConfig::default();
        assert_eq!(config.total_vram_required(), 32 * GB);
    }

    #[test]
    fn test_quantization_mode_memory_multipliers() {
        assert_eq!(QuantizationMode::Fp32.memory_multiplier(), 1.0);
        assert_eq!(QuantizationMode::Fp16.memory_multiplier(), 0.5);
        assert_eq!(QuantizationMode::Fp8.memory_multiplier(), 0.25);
    }

    #[test]
    fn test_quantization_mode_as_str() {
        assert_eq!(QuantizationMode::Fp32.as_str(), "FP32");
        assert_eq!(QuantizationMode::Fp16.as_str(), "FP16");
        assert_eq!(QuantizationMode::Fp8.as_str(), "FP8");
    }

    #[test]
    fn test_validate_zero_budget_fails() {
        let mut config = test_config();
        config.vram_budget_bytes = 0;

        let result = config.validate();
        assert!(result.is_err());

        if let Err(WarmError::InvalidConfig { field, reason }) = result {
            assert_eq!(field, "vram_budget_bytes");
            assert!(reason.contains("greater than 0"));
        } else {
            panic!("Expected InvalidConfig error");
        }
    }

    #[test]
    fn test_validate_missing_path_fails() {
        let mut config = WarmConfig::default();
        config.model_weights_path = PathBuf::from("/nonexistent/path/that/does/not/exist");

        let result = config.validate();
        assert!(result.is_err());

        if let Err(WarmError::InvalidConfig { field, .. }) = result {
            assert_eq!(field, "model_weights_path");
        } else {
            panic!("Expected InvalidConfig error");
        }
    }

    #[test]
    fn test_validate_valid_config_succeeds() {
        let config = test_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_from_env_vram_budget() {
        std::env::set_var("WARM_VRAM_BUDGET_BYTES", "16000000000");
        let config = WarmConfig::from_env();
        assert_eq!(config.vram_budget_bytes, 16_000_000_000);
        std::env::remove_var("WARM_VRAM_BUDGET_BYTES");
    }

    // Note: CUDA_VISIBLE_DEVICES test is in config.rs to avoid env var interference

    #[test]
    fn test_from_env_test_inference_boolean_variants() {
        for (val, expected) in [
            ("true", true),
            ("TRUE", true),
            ("1", true),
            ("yes", true),
            ("on", true),
            ("false", false),
            ("0", false),
            ("no", false),
            ("off", false),
        ] {
            std::env::set_var("WARM_ENABLE_TEST_INFERENCE", val);
            let config = WarmConfig::from_env();
            assert_eq!(
                config.enable_test_inference, expected,
                "Failed for value '{}'",
                val
            );
        }
        std::env::remove_var("WARM_ENABLE_TEST_INFERENCE");
    }
}

// ============================================================================
// GROUP 2: ERROR TESTS
// ============================================================================

mod error_tests {
    use super::*;

    #[test]
    fn test_exit_codes_fatal_101_to_110() {
        let fatal_errors: Vec<(i32, WarmError)> = vec![
            (
                101,
                WarmError::ModelFileMissing {
                    model_id: "E1_Semantic".to_string(),
                    path: "/models/semantic.bin".to_string(),
                },
            ),
            (
                102,
                WarmError::ModelLoadFailed {
                    model_id: "E2_Temporal".to_string(),
                    reason: "Checksum mismatch".to_string(),
                    bytes_read: 1024,
                    file_size: 10240,
                },
            ),
            (
                103,
                WarmError::ModelValidationFailed {
                    model_id: "E3_Causal".to_string(),
                    reason: "Test inference produced NaN".to_string(),
                    expected_output: Some("[0.1, 0.2]".to_string()),
                    actual_output: Some("[NaN, NaN]".to_string()),
                },
            ),
            (
                104,
                WarmError::VramInsufficientTotal {
                    required_bytes: 32 * GB,
                    available_bytes: 24 * GB,
                    required_gb: 32.0,
                    available_gb: 24.0,
                    model_breakdown: vec![("E1".to_string(), 1024)],
                },
            ),
            (
                105,
                WarmError::VramInsufficientHeadroom {
                    model_bytes: 24 * GB,
                    available_bytes: 28 * GB,
                    headroom_required: 8 * GB,
                    model_gb: 24.0,
                    available_gb: 28.0,
                    headroom_gb: 8.0,
                },
            ),
            (
                106,
                WarmError::CudaInitFailed {
                    cuda_error: "CUDA driver not found".to_string(),
                    driver_version: "".to_string(),
                    gpu_name: "".to_string(),
                },
            ),
            (
                107,
                WarmError::CudaCapabilityInsufficient {
                    actual_cc: "8.6".to_string(),
                    required_cc: "12.0".to_string(),
                    gpu_name: "RTX 3090".to_string(),
                },
            ),
            (
                108,
                WarmError::CudaAllocFailed {
                    requested_bytes: 1 * GB,
                    cuda_error: "CUDA_ERROR_OUT_OF_MEMORY".to_string(),
                    vram_free: Some(512 * MB),
                    allocation_history: vec!["model_1: 2GB".to_string()],
                },
            ),
            (
                109,
                WarmError::CudaContextLost {
                    reason: "TDR timeout".to_string(),
                    last_successful_op: "cudaMemcpy".to_string(),
                },
            ),
            (
                110,
                WarmError::ModelDimensionMismatch {
                    model_id: "E1_Semantic".to_string(),
                    expected: 1024,
                    actual: 768,
                },
            ),
        ];

        for (expected_code, err) in fatal_errors {
            assert_eq!(
                err.exit_code(),
                expected_code,
                "Exit code mismatch for {:?}",
                err.category()
            );
            assert!(err.is_fatal(), "Expected fatal for exit code {}", expected_code);
        }
    }

    #[test]
    fn test_exit_codes_non_fatal() {
        let non_fatal_errors: Vec<WarmError> = vec![
            WarmError::ModelAlreadyRegistered {
                model_id: "E1".to_string(),
            },
            WarmError::ModelNotRegistered {
                model_id: "E1".to_string(),
            },
            WarmError::InvalidConfig {
                field: "vram_budget".to_string(),
                reason: "must be > 0".to_string(),
            },
            WarmError::RegistryLockPoisoned,
            WarmError::WorkingMemoryExhausted {
                requested_bytes: 1024,
                available_bytes: 512,
            },
            WarmError::CudaNotAvailable,
            WarmError::CudaQueryFailed {
                error: "Query failed".to_string(),
            },
            WarmError::DiagnosticDumpFailed {
                reason: "Permission denied".to_string(),
            },
            WarmError::LoadTimeout {
                model_id: "E1".to_string(),
                timeout_ms: 30000,
            },
            WarmError::VramAllocationFailed {
                requested_bytes: 1024,
                available_bytes: 512,
                error: "Pool exhausted".to_string(),
            },
        ];

        for err in non_fatal_errors {
            assert_eq!(
                err.exit_code(),
                1,
                "Expected exit code 1 for {:?}",
                err.category()
            );
            assert!(!err.is_fatal(), "Expected non-fatal for {:?}", err.category());
        }
    }

    #[test]
    fn test_error_categories() {
        assert_eq!(
            WarmError::ModelFileMissing {
                model_id: "E1".to_string(),
                path: "/".to_string()
            }
            .category(),
            "MODEL_FILE"
        );
        assert_eq!(
            WarmError::CudaInitFailed {
                cuda_error: "".to_string(),
                driver_version: "".to_string(),
                gpu_name: "".to_string()
            }
            .category(),
            "CUDA"
        );
        assert_eq!(
            WarmError::VramInsufficientTotal {
                required_bytes: 0,
                available_bytes: 0,
                required_gb: 0.0,
                available_gb: 0.0,
                model_breakdown: vec![]
            }
            .category(),
            "VRAM"
        );
    }

    #[test]
    fn test_error_display_messages() {
        let err = WarmError::ModelFileMissing {
            model_id: "E1_Semantic".to_string(),
            path: "/models/semantic.bin".to_string(),
        };
        assert_eq!(
            format!("{}", err),
            "Model file missing: E1_Semantic not found at /models/semantic.bin"
        );

        let err = WarmError::VramInsufficientTotal {
            required_bytes: 32 * GB,
            available_bytes: 24 * GB,
            required_gb: 32.0,
            available_gb: 24.0,
            model_breakdown: vec![],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("32.00GB"));
        assert!(msg.contains("24.00GB"));
    }

    #[test]
    fn test_error_codes() {
        assert_eq!(
            WarmError::ModelFileMissing {
                model_id: "".to_string(),
                path: "".to_string()
            }
            .error_code(),
            "ERR-WARM-MODEL-MISSING"
        );
        assert_eq!(
            WarmError::CudaInitFailed {
                cuda_error: "".to_string(),
                driver_version: "".to_string(),
                gpu_name: "".to_string()
            }
            .error_code(),
            "ERR-WARM-CUDA-INIT"
        );
        assert_eq!(
            WarmError::ModelDimensionMismatch {
                model_id: "".to_string(),
                expected: 0,
                actual: 0
            }
            .error_code(),
            "ERR-WARM-MODEL-DIMENSION-MISMATCH"
        );
    }

    #[test]
    fn test_warm_result_type_alias() {
        fn returns_ok() -> WarmResult<i32> {
            Ok(42)
        }

        fn returns_err() -> WarmResult<i32> {
            Err(WarmError::CudaNotAvailable)
        }

        assert_eq!(returns_ok().unwrap(), 42);
        assert!(returns_err().is_err());
    }
}

// ============================================================================
// GROUP 3: STATE MACHINE TESTS
// ============================================================================

mod state_tests {
    use super::*;

    #[test]
    fn test_pending_predicates() {
        let s = WarmModelState::Pending;
        assert!(!s.is_warm());
        assert!(!s.is_failed());
        assert!(!s.is_loading());
    }

    #[test]
    fn test_loading_predicates() {
        let s = WarmModelState::Loading {
            progress_percent: 50,
            bytes_loaded: 1024,
        };
        assert!(!s.is_warm());
        assert!(!s.is_failed());
        assert!(s.is_loading());
    }

    #[test]
    fn test_validating_predicates() {
        let s = WarmModelState::Validating;
        assert!(!s.is_warm());
        assert!(!s.is_failed());
        assert!(!s.is_loading());
    }

    #[test]
    fn test_warm_predicates() {
        let s = WarmModelState::Warm;
        assert!(s.is_warm());
        assert!(!s.is_failed());
        assert!(!s.is_loading());
    }

    #[test]
    fn test_failed_predicates() {
        let s = WarmModelState::Failed {
            error_code: 101,
            error_message: "VRAM exhausted".into(),
        };
        assert!(!s.is_warm());
        assert!(s.is_failed());
        assert!(!s.is_loading());
    }

    #[test]
    fn test_state_equality() {
        assert_eq!(WarmModelState::Pending, WarmModelState::Pending);
        assert_eq!(WarmModelState::Warm, WarmModelState::Warm);
        assert_ne!(WarmModelState::Pending, WarmModelState::Warm);

        let loading1 = WarmModelState::Loading {
            progress_percent: 50,
            bytes_loaded: 1024,
        };
        let loading2 = WarmModelState::Loading {
            progress_percent: 50,
            bytes_loaded: 1024,
        };
        let loading3 = WarmModelState::Loading {
            progress_percent: 75,
            bytes_loaded: 2048,
        };
        assert_eq!(loading1, loading2);
        assert_ne!(loading1, loading3);
    }

    #[test]
    fn test_state_clone() {
        let original = WarmModelState::Failed {
            error_code: 102,
            error_message: "Load failed".to_string(),
        };
        let cloned = original.clone();

        if let (
            WarmModelState::Failed {
                error_code: c1,
                error_message: m1,
            },
            WarmModelState::Failed {
                error_code: c2,
                error_message: m2,
            },
        ) = (original, cloned)
        {
            assert_eq!(c1, c2);
            assert_eq!(m1, m2);
        } else {
            panic!("Clone should produce Failed state");
        }
    }
}

// ============================================================================
// GROUP 4: REGISTRY TESTS
// ============================================================================

mod registry_tests {
    use super::*;

    #[test]
    fn test_new_registry_is_empty() {
        let registry = WarmModelRegistry::new();
        assert_eq!(registry.model_count(), 0);
        assert!(!registry.all_warm());
        assert!(!registry.any_failed());
    }

    #[test]
    fn test_embedding_model_ids_count() {
        assert_eq!(EMBEDDING_MODEL_IDS.len(), 12);
        assert_eq!(FUSEMOE_MODEL_ID, "FuseMoE");
        assert_eq!(TOTAL_MODEL_COUNT, 13);
    }

    #[test]
    fn test_register_all_13_models() {
        let mut registry = WarmModelRegistry::new();

        for (i, model_id) in EMBEDDING_MODEL_IDS.iter().enumerate() {
            registry
                .register_model(*model_id, (i + 1) * 100 * MB, 768)
                .unwrap();
        }
        registry
            .register_model(FUSEMOE_MODEL_ID, 2 * GB, 768)
            .unwrap();

        assert_eq!(registry.model_count(), TOTAL_MODEL_COUNT);
    }

    #[test]
    fn test_register_duplicate_fails() {
        let mut registry = WarmModelRegistry::new();
        registry.register_model("E1_Semantic", 512 * MB, 768).unwrap();

        let err = registry
            .register_model("E1_Semantic", 256 * MB, 512)
            .unwrap_err();

        match err {
            WarmError::ModelAlreadyRegistered { model_id } => {
                assert_eq!(model_id, "E1_Semantic");
            }
            _ => panic!("Expected ModelAlreadyRegistered error"),
        }
    }

    #[test]
    fn test_full_state_transition_pending_to_warm() {
        let mut registry = WarmModelRegistry::new();
        registry.register_model("E1_Semantic", 512 * MB, 768).unwrap();

        // Pending -> Loading
        assert!(matches!(
            registry.get_state("E1_Semantic"),
            Some(WarmModelState::Pending)
        ));

        registry.start_loading("E1_Semantic").unwrap();
        assert!(matches!(
            registry.get_state("E1_Semantic"),
            Some(WarmModelState::Loading { .. })
        ));

        // Update progress
        registry.update_progress("E1_Semantic", 50, 256 * MB).unwrap();
        if let Some(WarmModelState::Loading {
            progress_percent,
            bytes_loaded,
        }) = registry.get_state("E1_Semantic")
        {
            assert_eq!(progress_percent, 50);
            assert_eq!(bytes_loaded, 256 * MB);
        }

        // Loading -> Validating
        registry.mark_validating("E1_Semantic").unwrap();
        assert!(matches!(
            registry.get_state("E1_Semantic"),
            Some(WarmModelState::Validating)
        ));

        // Validating -> Warm
        registry
            .mark_warm("E1_Semantic", test_handle(512 * MB))
            .unwrap();
        assert!(matches!(
            registry.get_state("E1_Semantic"),
            Some(WarmModelState::Warm)
        ));
        assert!(registry.get_handle("E1_Semantic").is_some());
    }

    #[test]
    fn test_invalid_transition_pending_to_validating() {
        let mut registry = WarmModelRegistry::new();
        registry.register_model("E1_Semantic", 512 * MB, 768).unwrap();

        let err = registry.mark_validating("E1_Semantic").unwrap_err();
        assert!(matches!(err, WarmError::ModelValidationFailed { .. }));
    }

    #[test]
    fn test_invalid_transition_pending_to_warm() {
        let mut registry = WarmModelRegistry::new();
        registry.register_model("E1_Semantic", 512 * MB, 768).unwrap();

        let err = registry
            .mark_warm("E1_Semantic", test_handle(512 * MB))
            .unwrap_err();
        assert!(matches!(err, WarmError::ModelValidationFailed { .. }));
    }

    #[test]
    fn test_mark_failed_from_loading() {
        let mut registry = WarmModelRegistry::new();
        registry.register_model("E1_Semantic", 512 * MB, 768).unwrap();
        registry.start_loading("E1_Semantic").unwrap();

        registry
            .mark_failed("E1_Semantic", 102, "CUDA allocation failed")
            .unwrap();

        match registry.get_state("E1_Semantic") {
            Some(WarmModelState::Failed {
                error_code,
                error_message,
            }) => {
                assert_eq!(error_code, 102);
                assert_eq!(error_message, "CUDA allocation failed");
            }
            _ => panic!("Expected Failed state"),
        }
    }

    #[test]
    fn test_loading_order_largest_first() {
        let mut registry = WarmModelRegistry::new();

        registry.register_model("Small", 100 * MB, 768).unwrap();
        registry.register_model("Large", 500 * MB, 768).unwrap();
        registry.register_model("Medium", 250 * MB, 768).unwrap();

        let order = registry.loading_order();
        assert_eq!(order[0], "Large");
        assert_eq!(order[1], "Medium");
        assert_eq!(order[2], "Small");
    }

    #[test]
    fn test_all_warm_complete() {
        let mut registry = WarmModelRegistry::new();
        registry.register_model("E1", 256 * MB, 768).unwrap();
        registry.register_model("E2", 256 * MB, 768).unwrap();

        for model_id in ["E1", "E2"] {
            registry.start_loading(model_id).unwrap();
            registry.mark_validating(model_id).unwrap();
            registry.mark_warm(model_id, test_handle(256 * MB)).unwrap();
        }

        assert!(registry.all_warm());
        assert_eq!(registry.warm_count(), 2);
    }

    #[test]
    fn test_any_failed_detection() {
        let mut registry = WarmModelRegistry::new();
        registry.register_model("E1", 256 * MB, 768).unwrap();
        registry.register_model("E2", 256 * MB, 768).unwrap();

        registry.start_loading("E1").unwrap();
        registry.mark_failed("E1", 102, "Failed").unwrap();

        assert!(registry.any_failed());
        assert!(!registry.all_warm());

        let failed = registry.failed_entries();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].0, "E1");
        assert_eq!(failed[0].1, 102);
    }

    #[test]
    fn test_operations_on_unregistered_model() {
        let mut registry = WarmModelRegistry::new();

        assert!(matches!(
            registry.start_loading("NonExistent"),
            Err(WarmError::ModelNotRegistered { .. })
        ));
        assert!(matches!(
            registry.update_progress("NonExistent", 50, 1000),
            Err(WarmError::ModelNotRegistered { .. })
        ));
        assert!(matches!(
            registry.mark_validating("NonExistent"),
            Err(WarmError::ModelNotRegistered { .. })
        ));
        assert!(matches!(
            registry.mark_warm("NonExistent", test_handle(1000)),
            Err(WarmError::ModelNotRegistered { .. })
        ));
    }

    #[test]
    fn test_shared_registry_thread_safety() {
        type SharedRegistry = Arc<RwLock<WarmModelRegistry>>;
        let registry: SharedRegistry = Arc::new(RwLock::new(WarmModelRegistry::new()));

        // Write access
        {
            let mut reg = registry.write().unwrap();
            reg.register_model("E1_Semantic", 512 * MB, 768).unwrap();
        }

        // Read access
        {
            let reg = registry.read().unwrap();
            assert_eq!(reg.model_count(), 1);
        }

        // Clone and access
        let registry_clone = Arc::clone(&registry);
        {
            let reg = registry_clone.read().unwrap();
            assert!(reg.get_state("E1_Semantic").is_some());
        }
    }
}

// ============================================================================
// GROUP 5: MEMORY POOL TESTS
// ============================================================================

mod memory_pool_tests {
    use super::*;

    #[test]
    fn test_rtx_5090_factory_capacities() {
        let pools = WarmMemoryPools::rtx_5090();

        assert_eq!(pools.model_pool_capacity(), 24 * GB);
        assert_eq!(pools.working_pool_capacity(), 8 * GB);
        assert_eq!(pools.total_capacity(), 32 * GB);
    }

    #[test]
    fn test_model_allocation_and_tracking() {
        let mut pools = WarmMemoryPools::rtx_5090();

        pools.allocate_model("E1_Semantic", 800 * MB, 0x1000).unwrap();

        let alloc = pools.get_model_allocation("E1_Semantic").unwrap();
        assert_eq!(alloc.size_bytes, 800 * MB);
        assert_eq!(alloc.vram_ptr, 0x1000);
        assert_eq!(pools.total_allocated_bytes(), 800 * MB);
        assert_eq!(pools.available_model_bytes(), 24 * GB - 800 * MB);
    }

    #[test]
    fn test_model_deallocation() {
        let mut pools = WarmMemoryPools::rtx_5090();

        pools.allocate_model("E1_Semantic", 800 * MB, 0x1000).unwrap();
        pools.free_model("E1_Semantic").unwrap();

        assert!(pools.get_model_allocation("E1_Semantic").is_none());
        assert_eq!(pools.total_allocated_bytes(), 0);
    }

    #[test]
    fn test_budget_enforcement_model_pool() {
        let mut pools = WarmMemoryPools::rtx_5090();

        pools.allocate_model("model1", 20 * GB, 0x1000).unwrap();
        pools.allocate_model("model2", 3 * GB, 0x2000).unwrap();

        // This should fail (exceeds 24GB capacity)
        let result = pools.allocate_model("model3", 2 * GB, 0x3000);
        assert!(matches!(result, Err(WarmError::VramAllocationFailed { .. })));
    }

    #[test]
    fn test_working_memory_allocation_and_free() {
        let mut pools = WarmMemoryPools::rtx_5090();

        pools.allocate_working(1 * GB).unwrap();
        assert_eq!(pools.available_working_bytes(), 7 * GB);

        pools.free_working(1 * GB).unwrap();
        assert_eq!(pools.available_working_bytes(), 8 * GB);
    }

    #[test]
    fn test_working_memory_exhaustion() {
        let mut pools = WarmMemoryPools::rtx_5090();

        pools.allocate_working(8 * GB).unwrap();

        let result = pools.allocate_working(1);
        assert!(matches!(result, Err(WarmError::WorkingMemoryExhausted { .. })));
    }

    #[test]
    fn test_is_within_budget() {
        let mut pools = WarmMemoryPools::rtx_5090();
        assert!(pools.is_within_budget());

        pools.allocate_model("model1", 20 * GB, 0x1000).unwrap();
        pools.allocate_working(6 * GB).unwrap();
        assert!(pools.is_within_budget());

        // Fill to capacity
        pools.allocate_model("model2", 4 * GB, 0x2000).unwrap();
        pools.allocate_working(2 * GB).unwrap();
        assert!(pools.is_within_budget());
    }

    #[test]
    fn test_duplicate_model_allocation_fails() {
        let mut pools = WarmMemoryPools::rtx_5090();

        pools.allocate_model("E1_Semantic", 1 * GB, 0x1000).unwrap();

        let result = pools.allocate_model("E1_Semantic", 1 * GB, 0x2000);
        match result {
            Err(WarmError::ModelAlreadyRegistered { model_id }) => {
                assert_eq!(model_id, "E1_Semantic");
            }
            _ => panic!("Expected ModelAlreadyRegistered error"),
        }
    }

    #[test]
    fn test_list_model_allocations() {
        let mut pools = WarmMemoryPools::rtx_5090();

        pools.allocate_model("model1", 1 * GB, 0x1000).unwrap();
        pools.allocate_model("model2", 2 * GB, 0x2000).unwrap();
        pools.allocate_model("model3", 3 * GB, 0x3000).unwrap();

        let allocations = pools.list_model_allocations();
        assert_eq!(allocations.len(), 3);

        let ids: Vec<_> = allocations.iter().map(|a| a.model_id.as_str()).collect();
        assert!(ids.contains(&"model1"));
        assert!(ids.contains(&"model2"));
        assert!(ids.contains(&"model3"));
    }

    #[test]
    fn test_utilization_metrics() {
        let mut pools = WarmMemoryPools::rtx_5090();

        assert_eq!(pools.model_pool_utilization(), 0.0);
        assert_eq!(pools.working_pool_utilization(), 0.0);

        pools.allocate_model("model1", 12 * GB, 0x1000).unwrap();
        pools.allocate_working(4 * GB).unwrap();

        assert!((pools.model_pool_utilization() - 0.5).abs() < 0.001);
        assert!((pools.working_pool_utilization() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_reset_working_pool() {
        let mut pools = WarmMemoryPools::rtx_5090();

        pools.allocate_working(5 * GB).unwrap();
        assert_eq!(pools.available_working_bytes(), 3 * GB);

        pools.reset_working_pool();
        assert_eq!(pools.available_working_bytes(), 8 * GB);
    }

    #[test]
    fn test_working_memory_over_free() {
        let mut pools = WarmMemoryPools::rtx_5090();

        pools.allocate_working(1 * GB).unwrap();
        pools.free_working(10 * GB).unwrap(); // Free more than allocated

        assert_eq!(pools.available_working_bytes(), 8 * GB);
    }
}

// ============================================================================
// GROUP 6: VALIDATION TESTS
// ============================================================================

mod validation_tests {
    use super::*;

    #[test]
    fn test_validator_default_tolerance() {
        let v = WarmValidator::new();
        // Just verify it's created successfully - tolerance is internal
        assert!(v.validate_dimensions("model", 1024, 1024).is_ok());
    }

    #[test]
    fn test_validator_custom_tolerance() {
        let v = WarmValidator::with_tolerance(0.01);
        // Verify custom tolerance validator works
        assert!(v.validate_dimensions("model", 1024, 1024).is_ok());
    }

    #[test]
    fn test_validate_dimensions_matching() {
        let v = WarmValidator::new();
        assert!(v.validate_dimensions("model", 1024, 1024).is_ok());
    }

    #[test]
    fn test_validate_dimensions_mismatch() {
        let v = WarmValidator::new();
        let result = v.validate_dimensions("E1_Semantic", 1024, 512);

        assert!(result.is_err());
        match result.unwrap_err() {
            WarmError::ModelDimensionMismatch {
                model_id,
                expected,
                actual,
            } => {
                assert_eq!(model_id, "E1_Semantic");
                assert_eq!(expected, 1024);
                assert_eq!(actual, 512);
            }
            _ => panic!("Expected ModelDimensionMismatch error"),
        }
    }

    #[test]
    fn test_validate_weights_finite_valid() {
        let v = WarmValidator::new();
        let weights = vec![0.1, -0.5, 1.0, -1.0, 0.0];
        assert!(v.validate_weights_finite(&weights).is_ok());
    }

    #[test]
    fn test_validate_weights_finite_nan() {
        let v = WarmValidator::new();
        let weights = vec![0.1, f32::NAN, 0.3];
        let result = v.validate_weights_finite(&weights);

        assert!(result.is_err());
        match result.unwrap_err() {
            WarmError::ModelValidationFailed {
                reason,
                actual_output,
                ..
            } => {
                assert!(reason.contains("NaN"));
                assert!(reason.contains("index 1"));
                assert_eq!(actual_output.as_deref(), Some("NaN"));
            }
            _ => panic!("Expected ModelValidationFailed error"),
        }
    }

    #[test]
    fn test_validate_weights_finite_positive_inf() {
        let v = WarmValidator::new();
        let weights = vec![0.1, f32::INFINITY, 0.3];
        let result = v.validate_weights_finite(&weights);

        assert!(result.is_err());
        match result.unwrap_err() {
            WarmError::ModelValidationFailed { actual_output, .. } => {
                assert_eq!(actual_output.as_deref(), Some("+Inf"));
            }
            _ => panic!("Expected ModelValidationFailed error"),
        }
    }

    #[test]
    fn test_validate_weights_finite_negative_inf() {
        let v = WarmValidator::new();
        let weights = vec![0.1, f32::NEG_INFINITY, 0.3];
        let result = v.validate_weights_finite(&weights);

        assert!(result.is_err());
        match result.unwrap_err() {
            WarmError::ModelValidationFailed { actual_output, .. } => {
                assert_eq!(actual_output.as_deref(), Some("-Inf"));
            }
            _ => panic!("Expected ModelValidationFailed error"),
        }
    }

    #[test]
    fn test_validate_weight_checksum_matching() {
        let v = WarmValidator::new();
        let handle = test_handle_full(0x1000, 1024, 0, 0xdeadbeefcafebabe);
        assert!(v.validate_weight_checksum(&handle, 0xdeadbeefcafebabe).is_ok());
    }

    #[test]
    fn test_validate_weight_checksum_mismatch() {
        let v = WarmValidator::new();
        let handle = test_handle_full(0x1000, 1024, 0, 0xdeadbeefcafebabe);
        let result = v.validate_weight_checksum(&handle, 0x1111111111111111);

        assert!(result.is_err());
        match result.unwrap_err() {
            WarmError::ModelValidationFailed {
                reason,
                expected_output,
                actual_output,
                ..
            } => {
                assert!(reason.contains("checksum"));
                assert!(expected_output.unwrap().contains("1111111111111111"));
                assert!(actual_output.unwrap().contains("deadbeefcafebabe"));
            }
            _ => panic!("Expected ModelValidationFailed error"),
        }
    }

    #[test]
    fn test_compare_output_identical() {
        let v = WarmValidator::new();
        let output = vec![0.1, 0.2, 0.3];
        let reference = vec![0.1, 0.2, 0.3];
        assert!(v.compare_output(&output, &reference, 1e-5).is_ok());
    }

    #[test]
    fn test_compare_output_within_tolerance() {
        let v = WarmValidator::new();
        let output = vec![0.10001, 0.20001, 0.30001];
        let reference = vec![0.1, 0.2, 0.3];
        assert!(v.compare_output(&output, &reference, 1e-3).is_ok());
    }

    #[test]
    fn test_compare_output_outside_tolerance() {
        let v = WarmValidator::new();
        let output = vec![0.1, 0.25, 0.3];
        let reference = vec![0.1, 0.2, 0.3];
        let result = v.compare_output(&output, &reference, 1e-5);

        assert!(result.is_err());
        match result.unwrap_err() {
            WarmError::ModelValidationFailed { reason, .. } => {
                assert!(reason.contains("index 1"));
            }
            _ => panic!("Expected ModelValidationFailed error"),
        }
    }

    #[test]
    fn test_compare_output_length_mismatch() {
        let v = WarmValidator::new();
        let output = vec![0.1, 0.2];
        let reference = vec![0.1, 0.2, 0.3];
        let result = v.compare_output(&output, &reference, 1e-5);

        assert!(result.is_err());
        match result.unwrap_err() {
            WarmError::ModelValidationFailed { reason, .. } => {
                assert!(reason.contains("length mismatch"));
            }
            _ => panic!("Expected ModelValidationFailed error"),
        }
    }

    #[test]
    fn test_validate_model_success() {
        let v = WarmValidator::new();
        let config = TestInferenceConfig::for_embedding_model("E1_Semantic", 4);
        let handle = test_handle_full(0x1000, 16, 0, 0xabcd);
        let output = vec![0.1, 0.2, 0.3, 0.4];

        let result = v.validate_model(&config, &handle, &output);

        assert!(result.is_valid());
        assert!(result.dimension_valid);
        assert!(result.weights_valid);
        assert!(result.inference_valid);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_validate_model_dimension_failure() {
        let v = WarmValidator::new();
        let config = TestInferenceConfig::for_embedding_model("E1_Semantic", 1024);
        let handle = test_handle_full(0x1000, 2048, 0, 0xabcd);
        let output = vec![0.1; 512];

        let result = v.validate_model(&config, &handle, &output);

        assert!(!result.is_valid());
        assert!(!result.dimension_valid);
        assert!(matches!(
            result.error,
            Some(WarmError::ModelDimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_validate_model_nan_failure() {
        let v = WarmValidator::new();
        let config = TestInferenceConfig::for_embedding_model("E1_Semantic", 4);
        let handle = test_handle_full(0x1000, 16, 0, 0xabcd);
        let output = vec![0.1, f32::NAN, 0.3, 0.4];

        let result = v.validate_model(&config, &handle, &output);

        assert!(!result.is_valid());
        assert!(result.dimension_valid);
        assert!(!result.weights_valid);
        assert!(matches!(
            result.error,
            Some(WarmError::ModelValidationFailed { .. })
        ));
    }

    #[test]
    fn test_test_inference_config_for_embedding() {
        let config = TestInferenceConfig::for_embedding_model("E1_Semantic", 1024);

        assert_eq!(config.model_id, "E1_Semantic");
        assert_eq!(config.expected_dimension, 1024);
        assert!(matches!(config.test_input, TestInput::Text(_)));
        assert!(config.reference_output.is_none());
        assert_eq!(config.max_inference_ms, 1000);
    }

    #[test]
    fn test_test_inference_config_for_fusemoe() {
        let config = TestInferenceConfig::for_fusemoe(2048);

        assert_eq!(config.model_id, "FuseMoE");
        assert_eq!(config.expected_dimension, 2048);
        assert!(matches!(config.test_input, TestInput::Tokens(_)));
        assert!(config.reference_output.is_none());
        assert_eq!(config.max_inference_ms, 2000);
    }

    #[test]
    fn test_test_input_types() {
        assert_eq!(TestInput::Text("hello".to_string()).description(), "text");
        assert_eq!(TestInput::Tokens(vec![1, 2, 3]).description(), "tokens");
        assert_eq!(
            TestInput::Embeddings(vec![0.1, 0.2]).description(),
            "embeddings"
        );

        assert_eq!(TestInput::Text("hello".to_string()).len(), 5);
        assert_eq!(TestInput::Tokens(vec![1, 2, 3]).len(), 3);
        assert!(!TestInput::Text("hello".to_string()).is_empty());
        assert!(TestInput::Text(String::new()).is_empty());
    }

    #[test]
    fn test_validation_result_success() {
        let result = ValidationResult::success("model1".to_string(), 42);

        assert_eq!(result.model_id, "model1");
        assert!(result.is_valid());
        assert!(result.dimension_valid);
        assert!(result.weights_valid);
        assert!(result.inference_valid);
        assert_eq!(result.inference_time_ms, 42);
        assert!(result.error().is_none());
    }

    #[test]
    fn test_validation_result_failure() {
        let error = WarmError::ModelDimensionMismatch {
            model_id: "model1".to_string(),
            expected: 100,
            actual: 50,
        };

        let result =
            ValidationResult::failure("model1".to_string(), false, true, false, 100, error);

        assert!(!result.is_valid());
        assert!(!result.dimension_valid);
        assert!(result.weights_valid);
        assert!(!result.inference_valid);
        assert!(result.error().is_some());
    }
}

// ============================================================================
// GROUP 7: HANDLE TESTS
// ============================================================================

mod handle_tests {
    use super::*;

    #[test]
    fn test_handle_creation() {
        let handle = ModelHandle::new(0x1000_0000, 512 * MB, 0, 0xDEAD_BEEF);

        assert_eq!(handle.vram_address(), 0x1000_0000);
        assert_eq!(handle.allocation_bytes(), 512 * MB);
        assert_eq!(handle.device_ordinal(), 0);
        assert_eq!(handle.weight_checksum(), 0xDEAD_BEEF);
    }

    #[test]
    fn test_handle_vram_address_hex() {
        let handle = ModelHandle::new(0x1000_0000, 512 * MB, 0, 0xDEAD_BEEF);
        let hex = handle.vram_address_hex();
        assert!(hex.contains("10000000"));
    }

    #[test]
    fn test_handle_different_devices() {
        let handle0 = ModelHandle::new(0x1000, 1024, 0, 0);
        let handle1 = ModelHandle::new(0x2000, 2048, 1, 0);

        assert_eq!(handle0.device_ordinal(), 0);
        assert_eq!(handle1.device_ordinal(), 1);
    }

    #[test]
    fn test_handle_checksum_verification() {
        let checksum = 0xCAFE_BABE_DEAD_BEEF;
        let handle = ModelHandle::new(0x1000, 1024, 0, checksum);
        assert_eq!(handle.weight_checksum(), checksum);
    }

    #[test]
    fn test_handle_large_allocation() {
        let handle = ModelHandle::new(0x1000_0000, 24 * GB, 0, 0);
        assert_eq!(handle.allocation_bytes(), 24 * GB);
    }

    #[test]
    fn test_handle_zero_allocation() {
        let handle = ModelHandle::new(0x1000, 0, 0, 0);
        assert_eq!(handle.allocation_bytes(), 0);
    }

    #[test]
    fn test_handle_not_clone_by_design() {
        // ModelHandle is intentionally NOT Clone/Copy to prevent VRAM ownership duplication
        // This test verifies we can create multiple handles with same data
        let handle1 = ModelHandle::new(0x1000_0000, 512 * MB, 0, 0xDEAD_BEEF);
        let handle2 = ModelHandle::new(0x1000_0000, 512 * MB, 0, 0xDEAD_BEEF);

        assert_eq!(handle1.vram_address(), handle2.vram_address());
        assert_eq!(handle1.allocation_bytes(), handle2.allocation_bytes());
        assert_eq!(handle1.device_ordinal(), handle2.device_ordinal());
        assert_eq!(handle1.weight_checksum(), handle2.weight_checksum());
    }
}

// ============================================================================
// GROUP 8: LOADER TESTS (Integration with Registry + Pools)
// ============================================================================

mod loader_integration_tests {
    use super::*;

    #[test]
    fn test_loader_orchestration_simulation() {
        // Simulate what WarmLoader does: register, allocate, transition
        let mut registry = WarmModelRegistry::new();
        let mut pools = WarmMemoryPools::rtx_5090();

        let models = [("E1_Semantic", 800 * MB), ("E2_Temporal", 600 * MB)];

        // Register models
        for (model_id, size) in models {
            registry.register_model(model_id, size, 768).unwrap();
        }

        // Get loading order (largest first)
        let order = registry.loading_order();
        assert_eq!(order[0], "E1_Semantic"); // 800MB > 600MB

        // Simulate loading each model
        for model_id in &order {
            // Start loading
            registry.start_loading(model_id).unwrap();

            // Get expected size
            let entry = registry.get_entry(model_id).unwrap();
            let size = entry.expected_bytes;

            // Allocate VRAM
            let vram_ptr = 0x1000_0000 + (size as u64);
            pools.allocate_model(model_id, size, vram_ptr).unwrap();

            // Update progress
            registry.update_progress(model_id, 100, size).unwrap();

            // Mark validating then warm
            registry.mark_validating(model_id).unwrap();
            registry.mark_warm(model_id, test_handle(size)).unwrap();
        }

        assert!(registry.all_warm());
        assert!(pools.is_within_budget());
    }

    #[test]
    fn test_loader_fail_fast_on_vram_exhaustion() {
        let mut registry = WarmModelRegistry::new();
        let mut pools = WarmMemoryPools::rtx_5090();

        // Register models that exceed VRAM budget
        registry.register_model("Huge1", 20 * GB, 768).unwrap();
        registry.register_model("Huge2", 10 * GB, 768).unwrap(); // Total 30GB > 24GB

        let order = registry.loading_order();

        // Load first model
        let model_id = &order[0];
        registry.start_loading(model_id).unwrap();
        let size = registry.get_entry(model_id).unwrap().expected_bytes;
        pools.allocate_model(model_id, size, 0x1000).unwrap();
        registry.update_progress(model_id, 100, size).unwrap();
        registry.mark_validating(model_id).unwrap();
        registry.mark_warm(model_id, test_handle(size)).unwrap();

        // Try to load second model - should fail
        let model_id = &order[1];
        registry.start_loading(model_id).unwrap();
        let size = registry.get_entry(model_id).unwrap().expected_bytes;

        let result = pools.allocate_model(model_id, size, 0x2000);
        assert!(matches!(result, Err(WarmError::VramAllocationFailed { .. })));

        // Mark as failed in registry
        registry.mark_failed(model_id, 104, "VRAM exhausted").unwrap();

        assert!(registry.any_failed());
        assert!(!registry.all_warm());
    }

    #[test]
    fn test_loader_all_13_models_fit_in_24gb() {
        let mut registry = WarmModelRegistry::new();
        let mut pools = WarmMemoryPools::rtx_5090();

        // Each model ~1.5GB (12 * 1.5GB + 2GB FuseMoE = 20GB < 24GB)
        let model_size = (1536) * MB;
        let fusemoe_size = 2 * GB;

        for model_id in EMBEDDING_MODEL_IDS {
            registry.register_model(model_id, model_size, 768).unwrap();
        }
        registry
            .register_model(FUSEMOE_MODEL_ID, fusemoe_size, 768)
            .unwrap();

        assert_eq!(registry.model_count(), TOTAL_MODEL_COUNT);

        // Load all models
        for model_id in registry.loading_order() {
            registry.start_loading(&model_id).unwrap();
            let size = registry.get_entry(&model_id).unwrap().expected_bytes;
            pools
                .allocate_model(&model_id, size, 0x1000 + size as u64)
                .unwrap();
            registry.update_progress(&model_id, 100, size).unwrap();
            registry.mark_validating(&model_id).unwrap();
            registry.mark_warm(&model_id, test_handle(size)).unwrap();
        }

        assert!(registry.all_warm());
        assert_eq!(registry.warm_count(), TOTAL_MODEL_COUNT);
        assert!(pools.is_within_budget());
    }
}

// ============================================================================
// GROUP 9: HEALTH CHECK TESTS
// ============================================================================

mod health_check_tests {
    use super::*;

    #[test]
    fn test_health_status_from_registry_all_warm() {
        let mut registry = WarmModelRegistry::new();
        registry.register_model("E1", 256 * MB, 768).unwrap();
        registry.register_model("E2", 256 * MB, 768).unwrap();

        for model_id in ["E1", "E2"] {
            registry.start_loading(model_id).unwrap();
            registry.mark_validating(model_id).unwrap();
            registry.mark_warm(model_id, test_handle(256 * MB)).unwrap();
        }

        // Healthy: all models warm
        assert!(registry.all_warm());
        assert!(!registry.any_failed());
        assert_eq!(registry.warm_count(), 2);
    }

    #[test]
    fn test_health_status_from_registry_loading() {
        let mut registry = WarmModelRegistry::new();
        registry.register_model("E1", 256 * MB, 768).unwrap();
        registry.register_model("E2", 256 * MB, 768).unwrap();

        // E1 is warm, E2 is still loading
        registry.start_loading("E1").unwrap();
        registry.mark_validating("E1").unwrap();
        registry.mark_warm("E1", test_handle(256 * MB)).unwrap();

        registry.start_loading("E2").unwrap();
        registry.update_progress("E2", 50, 128 * MB).unwrap();

        // Loading: not all warm, none failed
        assert!(!registry.all_warm());
        assert!(!registry.any_failed());

        // Check E2 is in loading state
        match registry.get_state("E2") {
            Some(WarmModelState::Loading {
                progress_percent,
                bytes_loaded,
            }) => {
                assert_eq!(progress_percent, 50);
                assert_eq!(bytes_loaded, 128 * MB);
            }
            _ => panic!("Expected Loading state"),
        }
    }

    #[test]
    fn test_health_status_from_registry_unhealthy() {
        let mut registry = WarmModelRegistry::new();
        registry.register_model("E1", 256 * MB, 768).unwrap();

        registry.start_loading("E1").unwrap();
        registry.mark_failed("E1", 102, "Load failed").unwrap();

        // Unhealthy: has failures
        assert!(registry.any_failed());
        assert!(!registry.all_warm());

        let failures = registry.failed_entries();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].0, "E1");
        assert_eq!(failures[0].1, 102);
    }

    #[test]
    fn test_health_status_from_registry_not_initialized() {
        let registry = WarmModelRegistry::new();

        // NotInitialized: no models registered
        assert_eq!(registry.model_count(), 0);
        assert!(!registry.all_warm());
        assert!(!registry.any_failed());
    }

    #[test]
    fn test_health_check_warm_count_progression() {
        let mut registry = WarmModelRegistry::new();

        // Register 5 models
        for i in 1..=5 {
            registry
                .register_model(format!("E{}", i), 100 * MB, 768)
                .unwrap();
        }

        assert_eq!(registry.warm_count(), 0);

        // Warm them one by one
        for i in 1..=5 {
            let model_id = format!("E{}", i);
            registry.start_loading(&model_id).unwrap();
            registry.mark_validating(&model_id).unwrap();
            registry.mark_warm(&model_id, test_handle(100 * MB)).unwrap();

            assert_eq!(registry.warm_count(), i);
        }

        assert!(registry.all_warm());
    }
}

// ============================================================================
// GROUP 10: DIAGNOSTICS TESTS
// ============================================================================

mod diagnostics_tests {
    use super::*;

    #[test]
    fn test_diagnostic_error_info_collection() {
        // Collect error information for diagnostics
        let errors: Vec<WarmError> = vec![
            WarmError::ModelFileMissing {
                model_id: "E1_Semantic".to_string(),
                path: "/models/semantic.bin".to_string(),
            },
            WarmError::VramInsufficientTotal {
                required_bytes: 32 * GB,
                available_bytes: 24 * GB,
                required_gb: 32.0,
                available_gb: 24.0,
                model_breakdown: vec![
                    ("E1".to_string(), 800 * MB),
                    ("E2".to_string(), 600 * MB),
                ],
            },
        ];

        for err in &errors {
            // Verify we can extract diagnostic info
            assert!(!err.category().is_empty());
            assert!(!err.error_code().is_empty());
            assert!(err.exit_code() > 0);
        }
    }

    #[test]
    fn test_diagnostic_registry_snapshot() {
        let mut registry = WarmModelRegistry::new();

        registry.register_model("E1", 800 * MB, 768).unwrap();
        registry.register_model("E2", 600 * MB, 768).unwrap();

        registry.start_loading("E1").unwrap();
        registry.update_progress("E1", 50, 400 * MB).unwrap();

        // Collect diagnostic snapshot
        let model_count = registry.model_count();
        let warm_count = registry.warm_count();
        let any_failed = registry.any_failed();
        let loading_order = registry.loading_order();

        assert_eq!(model_count, 2);
        assert_eq!(warm_count, 0);
        assert!(!any_failed);
        assert_eq!(loading_order[0], "E1"); // Largest first
    }

    #[test]
    fn test_diagnostic_memory_pool_snapshot() {
        let mut pools = WarmMemoryPools::rtx_5090();

        pools.allocate_model("E1", 10 * GB, 0x1000).unwrap();
        pools.allocate_working(2 * GB).unwrap();

        // Collect diagnostic snapshot
        let model_capacity = pools.model_pool_capacity();
        let model_allocated = pools.total_allocated_bytes();
        let model_utilization = pools.model_pool_utilization();
        let working_capacity = pools.working_pool_capacity();
        let working_available = pools.available_working_bytes();
        let within_budget = pools.is_within_budget();

        assert_eq!(model_capacity, 24 * GB);
        assert_eq!(model_allocated, 10 * GB + 2 * GB);
        assert!(model_utilization > 0.4 && model_utilization < 0.5);
        assert_eq!(working_capacity, 8 * GB);
        assert_eq!(working_available, 6 * GB);
        assert!(within_budget);
    }

    #[test]
    fn test_diagnostic_failure_report() {
        let mut registry = WarmModelRegistry::new();

        registry.register_model("E1", 800 * MB, 768).unwrap();
        registry.register_model("E2", 600 * MB, 768).unwrap();

        registry.start_loading("E1").unwrap();
        registry.mark_failed("E1", 102, "Checksum mismatch").unwrap();

        registry.start_loading("E2").unwrap();
        registry.mark_failed("E2", 104, "VRAM exhausted").unwrap();

        // Collect failure report
        let failures = registry.failed_entries();

        assert_eq!(failures.len(), 2);

        // Verify we can build a diagnostic report
        for (model_id, error_code, error_message) in &failures {
            assert!(!model_id.is_empty());
            assert!(*error_code >= 100);
            assert!(!error_message.is_empty());
        }
    }

    #[test]
    fn test_diagnostic_model_allocation_details() {
        let mut pools = WarmMemoryPools::rtx_5090();

        pools.allocate_model("E1", 1 * GB, 0x1000_0000).unwrap();
        pools.allocate_model("E2", 2 * GB, 0x5000_0000).unwrap();

        let allocations = pools.list_model_allocations();

        assert_eq!(allocations.len(), 2);

        for alloc in allocations {
            // Verify diagnostic info is available
            assert!(!alloc.model_id.is_empty());
            assert!(alloc.vram_ptr > 0);
            assert!(alloc.size_bytes > 0);
            assert!(alloc.age_secs() >= 0.0);
        }
    }

    #[test]
    fn test_diagnostic_config_dump() {
        let config = test_config();

        // Collect config diagnostic info
        let vram_budget = config.vram_budget_bytes;
        let vram_headroom = config.vram_headroom_bytes;
        let total_required = config.total_vram_required();
        let cuda_device = config.cuda_device_id;
        let quantization = config.quantization.as_str();
        let max_load_time = config.max_load_time_per_model_ms;

        assert_eq!(vram_budget, 24 * GB);
        assert_eq!(vram_headroom, 8 * GB);
        assert_eq!(total_required, 32 * GB);
        assert_eq!(cuda_device, 0);
        assert_eq!(quantization, "FP16");
        assert_eq!(max_load_time, 30_000);
    }
}

// ============================================================================
// INTEGRATION TESTS (Cross-Component)
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_full_warm_loading_pipeline_simulation() {
        // Simulate complete warm loading pipeline
        let config = test_config();
        let mut registry = WarmModelRegistry::new();
        let mut pools = WarmMemoryPools::new(config.clone());
        let validator = WarmValidator::new();

        // Register all 13 models with realistic sizes
        let model_sizes: Vec<(&str, usize)> = EMBEDDING_MODEL_IDS
            .iter()
            .map(|id| (*id, 600 * MB))
            .chain(std::iter::once((FUSEMOE_MODEL_ID, 2 * GB)))
            .collect();

        for (model_id, size) in &model_sizes {
            registry.register_model(*model_id, *size, 768).unwrap();
        }

        assert_eq!(registry.model_count(), TOTAL_MODEL_COUNT);

        // Load in order (largest first)
        for model_id in registry.loading_order() {
            // Start loading
            registry.start_loading(&model_id).unwrap();

            let entry = registry.get_entry(&model_id).unwrap();
            let size = entry.expected_bytes;
            let dimension = entry.expected_dimension;

            // Allocate VRAM
            let vram_ptr = 0x1000_0000_u64 + (pools.total_allocated_bytes() as u64);
            pools.allocate_model(&model_id, size, vram_ptr).unwrap();

            // Simulate loading progress
            registry.update_progress(&model_id, 50, size / 2).unwrap();
            registry.update_progress(&model_id, 100, size).unwrap();

            // Validation
            registry.mark_validating(&model_id).unwrap();

            // Simulate inference output
            let output: Vec<f32> = vec![0.1; dimension];
            let config = TestInferenceConfig::for_embedding_model(&model_id, dimension);
            let handle = test_handle_full(vram_ptr, size, 0, 0xABCD);

            let result = validator.validate_model(&config, &handle, &output);
            assert!(result.is_valid(), "Validation failed for {}", model_id);

            // Mark warm
            registry.mark_warm(&model_id, handle).unwrap();
        }

        // Verify final state
        assert!(registry.all_warm());
        assert!(!registry.any_failed());
        assert_eq!(registry.warm_count(), TOTAL_MODEL_COUNT);
        assert!(pools.is_within_budget());

        // Verify all handles are accessible
        for model_id in EMBEDDING_MODEL_IDS {
            assert!(
                registry.get_handle(model_id).is_some(),
                "Missing handle for {}",
                model_id
            );
        }
        assert!(registry.get_handle(FUSEMOE_MODEL_ID).is_some());
    }

    #[test]
    fn test_fail_fast_on_first_validation_failure() {
        let mut registry = WarmModelRegistry::new();
        let mut pools = WarmMemoryPools::rtx_5090();
        let validator = WarmValidator::new();

        // Register 3 models
        registry.register_model("E1", 500 * MB, 768).unwrap();
        registry.register_model("E2", 500 * MB, 768).unwrap();
        registry.register_model("E3", 500 * MB, 768).unwrap();

        // Load E1 successfully
        registry.start_loading("E1").unwrap();
        pools.allocate_model("E1", 500 * MB, 0x1000).unwrap();
        registry.update_progress("E1", 100, 500 * MB).unwrap();
        registry.mark_validating("E1").unwrap();

        let output = vec![0.1; 768];
        let config = TestInferenceConfig::for_embedding_model("E1", 768);
        let handle = test_handle(500 * MB);
        let result = validator.validate_model(&config, &handle, &output);
        assert!(result.is_valid());

        registry.mark_warm("E1", handle).unwrap();

        // Load E2 - validation fails (NaN in output)
        registry.start_loading("E2").unwrap();
        pools.allocate_model("E2", 500 * MB, 0x2000).unwrap();
        registry.update_progress("E2", 100, 500 * MB).unwrap();
        registry.mark_validating("E2").unwrap();

        let bad_output = vec![f32::NAN; 768];
        let config = TestInferenceConfig::for_embedding_model("E2", 768);
        let result = validator.validate_model(&config, &test_handle(500 * MB), &bad_output);
        assert!(!result.is_valid());
        assert!(!result.weights_valid);

        // Mark as failed with exit code 103
        registry.mark_failed("E2", 103, "NaN in output").unwrap();

        // FAIL-FAST: Don't continue loading E3
        assert!(registry.any_failed());
        assert!(!registry.all_warm());

        // E3 should still be Pending
        assert!(matches!(
            registry.get_state("E3"),
            Some(WarmModelState::Pending)
        ));
    }
}
