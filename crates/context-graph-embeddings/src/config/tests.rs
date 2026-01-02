//! Tests for configuration module.

use std::env;
use std::io::Write;
use std::path::PathBuf;

use serial_test::serial;
use tempfile::NamedTempFile;

use super::*;

// =========================================================================
// DEFAULT TESTS (5 tests)
// =========================================================================

#[test]
fn test_embedding_config_default() {
    let config = EmbeddingConfig::default();

    // Verify defaults match constitution.yaml specs
    assert_eq!(config.batch.max_batch_size, 32);
    assert_eq!(config.batch.max_wait_ms, 50);
    assert_eq!(config.fusion.num_experts, 8);
    assert_eq!(config.fusion.top_k, 4); // FIXED: constitution.yaml says 4, not 2
    assert_eq!(config.fusion.output_dim, 1536);
    assert_eq!(config.cache.max_entries, 100_000);
    assert!(config.gpu.enabled);
}

#[test]
fn test_model_registry_config_default() {
    let config = ModelPathConfig::default();
    assert_eq!(config.models_dir, "./models");
    assert!(config.lazy_loading);
    assert!(config.preload_models.is_empty());
    assert_eq!(config.max_loaded_models, 12);
}

#[test]
fn test_batch_config_default() {
    let config = BatchConfig::default();
    assert_eq!(config.max_batch_size, 32);
    assert_eq!(config.min_batch_size, 1);
    assert_eq!(config.max_wait_ms, 50);
    assert!(config.dynamic_batching);
    assert!(config.sort_by_length);
    assert_eq!(config.padding_strategy, PaddingStrategy::DynamicMax);
}

#[test]
fn test_fusion_config_default() {
    let config = FusionConfig::default();
    assert_eq!(config.num_experts, 8);
    assert_eq!(config.top_k, 4); // FIXED: constitution.yaml says 4, not 2
    assert_eq!(config.output_dim, 1536);
    assert_eq!(config.expert_hidden_dim, 4096);
    assert!((config.temperature - 1.0).abs() < f32::EPSILON);
    assert!((config.noise_std - 0.0).abs() < f32::EPSILON);
    assert!((config.laplace_alpha - 0.01).abs() < f32::EPSILON);
    assert!((config.capacity_factor - 1.25).abs() < f32::EPSILON);
    assert!((config.load_balance_coef - 0.01).abs() < f32::EPSILON);
}

#[test]
fn test_gpu_config_default() {
    let config = GpuConfig::default();
    assert!(config.enabled);
    assert_eq!(config.device_ids, vec![0]);
    assert!((config.memory_fraction - 0.9).abs() < f32::EPSILON);
    assert!(config.use_cuda_graphs);
    assert!(config.mixed_precision);
    assert!(!config.green_contexts);
    assert!(!config.gds_enabled);
}

// =========================================================================
// VALIDATION TESTS (12 tests)
// =========================================================================

#[test]
fn test_default_config_validates() {
    let config = EmbeddingConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_model_registry_empty_dir_fails() {
    let config = ModelPathConfig {
        models_dir: "".to_string(),
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("models_dir"));
}

#[test]
fn test_model_registry_invalid_preload_fails() {
    let config = ModelPathConfig {
        preload_models: vec!["invalid_model".to_string()],
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("invalid_model"));
}

#[test]
fn test_model_registry_valid_preload_succeeds() {
    let config = ModelPathConfig {
        preload_models: vec!["semantic".to_string(), "code".to_string()],
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_batch_zero_size_fails() {
    let config = BatchConfig {
        max_batch_size: 0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("max_batch_size"));
}

#[test]
fn test_batch_zero_wait_with_min_batch_greater_than_one_fails() {
    // max_wait_ms=0 is only invalid when min_batch_size > 1
    let config = BatchConfig {
        max_wait_ms: 0,
        min_batch_size: 4,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("max_wait_ms"));
}

#[test]
fn test_batch_zero_wait_with_min_batch_one_succeeds() {
    // Special case: max_wait_ms=0 is OK if min_batch_size=1
    let config = BatchConfig {
        min_batch_size: 1,
        max_wait_ms: 0,
        max_batch_size: 32,
        dynamic_batching: true,
        padding_strategy: PaddingStrategy::DynamicMax,
        sort_by_length: true,
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_batch_min_exceeds_max_fails() {
    let config = BatchConfig {
        min_batch_size: 64,
        max_batch_size: 32,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("min_batch_size"));
    assert!(msg.contains("cannot exceed"));
}

#[test]
fn test_fusion_zero_experts_fails() {
    let config = FusionConfig {
        num_experts: 0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("num_experts"));
}

#[test]
fn test_fusion_top_k_exceeds_experts_fails() {
    let config = FusionConfig {
        num_experts: 4,
        top_k: 8,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("top_k"));
}

#[test]
fn test_fusion_negative_laplace_fails() {
    let config = FusionConfig {
        laplace_alpha: -0.1,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("laplace_alpha"));
}

#[test]
fn test_cache_enabled_zero_entries_fails() {
    let config = CacheConfig {
        enabled: true,
        max_entries: 0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("max_entries"));
}

#[test]
fn test_gpu_empty_device_ids_when_enabled_fails() {
    let config = GpuConfig {
        enabled: true,
        device_ids: vec![],
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("device_ids"));
}

#[test]
fn test_gpu_memory_fraction_zero_fails() {
    let config = GpuConfig {
        memory_fraction: 0.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("memory_fraction"));
}

#[test]
fn test_gpu_memory_fraction_above_one_fails() {
    let config = GpuConfig {
        memory_fraction: 1.1,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("memory_fraction"));
}

#[test]
fn test_gpu_memory_fraction_nan_fails() {
    let config = GpuConfig {
        memory_fraction: f32::NAN,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("memory_fraction"));
}

// =========================================================================
// SERDE ROUNDTRIP TESTS (5 tests)
// =========================================================================

#[test]
fn test_serde_roundtrip_json() {
    let original = EmbeddingConfig::default();
    let json = serde_json::to_string(&original).unwrap();
    let restored: EmbeddingConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(original.batch.max_batch_size, restored.batch.max_batch_size);
    assert_eq!(original.fusion.num_experts, restored.fusion.num_experts);
}

#[test]
fn test_serde_roundtrip_toml() {
    let original = EmbeddingConfig::default();
    let toml_str = original.to_toml_string().unwrap();
    let restored = EmbeddingConfig::from_toml_str(&toml_str).unwrap();

    assert_eq!(original.batch.max_batch_size, restored.batch.max_batch_size);
    assert_eq!(original.fusion.num_experts, restored.fusion.num_experts);
}

#[test]
fn test_from_toml_str_custom_values() {
    let toml = r#"
[batch]
max_batch_size = 64
max_wait_ms = 100

[fusion]
num_experts = 16
top_k = 4

[cache]
enabled = false
"#;
    let config = EmbeddingConfig::from_toml_str(toml).unwrap();

    assert_eq!(config.batch.max_batch_size, 64);
    assert_eq!(config.batch.max_wait_ms, 100);
    assert_eq!(config.fusion.num_experts, 16);
    assert_eq!(config.fusion.top_k, 4);
    assert!(!config.cache.enabled);
}

#[test]
fn test_from_toml_str_partial_config() {
    // Only specify some values, rest should be defaults
    let toml = r#"
[gpu]
enabled = false
"#;
    let config = EmbeddingConfig::from_toml_str(toml).unwrap();

    assert!(!config.gpu.enabled);
    // Defaults still apply
    assert_eq!(config.batch.max_batch_size, 32);
    assert_eq!(config.fusion.num_experts, 8);
}

#[test]
fn test_from_toml_str_invalid_fails() {
    let toml = "invalid { toml } content";
    let result = EmbeddingConfig::from_toml_str(toml);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("TOML"));
}

// =========================================================================
// FILE LOADING TESTS (4 tests)
// =========================================================================

#[test]
fn test_from_file_success() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "[batch]").unwrap();
    writeln!(file, "max_batch_size = 128").unwrap();

    let config = EmbeddingConfig::from_file(file.path()).unwrap();
    assert_eq!(config.batch.max_batch_size, 128);
}

#[test]
fn test_from_file_missing_returns_config_error() {
    let result = EmbeddingConfig::from_file("/nonexistent/path/config.toml");
    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        EmbeddingError::ConfigError { message } => {
            assert!(message.contains("nonexistent"));
        }
        _ => panic!("Expected ConfigError, got {:?}", err),
    }
}

#[test]
fn test_from_file_invalid_toml_returns_config_error() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "not valid toml {{}}").unwrap();

    let result = EmbeddingConfig::from_file(file.path());
    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        EmbeddingError::ConfigError { message } => {
            assert!(message.contains("TOML"));
        }
        _ => panic!("Expected ConfigError, got {:?}", err),
    }
}

#[test]
fn test_from_file_empty_uses_defaults() {
    let file = NamedTempFile::new().unwrap();
    // Empty file

    let config = EmbeddingConfig::from_file(file.path()).unwrap();
    assert_eq!(config.batch.max_batch_size, 32); // Default
}

// =========================================================================
// ENVIRONMENT OVERRIDE TESTS (6 tests)
// These tests must run serially to avoid env var pollution
// =========================================================================

#[test]
#[serial]
fn test_env_override_models_dir() {
    env::set_var("EMBEDDING_MODELS_DIR", "/custom/models");
    let config = EmbeddingConfig::default().with_env_overrides();
    env::remove_var("EMBEDDING_MODELS_DIR");

    assert_eq!(config.models.models_dir, "/custom/models");
}

#[test]
#[serial]
fn test_env_override_gpu_enabled() {
    env::set_var("EMBEDDING_GPU_ENABLED", "false");
    let config = EmbeddingConfig::default().with_env_overrides();
    env::remove_var("EMBEDDING_GPU_ENABLED");

    assert!(!config.gpu.enabled);
}

#[test]
#[serial]
fn test_env_override_cache_max_entries() {
    env::set_var("EMBEDDING_CACHE_MAX_ENTRIES", "50000");
    let config = EmbeddingConfig::default().with_env_overrides();
    env::remove_var("EMBEDDING_CACHE_MAX_ENTRIES");

    assert_eq!(config.cache.max_entries, 50000);
}

#[test]
#[serial]
fn test_env_override_batch_max_size() {
    env::set_var("EMBEDDING_BATCH_MAX_SIZE", "64");
    let config = EmbeddingConfig::default().with_env_overrides();
    env::remove_var("EMBEDDING_BATCH_MAX_SIZE");

    assert_eq!(config.batch.max_batch_size, 64);
}

#[test]
#[serial]
fn test_env_override_invalid_value_ignored() {
    env::set_var("EMBEDDING_GPU_ENABLED", "not_a_bool");
    let config = EmbeddingConfig::default().with_env_overrides();
    env::remove_var("EMBEDDING_GPU_ENABLED");

    // Should keep default because "not_a_bool" can't be parsed
    assert!(config.gpu.enabled);
}

#[test]
#[serial]
fn test_env_override_lazy_loading() {
    env::set_var("EMBEDDING_LAZY_LOADING", "false");
    let config = EmbeddingConfig::default().with_env_overrides();
    env::remove_var("EMBEDDING_LAZY_LOADING");

    assert!(!config.models.lazy_loading);
}

// =========================================================================
// CONSTITUTION COMPLIANCE TESTS (5 tests)
// =========================================================================

#[test]
fn test_constitution_batch_defaults() {
    // constitution.yaml: max_batch_size = 32, max_wait_ms = 50
    let config = BatchConfig::default();
    assert_eq!(config.max_batch_size, 32);
    assert_eq!(config.max_wait_ms, 50);
}

#[test]
fn test_constitution_fusion_defaults() {
    // constitution.yaml: num_experts = 8, top_k = 4 (NOT 2!), output_dim = 1536
    let config = FusionConfig::default();
    assert_eq!(config.num_experts, 8);
    assert_eq!(config.top_k, 4); // FIXED: constitution.yaml fuse_moe.top_k = 4
    assert_eq!(config.output_dim, 1536);
    assert_eq!(config.expert_hidden_dim, 4096);
}

#[test]
fn test_constitution_cache_defaults() {
    // constitution.yaml: max_entries = 100000
    let config = CacheConfig::default();
    assert_eq!(config.max_entries, 100_000);
}

#[test]
fn test_constitution_gpu_memory_fraction() {
    // constitution.yaml: <24GB of 32GB = 0.75 max, default 0.9
    let config = GpuConfig::default();
    assert!((config.memory_fraction - 0.9).abs() < f32::EPSILON);
}

#[test]
fn test_fusion_laplace_alpha() {
    // constitution.yaml: fuse_moe.laplace_alpha = 0.01
    let config = FusionConfig::default();
    assert!((config.laplace_alpha - 0.01).abs() < f32::EPSILON);
}

// =========================================================================
// EDGE CASE TESTS (4 tests)
// =========================================================================

#[test]
fn test_fusion_nan_laplace_fails() {
    let config = FusionConfig {
        laplace_alpha: f32::NAN,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_fusion_capacity_factor_below_one_fails() {
    let config = FusionConfig {
        capacity_factor: 0.9,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_cache_disabled_with_zero_entries_succeeds() {
    // Zero entries is OK if cache is disabled
    let config = CacheConfig {
        enabled: false,
        max_entries: 0,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_nested_validation_error_includes_section() {
    let mut config = EmbeddingConfig::default();
    config.batch.max_batch_size = 0;

    let result = config.validate();
    assert!(result.is_err());
    // Error message should include [batch] section
    assert!(result.unwrap_err().to_string().contains("[batch]"));
}

// =========================================================================
// PADDING STRATEGY TESTS (6 tests)
// =========================================================================

#[test]
fn test_padding_strategy_default_is_dynamic_max() {
    assert_eq!(PaddingStrategy::default(), PaddingStrategy::DynamicMax);
}

#[test]
fn test_padding_strategy_all_variants() {
    let all = PaddingStrategy::all();
    assert_eq!(all.len(), 4);
    assert!(all.contains(&PaddingStrategy::MaxLength));
    assert!(all.contains(&PaddingStrategy::DynamicMax));
    assert!(all.contains(&PaddingStrategy::PowerOfTwo));
    assert!(all.contains(&PaddingStrategy::Bucket));
}

#[test]
fn test_padding_strategy_as_str() {
    assert_eq!(PaddingStrategy::MaxLength.as_str(), "max_length");
    assert_eq!(PaddingStrategy::DynamicMax.as_str(), "dynamic_max");
    assert_eq!(PaddingStrategy::PowerOfTwo.as_str(), "power_of_two");
    assert_eq!(PaddingStrategy::Bucket.as_str(), "bucket");
}

#[test]
fn test_padding_strategy_serde_roundtrip() {
    for strategy in PaddingStrategy::all() {
        let json = serde_json::to_string(strategy).unwrap();
        let restored: PaddingStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(*strategy, restored);
    }
}

#[test]
fn test_padding_strategy_serde_snake_case() {
    // Verify snake_case serialization
    let json = serde_json::to_string(&PaddingStrategy::DynamicMax).unwrap();
    assert_eq!(json, "\"dynamic_max\"");

    let json = serde_json::to_string(&PaddingStrategy::PowerOfTwo).unwrap();
    assert_eq!(json, "\"power_of_two\"");
}

#[test]
fn test_padding_strategy_copy() {
    // PaddingStrategy must be Copy for efficiency
    let a = PaddingStrategy::Bucket;
    let b = a; // Copy
    assert_eq!(a, b);
}

// =========================================================================
// BATCH CONFIG NEW FIELD TESTS (3 tests)
// =========================================================================

#[test]
fn test_batch_config_new_defaults() {
    let config = BatchConfig::default();
    assert_eq!(config.min_batch_size, 1);
    assert!(config.dynamic_batching);
    assert_eq!(config.padding_strategy, PaddingStrategy::DynamicMax);
}

#[test]
fn test_batch_config_toml_roundtrip() {
    let original = BatchConfig {
        max_batch_size: 64,
        min_batch_size: 4,
        max_wait_ms: 100,
        dynamic_batching: false,
        padding_strategy: PaddingStrategy::PowerOfTwo,
        sort_by_length: false,
    };

    let toml_str = toml::to_string(&original).unwrap();
    let restored: BatchConfig = toml::from_str(&toml_str).unwrap();

    assert_eq!(original.max_batch_size, restored.max_batch_size);
    assert_eq!(original.min_batch_size, restored.min_batch_size);
    assert_eq!(original.max_wait_ms, restored.max_wait_ms);
    assert_eq!(original.dynamic_batching, restored.dynamic_batching);
    assert_eq!(original.padding_strategy, restored.padding_strategy);
    assert_eq!(original.sort_by_length, restored.sort_by_length);
}

#[test]
fn test_batch_config_partial_toml_uses_defaults() {
    // Only specify max_batch_size, rest should be defaults
    let toml_str = r#"
max_batch_size = 64
"#;
    let config: BatchConfig = toml::from_str(toml_str).unwrap();

    assert_eq!(config.max_batch_size, 64);
    assert_eq!(config.min_batch_size, 1); // default
    assert_eq!(config.max_wait_ms, 50); // default
    assert!(config.dynamic_batching); // default
    assert_eq!(config.padding_strategy, PaddingStrategy::DynamicMax); // default
    assert!(config.sort_by_length); // default
}

// =========================================================================
// FUSION CONFIG NEW FIELD TESTS (M03-F14)
// =========================================================================

#[test]
fn test_fusion_config_default_top_k_is_4() {
    // CRITICAL: constitution.yaml specifies top_k = 4, NOT 2
    let config = FusionConfig::default();
    assert_eq!(config.top_k, 4);
}

#[test]
fn test_fusion_config_default_expert_hidden_dim() {
    let config = FusionConfig::default();
    assert_eq!(config.expert_hidden_dim, 4096);
}

#[test]
fn test_fusion_config_default_temperature() {
    let config = FusionConfig::default();
    assert!((config.temperature - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_fusion_config_default_noise_std() {
    let config = FusionConfig::default();
    assert!((config.noise_std - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_fusion_config_default_load_balance_coef() {
    let config = FusionConfig::default();
    assert!((config.load_balance_coef - 0.01).abs() < f32::EPSILON);
}

#[test]
fn test_fusion_validate_valid_default() {
    let config = FusionConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_fusion_validate_num_experts_zero() {
    let config = FusionConfig {
        num_experts: 0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("num_experts"));
}

#[test]
fn test_fusion_validate_top_k_zero() {
    let config = FusionConfig {
        top_k: 0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("top_k"));
}

#[test]
fn test_fusion_validate_top_k_exceeds_experts() {
    let config = FusionConfig {
        num_experts: 4,
        top_k: 8,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("top_k"));
}

#[test]
fn test_fusion_validate_top_k_equals_experts_succeeds() {
    // Edge case: top_k = num_experts should be valid
    let config = FusionConfig {
        num_experts: 8,
        top_k: 8,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_fusion_validate_output_dim_zero() {
    let config = FusionConfig {
        output_dim: 0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("output_dim"));
}

#[test]
fn test_fusion_validate_expert_hidden_dim_zero() {
    let config = FusionConfig {
        expert_hidden_dim: 0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("expert_hidden_dim"));
}

#[test]
fn test_fusion_validate_temperature_zero() {
    let config = FusionConfig {
        temperature: 0.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("temperature"));
}

#[test]
fn test_fusion_validate_temperature_negative() {
    let config = FusionConfig {
        temperature: -1.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("temperature"));
}

#[test]
fn test_fusion_validate_temperature_nan() {
    let config = FusionConfig {
        temperature: f32::NAN,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("temperature"));
}

#[test]
fn test_fusion_validate_temperature_very_small_succeeds() {
    // Edge case: very small temperature (still > 0) should be valid
    let config = FusionConfig {
        temperature: 0.001,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_fusion_validate_noise_std_negative() {
    let config = FusionConfig {
        noise_std: -0.1,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("noise_std"));
}

#[test]
fn test_fusion_validate_noise_std_nan() {
    let config = FusionConfig {
        noise_std: f32::NAN,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("noise_std"));
}

#[test]
fn test_fusion_validate_load_balance_coef_negative() {
    let config = FusionConfig {
        load_balance_coef: -0.01,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("load_balance_coef"));
}

#[test]
fn test_fusion_validate_load_balance_coef_nan() {
    let config = FusionConfig {
        load_balance_coef: f32::NAN,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("load_balance_coef"));
}

#[test]
fn test_fusion_for_inference_no_noise() {
    let config = FusionConfig::for_inference();
    assert!((config.noise_std - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_fusion_for_inference_no_load_balance() {
    let config = FusionConfig::for_inference();
    assert!((config.load_balance_coef - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_fusion_for_inference_validates() {
    let config = FusionConfig::for_inference();
    assert!(config.validate().is_ok());
}

#[test]
fn test_fusion_for_training_has_noise() {
    let config = FusionConfig::for_training();
    assert!((config.noise_std - 0.1).abs() < f32::EPSILON);
}

#[test]
fn test_fusion_for_training_has_load_balance() {
    let config = FusionConfig::for_training();
    assert!((config.load_balance_coef - 0.01).abs() < f32::EPSILON);
}

#[test]
fn test_fusion_for_training_validates() {
    let config = FusionConfig::for_training();
    assert!(config.validate().is_ok());
}

#[test]
fn test_fusion_is_inference_mode_true() {
    let config = FusionConfig::for_inference();
    assert!(config.is_inference_mode());
}

#[test]
fn test_fusion_is_inference_mode_false() {
    let config = FusionConfig::for_training();
    assert!(!config.is_inference_mode());
}

#[test]
fn test_fusion_is_inference_mode_default() {
    // Default config has noise_std = 0.0, so should be inference mode
    let config = FusionConfig::default();
    assert!(config.is_inference_mode());
}

#[test]
fn test_fusion_serde_roundtrip_json() {
    let original = FusionConfig::default();
    let json = serde_json::to_string(&original).unwrap();
    let restored: FusionConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(original.num_experts, restored.num_experts);
    assert_eq!(original.top_k, restored.top_k);
    assert_eq!(original.output_dim, restored.output_dim);
    assert_eq!(original.expert_hidden_dim, restored.expert_hidden_dim);
    assert!((original.temperature - restored.temperature).abs() < f32::EPSILON);
    assert!((original.noise_std - restored.noise_std).abs() < f32::EPSILON);
    assert!((original.load_balance_coef - restored.load_balance_coef).abs() < f32::EPSILON);
    assert!((original.capacity_factor - restored.capacity_factor).abs() < f32::EPSILON);
    assert!((original.laplace_alpha - restored.laplace_alpha).abs() < f32::EPSILON);
}

#[test]
fn test_fusion_serde_roundtrip_toml() {
    let original = FusionConfig {
        num_experts: 16,
        top_k: 6,
        output_dim: 2048,
        expert_hidden_dim: 8192,
        load_balance_coef: 0.05,
        capacity_factor: 1.5,
        temperature: 0.8,
        noise_std: 0.2,
        laplace_alpha: 0.02,
    };

    let toml_str = toml::to_string(&original).unwrap();
    let restored: FusionConfig = toml::from_str(&toml_str).unwrap();

    assert_eq!(original.num_experts, restored.num_experts);
    assert_eq!(original.top_k, restored.top_k);
    assert_eq!(original.output_dim, restored.output_dim);
    assert_eq!(original.expert_hidden_dim, restored.expert_hidden_dim);
    assert!((original.temperature - restored.temperature).abs() < f32::EPSILON);
    assert!((original.noise_std - restored.noise_std).abs() < f32::EPSILON);
    assert!((original.load_balance_coef - restored.load_balance_coef).abs() < f32::EPSILON);
    assert!((original.capacity_factor - restored.capacity_factor).abs() < f32::EPSILON);
    assert!((original.laplace_alpha - restored.laplace_alpha).abs() < f32::EPSILON);
}

#[test]
fn test_fusion_serde_partial_toml_uses_defaults() {
    // Only specify num_experts and top_k, rest should use defaults
    let toml_str = r#"
num_experts = 16
top_k = 8
"#;
    let config: FusionConfig = toml::from_str(toml_str).unwrap();

    assert_eq!(config.num_experts, 16);
    assert_eq!(config.top_k, 8);
    assert_eq!(config.output_dim, 1536); // default
    assert_eq!(config.expert_hidden_dim, 4096); // default
    assert!((config.temperature - 1.0).abs() < f32::EPSILON); // default
    assert!((config.noise_std - 0.0).abs() < f32::EPSILON); // default
    assert!((config.load_balance_coef - 0.01).abs() < f32::EPSILON); // default
}

// =========================================================================
// EDGE CASE TESTS FOR FUSION CONFIG
// =========================================================================

#[test]
fn test_fusion_edge_case_boundary_values() {
    // Test boundary values that should pass
    let config = FusionConfig {
        num_experts: 1,
        top_k: 1,
        output_dim: 1,
        expert_hidden_dim: 1,
        temperature: 0.0001, // Very small but > 0
        noise_std: 0.0,
        load_balance_coef: 0.0,
        capacity_factor: 1.0, // Exactly 1.0
        laplace_alpha: 0.0, // Exactly 0
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_fusion_edge_case_large_values() {
    // Test large but valid values
    let config = FusionConfig {
        num_experts: 1024,
        top_k: 512,
        output_dim: 65536,
        expert_hidden_dim: 32768,
        temperature: 100.0,
        noise_std: 10.0,
        load_balance_coef: 1.0,
        capacity_factor: 10.0,
        laplace_alpha: 1.0,
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_fusion_capacity_factor_nan() {
    let config = FusionConfig {
        capacity_factor: f32::NAN,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("capacity_factor"));
}

#[test]
fn test_fusion_laplace_alpha_nan() {
    let config = FusionConfig {
        laplace_alpha: f32::NAN,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("laplace_alpha"));
}

#[test]
fn test_fusion_laplace_alpha_negative() {
    let config = FusionConfig {
        laplace_alpha: -0.001,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("laplace_alpha"));
}

// =========================================================================
// EVICTION POLICY TESTS (M03-F15)
// =========================================================================

#[test]
fn test_eviction_policy_default() {
    assert_eq!(EvictionPolicy::default(), EvictionPolicy::Lru);
}

#[test]
fn test_eviction_policy_all_variants() {
    let all = EvictionPolicy::all();
    assert_eq!(all.len(), 4);
    assert!(all.contains(&EvictionPolicy::Lru));
    assert!(all.contains(&EvictionPolicy::Lfu));
    assert!(all.contains(&EvictionPolicy::TtlLru));
    assert!(all.contains(&EvictionPolicy::Arc));
}

#[test]
fn test_eviction_policy_as_str() {
    assert_eq!(EvictionPolicy::Lru.as_str(), "lru");
    assert_eq!(EvictionPolicy::Lfu.as_str(), "lfu");
    assert_eq!(EvictionPolicy::TtlLru.as_str(), "ttl_lru");
    assert_eq!(EvictionPolicy::Arc.as_str(), "arc");
}

#[test]
fn test_eviction_policy_serde_roundtrip() {
    for policy in EvictionPolicy::all() {
        let json = serde_json::to_string(policy).unwrap();
        let restored: EvictionPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(*policy, restored);
    }
}

#[test]
fn test_eviction_policy_serde_snake_case() {
    // Verify snake_case serialization
    let json = serde_json::to_string(&EvictionPolicy::TtlLru).unwrap();
    assert_eq!(json, "\"ttl_lru\"");

    let json = serde_json::to_string(&EvictionPolicy::Arc).unwrap();
    assert_eq!(json, "\"arc\"");
}

#[test]
fn test_eviction_policy_copy() {
    // EvictionPolicy must be Copy for efficiency
    let a = EvictionPolicy::Arc;
    let b = a; // Copy
    assert_eq!(a, b);
}

// =========================================================================
// CACHE CONFIG NEW TESTS (M03-F15)
// =========================================================================

#[test]
fn test_cache_config_default_enabled() {
    let config = CacheConfig::default();
    assert!(config.enabled);
}

#[test]
fn test_cache_config_default_max_entries() {
    let config = CacheConfig::default();
    assert_eq!(config.max_entries, 100_000);
}

#[test]
fn test_cache_config_default_max_bytes() {
    let config = CacheConfig::default();
    assert_eq!(config.max_bytes, 1_073_741_824);
}

#[test]
fn test_cache_config_default_ttl_seconds() {
    let config = CacheConfig::default();
    assert_eq!(config.ttl_seconds, None);
}

#[test]
fn test_cache_config_default_eviction_policy() {
    let config = CacheConfig::default();
    assert_eq!(config.eviction_policy, EvictionPolicy::Lru);
}

#[test]
fn test_cache_config_default_persist_to_disk() {
    let config = CacheConfig::default();
    assert!(!config.persist_to_disk);
}

#[test]
fn test_cache_config_default_disk_path() {
    let config = CacheConfig::default();
    assert_eq!(config.disk_path, None);
}

#[test]
fn test_cache_validate_max_bytes_zero_fails() {
    let config = CacheConfig {
        enabled: true,
        max_entries: 100,
        max_bytes: 0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("max_bytes"));
}

#[test]
fn test_cache_validate_persist_without_path_fails() {
    let config = CacheConfig {
        persist_to_disk: true,
        disk_path: None,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("disk_path"));
}

#[test]
fn test_cache_validate_persist_with_path_succeeds() {
    let config = CacheConfig {
        persist_to_disk: true,
        disk_path: Some(PathBuf::from("/tmp/cache")),
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_cache_disabled_allows_zero_bytes() {
    let config = CacheConfig {
        enabled: false,
        max_entries: 0,
        max_bytes: 0,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_cache_disabled_constructor() {
    let config = CacheConfig::disabled();
    assert!(!config.enabled);
    // Other fields should still have defaults
    assert_eq!(config.max_entries, 100_000);
    assert_eq!(config.max_bytes, 1_073_741_824);
    assert_eq!(config.eviction_policy, EvictionPolicy::Lru);
}

#[test]
fn test_cache_bytes_per_entry() {
    let config = CacheConfig::default();
    // 1GB / 100K = 10,737 bytes per entry
    assert_eq!(config.bytes_per_entry(), 10737);
}

#[test]
fn test_cache_bytes_per_entry_zero_entries() {
    let config = CacheConfig {
        max_entries: 0,
        ..Default::default()
    };
    assert_eq!(config.bytes_per_entry(), 0);
}

#[test]
fn test_cache_serde_roundtrip() {
    let original = CacheConfig {
        enabled: true,
        max_entries: 50_000,
        max_bytes: 500_000_000,
        ttl_seconds: Some(3600),
        eviction_policy: EvictionPolicy::Lfu,
        persist_to_disk: true,
        disk_path: Some(PathBuf::from("/var/cache")),
    };
    let json = serde_json::to_string(&original).unwrap();
    let restored: CacheConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(original.enabled, restored.enabled);
    assert_eq!(original.max_entries, restored.max_entries);
    assert_eq!(original.max_bytes, restored.max_bytes);
    assert_eq!(original.ttl_seconds, restored.ttl_seconds);
    assert_eq!(original.eviction_policy, restored.eviction_policy);
    assert_eq!(original.persist_to_disk, restored.persist_to_disk);
    assert_eq!(original.disk_path, restored.disk_path);
}

#[test]
fn test_cache_toml_with_all_new_fields() {
    let toml_str = r#"
enabled = true
max_entries = 50000
max_bytes = 500000000
ttl_seconds = 3600
eviction_policy = "lfu"
persist_to_disk = true
disk_path = "/var/cache"
"#;
    let config: CacheConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.max_entries, 50_000);
    assert_eq!(config.max_bytes, 500_000_000);
    assert_eq!(config.ttl_seconds, Some(3600));
    assert_eq!(config.eviction_policy, EvictionPolicy::Lfu);
    assert!(config.persist_to_disk);
    assert_eq!(config.disk_path, Some(PathBuf::from("/var/cache")));
}

// =========================================================================
// GPU CONFIG NEW TESTS (M03-F15)
// =========================================================================

#[test]
fn test_gpu_config_default_enabled() {
    let config = GpuConfig::default();
    assert!(config.enabled);
}

#[test]
fn test_gpu_config_default_device_ids() {
    let config = GpuConfig::default();
    assert_eq!(config.device_ids, vec![0]);
}

#[test]
fn test_gpu_config_default_memory_fraction() {
    let config = GpuConfig::default();
    assert!((config.memory_fraction - 0.9).abs() < f32::EPSILON);
}

#[test]
fn test_gpu_config_default_use_cuda_graphs() {
    let config = GpuConfig::default();
    assert!(config.use_cuda_graphs);
}

#[test]
fn test_gpu_config_default_mixed_precision() {
    let config = GpuConfig::default();
    assert!(config.mixed_precision);
}

#[test]
fn test_gpu_config_default_green_contexts() {
    let config = GpuConfig::default();
    assert!(!config.green_contexts);
}

#[test]
fn test_gpu_config_default_gds_enabled() {
    let config = GpuConfig::default();
    assert!(!config.gds_enabled);
}

#[test]
fn test_gpu_cpu_only_enabled() {
    let config = GpuConfig::cpu_only();
    assert!(!config.enabled);
}

#[test]
fn test_gpu_cpu_only_device_ids() {
    let config = GpuConfig::cpu_only();
    assert!(config.device_ids.is_empty());
}

#[test]
fn test_gpu_cpu_only_memory_fraction() {
    let config = GpuConfig::cpu_only();
    assert!((config.memory_fraction - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_gpu_cpu_only_all_features_disabled() {
    let config = GpuConfig::cpu_only();
    assert!(!config.use_cuda_graphs);
    assert!(!config.mixed_precision);
    assert!(!config.green_contexts);
    assert!(!config.gds_enabled);
}

#[test]
fn test_gpu_rtx_5090_optimized_enabled() {
    let config = GpuConfig::rtx_5090_optimized();
    assert!(config.enabled);
}

#[test]
fn test_gpu_rtx_5090_optimized_memory_fraction() {
    let config = GpuConfig::rtx_5090_optimized();
    assert!((config.memory_fraction - 0.75).abs() < f32::EPSILON);
}

#[test]
fn test_gpu_rtx_5090_optimized_all_features_enabled() {
    let config = GpuConfig::rtx_5090_optimized();
    assert!(config.use_cuda_graphs);
    assert!(config.mixed_precision);
    assert!(config.green_contexts);
    assert!(config.gds_enabled);
}

#[test]
fn test_gpu_is_gpu_enabled_true() {
    let config = GpuConfig::default();
    assert!(config.is_gpu_enabled());
}

#[test]
fn test_gpu_is_gpu_enabled_false_when_disabled() {
    let config = GpuConfig {
        enabled: false,
        ..Default::default()
    };
    assert!(!config.is_gpu_enabled());
}

#[test]
fn test_gpu_is_gpu_enabled_false_when_empty_devices() {
    let config = GpuConfig {
        enabled: true,
        device_ids: vec![],
        ..Default::default()
    };
    assert!(!config.is_gpu_enabled());
}

#[test]
fn test_gpu_memory_fraction_boundary_one() {
    // memory_fraction = 1.0 should be valid (inclusive upper bound)
    let config = GpuConfig {
        memory_fraction: 1.0,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_gpu_disabled_empty_devices_validates() {
    // Disabled GPU with empty device_ids should be valid
    let config = GpuConfig {
        enabled: false,
        device_ids: vec![],
        memory_fraction: 0.5,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_gpu_serde_roundtrip() {
    let original = GpuConfig {
        enabled: true,
        device_ids: vec![0, 1],
        memory_fraction: 0.8,
        use_cuda_graphs: true,
        mixed_precision: true,
        green_contexts: true,
        gds_enabled: true,
    };
    let json = serde_json::to_string(&original).unwrap();
    let restored: GpuConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(original.enabled, restored.enabled);
    assert_eq!(original.device_ids, restored.device_ids);
    assert!((original.memory_fraction - restored.memory_fraction).abs() < f32::EPSILON);
    assert_eq!(original.use_cuda_graphs, restored.use_cuda_graphs);
    assert_eq!(original.mixed_precision, restored.mixed_precision);
    assert_eq!(original.green_contexts, restored.green_contexts);
    assert_eq!(original.gds_enabled, restored.gds_enabled);
}

#[test]
fn test_gpu_toml_with_all_new_fields() {
    let toml_str = r#"
enabled = true
device_ids = [0, 1]
memory_fraction = 0.75
use_cuda_graphs = true
mixed_precision = true
green_contexts = true
gds_enabled = true
"#;
    let config: GpuConfig = toml::from_str(toml_str).unwrap();
    assert!(config.enabled);
    assert_eq!(config.device_ids, vec![0, 1]);
    assert!((config.memory_fraction - 0.75).abs() < f32::EPSILON);
    assert!(config.use_cuda_graphs);
    assert!(config.mixed_precision);
    assert!(config.green_contexts);
    assert!(config.gds_enabled);
}
