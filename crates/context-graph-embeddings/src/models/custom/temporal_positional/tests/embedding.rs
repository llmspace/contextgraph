//! Embedding and trait implementation tests for TemporalPositionalModel.

use chrono::Datelike;

use crate::error::EmbeddingError;
use crate::traits::EmbeddingModel;
use crate::types::{InputType, ModelId, ModelInput};

use super::super::{TemporalPositionalModel, TEMPORAL_POSITIONAL_DIMENSION};

// =========================================================================
// TRAIT IMPLEMENTATION TESTS
// =========================================================================

#[test]
fn test_model_id_is_temporal_positional() {
    let model = TemporalPositionalModel::new();

    assert_eq!(model.model_id(), ModelId::TemporalPositional);
}

#[test]
fn test_supported_input_types() {
    let model = TemporalPositionalModel::new();
    let types = model.supported_input_types();

    assert_eq!(types.len(), 1, "Should support exactly 1 input type");
    assert_eq!(types[0], InputType::Text, "Should support Text input");
}

#[test]
fn test_dimension_is_512() {
    let model = TemporalPositionalModel::new();

    // dimension() uses default impl that delegates to model_id().dimension()
    assert_eq!(model.dimension(), TEMPORAL_POSITIONAL_DIMENSION);
    assert_eq!(model.dimension(), 512);
}

#[test]
fn test_is_pretrained_returns_false() {
    let model = TemporalPositionalModel::new();

    // is_pretrained() uses default impl that delegates to model_id().is_pretrained()
    // ModelId::TemporalPositional.is_pretrained() should return false
    assert!(!model.is_pretrained(), "Custom models are not pretrained");
}

#[test]
fn test_latency_budget_is_2ms() {
    let model = TemporalPositionalModel::new();

    assert_eq!(model.latency_budget_ms(), 2, "Latency budget should be 2ms");
}

// =========================================================================
// EMBEDDING TESTS
// =========================================================================

#[tokio::test]
async fn test_embed_returns_512d_vector() {
    let model = TemporalPositionalModel::new();
    let input = ModelInput::text("test content").expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    println!("Vector length: {}", embedding.vector.len());
    assert_eq!(embedding.vector.len(), 512, "Must return exactly 512D");
}

#[tokio::test]
async fn test_embed_model_id_correct() {
    let model = TemporalPositionalModel::new();
    let input = ModelInput::text("test").expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    assert_eq!(embedding.model_id, ModelId::TemporalPositional);
}

#[tokio::test]
async fn test_embed_l2_normalized() {
    let model = TemporalPositionalModel::new();
    let input = ModelInput::text("test normalization").expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    let norm: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

    println!("L2 norm: {}", norm);
    println!("Deviation from 1.0: {}", (norm - 1.0).abs());

    assert!(
        (norm - 1.0).abs() < 0.001,
        "Vector MUST be L2 normalized, got norm = {}",
        norm
    );
}

#[tokio::test]
async fn test_embed_no_nan_values() {
    let model = TemporalPositionalModel::new();
    let input = ModelInput::text("test").expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    let has_nan = embedding.vector.iter().any(|x| x.is_nan());

    assert!(!has_nan, "Output must not contain NaN values");
}

#[tokio::test]
async fn test_embed_no_inf_values() {
    let model = TemporalPositionalModel::new();
    let input = ModelInput::text("test").expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    let has_inf = embedding.vector.iter().any(|x| x.is_infinite());

    assert!(!has_inf, "Output must not contain Inf values");
}

#[tokio::test]
async fn test_embed_records_latency() {
    let model = TemporalPositionalModel::new();
    let input = ModelInput::text("test latency").expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    println!("Latency: {} microseconds", embedding.latency_us);

    // Latency should be recorded (not necessarily non-zero for very fast ops)
    // Just verify the value is reasonable (< 2s as an upper sanity check)
    assert!(
        embedding.latency_us < 2_000_000,
        "Latency should be under 2 seconds"
    );
}

#[tokio::test]
async fn test_embed_latency_under_2ms() {
    let model = TemporalPositionalModel::new();
    let input = ModelInput::text("test performance").expect("Failed to create input");

    let start = std::time::Instant::now();
    let _embedding = model.embed(&input).await.expect("Embed should succeed");
    let elapsed = start.elapsed();

    println!("Elapsed: {:?}", elapsed);

    assert!(
        elapsed.as_millis() < 2,
        "Latency must be under 2ms, got {:?}",
        elapsed
    );
}

// =========================================================================
// TIMESTAMP PARSING TESTS
// =========================================================================

#[test]
fn test_parse_timestamp_iso8601() {
    let instruction = "timestamp:2024-01-15T10:30:00Z";
    let result = TemporalPositionalModel::parse_timestamp(instruction);

    assert!(result.is_some(), "Should parse ISO 8601");
    let dt = result.unwrap();
    assert_eq!(dt.year(), 2024);
    assert_eq!(dt.month(), 1);
    assert_eq!(dt.day(), 15);
}

#[test]
fn test_parse_timestamp_unix_epoch() {
    let instruction = "epoch:1705315800";
    let result = TemporalPositionalModel::parse_timestamp(instruction);

    assert!(result.is_some(), "Should parse Unix epoch");
}

#[test]
fn test_parse_timestamp_invalid() {
    let invalid_inputs = vec![
        "not a timestamp",
        "timestamp:invalid",
        "epoch:notanumber",
        "random text",
        "",
    ];

    for input in invalid_inputs {
        let result = TemporalPositionalModel::parse_timestamp(input);
        assert!(result.is_none(), "Should return None for '{}'", input);
    }
}

#[tokio::test]
async fn test_extract_timestamp_with_iso8601() {
    let model = TemporalPositionalModel::new();
    let input = ModelInput::text_with_instruction("content", "timestamp:2024-01-15T10:30:00Z")
        .expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    // Just verify it embeds successfully
    assert_eq!(embedding.vector.len(), 512);
}

#[tokio::test]
async fn test_extract_timestamp_fallback_to_now() {
    let model = TemporalPositionalModel::new();
    let input = ModelInput::text("no timestamp").expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    // Should use current time as fallback
    assert_eq!(embedding.vector.len(), 512);
}

// =========================================================================
// UNSUPPORTED INPUT TYPE TESTS
// =========================================================================

#[tokio::test]
async fn test_unsupported_code_input() {
    let model = TemporalPositionalModel::new();
    let input = ModelInput::code("fn main() {}", "rust").expect("Failed to create input");

    let result = model.embed(&input).await;

    assert!(result.is_err(), "Code input should be rejected");
    match result {
        Err(EmbeddingError::UnsupportedModality {
            model_id,
            input_type,
        }) => {
            assert_eq!(model_id, ModelId::TemporalPositional);
            assert_eq!(input_type, InputType::Code);
        }
        other => panic!("Expected UnsupportedModality error, got {:?}", other),
    }
}

// =========================================================================
// DIFFERENT BASE TESTS
// =========================================================================

#[tokio::test]
async fn test_different_base_produces_different_embeddings() {
    let model1 = TemporalPositionalModel::new(); // base = 10000
    let model2 = TemporalPositionalModel::with_base(5000.0).expect("model2");

    let input = ModelInput::text_with_instruction("content", "timestamp:2024-01-15T10:30:00Z")
        .expect("input");

    let emb1 = model1.embed(&input).await.expect("emb1");
    let emb2 = model2.embed(&input).await.expect("emb2");

    assert_ne!(
        emb1.vector, emb2.vector,
        "Different bases should produce different embeddings"
    );
}
