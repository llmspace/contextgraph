//! Tests for ConcatenatedEmbedding.

use super::ConcatenatedEmbedding;
use crate::error::EmbeddingError;
use crate::types::dimensions;
use crate::types::{ModelEmbedding, ModelId};

// ========== Construction Tests ==========

#[test]
fn test_new_creates_empty_struct() {
    let ce = ConcatenatedEmbedding::new();

    assert!(ce.embeddings.iter().all(|e| e.is_none()));
    assert!(ce.concatenated.is_empty());
    assert_eq!(ce.total_latency_us, 0);
    assert_eq!(ce.content_hash, 0);
    assert!(!ce.is_complete());
    assert_eq!(ce.filled_count(), 0);
}

#[test]
fn test_default_equals_new() {
    let ce1 = ConcatenatedEmbedding::new();
    let ce2 = ConcatenatedEmbedding::default();

    assert_eq!(ce1.total_latency_us, ce2.total_latency_us);
    assert_eq!(ce1.content_hash, ce2.content_hash);
    assert_eq!(ce1.filled_count(), ce2.filled_count());
}

// ========== Set Tests ==========

#[test]
fn test_set_places_at_correct_index() {
    let mut ce = ConcatenatedEmbedding::new();
    let mut emb = ModelEmbedding::new(ModelId::Semantic, vec![0.1; 1024], 1000);
    emb.set_projected(true);
    ce.set(emb);

    assert!(ce.embeddings[0].is_some()); // Semantic = 0
    assert_eq!(ce.filled_count(), 1);
    assert_eq!(ce.total_latency_us, 1000);
}

#[test]
fn test_set_all_models() {
    let mut ce = ConcatenatedEmbedding::new();

    for model_id in ModelId::all() {
        let dim = model_id.projected_dimension();
        let mut emb = ModelEmbedding::new(*model_id, vec![0.1; dim], 100);
        emb.set_projected(true);
        ce.set(emb);
    }

    assert!(ce.is_complete());
    assert_eq!(ce.filled_count(), 12);
    assert_eq!(ce.total_latency_us, 1200); // 12 * 100
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_set_wrong_dimension_panics() {
    let mut ce = ConcatenatedEmbedding::new();
    // Semantic requires 1024, but we provide 512
    let mut emb = ModelEmbedding::new(ModelId::Semantic, vec![0.1; 512], 1000);
    emb.set_projected(true);
    ce.set(emb); // Should panic
}

// ========== Get Tests ==========

#[test]
fn test_get_returns_correct_embedding() {
    let mut ce = ConcatenatedEmbedding::new();
    let mut emb = ModelEmbedding::new(ModelId::Causal, vec![0.1; 768], 500);
    emb.set_projected(true);
    ce.set(emb);

    let got = ce.get(ModelId::Causal);
    assert!(got.is_some());
    assert_eq!(got.unwrap().model_id, ModelId::Causal);
}

#[test]
fn test_get_returns_none_for_missing() {
    let ce = ConcatenatedEmbedding::new();
    assert!(ce.get(ModelId::Semantic).is_none());
}

// ========== Completion Tests ==========

#[test]
fn test_is_complete_only_when_all_12() {
    let mut ce = ConcatenatedEmbedding::new();

    // Fill 11 models
    for model_id in ModelId::all().iter().take(11) {
        let dim = model_id.projected_dimension();
        let mut emb = ModelEmbedding::new(*model_id, vec![0.1; dim], 100);
        emb.set_projected(true);
        ce.set(emb);
    }
    assert!(!ce.is_complete());
    assert_eq!(ce.filled_count(), 11);

    // Fill last model
    let mut emb = ModelEmbedding::new(ModelId::LateInteraction, vec![0.1; 128], 100);
    emb.set_projected(true);
    ce.set(emb);
    assert!(ce.is_complete());
    assert_eq!(ce.filled_count(), 12);
}

#[test]
fn test_missing_models_returns_correct_list() {
    let mut ce = ConcatenatedEmbedding::new();
    let missing = ce.missing_models();
    assert_eq!(missing.len(), 12);

    // Set one model
    let mut emb = ModelEmbedding::new(ModelId::Semantic, vec![0.1; 1024], 100);
    emb.set_projected(true);
    ce.set(emb);

    let missing = ce.missing_models();
    assert_eq!(missing.len(), 11);
    assert!(!missing.contains(&ModelId::Semantic));
}

// ========== Concatenation Tests ==========

#[test]
fn test_concatenate_produces_8320_vector() {
    let mut ce = create_complete_embedding();
    ce.concatenate();

    assert_eq!(ce.concatenated.len(), dimensions::TOTAL_CONCATENATED);
    assert_eq!(ce.concatenated.len(), 8320);
}

#[test]
fn test_concatenate_order_matches_model_order() {
    let mut ce = ConcatenatedEmbedding::new();

    // Set each model with unique value equal to its index
    for (i, model_id) in ModelId::all().iter().enumerate() {
        let dim = model_id.projected_dimension();
        let mut emb = ModelEmbedding::new(*model_id, vec![i as f32; dim], 100);
        emb.set_projected(true);
        ce.set(emb);
    }

    ce.concatenate();

    // Verify order: first 1024 elements should be 0.0 (Semantic)
    assert_eq!(ce.concatenated[0], 0.0);
    // Next 512 should be 1.0 (TemporalRecent)
    assert_eq!(ce.concatenated[1024], 1.0);
    // Last 128 should be 11.0 (LateInteraction)
    assert_eq!(ce.concatenated[8320 - 1], 11.0);
}

#[test]
fn test_content_hash_deterministic() {
    let mut ce1 = create_complete_embedding();
    let mut ce2 = create_complete_embedding();

    ce1.concatenate();
    ce2.concatenate();

    assert_eq!(ce1.content_hash, ce2.content_hash);
    assert_ne!(ce1.content_hash, 0);
}

#[test]
fn test_content_hash_differs_for_different_data() {
    let mut ce1 = create_complete_embedding();
    ce1.concatenate();
    let hash1 = ce1.content_hash;

    // Create another with different values
    let mut ce2 = ConcatenatedEmbedding::new();
    for (i, model_id) in ModelId::all().iter().enumerate() {
        let dim = model_id.projected_dimension();
        let mut emb = ModelEmbedding::new(*model_id, vec![(i as f32) * 0.1; dim], 100);
        emb.set_projected(true);
        ce2.set(emb);
    }
    ce2.concatenate();
    let hash2 = ce2.content_hash;

    assert_ne!(hash1, hash2);
}

#[test]
fn test_total_latency_sums_all() {
    let mut ce = ConcatenatedEmbedding::new();

    for model_id in ModelId::all() {
        let dim = model_id.projected_dimension();
        let mut emb = ModelEmbedding::new(*model_id, vec![0.1; dim], 100);
        emb.set_projected(true);
        ce.set(emb);
    }

    assert_eq!(ce.total_latency_us, 1200); // 12 * 100
}

#[test]
#[should_panic(expected = "Cannot concatenate")]
fn test_concatenate_panics_when_incomplete() {
    let mut ce = ConcatenatedEmbedding::new();
    ce.concatenate(); // Should panic
}

#[test]
fn test_total_dimension() {
    let mut ce = create_complete_embedding();
    ce.concatenate();
    assert_eq!(ce.total_dimension(), 8320);
}

#[test]
fn test_total_dimension_partial() {
    let mut ce = ConcatenatedEmbedding::new();
    let mut emb = ModelEmbedding::new(ModelId::Semantic, vec![0.1; 1024], 100);
    emb.set_projected(true);
    ce.set(emb);

    assert_eq!(ce.total_dimension(), 1024);
}

// ========== Validation Tests ==========

#[test]
fn test_validate_succeeds_for_valid_embeddings() {
    let ce = create_complete_embedding();
    assert!(ce.validate().is_ok());
}

#[test]
fn test_validate_detects_nan() {
    let mut ce = ConcatenatedEmbedding::new();
    let mut vector = vec![0.1; 1024];
    vector[500] = f32::NAN;
    let mut emb = ModelEmbedding::new(ModelId::Semantic, vector, 100);
    emb.set_projected(true);
    ce.set(emb);

    let result = ce.validate();
    assert!(result.is_err());
    match result.unwrap_err() {
        EmbeddingError::InvalidValue { index, value } => {
            assert_eq!(index, 500);
            assert!(value.is_nan());
        }
        _ => panic!("Expected InvalidValue error"),
    }
}

// ========== Get Slice Tests ==========

#[test]
fn test_get_slice_returns_correct_segment() {
    let mut ce = create_complete_embedding();
    ce.concatenate();

    let semantic_slice = ce.get_slice(ModelId::Semantic).unwrap();
    assert_eq!(semantic_slice.len(), 1024);
    assert!(semantic_slice.iter().all(|&v| (v - 0.5).abs() < 1e-6));

    let late_interaction_slice = ce.get_slice(ModelId::LateInteraction).unwrap();
    assert_eq!(late_interaction_slice.len(), 128);
}

#[test]
fn test_get_slice_none_before_concatenation() {
    let ce = ConcatenatedEmbedding::new();
    assert!(ce.get_slice(ModelId::Semantic).is_none());
}

// ========== Edge Case Tests ==========

#[test]
fn edge_case_empty_struct() {
    let ce = ConcatenatedEmbedding::new();
    println!(
        "BEFORE: filled={}, complete={}",
        ce.filled_count(),
        ce.is_complete()
    );

    let missing = ce.missing_models();

    println!("AFTER: missing_count={}", missing.len());
    assert_eq!(missing.len(), 12);
    println!("Edge Case 1 PASSED: Empty struct returns all 12 models as missing");
}

#[test]
fn edge_case_overwrite() {
    let mut ce = ConcatenatedEmbedding::new();
    let mut emb1 = ModelEmbedding::new(ModelId::Semantic, vec![1.0; 1024], 100);
    emb1.set_projected(true);
    ce.set(emb1);

    println!(
        "BEFORE: latency={}, first_value={}",
        ce.total_latency_us,
        ce.embeddings[0].as_ref().unwrap().vector[0]
    );

    let mut emb2 = ModelEmbedding::new(ModelId::Semantic, vec![2.0; 1024], 200);
    emb2.set_projected(true);
    ce.set(emb2);

    println!(
        "AFTER: latency={}, first_value={}",
        ce.total_latency_us,
        ce.embeddings[0].as_ref().unwrap().vector[0]
    );

    // Latency should be replaced (old subtracted, new added)
    assert_eq!(ce.total_latency_us, 200);
    assert_eq!(ce.embeddings[0].as_ref().unwrap().vector[0], 2.0);
    println!("Edge Case 2 PASSED: Overwrite replaces embedding and updates latency correctly");
}

#[test]
fn edge_case_max_latency() {
    let mut ce = ConcatenatedEmbedding::new();
    let mut emb = ModelEmbedding::new(ModelId::Semantic, vec![0.1; 1024], u64::MAX);
    emb.set_projected(true);

    println!("BEFORE: total_latency={}", ce.total_latency_us);
    ce.set(emb);
    println!("AFTER: total_latency={}", ce.total_latency_us);

    assert_eq!(ce.total_latency_us, u64::MAX);
    println!("Edge Case 3 PASSED: u64::MAX latency handled correctly");
}

// ========== Source of Truth Verification ==========

#[test]
fn verify_source_of_truth() {
    // The concatenated vector IS the source of truth
    let mut ce = create_complete_embedding();
    ce.concatenate();

    // 1. Verify vector exists in memory
    assert!(!ce.concatenated.is_empty());
    println!(
        "SOURCE OF TRUTH: concatenated.len() = {}",
        ce.concatenated.len()
    );

    // 2. Verify dimensions match specification
    assert_eq!(ce.concatenated.len(), dimensions::TOTAL_CONCATENATED);
    println!(
        "DIMENSION CHECK: {} == {} (expected)",
        ce.concatenated.len(),
        dimensions::TOTAL_CONCATENATED
    );

    // 3. Verify hash is non-zero
    assert_ne!(ce.content_hash, 0);
    println!("HASH CHECK: content_hash = {} (non-zero)", ce.content_hash);

    // 4. Read back individual slices
    for (i, model_id) in ModelId::all().iter().enumerate() {
        let offset = dimensions::offset_by_index(i);
        let dim = dimensions::projected_dimension_by_index(i);
        let slice = &ce.concatenated[offset..offset + dim];

        println!(
            "MODEL {:?}: offset={}, dim={}, slice_len={}",
            model_id,
            offset,
            dim,
            slice.len()
        );
        assert_eq!(slice.len(), dim);
    }

    println!("VERIFICATION COMPLETE: All checks passed");
}

// ========== Helper Functions ==========

fn create_complete_embedding() -> ConcatenatedEmbedding {
    let mut ce = ConcatenatedEmbedding::new();
    for model_id in ModelId::all() {
        let dim = model_id.projected_dimension();
        let mut emb = ModelEmbedding::new(*model_id, vec![0.5; dim], 100);
        emb.set_projected(true);
        ce.set(emb);
    }
    ce
}
