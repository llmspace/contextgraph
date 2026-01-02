//! Edge case handling tests for TemporalPositionalModel.

use crate::traits::EmbeddingModel;
use crate::types::ModelInput;

use super::super::TemporalPositionalModel;
use super::cosine_similarity;

// =========================================================================
// EDGE CASE TESTS (MANDATORY per spec)
// =========================================================================

#[tokio::test]
async fn test_edge_case_1_timestamps_one_second_apart() {
    let model = TemporalPositionalModel::new();

    println!("=== EDGE CASE 1: Timestamps 1 Second Apart ===");

    let ts1 = "timestamp:2024-01-15T10:30:00Z";
    let ts2 = "timestamp:2024-01-15T10:30:01Z";

    println!("BEFORE: ts1 = {}", ts1);
    println!("BEFORE: ts2 = {}", ts2);

    let input1 = ModelInput::text_with_instruction("content", ts1).expect("input1");
    let input2 = ModelInput::text_with_instruction("content", ts2).expect("input2");

    let emb1 = model.embed(&input1).await.expect("Embed ts1");
    let emb2 = model.embed(&input2).await.expect("Embed ts2");

    println!("AFTER: emb1[0..5] = {:?}", &emb1.vector[0..5]);
    println!("AFTER: emb2[0..5] = {:?}", &emb2.vector[0..5]);

    let cosine_sim = cosine_similarity(&emb1.vector, &emb2.vector);

    println!("AFTER: cosine similarity = {}", cosine_sim);

    assert_eq!(emb1.vector.len(), 512);
    assert_eq!(emb2.vector.len(), 512);
    // 1 second apart should be similar but NOT identical
    assert_ne!(
        emb1.vector, emb2.vector,
        "Different timestamps must produce different embeddings"
    );
    assert!(
        cosine_sim > 0.9,
        "Close timestamps should be similar but distinct, got {}",
        cosine_sim
    );
    assert!(cosine_sim < 1.0, "But not identical, got {}", cosine_sim);
}

#[tokio::test]
async fn test_edge_case_2_unix_epoch_zero() {
    let model = TemporalPositionalModel::new();

    println!("=== EDGE CASE 2: Unix Epoch Zero ===");

    let epoch_zero = "epoch:0";
    println!("BEFORE: epoch = {}", epoch_zero);

    let input = ModelInput::text_with_instruction("content", epoch_zero).expect("input");

    let embedding = model.embed(&input).await.expect("Embed epoch 0");

    println!("AFTER: vector[0..5] = {:?}", &embedding.vector[0..5]);
    println!(
        "AFTER: all values finite = {}",
        embedding.vector.iter().all(|x| x.is_finite())
    );

    let norm: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("AFTER: L2 norm = {}", norm);

    // For pos=0, sin(0) = 0, cos(0) = 1 for all dimensions
    // After normalization, values should be consistent
    assert_eq!(embedding.vector.len(), 512);
    assert!(
        embedding.vector.iter().all(|x| x.is_finite()),
        "Must handle epoch 0"
    );
    assert!((norm - 1.0).abs() < 0.001, "Must be L2 normalized");
}

#[tokio::test]
async fn test_edge_case_3_no_timestamp_fallback() {
    let model = TemporalPositionalModel::new();

    println!("=== EDGE CASE 3: No Timestamp (Fallback) ===");
    println!("BEFORE: input has no timestamp instruction");

    let input = ModelInput::text("content without timestamp").expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    println!("AFTER: vector[0..5] = {:?}", &embedding.vector[0..5]);
    println!("AFTER: Should use current time as fallback");

    assert_eq!(embedding.vector.len(), 512);
    assert!(embedding.vector.iter().all(|x| x.is_finite()));
}

#[tokio::test]
async fn test_edge_case_very_large_timestamp() {
    let model = TemporalPositionalModel::new();

    println!("=== EDGE CASE: Very Large Timestamp (Year 3000) ===");

    // Year 3000 = approximately 32503680000 seconds
    let input = ModelInput::text_with_instruction("content", "epoch:32503680000")
        .expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    println!("AFTER: vector[0..5] = {:?}", &embedding.vector[0..5]);
    println!(
        "AFTER: all values finite = {}",
        embedding.vector.iter().all(|x| x.is_finite())
    );

    assert_eq!(embedding.vector.len(), 512);
    assert!(
        embedding.vector.iter().all(|x| x.is_finite()),
        "Must handle far future"
    );
}

#[tokio::test]
async fn test_edge_case_negative_timestamp() {
    let model = TemporalPositionalModel::new();

    println!("=== EDGE CASE: Negative Timestamp (Before 1970) ===");

    // -86400 = 1 day before Unix epoch (Dec 31, 1969)
    let input = ModelInput::text_with_instruction("content", "epoch:-86400")
        .expect("Failed to create input");

    let embedding = model.embed(&input).await.expect("Embed should succeed");

    println!("AFTER: vector[0..5] = {:?}", &embedding.vector[0..5]);
    println!(
        "AFTER: all values finite = {}",
        embedding.vector.iter().all(|x| x.is_finite())
    );

    assert_eq!(embedding.vector.len(), 512);
    assert!(
        embedding.vector.iter().all(|x| x.is_finite()),
        "Must handle pre-1970 dates"
    );
}
