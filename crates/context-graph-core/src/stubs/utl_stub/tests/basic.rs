//! Basic functionality tests for UTL processor.

use crate::stubs::utl_stub::StubUtlProcessor;
use crate::traits::UtlProcessor;
use crate::types::{EmotionalState, UtlContext};

#[tokio::test]
async fn test_compute_learning_score() {
    let processor = StubUtlProcessor::new();
    let context = UtlContext::default();

    let score = processor
        .compute_learning_score("test input", &context)
        .await
        .unwrap();

    assert!((0.0..=1.0).contains(&score));
}

#[tokio::test]
async fn test_deterministic_output() {
    let processor = StubUtlProcessor::new();
    let context = UtlContext::default();

    let score1 = processor
        .compute_surprise("same input", &context)
        .await
        .unwrap();
    let score2 = processor
        .compute_surprise("same input", &context)
        .await
        .unwrap();

    assert_eq!(score1, score2);
}

#[tokio::test]
async fn test_emotional_weight_modifier() {
    let processor = StubUtlProcessor::new();

    let neutral_ctx = UtlContext {
        emotional_state: EmotionalState::Neutral,
        ..Default::default()
    };
    let curious_ctx = UtlContext {
        emotional_state: EmotionalState::Curious,
        ..Default::default()
    };

    let neutral_weight = processor
        .compute_emotional_weight("test", &neutral_ctx)
        .await
        .unwrap();
    let curious_weight = processor
        .compute_emotional_weight("test", &curious_ctx)
        .await
        .unwrap();

    // Curious should have higher weight (1.2 vs 1.0)
    assert!(curious_weight > neutral_weight);
    assert_eq!(neutral_weight, 1.0);
    assert_eq!(curious_weight, 1.2);
}

#[tokio::test]
async fn test_fallback_when_no_embeddings() {
    // Test graceful fallback to context values when embeddings unavailable
    let processor = StubUtlProcessor::new();

    let context = UtlContext {
        prior_entropy: 0.8,
        current_coherence: 0.6,
        // No embeddings provided
        ..Default::default()
    };

    let surprise = processor.compute_surprise("test", &context).await.unwrap();
    let coherence = processor
        .compute_coherence_change("test", &context)
        .await
        .unwrap();
    let alignment = processor.compute_alignment("test", &context).await.unwrap();

    // Without embeddings, falls back to context values
    assert_eq!(surprise, 0.8, "Should use prior_entropy as fallback");
    assert_eq!(coherence, 0.6, "Should use current_coherence as fallback");
    assert_eq!(alignment, 1.0, "Should default to full alignment");
}

#[test]
fn test_get_status_shows_real_computation() {
    let processor = StubUtlProcessor::new();
    let status = processor.get_status();

    // Verify status indicates real computation mode
    assert_eq!(status["computation_mode"], "real");
    assert!(status["formula"].as_str().unwrap().contains("sigmoid"));
    assert!(status["thresholds"]["edge_similarity"].as_f64().is_some());
    assert!(status["thresholds"]["knn_k"].as_u64().is_some());
}
