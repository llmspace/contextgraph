//! Learning Layer tests - REAL implementations, NO MOCKS.

use std::time::Duration;

use crate::layers::learning::{LearningLayer, DEFAULT_LEARNING_RATE};
use crate::traits::NervousLayer;
use crate::types::{LayerId, LayerInput, LayerResult};

#[tokio::test]
async fn test_learning_layer_process() {
    let layer = LearningLayer::new();
    let input = LayerInput::new("test-123".to_string(), "test content".to_string());

    let result = layer.process(input).await.unwrap();

    assert_eq!(result.layer, LayerId::Learning);
    assert!(result.result.success);
    assert!(result.result.data.get("weight_delta").is_some());
    assert!(result.result.data.get("surprise").is_some());
    assert!(result.result.data.get("coherence_w").is_some());

    println!("[VERIFIED] LearningLayer.process() returns valid output");
}

#[tokio::test]
async fn test_learning_layer_with_context() {
    let layer = LearningLayer::new();

    // Create input with L1 and L3 context
    let mut input = LayerInput::new("test-456".to_string(), "test content".to_string());
    input.context.layer_results.push(LayerResult::success(
        LayerId::Sensing,
        serde_json::json!({
            "delta_s": 0.7,
            "scrubbed": false,
        }),
    ));
    input.context.layer_results.push(LayerResult::success(
        LayerId::Memory,
        serde_json::json!({
            "retrieval_count": 2,
        }),
    ));
    input.context.pulse.coherence = 0.8;

    let result = layer.process(input).await.unwrap();

    assert!(result.result.success);

    let delta = result.result.data["weight_delta"].as_f64().unwrap();
    assert!(
        delta > 0.0,
        "Should have positive delta with high surprise/coherence"
    );

    println!("[VERIFIED] LearningLayer uses L1/L3 context: delta = {}", delta);
}

#[tokio::test]
async fn test_learning_layer_properties() {
    let layer = LearningLayer::new();

    assert_eq!(layer.layer_id(), LayerId::Learning);
    assert_eq!(layer.latency_budget(), Duration::from_millis(10));
    assert_eq!(layer.layer_name(), "Learning Layer");
    assert!((layer.learning_rate() - DEFAULT_LEARNING_RATE).abs() < 1e-9);

    println!("[VERIFIED] LearningLayer properties correct");
}

#[tokio::test]
async fn test_learning_layer_health_check() {
    let layer = LearningLayer::new();
    let healthy = layer.health_check().await.unwrap();

    assert!(healthy, "LearningLayer should be healthy");
    println!("[VERIFIED] health_check passes");
}

#[tokio::test]
async fn test_learning_layer_custom_config() {
    let layer = LearningLayer::new()
        .with_learning_rate(0.001)
        .with_consolidation_threshold(0.05);

    assert!((layer.learning_rate() - 0.001).abs() < 1e-9);
    assert!((layer.consolidation_threshold() - 0.05).abs() < 1e-9);

    println!("[VERIFIED] Custom configuration works");
}

#[tokio::test]
async fn test_consolidation_trigger() {
    // Use high learning rate to trigger consolidation
    let layer = LearningLayer::new()
        .with_learning_rate(1.0)
        .with_consolidation_threshold(0.01);

    let mut input =
        LayerInput::new("test-789".to_string(), "trigger consolidation".to_string());
    input.context.pulse.entropy = 0.9; // High surprise
    input.context.pulse.coherence = 0.9; // High coherence

    let result = layer.process(input).await.unwrap();

    let should_consolidate = result.result.data["should_consolidate"].as_bool().unwrap();
    assert!(
        should_consolidate,
        "Should trigger consolidation with high delta"
    );
    assert!(layer.consolidation_trigger_count() > 0);

    println!("[VERIFIED] Consolidation triggers correctly");
}

#[tokio::test]
async fn test_pulse_update_on_positive_learning() {
    let layer = LearningLayer::new().with_learning_rate(0.1); // Higher rate for visible effect

    let mut input = LayerInput::new("pulse-test".to_string(), "Test pulse update".to_string());
    input.context.pulse.coherence = 0.5;
    input.context.pulse.entropy = 0.8; // High surprise

    let initial_coherence = input.context.pulse.coherence;
    let result = layer.process(input).await.unwrap();

    // Positive learning should slightly increase coherence
    let weight_delta = result.result.data["weight_delta"].as_f64().unwrap() as f32;
    if weight_delta > 0.0 {
        // Coherence should increase (or stay same if at max)
        assert!(
            result.pulse.coherence >= initial_coherence || result.pulse.coherence >= 0.99,
            "Coherence should increase with positive learning"
        );
    }

    // coherence_delta should reflect the weight delta
    assert!(
        (result.pulse.coherence_delta - weight_delta).abs() < 1e-6,
        "coherence_delta should equal weight_delta"
    );

    println!("[VERIFIED] Pulse updated correctly on positive learning");
}
