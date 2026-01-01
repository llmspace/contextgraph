//! Integration and cross-layer tests for stub layers.
//!
//! This module contains tests that verify:
//! - Cross-layer interactions and pipeline processing
//! - Layer output validation
//! - Health check verification

use crate::traits::NervousLayer;
use crate::types::{LayerId, LayerInput};

use super::{
    StubCoherenceLayer, StubLearningLayer, StubMemoryLayer, StubReflexLayer, StubSensingLayer,
};

/// Helper to create test input for integration testing.
fn test_input(content: &str) -> LayerInput {
    LayerInput::new("test-request-123".to_string(), content.to_string())
}

#[tokio::test]
async fn test_all_layers_healthy() {
    let sensing = StubSensingLayer::new();
    let reflex = StubReflexLayer::new();
    let memory = StubMemoryLayer::new();
    let learning = StubLearningLayer::new();
    let coherence = StubCoherenceLayer::new();

    assert!(sensing.health_check().await.unwrap());
    assert!(reflex.health_check().await.unwrap());
    assert!(memory.health_check().await.unwrap());
    assert!(learning.health_check().await.unwrap());
    assert!(coherence.health_check().await.unwrap());
}

#[tokio::test]
async fn test_pipeline_processing() {
    // Test that layers can be chained (context accumulates)
    let sensing = StubSensingLayer::new();
    let reflex = StubReflexLayer::new();
    let memory = StubMemoryLayer::new();
    let learning = StubLearningLayer::new();
    let coherence = StubCoherenceLayer::new();

    let mut input = test_input("pipeline test");

    let out1 = sensing.process(input.clone()).await.unwrap();
    input
        .context
        .add_latency(std::time::Duration::from_micros(out1.duration_us));
    input.context.layer_results.push(out1.result.clone());

    let out2 = reflex.process(input.clone()).await.unwrap();
    input
        .context
        .add_latency(std::time::Duration::from_micros(out2.duration_us));
    input.context.layer_results.push(out2.result.clone());

    let out3 = memory.process(input.clone()).await.unwrap();
    input
        .context
        .add_latency(std::time::Duration::from_micros(out3.duration_us));
    input.context.layer_results.push(out3.result.clone());

    let out4 = learning.process(input.clone()).await.unwrap();
    input
        .context
        .add_latency(std::time::Duration::from_micros(out4.duration_us));
    input.context.layer_results.push(out4.result.clone());

    let out5 = coherence.process(input.clone()).await.unwrap();
    input
        .context
        .add_latency(std::time::Duration::from_micros(out5.duration_us));
    input.context.layer_results.push(out5.result.clone());

    // All layers should have run successfully
    assert_eq!(input.context.layer_results.len(), 5);
    for result in &input.context.layer_results {
        assert!(result.success);
    }
}

#[tokio::test]
async fn test_layer_output_coherence_ranges() {
    let layers: Vec<Box<dyn NervousLayer>> = vec![
        Box::new(StubSensingLayer::new()),
        Box::new(StubReflexLayer::new()),
        Box::new(StubMemoryLayer::new()),
        Box::new(StubLearningLayer::new()),
        Box::new(StubCoherenceLayer::new()),
    ];

    for layer in layers {
        let input = LayerInput::new(
            "test-request".to_string(),
            "test content for range validation".to_string(),
        );
        let output = layer.process(input).await.unwrap();

        assert!(
            output.pulse.entropy >= 0.0 && output.pulse.entropy <= 1.0,
            "{} entropy {} must be in [0.0, 1.0]",
            layer.layer_name(),
            output.pulse.entropy
        );
        assert!(
            output.pulse.coherence >= 0.0 && output.pulse.coherence <= 1.0,
            "{} coherence {} must be in [0.0, 1.0]",
            layer.layer_name(),
            output.pulse.coherence
        );
    }
}

#[tokio::test]
async fn test_layer_output_correct_layer_id() {
    let test_cases = vec![
        (
            Box::new(StubSensingLayer::new()) as Box<dyn NervousLayer>,
            LayerId::Sensing,
        ),
        (Box::new(StubReflexLayer::new()), LayerId::Reflex),
        (Box::new(StubMemoryLayer::new()), LayerId::Memory),
        (Box::new(StubLearningLayer::new()), LayerId::Learning),
        (Box::new(StubCoherenceLayer::new()), LayerId::Coherence),
    ];

    for (layer, expected_id) in test_cases {
        let input = LayerInput::new("test-request".to_string(), "test content".to_string());
        let output = layer.process(input).await.unwrap();

        assert_eq!(
            output.layer,
            expected_id,
            "{} must report correct LayerId {:?}, got {:?}",
            layer.layer_name(),
            expected_id,
            output.layer
        );
        assert_eq!(
            output.result.layer,
            expected_id,
            "{} result must report correct LayerId {:?}, got {:?}",
            layer.layer_name(),
            expected_id,
            output.result.layer
        );
    }
}
