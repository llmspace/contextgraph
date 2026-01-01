//! Latency budget enforcement tests (TC-GHOST-007).
//!
//! This module contains tests that verify layer latency budget compliance.

use std::time::{Duration, Instant};

use crate::traits::NervousLayer;
use crate::types::LayerInput;

use super::{
    StubCoherenceLayer, StubLearningLayer, StubMemoryLayer, StubReflexLayer, StubSensingLayer,
};

/// Helper to create test input.
fn test_input(content: &str) -> LayerInput {
    LayerInput::new("test-request-123".to_string(), content.to_string())
}

#[tokio::test]
async fn test_layer_execution_within_latency_budget_sensing() {
    let layer = StubSensingLayer::new();
    let input = test_input("test content for sensing layer latency budget");

    let budget = layer.latency_budget();
    let start = Instant::now();
    let result = layer.process(input).await;
    let elapsed = start.elapsed();

    assert!(result.is_ok(), "Sensing layer must process successfully");
    assert!(
        elapsed <= budget,
        "Sensing layer took {:?} but budget is {:?}",
        elapsed,
        budget
    );
}

#[tokio::test]
async fn test_layer_execution_within_latency_budget_reflex() {
    let layer = StubReflexLayer::new();
    let input = test_input("test content for reflex layer latency budget");

    let budget = layer.latency_budget();
    let start = Instant::now();
    let result = layer.process(input).await;
    let elapsed = start.elapsed();

    assert!(result.is_ok(), "Reflex layer must process successfully");
    assert!(
        elapsed <= budget,
        "Reflex layer took {:?} but budget is {:?}",
        elapsed,
        budget
    );
}

#[tokio::test]
async fn test_layer_execution_within_latency_budget_memory() {
    let layer = StubMemoryLayer::new();
    let input = test_input("test content for memory layer latency budget");

    let budget = layer.latency_budget();
    let start = Instant::now();
    let result = layer.process(input).await;
    let elapsed = start.elapsed();

    assert!(result.is_ok(), "Memory layer must process successfully");
    assert!(
        elapsed <= budget,
        "Memory layer took {:?} but budget is {:?}",
        elapsed,
        budget
    );
}

#[tokio::test]
async fn test_layer_execution_within_latency_budget_learning() {
    let layer = StubLearningLayer::new();
    let input = test_input("test content for learning layer latency budget");

    let budget = layer.latency_budget();
    let start = Instant::now();
    let result = layer.process(input).await;
    let elapsed = start.elapsed();

    assert!(result.is_ok(), "Learning layer must process successfully");
    assert!(
        elapsed <= budget,
        "Learning layer took {:?} but budget is {:?}",
        elapsed,
        budget
    );
}

#[tokio::test]
async fn test_layer_execution_within_latency_budget_coherence() {
    let layer = StubCoherenceLayer::new();
    let input = test_input("test content for coherence layer latency budget");

    let budget = layer.latency_budget();
    let start = Instant::now();
    let result = layer.process(input).await;
    let elapsed = start.elapsed();

    assert!(result.is_ok(), "Coherence layer must process successfully");
    assert!(
        elapsed <= budget,
        "Coherence layer took {:?} but budget is {:?}",
        elapsed,
        budget
    );
}

#[tokio::test]
async fn test_all_layers_within_latency_budget() {
    let layers: Vec<(Box<dyn NervousLayer>, &str)> = vec![
        (Box::new(StubSensingLayer::new()), "Sensing"),
        (Box::new(StubReflexLayer::new()), "Reflex"),
        (Box::new(StubMemoryLayer::new()), "Memory"),
        (Box::new(StubLearningLayer::new()), "Learning"),
        (Box::new(StubCoherenceLayer::new()), "Coherence"),
    ];

    for (layer, name) in layers {
        let input = LayerInput::new(
            "test-request".to_string(),
            format!("test content for {}", name),
        );

        let budget = layer.latency_budget();
        let start = Instant::now();
        let result = layer.process(input).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "{} layer must process successfully", name);
        assert!(
            elapsed <= budget,
            "{} layer took {:?} but budget is {:?}",
            name,
            elapsed,
            budget
        );
    }
}

#[tokio::test]
async fn test_layer_reported_duration_within_budget() {
    let layers: Vec<Box<dyn NervousLayer>> = vec![
        Box::new(StubSensingLayer::new()),
        Box::new(StubReflexLayer::new()),
        Box::new(StubMemoryLayer::new()),
        Box::new(StubLearningLayer::new()),
        Box::new(StubCoherenceLayer::new()),
    ];

    for layer in layers {
        let input = LayerInput::new("test-request".to_string(), "test content".to_string());
        let output = layer.process(input).await.unwrap();

        let budget_us = layer.latency_budget().as_micros() as u64;
        assert!(
            output.duration_us < budget_us,
            "{} reported duration {}us exceeds budget {}us",
            layer.layer_name(),
            output.duration_us,
            budget_us
        );
    }
}

#[tokio::test]
async fn test_full_pipeline_within_total_budget() {
    let sensing = StubSensingLayer::new();
    let reflex = StubReflexLayer::new();
    let memory = StubMemoryLayer::new();
    let learning = StubLearningLayer::new();
    let coherence = StubCoherenceLayer::new();

    // Total budget: 5ms + 100us + 1ms + 10ms + 10ms = 26.1ms
    let total_budget = Duration::from_micros(26100);

    let input = test_input("full pipeline test content");
    let start = Instant::now();

    let _ = sensing.process(input.clone()).await.unwrap();
    let _ = reflex.process(input.clone()).await.unwrap();
    let _ = memory.process(input.clone()).await.unwrap();
    let _ = learning.process(input.clone()).await.unwrap();
    let _ = coherence.process(input).await.unwrap();

    let elapsed = start.elapsed();

    assert!(
        elapsed <= total_budget,
        "Full pipeline took {:?} but total budget is {:?}",
        elapsed,
        total_budget
    );
}
