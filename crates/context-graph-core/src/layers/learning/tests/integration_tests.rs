//! Integration tests for L4 Learning Layer - full pipeline context.

use crate::layers::learning::{LearningLayer, GRADIENT_CLIP};
use crate::traits::NervousLayer;
use crate::types::{LayerId, LayerInput, LayerResult};

#[tokio::test]
async fn test_learning_layer_full_pipeline_context() {
    let layer = LearningLayer::new();

    // Simulate full L1 -> L2 -> L3 -> L4 pipeline context
    let mut input = LayerInput::new(
        "pipeline-test".to_string(),
        "Full pipeline test".to_string(),
    );

    // L1 Sensing result
    input.context.layer_results.push(LayerResult::success(
        LayerId::Sensing,
        serde_json::json!({
            "delta_s": 0.6,
            "scrubbed_content": "Full pipeline test",
            "pii_found": false,
            "duration_us": 100,
        }),
    ));

    // L2 Reflex result (cache miss)
    input.context.layer_results.push(LayerResult::success(
        LayerId::Reflex,
        serde_json::json!({
            "cache_hit": false,
            "query_norm": 1.0,
        }),
    ));

    // L3 Memory result
    input.context.layer_results.push(LayerResult::success(
        LayerId::Memory,
        serde_json::json!({
            "retrieval_count": 3,
            "memories": [],
            "duration_us": 500,
        }),
    ));

    // Set pulse state
    input.context.pulse.coherence = 0.75;
    input.context.pulse.entropy = 0.55;

    let result = layer.process(input).await.unwrap();

    assert!(result.result.success);

    // Verify all expected fields
    let data = &result.result.data;
    assert!(data.get("weight_delta").is_some());
    assert!(data.get("surprise").is_some());
    assert!(data.get("coherence_w").is_some());
    assert!(data.get("learning_rate").is_some());
    assert!(data.get("should_consolidate").is_some());
    assert!(data.get("duration_us").is_some());
    assert!(data.get("within_budget").is_some());

    // Verify reasonable values
    let delta = data["weight_delta"].as_f64().unwrap() as f32;
    let surprise = data["surprise"].as_f64().unwrap() as f32;
    let coherence = data["coherence_w"].as_f64().unwrap() as f32;

    assert!((0.0..=1.0).contains(&surprise));
    assert!((0.0..=1.0).contains(&coherence));
    assert!(delta.abs() <= GRADIENT_CLIP);

    println!("[VERIFIED] Full pipeline context processed correctly");
    println!("  Surprise: {}", surprise);
    println!("  Coherence: {}", coherence);
    println!("  Delta: {}", delta);
}
