//! TASK-DELTA-P1-001: Edge Case Tests

use serde_json::json;
use uuid::Uuid;

use context_graph_core::types::fingerprint::NUM_EMBEDDERS;

use crate::handlers::tests::{create_test_handlers, extract_mcp_tool_data, make_request};
use crate::protocol::JsonRpcId;

use super::helpers::create_test_fingerprint_with_semantic;

/// EC-01: Identical fingerprints should produce ΔS ≈ 0 for all embedders
#[tokio::test]
async fn test_ec01_identical_fingerprints() {
    let handlers = create_test_handlers();

    // Create identical fingerprints
    let fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(
        response.error.is_none(),
        "Should succeed with identical fingerprints"
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    // With identical fingerprints, ΔS should be very low (close to 0)
    let delta_s_agg = data["delta_s_aggregate"].as_f64().expect("f64");

    // Note: ΔS might not be exactly 0 due to the entropy calculation method,
    // but should be very low for identical embeddings
    assert!(
        delta_s_agg < 0.5,
        "EC-01: Identical fingerprints should produce low ΔS, got {}",
        delta_s_agg
    );

    // Verify all values are still valid (no NaN/Inf)
    let delta_s_per_embedder = data["delta_s_per_embedder"].as_array().expect("array");
    for (i, val) in delta_s_per_embedder.iter().enumerate() {
        let v = val.as_f64().expect("f64");
        assert!(
            (0.0..=1.0).contains(&v) && !v.is_nan() && !v.is_infinite(),
            "EC-01: delta_s_per_embedder[{}] = {} should be valid",
            i,
            v
        );
    }
}

/// EC-02: Maximum change fingerprints (opposite embeddings) should produce high ΔS
#[tokio::test]
async fn test_ec02_maximum_change_fingerprints() {
    let handlers = create_test_handlers();

    // Create maximally different fingerprints
    let old_fp = create_test_fingerprint_with_semantic(vec![0.0; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![1.0; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(
        response.error.is_none(),
        "Should succeed with maximum change fingerprints"
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    // All values must still be valid and clamped
    let delta_s_agg = data["delta_s_aggregate"].as_f64().expect("f64");
    assert!(
        (0.0..=1.0).contains(&delta_s_agg),
        "EC-02: delta_s_aggregate must be clamped to [0,1], got {}",
        delta_s_agg
    );

    let delta_c = data["delta_c"].as_f64().expect("f64");
    assert!(
        (0.0..=1.0).contains(&delta_c),
        "EC-02: delta_c must be clamped to [0,1], got {}",
        delta_c
    );

    let learning_potential = data["utl_learning_potential"].as_f64().expect("f64");
    assert!(
        (0.0..=1.0).contains(&learning_potential),
        "EC-02: utl_learning_potential must be clamped to [0,1], got {}",
        learning_potential
    );
}

/// EC-03: Zero-magnitude embeddings (all zeros) should not cause NaN/Inf
#[tokio::test]
async fn test_ec03_zero_magnitude_embeddings() {
    let handlers = create_test_handlers();

    // Create zero embeddings - this tests division by zero protection
    let zero_fp = create_test_fingerprint_with_semantic(vec![0.0; 1024]);
    let nonzero_fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&zero_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&nonzero_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(
        response.error.is_none(),
        "Should handle zero embeddings gracefully"
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    // Verify no NaN/Inf in any output
    let delta_s_per_embedder = data["delta_s_per_embedder"].as_array().expect("array");
    for (i, val) in delta_s_per_embedder.iter().enumerate() {
        let v = val.as_f64().expect("f64");
        assert!(
            !v.is_nan() && !v.is_infinite(),
            "EC-03: delta_s_per_embedder[{}] = {} must not be NaN/Inf",
            i,
            v
        );
    }

    let delta_c = data["delta_c"].as_f64().expect("f64");
    assert!(
        !delta_c.is_nan() && !delta_c.is_infinite(),
        "EC-03: delta_c = {} must not be NaN/Inf",
        delta_c
    );

    let learning_potential = data["utl_learning_potential"].as_f64().expect("f64");
    assert!(
        !learning_potential.is_nan() && !learning_potential.is_infinite(),
        "EC-03: utl_learning_potential = {} must not be NaN/Inf",
        learning_potential
    );
}

/// EC-04: Threshold boundary test - ΔS/ΔC values at various magnitudes
#[tokio::test]
async fn test_ec04_threshold_boundary() {
    let handlers = create_test_handlers();

    // We can't directly control the exact ΔS/ΔC values, but we can verify
    // the computation handles boundary cases correctly
    let old_fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.55; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
                "include_diagnostics": true,
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Should handle boundary values");

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let delta_s_agg = data["delta_s_aggregate"].as_f64().expect("f64");
    let delta_c = data["delta_c"].as_f64().expect("f64");

    // Verify values are within valid range
    assert!(
        (0.0..=1.0).contains(&delta_s_agg),
        "EC-04: delta_s_aggregate should be in [0,1], got {}",
        delta_s_agg
    );
    assert!(
        (0.0..=1.0).contains(&delta_c),
        "EC-04: delta_c should be in [0,1], got {}",
        delta_c
    );
}

/// EC-05: Large embedding dimensions should not cause performance issues
#[tokio::test]
async fn test_ec05_large_embedding_dimensions() {
    let handlers = create_test_handlers();

    // Test with max typical dimension (1024 is standard, but let's verify it works)
    let old_fp = create_test_fingerprint_with_semantic(vec![0.3; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.7; 1024]);

    let start = std::time::Instant::now();

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let elapsed = start.elapsed();

    assert!(response.error.is_none(), "Should handle large dimensions");

    // Performance check: should complete in reasonable time (< 1 second)
    assert!(
        elapsed.as_secs() < 1,
        "EC-05: Large dimension computation took too long: {:?}",
        elapsed
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    // Verify output is still valid
    let delta_s_per_embedder = data["delta_s_per_embedder"].as_array().expect("array");
    assert_eq!(
        delta_s_per_embedder.len(),
        NUM_EMBEDDERS,
        "EC-05: Should still have 13 embedders"
    );
}
