//! UTL Handler Tests
//!
//! TASK-UTL-P1-001: Tests for gwt/compute_delta_sc handler.
//! Tests verify:
//! - Per-embedder ΔS computation
//! - Aggregate ΔS/ΔC values
//! - Johari quadrant classification
//! - AP-10 compliance (all values in [0,1], no NaN/Inf)
//! - FAIL FAST error handling

use serde_json::json;
use uuid::Uuid;

use context_graph_core::johari::NUM_EMBEDDERS;
use context_graph_core::types::fingerprint::{
    JohariFingerprint, PurposeVector, SemanticFingerprint, TeleologicalFingerprint,
};

use crate::protocol::JsonRpcId;

use super::{create_test_handlers, make_request};

// ============================================================================
// TASK-UTL-P1-001: gwt/compute_delta_sc Tests
// ============================================================================

/// Create a test TeleologicalFingerprint with specified semantic values.
///
/// Uses zeroed base with modified e1_semantic for testing ΔS computation.
fn create_test_fingerprint_with_semantic(semantic_values: Vec<f32>) -> TeleologicalFingerprint {
    let mut semantic = SemanticFingerprint::zeroed();
    semantic.e1_semantic = semantic_values;
    TeleologicalFingerprint::new(
        semantic,
        PurposeVector::default(),
        JohariFingerprint::zeroed(),
        [0u8; 32],
    )
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_valid() {
    let handlers = create_test_handlers();

    // Create two fingerprints with different semantic values
    let old_fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.7; 1024]);

    // Must call through tools/call with name + arguments
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize old_fp"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize new_fp"),
            }
        })),
    );
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "gwt/compute_delta_sc should succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    // Verify required fields exist
    assert!(
        data.get("delta_s_per_embedder").is_some(),
        "Should have delta_s_per_embedder"
    );
    assert!(
        data.get("delta_s_aggregate").is_some(),
        "Should have delta_s_aggregate"
    );
    assert!(data.get("delta_c").is_some(), "Should have delta_c");
    assert!(
        data.get("johari_quadrants").is_some(),
        "Should have johari_quadrants"
    );
    assert!(
        data.get("johari_aggregate").is_some(),
        "Should have johari_aggregate"
    );
    assert!(
        data.get("utl_learning_potential").is_some(),
        "Should have utl_learning_potential"
    );
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_per_embedder_count() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.3; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.8; 1024]);

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
    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    // Verify 13 per-embedder values
    let delta_s_per_embedder = data["delta_s_per_embedder"]
        .as_array()
        .expect("delta_s_per_embedder should be array");
    assert_eq!(
        delta_s_per_embedder.len(),
        NUM_EMBEDDERS,
        "Should have exactly 13 per-embedder ΔS values"
    );

    // Verify 13 Johari quadrants
    let johari_quadrants = data["johari_quadrants"]
        .as_array()
        .expect("johari_quadrants should be array");
    assert_eq!(
        johari_quadrants.len(),
        NUM_EMBEDDERS,
        "Should have exactly 13 Johari quadrants"
    );
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_ap10_range_compliance() {
    let handlers = create_test_handlers();

    // Use different values to get non-trivial ΔS
    let old_fp = create_test_fingerprint_with_semantic(vec![0.1; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.9; 1024]);

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
    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    // AP-10: All values must be in [0, 1]
    let delta_s_per_embedder = data["delta_s_per_embedder"]
        .as_array()
        .expect("array");
    for (i, val) in delta_s_per_embedder.iter().enumerate() {
        let v = val.as_f64().expect("f64");
        assert!(
            (0.0..=1.0).contains(&v),
            "delta_s_per_embedder[{}] = {} not in [0,1]",
            i,
            v
        );
        assert!(!v.is_nan(), "delta_s_per_embedder[{}] is NaN", i);
        assert!(!v.is_infinite(), "delta_s_per_embedder[{}] is Inf", i);
    }

    let delta_s_agg = data["delta_s_aggregate"].as_f64().expect("f64");
    assert!(
        (0.0..=1.0).contains(&delta_s_agg),
        "delta_s_aggregate = {} not in [0,1]",
        delta_s_agg
    );

    let delta_c = data["delta_c"].as_f64().expect("f64");
    assert!(
        (0.0..=1.0).contains(&delta_c),
        "delta_c = {} not in [0,1]",
        delta_c
    );

    let learning_potential = data["utl_learning_potential"].as_f64().expect("f64");
    assert!(
        (0.0..=1.0).contains(&learning_potential),
        "utl_learning_potential = {} not in [0,1]",
        learning_potential
    );
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_johari_quadrant_values() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.6; 1024]);

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
    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    // Verify Johari quadrants are valid enum values
    let valid_quadrants = ["Open", "Blind", "Hidden", "Unknown"];
    let johari_quadrants = data["johari_quadrants"]
        .as_array()
        .expect("array");

    for (i, quadrant) in johari_quadrants.iter().enumerate() {
        let q = quadrant.as_str().expect("string");
        assert!(
            valid_quadrants.contains(&q),
            "johari_quadrants[{}] = '{}' is not a valid quadrant",
            i,
            q
        );
    }

    let johari_agg = data["johari_aggregate"].as_str().expect("string");
    assert!(
        valid_quadrants.contains(&johari_agg),
        "johari_aggregate = '{}' is not a valid quadrant",
        johari_agg
    );
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_with_diagnostics() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.4; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.7; 1024]);

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
    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    // Verify diagnostics are included
    assert!(
        data.get("diagnostics").is_some(),
        "Should have diagnostics when include_diagnostics=true"
    );

    let diagnostics = &data["diagnostics"];
    assert!(
        diagnostics.get("per_embedder").is_some(),
        "diagnostics should have per_embedder"
    );
    assert!(
        diagnostics.get("johari_threshold").is_some(),
        "diagnostics should have johari_threshold"
    );
    assert!(
        diagnostics.get("coherence_config").is_some(),
        "diagnostics should have coherence_config"
    );
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_custom_johari_threshold() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.6; 1024]);

    // Test with custom johari_threshold
    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
                "johari_threshold": 0.4,
                "include_diagnostics": true,
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");
    let data = super::extract_mcp_tool_data(&result);

    // Verify threshold was applied (clamped to [0.35, 0.65])
    let threshold = data["diagnostics"]["johari_threshold"]
        .as_f64()
        .expect("threshold");
    assert!(
        (0.35..=0.65).contains(&threshold),
        "johari_threshold {} should be clamped to [0.35, 0.65]",
        threshold
    );
}

// ============================================================================
// FAIL FAST Error Cases (TASK-UTL-P1-001)
// ============================================================================

#[tokio::test]
async fn test_gwt_compute_delta_sc_missing_vertex_id() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.6; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let error = response.error.expect("Should have error for missing vertex_id");

    assert!(
        error.message.contains("vertex_id"),
        "Error message should mention vertex_id: {:?}",
        error.message
    );
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_invalid_vertex_id() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);
    let new_fp = create_test_fingerprint_with_semantic(vec![0.6; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": "not-a-valid-uuid",
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let error = response.error.expect("Should have error for invalid UUID");

    assert!(
        error.message.contains("UUID") || error.message.contains("uuid") || error.message.contains("Invalid"),
        "Error message should indicate invalid UUID: {:?}",
        error.message
    );
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_missing_old_fingerprint() {
    let handlers = create_test_handlers();

    let new_fp = create_test_fingerprint_with_semantic(vec![0.6; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "new_fingerprint": serde_json::to_value(&new_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let error = response.error.expect("Should have error for missing old_fingerprint");

    assert!(
        error.message.contains("old_fingerprint"),
        "Error message should mention old_fingerprint: {:?}",
        error.message
    );
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_missing_new_fingerprint() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.5; 1024]);

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": serde_json::to_value(&old_fp).expect("serialize"),
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let error = response.error.expect("Should have error for missing new_fingerprint");

    assert!(
        error.message.contains("new_fingerprint"),
        "Error message should mention new_fingerprint: {:?}",
        error.message
    );
}

#[tokio::test]
async fn test_gwt_compute_delta_sc_invalid_fingerprint_json() {
    let handlers = create_test_handlers();

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "gwt/compute_delta_sc",
            "arguments": {
                "vertex_id": Uuid::new_v4().to_string(),
                "old_fingerprint": { "invalid": "structure" },
                "new_fingerprint": { "also": "invalid" },
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let error = response.error.expect("Should have error for invalid fingerprint JSON");

    assert!(
        error.message.contains("parse") || error.message.contains("fingerprint") || error.message.contains("Failed"),
        "Error message should indicate parse failure: {:?}",
        error.message
    );
}

#[tokio::test]
async fn test_utl_compute_valid() {
    let handlers = create_test_handlers();
    // Handler expects 'input' parameter, not 'content'
    let params = json!({
        "input": "Test content for UTL computation"
    });
    let request = make_request("utl/compute", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(response.error.is_none(), "utl/compute should succeed");
    let result = response.result.expect("Should have result");

    // Handler returns only learningScore
    assert!(
        result.get("learningScore").is_some(),
        "Should have learningScore"
    );
}

#[tokio::test]
async fn test_utl_compute_missing_input() {
    let handlers = create_test_handlers();
    let params = json!({});
    let request = make_request("utl/compute", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_some(),
        "utl/compute should fail without input"
    );
}

#[tokio::test]
async fn test_utl_compute_with_input() {
    let handlers = create_test_handlers();
    let params = json!({
        "input": "Learning about neural networks"
    });
    let request = make_request("utl/compute", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "utl/compute with input should succeed"
    );
}

#[tokio::test]
async fn test_utl_metrics_valid() {
    let handlers = create_test_handlers();
    // Handler expects 'input' parameter
    let params = json!({
        "input": "Test input for metrics"
    });
    let request = make_request("utl/metrics", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(response.error.is_none(), "utl/metrics should succeed");
    let result = response.result.expect("Should have result");

    // Verify metrics fields
    assert!(result.get("entropy").is_some(), "Should have entropy");
    assert!(result.get("coherence").is_some(), "Should have coherence");
    assert!(
        result.get("learningScore").is_some(),
        "Should have learningScore"
    );
}

#[tokio::test]
async fn test_utl_compute_learning_score_range() {
    let handlers = create_test_handlers();
    let params = json!({
        "input": "Simple test content"
    });
    let request = make_request("utl/compute", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");

    let learning_score = result
        .get("learningScore")
        .and_then(|v| v.as_f64())
        .expect("Should have learningScore");

    assert!(
        (0.0..=1.0).contains(&learning_score),
        "learningScore should be between 0 and 1, got {}",
        learning_score
    );
}

#[tokio::test]
async fn test_utl_metrics_entropy_range() {
    let handlers = create_test_handlers();
    let params = json!({
        "input": "Test entropy calculation"
    });
    let request = make_request("utl/metrics", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");

    let entropy = result
        .get("entropy")
        .and_then(|v| v.as_f64())
        .expect("Should have entropy");

    assert!(
        (0.0..=1.0).contains(&entropy),
        "entropy should be between 0 and 1, got {}",
        entropy
    );
}
