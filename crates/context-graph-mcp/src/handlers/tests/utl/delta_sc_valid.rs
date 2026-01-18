//! Valid gwt/compute_delta_sc tests
//!
//! TASK-UTL-P1-001: Tests for successful gwt/compute_delta_sc operations.

use serde_json::json;
use uuid::Uuid;

use context_graph_core::types::fingerprint::NUM_EMBEDDERS;

use crate::handlers::tests::{create_test_handlers, extract_mcp_tool_data, make_request};
use crate::protocol::JsonRpcId;

use super::helpers::create_test_fingerprint_with_semantic;

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
    let data = extract_mcp_tool_data(&result);

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
    let data = extract_mcp_tool_data(&result);

    // Verify 13 per-embedder values
    let delta_s_per_embedder = data["delta_s_per_embedder"]
        .as_array()
        .expect("delta_s_per_embedder should be array");
    assert_eq!(
        delta_s_per_embedder.len(),
        NUM_EMBEDDERS,
        "Should have exactly 13 per-embedder ΔS values"
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
    let data = extract_mcp_tool_data(&result);

    // AP-10: All values must be in [0, 1]
    let delta_s_per_embedder = data["delta_s_per_embedder"].as_array().expect("array");
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
    let data = extract_mcp_tool_data(&result);

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
        diagnostics.get("coherence_config").is_some(),
        "diagnostics should have coherence_config"
    );
}
