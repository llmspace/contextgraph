//! TASK-DELTA-P1-001: Full State Verification (FSV) Tests

use serde_json::json;
use uuid::Uuid;

use context_graph_core::types::fingerprint::NUM_EMBEDDERS;

use crate::handlers::tests::{create_test_handlers, extract_mcp_tool_data, make_request};
use crate::protocol::JsonRpcId;

use super::helpers::create_test_fingerprint_with_semantic;

/// FSV-01: Verify ΔC formula matches constitution.yaml line 166:
/// ΔC = 0.4×Connectivity + 0.4×ClusterFit + 0.2×Consistency
///
/// We verify this by enabling diagnostics and checking the component values.
#[tokio::test]
async fn test_fsv_delta_c_formula_verification() {
    let handlers = create_test_handlers();

    // Create fingerprints with different values to get non-trivial components
    let old_fp = create_test_fingerprint_with_semantic(vec![0.3; 1024]);
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
    assert!(
        response.error.is_none(),
        "Should succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    // Verify diagnostics contains delta_c_components
    let diagnostics = &data["diagnostics"];
    assert!(
        diagnostics.get("delta_c_components").is_some(),
        "Should have delta_c_components"
    );

    let components = &diagnostics["delta_c_components"];
    let connectivity = components["connectivity"].as_f64().expect("connectivity") as f32;
    let cluster_fit = components["cluster_fit"].as_f64().expect("cluster_fit") as f32;
    let consistency = components["consistency"].as_f64().expect("consistency") as f32;

    // Verify weights are correct per constitution.yaml (with f32/f64 tolerance)
    let weights = &components["weights"];
    let alpha = weights["alpha_connectivity"].as_f64().unwrap();
    let beta = weights["beta_cluster_fit"].as_f64().unwrap();
    let gamma = weights["gamma_consistency"].as_f64().unwrap();

    assert!(
        (alpha - 0.4).abs() < 0.0001,
        "alpha (connectivity weight) should be ~0.4, got {}",
        alpha
    );
    assert!(
        (beta - 0.4).abs() < 0.0001,
        "beta (cluster_fit weight) should be ~0.4, got {}",
        beta
    );
    assert!(
        (gamma - 0.2).abs() < 0.0001,
        "gamma (consistency weight) should be ~0.2, got {}",
        gamma
    );

    // Verify delta_c matches the formula: 0.4*Connectivity + 0.4*ClusterFit + 0.2*Consistency
    let expected_delta_c = 0.4 * connectivity + 0.4 * cluster_fit + 0.2 * consistency;
    let actual_delta_c = data["delta_c"].as_f64().expect("delta_c") as f32;

    // Allow small floating-point tolerance and clamping effects
    let diff = (expected_delta_c.clamp(0.0, 1.0) - actual_delta_c).abs();
    assert!(
        diff < 0.01,
        "ΔC formula mismatch: expected {:.6} (0.4×{:.4} + 0.4×{:.4} + 0.2×{:.4}), got {:.6}",
        expected_delta_c,
        connectivity,
        cluster_fit,
        consistency,
        actual_delta_c
    );
}

/// FSV-02: Verify UTL learning potential formula:
/// utl_learning_potential = delta_s_aggregate × delta_c
#[tokio::test]
async fn test_fsv_utl_learning_potential_formula() {
    let handlers = create_test_handlers();

    let old_fp = create_test_fingerprint_with_semantic(vec![0.2; 1024]);
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
    assert!(response.error.is_none(), "Should succeed");

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let delta_s_agg = data["delta_s_aggregate"]
        .as_f64()
        .expect("delta_s_aggregate") as f32;
    let delta_c = data["delta_c"].as_f64().expect("delta_c") as f32;
    let learning_potential = data["utl_learning_potential"]
        .as_f64()
        .expect("utl_learning_potential") as f32;

    // Verify formula: learning_potential = delta_s_aggregate * delta_c
    let expected = (delta_s_agg * delta_c).clamp(0.0, 1.0);
    let diff = (expected - learning_potential).abs();

    assert!(
        diff < 0.001,
        "UTL learning potential formula mismatch: expected {:.6} ({:.4} × {:.4}), got {:.6}",
        expected,
        delta_s_agg,
        delta_c,
        learning_potential
    );
}

/// FSV-03: Verify each per-embedder ΔS value is properly clamped to [0, 1]
#[tokio::test]
async fn test_fsv_delta_s_clamping() {
    let handlers = create_test_handlers();

    // Use extreme values that might produce out-of-range results before clamping
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
    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let delta_s_per_embedder = data["delta_s_per_embedder"]
        .as_array()
        .expect("delta_s_per_embedder should be array");

    // Verify ALL 13 values are in [0, 1]
    for (i, val) in delta_s_per_embedder.iter().enumerate() {
        let v = val.as_f64().expect("f64");
        assert!(
            (0.0..=1.0).contains(&v),
            "FSV: delta_s_per_embedder[{}] = {} MUST be clamped to [0,1]",
            i,
            v
        );
        assert!(!v.is_nan(), "FSV: delta_s_per_embedder[{}] is NaN", i);
        assert!(!v.is_infinite(), "FSV: delta_s_per_embedder[{}] is Inf", i);
    }

    // Verify aggregate is also clamped
    let delta_s_agg = data["delta_s_aggregate"].as_f64().expect("f64");
    assert!(
        (0.0..=1.0).contains(&delta_s_agg),
        "FSV: delta_s_aggregate = {} MUST be in [0,1]",
        delta_s_agg
    );
}

/// FSV-04: Verify aggregate ΔS is mean of per-embedder values
#[tokio::test]
async fn test_fsv_delta_s_aggregate_is_mean() {
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
            }
        })),
    );

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    let delta_s_per_embedder = data["delta_s_per_embedder"].as_array().expect("array");

    // Calculate expected mean
    let sum: f64 = delta_s_per_embedder
        .iter()
        .map(|v| v.as_f64().unwrap())
        .sum();
    let expected_mean = (sum / NUM_EMBEDDERS as f64).clamp(0.0, 1.0);
    let actual_agg = data["delta_s_aggregate"].as_f64().expect("f64");

    let diff = (expected_mean - actual_agg).abs();
    assert!(
        diff < 0.001,
        "FSV: delta_s_aggregate should be mean of per-embedder values. Expected {:.6}, got {:.6}",
        expected_mean,
        actual_agg
    );
}
