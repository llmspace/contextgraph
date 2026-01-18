//! purpose/drift_check Tests
//!
//! Tests for the purpose/drift_check MCP endpoint.
//! Detects alignment drift across fingerprints.

use serde_json::json;

use crate::protocol::JsonRpcId;

use super::super::{create_test_handlers, create_test_handlers_no_goals, make_request};

/// Test purpose/drift_check with valid fingerprints.
#[tokio::test]
async fn test_drift_check_valid_fingerprints() {
    let handlers = create_test_handlers();

    // Store multiple fingerprints
    let mut fingerprint_ids = Vec::new();
    for i in 0..3 {
        let store_params = json!({
            "content": format!("Drift check test content number {}", i),
            "importance": 0.7
        });
        let store_request = make_request(
            "memory/store",
            Some(JsonRpcId::Number(i as i64 + 1)),
            Some(store_params),
        );
        let store_response = handlers.dispatch(store_request).await;
        let fp_id = store_response
            .result
            .unwrap()
            .get("fingerprintId")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        fingerprint_ids.push(fp_id);
    }

    // Check drift
    let drift_params = json!({
        "fingerprint_ids": fingerprint_ids,
        "threshold": 0.1
    });
    let drift_request = make_request(
        "purpose/drift_check",
        Some(JsonRpcId::Number(10)),
        Some(drift_params),
    );
    let response = handlers.dispatch(drift_request).await;

    assert!(
        response.error.is_none(),
        "purpose/drift_check should succeed"
    );
    let result = response.result.expect("Should have result");

    // TASK-INTEG-002: Verify NEW response structure with TeleologicalDriftDetector
    // Response now includes: overall_drift, per_embedder_drift, most_drifted_embedders, recommendations, trend

    // Verify overall_drift (5-level classification: None, Low, Medium, High, Critical)
    let overall_drift = result
        .get("overall_drift")
        .expect("Must have overall_drift");
    assert!(
        overall_drift.get("level").is_some(),
        "Must have drift level"
    );
    assert!(
        overall_drift.get("similarity").is_some(),
        "Must have similarity"
    );
    assert!(
        overall_drift.get("drift_score").is_some(),
        "Must have drift_score"
    );
    assert!(
        overall_drift.get("has_drifted").is_some(),
        "Must have has_drifted"
    );

    // Verify per_embedder_drift (exactly 13 entries, one per embedder E1-E13)
    let per_embedder = result.get("per_embedder_drift").and_then(|v| v.as_array());
    assert!(per_embedder.is_some(), "Must have per_embedder_drift array");
    assert_eq!(
        per_embedder.unwrap().len(),
        13,
        "Must have exactly 13 embedder entries"
    );

    // Verify most_drifted_embedders (top 5, sorted worst-first)
    let most_drifted = result
        .get("most_drifted_embedders")
        .and_then(|v| v.as_array());
    assert!(most_drifted.is_some(), "Must have most_drifted_embedders");
    assert!(
        most_drifted.unwrap().len() <= 5,
        "Must have at most 5 most drifted"
    );

    // Verify recommendations array
    assert!(
        result
            .get("recommendations")
            .and_then(|v| v.as_array())
            .is_some(),
        "Must have recommendations"
    );

    // Verify analyzed_count and check_time_ms
    assert!(
        result.get("analyzed_count").is_some(),
        "Must have analyzed_count"
    );
    assert!(
        result.get("check_time_ms").is_some(),
        "Must have check_time_ms"
    );
    assert!(result.get("timestamp").is_some(), "Must have timestamp");
}

/// Test purpose/drift_check fails with missing fingerprint_ids.
#[tokio::test]
async fn test_drift_check_missing_ids_fails() {
    let handlers = create_test_handlers();

    let drift_params = json!({
        "threshold": 0.1
    });
    let drift_request = make_request(
        "purpose/drift_check",
        Some(JsonRpcId::Number(1)),
        Some(drift_params),
    );
    let response = handlers.dispatch(drift_request).await;

    assert!(
        response.error.is_some(),
        "purpose/drift_check must fail without fingerprint_ids"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32602,
        "Should return INVALID_PARAMS error code"
    );
    assert!(
        error.message.contains("fingerprint_ids"),
        "Error should mention missing fingerprint_ids"
    );
}

/// Test purpose/drift_check fails with empty fingerprint_ids array.
#[tokio::test]
async fn test_drift_check_empty_ids_fails() {
    let handlers = create_test_handlers();

    let drift_params = json!({
        "fingerprint_ids": []
    });
    let drift_request = make_request(
        "purpose/drift_check",
        Some(JsonRpcId::Number(1)),
        Some(drift_params),
    );
    let response = handlers.dispatch(drift_request).await;

    assert!(
        response.error.is_some(),
        "purpose/drift_check must fail with empty fingerprint_ids"
    );
    let error = response.error.unwrap();
    assert_eq!(
        error.code, -32602,
        "Should return INVALID_PARAMS error code"
    );
}

/// Test autonomous operation: store succeeds without goals configured.
///
/// AUTONOMOUS OPERATION: Per contextprd.md, memory storage works without goals
/// by using a default purpose vector [0.0; 13]. The 13-embedding array IS the
/// teleological vector - purpose alignment is secondary metadata.
///
/// Note: drift_check compares alignment drift across fingerprints,
/// so this test focuses on verifying that store works autonomously. Drift detection
/// becomes meaningful only after goals are established.
#[tokio::test]
async fn test_store_autonomous_operation_for_drift() {
    let handlers = create_test_handlers_no_goals();

    // Store content - should SUCCEED without goals (AUTONOMOUS OPERATION)
    let store_params = json!({
        "content": "Test content for autonomous drift check",
        "importance": 0.8
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    let store_response = handlers.dispatch(store_request).await;

    // Store should succeed with default purpose vector
    assert!(
        store_response.error.is_none(),
        "memory/store must succeed without goals (AUTONOMOUS OPERATION). Error: {:?}",
        store_response.error
    );
    let result = store_response.result.expect("Should have result");
    let fingerprint_id = result
        .get("fingerprintId")
        .expect("Must return fingerprintId");

    // Verify the response contains expected fields demonstrating autonomous storage
    assert!(
        result.get("embeddingLatencyMs").is_some(),
        "Must include embedding latency"
    );

    println!(
        "Successfully stored fingerprint {} autonomously (no goals)",
        fingerprint_id
    );
}

/// TASK-INTEG-002: Test purpose/drift_check FAILS FAST on not-found fingerprints.
/// Per FAIL FAST design, missing fingerprints should return an error, not succeed.
#[tokio::test]
async fn test_drift_check_not_found_fingerprints() {
    let handlers = create_test_handlers();

    let drift_params = json!({
        "fingerprint_ids": [
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000002"
        ]
    });
    let drift_request = make_request(
        "purpose/drift_check",
        Some(JsonRpcId::Number(1)),
        Some(drift_params),
    );
    let response = handlers.dispatch(drift_request).await;

    // FAIL FAST: Not-found fingerprints should return error
    assert!(
        response.error.is_some(),
        "purpose/drift_check must FAIL FAST when fingerprints not found"
    );
    let error = response.error.unwrap();

    // Verify error code is FINGERPRINT_NOT_FOUND (-32010)
    assert_eq!(
        error.code, -32010,
        "Must return FINGERPRINT_NOT_FOUND error code"
    );
    assert!(
        error.message.contains("not found") || error.message.contains("No fingerprint"),
        "Error message must indicate fingerprint not found"
    );
}
