//! FAIL FAST Error Cases for gwt/compute_delta_sc
//!
//! TASK-UTL-P1-001: Tests for error handling in gwt/compute_delta_sc.

use serde_json::json;
use uuid::Uuid;

use crate::handlers::tests::{create_test_handlers, make_request};
use crate::protocol::JsonRpcId;

use super::helpers::create_test_fingerprint_with_semantic;

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
    let error = response
        .error
        .expect("Should have error for missing vertex_id");

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
        error.message.contains("UUID")
            || error.message.contains("uuid")
            || error.message.contains("Invalid"),
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
    let error = response
        .error
        .expect("Should have error for missing old_fingerprint");

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
    let error = response
        .error
        .expect("Should have error for missing new_fingerprint");

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
    let error = response
        .error
        .expect("Should have error for invalid fingerprint JSON");

    assert!(
        error.message.contains("parse")
            || error.message.contains("fingerprint")
            || error.message.contains("Failed"),
        "Error message should indicate parse failure: {:?}",
        error.message
    );
}
