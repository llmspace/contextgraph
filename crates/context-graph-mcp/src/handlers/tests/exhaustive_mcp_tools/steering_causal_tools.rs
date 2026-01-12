//! Steering and Causal MCP tools tests (2 tools):
//! - get_steering_feedback (Steering)
//! - omni_infer (Causal)

use serde_json::json;

use crate::handlers::tests::create_test_handlers;
use super::helpers::{make_tool_call, assert_success, assert_tool_error, get_tool_data};
use super::synthetic_data;

// ============================================================================
// STEERING TOOLS (1)
// ============================================================================

#[tokio::test]
async fn test_get_steering_feedback_basic() {
    let handlers = create_test_handlers();
    let request = make_tool_call("get_steering_feedback", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_steering_feedback");

    let data = get_tool_data(&response);
    // Should return SteeringReward in [-1, 1]
    if let Some(reward) = data.get("reward").and_then(|v| v.as_f64()) {
        assert!(
            (-1.0..=1.0).contains(&reward),
            "SteeringReward {} must be in [-1, 1]",
            reward
        );
    }
}

// ============================================================================
// CAUSAL TOOLS (1)
// ============================================================================

#[tokio::test]
async fn test_omni_infer_forward() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "omni_infer",
        json!({
            "source": synthetic_data::uuids::VALID_SOURCE,
            "target": synthetic_data::uuids::VALID_TARGET,
            "direction": "forward"
        }),
    );

    let response = handlers.dispatch(request).await;
    // May fail if nodes don't exist, but should not be JSON-RPC error
    assert!(response.error.is_none(), "Should not be JSON-RPC error");
}

#[tokio::test]
async fn test_omni_infer_backward() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "omni_infer",
        json!({
            "source": synthetic_data::uuids::VALID_SOURCE,
            "target": synthetic_data::uuids::VALID_TARGET,
            "direction": "backward"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Should not be JSON-RPC error");
}

#[tokio::test]
async fn test_omni_infer_bidirectional() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "omni_infer",
        json!({
            "source": synthetic_data::uuids::VALID_SOURCE,
            "target": synthetic_data::uuids::VALID_TARGET,
            "direction": "bidirectional"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Should not be JSON-RPC error");
}

#[tokio::test]
async fn test_omni_infer_abduction() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "omni_infer",
        json!({
            "source": synthetic_data::uuids::VALID_SOURCE,
            "direction": "abduction"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Should not be JSON-RPC error");
}

#[tokio::test]
async fn test_omni_infer_missing_source() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "omni_infer",
        json!({
            "target": synthetic_data::uuids::VALID_TARGET
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_tool_error(&response, "omni_infer");
}
