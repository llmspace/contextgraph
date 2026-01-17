//! Dream MCP tools tests (4 tools):
//! - get_dream_status
//! - trigger_dream
//! - abort_dream
//! - get_amortized_shortcuts

use serde_json::json;

use super::helpers::{assert_success, get_tool_data, make_tool_call};
use super::synthetic_data;
use crate::handlers::tests::create_test_handlers_with_all_components;

// -------------------------------------------------------------------------
// get_dream_status
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_get_dream_status_basic() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call("get_dream_status", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_dream_status");

    let data = get_tool_data(&response);
    assert!(
        data.get("state").is_some()
            || data.get("dream_state").is_some()
            || data.get("current_state").is_some(),
        "Must have dream state"
    );
}

#[tokio::test]
async fn test_get_dream_status_valid_state() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call("get_dream_status", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_dream_status");

    let data = get_tool_data(&response);
    if let Some(state) = data
        .get("state")
        .or(data.get("dream_state"))
        .or(data.get("current_state"))
        .and_then(|v| v.as_str())
    {
        assert!(
            synthetic_data::dream::VALID_STATES.contains(&state),
            "Dream state '{}' must be one of {:?}",
            state,
            synthetic_data::dream::VALID_STATES
        );
    }
}

// -------------------------------------------------------------------------
// trigger_dream - TASK-35: Updated to require rationale and use TriggerManager
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_trigger_dream_basic_with_rationale() {
    let handlers = create_test_handlers_with_all_components();
    // TASK-35: rationale is now REQUIRED
    let request = make_tool_call(
        "trigger_dream",
        json!({
            "rationale": "Test manual trigger for FSV verification"
        }),
    );

    let response = handlers.dispatch(request).await;
    // Should succeed with valid rationale
    assert!(response.error.is_none(), "Should not be JSON-RPC error");

    let data = get_tool_data(&response);
    assert_eq!(data["triggered"], true, "Manual trigger should be accepted");
    assert_eq!(
        data["trigger_reason"], "Manual",
        "Trigger reason should be Manual"
    );
}

#[tokio::test]
async fn test_trigger_dream_missing_rationale_fails_fast() {
    let handlers = create_test_handlers_with_all_components();
    // TASK-35: Missing rationale should fail with INVALID_PARAMS
    let request = make_tool_call("trigger_dream", json!({}));

    let response = handlers.dispatch(request).await;
    assert!(
        response.error.is_some(),
        "Should return error for missing rationale"
    );
    let error = response.error.unwrap();
    assert!(
        error.message.contains("rationale"),
        "Error should mention rationale: {}",
        error.message
    );
}

#[tokio::test]
async fn test_trigger_dream_empty_rationale_fails_fast() {
    let handlers = create_test_handlers_with_all_components();
    // TASK-35: Empty/whitespace rationale should fail
    let request = make_tool_call(
        "trigger_dream",
        json!({
            "rationale": "   "
        }),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_some(), "Empty rationale should fail");
}

#[tokio::test]
async fn test_trigger_dream_with_force() {
    let handlers = create_test_handlers_with_all_components();
    // TASK-35: force=true should bypass GPU check
    let request = make_tool_call(
        "trigger_dream",
        json!({
            "rationale": "Emergency consolidation test",
            "force": true
        }),
    );

    let response = handlers.dispatch(request).await;
    assert!(
        response.error.is_none(),
        "Force should not cause JSON-RPC error"
    );

    let data = get_tool_data(&response);
    assert_eq!(data["triggered"], true, "Force should allow trigger");
    assert_eq!(data["forced"], true, "forced field should be true");
}

#[tokio::test]
async fn test_trigger_dream_returns_manual_trigger_reason() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call(
        "trigger_dream",
        json!({
            "rationale": "Verify trigger_reason field in response"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Valid request should succeed");

    let data = get_tool_data(&response);
    // TASK-35: Response must include trigger_reason = "Manual"
    assert_eq!(
        data.get("trigger_reason").and_then(|v| v.as_str()),
        Some("Manual"),
        "trigger_reason must be 'Manual' for successful manual trigger"
    );
}

// -------------------------------------------------------------------------
// abort_dream
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_abort_dream_basic() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call("abort_dream", json!({}));

    let response = handlers.dispatch(request).await;
    // May fail if not dreaming, but should not be JSON-RPC error
    assert!(response.error.is_none(), "Should not be JSON-RPC error");
}

// -------------------------------------------------------------------------
// get_amortized_shortcuts
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_get_amortized_shortcuts_basic() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call("get_amortized_shortcuts", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_amortized_shortcuts");
}

#[tokio::test]
async fn test_get_amortized_shortcuts_with_params() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call(
        "get_amortized_shortcuts",
        json!({
            "min_confidence": 0.8,
            "limit": 10
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_amortized_shortcuts");
}
