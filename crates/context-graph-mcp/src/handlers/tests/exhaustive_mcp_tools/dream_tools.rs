//! Dream MCP tools tests (4 tools):
//! - get_dream_status
//! - trigger_dream
//! - abort_dream
//! - get_amortized_shortcuts

use serde_json::json;

use crate::handlers::tests::create_test_handlers_with_all_components;
use super::helpers::{make_tool_call, assert_success, get_tool_data};
use super::synthetic_data;

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
        data.get("state").is_some() || data.get("dream_state").is_some() || data.get("current_state").is_some(),
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
// trigger_dream
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_trigger_dream_basic() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call("trigger_dream", json!({}));

    let response = handlers.dispatch(request).await;
    // May fail if system is not idle, but should not be JSON-RPC error
    assert!(response.error.is_none(), "Should not be JSON-RPC error");
}

#[tokio::test]
async fn test_trigger_dream_with_force() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call(
        "trigger_dream",
        json!({
            "force": true
        }),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Should not be JSON-RPC error");
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
