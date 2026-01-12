//! Neuromod MCP tools tests (2 tools):
//! - get_neuromodulation_state
//! - adjust_neuromodulator

use serde_json::json;

use crate::handlers::tests::create_test_handlers_with_all_components;
use super::helpers::{make_tool_call, assert_success, assert_tool_error, get_tool_data};

// -------------------------------------------------------------------------
// get_neuromodulation_state
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_get_neuromodulation_state_basic() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call("get_neuromodulation_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_neuromodulation_state");

    let data = get_tool_data(&response);
    // Should have 4 neuromodulators
    assert!(
        data.get("dopamine").is_some() || data.get("modulators").is_some(),
        "Must have neuromodulator data"
    );
}

// -------------------------------------------------------------------------
// adjust_neuromodulator
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_adjust_neuromodulator_dopamine() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call(
        "adjust_neuromodulator",
        json!({
            "modulator": "dopamine",
            "delta": 0.5
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "adjust_neuromodulator");
}

#[tokio::test]
async fn test_adjust_neuromodulator_serotonin() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call(
        "adjust_neuromodulator",
        json!({
            "modulator": "serotonin",
            "delta": 0.1
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "adjust_neuromodulator");
}

#[tokio::test]
async fn test_adjust_neuromodulator_noradrenaline() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call(
        "adjust_neuromodulator",
        json!({
            "modulator": "noradrenaline",
            "delta": -0.2
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "adjust_neuromodulator");
}

#[tokio::test]
async fn test_adjust_neuromodulator_missing_params() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call("adjust_neuromodulator", json!({}));

    let response = handlers.dispatch(request).await;
    assert_tool_error(&response, "adjust_neuromodulator");
}

#[tokio::test]
async fn test_adjust_neuromodulator_missing_delta() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call(
        "adjust_neuromodulator",
        json!({
            "modulator": "dopamine"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_tool_error(&response, "adjust_neuromodulator");
}
