//! GWT/Consciousness MCP tools tests (6 tools):
//! - get_consciousness_state
//! - get_kuramoto_sync
//! - get_workspace_status
//! - get_ego_state
//! - trigger_workspace_broadcast
//! - adjust_coupling

use serde_json::json;
use uuid::Uuid;

use crate::handlers::tests::create_test_handlers_with_warm_gwt;
use super::helpers::{make_tool_call, assert_success, assert_tool_error, get_tool_data};
use super::synthetic_data;

// -------------------------------------------------------------------------
// get_consciousness_state
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_get_consciousness_state_basic() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_consciousness_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_consciousness_state");

    let data = get_tool_data(&response);
    // Verify consciousness equation components: C = I × R × D
    assert!(
        data.get("consciousness_level").is_some() || data.get("C").is_some(),
        "Must have consciousness level"
    );
}

#[tokio::test]
async fn test_get_consciousness_state_with_session() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call(
        "get_consciousness_state",
        json!({
            "session_id": "test-session-123"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_consciousness_state");
}

#[tokio::test]
async fn test_consciousness_level_in_range() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_consciousness_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_consciousness_state");

    let data = get_tool_data(&response);
    if let Some(c) = data
        .get("consciousness_level")
        .or(data.get("C"))
        .and_then(|v| v.as_f64())
    {
        assert!(
            (synthetic_data::consciousness::C_MIN..=synthetic_data::consciousness::C_MAX)
                .contains(&c),
            "Consciousness level {} must be in [0,1]",
            c
        );
    }
}

// -------------------------------------------------------------------------
// get_kuramoto_sync
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_get_kuramoto_sync_basic() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_kuramoto_sync", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_kuramoto_sync");

    let data = get_tool_data(&response);
    assert!(data.get("r").is_some(), "Must have order parameter r");
    assert!(data.get("psi").is_some() || data.get("mean_phase").is_some(), "Must have mean phase");
}

#[tokio::test]
async fn test_kuramoto_order_parameter_in_range() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_kuramoto_sync", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_kuramoto_sync");

    let data = get_tool_data(&response);
    let r = data["r"].as_f64().expect("r must be f64");

    assert!(
        (synthetic_data::kuramoto::ORDER_PARAM_MIN..=synthetic_data::kuramoto::ORDER_PARAM_MAX)
            .contains(&r),
        "Order parameter r={} must be in [0,1]",
        r
    );
}

#[tokio::test]
async fn test_kuramoto_warm_state_synchronized() {
    // Warm GWT should have synchronized Kuramoto (r ≈ 1.0)
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_kuramoto_sync", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_kuramoto_sync");

    let data = get_tool_data(&response);
    let r = data["r"].as_f64().expect("r must be f64");

    assert!(
        r >= synthetic_data::kuramoto::SYNC_THRESHOLD,
        "Warm GWT should have r >= {} (synchronized), got {}",
        synthetic_data::kuramoto::SYNC_THRESHOLD,
        r
    );
}

#[tokio::test]
async fn test_kuramoto_13_oscillators() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_kuramoto_sync", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_kuramoto_sync");

    let data = get_tool_data(&response);
    if let Some(phases) = data.get("phases").and_then(|v| v.as_array()) {
        assert_eq!(
            phases.len(),
            synthetic_data::kuramoto::NUM_OSCILLATORS,
            "Must have exactly 13 oscillator phases"
        );
    }
}

// -------------------------------------------------------------------------
// get_workspace_status
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_get_workspace_status_basic() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_workspace_status", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_workspace_status");

    let data = get_tool_data(&response);
    // Should have workspace-related fields
    assert!(
        data.get("active_memory").is_some()
            || data.get("broadcast_state").is_some()
            || data.get("state").is_some(),
        "Must have workspace state info"
    );
}

// -------------------------------------------------------------------------
// get_ego_state
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_get_ego_state_basic() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_ego_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_ego_state");

    let data = get_tool_data(&response);
    assert!(
        data.get("purpose_vector").is_some(),
        "Must have purpose_vector (13D)"
    );
}

#[tokio::test]
async fn test_ego_state_purpose_vector_13d() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_ego_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_ego_state");

    let data = get_tool_data(&response);
    if let Some(pv) = data.get("purpose_vector").and_then(|v| v.as_array()) {
        assert_eq!(
            pv.len(),
            13,
            "Purpose vector must be 13-dimensional, got {}",
            pv.len()
        );
    }
}

#[tokio::test]
async fn test_ego_state_warm_nonzero_purpose() {
    // Warm GWT should have non-zero purpose vector
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_ego_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_ego_state");

    let data = get_tool_data(&response);
    if let Some(pv) = data.get("purpose_vector").and_then(|v| v.as_array()) {
        let sum: f64 = pv.iter().filter_map(|v| v.as_f64()).sum();
        assert!(
            sum > 0.0,
            "Warm GWT should have non-zero purpose vector sum"
        );
    }
}

// -------------------------------------------------------------------------
// trigger_workspace_broadcast
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_trigger_workspace_broadcast_basic() {
    let handlers = create_test_handlers_with_warm_gwt();
    let memory_id = Uuid::new_v4().to_string();

    let request = make_tool_call(
        "trigger_workspace_broadcast",
        json!({
            "memory_id": memory_id
        }),
    );

    let response = handlers.dispatch(request).await;
    // May fail if memory doesn't exist, but should not be JSON-RPC error
    assert!(
        response.error.is_none(),
        "Should not be JSON-RPC error"
    );
}

#[tokio::test]
async fn test_trigger_workspace_broadcast_with_params() {
    let handlers = create_test_handlers_with_warm_gwt();
    let memory_id = Uuid::new_v4().to_string();

    let request = make_tool_call(
        "trigger_workspace_broadcast",
        json!({
            "memory_id": memory_id,
            "importance": 0.9,
            "alignment": 0.8,
            "force": true
        }),
    );

    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Should not be JSON-RPC error");
}

#[tokio::test]
async fn test_trigger_workspace_broadcast_missing_memory_id() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("trigger_workspace_broadcast", json!({}));

    let response = handlers.dispatch(request).await;
    assert_tool_error(&response, "trigger_workspace_broadcast");
}

// -------------------------------------------------------------------------
// adjust_coupling
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_adjust_coupling_basic() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call(
        "adjust_coupling",
        json!({
            "new_K": 2.0
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "adjust_coupling");

    let data = get_tool_data(&response);
    assert!(
        data.get("old_K").is_some() || data.get("new_K").is_some(),
        "Must return coupling values"
    );
}

#[tokio::test]
async fn test_adjust_coupling_boundary_min() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call(
        "adjust_coupling",
        json!({
            "new_K": synthetic_data::kuramoto::COUPLING_MIN
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "adjust_coupling");
}

#[tokio::test]
async fn test_adjust_coupling_boundary_max() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call(
        "adjust_coupling",
        json!({
            "new_K": synthetic_data::kuramoto::COUPLING_MAX
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "adjust_coupling");
}

#[tokio::test]
async fn test_adjust_coupling_missing_new_k() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("adjust_coupling", json!({}));

    let response = handlers.dispatch(request).await;
    assert_tool_error(&response, "adjust_coupling");
}
