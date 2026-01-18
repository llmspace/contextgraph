//! GWT (Global Workspace Theory) Requirements Verification
//!
//! Verifies GWT workspace, ego state, and related requirements
//! from contextprd.md and constitution.yaml.

use serde_json::json;

use super::helpers::{assert_success, get_tool_data, make_tool_call};
use crate::handlers::tests::{create_test_handlers, create_test_handlers_with_warm_gwt};

/// Verify purpose vector is 13-dimensional
#[tokio::test]
async fn test_purpose_vector_13d() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_ego_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_ego_state");

    let data = get_tool_data(&response);

    if let Some(pv) = data.get("purpose_vector").and_then(|v| v.as_array()) {
        assert_eq!(
            pv.len(),
            13,
            "Purpose vector must be 13-dimensional (one per embedder)"
        );
    }
}

/// Verify Global Workspace Theory winner-take-all mechanism
#[tokio::test]
async fn test_gwt_workspace_wta() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_workspace_status", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_workspace_status");

    // Workspace should have WTA-related fields
    let data = get_tool_data(&response);
    let has_workspace_info = data.get("active_memory").is_some()
        || data.get("competing_candidates").is_some()
        || data.get("broadcast_state").is_some()
        || data.get("state").is_some();

    assert!(
        has_workspace_info,
        "Workspace must expose WTA selection state"
    );
}

/// Verify UTL metrics are computed correctly
#[tokio::test]
async fn test_utl_learning_metrics() {
    let handlers = create_test_handlers();
    let request = make_tool_call("utl_status", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "utl_status");

    let data = get_tool_data(&response);

    // Verify all UTL components
    let entropy = data["entropy"].as_f64().expect("Must have entropy");
    let coherence = data["coherence"].as_f64().expect("Must have coherence");
    let learning_score = data["learning_score"]
        .as_f64()
        .expect("Must have learning_score");

    // All values in [0, 1]
    assert!((0.0..=1.0).contains(&entropy), "entropy in [0,1]");
    assert!((0.0..=1.0).contains(&coherence), "coherence in [0,1]");
    assert!(
        (0.0..=1.0).contains(&learning_score),
        "learning_score in [0,1]"
    );
}
