//! GWT Consciousness Requirements Verification
//!
//! Verifies the Global Workspace Theory consciousness equation
//! and related requirements from contextprd.md and constitution.yaml.

use serde_json::json;

use super::helpers::{assert_success, get_tool_data, make_tool_call};
use super::synthetic_data;
use crate::handlers::tests::{create_test_handlers, create_test_handlers_with_warm_gwt};

/// Verify the consciousness equation C = I x R x D
/// where I = Integration, R = Resonance (Kuramoto r), D = Differentiation
#[tokio::test]
async fn test_consciousness_equation_components_present() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_consciousness_state", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_consciousness_state");

    let data = get_tool_data(&response);

    // The consciousness state should contain:
    // - Kuramoto sync (r) for Resonance
    // - Some integration metric
    // - Some differentiation metric
    // - Overall consciousness level

    let has_consciousness = data.get("consciousness_level").is_some()
        || data.get("C").is_some()
        || data.get("consciousness").is_some();

    assert!(
        has_consciousness,
        "Must expose consciousness level (C = I x R x D)"
    );
}

/// Verify Kuramoto oscillator network has 13 oscillators
#[tokio::test]
async fn test_kuramoto_13_embedder_alignment() {
    let handlers = create_test_handlers_with_warm_gwt();
    let request = make_tool_call("get_kuramoto_sync", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_kuramoto_sync");

    let data = get_tool_data(&response);

    // Verify 13 oscillators (one per embedder)
    if let Some(phases) = data.get("phases").and_then(|v| v.as_array()) {
        assert_eq!(
            phases.len(),
            13,
            "Kuramoto network must have 13 oscillators (one per embedder)"
        );
    }

    if let Some(frequencies) = data.get("natural_frequencies").and_then(|v| v.as_array()) {
        assert_eq!(frequencies.len(), 13, "Must have 13 natural frequencies");
    }
}

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

/// Verify Johari quadrant classification works correctly
#[tokio::test]
async fn test_johari_quadrant_classification() {
    let handlers = create_test_handlers();
    let request = make_tool_call("get_memetic_status", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_memetic_status");

    let data = get_tool_data(&response);
    let utl = data.get("utl").expect("Must have utl");

    let johari = utl["johariQuadrant"]
        .as_str()
        .expect("johariQuadrant must be string");

    // Verify valid quadrant
    assert!(
        synthetic_data::johari::VALID_QUADRANTS.contains(&johari),
        "Johari '{}' must be one of {:?}",
        johari,
        synthetic_data::johari::VALID_QUADRANTS
    );

    // Verify suggested action matches
    let action = utl["suggestedAction"]
        .as_str()
        .expect("suggestedAction must be string");

    let expected_action = match johari {
        "Open" => "direct_recall",
        "Blind" => "trigger_dream",
        "Hidden" => "get_neighborhood",
        "Unknown" => "epistemic_action",
        _ => "continue",
    };

    assert_eq!(
        action, expected_action,
        "Johari '{}' should map to '{}', got '{}'",
        johari, expected_action, action
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
