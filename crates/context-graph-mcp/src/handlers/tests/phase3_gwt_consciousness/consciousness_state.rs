//! Scenario 1: Consciousness State Verification Tests
//!
//! Tests get_consciousness_state tool:
//! - Valid consciousness level and metrics
//! - Fail fast without GWT initialization

use serde_json::json;

use crate::handlers::tests::{create_test_handlers_with_warm_gwt, extract_mcp_tool_data};
use crate::protocol::{error_codes, JsonRpcId, JsonRpcRequest};
use crate::tools::tool_names;

/// FSV Test: get_consciousness_state returns valid consciousness level and metrics.
///
/// Source of Truth: GWT providers (Kuramoto, Workspace, MetaCognitive, SelfEgo)
/// Expected: Response contains C, r, psi, state, workspace, identity with valid values.
#[tokio::test]
async fn test_get_consciousness_state_returns_valid_data() {
    // SETUP: Create handlers with WARM GWT state (synchronized Kuramoto)
    let handlers = create_test_handlers_with_warm_gwt();

    // EXECUTE: Call get_consciousness_state via tools/call
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_CONSCIOUSNESS_STATE,
            "arguments": {}
        })),
    };
    let response = handlers.dispatch(request).await;

    // VERIFY: Response is successful
    assert!(
        response.error.is_none(),
        "[FSV] Expected success, got error: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    // FSV: Verify C (consciousness level) is in valid range [0, 1]
    let c_value = data
        .get("C")
        .and_then(|v| v.as_f64())
        .expect("C must exist");
    assert!(
        (0.0..=1.0).contains(&c_value),
        "[FSV] C must be in [0, 1], got {}",
        c_value
    );

    // FSV: Verify r (order parameter) is in valid range [0, 1]
    let r = data
        .get("r")
        .and_then(|v| v.as_f64())
        .expect("r must exist");
    assert!(
        (0.0..=1.0).contains(&r),
        "[FSV] r must be in [0, 1], got {}",
        r
    );

    // FSV: For synchronized network, r should be high (> 0.99)
    assert!(
        r > 0.99,
        "[FSV] Synchronized network should have r > 0.99, got {}",
        r
    );

    // FSV: Verify state is valid (DORMANT, FRAGMENTED, EMERGING, CONSCIOUS, HYPERSYNC)
    let state = data
        .get("state")
        .and_then(|v| v.as_str())
        .expect("state must exist");
    let valid_states = [
        "DORMANT",
        "FRAGMENTED",
        "EMERGING",
        "CONSCIOUS",
        "HYPERSYNC",
    ];
    assert!(
        valid_states.contains(&state),
        "[FSV] Invalid state: {}, expected one of {:?}",
        state,
        valid_states
    );

    // FSV: For high r (> 0.95), state should be CONSCIOUS or HYPERSYNC
    assert!(
        ["CONSCIOUS", "HYPERSYNC"].contains(&state),
        "[FSV] High r ({}) should give CONSCIOUS or HYPERSYNC, got {}",
        r,
        state
    );

    // FSV: Verify workspace data exists and has valid structure
    let workspace = data.get("workspace").expect("workspace must exist");
    assert!(
        workspace.get("coherence_threshold").is_some(),
        "[FSV] workspace.coherence_threshold must exist"
    );
    assert!(
        workspace.get("is_broadcasting").is_some(),
        "[FSV] workspace.is_broadcasting must exist"
    );
    assert!(
        workspace.get("has_conflict").is_some(),
        "[FSV] workspace.has_conflict must exist"
    );

    // FSV: Verify identity has 13-element purpose vector
    let identity = data.get("identity").expect("identity must exist");
    let pv = identity
        .get("purpose_vector")
        .and_then(|v| v.as_array())
        .expect("purpose_vector must exist");
    assert_eq!(
        pv.len(),
        13,
        "[FSV] Purpose vector must have 13 elements (one per embedder), got {}",
        pv.len()
    );

    // FSV: Verify identity has coherence and status
    assert!(
        identity.get("coherence").is_some(),
        "[FSV] identity.coherence must exist"
    );
    assert!(
        identity.get("status").is_some(),
        "[FSV] identity.status must exist"
    );
    assert!(
        identity.get("trajectory_length").is_some(),
        "[FSV] identity.trajectory_length must exist"
    );

    // FSV: Verify component_analysis exists
    let component_analysis = data
        .get("component_analysis")
        .expect("component_analysis must exist");
    assert!(
        component_analysis.get("integration_sufficient").is_some(),
        "[FSV] component_analysis.integration_sufficient must exist"
    );
    assert!(
        component_analysis.get("reflection_sufficient").is_some(),
        "[FSV] component_analysis.reflection_sufficient must exist"
    );
    assert!(
        component_analysis
            .get("differentiation_sufficient")
            .is_some(),
        "[FSV] component_analysis.differentiation_sufficient must exist"
    );

    println!("[FSV] Phase 3 - get_consciousness_state verification PASSED");
    println!("[FSV]   C={:.4}, r={:.4}, state={}", c_value, r, state);
    println!("[FSV]   purpose_vector.len={}", pv.len());
}

/// FSV Test: get_consciousness_state FAIL FAST without GWT initialization.
///
/// Source of Truth: kuramoto_network = None in Handlers
/// Expected: Error code -32060 (GWT_NOT_INITIALIZED)
#[tokio::test]
async fn test_get_consciousness_state_fails_without_gwt() {
    // SETUP: Create handlers WITHOUT GWT (basic Handlers::new())
    let handlers = super::super::create_test_handlers();

    // EXECUTE: Call get_consciousness_state
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_CONSCIOUSNESS_STATE,
            "arguments": {}
        })),
    };
    let response = handlers.dispatch(request).await;

    // VERIFY: Must FAIL FAST with correct error code
    assert!(
        response.error.is_some(),
        "[FSV] Should have error without GWT"
    );
    let error = response.error.expect("Should have error");
    assert_eq!(
        error.code,
        error_codes::GWT_NOT_INITIALIZED,
        "[FSV] Error code must be GWT_NOT_INITIALIZED (-32060), got {}",
        error.code
    );

    println!("[FSV] Phase 3 - get_consciousness_state FAIL FAST verification PASSED");
}
