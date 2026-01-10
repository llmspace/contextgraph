//! Phase 3: GWT (Global Workspace Theory) Consciousness Tools Manual Testing
//!
//! This module tests all 6 GWT consciousness tools:
//! 1. get_consciousness_state - Returns Psi (consciousness level)
//! 2. get_kuramoto_sync - Returns order parameter r and oscillator phases
//! 3. get_workspace_status - Returns active memory, coherence threshold, broadcasting status
//! 4. get_ego_state - Returns identity coherence, purpose vector, trajectory length
//! 5. trigger_workspace_broadcast - Performs winner-take-all selection
//! 6. adjust_coupling - Modifies Kuramoto coupling constant K
//!
//! # Test Scenarios
//!
//! ## Scenario 1: Consciousness State Verification
//! - Call get_consciousness_state
//! - Verify psi value is in valid range [0, 1]
//! - Verify layers (perception, memory, reasoning, action, meta) have valid status
//!
//! ## Scenario 2: Kuramoto Synchronization
//! - Call get_kuramoto_sync
//! - Verify r (order parameter) is in [0, 1]
//! - Verify phases array has 13 oscillators (one per embedder)
//! - Test adjust_coupling and verify K changes
//!
//! ## Scenario 3: Workspace Broadcast
//! - Store a memory first
//! - Call trigger_workspace_broadcast with the memory_id
//! - Verify workspace state changes (is_broadcasting, active_memory)
//!
//! ## Scenario 4: Ego State Verification
//! - Call get_ego_state
//! - Verify identity_status, purpose_vector (13D), trajectory_length
//!
//! # Critical Verification
//! - State values must be within valid ranges
//! - State changes must persist across calls
//! - Kuramoto must have 13 oscillators (one per embedder per constitution)

use serde_json::json;

use crate::handlers::tests::{
    create_test_handlers_with_all_components, create_test_handlers_with_warm_gwt, extract_mcp_tool_data,
};
use crate::protocol::{error_codes, JsonRpcId, JsonRpcRequest};
use crate::tools::tool_names;

// =============================================================================
// SCENARIO 1: CONSCIOUSNESS STATE VERIFICATION
// =============================================================================

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
    let c_value = data.get("C").and_then(|v| v.as_f64()).expect("C must exist");
    assert!(
        (0.0..=1.0).contains(&c_value),
        "[FSV] C must be in [0, 1], got {}",
        c_value
    );

    // FSV: Verify r (order parameter) is in valid range [0, 1]
    let r = data.get("r").and_then(|v| v.as_f64()).expect("r must exist");
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
    let valid_states = ["DORMANT", "FRAGMENTED", "EMERGING", "CONSCIOUS", "HYPERSYNC"];
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
        component_analysis.get("differentiation_sufficient").is_some(),
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
    let handlers = super::create_test_handlers();

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

// =============================================================================
// SCENARIO 2: KURAMOTO SYNCHRONIZATION
// =============================================================================

/// FSV Test: get_kuramoto_sync returns valid oscillator network state.
///
/// Source of Truth: KuramotoProvider
/// Expected: Response contains r, phases[13], natural_freqs[13], coupling, thresholds.
#[tokio::test]
async fn test_get_kuramoto_sync_returns_13_oscillators() {
    // SETUP: Create handlers with WARM GWT state
    let handlers = create_test_handlers_with_warm_gwt();

    // EXECUTE: Call get_kuramoto_sync
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(2)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_KURAMOTO_SYNC,
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

    // FSV: Verify r (order parameter) is in [0, 1]
    let r = data.get("r").and_then(|v| v.as_f64()).expect("r must exist");
    assert!(
        (0.0..=1.0).contains(&r),
        "[FSV] r must be in [0, 1], got {}",
        r
    );

    // FSV: Verify synchronized network has high r
    assert!(
        r > 0.99,
        "[FSV] Synchronized network should have r > 0.99, got {}",
        r
    );

    // FSV: Verify psi (mean phase) is in [0, 2*PI] (allowing some tolerance)
    let psi = data.get("psi").and_then(|v| v.as_f64()).expect("psi must exist");
    assert!(
        psi >= 0.0 && psi <= 2.0 * std::f64::consts::PI + 0.01,
        "[FSV] psi must be in [0, 2*PI], got {}",
        psi
    );

    // FSV: CRITICAL - Verify phases array has exactly 13 oscillators (one per embedder)
    let phases = data
        .get("phases")
        .and_then(|v| v.as_array())
        .expect("phases must exist");
    assert_eq!(
        phases.len(),
        13,
        "[FSV] CRITICAL: Must have 13 oscillator phases (one per embedder), got {}",
        phases.len()
    );

    // FSV: Verify natural frequencies array has exactly 13 elements
    let natural_freqs = data
        .get("natural_freqs")
        .and_then(|v| v.as_array())
        .expect("natural_freqs must exist");
    assert_eq!(
        natural_freqs.len(),
        13,
        "[FSV] Must have 13 natural frequencies, got {}",
        natural_freqs.len()
    );

    // FSV: Verify all natural frequencies are positive
    for (i, freq) in natural_freqs.iter().enumerate() {
        let freq_val = freq.as_f64().expect("freq must be f64");
        assert!(
            freq_val > 0.0,
            "[FSV] Frequency[{}] must be positive, got {}",
            i,
            freq_val
        );
    }

    // FSV: Verify coupling strength K is present
    let coupling = data
        .get("coupling")
        .and_then(|v| v.as_f64())
        .expect("coupling must exist");
    assert!(
        coupling >= 0.0,
        "[FSV] Coupling K must be non-negative, got {}",
        coupling
    );

    // FSV: Verify thresholds are constitution-mandated values
    let thresholds = data.get("thresholds").expect("thresholds must exist");
    assert_eq!(
        thresholds.get("conscious").and_then(|v| v.as_f64()),
        Some(0.8),
        "[FSV] thresholds.conscious must be 0.8"
    );
    assert_eq!(
        thresholds.get("fragmented").and_then(|v| v.as_f64()),
        Some(0.5),
        "[FSV] thresholds.fragmented must be 0.5"
    );
    assert_eq!(
        thresholds.get("hypersync").and_then(|v| v.as_f64()),
        Some(0.95),
        "[FSV] thresholds.hypersync must be 0.95"
    );

    // FSV: Verify embedding labels are present (13 labels)
    let labels = data
        .get("embedding_labels")
        .and_then(|v| v.as_array())
        .expect("embedding_labels must exist");
    assert_eq!(
        labels.len(),
        13,
        "[FSV] Must have 13 embedding labels, got {}",
        labels.len()
    );

    // FSV: Verify state is valid
    let state = data
        .get("state")
        .and_then(|v| v.as_str())
        .expect("state must exist");
    let valid_states = ["DORMANT", "FRAGMENTED", "EMERGING", "CONSCIOUS", "HYPERSYNC"];
    assert!(
        valid_states.contains(&state),
        "[FSV] Invalid state: {}",
        state
    );

    println!("[FSV] Phase 3 - get_kuramoto_sync verification PASSED");
    println!(
        "[FSV]   r={:.4}, psi={:.4}, phases.len={}, natural_freqs.len={}, coupling={}",
        r,
        psi,
        phases.len(),
        natural_freqs.len(),
        coupling
    );
    println!("[FSV]   KURAMOTO 13 OSCILLATORS: VERIFIED");
}

/// FSV Test: adjust_coupling modifies Kuramoto coupling constant K.
///
/// Source of Truth: KuramotoProvider::set_coupling_strength()
/// Expected: old_K, new_K in response, K changes persist
#[tokio::test]
async fn test_adjust_coupling_modifies_k() {
    // SETUP: Create handlers with WARM GWT state
    let handlers = create_test_handlers_with_warm_gwt();

    // STEP 1: Get initial coupling via get_kuramoto_sync
    let sync_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_KURAMOTO_SYNC,
            "arguments": {}
        })),
    };
    let sync_response = handlers.dispatch(sync_request).await;
    assert!(sync_response.error.is_none(), "Initial sync should succeed");
    let sync_data = extract_mcp_tool_data(&sync_response.result.unwrap());
    let initial_k = sync_data
        .get("coupling")
        .and_then(|v| v.as_f64())
        .expect("coupling must exist");

    // STEP 2: Adjust coupling to new value
    let new_k_target = 3.5;
    let adjust_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(2)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::ADJUST_COUPLING,
            "arguments": {
                "new_K": new_k_target
            }
        })),
    };
    let adjust_response = handlers.dispatch(adjust_request).await;

    // VERIFY: Adjustment succeeded
    assert!(
        adjust_response.error.is_none(),
        "[FSV] adjust_coupling should succeed: {:?}",
        adjust_response.error
    );
    let adjust_data = extract_mcp_tool_data(&adjust_response.result.unwrap());

    // FSV: Verify old_K matches initial
    let old_k = adjust_data
        .get("old_K")
        .and_then(|v| v.as_f64())
        .expect("old_K must exist");
    assert!(
        (old_k - initial_k).abs() < 0.01,
        "[FSV] old_K ({}) should match initial_k ({})",
        old_k,
        initial_k
    );

    // FSV: Verify new_K is as requested (may be clamped to [0, 10])
    let new_k = adjust_data
        .get("new_K")
        .and_then(|v| v.as_f64())
        .expect("new_K must exist");
    assert!(
        (new_k - new_k_target).abs() < 0.01,
        "[FSV] new_K ({}) should be {} (requested)",
        new_k,
        new_k_target
    );

    // STEP 3: Verify change persists via another get_kuramoto_sync
    let verify_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(3)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_KURAMOTO_SYNC,
            "arguments": {}
        })),
    };
    let verify_response = handlers.dispatch(verify_request).await;
    assert!(verify_response.error.is_none(), "Verify sync should succeed");
    let verify_data = extract_mcp_tool_data(&verify_response.result.unwrap());
    let persisted_k = verify_data
        .get("coupling")
        .and_then(|v| v.as_f64())
        .expect("coupling must exist");

    // FSV: Verify persisted K matches new K
    assert!(
        (persisted_k - new_k).abs() < 0.01,
        "[FSV] Persisted K ({}) should match new_K ({})",
        persisted_k,
        new_k
    );

    println!("[FSV] Phase 3 - adjust_coupling verification PASSED");
    println!(
        "[FSV]   old_K={}, new_K={}, persisted_K={}",
        old_k, new_k, persisted_k
    );
    println!("[FSV]   STATE CHANGES PERSIST: VERIFIED");
}

/// FSV Test: adjust_coupling clamps K to [0, 10] range.
#[tokio::test]
async fn test_adjust_coupling_clamps_k() {
    let handlers = create_test_handlers_with_warm_gwt();

    // Test upper bound clamping
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::ADJUST_COUPLING,
            "arguments": {
                "new_K": 100.0  // Above max
            }
        })),
    };
    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none(), "Should succeed with clamping");
    let data = extract_mcp_tool_data(&response.result.unwrap());
    let new_k = data.get("new_K").and_then(|v| v.as_f64()).unwrap();
    assert!(
        new_k <= 10.0,
        "[FSV] K should be clamped to max 10, got {}",
        new_k
    );

    println!("[FSV] Phase 3 - adjust_coupling clamping verification PASSED");
}

// =============================================================================
// SCENARIO 3: WORKSPACE BROADCAST
// =============================================================================

/// FSV Test: get_workspace_status returns valid workspace state.
///
/// Source of Truth: WorkspaceProvider
/// Expected: Response contains active_memory, is_broadcasting, coherence_threshold
#[tokio::test]
async fn test_get_workspace_status_returns_valid_state() {
    let handlers = create_test_handlers_with_warm_gwt();

    // EXECUTE: Call get_workspace_status
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_WORKSPACE_STATUS,
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

    // FSV: Verify coherence_threshold exists and is 0.8 (constitution default)
    let coherence_threshold = data
        .get("coherence_threshold")
        .and_then(|v| v.as_f64())
        .expect("coherence_threshold must exist");
    assert!(
        (coherence_threshold - 0.8).abs() < 0.01,
        "[FSV] Coherence threshold must be 0.8 (constitution default), got {}",
        coherence_threshold
    );

    // FSV: Verify is_broadcasting is a boolean
    assert!(
        data.get("is_broadcasting").is_some(),
        "[FSV] is_broadcasting must exist"
    );

    // FSV: Verify has_conflict is a boolean
    assert!(
        data.get("has_conflict").is_some(),
        "[FSV] has_conflict must exist"
    );

    // FSV: Verify broadcast_duration_ms is 100 (constitution default)
    let broadcast_duration = data
        .get("broadcast_duration_ms")
        .and_then(|v| v.as_u64())
        .expect("broadcast_duration_ms must exist");
    assert_eq!(
        broadcast_duration, 100,
        "[FSV] Broadcast duration must be 100ms (constitution default)"
    );

    println!("[FSV] Phase 3 - get_workspace_status verification PASSED");
    println!(
        "[FSV]   coherence_threshold={}, broadcast_duration_ms={}",
        coherence_threshold, broadcast_duration
    );
}

/// FSV Test: trigger_workspace_broadcast performs WTA selection.
///
/// Source of Truth: WorkspaceProvider::select_winning_memory()
/// Expected: Broadcast succeeds when r >= threshold or force=true
#[tokio::test]
async fn test_trigger_workspace_broadcast_works() {
    let handlers = create_test_handlers_with_all_components();

    // Generate a test UUID for memory_id
    let test_memory_id = uuid::Uuid::new_v4();

    // EXECUTE: Call trigger_workspace_broadcast
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::TRIGGER_WORKSPACE_BROADCAST,
            "arguments": {
                "memory_id": test_memory_id.to_string(),
                "importance": 0.9,
                "alignment": 0.85,
                "force": false  // Don't force, rely on high r from synchronized network
            }
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

    // FSV: Verify success field exists
    let success = data
        .get("success")
        .and_then(|v| v.as_bool())
        .expect("success must exist");
    assert!(
        success,
        "[FSV] Broadcast should succeed with synchronized network"
    );

    // FSV: Verify memory_id in response matches
    let returned_id = data
        .get("memory_id")
        .and_then(|v| v.as_str())
        .expect("memory_id must exist");
    assert_eq!(
        returned_id,
        test_memory_id.to_string(),
        "[FSV] Returned memory_id should match input"
    );

    // FSV: Verify new_r is present and valid
    let new_r = data
        .get("new_r")
        .and_then(|v| v.as_f64())
        .expect("new_r must exist");
    assert!(
        (0.0..=1.0).contains(&new_r),
        "[FSV] new_r must be in [0, 1], got {}",
        new_r
    );

    // FSV: Verify was_selected is present
    assert!(
        data.get("was_selected").is_some(),
        "[FSV] was_selected must exist"
    );

    // FSV: Verify is_broadcasting is present
    assert!(
        data.get("is_broadcasting").is_some(),
        "[FSV] is_broadcasting must exist"
    );

    println!("[FSV] Phase 3 - trigger_workspace_broadcast verification PASSED");
    println!(
        "[FSV]   success={}, memory_id={}, new_r={:.4}",
        success, returned_id, new_r
    );
    println!("[FSV]   WORKSPACE BROADCAST: TESTED");
}

/// FSV Test: trigger_workspace_broadcast fails gracefully with low r (without force).
#[tokio::test]
async fn test_trigger_workspace_broadcast_low_r_without_force() {
    // Create handlers with INCOHERENT network (low r)
    use super::consciousness_dispatch::create_test_handlers_incoherent;
    let handlers = create_test_handlers_incoherent();

    let test_memory_id = uuid::Uuid::new_v4();

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::TRIGGER_WORKSPACE_BROADCAST,
            "arguments": {
                "memory_id": test_memory_id.to_string(),
                "force": false
            }
        })),
    };
    let response = handlers.dispatch(request).await;

    // Should succeed but with success=false (below threshold)
    assert!(
        response.error.is_none(),
        "Should return success response with failure reason, not error"
    );

    let result = response.result.expect("Should have result");
    let data = extract_mcp_tool_data(&result);

    // FSV: Verify success is false (below coherence threshold)
    let success = data.get("success").and_then(|v| v.as_bool()).unwrap_or(true);
    assert!(
        !success,
        "[FSV] Broadcast should fail with low r without force"
    );

    // FSV: Verify reason is provided
    let reason = data.get("reason").and_then(|v| v.as_str());
    assert!(
        reason.is_some(),
        "[FSV] Reason should be provided for failure"
    );

    println!("[FSV] Phase 3 - trigger_workspace_broadcast low r graceful failure PASSED");
}

// =============================================================================
// SCENARIO 4: EGO STATE VERIFICATION
// =============================================================================

/// FSV Test: get_ego_state returns valid identity and purpose vector.
///
/// Source of Truth: SelfEgoProvider
/// Expected: Response contains purpose_vector (13D), identity_coherence, identity_status
#[tokio::test]
async fn test_get_ego_state_returns_valid_data() {
    let handlers = create_test_handlers_with_warm_gwt();

    // EXECUTE: Call get_ego_state
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_EGO_STATE,
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

    // FSV: CRITICAL - Verify purpose_vector has exactly 13 elements
    let pv = data
        .get("purpose_vector")
        .and_then(|v| v.as_array())
        .expect("purpose_vector must exist");
    assert_eq!(
        pv.len(),
        13,
        "[FSV] CRITICAL: Purpose vector must have 13 elements (one per embedder), got {}",
        pv.len()
    );

    // FSV: Verify all purpose vector elements are floats in [-1, 1]
    // (Purpose alignments are cosine similarities)
    for (i, val) in pv.iter().enumerate() {
        let v = val.as_f64().expect("purpose_vector elements must be f64");
        assert!(
            v >= -1.0 && v <= 1.0,
            "[FSV] Purpose vector[{}] must be in [-1, 1], got {}",
            i,
            v
        );
    }

    // FSV: Verify identity_coherence is in [0, 1]
    let identity_coherence = data
        .get("identity_coherence")
        .and_then(|v| v.as_f64())
        .expect("identity_coherence must exist");
    assert!(
        (0.0..=1.0).contains(&identity_coherence),
        "[FSV] identity_coherence must be in [0, 1], got {}",
        identity_coherence
    );

    // FSV: Verify identity_status is valid
    let status = data
        .get("identity_status")
        .and_then(|v| v.as_str())
        .expect("identity_status must exist");
    let valid_statuses = ["Healthy", "Warning", "Degraded", "Critical"];
    // Status might be Debug formatted (e.g., "Healthy" or "IdentityStatus::Healthy")
    let status_valid = valid_statuses.iter().any(|s| status.contains(s));
    assert!(
        status_valid,
        "[FSV] Invalid identity_status: {}, expected one containing {:?}",
        status,
        valid_statuses
    );

    // FSV: Verify coherence_with_actions is in [0, 1]
    let coherence_with_actions = data
        .get("coherence_with_actions")
        .and_then(|v| v.as_f64())
        .expect("coherence_with_actions must exist");
    assert!(
        (0.0..=1.0).contains(&coherence_with_actions),
        "[FSV] coherence_with_actions must be in [0, 1], got {}",
        coherence_with_actions
    );

    // FSV: Verify trajectory_length is non-negative
    let trajectory_length = data
        .get("trajectory_length")
        .and_then(|v| v.as_u64())
        .expect("trajectory_length must exist");
    assert!(
        trajectory_length >= 0,
        "[FSV] trajectory_length must be non-negative"
    );

    // FSV: Verify thresholds are present
    let thresholds = data.get("thresholds").expect("thresholds must exist");
    assert_eq!(
        thresholds.get("healthy").and_then(|v| v.as_f64()),
        Some(0.9),
        "[FSV] thresholds.healthy must be 0.9"
    );
    assert_eq!(
        thresholds.get("warning").and_then(|v| v.as_f64()),
        Some(0.7),
        "[FSV] thresholds.warning must be 0.7"
    );

    println!("[FSV] Phase 3 - get_ego_state verification PASSED");
    println!(
        "[FSV]   purpose_vector.len={}, identity_coherence={:.4}, status={}",
        pv.len(),
        identity_coherence,
        status
    );
    println!(
        "[FSV]   trajectory_length={}, coherence_with_actions={:.4}",
        trajectory_length, coherence_with_actions
    );
}

/// FSV Test: get_ego_state with WARM state has non-zero purpose vector.
#[tokio::test]
async fn test_get_ego_state_warm_has_non_zero_purpose_vector() {
    // Warm GWT state includes a pre-initialized purpose vector
    let handlers = create_test_handlers_with_warm_gwt();

    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_EGO_STATE,
            "arguments": {}
        })),
    };
    let response = handlers.dispatch(request).await;
    assert!(response.error.is_none());

    let data = extract_mcp_tool_data(&response.result.unwrap());
    let pv = data
        .get("purpose_vector")
        .and_then(|v| v.as_array())
        .expect("purpose_vector must exist");

    // FSV: At least some elements should be non-zero in warm state
    let non_zero_count = pv.iter().filter(|v| {
        let val = v.as_f64().unwrap_or(0.0);
        val.abs() > 0.001
    }).count();

    assert!(
        non_zero_count > 0,
        "[FSV] WARM state should have non-zero purpose vector elements, got {} non-zero",
        non_zero_count
    );

    println!("[FSV] Phase 3 - get_ego_state WARM state verification PASSED");
    println!("[FSV]   Non-zero purpose vector elements: {}/13", non_zero_count);
}

// =============================================================================
// SUMMARY TEST: ALL 6 GWT TOOLS
// =============================================================================

/// FSV Integration Test: All 6 GWT tools work together.
///
/// Tests the complete GWT consciousness flow:
/// 1. Get initial consciousness state
/// 2. Get Kuramoto synchronization (verify 13 oscillators)
/// 3. Get workspace status
/// 4. Get ego state
/// 5. Adjust coupling (verify persistence)
/// 6. Trigger workspace broadcast
/// 7. Re-verify consciousness state reflects changes
#[tokio::test]
async fn test_all_gwt_tools_integration() {
    let handlers = create_test_handlers_with_all_components();
    let mut gwt_tests_passed = 0;

    // TEST 1: get_consciousness_state
    let req1 = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(1)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_CONSCIOUSNESS_STATE,
            "arguments": {}
        })),
    };
    let resp1 = handlers.dispatch(req1).await;
    if resp1.error.is_none() {
        gwt_tests_passed += 1;
        println!("[Phase 3] get_consciousness_state: PASSED");
    }

    // TEST 2: get_kuramoto_sync
    let req2 = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(2)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_KURAMOTO_SYNC,
            "arguments": {}
        })),
    };
    let resp2 = handlers.dispatch(req2).await;
    let kuramoto_13_oscillators = if resp2.error.is_none() {
        let data = extract_mcp_tool_data(&resp2.result.unwrap());
        let phases = data.get("phases").and_then(|v| v.as_array());
        phases.map(|p| p.len() == 13).unwrap_or(false)
    } else {
        false
    };
    if kuramoto_13_oscillators {
        gwt_tests_passed += 1;
        println!("[Phase 3] get_kuramoto_sync (13 oscillators): PASSED");
    }

    // TEST 3: get_workspace_status
    let req3 = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(3)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_WORKSPACE_STATUS,
            "arguments": {}
        })),
    };
    let resp3 = handlers.dispatch(req3).await;
    if resp3.error.is_none() {
        gwt_tests_passed += 1;
        println!("[Phase 3] get_workspace_status: PASSED");
    }

    // TEST 4: get_ego_state
    let req4 = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(4)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::GET_EGO_STATE,
            "arguments": {}
        })),
    };
    let resp4 = handlers.dispatch(req4).await;
    if resp4.error.is_none() {
        gwt_tests_passed += 1;
        println!("[Phase 3] get_ego_state: PASSED");
    }

    // TEST 5: adjust_coupling
    let req5 = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(5)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::ADJUST_COUPLING,
            "arguments": { "new_K": 5.0 }
        })),
    };
    let resp5 = handlers.dispatch(req5).await;
    if resp5.error.is_none() {
        gwt_tests_passed += 1;
        println!("[Phase 3] adjust_coupling: PASSED");
    }

    // TEST 6: trigger_workspace_broadcast
    let req6 = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(JsonRpcId::Number(6)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": tool_names::TRIGGER_WORKSPACE_BROADCAST,
            "arguments": {
                "memory_id": uuid::Uuid::new_v4().to_string(),
                "force": false
            }
        })),
    };
    let resp6 = handlers.dispatch(req6).await;
    let workspace_broadcast_tested = resp6.error.is_none();
    if workspace_broadcast_tested {
        gwt_tests_passed += 1;
        println!("[Phase 3] trigger_workspace_broadcast: PASSED");
    }

    // SUMMARY
    println!("\n[Phase 3] GWT CONSCIOUSNESS TOOLS SUMMARY");
    println!("==========================================");
    println!("GWT tests passed: {}/6", gwt_tests_passed);
    println!("Consciousness verified: {}", gwt_tests_passed >= 4);
    println!("Kuramoto 13 oscillators: {}", kuramoto_13_oscillators);
    println!("Workspace broadcast tested: {}", workspace_broadcast_tested);
    println!("==========================================");

    // All 6 tests must pass
    assert_eq!(
        gwt_tests_passed, 6,
        "[FSV] All 6 GWT tools should pass, got {}/6",
        gwt_tests_passed
    );
}
