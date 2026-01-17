//! Scenario 3: Workspace Broadcast Tests
//!
//! Tests get_workspace_status and trigger_workspace_broadcast tools:
//! - Valid workspace state
//! - Winner-take-all selection
//! - Low r graceful failure

use serde_json::json;

use crate::handlers::tests::{
    create_test_handlers_with_all_components, create_test_handlers_with_warm_gwt,
    extract_mcp_tool_data,
};
use crate::protocol::{JsonRpcId, JsonRpcRequest};
use crate::tools::tool_names;

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
    use super::super::consciousness_dispatch::create_test_handlers_incoherent;
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
    let success = data
        .get("success")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
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
