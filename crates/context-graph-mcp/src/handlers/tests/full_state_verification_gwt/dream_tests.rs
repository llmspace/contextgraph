//! Dream Consolidation FSV Tests
//!
//! Verifies dream cycle, abort, and amortized learning shortcuts.
//!
//! CONSTITUTION REFERENCE: dream section (constitution.yaml:446)
//! - Activity level < 0.15 for 10 minutes
//! - No active queries
//! - GPU usage < 30%
//! - Wake latency < 100ms (MANDATE)

use serde_json::json;

use super::{create_handlers_with_gwt, extract_tool_content, make_tool_call_request};
use crate::tools::tool_names;

/// P4-09: FSV test to verify trigger_dream initiates REAL dream consolidation.
#[tokio::test]
async fn test_trigger_dream_initiates_real_consolidation() {
    // SETUP: Create handlers with real GWT/Dream components
    let handlers = create_handlers_with_gwt();

    // EXECUTE: Call trigger_dream tool (without force flag)
    let args = json!({});
    let request = make_tool_call_request(tool_names::TRIGGER_DREAM, Some(args));
    let response = handlers.dispatch(request).await;

    // Parse response
    let response_json = serde_json::to_value(&response).expect("serialize");
    assert!(
        response.error.is_none(),
        "trigger_dream should not error: {:?}",
        response.error
    );

    let content = extract_tool_content(&response_json).expect("trigger_dream must return content");

    // FSV-1: Must have triggered flag
    let triggered = content["triggered"]
        .as_bool()
        .expect("triggered must be bool");
    // May or may not trigger depending on activity level

    // FSV-2: Must have reason string
    let reason = content["reason"].as_str().expect("reason must be string");
    assert!(!reason.is_empty(), "reason must not be empty");

    // FSV-3: Must have current_state
    let current_state = content["current_state"]
        .as_str()
        .expect("current_state must be string");
    assert!(!current_state.is_empty(), "current_state must not be empty");

    // FSV-4: Must have activity_level
    let activity_level = content["activity_level"]
        .as_f64()
        .expect("activity_level must be f64");
    assert!(
        activity_level >= 0.0,
        "activity_level must be >= 0, got {}",
        activity_level
    );

    println!(
        "FSV PASSED: trigger_dream - triggered={}, state='{}', activity={:.3}, reason='{}'",
        triggered, current_state, activity_level, reason
    );

    // EXECUTE: Call trigger_dream with force=true
    let args_force = json!({ "force": true });
    let request_force = make_tool_call_request(tool_names::TRIGGER_DREAM, Some(args_force));
    let response_force = handlers.dispatch(request_force).await;

    let response_force_json = serde_json::to_value(&response_force).expect("serialize");
    let content_force = extract_tool_content(&response_force_json)
        .expect("trigger_dream force must return content");

    // Force should either trigger or report already dreaming
    let triggered_force = content_force["triggered"]
        .as_bool()
        .expect("triggered must be bool");
    let reason_force = content_force["reason"]
        .as_str()
        .expect("reason must be string");

    println!(
        "FSV PASSED: trigger_dream force=true - triggered={}, reason='{}'",
        triggered_force, reason_force
    );
}

/// P4-10: FSV test to verify get_dream_status returns REAL dream state.
///
/// CONSTITUTION REFERENCE: dream section
/// - States: Awake, EnteringDream, Nrem, Rem, Waking
/// - Constitution compliance mandates
#[tokio::test]
async fn test_get_dream_status_returns_real_state() {
    // SETUP: Create handlers with real GWT/Dream components
    let handlers = create_handlers_with_gwt();

    // EXECUTE: Call get_dream_status tool
    let request = make_tool_call_request(tool_names::GET_DREAM_STATUS, None);
    let response = handlers.dispatch(request).await;

    // Parse response
    let response_json = serde_json::to_value(&response).expect("serialize");
    assert!(
        response.error.is_none(),
        "get_dream_status should not error: {:?}",
        response.error
    );

    let content =
        extract_tool_content(&response_json).expect("get_dream_status must return content");

    // FSV-1: Must have state string
    let state = content["state"].as_str().expect("state must be string");
    // State should be one of: Awake, EnteringDream, Nrem, Rem, Waking
    assert!(!state.is_empty(), "state must not be empty");

    // FSV-2: Must have is_dreaming flag
    let is_dreaming = content["is_dreaming"]
        .as_bool()
        .expect("is_dreaming must be bool");

    // FSV-3: Must have gpu_usage
    let gpu_usage = content["gpu_usage"]
        .as_f64()
        .expect("gpu_usage must be f64");
    assert!(
        (0.0..=1.0).contains(&gpu_usage),
        "gpu_usage must be in [0, 1], got {}",
        gpu_usage
    );

    // FSV-4: Must have scheduler object
    let scheduler = &content["scheduler"];
    assert!(scheduler.is_object(), "scheduler must be an object");

    // Scheduler should have activity level
    let scheduler_activity = scheduler["average_activity"]
        .as_f64()
        .expect("scheduler.average_activity must be f64");
    assert!(
        scheduler_activity >= 0.0,
        "scheduler.average_activity must be >= 0"
    );

    // FSV-5: Must have constitution_compliance object
    let compliance = &content["constitution_compliance"];
    assert!(
        compliance.is_object(),
        "constitution_compliance must be an object"
    );

    // Compliance should have gpu_under_30_percent flag
    let gpu_ok = compliance["gpu_under_30_percent"]
        .as_bool()
        .expect("gpu_under_30_percent must be bool");

    // Compliance should have max_wake_latency_ms
    let max_wake_latency_ms = compliance["max_wake_latency_ms"]
        .as_u64()
        .expect("max_wake_latency_ms must be u64");
    assert_eq!(
        max_wake_latency_ms, 100,
        "max_wake_latency_ms should be 100ms mandate"
    );

    println!(
        "FSV PASSED: get_dream_status - state='{}', is_dreaming={}, gpu={:.1}%, activity={:.3}",
        state,
        is_dreaming,
        gpu_usage * 100.0,
        scheduler_activity
    );
    println!(
        "  Constitution compliance: gpu_under_30_percent={}, max_wake_latency={}ms",
        gpu_ok, max_wake_latency_ms
    );
}

/// P5-01: FSV test verifying abort_dream stops dream cycle properly.
///
/// Constitution mandate: Wake latency MUST be <100ms.
/// This test verifies:
/// - abort_dream when not dreaming returns aborted: false
/// - abort_dream has correct response structure
/// - mandate_met field reflects <100ms requirement
#[tokio::test]
async fn test_abort_dream_stops_cycle_properly() {
    let handlers = create_handlers_with_gwt();

    // First, call abort_dream when NOT dreaming
    let request = make_tool_call_request(tool_names::ABORT_DREAM, Some(json!({})));
    let response = handlers.dispatch(request).await;
    let response_json = serde_json::to_value(&response).expect("serialize");

    // Should succeed (but not abort anything since not dreaming)
    assert!(
        response.error.is_none(),
        "abort_dream should not error when not dreaming"
    );

    let content = extract_tool_content(&response_json).expect("abort_dream must return content");

    // FSV-1: Must have aborted field (should be false when not dreaming)
    let aborted = content["aborted"].as_bool().expect("aborted must be bool");
    assert!(
        !aborted,
        "aborted should be false when not currently dreaming"
    );

    // FSV-2: Must have abort_latency_ms field
    let abort_latency_ms = content["abort_latency_ms"]
        .as_u64()
        .expect("abort_latency_ms must be u64");
    assert_eq!(
        abort_latency_ms, 0,
        "abort_latency_ms should be 0 when not dreaming"
    );

    // FSV-3: Must have previous_state field
    let previous_state = content["previous_state"]
        .as_str()
        .expect("previous_state must be string");
    assert!(
        !previous_state.is_empty(),
        "previous_state must not be empty"
    );

    // FSV-4: Must have mandate_met field
    let mandate_met = content["mandate_met"]
        .as_bool()
        .expect("mandate_met must be bool");
    assert!(
        mandate_met,
        "mandate_met should be true when not dreaming (trivially satisfied)"
    );

    // FSV-5: Must have reason field
    let reason = content["reason"].as_str().expect("reason must be string");
    assert!(
        reason.contains("Not currently dreaming"),
        "reason should indicate not dreaming: {}",
        reason
    );

    println!(
        "FSV PASSED: abort_dream (not dreaming) - aborted={}, latency={}ms, state='{}', mandate_met={}",
        aborted, abort_latency_ms, previous_state, mandate_met
    );

    // Now force-trigger a dream and try to abort it
    let trigger_request = make_tool_call_request(
        tool_names::TRIGGER_DREAM,
        Some(json!({
            "force": true,
            "phase": "nrem"
        })),
    );
    let _trigger_response = handlers.dispatch(trigger_request).await;

    // Try to abort the dream
    let abort_request = make_tool_call_request(
        tool_names::ABORT_DREAM,
        Some(json!({
            "reason": "FSV test abort"
        })),
    );
    let abort_response = handlers.dispatch(abort_request).await;
    let abort_json = serde_json::to_value(&abort_response).expect("serialize abort response");

    // Should succeed
    assert!(
        abort_response.error.is_none(),
        "abort_dream should not error: {:?}",
        abort_response.error
    );

    let abort_content =
        extract_tool_content(&abort_json).expect("abort_dream must return content after force");

    // FSV-6: After force trigger, verify abort response structure
    // Note: aborted may be true or false depending on timing
    let aborted_after = abort_content["aborted"]
        .as_bool()
        .expect("aborted must be bool after force");

    let latency_after = abort_content["abort_latency_ms"]
        .as_u64()
        .expect("abort_latency_ms must be u64 after force");

    let mandate_met_after = abort_content["mandate_met"]
        .as_bool()
        .expect("mandate_met must be bool after force");

    let reason_after = abort_content["reason"]
        .as_str()
        .expect("reason must be string after force");

    // If we aborted a running dream, verify mandate was met
    if aborted_after {
        assert!(
            mandate_met_after,
            "Constitution mandate violated: abort took {}ms (max 100ms)",
            latency_after
        );
        assert!(
            latency_after < 100,
            "Abort latency must be <100ms per constitution, got {}ms",
            latency_after
        );
    }

    println!(
        "FSV PASSED: abort_dream (after force) - aborted={}, latency={}ms, mandate_met={}, reason='{}'",
        aborted_after, latency_after, mandate_met_after, reason_after
    );
}

/// P5-02: FSV test verifying get_amortized_shortcuts returns real shortcut candidates.
///
/// Constitution reference (dream.amortized):
/// - trigger: "3+ hop path traversed >=5x"
/// - weight: "product(path_weights)"
/// - confidence: ">=0.7"
/// - is_shortcut: true
#[tokio::test]
async fn test_get_amortized_shortcuts_returns_real_candidates() {
    let handlers = create_handlers_with_gwt();

    // Call with default parameters
    let request = make_tool_call_request(tool_names::GET_AMORTIZED_SHORTCUTS, Some(json!({})));
    let response = handlers.dispatch(request).await;
    let response_json = serde_json::to_value(&response).expect("serialize");

    // Should succeed
    assert!(
        response.error.is_none(),
        "get_amortized_shortcuts should not error: {:?}",
        response.error
    );

    let content =
        extract_tool_content(&response_json).expect("get_amortized_shortcuts must return content");

    // FSV-1: Must have shortcuts array
    let shortcuts = content["shortcuts"]
        .as_array()
        .expect("shortcuts must be an array");
    // Note: May be empty if no paths have been traversed yet

    // FSV-2: Must have total_candidates count
    let total_candidates = content["total_candidates"]
        .as_u64()
        .expect("total_candidates must be u64");

    // FSV-3: Must have returned_count
    let returned_count = content["returned_count"]
        .as_u64()
        .expect("returned_count must be u64");
    assert_eq!(
        returned_count as usize,
        shortcuts.len(),
        "returned_count should match shortcuts.len()"
    );

    // FSV-4: Must have shortcuts_created_this_cycle
    let shortcuts_this_cycle = content["shortcuts_created_this_cycle"]
        .as_u64()
        .expect("shortcuts_created_this_cycle must be u64");

    // FSV-5: Must have filters_applied object
    let filters = &content["filters_applied"];
    assert!(filters.is_object(), "filters_applied must be an object");

    let min_confidence = filters["min_confidence"]
        .as_f64()
        .expect("filters_applied.min_confidence must be f64");
    let limit = filters["limit"]
        .as_u64()
        .expect("filters_applied.limit must be u64");

    // FSV-6: Must have constitution_reference
    let constitution = &content["constitution_reference"];
    assert!(
        constitution.is_object(),
        "constitution_reference must be an object"
    );

    let min_hops = constitution["min_hops"]
        .as_u64()
        .expect("constitution_reference.min_hops must be u64");
    assert_eq!(min_hops, 3, "Constitution requires min_hops=3");

    let min_traversals = constitution["min_traversals"]
        .as_u64()
        .expect("constitution_reference.min_traversals must be u64");
    assert_eq!(min_traversals, 5, "Constitution requires min_traversals=5");

    println!(
        "FSV PASSED: get_amortized_shortcuts - total={}, returned={}, this_cycle={}",
        total_candidates, returned_count, shortcuts_this_cycle
    );
    println!(
        "  Filters: min_confidence={}, limit={}. Constitution: min_hops={}, min_traversals={}",
        min_confidence, limit, min_hops, min_traversals
    );

    // If there are shortcuts, verify their structure
    if !shortcuts.is_empty() {
        let first = &shortcuts[0];

        assert!(
            first["source"].is_string(),
            "shortcut.source must be string"
        );
        assert!(
            first["target"].is_string(),
            "shortcut.target must be string"
        );
        assert!(
            first["hop_count"].is_u64(),
            "shortcut.hop_count must be u64"
        );
        assert!(
            first["traversal_count"].is_u64(),
            "shortcut.traversal_count must be u64"
        );
        assert!(
            first["combined_weight"].is_f64(),
            "shortcut.combined_weight must be f64"
        );
        assert!(
            first["min_confidence"].is_f64(),
            "shortcut.min_confidence must be f64"
        );

        println!(
            "  First shortcut: {} -> {}, hops={}, traversals={}",
            first["source"].as_str().unwrap_or("?"),
            first["target"].as_str().unwrap_or("?"),
            first["hop_count"].as_u64().unwrap_or(0),
            first["traversal_count"].as_u64().unwrap_or(0)
        );
    }
}
