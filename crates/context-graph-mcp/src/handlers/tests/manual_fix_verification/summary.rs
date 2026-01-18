//! Summary Test: Collect All Evidence
//!
//! Runs all verifications and prints consolidated evidence.

use crate::handlers::tests::{create_test_handlers, create_test_handlers_no_goals, make_request};
use crate::protocol::JsonRpcId;
use serde_json::json;

/// Summary test that runs all verifications and prints consolidated evidence.
#[tokio::test]
async fn test_all_fixes_summary() {
    println!("\n");
    println!("{}", "#".repeat(70));
    println!("#  MANUAL FIX VERIFICATION SUMMARY");
    println!("#  Tests for Issues 1-3 in context-graph MCP server");
    println!("{}", "#".repeat(70));

    // Run a quick verification of each fix
    let handlers = create_test_handlers();
    let handlers_no_ns = create_test_handlers_no_goals();

    // Issue 1: search_teleological query_content
    let req1 = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "search_teleological",
            "arguments": {"query_content": "test"}
        })),
    );
    let res1 = handlers.dispatch(req1).await;
    let issue1_pass = res1.error.is_none();

    // Issue 3a: get_autonomous_status without strategic goal
    let req3a = make_request(
        "tools/call",
        Some(JsonRpcId::Number(2)),
        Some(json!({
            "name": "get_autonomous_status",
            "arguments": {}
        })),
    );
    let res3a = handlers_no_ns.dispatch(req3a).await;
    let issue3a_pass = res3a.error.is_none()
        && res3a
            .result
            .as_ref()
            .map(|r| !r.get("isError").and_then(|v| v.as_bool()).unwrap_or(true))
            .unwrap_or(false);

    // TASK-P0-001: Issue 3b (auto_bootstrap) test removed
    // The auto_bootstrap tool was removed per ARCH-03 (goals emerge from topic clustering)
    let issue3b_pass = true; // Removed - no longer testable

    // Edge case: empty query_content FAIL FAST
    let req_edge = make_request(
        "tools/call",
        Some(JsonRpcId::Number(4)),
        Some(json!({
            "name": "search_teleological",
            "arguments": {"query_content": ""}
        })),
    );
    let res_edge = handlers.dispatch(req_edge).await;
    let edge_pass = res_edge.error.is_none()
        && res_edge
            .result
            .as_ref()
            .map(|r| r.get("isError").and_then(|v| v.as_bool()).unwrap_or(false))
            .unwrap_or(false);

    println!("\n{}", "=".repeat(70));
    println!("VERIFICATION RESULTS:");
    println!("{}", "=".repeat(70));
    println!(
        "Issue 1 - search_teleological query_content: {}",
        if issue1_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "Issue 3a - get_autonomous_status without strategic goal: {}",
        if issue3a_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "Issue 3b - auto_bootstrap (REMOVED per TASK-P0-001): {}",
        if issue3b_pass { "SKIPPED" } else { "SKIPPED" }
    );
    println!(
        "Edge case - FAIL FAST on empty query_content: {}",
        if edge_pass { "PASS" } else { "FAIL" }
    );
    println!("{}", "=".repeat(70));

    let all_pass = issue1_pass && issue3a_pass && issue3b_pass && edge_pass;
    println!(
        "\nOVERALL: {}",
        if all_pass {
            "ALL TESTS PASSED"
        } else {
            "SOME TESTS FAILED"
        }
    );
    println!("\n");

    assert!(all_pass, "Not all verification tests passed");
}
