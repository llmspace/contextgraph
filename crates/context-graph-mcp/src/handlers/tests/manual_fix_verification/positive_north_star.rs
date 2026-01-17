//! Positive Cases: Handlers with North Star configured
//!
//! Tests verifying correct behavior when North Star IS properly configured.

use crate::handlers::tests::{create_test_handlers, make_request};
use crate::protocol::JsonRpcId;
use serde_json::json;

/// Positive case: get_autonomous_status WITH North Star configured.
#[tokio::test]
async fn test_autonomous_status_with_north_star() {
    println!("\n{}", "=".repeat(60));
    println!("POSITIVE VERIFICATION: get_autonomous_status with North Star");
    println!("{}", "=".repeat(60));

    // Use handlers WITH North Star configured
    let handlers = create_test_handlers();
    println!("[BEFORE] Handlers created WITH North Star configured");

    let request = make_request(
        "tools/call",
        Some(JsonRpcId::Number(1)),
        Some(json!({
            "name": "get_autonomous_status",
            "arguments": {
                "include_metrics": true
            }
        })),
    );

    println!("[EXECUTE] Calling get_autonomous_status with North Star...");
    let response = handlers.dispatch(request).await;

    assert!(response.error.is_none(), "Should not have protocol error");
    let result = response.result.expect("Must have result");

    let is_error = result
        .get("isError")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    assert!(!is_error, "Should not be error with North Star");
    println!("[VERIFY] Not an error response - PASS");

    if let Some(content) = result.get("content").and_then(|v| v.as_array()) {
        if let Some(first) = content.first() {
            if let Some(text) = first.get("text").and_then(|v| v.as_str()) {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
                    // VERIFY: strategic_goals.configured should be true
                    if let Some(ns) = parsed.get("strategic_goals") {
                        let configured = ns
                            .get("configured")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        assert!(
                            configured,
                            "[FAIL] strategic_goals.configured should be true"
                        );
                        println!("[VERIFY] strategic_goals.configured = true - PASS");

                        if let Some(goal_id) = ns.get("goal_id").and_then(|v| v.as_str()) {
                            println!("[EVIDENCE] North Star goal_id: {}", goal_id);
                        }
                    }

                    // VERIFY: overall_health should NOT be not_configured
                    if let Some(health) = parsed.get("overall_health") {
                        if let Some(status) = health.get("status").and_then(|v| v.as_str()) {
                            assert_ne!(status, "not_configured", "Should have valid health status");
                            println!("[VERIFY] overall_health.status = \"{}\" - PASS", status);
                        }
                        if let Some(score) = health.get("score").and_then(|v| v.as_f64()) {
                            println!("[EVIDENCE] Health score: {}", score);
                        }
                    }
                }
            }
        }
    }

    println!("\n[POSITIVE get_autonomous_status VERIFICATION COMPLETE]\n");
}

// TASK-P0-001: Removed test_bootstrap_with_existing_north_star
// The auto_bootstrap_north_star tool was removed per ARCH-03 (goals emerge from topic clustering).
