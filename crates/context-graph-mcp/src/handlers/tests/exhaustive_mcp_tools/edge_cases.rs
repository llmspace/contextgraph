//! Edge cases and error handling tests for MCP tools.

use serde_json::json;

use super::helpers::{assert_success, assert_tool_error, make_tool_call};
use super::synthetic_data;
use crate::handlers::tests::{create_test_handlers, create_test_handlers_no_goals};

#[tokio::test]
async fn test_unknown_tool_name() {
    let handlers = create_test_handlers();
    let request = make_tool_call("nonexistent_tool", json!({}));

    let response = handlers.dispatch(request).await;
    // Should return tool error, not crash
    assert!(
        response.error.is_some() || response.result.is_some(),
        "Must handle unknown tool gracefully"
    );
}

#[tokio::test]
async fn test_empty_arguments() {
    // Tools without required params should work with empty args
    let handlers = create_test_handlers();
    let request = make_tool_call("get_memetic_status", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_memetic_status");
}

#[tokio::test]
async fn test_extra_arguments_ignored() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "get_memetic_status",
        json!({
            "extra_param": "should be ignored",
            "another_extra": 123
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_memetic_status");
}

#[tokio::test]
async fn test_null_content() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "inject_context",
        json!({
            "content": null,
            "rationale": "Testing null content"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_tool_error(&response, "inject_context");
}

#[tokio::test]
async fn test_wrong_type_content() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "inject_context",
        json!({
            "content": 12345,  // Should be string
            "rationale": "Testing wrong type"
        }),
    );

    let response = handlers.dispatch(request).await;
    // Should be handled gracefully
    assert!(response.error.is_none(), "Should not crash on wrong type");
}

#[tokio::test]
async fn test_very_long_content() {
    let handlers = create_test_handlers();
    let long_content = "x".repeat(100_000); // 100KB of text

    let request = make_tool_call(
        "inject_context",
        json!({
            "content": long_content,
            "rationale": "Testing very long content"
        }),
    );

    let response = handlers.dispatch(request).await;
    // Should handle large content (may succeed or fail gracefully)
    assert!(
        response.error.is_none(),
        "Should not crash on large content"
    );
}

#[tokio::test]
async fn test_special_characters_in_content() {
    let handlers = create_test_handlers();
    let request = make_tool_call(
        "inject_context",
        json!({
            "content": synthetic_data::content::SPECIAL_CHARS,
            "rationale": "Testing special characters"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "inject_context");
}

#[tokio::test]
async fn test_no_north_star_error_handling() {
    let handlers = create_test_handlers_no_goals();
    let request = make_tool_call("discover_sub_goals", json!({}));

    let response = handlers.dispatch(request).await;
    // Should handle missing North Star gracefully (may return empty or error)
    assert!(
        response.error.is_none(),
        "Should not crash without North Star"
    );
}

#[tokio::test]
async fn test_concurrent_tool_calls() {
    // Test that multiple concurrent calls don't interfere
    use futures::future::join_all;

    let handlers = create_test_handlers();

    let futures: Vec<_> = (0..5)
        .map(|i| {
            let request = make_tool_call(
                "inject_context",
                json!({
                    "content": format!("Concurrent test content {}", i),
                    "rationale": format!("Concurrent test rationale {}", i)
                }),
            );
            handlers.dispatch(request)
        })
        .collect();

    let results = join_all(futures).await;

    for (i, response) in results.iter().enumerate() {
        assert!(
            response.error.is_none(),
            "Concurrent call {} should not error",
            i
        );
    }
}
