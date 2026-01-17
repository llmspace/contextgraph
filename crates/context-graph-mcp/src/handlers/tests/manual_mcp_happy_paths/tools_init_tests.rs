//! Tools and initialization handler happy path tests
//!
//! Tests for tools/list, initialize

use serde_json::json;

use super::common::{create_test_handlers_with_rocksdb_store_access, make_request};

/// Test 14: tools/list - List available tools
#[tokio::test]
async fn test_14_tools_list() {
    println!("\n========================================================================================================");
    println!("TEST 14: tools/list");
    println!("========================================================================================================");

    let (handlers, _store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    let request = make_request("tools/list", 1, None);
    let response = handlers.dispatch(request).await;

    println!(
        "Response (truncated): {}",
        serde_json::to_string_pretty(&response)
            .unwrap()
            .chars()
            .take(2000)
            .collect::<String>()
    );

    assert!(
        response.error.is_none(),
        "Should not have error: {:?}",
        response.error
    );
    let result = response.result.expect("Should have result");

    println!("\n[VERIFICATION]");
    if let Some(tools) = result.get("tools") {
        if let Some(arr) = tools.as_array() {
            println!("  Total tools: {}", arr.len());
            println!("  First 5 tools:");
            for tool in arr.iter().take(5) {
                if let Some(name) = tool.get("name") {
                    println!("    - {}", name);
                }
            }
        }
    }

    println!("\n[PASSED] tools/list works correctly");
}

/// Test 15: initialize - Initialize MCP connection
#[tokio::test]
async fn test_15_initialize() {
    println!("\n========================================================================================================");
    println!("TEST 15: initialize");
    println!("========================================================================================================");

    let (handlers, _store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    let init_params = json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {
            "name": "test-client",
            "version": "1.0.0"
        }
    });
    let request = make_request("initialize", 1, Some(init_params));
    let response = handlers.dispatch(request).await;

    println!(
        "Response: {}",
        serde_json::to_string_pretty(&response).unwrap()
    );

    assert!(
        response.error.is_none(),
        "Should not have error: {:?}",
        response.error
    );
    let result = response.result.expect("Should have result");

    println!("\n[VERIFICATION]");
    println!(
        "  protocolVersion: {}",
        result.get("protocolVersion").unwrap_or(&json!("?"))
    );
    if let Some(server_info) = result.get("serverInfo") {
        println!(
            "  serverInfo.name: {}",
            server_info.get("name").unwrap_or(&json!("?"))
        );
        println!(
            "  serverInfo.version: {}",
            server_info.get("version").unwrap_or(&json!("?"))
        );
    }

    println!("\n[PASSED] initialize works correctly");
}
