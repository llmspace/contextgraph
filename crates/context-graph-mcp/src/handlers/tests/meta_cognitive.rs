//! Meta-Cognitive and System Handler Tests

use serde_json::json;

use crate::protocol::JsonRpcId;

use super::{create_test_handlers, make_request};

// =========================================================================
// get_graph_manifest Tool Tests
// =========================================================================

#[tokio::test]
async fn test_tools_call_get_graph_manifest() {
    let handlers = create_test_handlers();
    let params = json!({
        "name": "get_graph_manifest",
        "arguments": {}
    });
    let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "get_graph_manifest should not return an error"
    );
    let result = response.result.expect("tools/call must return a result");

    // Verify MCP format
    let content = result.get("content").unwrap().as_array().unwrap();
    let text = content[0].get("text").unwrap().as_str().unwrap();
    let parsed_text: serde_json::Value = serde_json::from_str(text).unwrap();

    // Verify expected fields
    assert!(
        parsed_text.get("architecture").is_some(),
        "Response must contain architecture"
    );
    assert!(
        parsed_text.get("layers").is_some(),
        "Response must contain layers"
    );
    assert!(
        parsed_text.get("utl").is_some(),
        "Response must contain utl"
    );
}

#[tokio::test]
async fn test_graph_manifest_layers_structure() {
    let handlers = create_test_handlers();
    let params = json!({
        "name": "get_graph_manifest",
        "arguments": {}
    });
    let request = make_request("tools/call", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("tools/call must return a result");
    let content = result.get("content").unwrap().as_array().unwrap();
    let text = content[0].get("text").unwrap().as_str().unwrap();
    let parsed_text: serde_json::Value = serde_json::from_str(text).unwrap();

    let layers = parsed_text
        .get("layers")
        .expect("Response must contain layers")
        .as_array()
        .expect("layers must be an array");

    // Verify 5-layer bio-nervous architecture
    assert_eq!(layers.len(), 5, "Must have exactly 5 layers");

    for layer in layers {
        assert!(layer.get("name").is_some(), "Layer must have name");
        assert!(
            layer.get("description").is_some(),
            "Layer must have description"
        );
        assert!(layer.get("status").is_some(), "Layer must have status");
    }
}

// =========================================================================
// System Status Tests
// =========================================================================

#[tokio::test]
async fn test_system_status() {
    let handlers = create_test_handlers();
    let request = make_request("system/status", Some(JsonRpcId::Number(1)), None);

    let response = handlers.dispatch(request).await;

    assert!(response.error.is_none(), "system/status should succeed");
    let result = response.result.expect("Should have result");

    assert!(result.get("status").is_some(), "Should have status field");
    assert!(result.get("phase").is_some(), "Should have phase field");
}

// =========================================================================
// System Health Tests
// =========================================================================

#[tokio::test]
async fn test_system_health() {
    let handlers = create_test_handlers();
    let request = make_request("system/health", Some(JsonRpcId::Number(1)), None);

    let response = handlers.dispatch(request).await;

    assert!(response.error.is_none(), "system/health should succeed");
    let result = response.result.expect("Should have result");

    assert!(result.get("healthy").is_some(), "Should have healthy field");
}

#[tokio::test]
async fn test_system_health_components() {
    let handlers = create_test_handlers();
    let request = make_request("system/health", Some(JsonRpcId::Number(1)), None);

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");

    if let Some(components) = result.get("components") {
        assert!(components.is_object(), "components should be an object");
    }
}
