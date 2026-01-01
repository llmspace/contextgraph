//! Legacy Memory Handler Tests

use serde_json::json;

use crate::protocol::JsonRpcId;

use super::{create_test_handlers, make_request};

#[tokio::test]
async fn test_memory_store_valid() {
    let handlers = create_test_handlers();
    let params = json!({
        "content": "Test memory content",
        "importance": 0.8
    });
    let request = make_request("memory/store", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(response.error.is_none(), "memory/store should succeed");
    let result = response.result.expect("Should have result");
    assert!(result.get("nodeId").is_some(), "Should return nodeId");
}

#[tokio::test]
async fn test_memory_store_missing_content() {
    let handlers = create_test_handlers();
    let params = json!({
        "importance": 0.5
    });
    let request = make_request("memory/store", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_some(),
        "memory/store should fail without content"
    );
}

#[tokio::test]
async fn test_memory_retrieve_valid() {
    let handlers = create_test_handlers();

    // First store a memory
    let store_params = json!({
        "content": "Retrievable content",
        "importance": 0.7
    });
    let store_request =
        make_request("memory/store", Some(JsonRpcId::Number(1)), Some(store_params));
    let store_response = handlers.dispatch(store_request).await;
    let node_id = store_response
        .result
        .unwrap()
        .get("nodeId")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();

    // Then retrieve it
    let retrieve_params = json!({
        "nodeId": node_id
    });
    let retrieve_request = make_request(
        "memory/retrieve",
        Some(JsonRpcId::Number(2)),
        Some(retrieve_params),
    );

    let response = handlers.dispatch(retrieve_request).await;

    assert!(response.error.is_none(), "memory/retrieve should succeed");
    let result = response.result.expect("Should have result");
    // Response returns node object with content inside
    let node = result.get("node").expect("Should return node object");
    assert!(node.get("content").is_some(), "Node should have content");
}

#[tokio::test]
async fn test_memory_search_valid() {
    let handlers = create_test_handlers();
    let params = json!({
        "query": "test search",
        "topK": 5
    });
    let request = make_request("memory/search", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(response.error.is_none(), "memory/search should succeed");
    let result = response.result.expect("Should have result");
    assert!(result.get("results").is_some(), "Should return results");
}

#[tokio::test]
async fn test_memory_delete_valid() {
    let handlers = create_test_handlers();

    // First store a memory
    let store_params = json!({
        "content": "Content to delete",
        "importance": 0.5
    });
    let store_request =
        make_request("memory/store", Some(JsonRpcId::Number(1)), Some(store_params));
    let store_response = handlers.dispatch(store_request).await;
    let node_id = store_response
        .result
        .unwrap()
        .get("nodeId")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();

    // Then delete it
    let delete_params = json!({
        "nodeId": node_id
    });
    let delete_request = make_request(
        "memory/delete",
        Some(JsonRpcId::Number(2)),
        Some(delete_params),
    );

    let response = handlers.dispatch(delete_request).await;

    assert!(response.error.is_none(), "memory/delete should succeed");
}
