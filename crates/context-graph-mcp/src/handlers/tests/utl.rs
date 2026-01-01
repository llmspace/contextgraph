//! UTL Handler Tests

use serde_json::json;

use crate::protocol::JsonRpcId;

use super::{create_test_handlers, make_request};

#[tokio::test]
async fn test_utl_compute_valid() {
    let handlers = create_test_handlers();
    // Handler expects 'input' parameter, not 'content'
    let params = json!({
        "input": "Test content for UTL computation"
    });
    let request = make_request("utl/compute", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(response.error.is_none(), "utl/compute should succeed");
    let result = response.result.expect("Should have result");

    // Handler returns only learningScore
    assert!(
        result.get("learningScore").is_some(),
        "Should have learningScore"
    );
}

#[tokio::test]
async fn test_utl_compute_missing_input() {
    let handlers = create_test_handlers();
    let params = json!({});
    let request = make_request("utl/compute", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_some(),
        "utl/compute should fail without input"
    );
}

#[tokio::test]
async fn test_utl_compute_with_input() {
    let handlers = create_test_handlers();
    let params = json!({
        "input": "Learning about neural networks"
    });
    let request = make_request("utl/compute", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "utl/compute with input should succeed"
    );
}

#[tokio::test]
async fn test_utl_metrics_valid() {
    let handlers = create_test_handlers();
    // Handler expects 'input' parameter
    let params = json!({
        "input": "Test input for metrics"
    });
    let request = make_request("utl/metrics", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;

    assert!(response.error.is_none(), "utl/metrics should succeed");
    let result = response.result.expect("Should have result");

    // Verify metrics fields
    assert!(result.get("entropy").is_some(), "Should have entropy");
    assert!(result.get("coherence").is_some(), "Should have coherence");
    assert!(
        result.get("learningScore").is_some(),
        "Should have learningScore"
    );
}

#[tokio::test]
async fn test_utl_compute_learning_score_range() {
    let handlers = create_test_handlers();
    let params = json!({
        "input": "Simple test content"
    });
    let request = make_request("utl/compute", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");

    let learning_score = result
        .get("learningScore")
        .and_then(|v| v.as_f64())
        .expect("Should have learningScore");

    assert!(
        (0.0..=1.0).contains(&learning_score),
        "learningScore should be between 0 and 1, got {}",
        learning_score
    );
}

#[tokio::test]
async fn test_utl_metrics_entropy_range() {
    let handlers = create_test_handlers();
    let params = json!({
        "input": "Test entropy calculation"
    });
    let request = make_request("utl/metrics", Some(JsonRpcId::Number(1)), Some(params));

    let response = handlers.dispatch(request).await;
    let result = response.result.expect("Should have result");

    let entropy = result
        .get("entropy")
        .and_then(|v| v.as_f64())
        .expect("Should have entropy");

    assert!(
        (0.0..=1.0).contains(&entropy),
        "entropy should be between 0 and 1, got {}",
        entropy
    );
}
