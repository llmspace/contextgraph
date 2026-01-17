//! ATC MCP tools tests (3 tools):
//! - get_threshold_status
//! - get_calibration_metrics
//! - trigger_recalibration

use serde_json::json;

use super::helpers::{assert_success, make_tool_call};
use super::synthetic_data;
use crate::handlers::tests::create_test_handlers_with_all_components;

// -------------------------------------------------------------------------
// get_threshold_status
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_get_threshold_status_basic() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call("get_threshold_status", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_threshold_status");
}

#[tokio::test]
async fn test_get_threshold_status_with_domain() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call(
        "get_threshold_status",
        json!({
            "domain": "Code"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_threshold_status");
}

#[tokio::test]
async fn test_get_threshold_status_with_embedder() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call(
        "get_threshold_status",
        json!({
            "embedder_id": 1
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_threshold_status");
}

// -------------------------------------------------------------------------
// get_calibration_metrics
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_get_calibration_metrics_basic() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call("get_calibration_metrics", json!({}));

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_calibration_metrics");
}

#[tokio::test]
async fn test_get_calibration_metrics_with_timeframe() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call(
        "get_calibration_metrics",
        json!({
            "timeframe": "7d"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "get_calibration_metrics");
}

// -------------------------------------------------------------------------
// trigger_recalibration
// -------------------------------------------------------------------------

#[tokio::test]
async fn test_trigger_recalibration_level_1() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call(
        "trigger_recalibration",
        json!({
            "level": synthetic_data::atc::LEVEL_EWMA
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "trigger_recalibration");
}

#[tokio::test]
async fn test_trigger_recalibration_level_2() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call(
        "trigger_recalibration",
        json!({
            "level": synthetic_data::atc::LEVEL_TEMPERATURE
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "trigger_recalibration");
}

#[tokio::test]
async fn test_trigger_recalibration_level_3() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call(
        "trigger_recalibration",
        json!({
            "level": synthetic_data::atc::LEVEL_BANDIT
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "trigger_recalibration");
}

#[tokio::test]
async fn test_trigger_recalibration_level_4() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call(
        "trigger_recalibration",
        json!({
            "level": synthetic_data::atc::LEVEL_BAYESIAN
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "trigger_recalibration");
}

#[tokio::test]
async fn test_trigger_recalibration_missing_level() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call("trigger_recalibration", json!({}));

    let response = handlers.dispatch(request).await;
    // Handler returns JSON-RPC error for missing required 'level' parameter
    assert!(
        response.error.is_some(),
        "trigger_recalibration should return JSON-RPC error for missing level"
    );
}

#[tokio::test]
async fn test_trigger_recalibration_with_domain() {
    let handlers = create_test_handlers_with_all_components();
    let request = make_tool_call(
        "trigger_recalibration",
        json!({
            "level": 2,
            "domain": "Medical"
        }),
    );

    let response = handlers.dispatch(request).await;
    assert_success(&response, "trigger_recalibration");
}
