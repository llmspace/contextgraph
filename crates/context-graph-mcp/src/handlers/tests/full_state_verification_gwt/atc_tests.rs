//! ATC (Adaptive Threshold Calibration) FSV Tests
//!
//! Verifies threshold status, calibration metrics, and recalibration.
//!
//! CONSTITUTION REFERENCE: adaptive_thresholds section
//! - Level 1: EWMA Drift Tracker (per-query)
//! - Level 2: Temperature Scaling (hourly, per-embedder)
//! - Level 3: Bandit Threshold Selector (session)
//! - Level 4: Bayesian Meta-Optimizer (weekly)

use serde_json::json;

use super::{create_handlers_with_gwt, extract_tool_content, make_tool_call_request};
use crate::tools::tool_names;

/// P4-06: FSV test to verify get_threshold_status returns REAL ATC threshold data.
///
/// Threshold priors: theta_opt=0.75, theta_acc=0.70, theta_warn=0.55, theta_dup=0.90, theta_edge=0.70
#[tokio::test]
async fn test_get_threshold_status_returns_real_atc_data() {
    // SETUP: Create handlers with real GWT/ATC components
    let handlers = create_handlers_with_gwt();

    // EXECUTE: Call get_threshold_status tool
    let request = make_tool_call_request(tool_names::GET_THRESHOLD_STATUS, None);
    let response = handlers.dispatch(request).await;

    // Parse response
    let response_json = serde_json::to_value(&response).expect("serialize");
    assert!(
        response.error.is_none(),
        "get_threshold_status should not error"
    );

    let content =
        extract_tool_content(&response_json).expect("get_threshold_status must return content");

    // FSV-1: Must have domain (defaults to "General")
    let domain = content["domain"].as_str().expect("domain must be string");
    assert!(!domain.is_empty(), "domain must not be empty");

    // FSV-2: Must have thresholds object with domain_thresholds
    let thresholds = &content["thresholds"];
    assert!(thresholds.is_object(), "thresholds must be an object");

    // FSV-3: Must have calibration object with ECE, MCE, Brier
    let calibration = &content["calibration"];
    assert!(calibration.is_object(), "calibration must be an object");

    let ece = calibration["ece"].as_f64().expect("ece must be f64");
    assert!(
        (0.0..=1.0).contains(&ece),
        "ECE must be in [0, 1], got {}",
        ece
    );

    let mce = calibration["mce"].as_f64().expect("mce must be f64");
    assert!(
        (0.0..=1.0).contains(&mce),
        "MCE must be in [0, 1], got {}",
        mce
    );

    let brier = calibration["brier"].as_f64().expect("brier must be f64");
    assert!(
        (0.0..=1.0).contains(&brier),
        "Brier score must be in [0, 1], got {}",
        brier
    );

    // FSV-4: Must have calibration status
    let status = calibration["status"]
        .as_str()
        .expect("status must be string");
    assert!(!status.is_empty(), "calibration status must not be empty");

    // FSV-5: Must have sample_count
    let sample_count = calibration["sample_count"]
        .as_u64()
        .expect("sample_count must be u64");
    // Sample count can be 0 for fresh ATC

    // FSV-6: Must have drift_scores (can be empty array for fresh ATC)
    assert!(
        content.get("drift_scores").is_some(),
        "drift_scores must be present"
    );

    // FSV-7: Must have recalibration flags (booleans)
    let should_level2 = content["should_recalibrate_level2"]
        .as_bool()
        .expect("should_recalibrate_level2 must be bool");
    let should_level3 = content["should_explore_level3"]
        .as_bool()
        .expect("should_explore_level3 must be bool");
    let should_level4 = content["should_optimize_level4"]
        .as_bool()
        .expect("should_optimize_level4 must be bool");

    println!(
        "FSV PASSED: get_threshold_status returned domain={}, ECE={:.4}, MCE={:.4}, Brier={:.4}, status={}, samples={}",
        domain, ece, mce, brier, status, sample_count
    );
    println!(
        "  Recalibration flags: level2={}, level3={}, level4={}",
        should_level2, should_level3, should_level4
    );
}

/// P4-07: FSV test to verify get_calibration_metrics returns REAL calibration data.
///
/// CONSTITUTION REFERENCE: adaptive_thresholds section
/// - ECE target < 0.05
/// - MCE target < 0.10
/// - Brier target < 0.10
#[tokio::test]
async fn test_get_calibration_metrics_returns_real_data() {
    // SETUP: Create handlers with real GWT/ATC components
    let handlers = create_handlers_with_gwt();

    // EXECUTE: Call get_calibration_metrics tool
    let request = make_tool_call_request(tool_names::GET_CALIBRATION_METRICS, None);
    let response = handlers.dispatch(request).await;

    // Parse response
    let response_json = serde_json::to_value(&response).expect("serialize");
    assert!(
        response.error.is_none(),
        "get_calibration_metrics should not error"
    );

    let content =
        extract_tool_content(&response_json).expect("get_calibration_metrics must return content");

    // FSV-1: Must have metrics object
    let metrics = &content["metrics"];
    assert!(metrics.is_object(), "metrics must be an object");

    // FSV-2: Must have ECE (Expected Calibration Error)
    let ece = metrics["ece"].as_f64().expect("ece must be f64");
    assert!(
        (0.0..=1.0).contains(&ece),
        "ECE must be in [0, 1], got {}",
        ece
    );

    // FSV-3: Must have MCE (Maximum Calibration Error)
    let mce = metrics["mce"].as_f64().expect("mce must be f64");
    assert!(
        (0.0..=1.0).contains(&mce),
        "MCE must be in [0, 1], got {}",
        mce
    );

    // FSV-4: Must have Brier score
    let brier = metrics["brier"].as_f64().expect("brier must be f64");
    assert!(
        (0.0..=1.0).contains(&brier),
        "Brier score must be in [0, 1], got {}",
        brier
    );

    // FSV-5: Must have status at top level
    let status = content["status"].as_str().expect("status must be string");
    assert!(!status.is_empty(), "status must not be empty");

    // FSV-6: Must have sample_count in metrics
    let sample_count = metrics["sample_count"]
        .as_u64()
        .expect("sample_count must be u64");

    // FSV-7: Must have targets embedded in metrics
    let ece_target = metrics["ece_target"]
        .as_f64()
        .expect("ece_target must be f64");
    let mce_target = metrics["mce_target"]
        .as_f64()
        .expect("mce_target must be f64");
    let brier_target = metrics["brier_target"]
        .as_f64()
        .expect("brier_target must be f64");

    // FSV-8: Check if we're meeting calibration targets
    let meets_ece = ece <= ece_target;
    let meets_mce = mce <= mce_target;
    let meets_brier = brier <= brier_target;

    // FSV-9: Must have recommendations object
    let recommendations = &content["recommendations"];
    assert!(
        recommendations.is_object(),
        "recommendations must be an object"
    );
    assert!(
        recommendations.get("level2_recalibration_needed").is_some(),
        "must have level2_recalibration_needed"
    );

    println!(
        "FSV PASSED: get_calibration_metrics returned ECE={:.4} (target<{:.2}), MCE={:.4} (target<{:.2}), Brier={:.4} (target<{:.2})",
        ece, ece_target, mce, mce_target, brier, brier_target
    );
    println!(
        "  Status: {}, samples={}, meets_targets: ECE={}, MCE={}, Brier={}",
        status, sample_count, meets_ece, meets_mce, meets_brier
    );
}

/// P4-08: FSV test to verify trigger_recalibration performs REAL recalibration.
///
/// CONSTITUTION REFERENCE: adaptive_thresholds section
/// - Level 1: EWMA Drift Tracker (per-query)
/// - Level 2: Temperature Scaling (hourly, per-embedder)
/// - Level 3: Thompson Sampling Bandit (session)
/// - Level 4: Bayesian Meta-Optimizer (weekly)
#[tokio::test]
async fn test_trigger_recalibration_performs_real_calibration() {
    // SETUP: Create handlers with real GWT/ATC components
    let handlers = create_handlers_with_gwt();

    // Test all 4 levels of recalibration
    for level in 1..=4u64 {
        // EXECUTE: Call trigger_recalibration tool with level
        let args = json!({ "level": level });
        let request = make_tool_call_request(tool_names::TRIGGER_RECALIBRATION, Some(args));
        let response = handlers.dispatch(request).await;

        // Parse response
        let response_json = serde_json::to_value(&response).expect("serialize");
        assert!(
            response.error.is_none(),
            "trigger_recalibration level {} should not error: {:?}",
            level,
            response.error
        );

        let content = extract_tool_content(&response_json)
            .unwrap_or_else(|| panic!("trigger_recalibration level {} must return content", level));

        // FSV-1: Must have success flag
        let success = content["success"].as_bool().expect("success must be bool");
        assert!(success, "recalibration should succeed");

        // FSV-2: Must have recalibration object with level details
        let recalibration = &content["recalibration"];
        assert!(recalibration.is_object(), "recalibration must be an object");

        let returned_level = recalibration["level"].as_u64().expect("level must be u64");
        assert_eq!(
            returned_level, level,
            "returned level must match requested level"
        );

        let level_name = recalibration["level_name"]
            .as_str()
            .expect("level_name must be string");
        assert!(!level_name.is_empty(), "level_name must not be empty");

        let action = recalibration["action"]
            .as_str()
            .expect("action must be string");
        // Action should be one of: "reported", "recalibrated", "initialized", "triggered", "skipped"
        assert!(
            [
                "reported",
                "recalibrated",
                "initialized",
                "triggered",
                "skipped"
            ]
            .contains(&action),
            "action should be valid: {}",
            action
        );

        // FSV-3: Must have metrics_before and metrics_after
        let metrics_before = &content["metrics_before"];
        assert!(
            metrics_before.is_object(),
            "metrics_before must be an object"
        );
        assert!(
            metrics_before.get("ece").is_some(),
            "metrics_before must have ece"
        );

        let metrics_after = &content["metrics_after"];
        assert!(metrics_after.is_object(), "metrics_after must be an object");
        assert!(
            metrics_after.get("ece").is_some(),
            "metrics_after must have ece"
        );

        println!(
            "FSV PASSED: trigger_recalibration level {} - name='{}', action='{}'",
            level, level_name, action
        );
    }

    println!("FSV PASSED: All 4 ATC recalibration levels verified");
}
