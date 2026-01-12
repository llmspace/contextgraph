//! Ego State FSV Tests
//!
//! Verifies purpose vector and identity continuity.

use super::{create_handlers_with_gwt, extract_tool_content, make_tool_call_request};
use crate::tools::tool_names;

#[tokio::test]
async fn test_get_ego_state_returns_real_identity_data() {
    // SETUP: Create handlers with real GWT components
    let handlers = create_handlers_with_gwt();

    // EXECUTE: Call get_ego_state tool
    let request = make_tool_call_request(tool_names::GET_EGO_STATE, None);
    let response = handlers.dispatch(request).await;

    // VERIFY: Response is successful
    let response_json = serde_json::to_value(&response).expect("serialize response");
    assert!(
        response_json.get("error").is_none(),
        "Expected success, got error: {:?}",
        response_json.get("error")
    );

    // VERIFY: Extract and validate tool content
    let content = extract_tool_content(&response_json).expect("Tool response must have content");

    // FSV-1: purpose_vector must have 13 dimensions
    let purpose_vector = content["purpose_vector"]
        .as_array()
        .expect("purpose_vector must be array");
    assert_eq!(
        purpose_vector.len(),
        13,
        "Purpose vector must have 13 dimensions, got {}",
        purpose_vector.len()
    );

    // FSV-2: All purpose vector elements must be valid floats in [-1, 1]
    for (i, pv) in purpose_vector.iter().enumerate() {
        let p = pv.as_f64().expect("purpose element must be f64");
        assert!(
            (-1.0..=1.0).contains(&p),
            "Purpose vector[{}]={} must be in [-1, 1]",
            i,
            p
        );
    }

    // FSV-3: identity_coherence must be in [0, 1]
    let identity_coherence = content["identity_coherence"]
        .as_f64()
        .expect("identity_coherence must be f64");
    assert!(
        (0.0..=1.0).contains(&identity_coherence),
        "Identity coherence={} must be in [0, 1]",
        identity_coherence
    );

    // FSV-4: coherence_with_actions must be in [0, 1]
    let coherence_with_actions = content["coherence_with_actions"]
        .as_f64()
        .expect("coherence_with_actions must be f64");
    assert!(
        (0.0..=1.0).contains(&coherence_with_actions),
        "Coherence with actions={} must be in [0, 1]",
        coherence_with_actions
    );

    // FSV-5: identity_status must be valid status
    let identity_status = content["identity_status"]
        .as_str()
        .expect("identity_status must be string");
    let valid_statuses = ["Healthy", "Warning", "Degraded", "Critical"];
    assert!(
        valid_statuses.iter().any(|s| identity_status.contains(s)),
        "Identity status '{}' must contain one of {:?}",
        identity_status,
        valid_statuses
    );

    // FSV-6: trajectory_length must be valid (u64 is always non-negative)
    let trajectory_length = content["trajectory_length"]
        .as_u64()
        .expect("trajectory_length must be u64");
    // u64 is always >= 0, so we just verify we got a valid value
    let _ = trajectory_length; // Acknowledge the value is valid

    // FSV-7: Thresholds must be present
    let thresholds = &content["thresholds"];
    assert!(thresholds.is_object(), "thresholds must be an object");
    assert_eq!(
        thresholds["healthy"].as_f64(),
        Some(0.9),
        "Healthy threshold must be 0.9"
    );
    assert_eq!(
        thresholds["warning"].as_f64(),
        Some(0.7),
        "Warning threshold must be 0.7"
    );

    println!(
        "FSV PASSED: Ego state returned REAL data: pv_len={}, coherence={:.4}, status={}",
        purpose_vector.len(),
        identity_coherence,
        identity_status
    );
}
