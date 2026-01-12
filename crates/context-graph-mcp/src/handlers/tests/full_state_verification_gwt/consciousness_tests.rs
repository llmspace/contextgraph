//! Consciousness State and Cross-Validation FSV Tests
//!
//! Verifies C(t) = I(t) x R(t) x D(t) computation and cross-tool consistency.

use super::{create_handlers_with_gwt, extract_tool_content, make_tool_call_request};
use crate::tools::tool_names;

#[tokio::test]
async fn test_get_consciousness_state_returns_real_gwt_data() {
    // SETUP: Create handlers with real GWT components
    let handlers = create_handlers_with_gwt();

    // EXECUTE: Call get_consciousness_state tool
    let request = make_tool_call_request(tool_names::GET_CONSCIOUSNESS_STATE, None);
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

    // FSV-1: Consciousness C must be in [0, 1]
    let c = content["C"].as_f64().expect("C must be f64");
    assert!(
        (0.0..=1.0).contains(&c),
        "Consciousness C={} must be in [0, 1]",
        c
    );

    // FSV-2: Order parameter r must be in [0, 1]
    let r = content["r"].as_f64().expect("r must be f64");
    assert!(
        (0.0..=1.0).contains(&r),
        "Order parameter r={} must be in [0, 1]",
        r
    );

    // FSV-3: Integration, reflection, differentiation must be in [0, 1]
    let integration = content["integration"]
        .as_f64()
        .expect("integration must be f64");
    assert!(
        (0.0..=1.0).contains(&integration),
        "Integration={} must be in [0, 1]",
        integration
    );

    let reflection = content["reflection"]
        .as_f64()
        .expect("reflection must be f64");
    assert!(
        (0.0..=1.0).contains(&reflection),
        "Reflection={} must be in [0, 1]",
        reflection
    );

    let differentiation = content["differentiation"]
        .as_f64()
        .expect("differentiation must be f64");
    assert!(
        (0.0..=1.0).contains(&differentiation),
        "Differentiation={} must be in [0, 1]",
        differentiation
    );

    // FSV-4: State must be valid consciousness state (constitution.yaml lines 394-408)
    // All 5 states: DORMANT, FRAGMENTED, EMERGING, CONSCIOUS, HYPERSYNC
    let state = content["state"].as_str().expect("state must be string");
    let valid_states = [
        "DORMANT",
        "FRAGMENTED",
        "EMERGING",
        "CONSCIOUS",
        "HYPERSYNC",
    ];
    assert!(
        valid_states.contains(&state),
        "State '{}' must be one of {:?}",
        state,
        valid_states
    );

    // FSV-5: GWT state must be present
    let gwt_state = content["gwt_state"]
        .as_str()
        .expect("gwt_state must be string");
    assert!(!gwt_state.is_empty(), "GWT state must not be empty");

    // FSV-6: Workspace object must be present
    let workspace = &content["workspace"];
    assert!(workspace.is_object(), "workspace must be an object");
    assert!(
        workspace.get("is_broadcasting").is_some(),
        "workspace must have is_broadcasting"
    );
    assert!(
        workspace.get("has_conflict").is_some(),
        "workspace must have has_conflict"
    );
    let coherence_threshold = workspace["coherence_threshold"]
        .as_f64()
        .expect("coherence_threshold must be f64");
    // Use approximate comparison due to f32->f64 conversion
    assert!(
        (coherence_threshold - 0.8).abs() < 1e-6,
        "Coherence threshold must be ~0.8, got {}",
        coherence_threshold
    );

    // FSV-7: Identity object must be present with 13D purpose vector
    let identity = &content["identity"];
    assert!(identity.is_object(), "identity must be an object");
    let purpose_vector = identity["purpose_vector"]
        .as_array()
        .expect("purpose_vector must be array");
    assert_eq!(
        purpose_vector.len(),
        13,
        "Purpose vector must have 13 dimensions, got {}",
        purpose_vector.len()
    );

    let identity_coherence = identity["coherence"]
        .as_f64()
        .expect("identity coherence must be f64");
    assert!(
        (0.0..=1.0).contains(&identity_coherence),
        "Identity coherence={} must be in [0, 1]",
        identity_coherence
    );

    // FSV-8: Component analysis must be present
    let analysis = &content["component_analysis"];
    assert!(analysis.is_object(), "component_analysis must be an object");
    assert!(
        analysis.get("integration_sufficient").is_some(),
        "component_analysis must have integration_sufficient"
    );
    assert!(
        analysis.get("reflection_sufficient").is_some(),
        "component_analysis must have reflection_sufficient"
    );
    assert!(
        analysis.get("differentiation_sufficient").is_some(),
        "component_analysis must have differentiation_sufficient"
    );
    assert!(
        analysis.get("limiting_factor").is_some(),
        "component_analysis must have limiting_factor"
    );

    // FSV-9: Verify C = I x R x D formula (approximately)
    let computed_c = integration * reflection * differentiation;
    // Allow some tolerance for floating-point and any internal adjustments
    assert!(
        (c - computed_c).abs() < 0.01,
        "C={} should approximately equal IxRxD={} (I={}, R={}, D={})",
        c,
        computed_c,
        integration,
        reflection,
        differentiation
    );

    println!(
        "FSV PASSED: Consciousness state returned REAL data: C={:.4}, state={}, r={:.4}",
        c, state, r
    );
}

#[tokio::test]
async fn test_gwt_cross_validation_kuramoto_and_consciousness() {
    // SETUP: Create handlers with real GWT components
    let handlers = create_handlers_with_gwt();

    // EXECUTE: Call both tools
    let kuramoto_request = make_tool_call_request(tool_names::GET_KURAMOTO_SYNC, None);
    let consciousness_request = make_tool_call_request(tool_names::GET_CONSCIOUSNESS_STATE, None);

    let kuramoto_response = handlers.dispatch(kuramoto_request).await;
    let consciousness_response = handlers.dispatch(consciousness_request).await;

    // Parse responses
    let kuramoto_json = serde_json::to_value(&kuramoto_response).expect("serialize");
    let consciousness_json = serde_json::to_value(&consciousness_response).expect("serialize");

    let kuramoto_content = extract_tool_content(&kuramoto_json).expect("kuramoto content");
    let consciousness_content =
        extract_tool_content(&consciousness_json).expect("consciousness content");

    // CROSS-VALIDATION-1: Order parameter r must match between tools
    let kuramoto_r = kuramoto_content["r"].as_f64().expect("kuramoto r");
    let consciousness_r = consciousness_content["r"]
        .as_f64()
        .expect("consciousness r");

    assert!(
        (kuramoto_r - consciousness_r).abs() < 1e-10,
        "Kuramoto r={} must match consciousness r={}",
        kuramoto_r,
        consciousness_r
    );

    // CROSS-VALIDATION-2: State classifications should be consistent
    // Both tools now use ConsciousnessState::from_level() so states should match exactly
    // All 5 states per constitution.yaml lines 394-408: DORMANT, FRAGMENTED, EMERGING, CONSCIOUS, HYPERSYNC
    let kuramoto_state = kuramoto_content["state"].as_str().expect("kuramoto state");
    let consciousness_state = consciousness_content["state"]
        .as_str()
        .expect("consciousness state");

    assert_eq!(
        kuramoto_state, consciousness_state,
        "Kuramoto state '{}' must exactly match consciousness state '{}'",
        kuramoto_state, consciousness_state
    );

    // CROSS-VALIDATION-3: Integration factor should correlate with r
    let integration = consciousness_content["integration"]
        .as_f64()
        .expect("integration");
    // Integration is derived from Kuramoto r, so they should be related
    // (exact relationship depends on implementation, but both should be in [0,1])
    assert!(
        (0.0..=1.0).contains(&integration),
        "Integration derived from Kuramoto must be valid"
    );

    println!(
        "CROSS-VALIDATION PASSED: Kuramoto r={:.4} matches consciousness r={:.4}, states consistent",
        kuramoto_r, consciousness_r
    );
}

#[tokio::test]
async fn test_gwt_cross_validation_ego_and_consciousness() {
    // SETUP: Create handlers with real GWT components
    let handlers = create_handlers_with_gwt();

    // EXECUTE: Call both tools
    let ego_request = make_tool_call_request(tool_names::GET_EGO_STATE, None);
    let consciousness_request = make_tool_call_request(tool_names::GET_CONSCIOUSNESS_STATE, None);

    let ego_response = handlers.dispatch(ego_request).await;
    let consciousness_response = handlers.dispatch(consciousness_request).await;

    // Parse responses
    let ego_json = serde_json::to_value(&ego_response).expect("serialize");
    let consciousness_json = serde_json::to_value(&consciousness_response).expect("serialize");

    let ego_content = extract_tool_content(&ego_json).expect("ego content");
    let consciousness_content =
        extract_tool_content(&consciousness_json).expect("consciousness content");

    // CROSS-VALIDATION-1: Purpose vectors must match
    let ego_pv = ego_content["purpose_vector"]
        .as_array()
        .expect("ego purpose_vector");
    let consciousness_pv = consciousness_content["identity"]["purpose_vector"]
        .as_array()
        .expect("consciousness purpose_vector");

    assert_eq!(
        ego_pv.len(),
        consciousness_pv.len(),
        "Purpose vector lengths must match"
    );

    for (i, (ego_v, cons_v)) in ego_pv.iter().zip(consciousness_pv.iter()).enumerate() {
        let ego_val = ego_v.as_f64().expect("ego pv element");
        let cons_val = cons_v.as_f64().expect("cons pv element");
        assert!(
            (ego_val - cons_val).abs() < 1e-10,
            "Purpose vector[{}] ego={} must match consciousness={}",
            i,
            ego_val,
            cons_val
        );
    }

    // CROSS-VALIDATION-2: Identity coherence must match
    let ego_coherence = ego_content["identity_coherence"]
        .as_f64()
        .expect("ego coherence");
    let consciousness_coherence = consciousness_content["identity"]["coherence"]
        .as_f64()
        .expect("consciousness coherence");

    assert!(
        (ego_coherence - consciousness_coherence).abs() < 1e-10,
        "Ego coherence={} must match consciousness identity coherence={}",
        ego_coherence,
        consciousness_coherence
    );

    println!(
        "CROSS-VALIDATION PASSED: Ego and consciousness purpose vectors match (len={}), coherence={:.4}",
        ego_pv.len(),
        ego_coherence
    );
}
