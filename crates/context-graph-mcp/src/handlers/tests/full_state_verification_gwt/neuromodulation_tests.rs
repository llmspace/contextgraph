//! Neuromodulation FSV Tests (P5-03, P5-04)
//!
//! Verifies 4-modulator control system per constitution.yaml:
//! - Dopamine (DA): [1, 5] - Controls Hopfield beta
//! - Serotonin (5HT): [0, 1] - Scales embedding space weights E1-E13
//! - Noradrenaline (NE): [0.5, 2] - Controls attention temperature
//! - Acetylcholine (ACh): [0.001, 0.002] - UTL learning rate (READ-ONLY)

use serde_json::json;

use super::{create_handlers_with_gwt, extract_tool_content, make_tool_call_request};
use crate::tools::tool_names;

/// P5-03: FSV test verifying get_neuromodulation_state returns REAL modulator levels.
#[tokio::test]
async fn test_get_neuromodulation_state_returns_real_modulator_levels() {
    let handlers = create_handlers_with_gwt();

    let request = make_tool_call_request(tool_names::GET_NEUROMODULATION_STATE, None);
    let response = handlers.dispatch(request).await;
    let response_json = serde_json::to_value(&response).expect("serialize");

    // Must have result, no error
    assert!(
        response.error.is_none(),
        "get_neuromodulation_state must not return error: {:?}",
        response.error
    );

    // Extract content from MCP tool result format
    let content = extract_tool_content(&response_json)
        .expect("get_neuromodulation_state must return content");

    // FSV-1: Must have dopamine with correct structure
    let dopamine = &content["dopamine"];
    assert!(dopamine.is_object(), "dopamine must be an object");

    let da_level = dopamine["level"]
        .as_f64()
        .expect("dopamine.level must be f64");
    let da_range = &dopamine["range"];
    let da_min = da_range["min"]
        .as_f64()
        .expect("dopamine.range.min must be f64");
    let _da_baseline = da_range["baseline"]
        .as_f64()
        .expect("dopamine.range.baseline must be f64");
    let da_max = da_range["max"]
        .as_f64()
        .expect("dopamine.range.max must be f64");

    // Constitution mandates DA range [1, 5]
    assert!(
        da_min >= 1.0 && da_max <= 5.0,
        "DA range must be within [1, 5], got [{}, {}]",
        da_min,
        da_max
    );
    assert!(
        da_level >= da_min && da_level <= da_max,
        "DA level {} must be within range [{}, {}]",
        da_level,
        da_min,
        da_max
    );
    assert_eq!(
        dopamine["parameter"].as_str(),
        Some("hopfield.beta"),
        "DA parameter must be hopfield.beta"
    );

    // FSV-2: Must have serotonin with correct structure
    let serotonin = &content["serotonin"];
    assert!(serotonin.is_object(), "serotonin must be an object");

    let sht_level = serotonin["level"]
        .as_f64()
        .expect("serotonin.level must be f64");
    let sht_range = &serotonin["range"];
    let sht_min = sht_range["min"]
        .as_f64()
        .expect("serotonin.range.min must be f64");
    let sht_max = sht_range["max"]
        .as_f64()
        .expect("serotonin.range.max must be f64");

    // Constitution mandates 5HT range [0, 1]
    assert!(
        sht_min >= 0.0 && sht_max <= 1.0,
        "5HT range must be within [0, 1], got [{}, {}]",
        sht_min,
        sht_max
    );
    assert!(
        sht_level >= sht_min && sht_level <= sht_max,
        "5HT level {} must be within range [{}, {}]",
        sht_level,
        sht_min,
        sht_max
    );

    // 5HT must have space_weights array (13 embedder weights)
    let space_weights = serotonin["space_weights"]
        .as_array()
        .expect("serotonin.space_weights must be array");
    assert_eq!(
        space_weights.len(),
        13,
        "5HT space_weights must have 13 elements (E1-E13)"
    );

    // FSV-3: Must have noradrenaline with correct structure
    let noradrenaline = &content["noradrenaline"];
    assert!(noradrenaline.is_object(), "noradrenaline must be an object");

    let ne_level = noradrenaline["level"]
        .as_f64()
        .expect("noradrenaline.level must be f64");
    let ne_range = &noradrenaline["range"];
    let ne_min = ne_range["min"]
        .as_f64()
        .expect("noradrenaline.range.min must be f64");
    let ne_max = ne_range["max"]
        .as_f64()
        .expect("noradrenaline.range.max must be f64");

    // Constitution mandates NE range [0.5, 2]
    assert!(
        ne_min >= 0.5 && ne_max <= 2.0,
        "NE range must be within [0.5, 2], got [{}, {}]",
        ne_min,
        ne_max
    );
    assert!(
        ne_level >= ne_min && ne_level <= ne_max,
        "NE level {} must be within range [{}, {}]",
        ne_level,
        ne_min,
        ne_max
    );
    assert_eq!(
        noradrenaline["parameter"].as_str(),
        Some("attention.temp"),
        "NE parameter must be attention.temp"
    );

    // FSV-4: Must have acetylcholine with correct structure (READ-ONLY)
    let acetylcholine = &content["acetylcholine"];
    assert!(acetylcholine.is_object(), "acetylcholine must be an object");

    let ach_level = acetylcholine["level"]
        .as_f64()
        .expect("acetylcholine.level must be f64");
    let ach_range = &acetylcholine["range"];
    let ach_min = ach_range["min"]
        .as_f64()
        .expect("acetylcholine.range.min must be f64");
    let ach_max = ach_range["max"]
        .as_f64()
        .expect("acetylcholine.range.max must be f64");

    // Constitution mandates ACh range [0.001, 0.002] (with f32 precision tolerance)
    // f32 0.001 = 0.0010000000474974513, f32 0.002 = 0.0020000000949949026
    let epsilon = 0.0001; // Allow for f32 representation imprecision
    assert!(
        ach_min >= (0.001 - epsilon) && ach_max <= (0.002 + epsilon),
        "ACh range must be within [0.001, 0.002] (+-epsilon), got [{}, {}]",
        ach_min,
        ach_max
    );
    assert!(
        ach_level >= (ach_min - epsilon) && ach_level <= (ach_max + epsilon),
        "ACh level {} must be within range [{}, {}]",
        ach_level,
        ach_min,
        ach_max
    );
    assert_eq!(
        acetylcholine["read_only"].as_bool(),
        Some(true),
        "ACh must be marked read_only"
    );
    assert_eq!(
        acetylcholine["parameter"].as_str(),
        Some("utl.lr"),
        "ACh parameter must be utl.lr"
    );

    // FSV-5: Must have derived_parameters with computed values
    let derived = &content["derived_parameters"];
    assert!(derived.is_object(), "derived_parameters must be an object");

    let hopfield_beta = derived["hopfield_beta"]
        .as_f64()
        .expect("derived_parameters.hopfield_beta must be f64");
    let attention_temp = derived["attention_temp"]
        .as_f64()
        .expect("derived_parameters.attention_temp must be f64");
    let utl_lr = derived["utl_learning_rate"]
        .as_f64()
        .expect("derived_parameters.utl_learning_rate must be f64");

    // Derived values must be positive
    assert!(hopfield_beta > 0.0, "hopfield_beta must be positive");
    assert!(attention_temp > 0.0, "attention_temp must be positive");
    assert!(utl_lr > 0.0, "utl_learning_rate must be positive");

    // FSV-6: Must have constitution_reference
    let constitution = &content["constitution_reference"];
    assert!(
        constitution.is_object(),
        "constitution_reference must be an object"
    );

    println!("FSV PASSED: get_neuromodulation_state - All 4 modulators verified");
    println!(
        "  DA={:.3} [{}, {}], 5HT={:.3} [{}, {}]",
        da_level, da_min, da_max, sht_level, sht_min, sht_max
    );
    println!(
        "  NE={:.3} [{}, {}], ACh={:.6} [{}, {}] (read-only)",
        ne_level, ne_min, ne_max, ach_level, ach_min, ach_max
    );
    println!(
        "  Derived: hopfield_beta={:.3}, attention_temp={:.3}, utl_lr={:.6}",
        hopfield_beta, attention_temp, utl_lr
    );
    println!(
        "  5HT space_weights: {} elements for E1-E13",
        space_weights.len()
    );
}

/// P5-04: FSV test verifying adjust_neuromodulator modifies REAL modulator levels.
///
/// TASK-NEUROMOD-MCP: Verify:
/// - DA, 5HT, NE can be adjusted
/// - ACh adjustment returns error (READ-ONLY, managed by GWT)
/// - Values are clamped to constitution-mandated ranges
#[tokio::test]
async fn test_adjust_neuromodulator_modifies_real_levels() {
    let handlers = create_handlers_with_gwt();

    // PART 1: Get initial dopamine level
    let initial_request = make_tool_call_request(tool_names::GET_NEUROMODULATION_STATE, None);
    let initial_response = handlers.dispatch(initial_request).await;
    let initial_json = serde_json::to_value(&initial_response).expect("serialize");
    let initial_content =
        extract_tool_content(&initial_json).expect("get_neuromodulation_state must return content");

    let initial_da = initial_content["dopamine"]["level"]
        .as_f64()
        .expect("initial dopamine.level must be f64");

    println!("Initial dopamine level: {}", initial_da);

    // PART 2: Adjust dopamine by +0.5
    let adjust_request = make_tool_call_request(
        tool_names::ADJUST_NEUROMODULATOR,
        Some(json!({
            "modulator": "dopamine",
            "delta": 0.5
        })),
    );
    let adjust_response = handlers.dispatch(adjust_request).await;
    let adjust_json = serde_json::to_value(&adjust_response).expect("serialize");

    assert!(
        adjust_response.error.is_none(),
        "adjust_neuromodulator should not error: {:?}",
        adjust_json.get("error")
    );

    let adjust_content =
        extract_tool_content(&adjust_json).expect("adjust_neuromodulator must return content");

    // FSV-1: Must have modulator name
    assert_eq!(
        adjust_content["modulator"].as_str(),
        Some("dopamine"),
        "modulator must be dopamine"
    );

    // FSV-2: Must have old_level
    let old_level = adjust_content["old_level"]
        .as_f64()
        .expect("old_level must be f64");
    assert!(
        (old_level - initial_da).abs() < 0.0001,
        "old_level {} should match initial {}",
        old_level,
        initial_da
    );

    // FSV-3: Must have new_level
    let new_level = adjust_content["new_level"]
        .as_f64()
        .expect("new_level must be f64");

    // FSV-4: Must have delta_requested and delta_applied
    let delta_requested = adjust_content["delta_requested"]
        .as_f64()
        .expect("delta_requested must be f64");
    let delta_applied = adjust_content["delta_applied"]
        .as_f64()
        .expect("delta_applied must be f64");

    assert!(
        (delta_requested - 0.5).abs() < 0.0001,
        "delta_requested {} should be 0.5",
        delta_requested
    );

    // New level should be old + delta_applied (may be clamped)
    assert!(
        ((new_level - old_level) - delta_applied).abs() < 0.0001,
        "new_level {} - old_level {} should equal delta_applied {}",
        new_level,
        old_level,
        delta_applied
    );

    // FSV-5: Must have range object
    let range = &adjust_content["range"];
    assert!(range.is_object(), "range must be an object");
    let range_min = range["min"].as_f64().expect("range.min must be f64");
    let range_max = range["max"].as_f64().expect("range.max must be f64");

    // New level must be within range
    assert!(
        new_level >= range_min && new_level <= range_max,
        "new_level {} must be within [{}, {}]",
        new_level,
        range_min,
        range_max
    );

    // FSV-6: Must have clamped flag
    let clamped = adjust_content["clamped"]
        .as_bool()
        .expect("clamped must be bool");

    println!(
        "FSV: adjust_neuromodulator dopamine - old={:.3}, new={:.3}, delta_req={:.3}, delta_app={:.3}, clamped={}",
        old_level, new_level, delta_requested, delta_applied, clamped
    );

    // PART 3: Verify state changed
    let verify_request = make_tool_call_request(tool_names::GET_NEUROMODULATION_STATE, None);
    let verify_response = handlers.dispatch(verify_request).await;
    let verify_json = serde_json::to_value(&verify_response).expect("serialize");
    let verify_content =
        extract_tool_content(&verify_json).expect("get_neuromodulation_state must return content");

    let verify_da = verify_content["dopamine"]["level"]
        .as_f64()
        .expect("verify dopamine.level must be f64");

    assert!(
        (verify_da - new_level).abs() < 0.0001,
        "Verified dopamine {} should match new_level {}",
        verify_da,
        new_level
    );

    println!(
        "FSV: State verified - dopamine changed from {:.3} to {:.3}",
        initial_da, verify_da
    );

    // PART 4: Test that ACh adjustment returns error (READ-ONLY)
    let ach_adjust_request = make_tool_call_request(
        tool_names::ADJUST_NEUROMODULATOR,
        Some(json!({
            "modulator": "acetylcholine",
            "delta": 0.0001
        })),
    );
    let ach_response = handlers.dispatch(ach_adjust_request).await;
    let ach_json = serde_json::to_value(&ach_response).expect("serialize");

    // ACh adjustment MUST fail (it's read-only per constitution)
    assert!(
        ach_response.error.is_some(),
        "Adjusting ACh must return error (read-only)"
    );

    let error_msg = ach_json["error"]["message"].as_str().unwrap_or("");
    assert!(
        error_msg.to_lowercase().contains("read-only")
            || error_msg.to_lowercase().contains("read only")
            || error_msg.to_lowercase().contains("gwt"),
        "Error message should mention read-only or GWT: {}",
        error_msg
    );

    println!(
        "FSV: ACh adjustment correctly rejected (read-only): {}",
        error_msg
    );

    // PART 5: Test clamping at max boundary (increase DA to exceed max)
    let big_delta = 10.0; // This should exceed the [1, 5] range
    let clamp_request = make_tool_call_request(
        tool_names::ADJUST_NEUROMODULATOR,
        Some(json!({
            "modulator": "dopamine",
            "delta": big_delta
        })),
    );
    let clamp_response = handlers.dispatch(clamp_request).await;
    let clamp_json = serde_json::to_value(&clamp_response).expect("serialize");

    let clamp_content = extract_tool_content(&clamp_json)
        .expect("adjust_neuromodulator must return content for clamp test");

    let clamp_new = clamp_content["new_level"]
        .as_f64()
        .expect("clamp new_level must be f64");
    let clamp_clamped = clamp_content["clamped"]
        .as_bool()
        .expect("clamp clamped must be bool");

    // Should be clamped to max (5.0)
    assert!(
        clamp_clamped,
        "Value should be clamped when exceeding range"
    );
    assert!(
        clamp_new <= 5.0 + 0.0001,
        "Clamped value {} should be at max 5.0",
        clamp_new
    );

    println!(
        "FSV: Clamping works - requested delta={}, clamped to max={:.3}",
        big_delta, clamp_new
    );

    println!("FSV PASSED: adjust_neuromodulator - all validations complete");
}
