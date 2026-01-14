//! Kuramoto Synchronization FSV Tests
//!
//! Verifies the 13-oscillator network state returns REAL data.
//! TASK-39: Adds get_kuramoto_state tool with is_running status.

use super::{create_handlers_with_gwt, extract_tool_content, make_tool_call_request};
use crate::tools::tool_names;

#[tokio::test]
async fn test_get_kuramoto_sync_returns_real_oscillator_data() {
    // SETUP: Create handlers with real GWT components
    let handlers = create_handlers_with_gwt();

    // EXECUTE: Call get_kuramoto_sync tool
    let request = make_tool_call_request(tool_names::GET_KURAMOTO_SYNC, None);
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

    // FSV-1: Order parameter r must be in [0, 1]
    let r = content["r"].as_f64().expect("r must be f64");
    assert!(
        (0.0..=1.0).contains(&r),
        "Order parameter r={} must be in [0, 1]",
        r
    );

    // FSV-2: Mean phase psi must be in [0, 2pi]
    let psi = content["psi"].as_f64().expect("psi must be f64");
    assert!(
        (0.0..=std::f64::consts::TAU).contains(&psi),
        "Mean phase psi={} must be in [0, 2pi]",
        psi
    );

    // FSV-3: Must have exactly 13 oscillator phases
    let phases = content["phases"].as_array().expect("phases must be array");
    assert_eq!(
        phases.len(),
        13,
        "Must have exactly 13 oscillator phases, got {}",
        phases.len()
    );

    // FSV-4: All phases must be valid floats in [0, 2pi]
    for (i, phase) in phases.iter().enumerate() {
        let p = phase.as_f64().expect("phase must be f64");
        assert!(
            (0.0..=std::f64::consts::TAU).contains(&p),
            "Phase[{}]={} must be in [0, 2pi]",
            i,
            p
        );
    }

    // FSV-5: Must have exactly 13 natural frequencies
    let freqs = content["natural_freqs"]
        .as_array()
        .expect("natural_freqs must be array");
    assert_eq!(
        freqs.len(),
        13,
        "Must have exactly 13 natural frequencies, got {}",
        freqs.len()
    );

    // FSV-6: Natural frequencies must be positive (Hz)
    for (i, freq) in freqs.iter().enumerate() {
        let f = freq.as_f64().expect("freq must be f64");
        assert!(f > 0.0, "Natural frequency[{}]={} must be positive", i, f);
    }

    // FSV-7: Coupling strength K must be positive
    let coupling = content["coupling"].as_f64().expect("coupling must be f64");
    assert!(
        coupling > 0.0,
        "Coupling strength K={} must be positive",
        coupling
    );

    // FSV-8: State must be one of valid states (constitution.yaml lines 394-408)
    // All 5 states per constitution: DORMANT, FRAGMENTED, EMERGING, CONSCIOUS, HYPERSYNC
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

    // FSV-9: Synchronization must equal r
    let sync = content["synchronization"]
        .as_f64()
        .expect("sync must be f64");
    assert!(
        (sync - r).abs() < 1e-10,
        "synchronization={} must equal r={}",
        sync,
        r
    );

    // FSV-10: Elapsed time must be non-negative
    let elapsed = content["elapsed_seconds"]
        .as_f64()
        .expect("elapsed_seconds must be f64");
    assert!(
        elapsed >= 0.0,
        "Elapsed time {} must be non-negative",
        elapsed
    );

    // FSV-11: Must have 13 embedding labels
    let labels = content["embedding_labels"]
        .as_array()
        .expect("embedding_labels must be array");
    assert_eq!(
        labels.len(),
        13,
        "Must have exactly 13 embedding labels, got {}",
        labels.len()
    );

    // FSV-12: Thresholds must be present and valid
    let thresholds = &content["thresholds"];
    assert_eq!(
        thresholds["conscious"].as_f64(),
        Some(0.8),
        "Conscious threshold must be 0.8"
    );
    assert_eq!(
        thresholds["fragmented"].as_f64(),
        Some(0.5),
        "Fragmented threshold must be 0.5"
    );
    assert_eq!(
        thresholds["hypersync"].as_f64(),
        Some(0.95),
        "Hypersync threshold must be 0.95"
    );

    println!(
        "FSV PASSED: Kuramoto sync returned REAL data: r={:.4}, state={}, phases={}, freqs={}",
        r,
        state,
        phases.len(),
        freqs.len()
    );
}

/// TASK-39: Full State Verification for get_kuramoto_state tool
///
/// This test verifies:
/// 1. is_running field reflects stepper state (starts as false)
/// 2. All 13 oscillator phases are returned and valid
/// 3. All 13 natural frequencies are returned and valid (per constitution.yaml line 221)
/// 4. Coupling strength K is returned
/// 5. Order parameter r is in [0, 1]
/// 6. Mean phase psi is in [0, 2pi]
///
/// Uses REAL handlers with GWT providers, no mocks.
#[tokio::test]
async fn test_get_kuramoto_state_returns_stepper_status_and_network_data() {
    println!("\n=== FSV TEST: get_kuramoto_state (TASK-39) ===");

    // ====================================================================
    // SETUP: Create handlers with real GWT components
    // ====================================================================
    let handlers = create_handlers_with_gwt();
    println!("SETUP: Created Handlers with with_default_gwt()");

    // ====================================================================
    // STATE BEFORE: Stepper not started - is_running should be false
    // ====================================================================
    let request = make_tool_call_request(tool_names::GET_KURAMOTO_STATE, None);
    let response = handlers.dispatch(request).await;

    let response_json = serde_json::to_value(&response).expect("serialize response");
    assert!(
        response_json.get("error").is_none(),
        "Expected success, got error: {:?}",
        response_json.get("error")
    );

    let content = extract_tool_content(&response_json).expect("Tool response must have content");

    // FSV-1: is_running must be boolean and initially false (stepper not started)
    let is_running = content["is_running"]
        .as_bool()
        .expect("is_running must be boolean");
    assert!(
        !is_running,
        "is_running should be false initially (stepper not started)"
    );
    println!("FSV-1 PASS: is_running = {} (stepper not started)", is_running);

    // FSV-2: Must have exactly 13 oscillator phases
    let phases = content["phases"].as_array().expect("phases must be array");
    assert_eq!(
        phases.len(),
        13,
        "Must have exactly 13 oscillator phases, got {}",
        phases.len()
    );

    // FSV-3: All phases must be valid floats in [0, 2pi]
    for (i, phase) in phases.iter().enumerate() {
        let p = phase.as_f64().expect("phase must be f64");
        assert!(
            (0.0..=std::f64::consts::TAU).contains(&p),
            "Phase[{}]={} must be in [0, 2pi]",
            i,
            p
        );
    }
    println!("FSV-2,3 PASS: 13 phases, all in valid range [0, 2pi]");

    // FSV-4: Must have exactly 13 natural frequencies
    let frequencies = content["frequencies"]
        .as_array()
        .expect("frequencies must be array");
    assert_eq!(
        frequencies.len(),
        13,
        "Must have exactly 13 natural frequencies, got {}",
        frequencies.len()
    );

    // FSV-5: Natural frequencies must be positive (Hz per constitution.yaml line 221)
    // Expected: E1=40Hz, E2-4=8Hz, E5=25Hz, E6=4Hz, E7=25Hz, E8=12Hz, E9=80Hz, E10=40Hz, E11=15Hz, E12=60Hz, E13=4Hz
    for (i, freq) in frequencies.iter().enumerate() {
        let f = freq.as_f64().expect("freq must be f64");
        assert!(f > 0.0, "Natural frequency[{}]={} must be positive", i, f);
    }
    println!("FSV-4,5 PASS: 13 frequencies, all positive");

    // FSV-6: Coupling strength K must be present and positive
    let coupling = content["coupling"].as_f64().expect("coupling must be f64");
    assert!(
        coupling > 0.0,
        "Coupling strength K={} must be positive",
        coupling
    );
    println!("FSV-6 PASS: coupling K = {}", coupling);

    // FSV-7: Order parameter r must be in [0, 1]
    let r = content["order_parameter"]
        .as_f64()
        .expect("order_parameter must be f64");
    assert!(
        (0.0..=1.0).contains(&r),
        "Order parameter r={} must be in [0, 1]",
        r
    );
    println!("FSV-7 PASS: order_parameter r = {:.6}", r);

    // FSV-8: Mean phase psi must be in [0, 2pi]
    let psi = content["mean_phase"]
        .as_f64()
        .expect("mean_phase must be f64");
    assert!(
        (0.0..=std::f64::consts::TAU).contains(&psi),
        "Mean phase psi={} must be in [0, 2pi]",
        psi
    );
    println!("FSV-8 PASS: mean_phase psi = {:.6}", psi);

    // ====================================================================
    // START STEPPER: is_running should become true
    // ====================================================================
    let start_result = handlers.start_kuramoto_stepper();
    assert!(start_result.is_ok(), "start_kuramoto_stepper must succeed");
    println!("ACTION: Started kuramoto stepper");

    // Call tool again to verify is_running changed
    let request2 = make_tool_call_request(tool_names::GET_KURAMOTO_STATE, None);
    let response2 = handlers.dispatch(request2).await;
    let response_json2 = serde_json::to_value(&response2).expect("serialize response");
    let content2 = extract_tool_content(&response_json2).expect("Tool response must have content");

    let is_running_after_start = content2["is_running"]
        .as_bool()
        .expect("is_running must be boolean");
    assert!(
        is_running_after_start,
        "is_running should be true after starting stepper"
    );
    println!(
        "FSV-9 PASS: is_running = {} (after stepper start)",
        is_running_after_start
    );

    // ====================================================================
    // STOP STEPPER: is_running should become false again
    // ====================================================================
    let stop_result = handlers.stop_kuramoto_stepper().await;
    assert!(stop_result.is_ok(), "stop_kuramoto_stepper must succeed");
    println!("ACTION: Stopped kuramoto stepper");

    // Call tool again to verify is_running changed back
    let request3 = make_tool_call_request(tool_names::GET_KURAMOTO_STATE, None);
    let response3 = handlers.dispatch(request3).await;
    let response_json3 = serde_json::to_value(&response3).expect("serialize response");
    let content3 = extract_tool_content(&response_json3).expect("Tool response must have content");

    let is_running_after_stop = content3["is_running"]
        .as_bool()
        .expect("is_running must be boolean");
    assert!(
        !is_running_after_stop,
        "is_running should be false after stopping stepper"
    );
    println!(
        "FSV-10 PASS: is_running = {} (after stepper stop)",
        is_running_after_stop
    );

    // ====================================================================
    // EVIDENCE OF SUCCESS
    // ====================================================================
    println!("\n=== FSV EVIDENCE (TASK-39) ===");
    println!("✓ is_running reflects stepper state (false → true → false)");
    println!("✓ phases: {} oscillators, all in [0, 2pi]", phases.len());
    println!(
        "✓ frequencies: {} values, all positive (Hz)",
        frequencies.len()
    );
    println!("✓ coupling: K = {}", coupling);
    println!("✓ order_parameter: r = {:.6} in [0, 1]", r);
    println!("✓ mean_phase: psi = {:.6} in [0, 2pi]", psi);
    println!("=== FSV TEST PASSED (TASK-39) ===\n");
}

/// TASK-39 Edge Case: get_kuramoto_state without GWT initialized
///
/// Verifies FAIL FAST behavior when Kuramoto network is not initialized.
/// Should return JSON-RPC error with GWT_NOT_INITIALIZED code, not crash.
#[tokio::test]
async fn test_get_kuramoto_state_fails_fast_without_gwt() {
    use crate::handlers::Handlers;
    use context_graph_core::johari::DynDefaultJohariManager;
    use context_graph_core::stubs::{InMemoryTeleologicalStore, StubMultiArrayProvider, StubUtlProcessor};
    use context_graph_core::traits::{MultiArrayEmbeddingProvider, TeleologicalMemoryStore, UtlProcessor};
    use std::sync::Arc;
    use context_graph_core::alignment::DefaultAlignmentCalculator;
    use context_graph_core::purpose::GoalHierarchy;

    println!("\n=== EDGE CASE TEST: get_kuramoto_state without GWT ===");

    // Create handlers WITHOUT GWT (using with_johari_manager constructor)
    // This is a minimal Handlers that has NO Kuramoto network
    let teleological_store: Arc<dyn TeleologicalMemoryStore> =
        Arc::new(InMemoryTeleologicalStore::new());
    let utl_processor: Arc<dyn UtlProcessor> = Arc::new(StubUtlProcessor::new());
    let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> =
        Arc::new(StubMultiArrayProvider::new());
    let alignment_calculator = Arc::new(DefaultAlignmentCalculator::new());
    let goal_hierarchy = Arc::new(parking_lot::RwLock::new(GoalHierarchy::new()));
    let johari_manager: Arc<dyn context_graph_core::johari::JohariTransitionManager> =
        Arc::new(DynDefaultJohariManager::new(Arc::clone(&teleological_store)));

    // Use with_johari_manager - has NO GWT providers (no kuramoto_network)
    // This constructor only takes 6 args and creates stubs for the rest
    let handlers = Handlers::with_johari_manager(
        teleological_store,
        utl_processor,
        multi_array_provider,
        alignment_calculator,
        goal_hierarchy,
        johari_manager,
    );

    // Call get_kuramoto_state - should FAIL FAST
    let request = make_tool_call_request(tool_names::GET_KURAMOTO_STATE, None);
    let response = handlers.dispatch(request).await;

    let response_json = serde_json::to_value(&response).expect("serialize response");

    // VERIFY: Should be a JSON-RPC error
    let error = response_json
        .get("error")
        .expect("Expected error response when GWT not initialized");

    // Verify error code is GWT_NOT_INITIALIZED (-32060)
    let error_code = error["code"].as_i64().expect("error code must be i64");
    assert_eq!(
        error_code, -32060,
        "Error code should be GWT_NOT_INITIALIZED (-32060), got {}",
        error_code
    );

    // Verify error message mentions the issue
    let error_message = error["message"].as_str().expect("error message must be string");
    assert!(
        error_message.contains("Kuramoto") || error_message.contains("not initialized"),
        "Error message should mention Kuramoto or initialization: {}",
        error_message
    );

    println!("EDGE CASE PASS: get_kuramoto_state fails fast with error code {}", error_code);
    println!("Error message: {}", error_message);
    println!("=== EDGE CASE TEST PASSED ===\n");
}
