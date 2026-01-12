//! Kuramoto Synchronization FSV Tests
//!
//! Verifies the 13-oscillator network state returns REAL data.

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
