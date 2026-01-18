//! Tests for coherence state machine

use super::*;

#[test]
fn test_coherence_state_from_level() {
    assert_eq!(
        CoherenceState::from_level(0.1),
        CoherenceState::Dormant
    );
    assert_eq!(
        CoherenceState::from_level(0.4),
        CoherenceState::Fragmented
    );
    assert_eq!(
        CoherenceState::from_level(0.65),
        CoherenceState::Emerging
    );
    assert_eq!(
        CoherenceState::from_level(0.85),
        CoherenceState::Stable
    );
    assert_eq!(
        CoherenceState::from_level(0.97),
        CoherenceState::Hypersync
    );
}

#[test]
fn test_coherence_state_name() {
    assert_eq!(CoherenceState::Dormant.name(), "DORMANT");
    assert_eq!(CoherenceState::Stable.name(), "STABLE");
    assert_eq!(CoherenceState::Hypersync.name(), "HYPERSYNC");
}

#[tokio::test]
async fn test_state_machine_dormant_to_fragmented() {
    let mut sm = StateMachineManager::new();
    assert_eq!(sm.current_state(), CoherenceState::Dormant);

    sm.update(0.4).await.unwrap();
    assert_eq!(sm.current_state(), CoherenceState::Fragmented);
}

#[tokio::test]
async fn test_state_machine_progression() {
    let mut sm = StateMachineManager::new();

    // Dormant -> Fragmented
    sm.update(0.4).await.unwrap();
    assert_eq!(sm.current_state(), CoherenceState::Fragmented);

    // Fragmented -> Emerging
    sm.update(0.65).await.unwrap();
    assert_eq!(sm.current_state(), CoherenceState::Emerging);

    // Emerging -> Stable
    sm.update(0.85).await.unwrap();
    assert_eq!(sm.current_state(), CoherenceState::Stable);

    // Stable -> Hypersync
    sm.update(0.97).await.unwrap();
    assert_eq!(sm.current_state(), CoherenceState::Hypersync);
}

#[tokio::test]
async fn test_state_machine_regression() {
    let mut sm = StateMachineManager::new();

    // Start stable
    sm.update(0.85).await.unwrap();
    assert!(sm.is_stable());

    // Drop to fragmented
    sm.update(0.4).await.unwrap();
    assert!(sm.is_fragmented());
    assert!(!sm.is_stable());
}

#[tokio::test]
async fn test_state_machine_just_became_stable() {
    let mut sm = StateMachineManager::new();

    sm.update(0.85).await.unwrap();
    assert!(sm.just_became_stable());

    // After 2 seconds, should not be "just" (threshold is 1000ms)
    // Use 2000ms to ensure we're well past the threshold and avoid timing flakiness
    tokio::time::sleep(std::time::Duration::from_millis(2000)).await;
    assert!(!sm.just_became_stable());
}

#[tokio::test]
async fn test_state_machine_hypersync_detection() {
    let mut sm = StateMachineManager::new();

    sm.update(0.97).await.unwrap();
    assert!(sm.is_hypersync());
    assert!(sm.is_stable()); // Hypersync is a form of stable coherence
}

#[test]
fn test_state_machine_time_in_state() {
    let sm = StateMachineManager::new();
    let time = sm.time_in_state();

    assert!(time.num_seconds() >= 0);
}

#[tokio::test]
async fn test_state_machine_last_transition() {
    let mut sm = StateMachineManager::new();

    sm.update(0.85).await.unwrap();
    let trans = sm.last_transition();

    assert!(trans.is_some());
    let t = trans.unwrap();
    assert_eq!(t.from, CoherenceState::Dormant);
    assert_eq!(t.to, CoherenceState::Stable);
}
