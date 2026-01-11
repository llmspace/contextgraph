//! Tests for consciousness state machine

use super::*;

#[test]
fn test_consciousness_state_from_level() {
    assert_eq!(
        ConsciousnessState::from_level(0.1),
        ConsciousnessState::Dormant
    );
    assert_eq!(
        ConsciousnessState::from_level(0.4),
        ConsciousnessState::Fragmented
    );
    assert_eq!(
        ConsciousnessState::from_level(0.65),
        ConsciousnessState::Emerging
    );
    assert_eq!(
        ConsciousnessState::from_level(0.85),
        ConsciousnessState::Conscious
    );
    assert_eq!(
        ConsciousnessState::from_level(0.97),
        ConsciousnessState::Hypersync
    );
}

#[test]
fn test_consciousness_state_name() {
    assert_eq!(ConsciousnessState::Dormant.name(), "DORMANT");
    assert_eq!(ConsciousnessState::Conscious.name(), "CONSCIOUS");
    assert_eq!(ConsciousnessState::Hypersync.name(), "HYPERSYNC");
}

#[tokio::test]
async fn test_state_machine_dormant_to_fragmented() {
    let mut sm = StateMachineManager::new();
    assert_eq!(sm.current_state(), ConsciousnessState::Dormant);

    sm.update(0.4).await.unwrap();
    assert_eq!(sm.current_state(), ConsciousnessState::Fragmented);
}

#[tokio::test]
async fn test_state_machine_progression() {
    let mut sm = StateMachineManager::new();

    // Dormant → Fragmented
    sm.update(0.4).await.unwrap();
    assert_eq!(sm.current_state(), ConsciousnessState::Fragmented);

    // Fragmented → Emerging
    sm.update(0.65).await.unwrap();
    assert_eq!(sm.current_state(), ConsciousnessState::Emerging);

    // Emerging → Conscious
    sm.update(0.85).await.unwrap();
    assert_eq!(sm.current_state(), ConsciousnessState::Conscious);

    // Conscious → Hypersync
    sm.update(0.97).await.unwrap();
    assert_eq!(sm.current_state(), ConsciousnessState::Hypersync);
}

#[tokio::test]
async fn test_state_machine_regression() {
    let mut sm = StateMachineManager::new();

    // Start conscious
    sm.update(0.85).await.unwrap();
    assert!(sm.is_conscious());

    // Drop to fragmented
    sm.update(0.4).await.unwrap();
    assert!(sm.is_fragmented());
    assert!(!sm.is_conscious());
}

#[tokio::test]
async fn test_state_machine_just_became_conscious() {
    let mut sm = StateMachineManager::new();

    sm.update(0.85).await.unwrap();
    assert!(sm.just_became_conscious());

    // After 2 seconds, should not be "just" (threshold is 1000ms)
    // Use 2000ms to ensure we're well past the threshold and avoid timing flakiness
    tokio::time::sleep(std::time::Duration::from_millis(2000)).await;
    assert!(!sm.just_became_conscious());
}

#[tokio::test]
async fn test_state_machine_hypersync_detection() {
    let mut sm = StateMachineManager::new();

    sm.update(0.97).await.unwrap();
    assert!(sm.is_hypersync());
    assert!(sm.is_conscious()); // Hypersync is a form of consciousness
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
    assert_eq!(t.from, ConsciousnessState::Dormant);
    assert_eq!(t.to, ConsciousnessState::Conscious);
}
