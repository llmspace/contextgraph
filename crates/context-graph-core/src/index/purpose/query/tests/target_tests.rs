//! Tests for PurposeQueryTarget.

use uuid::Uuid;

use super::helpers::create_purpose_vector;
use crate::index::purpose::query::PurposeQueryTarget;

#[test]
fn test_purpose_query_target_vector() {
    let pv = create_purpose_vector(0.5);
    let target = PurposeQueryTarget::vector(pv.clone());

    match target {
        PurposeQueryTarget::Vector(v) => {
            assert_eq!(v.alignments, pv.alignments);
        }
        _ => panic!("Expected Vector variant"),
    }

    println!("[VERIFIED] PurposeQueryTarget::vector creates Vector variant");
}

#[test]
fn test_purpose_query_target_pattern_valid() {
    let target = PurposeQueryTarget::pattern(5, 0.7).unwrap();

    match target {
        PurposeQueryTarget::Pattern {
            min_cluster_size,
            coherence_threshold,
        } => {
            assert_eq!(min_cluster_size, 5);
            assert!((coherence_threshold - 0.7).abs() < f32::EPSILON);
        }
        _ => panic!("Expected Pattern variant"),
    }

    println!("[VERIFIED] PurposeQueryTarget::pattern creates Pattern variant with valid params");
}

#[test]
fn test_purpose_query_target_pattern_boundary_values() {
    // Test coherence_threshold = 0.0
    let target = PurposeQueryTarget::pattern(1, 0.0).unwrap();
    if let PurposeQueryTarget::Pattern {
        coherence_threshold,
        ..
    } = target
    {
        assert_eq!(coherence_threshold, 0.0);
    }

    // Test coherence_threshold = 1.0
    let target = PurposeQueryTarget::pattern(1, 1.0).unwrap();
    if let PurposeQueryTarget::Pattern {
        coherence_threshold,
        ..
    } = target
    {
        assert_eq!(coherence_threshold, 1.0);
    }

    println!("[VERIFIED] PurposeQueryTarget::pattern accepts boundary values 0.0 and 1.0");
}

#[test]
fn test_purpose_query_target_pattern_invalid_cluster_size() {
    let result = PurposeQueryTarget::pattern(0, 0.5);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("min_cluster_size"));

    println!(
        "[VERIFIED] FAIL FAST: PurposeQueryTarget::pattern rejects min_cluster_size=0: {}",
        msg
    );
}

#[test]
fn test_purpose_query_target_pattern_invalid_coherence_over() {
    let result = PurposeQueryTarget::pattern(5, 1.5);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("coherence_threshold"));
    assert!(msg.contains("1.5"));

    println!(
        "[VERIFIED] FAIL FAST: PurposeQueryTarget::pattern rejects coherence_threshold=1.5: {}",
        msg
    );
}

#[test]
fn test_purpose_query_target_pattern_invalid_coherence_under() {
    let result = PurposeQueryTarget::pattern(5, -0.1);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("coherence_threshold"));

    println!(
        "[VERIFIED] FAIL FAST: PurposeQueryTarget::pattern rejects coherence_threshold=-0.1: {}",
        msg
    );
}

#[test]
fn test_purpose_query_target_from_memory() {
    let id = Uuid::new_v4();
    let target = PurposeQueryTarget::from_memory(id);

    match target {
        PurposeQueryTarget::FromMemory(mem_id) => {
            assert_eq!(mem_id, id);
        }
        _ => panic!("Expected FromMemory variant"),
    }

    println!("[VERIFIED] PurposeQueryTarget::from_memory creates FromMemory variant");
}

#[test]
fn test_purpose_query_target_requires_memory_lookup() {
    let pv = create_purpose_vector(0.5);

    assert!(!PurposeQueryTarget::vector(pv).requires_memory_lookup());
    assert!(!PurposeQueryTarget::pattern(5, 0.7)
        .unwrap()
        .requires_memory_lookup());
    assert!(PurposeQueryTarget::from_memory(Uuid::new_v4()).requires_memory_lookup());

    println!("[VERIFIED] requires_memory_lookup returns true only for FromMemory");
}
