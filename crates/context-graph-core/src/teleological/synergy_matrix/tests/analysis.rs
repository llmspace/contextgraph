//! Analysis function tests for SynergyMatrix.

use crate::teleological::synergy_matrix::{
    SynergyMatrix, CROSS_CORRELATION_COUNT, SYNERGY_DIM,
};

#[test]
fn test_synergy_matrix_average_synergy() {
    let matrix = SynergyMatrix::with_base_synergies();
    let avg = matrix.average_synergy();

    // Average should be in (0, 1)
    assert!(avg > 0.0 && avg < 1.0);

    // Count unique pairs: 13 * 12 / 2 = 78
    // Verify by manual calculation
    let mut sum = 0.0f32;
    for i in 0..SYNERGY_DIM {
        for j in (i + 1)..SYNERGY_DIM {
            sum += matrix.values[i][j];
        }
    }
    let expected = sum / CROSS_CORRELATION_COUNT as f32;

    assert!((avg - expected).abs() < f32::EPSILON);
}

#[test]
fn test_synergy_matrix_high_synergy_pairs() {
    let matrix = SynergyMatrix::with_base_synergies();
    let high_pairs = matrix.high_synergy_pairs(0.9);

    // Should include (0, 4) = E1_Semantic + E5_Analogical
    assert!(high_pairs.contains(&(0, 4)));
    // Should include (1, 2) = E2_Episodic + E3_Temporal
    assert!(high_pairs.contains(&(1, 2)));
}

#[test]
fn test_synergy_matrix_to_cross_correlations() {
    let matrix = SynergyMatrix::with_base_synergies();
    let cross = matrix.to_cross_correlations();

    assert_eq!(cross.len(), CROSS_CORRELATION_COUNT);

    // Verify first few values
    // (0,1) = E1_Semantic + E2_Episodic = 0.6
    assert!((cross[0] - 0.6).abs() < f32::EPSILON);
    // (0,2) = E1_Semantic + E3_Temporal = 0.3
    assert!((cross[1] - 0.3).abs() < f32::EPSILON);
}

#[test]
fn test_synergy_matrix_flat_to_indices() {
    // First pair (0, 1)
    assert_eq!(SynergyMatrix::flat_to_indices(0), (0, 1));
    // Second pair (0, 2)
    assert_eq!(SynergyMatrix::flat_to_indices(1), (0, 2));
    // Last of first row (0, 12)
    assert_eq!(SynergyMatrix::flat_to_indices(11), (0, 12));
    // First of second row (1, 2)
    assert_eq!(SynergyMatrix::flat_to_indices(12), (1, 2));
    // Last pair (11, 12)
    assert_eq!(SynergyMatrix::flat_to_indices(77), (11, 12));
}

#[test]
fn test_synergy_matrix_indices_to_flat() {
    assert_eq!(SynergyMatrix::indices_to_flat(0, 1), 0);
    assert_eq!(SynergyMatrix::indices_to_flat(0, 2), 1);
    assert_eq!(SynergyMatrix::indices_to_flat(0, 12), 11);
    assert_eq!(SynergyMatrix::indices_to_flat(1, 2), 12);
    assert_eq!(SynergyMatrix::indices_to_flat(11, 12), 77);
}

#[test]
fn test_synergy_matrix_roundtrip_indices() {
    // Test all 78 pairs roundtrip correctly
    for flat in 0..CROSS_CORRELATION_COUNT {
        let (i, j) = SynergyMatrix::flat_to_indices(flat);
        let back = SynergyMatrix::indices_to_flat(i, j);
        assert_eq!(
            flat, back,
            "Roundtrip failed: {} -> ({}, {}) -> {}",
            flat, i, j, back
        );
    }
}

#[test]
fn test_synergy_matrix_weights() {
    let mut matrix = SynergyMatrix::new();

    // Default weight is 1.0
    assert!((matrix.get_weight(0, 1) - 1.0).abs() < f32::EPSILON);

    // Set weight
    matrix.set_weight(0, 1, 2.5);
    assert!((matrix.get_weight(0, 1) - 2.5).abs() < f32::EPSILON);
    assert!((matrix.get_weight(1, 0) - 2.5).abs() < f32::EPSILON); // Symmetric
}

#[test]
fn test_synergy_matrix_weighted_synergy() {
    let mut matrix = SynergyMatrix::with_base_synergies();
    matrix.set_weight(0, 4, 2.0);

    // E1_Semantic + E5_Analogical = 0.9, weight = 2.0
    let weighted = matrix.get_weighted_synergy(0, 4);
    assert!((weighted - 1.8).abs() < f32::EPSILON);
}

#[test]
fn test_synergy_matrix_serialization() {
    let matrix = SynergyMatrix::with_base_synergies();
    let json = serde_json::to_string(&matrix).unwrap();
    let deserialized: SynergyMatrix = serde_json::from_str(&json).unwrap();

    assert_eq!(matrix.sample_count, deserialized.sample_count);
    assert!((matrix.get_synergy(0, 4) - deserialized.get_synergy(0, 4)).abs() < f32::EPSILON);
}
