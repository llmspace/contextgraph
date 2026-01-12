//! Basic constructor and accessor tests for SynergyMatrix.

use crate::teleological::synergy_matrix::{SynergyMatrix, SYNERGY_DIM};

#[test]
fn test_synergy_matrix_new() {
    let matrix = SynergyMatrix::new();

    // Diagonal should be 1.0
    for i in 0..SYNERGY_DIM {
        assert!(
            (matrix.values[i][i] - 1.0).abs() < f32::EPSILON,
            "Diagonal [{i}][{i}] should be 1.0"
        );
    }

    // Off-diagonal should be 0.0
    for i in 0..SYNERGY_DIM {
        for j in 0..SYNERGY_DIM {
            if i != j {
                assert!(
                    matrix.values[i][j].abs() < f32::EPSILON,
                    "Off-diagonal [{i}][{j}] should be 0.0"
                );
            }
        }
    }
}

#[test]
fn test_synergy_matrix_with_base_synergies() {
    let matrix = SynergyMatrix::with_base_synergies();

    // Verify diagonal is 1.0
    for i in 0..SYNERGY_DIM {
        assert!(
            (matrix.values[i][i] - 1.0).abs() < f32::EPSILON,
            "Diagonal [{i}][{i}] should be 1.0"
        );
    }

    // Verify some known synergy values from teleoplan.md
    // E1_Semantic + E5_Analogical = strong (0.9)
    assert!((matrix.get_synergy(0, 4) - 0.9).abs() < f32::EPSILON);
    // E2_Episodic + E3_Temporal = strong (0.9)
    assert!((matrix.get_synergy(1, 2) - 0.9).abs() < f32::EPSILON);
    // E6_Code + E13_Sparse = strong (0.9)
    assert!((matrix.get_synergy(5, 12) - 0.9).abs() < f32::EPSILON);
}

#[test]
fn test_synergy_matrix_symmetry() {
    let matrix = SynergyMatrix::with_base_synergies();

    for i in 0..SYNERGY_DIM {
        for j in 0..SYNERGY_DIM {
            assert!(
                (matrix.values[i][j] - matrix.values[j][i]).abs() < f32::EPSILON,
                "Matrix should be symmetric: [{i}][{j}] != [{j}][{i}]"
            );
        }
    }

    assert!(matrix.is_symmetric(f32::EPSILON));
}

#[test]
fn test_synergy_matrix_values_in_range() {
    let matrix = SynergyMatrix::with_base_synergies();

    for i in 0..SYNERGY_DIM {
        for j in 0..SYNERGY_DIM {
            let v = matrix.values[i][j];
            assert!(
                (0.0..=1.0).contains(&v),
                "Value [{i}][{j}] = {v} out of range [0.0, 1.0]"
            );
        }
    }

    assert!(matrix.values_in_range());
}

#[test]
fn test_synergy_matrix_get_set_synergy() {
    let mut matrix = SynergyMatrix::new();

    matrix.set_synergy(0, 5, 0.7);

    assert!((matrix.get_synergy(0, 5) - 0.7).abs() < f32::EPSILON);
    assert!((matrix.get_synergy(5, 0) - 0.7).abs() < f32::EPSILON); // Symmetric
}

#[test]
fn test_synergy_matrix_default() {
    let matrix = SynergyMatrix::default();

    // Default should use base synergies
    assert!((matrix.get_synergy(0, 4) - 0.9).abs() < f32::EPSILON);
}
