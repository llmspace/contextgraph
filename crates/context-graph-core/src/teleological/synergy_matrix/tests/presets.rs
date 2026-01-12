//! Predefined constructor tests for SynergyMatrix (TASK-CORE-004).

use crate::teleological::synergy_matrix::{SynergyMatrix, SYNERGY_DIM};

#[test]
fn test_semantic_focused_constructor() {
    let matrix = SynergyMatrix::semantic_focused();

    // Matrix should be valid
    assert!(
        matrix.validate().is_ok(),
        "semantic_focused should produce valid matrix"
    );

    // Verify boosted synergies
    assert!(
        (matrix.get_synergy(0, 4) - 0.95).abs() < f32::EPSILON,
        "E1+E5 should be boosted to 0.95"
    );
    assert!(
        (matrix.get_synergy(0, 10) - 0.95).abs() < f32::EPSILON,
        "E1+E11 should be boosted to 0.95"
    );
    assert!(
        (matrix.get_synergy(0, 11) - 0.95).abs() < f32::EPSILON,
        "E1+E12 should be boosted to 0.95"
    );
    assert!(
        (matrix.get_synergy(4, 10) - 0.95).abs() < f32::EPSILON,
        "E5+E11 should be boosted to 0.95"
    );

    assert!(matrix.is_symmetric(f32::EPSILON));
    assert!(matrix.has_unit_diagonal(f32::EPSILON));
}

#[test]
fn test_code_heavy_constructor() {
    let matrix = SynergyMatrix::code_heavy();

    assert!(
        matrix.validate().is_ok(),
        "code_heavy should produce valid matrix"
    );

    assert!(
        (matrix.get_synergy(5, 3) - 0.95).abs() < f32::EPSILON,
        "E6+E4 should be boosted to 0.95"
    );
    assert!(
        (matrix.get_synergy(5, 6) - 0.95).abs() < f32::EPSILON,
        "E6+E7 should be boosted to 0.95"
    );
    assert!(
        (matrix.get_synergy(5, 7) - 0.95).abs() < f32::EPSILON,
        "E6+E8 should be boosted to 0.95"
    );
    assert!(
        (matrix.get_synergy(5, 12) - 0.95).abs() < f32::EPSILON,
        "E6+E13 should be boosted to 0.95"
    );

    assert!(matrix.is_symmetric(f32::EPSILON));
}

#[test]
fn test_temporal_focused_constructor() {
    let matrix = SynergyMatrix::temporal_focused();

    assert!(
        matrix.validate().is_ok(),
        "temporal_focused should produce valid matrix"
    );

    assert!(
        (matrix.get_synergy(1, 2) - 0.95).abs() < f32::EPSILON,
        "E2+E3 should be boosted to 0.95"
    );
    assert!(
        (matrix.get_synergy(2, 3) - 0.95).abs() < f32::EPSILON,
        "E3+E4 should be boosted to 0.95"
    );
}

#[test]
fn test_causal_reasoning_constructor() {
    let matrix = SynergyMatrix::causal_reasoning();

    assert!(
        matrix.validate().is_ok(),
        "causal_reasoning should produce valid matrix"
    );

    assert!(
        (matrix.get_synergy(3, 2) - 0.95).abs() < f32::EPSILON,
        "E4+E3 should be boosted to 0.95"
    );
    assert!(
        (matrix.get_synergy(3, 5) - 0.95).abs() < f32::EPSILON,
        "E4+E6 should be boosted to 0.95"
    );
    assert!(
        (matrix.get_synergy(3, 6) - 0.95).abs() < f32::EPSILON,
        "E4+E7 should be boosted to 0.95"
    );
    assert!(
        (matrix.get_synergy(3, 10) - 0.95).abs() < f32::EPSILON,
        "E4+E11 should be boosted to 0.95"
    );
    assert!(
        (matrix.get_synergy(3, 11) - 0.95).abs() < f32::EPSILON,
        "E4+E12 should be boosted to 0.95"
    );
}

#[test]
fn test_relational_constructor() {
    let matrix = SynergyMatrix::relational();

    assert!(
        matrix.validate().is_ok(),
        "relational should produce valid matrix"
    );

    assert!(
        (matrix.get_synergy(4, 7) - 0.9).abs() < f32::EPSILON,
        "E5+E8 should be 0.9"
    );
    assert!(
        (matrix.get_synergy(4, 8) - 0.9).abs() < f32::EPSILON,
        "E5+E9 should be 0.9"
    );
    assert!(
        (matrix.get_synergy(7, 8) - 0.9).abs() < f32::EPSILON,
        "E8+E9 should be 0.9"
    );
    assert!(
        (matrix.get_synergy(8, 9) - 0.95).abs() < f32::EPSILON,
        "E9+E10 should be boosted to 0.95"
    );
}

#[test]
fn test_qualitative_constructor() {
    let matrix = SynergyMatrix::qualitative();

    assert!(
        matrix.validate().is_ok(),
        "qualitative should produce valid matrix"
    );

    assert!(
        (matrix.get_synergy(9, 10) - 0.9).abs() < f32::EPSILON,
        "E10+E11 should be 0.9"
    );
    assert!(
        (matrix.get_synergy(8, 9) - 0.95).abs() < f32::EPSILON,
        "E9+E10 should be boosted to 0.95"
    );
}

#[test]
fn test_balanced_constructor() {
    let matrix = SynergyMatrix::balanced();

    assert!(
        matrix.validate().is_ok(),
        "balanced should produce valid matrix"
    );

    // All off-diagonal values should be 0.6
    for i in 0..SYNERGY_DIM {
        for j in 0..SYNERGY_DIM {
            if i == j {
                assert!(
                    (matrix.get_synergy(i, j) - 1.0).abs() < f32::EPSILON,
                    "Diagonal should be 1.0"
                );
            } else {
                assert!(
                    (matrix.get_synergy(i, j) - 0.6).abs() < f32::EPSILON,
                    "Off-diagonal [{}, {}] should be 0.6",
                    i,
                    j
                );
            }
        }
    }
}

#[test]
fn test_identity_constructor() {
    let matrix = SynergyMatrix::identity();

    assert!(
        matrix.validate().is_ok(),
        "identity should produce valid matrix"
    );

    for i in 0..SYNERGY_DIM {
        for j in 0..SYNERGY_DIM {
            if i == j {
                assert!(
                    (matrix.get_synergy(i, j) - 1.0).abs() < f32::EPSILON,
                    "Diagonal should be 1.0"
                );
            } else {
                assert!(
                    matrix.get_synergy(i, j).abs() < f32::EPSILON,
                    "Off-diagonal [{}, {}] should be 0.0",
                    i,
                    j
                );
            }
        }
    }
}

#[test]
fn test_all_predefined_matrices_are_valid() {
    let matrices = [
        ("base", SynergyMatrix::with_base_synergies()),
        ("semantic_focused", SynergyMatrix::semantic_focused()),
        ("code_heavy", SynergyMatrix::code_heavy()),
        ("temporal_focused", SynergyMatrix::temporal_focused()),
        ("causal_reasoning", SynergyMatrix::causal_reasoning()),
        ("relational", SynergyMatrix::relational()),
        ("qualitative", SynergyMatrix::qualitative()),
        ("balanced", SynergyMatrix::balanced()),
        ("identity", SynergyMatrix::identity()),
    ];

    for (name, matrix) in matrices.iter() {
        let result = matrix.validate();
        assert!(
            result.is_ok(),
            "{} matrix validation failed: {:?}",
            name,
            result
        );
        assert!(
            matrix.is_valid(),
            "{} matrix is_valid() returned false",
            name
        );
        assert!(
            matrix.is_symmetric(f32::EPSILON),
            "{} matrix is not symmetric",
            name
        );
        assert!(
            matrix.has_unit_diagonal(f32::EPSILON),
            "{} matrix has non-unity diagonal",
            name
        );
        assert!(
            matrix.values_in_range(),
            "{} matrix has values out of range",
            name
        );
    }
}

#[test]
fn test_predefined_matrix_properties() {
    let semantic = SynergyMatrix::semantic_focused();
    let base = SynergyMatrix::with_base_synergies();

    // E1_Semantic row average should be higher in semantic_focused
    let semantic_e1_avg: f32 = (0..SYNERGY_DIM)
        .filter(|&j| j != 0)
        .map(|j| semantic.get_synergy(0, j))
        .sum::<f32>()
        / 12.0;
    let base_e1_avg: f32 = (0..SYNERGY_DIM)
        .filter(|&j| j != 0)
        .map(|j| base.get_synergy(0, j))
        .sum::<f32>()
        / 12.0;

    assert!(
        semantic_e1_avg > base_e1_avg,
        "semantic_focused E1 average ({}) should be higher than base ({})",
        semantic_e1_avg,
        base_e1_avg
    );

    let code = SynergyMatrix::code_heavy();

    // E6_Code row average should be higher in code_heavy
    let code_e6_avg: f32 = (0..SYNERGY_DIM)
        .filter(|&j| j != 5)
        .map(|j| code.get_synergy(5, j))
        .sum::<f32>()
        / 12.0;
    let base_e6_avg: f32 = (0..SYNERGY_DIM)
        .filter(|&j| j != 5)
        .map(|j| base.get_synergy(5, j))
        .sum::<f32>()
        / 12.0;

    assert!(
        code_e6_avg > base_e6_avg,
        "code_heavy E6 average ({}) should be higher than base ({})",
        code_e6_avg,
        base_e6_avg
    );
}
