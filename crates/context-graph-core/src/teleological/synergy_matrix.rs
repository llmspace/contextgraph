//! 13x13 Cross-embedding synergy matrix for teleological fusion.
//!
//! The synergy matrix captures the strength of relationships between different
//! embedding spaces. High synergy pairs should have their cross-correlations
//! amplified in teleological fusion.
//!
//! From teleoplan.md:
//! - Diagonal is always 1.0 (self-synergy)
//! - Matrix must be symmetric
//! - Values in [0.0, 1.0]
//! - Base synergies: weak (0.3), moderate (0.6), strong (0.9)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Number of embedding dimensions in the synergy matrix.
pub const SYNERGY_DIM: usize = 13;

/// Number of unique cross-embedding pairs (upper triangle excluding diagonal).
/// Formula: n * (n - 1) / 2 = 13 * 12 / 2 = 78
pub const CROSS_CORRELATION_COUNT: usize = 78;

/// 13x13 cross-embedding synergy matrix per teleoplan.md.
///
/// Captures the strength of relationships between different embedding spaces.
/// Used for weighting cross-correlations in teleological fusion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynergyMatrix {
    /// 13x13 matrix of synergy values, symmetric with diagonal = 1.0
    pub values: [[f32; SYNERGY_DIM]; SYNERGY_DIM],
    /// Per-cell weights for adaptive learning
    pub weights: [[f32; SYNERGY_DIM]; SYNERGY_DIM],
    /// When this matrix was computed/updated
    pub computed_at: DateTime<Utc>,
    /// Number of samples used to compute/refine the matrix
    pub sample_count: u64,
}

impl SynergyMatrix {
    /// Base synergies from teleoplan.md synergy matrix.
    ///
    /// Values: weak (0.3), moderate (0.6), strong (0.9)
    ///
    /// Index mapping:
    /// - 0: E1_Semantic
    /// - 1: E2_Episodic
    /// - 2: E3_Temporal
    /// - 3: E4_Causal
    /// - 4: E5_Analogical
    /// - 5: E6_Code
    /// - 6: E7_Procedural
    /// - 7: E8_Spatial
    /// - 8: E9_Social
    /// - 9: E10_Emotional
    /// - 10: E11_Abstract
    /// - 11: E12_Factual
    /// - 12: E13_Sparse
    #[rustfmt::skip]
    pub const BASE_SYNERGIES: [[f32; SYNERGY_DIM]; SYNERGY_DIM] = [
        // E1_Semantic
        [1.0, 0.6, 0.3, 0.6, 0.9, 0.6, 0.6, 0.3, 0.3, 0.6, 0.9, 0.9, 0.6],
        // E2_Episodic
        [0.6, 1.0, 0.9, 0.6, 0.3, 0.3, 0.3, 0.6, 0.9, 0.9, 0.3, 0.6, 0.3],
        // E3_Temporal
        [0.3, 0.9, 1.0, 0.9, 0.3, 0.3, 0.9, 0.3, 0.6, 0.3, 0.3, 0.6, 0.3],
        // E4_Causal
        [0.6, 0.6, 0.9, 1.0, 0.6, 0.9, 0.9, 0.3, 0.6, 0.3, 0.9, 0.9, 0.3],
        // E5_Analogical
        [0.9, 0.3, 0.3, 0.6, 1.0, 0.6, 0.3, 0.6, 0.6, 0.6, 0.9, 0.3, 0.3],
        // E6_Code
        [0.6, 0.3, 0.3, 0.9, 0.6, 1.0, 0.9, 0.9, 0.3, 0.3, 0.6, 0.6, 0.9],
        // E7_Procedural
        [0.6, 0.3, 0.9, 0.9, 0.3, 0.9, 1.0, 0.6, 0.3, 0.3, 0.3, 0.6, 0.6],
        // E8_Spatial
        [0.3, 0.6, 0.3, 0.3, 0.6, 0.9, 0.6, 1.0, 0.6, 0.3, 0.6, 0.3, 0.3],
        // E9_Social
        [0.3, 0.9, 0.6, 0.6, 0.6, 0.3, 0.3, 0.6, 1.0, 0.9, 0.3, 0.6, 0.3],
        // E10_Emotional
        [0.6, 0.9, 0.3, 0.3, 0.6, 0.3, 0.3, 0.3, 0.9, 1.0, 0.6, 0.3, 0.3],
        // E11_Abstract
        [0.9, 0.3, 0.3, 0.9, 0.9, 0.6, 0.3, 0.6, 0.3, 0.6, 1.0, 0.6, 0.3],
        // E12_Factual
        [0.9, 0.6, 0.6, 0.9, 0.3, 0.6, 0.6, 0.3, 0.6, 0.3, 0.6, 1.0, 0.9],
        // E13_Sparse
        [0.6, 0.3, 0.3, 0.3, 0.3, 0.9, 0.6, 0.3, 0.3, 0.3, 0.3, 0.9, 1.0],
    ];

    /// Create a new empty synergy matrix with identity diagonal.
    ///
    /// All synergy values are 0.0 except diagonal which is 1.0.
    pub fn new() -> Self {
        let mut values = [[0.0f32; SYNERGY_DIM]; SYNERGY_DIM];
        let mut weights = [[1.0f32; SYNERGY_DIM]; SYNERGY_DIM];

        // Set diagonal to 1.0
        for i in 0..SYNERGY_DIM {
            values[i][i] = 1.0;
            weights[i][i] = 1.0;
        }

        Self {
            values,
            weights,
            computed_at: Utc::now(),
            sample_count: 0,
        }
    }

    /// Create a synergy matrix initialized with base synergies from teleoplan.md.
    pub fn with_base_synergies() -> Self {
        Self {
            values: Self::BASE_SYNERGIES,
            weights: [[1.0f32; SYNERGY_DIM]; SYNERGY_DIM],
            computed_at: Utc::now(),
            sample_count: 0,
        }
    }

    /// Get synergy value between embeddings i and j.
    ///
    /// # Panics
    ///
    /// Panics if `i >= SYNERGY_DIM` or `j >= SYNERGY_DIM` (FAIL FAST).
    #[inline]
    pub fn get_synergy(&self, i: usize, j: usize) -> f32 {
        assert!(
            i < SYNERGY_DIM,
            "FAIL FAST: synergy index i={} out of bounds (max {})",
            i,
            SYNERGY_DIM - 1
        );
        assert!(
            j < SYNERGY_DIM,
            "FAIL FAST: synergy index j={} out of bounds (max {})",
            j,
            SYNERGY_DIM - 1
        );
        self.values[i][j]
    }

    /// Set synergy value between embeddings i and j.
    ///
    /// Automatically maintains symmetry (sets both [i][j] and [j][i]).
    ///
    /// # Panics
    ///
    /// - Panics if `i >= SYNERGY_DIM` or `j >= SYNERGY_DIM` (FAIL FAST)
    /// - Panics if `value < 0.0` or `value > 1.0` (FAIL FAST)
    /// - Panics if attempting to set diagonal to non-1.0 value (FAIL FAST)
    #[inline]
    pub fn set_synergy(&mut self, i: usize, j: usize, value: f32) {
        assert!(
            i < SYNERGY_DIM,
            "FAIL FAST: synergy index i={} out of bounds (max {})",
            i,
            SYNERGY_DIM - 1
        );
        assert!(
            j < SYNERGY_DIM,
            "FAIL FAST: synergy index j={} out of bounds (max {})",
            j,
            SYNERGY_DIM - 1
        );
        assert!(
            (0.0..=1.0).contains(&value),
            "FAIL FAST: synergy value {} must be in [0.0, 1.0]",
            value
        );
        if i == j {
            assert!(
                (value - 1.0).abs() < f32::EPSILON,
                "FAIL FAST: diagonal synergy must be 1.0, got {}",
                value
            );
        }

        self.values[i][j] = value;
        self.values[j][i] = value; // Maintain symmetry
        self.computed_at = Utc::now();
    }

    /// Get weight for synergy between embeddings i and j.
    ///
    /// # Panics
    ///
    /// Panics if `i >= SYNERGY_DIM` or `j >= SYNERGY_DIM` (FAIL FAST).
    #[inline]
    pub fn get_weight(&self, i: usize, j: usize) -> f32 {
        assert!(
            i < SYNERGY_DIM,
            "FAIL FAST: weight index i={} out of bounds (max {})",
            i,
            SYNERGY_DIM - 1
        );
        assert!(
            j < SYNERGY_DIM,
            "FAIL FAST: weight index j={} out of bounds (max {})",
            j,
            SYNERGY_DIM - 1
        );
        self.weights[i][j]
    }

    /// Set weight for synergy between embeddings i and j.
    ///
    /// Automatically maintains symmetry.
    ///
    /// # Panics
    ///
    /// - Panics if indices out of bounds (FAIL FAST)
    /// - Panics if weight is negative (FAIL FAST)
    #[inline]
    pub fn set_weight(&mut self, i: usize, j: usize, weight: f32) {
        assert!(
            i < SYNERGY_DIM,
            "FAIL FAST: weight index i={} out of bounds (max {})",
            i,
            SYNERGY_DIM - 1
        );
        assert!(
            j < SYNERGY_DIM,
            "FAIL FAST: weight index j={} out of bounds (max {})",
            j,
            SYNERGY_DIM - 1
        );
        assert!(
            weight >= 0.0,
            "FAIL FAST: weight {} must be non-negative",
            weight
        );

        self.weights[i][j] = weight;
        self.weights[j][i] = weight; // Maintain symmetry
    }

    /// Check if the matrix is symmetric within tolerance.
    pub fn is_symmetric(&self, tolerance: f32) -> bool {
        for i in 0..SYNERGY_DIM {
            for j in (i + 1)..SYNERGY_DIM {
                if (self.values[i][j] - self.values[j][i]).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Check if diagonal values are all 1.0 within tolerance.
    pub fn has_unit_diagonal(&self, tolerance: f32) -> bool {
        for i in 0..SYNERGY_DIM {
            if (self.values[i][i] - 1.0).abs() > tolerance {
                return false;
            }
        }
        true
    }

    /// Check if all values are in [0.0, 1.0].
    pub fn values_in_range(&self) -> bool {
        for row in &self.values {
            for &value in row {
                if !(0.0..=1.0).contains(&value) {
                    return false;
                }
            }
        }
        true
    }

    /// Validate the matrix satisfies all invariants.
    ///
    /// # Panics
    ///
    /// Panics if any invariant is violated (FAIL FAST).
    pub fn validate(&self) {
        assert!(
            self.is_symmetric(f32::EPSILON),
            "FAIL FAST: synergy matrix must be symmetric"
        );
        assert!(
            self.has_unit_diagonal(f32::EPSILON),
            "FAIL FAST: synergy matrix diagonal must be 1.0"
        );
        assert!(
            self.values_in_range(),
            "FAIL FAST: synergy values must be in [0.0, 1.0]"
        );
    }

    /// Get weighted synergy value (value * weight).
    #[inline]
    pub fn get_weighted_synergy(&self, i: usize, j: usize) -> f32 {
        self.get_synergy(i, j) * self.get_weight(i, j)
    }

    /// Compute average synergy across all pairs (excluding diagonal).
    pub fn average_synergy(&self) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0;

        for i in 0..SYNERGY_DIM {
            for j in (i + 1)..SYNERGY_DIM {
                sum += self.values[i][j];
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f32
        } else {
            0.0
        }
    }

    /// Get indices of high synergy pairs (value >= threshold).
    pub fn high_synergy_pairs(&self, threshold: f32) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..SYNERGY_DIM {
            for j in (i + 1)..SYNERGY_DIM {
                if self.values[i][j] >= threshold {
                    pairs.push((i, j));
                }
            }
        }
        pairs
    }

    /// Flatten upper triangle to cross-correlation array.
    ///
    /// Returns 78 values corresponding to unique pairs (i, j) where i < j.
    /// Order: (0,1), (0,2), ..., (0,12), (1,2), (1,3), ..., (11,12)
    pub fn to_cross_correlations(&self) -> [f32; CROSS_CORRELATION_COUNT] {
        let mut result = [0.0f32; CROSS_CORRELATION_COUNT];
        let mut idx = 0;

        for i in 0..SYNERGY_DIM {
            for j in (i + 1)..SYNERGY_DIM {
                result[idx] = self.values[i][j];
                idx += 1;
            }
        }

        result
    }

    /// Convert flat index (0-77) to matrix indices (i, j).
    ///
    /// # Panics
    ///
    /// Panics if `flat_idx >= CROSS_CORRELATION_COUNT` (FAIL FAST).
    pub fn flat_to_indices(flat_idx: usize) -> (usize, usize) {
        assert!(
            flat_idx < CROSS_CORRELATION_COUNT,
            "FAIL FAST: flat index {} out of bounds (max {})",
            flat_idx,
            CROSS_CORRELATION_COUNT - 1
        );

        // Inverse of triangular number formula
        let mut i = 0;
        let mut offset = 0;

        while offset + (SYNERGY_DIM - 1 - i) <= flat_idx {
            offset += SYNERGY_DIM - 1 - i;
            i += 1;
        }

        let j = flat_idx - offset + i + 1;
        (i, j)
    }

    /// Convert matrix indices (i, j) to flat index (0-77).
    ///
    /// # Panics
    ///
    /// Panics if indices out of bounds or i >= j (FAIL FAST).
    pub fn indices_to_flat(i: usize, j: usize) -> usize {
        assert!(
            i < SYNERGY_DIM && j < SYNERGY_DIM,
            "FAIL FAST: indices ({}, {}) out of bounds",
            i,
            j
        );
        assert!(i < j, "FAIL FAST: i ({}) must be less than j ({})", i, j);

        let mut flat_idx = 0;
        for row in 0..i {
            flat_idx += SYNERGY_DIM - 1 - row;
        }
        flat_idx + (j - i - 1)
    }
}

impl Default for SynergyMatrix {
    fn default() -> Self {
        Self::with_base_synergies()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        println!("[PASS] SynergyMatrix::new creates identity-diagonal matrix");
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

        println!("[PASS] SynergyMatrix::with_base_synergies matches teleoplan.md");
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
        println!("[PASS] BASE_SYNERGIES matrix is symmetric");
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
        println!("[PASS] All synergy values in [0.0, 1.0]");
    }

    #[test]
    fn test_synergy_matrix_get_set_synergy() {
        let mut matrix = SynergyMatrix::new();

        matrix.set_synergy(0, 5, 0.7);

        assert!((matrix.get_synergy(0, 5) - 0.7).abs() < f32::EPSILON);
        assert!((matrix.get_synergy(5, 0) - 0.7).abs() < f32::EPSILON); // Symmetric

        println!("[PASS] set_synergy maintains symmetry");
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_synergy_matrix_get_out_of_bounds_i() {
        let matrix = SynergyMatrix::new();
        let _ = matrix.get_synergy(13, 0);
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_synergy_matrix_get_out_of_bounds_j() {
        let matrix = SynergyMatrix::new();
        let _ = matrix.get_synergy(0, 13);
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_synergy_matrix_set_value_too_high() {
        let mut matrix = SynergyMatrix::new();
        matrix.set_synergy(0, 1, 1.5);
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_synergy_matrix_set_value_negative() {
        let mut matrix = SynergyMatrix::new();
        matrix.set_synergy(0, 1, -0.1);
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_synergy_matrix_set_diagonal_not_one() {
        let mut matrix = SynergyMatrix::new();
        matrix.set_synergy(3, 3, 0.5);
    }

    #[test]
    fn test_synergy_matrix_validate() {
        let matrix = SynergyMatrix::with_base_synergies();
        matrix.validate(); // Should not panic

        println!("[PASS] validate() passes for valid matrix");
    }

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
        println!("[PASS] average_synergy = {avg:.4}");
    }

    #[test]
    fn test_synergy_matrix_high_synergy_pairs() {
        let matrix = SynergyMatrix::with_base_synergies();
        let high_pairs = matrix.high_synergy_pairs(0.9);

        // Should include (0, 4) = E1_Semantic + E5_Analogical
        assert!(high_pairs.contains(&(0, 4)));
        // Should include (1, 2) = E2_Episodic + E3_Temporal
        assert!(high_pairs.contains(&(1, 2)));

        println!(
            "[PASS] high_synergy_pairs(0.9) found {} pairs",
            high_pairs.len()
        );
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

        println!("[PASS] to_cross_correlations produces 78 values");
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

        println!("[PASS] flat_to_indices conversions correct");
    }

    #[test]
    fn test_synergy_matrix_indices_to_flat() {
        assert_eq!(SynergyMatrix::indices_to_flat(0, 1), 0);
        assert_eq!(SynergyMatrix::indices_to_flat(0, 2), 1);
        assert_eq!(SynergyMatrix::indices_to_flat(0, 12), 11);
        assert_eq!(SynergyMatrix::indices_to_flat(1, 2), 12);
        assert_eq!(SynergyMatrix::indices_to_flat(11, 12), 77);

        println!("[PASS] indices_to_flat conversions correct");
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

        println!("[PASS] All 78 index roundtrips correct");
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_synergy_matrix_flat_to_indices_out_of_bounds() {
        let _ = SynergyMatrix::flat_to_indices(78);
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_synergy_matrix_indices_to_flat_invalid() {
        let _ = SynergyMatrix::indices_to_flat(5, 3); // i must be < j
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

        println!("[PASS] Weight get/set works correctly");
    }

    #[test]
    fn test_synergy_matrix_weighted_synergy() {
        let mut matrix = SynergyMatrix::with_base_synergies();
        matrix.set_weight(0, 4, 2.0);

        // E1_Semantic + E5_Analogical = 0.9, weight = 2.0
        let weighted = matrix.get_weighted_synergy(0, 4);
        assert!((weighted - 1.8).abs() < f32::EPSILON);

        println!("[PASS] get_weighted_synergy = {weighted}");
    }

    #[test]
    fn test_synergy_matrix_serialization() {
        let matrix = SynergyMatrix::with_base_synergies();
        let json = serde_json::to_string(&matrix).unwrap();
        let deserialized: SynergyMatrix = serde_json::from_str(&json).unwrap();

        assert_eq!(matrix.sample_count, deserialized.sample_count);
        assert!((matrix.get_synergy(0, 4) - deserialized.get_synergy(0, 4)).abs() < f32::EPSILON);

        println!("[PASS] Serialization roundtrip successful");
    }

    #[test]
    fn test_synergy_matrix_default() {
        let matrix = SynergyMatrix::default();

        // Default should use base synergies
        assert!((matrix.get_synergy(0, 4) - 0.9).abs() < f32::EPSILON);

        println!("[PASS] Default uses base synergies");
    }
}
