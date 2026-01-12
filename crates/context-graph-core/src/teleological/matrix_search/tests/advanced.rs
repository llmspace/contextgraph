//! Advanced tests for teleological matrix search.
//!
//! Tests for collection search, clustering, centroids, and comprehensive comparison.

use crate::teleological::comparison_error::ComparisonValidationError;
use crate::teleological::matrix_search::{
    ComponentWeights, MatrixSearchConfig, TeleologicalMatrixSearch,
};

use super::basic::{make_test_vector, make_varied_test_vector};

#[test]
fn test_matrix_search_collection() {
    let search = TeleologicalMatrixSearch::new();
    let query = make_test_vector(0.8, 0.7);

    let candidates = vec![
        make_test_vector(0.8, 0.7),
        make_varied_test_vector(200),
        make_varied_test_vector(500),
    ];

    let results = search.search(&query, &candidates);

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, 0);
    assert!(
        results[0].1 >= results[1].1,
        "First result should have highest similarity"
    );
}

#[test]
fn test_matrix_search_with_threshold() {
    let config = MatrixSearchConfig {
        min_similarity: 0.5,
        ..Default::default()
    };
    let search = TeleologicalMatrixSearch::with_config(config);

    let query = make_test_vector(0.8, 0.7);
    let candidates = vec![make_test_vector(0.8, 0.7), make_test_vector(0.1, 0.1)];

    let results = search.search(&query, &candidates);

    assert!(!results.is_empty());
    for (_, sim) in &results {
        assert!(*sim >= 0.5, "All results should be above threshold");
    }
}

#[test]
fn test_pairwise_similarity_matrix() {
    let search = TeleologicalMatrixSearch::new();
    let vectors = vec![
        make_test_vector(0.8, 0.6),
        make_test_vector(0.7, 0.5),
        make_test_vector(0.6, 0.4),
    ];

    let matrix = search.pairwise_similarity_matrix(&vectors);

    assert_eq!(matrix.len(), 3);
    assert_eq!(matrix[0].len(), 3);

    for (i, row) in matrix.iter().enumerate() {
        assert!((row[i] - 1.0).abs() < 0.01);
    }

    for (i, row_i) in matrix.iter().enumerate() {
        for (j, &val) in row_i.iter().enumerate() {
            assert!((val - matrix[j][i]).abs() < 0.001);
        }
    }
}

#[test]
fn test_find_clusters() {
    let search = TeleologicalMatrixSearch::new();
    let vectors = vec![
        make_test_vector(0.9, 0.8),
        make_test_vector(0.9, 0.8),
        make_varied_test_vector(100),
        make_varied_test_vector(500),
    ];

    let clusters = search.find_clusters(&vectors, 0.99);
    assert!(!clusters.is_empty(), "Should find at least 1 cluster");
}

#[test]
fn test_compute_centroid() {
    let search = TeleologicalMatrixSearch::new();
    let vectors = vec![make_test_vector(0.8, 0.6), make_test_vector(0.6, 0.4)];

    let centroid = search.compute_centroid(&vectors);

    assert!((centroid.purpose_vector.alignments[0] - 0.7).abs() < 0.01);
    assert!((centroid.cross_correlations[0] - 0.5).abs() < 0.01);
}

#[test]
fn test_comprehensive_comparison() {
    let search = TeleologicalMatrixSearch::new();
    let tv1 = make_test_vector(0.8, 0.6);
    let tv2 = make_test_vector(0.7, 0.5);

    let comp = search.comprehensive_comparison(&tv1, &tv2);

    assert!(comp.full.overall > 0.0);
    assert!(comp.purpose_only > 0.0);
    assert!(comp.correlations_only > 0.0);
    assert!(comp.groups_only > 0.0);
    assert!(!comp.per_group.is_empty());
    assert!(comp.per_embedder_pattern.iter().all(|&v| v > 0.0));
}

#[test]
fn test_component_weights_validation() {
    let mut weights = ComponentWeights::default();
    assert!(
        weights.validate().is_ok(),
        "Default weights should sum to 1.0"
    );
    assert!(weights.is_valid(), "Default weights should be valid");

    weights.purpose_vector = 0.5;
    let err = weights.validate();
    assert!(err.is_err(), "Modified weights should not sum to 1.0");

    match err {
        Err(ComparisonValidationError::WeightsNotNormalized { actual_sum, .. }) => {
            assert!((actual_sum - 1.1).abs() < 0.01, "Sum should be ~1.1");
        }
        _ => panic!("Expected WeightsNotNormalized error"),
    }

    weights.normalize();
    assert!(
        weights.validate().is_ok(),
        "Normalized weights should sum to 1.0"
    );
    assert!(weights.is_valid(), "Normalized weights should be valid");

    let bad_weights = ComponentWeights {
        confidence: -0.5,
        ..Default::default()
    };
    let range_err = bad_weights.validate();
    assert!(range_err.is_err(), "Negative weight should fail validation");
    match range_err {
        Err(ComparisonValidationError::WeightOutOfRange {
            field_name, value, ..
        }) => {
            assert_eq!(field_name, "confidence");
            assert!((value - (-0.5)).abs() < f32::EPSILON);
        }
        _ => panic!("Expected WeightOutOfRange error"),
    }
}
