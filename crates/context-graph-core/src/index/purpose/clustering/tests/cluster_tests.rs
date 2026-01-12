//! Tests for PurposeCluster and ClusteringResult types.

use uuid::Uuid;

use crate::index::config::PURPOSE_VECTOR_DIM;
use crate::index::purpose::clustering::types::{ClusteringResult, PurposeCluster};
use crate::index::purpose::entry::GoalId;

// =========================================================================
// PurposeCluster Tests
// =========================================================================

#[test]
fn test_purpose_cluster_new() {
    let centroid = [0.5; PURPOSE_VECTOR_DIM];
    let members = vec![Uuid::new_v4(), Uuid::new_v4()];
    let cluster = PurposeCluster::new(centroid, members.clone(), 0.85, Some(GoalId::new("test")));

    assert_eq!(cluster.centroid, centroid);
    assert_eq!(cluster.members.len(), 2);
    assert!((cluster.coherence - 0.85).abs() < f32::EPSILON);
    assert!(cluster.dominant_goal.is_some());

    println!("[VERIFIED] PurposeCluster::new creates cluster with all fields");
}

#[test]
fn test_purpose_cluster_len_and_is_empty() {
    let empty_cluster = PurposeCluster::new([0.5; PURPOSE_VECTOR_DIM], vec![], 0.0, None);

    assert!(empty_cluster.is_empty());
    assert_eq!(empty_cluster.len(), 0);

    let filled_cluster = PurposeCluster::new(
        [0.5; PURPOSE_VECTOR_DIM],
        vec![Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()],
        0.9,
        None,
    );

    assert!(!filled_cluster.is_empty());
    assert_eq!(filled_cluster.len(), 3);

    println!("[VERIFIED] PurposeCluster len and is_empty work correctly");
}

// =========================================================================
// ClusteringResult Tests
// =========================================================================

#[test]
fn test_clustering_result_new() {
    let clusters = vec![
        PurposeCluster::new([0.2; PURPOSE_VECTOR_DIM], vec![Uuid::new_v4()], 0.8, None),
        PurposeCluster::new([0.8; PURPOSE_VECTOR_DIM], vec![Uuid::new_v4()], 0.9, None),
    ];

    let result = ClusteringResult::new(clusters, 25, true, 0.5);

    assert_eq!(result.num_clusters(), 2);
    assert_eq!(result.iterations, 25);
    assert!(result.converged);
    assert!((result.wcss - 0.5).abs() < f32::EPSILON);

    println!("[VERIFIED] ClusteringResult::new creates result with all fields");
}

#[test]
fn test_clustering_result_total_points() {
    let clusters = vec![
        PurposeCluster::new(
            [0.2; PURPOSE_VECTOR_DIM],
            vec![Uuid::new_v4(), Uuid::new_v4()],
            0.8,
            None,
        ),
        PurposeCluster::new(
            [0.5; PURPOSE_VECTOR_DIM],
            vec![Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()],
            0.9,
            None,
        ),
        PurposeCluster::new([0.8; PURPOSE_VECTOR_DIM], vec![Uuid::new_v4()], 0.7, None),
    ];

    let result = ClusteringResult::new(clusters, 10, true, 0.3);

    assert_eq!(result.total_points(), 6); // 2 + 3 + 1

    println!("[VERIFIED] ClusteringResult::total_points returns correct count");
}

#[test]
fn test_clustering_result_avg_cluster_size() {
    let clusters = vec![
        PurposeCluster::new(
            [0.2; PURPOSE_VECTOR_DIM],
            vec![Uuid::new_v4(), Uuid::new_v4()],
            0.8,
            None,
        ),
        PurposeCluster::new(
            [0.5; PURPOSE_VECTOR_DIM],
            vec![
                Uuid::new_v4(),
                Uuid::new_v4(),
                Uuid::new_v4(),
                Uuid::new_v4(),
            ],
            0.9,
            None,
        ),
    ];

    let result = ClusteringResult::new(clusters, 10, true, 0.3);

    assert!((result.avg_cluster_size() - 3.0).abs() < f32::EPSILON); // (2 + 4) / 2

    println!("[VERIFIED] ClusteringResult::avg_cluster_size returns correct average");
}

#[test]
fn test_clustering_result_avg_coherence() {
    let clusters = vec![
        PurposeCluster::new([0.2; PURPOSE_VECTOR_DIM], vec![], 0.8, None),
        PurposeCluster::new([0.5; PURPOSE_VECTOR_DIM], vec![], 0.9, None),
        PurposeCluster::new([0.8; PURPOSE_VECTOR_DIM], vec![], 0.7, None),
    ];

    let result = ClusteringResult::new(clusters, 10, true, 0.3);

    let expected = (0.8 + 0.9 + 0.7) / 3.0;
    assert!((result.avg_coherence() - expected).abs() < f32::EPSILON);

    println!("[VERIFIED] ClusteringResult::avg_coherence returns correct average");
}
