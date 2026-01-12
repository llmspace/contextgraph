//! Edge case and boundary condition tests for clustering.

use uuid::Uuid;

use crate::index::config::PURPOSE_VECTOR_DIM;
use crate::index::purpose::clustering::clusterer::{KMeansPurposeClustering, StandardKMeans};
use crate::index::purpose::clustering::config::KMeansConfig;
use crate::index::purpose::clustering::types::PurposeCluster;
use crate::index::purpose::entry::GoalId;

use super::helpers::{create_clustered_entries, create_entry};

#[test]
fn test_cluster_max_iterations_reached() {
    let clusterer = StandardKMeans::new();

    // Create entries that won't converge quickly
    let entries: Vec<_> = (0..20)
        .map(|i| create_entry(i as f32 / 20.0, &format!("goal_{}", i % 5)))
        .collect();

    // Very few iterations
    let config = KMeansConfig::new(5, 2, 1e-10).unwrap();

    let result = clusterer.cluster_purposes(&entries, &config).unwrap();

    // Should reach max iterations without converging
    assert_eq!(result.iterations, 2);
    // May or may not converge in 2 iterations

    println!(
        "[VERIFIED] Clustering respects max_iterations limit (converged={})",
        result.converged
    );
}

#[test]
fn test_cluster_preserves_all_memory_ids() {
    let clusterer = StandardKMeans::new();
    let entries = create_clustered_entries();
    let original_ids: std::collections::HashSet<Uuid> =
        entries.iter().map(|e| e.memory_id).collect();

    let config = KMeansConfig::new(3, 100, 1e-6).unwrap();
    let result = clusterer.cluster_purposes(&entries, &config).unwrap();

    let clustered_ids: std::collections::HashSet<Uuid> = result
        .clusters
        .iter()
        .flat_map(|c| c.members.iter().copied())
        .collect();

    assert_eq!(original_ids, clustered_ids);

    println!("[VERIFIED] All memory IDs preserved after clustering");
}

#[test]
fn test_cluster_result_clone_and_debug() {
    let clusterer = StandardKMeans::new();
    let entries = vec![create_entry(0.5, "test")];
    let config = KMeansConfig::new(1, 10, 1e-6).unwrap();

    let result = clusterer.cluster_purposes(&entries, &config).unwrap();

    // Test Clone
    let cloned = result.clone();
    assert_eq!(cloned.num_clusters(), result.num_clusters());
    assert_eq!(cloned.iterations, result.iterations);

    // Test Debug
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("ClusteringResult"));

    println!("[VERIFIED] ClusteringResult implements Clone and Debug");
}

#[test]
fn test_purpose_cluster_clone_and_debug() {
    let cluster = PurposeCluster::new(
        [0.5; PURPOSE_VECTOR_DIM],
        vec![Uuid::new_v4()],
        0.9,
        Some(GoalId::new("test")),
    );

    // Test Clone
    let cloned = cluster.clone();
    assert_eq!(cloned.len(), cluster.len());
    assert_eq!(cloned.coherence, cluster.coherence);

    // Test Debug
    let debug_str = format!("{:?}", cluster);
    assert!(debug_str.contains("PurposeCluster"));

    println!("[VERIFIED] PurposeCluster implements Clone and Debug");
}
