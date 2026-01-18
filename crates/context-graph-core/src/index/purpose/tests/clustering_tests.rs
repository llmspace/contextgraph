//! Clustering Tests - K-means with real 13D vectors

use super::helpers::{create_clustered_entries, create_entry};
use crate::index::purpose::clustering::{KMeansConfig, KMeansPurposeClustering, StandardKMeans};
use std::collections::HashSet;
use uuid::Uuid;

#[test]
fn test_kmeans_with_real_13d_purpose_vectors() {
    let clusterer = StandardKMeans::new();
    let entries = create_clustered_entries();
    let config = KMeansConfig::new(3, 100, 1e-6).unwrap();

    println!("[BEFORE] entries={}, k={}", entries.len(), config.k);

    let result = clusterer.cluster_purposes(&entries, &config).unwrap();

    println!(
        "[AFTER] clusters={}, iterations={}, WCSS={:.4}",
        result.num_clusters(),
        result.iterations,
        result.wcss
    );

    assert_eq!(result.num_clusters(), 3);
    assert_eq!(result.total_points(), 15);

    for (i, cluster) in result.clusters.iter().enumerate() {
        assert!(!cluster.is_empty(), "Cluster {} should not be empty", i);
        println!(
            "  Cluster {}: {} members, coherence={:.4}",
            i,
            cluster.len(),
            cluster.coherence
        );
    }

    println!("[VERIFIED] K-means with real 13D purpose vectors produces 3 clusters");
}

#[test]
fn test_kmeans_convergence_detection() {
    let clusterer = StandardKMeans::new();
    let entries = create_clustered_entries();
    let config = KMeansConfig::new(3, 500, 1e-6).unwrap();

    let result = clusterer.cluster_purposes(&entries, &config).unwrap();

    assert!(
        result.converged,
        "Should converge with well-separated clusters"
    );
    assert!(
        result.iterations < config.max_iterations,
        "Should converge before max_iterations"
    );

    println!(
        "[VERIFIED] Convergence detected at iteration {} < max {}",
        result.iterations, config.max_iterations
    );
}

#[test]
fn test_cluster_coherence_calculation() {
    let clusterer = StandardKMeans::new();

    // Create tightly grouped entries (high coherence expected)
    let entries: Vec<_> = (0..10)
        .map(|i| create_entry(0.5 + i as f32 * 0.005, "tight"))
        .collect();

    let config = KMeansConfig::new(1, 100, 1e-6).unwrap();
    let result = clusterer.cluster_purposes(&entries, &config).unwrap();

    let coherence = result.clusters[0].coherence;
    assert!(coherence > 0.9, "Tight cluster should have high coherence");

    println!(
        "[VERIFIED] Cluster coherence calculation: {:.4} > 0.9",
        coherence
    );
}

#[test]
fn test_clustering_single_point_edge_case() {
    let clusterer = StandardKMeans::new();
    let entries = vec![create_entry(0.5, "single")];
    let config = KMeansConfig::new(1, 100, 1e-6).unwrap();

    let result = clusterer.cluster_purposes(&entries, &config).unwrap();

    assert_eq!(result.num_clusters(), 1);
    assert_eq!(result.total_points(), 1);
    assert!(result.converged);

    println!("[VERIFIED] Single point clustering works (edge case)");
}

#[test]
fn test_clustering_k_greater_than_n_rejected() {
    let clusterer = StandardKMeans::new();
    let entries = vec![create_entry(0.5, "single")];
    let config = KMeansConfig::new(5, 100, 1e-6).unwrap(); // k=5 but only 1 entry

    let result = clusterer.cluster_purposes(&entries, &config);

    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("k (5)"));

    println!("[VERIFIED] FAIL FAST: Clustering rejects k > n: {}", msg);
}

#[test]
fn test_kmeans_config_boundary_values() {
    // Valid boundary k=1
    let config = KMeansConfig::with_k(1).unwrap();
    assert_eq!(config.k, 1);

    // Valid large max_iterations
    let config = KMeansConfig::new(5, 10000, 1e-6).unwrap();
    assert_eq!(config.max_iterations, 10000);

    // Valid small convergence_threshold
    let config = KMeansConfig::new(5, 100, 1e-10).unwrap();
    assert!(config.convergence_threshold > 0.0);

    println!("[VERIFIED] KMeansConfig accepts valid boundary values");
}

#[test]
fn test_clustering_preserves_all_memory_ids() {
    let clusterer = StandardKMeans::new();
    let entries = create_clustered_entries();
    let original_ids: HashSet<Uuid> = entries.iter().map(|e| e.memory_id).collect();

    let config = KMeansConfig::new(3, 100, 1e-6).unwrap();
    let result = clusterer.cluster_purposes(&entries, &config).unwrap();

    let clustered_ids: HashSet<Uuid> = result
        .clusters
        .iter()
        .flat_map(|c| c.members.iter().copied())
        .collect();

    assert_eq!(original_ids, clustered_ids);

    println!("[VERIFIED] Clustering preserves all memory IDs");
}

#[test]
fn test_clustering_wcss_decreases_with_more_clusters() {
    let clusterer = StandardKMeans::new();
    let entries = create_clustered_entries();

    let result_k1 = clusterer
        .cluster_purposes(&entries, &KMeansConfig::with_k(1).unwrap())
        .unwrap();

    let result_k2 = clusterer
        .cluster_purposes(&entries, &KMeansConfig::with_k(2).unwrap())
        .unwrap();

    let result_k3 = clusterer
        .cluster_purposes(&entries, &KMeansConfig::with_k(3).unwrap())
        .unwrap();

    // WCSS should decrease or stay same as k increases
    assert!(result_k1.wcss >= result_k2.wcss);
    assert!(result_k2.wcss >= result_k3.wcss);

    println!(
        "[VERIFIED] WCSS decreases with increasing k: k1={:.4}, k2={:.4}, k3={:.4}",
        result_k1.wcss, result_k2.wcss, result_k3.wcss
    );
}
