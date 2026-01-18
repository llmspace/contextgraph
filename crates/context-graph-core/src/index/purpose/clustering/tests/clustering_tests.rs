//! Tests for StandardKMeans clustering algorithm.

use uuid::Uuid;

use crate::index::purpose::clustering::clusterer::{KMeansPurposeClustering, StandardKMeans};
use crate::index::purpose::clustering::config::KMeansConfig;
use crate::index::purpose::entry::{GoalId, PurposeIndexEntry, PurposeMetadata};

use super::helpers::{create_clustered_entries, create_entry, create_purpose_vector};

#[test]
fn test_cluster_empty_entries_fails() {
    let clusterer = StandardKMeans::new();
    let entries: Vec<PurposeIndexEntry> = vec![];
    let config = KMeansConfig::default();

    let result = clusterer.cluster_purposes(&entries, &config);

    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("empty"));

    println!(
        "[VERIFIED] FAIL FAST: clustering rejects empty entries: {}",
        msg
    );
}

#[test]
fn test_cluster_k_greater_than_entries_fails() {
    let clusterer = StandardKMeans::new();
    let entries = vec![create_entry(0.5, "goal")];
    let config = KMeansConfig::new(5, 100, 1e-6).unwrap(); // k=5 but only 1 entry

    let result = clusterer.cluster_purposes(&entries, &config);

    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("k (5)"));
    assert!(msg.contains("entries.len() (1)"));

    println!(
        "[VERIFIED] FAIL FAST: clustering rejects k > entries.len(): {}",
        msg
    );
}

#[test]
fn test_cluster_single_point() {
    let clusterer = StandardKMeans::new();
    let entries = vec![create_entry(0.5, "single_goal")];
    let config = KMeansConfig::new(1, 100, 1e-6).unwrap();

    println!("[BEFORE] entries.len()={}", entries.len());

    let result = clusterer.cluster_purposes(&entries, &config).unwrap();

    println!(
        "[AFTER] clusters={}, points={}",
        result.num_clusters(),
        result.total_points()
    );

    assert_eq!(result.num_clusters(), 1);
    assert_eq!(result.total_points(), 1);
    assert!(result.converged);
    assert_eq!(result.clusters[0].len(), 1);

    println!("[VERIFIED] Single point clustering works correctly");
}

#[test]
fn test_cluster_all_same_points() {
    let clusterer = StandardKMeans::new();

    // All entries have the same purpose vector
    let base_pv = create_purpose_vector(0.5, 0.0);
    let entries: Vec<PurposeIndexEntry> = (0..5)
        .map(|_| {
            let metadata = PurposeMetadata::new(GoalId::new("same"), 0.9).unwrap();
            PurposeIndexEntry::new(Uuid::new_v4(), base_pv.clone(), metadata)
        })
        .collect();

    let config = KMeansConfig::new(2, 100, 1e-6).unwrap();

    println!("[BEFORE] entries with identical vectors: {}", entries.len());

    let result = clusterer.cluster_purposes(&entries, &config).unwrap();

    println!(
        "[AFTER] clusters={}, total_points={}",
        result.num_clusters(),
        result.total_points()
    );

    assert_eq!(result.num_clusters(), 2);
    assert_eq!(result.total_points(), 5);
    // WCSS should be very low since all points are the same
    assert!(result.wcss < 0.1);

    println!("[VERIFIED] All same points clustering works (edge case)");
}

#[test]
fn test_cluster_distinct_clusters() {
    let clusterer = StandardKMeans::new();
    let entries = create_clustered_entries();
    let config = KMeansConfig::new(3, 100, 1e-6).unwrap();

    println!(
        "[BEFORE] entries={}, expecting 3 distinct clusters",
        entries.len()
    );

    let result = clusterer.cluster_purposes(&entries, &config).unwrap();

    println!(
        "[AFTER] clusters={}, iterations={}, converged={}, WCSS={:.4}",
        result.num_clusters(),
        result.iterations,
        result.converged,
        result.wcss
    );

    assert_eq!(result.num_clusters(), 3);
    assert_eq!(result.total_points(), 15);

    // All clusters should have members
    for (i, cluster) in result.clusters.iter().enumerate() {
        assert!(!cluster.is_empty(), "Cluster {} should not be empty", i);
        println!(
            "  Cluster {}: {} members, coherence={:.4}, goal={:?}",
            i,
            cluster.len(),
            cluster.coherence,
            cluster.dominant_goal
        );
    }

    println!("[VERIFIED] Clustering produces 3 non-empty clusters for distinct data");
}

#[test]
fn test_cluster_convergence_detection() {
    let clusterer = StandardKMeans::new();
    let entries = create_clustered_entries();
    let config = KMeansConfig::new(3, 500, 1e-6).unwrap();

    let result = clusterer.cluster_purposes(&entries, &config).unwrap();

    // With well-separated clusters, should converge
    assert!(result.converged);
    assert!(result.iterations < config.max_iterations);

    println!(
        "[VERIFIED] Convergence detected at iteration {} < max {}",
        result.iterations, config.max_iterations
    );
}

#[test]
fn test_cluster_dominant_goal_detection() {
    let clusterer = StandardKMeans::new();
    let entries = create_clustered_entries();
    let config = KMeansConfig::new(3, 100, 1e-6).unwrap();

    let result = clusterer.cluster_purposes(&entries, &config).unwrap();

    // Each cluster should have a dominant goal
    let goals_found: Vec<_> = result
        .clusters
        .iter()
        .filter_map(|c| c.dominant_goal.as_ref())
        .map(|g| g.as_str().to_string())
        .collect();

    println!("[BEFORE] Expected goals: goal_low, goal_mid, goal_high");
    println!("[AFTER] Found goals: {:?}", goals_found);

    // We should find at least some goals
    assert!(!goals_found.is_empty());

    println!("[VERIFIED] Dominant goals detected for clusters");
}

#[test]
fn test_cluster_coherence_computation() {
    let clusterer = StandardKMeans::new();

    // Create tightly grouped entries (high coherence expected)
    let entries: Vec<PurposeIndexEntry> = (0..10)
        .map(|i| create_entry(0.5 + i as f32 * 0.005, "tight"))
        .collect();

    let config = KMeansConfig::new(1, 100, 1e-6).unwrap();

    let result = clusterer.cluster_purposes(&entries, &config).unwrap();

    // Single cluster with tight grouping should have high coherence
    let coherence = result.clusters[0].coherence;
    println!(
        "[RESULT] Tight cluster coherence: {:.4} (expected > 0.9)",
        coherence
    );
    assert!(coherence > 0.9);

    println!("[VERIFIED] Coherence correctly computed for tight cluster");
}

#[test]
fn test_cluster_wcss_decreases_with_more_clusters() {
    let clusterer = StandardKMeans::new();
    let entries = create_clustered_entries();

    let result_k1 = clusterer
        .cluster_purposes(&entries, &KMeansConfig::new(1, 100, 1e-6).unwrap())
        .unwrap();

    let result_k2 = clusterer
        .cluster_purposes(&entries, &KMeansConfig::new(2, 100, 1e-6).unwrap())
        .unwrap();

    let result_k3 = clusterer
        .cluster_purposes(&entries, &KMeansConfig::new(3, 100, 1e-6).unwrap())
        .unwrap();

    println!(
        "[RESULT] WCSS: k=1: {:.4}, k=2: {:.4}, k=3: {:.4}",
        result_k1.wcss, result_k2.wcss, result_k3.wcss
    );

    // WCSS should decrease (or stay same) as k increases
    assert!(result_k1.wcss >= result_k2.wcss);
    assert!(result_k2.wcss >= result_k3.wcss);

    println!("[VERIFIED] WCSS decreases with increasing k");
}

#[test]
fn test_cluster_various_k_values() {
    let clusterer = StandardKMeans::new();
    let entries = create_clustered_entries(); // 15 entries

    for k in [1, 2, 3, 5, 10, 15] {
        let config = KMeansConfig::new(k, 100, 1e-6).unwrap();
        let result = clusterer.cluster_purposes(&entries, &config).unwrap();

        assert_eq!(result.num_clusters(), k);
        assert_eq!(result.total_points(), 15);

        println!(
            "  k={}: clusters={}, iterations={}, WCSS={:.4}",
            k,
            result.num_clusters(),
            result.iterations,
            result.wcss
        );
    }

    println!("[VERIFIED] Clustering works for various k values");
}
