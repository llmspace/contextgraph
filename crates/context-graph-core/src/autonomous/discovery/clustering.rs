//! K-means clustering implementation for teleological arrays.
//!
//! This module implements the K-means clustering algorithm using full 13-embedder
//! TeleologicalArrays and the TeleologicalComparator for distance/similarity.

use crate::teleological::comparator::TeleologicalComparator;
use crate::types::fingerprint::TeleologicalArray;

use super::centroid::{compute_centroid, create_zeroed_fingerprint};
use super::types::{Cluster, DiscoveryConfig, NumClusters};

/// Cluster arrays using K-means on full teleological arrays.
///
/// # Arguments
///
/// * `arrays` - Slice of references to teleological arrays to cluster
/// * `config` - Discovery configuration with clustering parameters
/// * `comparator` - TeleologicalComparator for similarity computation
///
/// # Returns
///
/// Vector of clusters with members, centroids, and coherence scores.
pub fn cluster_arrays(
    arrays: &[&TeleologicalArray],
    config: &DiscoveryConfig,
    comparator: &TeleologicalComparator,
) -> Vec<Cluster> {
    let n = arrays.len();

    // Determine number of clusters
    let k = match &config.num_clusters {
        NumClusters::Auto => {
            // sqrt(n/2) heuristic
            let k = ((n as f32 / 2.0).sqrt().ceil() as usize).max(2);
            k.min(n / config.min_cluster_size) // Don't exceed data capacity
        }
        NumClusters::Fixed(k) => *k,
        NumClusters::Range { min, max } => {
            // Use elbow method approximation
            let k_auto = ((n as f32 / 2.0).sqrt().ceil() as usize).max(2);
            k_auto.clamp(*min, *max)
        }
    };

    assert!(
        k >= 1,
        "FAIL FAST: Cannot form clusters with k=0. n={}, min_cluster_size={}",
        n,
        config.min_cluster_size
    );

    eprintln!("[K-means] Clustering {} arrays into {} clusters", n, k);

    // Initialize centroids using k-means++ strategy
    let mut centroids: Vec<TeleologicalArray> =
        initialize_centroids_kmeans_pp(arrays, k, comparator);
    let mut assignments: Vec<usize> = vec![0; n];
    let mut iteration = 0;

    loop {
        iteration += 1;

        // Assignment step: assign each array to nearest centroid
        let mut changed = false;
        for (i, array) in arrays.iter().enumerate() {
            let nearest = find_nearest_centroid(array, &centroids, comparator);
            if nearest != assignments[i] {
                changed = true;
                assignments[i] = nearest;
            }
        }

        // Check convergence
        if !changed || iteration >= config.max_iterations {
            eprintln!(
                "[K-means] Converged after {} iterations (changed={})",
                iteration, changed
            );
            break;
        }

        // Update step: recompute centroids
        centroids = recompute_centroids(arrays, &assignments, k);
    }

    // Build cluster objects
    let mut clusters: Vec<Cluster> = Vec::with_capacity(k);
    for cluster_id in 0..k {
        let members: Vec<usize> = assignments
            .iter()
            .enumerate()
            .filter(|(_, &a)| a == cluster_id)
            .map(|(i, _)| i)
            .collect();

        if members.is_empty() {
            continue; // Skip empty clusters
        }

        let member_arrays: Vec<&TeleologicalArray> = members.iter().map(|&i| arrays[i]).collect();

        let centroid = compute_centroid(&member_arrays);
        let coherence = compute_cluster_coherence(&member_arrays, &centroid, comparator);

        clusters.push(Cluster {
            members,
            centroid,
            coherence,
        });
    }

    clusters
}

/// Initialize centroids using k-means++ strategy.
fn initialize_centroids_kmeans_pp(
    arrays: &[&TeleologicalArray],
    k: usize,
    comparator: &TeleologicalComparator,
) -> Vec<TeleologicalArray> {
    let mut centroids: Vec<TeleologicalArray> = Vec::with_capacity(k);

    // First centroid: pick the first array (deterministic for reproducibility)
    centroids.push(arrays[0].clone());

    // Remaining centroids: pick proportional to squared distance
    for _ in 1..k {
        let mut max_min_dist = 0.0_f32;
        let mut best_idx = 0;

        for (i, array) in arrays.iter().enumerate() {
            // Find minimum distance to existing centroids
            let mut min_dist = f32::MAX;
            for centroid in &centroids {
                let result = comparator.compare(array, centroid);
                let similarity = result.map(|r| r.overall).unwrap_or(0.0);
                let distance = 1.0 - similarity;
                min_dist = min_dist.min(distance);
            }

            // Pick array with maximum minimum distance (furthest from all centroids)
            if min_dist > max_min_dist {
                max_min_dist = min_dist;
                best_idx = i;
            }
        }

        centroids.push(arrays[best_idx].clone());
    }

    centroids
}

/// Find nearest centroid for an array.
fn find_nearest_centroid(
    array: &TeleologicalArray,
    centroids: &[TeleologicalArray],
    comparator: &TeleologicalComparator,
) -> usize {
    let mut best_idx = 0;
    let mut best_similarity = f32::NEG_INFINITY;

    for (i, centroid) in centroids.iter().enumerate() {
        let result = comparator.compare(array, centroid);
        let similarity = result.map(|r| r.overall).unwrap_or(0.0);
        if similarity > best_similarity {
            best_similarity = similarity;
            best_idx = i;
        }
    }

    best_idx
}

/// Recompute centroids from assignments.
fn recompute_centroids(
    arrays: &[&TeleologicalArray],
    assignments: &[usize],
    k: usize,
) -> Vec<TeleologicalArray> {
    (0..k)
        .map(|cluster_id| {
            let members: Vec<&TeleologicalArray> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &a)| a == cluster_id)
                .map(|(i, _)| arrays[i])
                .collect();

            if members.is_empty() {
                // Keep the previous centroid if cluster is empty
                // This shouldn't happen with proper k-means++ initialization
                create_zeroed_fingerprint()
            } else {
                compute_centroid(&members)
            }
        })
        .collect()
}

/// Compute intra-cluster coherence.
pub fn compute_cluster_coherence(
    members: &[&TeleologicalArray],
    centroid: &TeleologicalArray,
    comparator: &TeleologicalComparator,
) -> f32 {
    if members.is_empty() {
        return 0.0;
    }

    let similarities: Vec<f32> = members
        .iter()
        .filter_map(|m| comparator.compare(m, centroid).ok().map(|r| r.overall))
        .collect();

    if similarities.is_empty() {
        return 0.0;
    }

    similarities.iter().sum::<f32>() / similarities.len() as f32
}
