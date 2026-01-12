//! Clustering algorithms and helper functions.
//!
//! Contains k-means++ initialization and centroid computation.

use std::collections::HashMap;
use uuid::Uuid;

use crate::index::config::PURPOSE_VECTOR_DIM;

use super::super::entry::{GoalId, PurposeIndexEntry};
use super::metrics::{euclidean_distance, euclidean_distance_squared};
use super::types::PurposeCluster;

/// Initialize centroids using k-means++ algorithm.
///
/// K-means++ provides better initial centroids by choosing them
/// with probability proportional to squared distance from existing centroids.
pub fn kmeans_plus_plus_init(
    vectors: &[[f32; PURPOSE_VECTOR_DIM]],
    k: usize,
) -> Vec<[f32; PURPOSE_VECTOR_DIM]> {
    let n = vectors.len();
    let mut centroids = Vec::with_capacity(k);

    // Choose first centroid uniformly at random
    // Use deterministic selection for reproducibility in tests
    let first_idx = 0;
    centroids.push(vectors[first_idx]);

    // Distance from each point to nearest centroid
    let mut min_distances = vec![f32::MAX; n];

    for _ in 1..k {
        // Update distances
        let last_centroid = centroids.last().unwrap();
        for (i, vector) in vectors.iter().enumerate() {
            let dist = euclidean_distance_squared(vector, last_centroid);
            if dist < min_distances[i] {
                min_distances[i] = dist;
            }
        }

        // Select next centroid with probability proportional to D^2
        // Use deterministic weighted selection for reproducibility
        let total: f32 = min_distances.iter().sum();
        if total == 0.0 {
            // All points are at centroid locations, pick next available
            for (i, _) in vectors.iter().enumerate() {
                if !centroids
                    .iter()
                    .any(|c| euclidean_distance_squared(c, &vectors[i]) < 1e-10)
                {
                    centroids.push(vectors[i]);
                    break;
                }
            }
        } else {
            // Find the point with maximum distance (deterministic approximation)
            let max_idx = min_distances
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            centroids.push(vectors[max_idx]);
        }
    }

    centroids
}

/// Compute new centroids as mean of assigned points.
pub fn compute_centroids(
    vectors: &[[f32; PURPOSE_VECTOR_DIM]],
    assignments: &[usize],
    k: usize,
) -> Vec<[f32; PURPOSE_VECTOR_DIM]> {
    let mut sums = vec![[0.0f32; PURPOSE_VECTOR_DIM]; k];
    let mut counts = vec![0usize; k];

    for (i, &cluster) in assignments.iter().enumerate() {
        counts[cluster] += 1;
        for d in 0..PURPOSE_VECTOR_DIM {
            sums[cluster][d] += vectors[i][d];
        }
    }

    sums.into_iter()
        .zip(counts)
        .map(|(mut sum, count)| {
            if count > 0 {
                for elem in sum.iter_mut() {
                    *elem /= count as f32;
                }
            }
            sum
        })
        .collect()
}

/// Build cluster objects with metadata.
pub fn build_clusters(
    entries: &[PurposeIndexEntry],
    assignments: &[usize],
    centroids: &[[f32; PURPOSE_VECTOR_DIM]],
    k: usize,
) -> Vec<PurposeCluster> {
    let mut cluster_members: Vec<Vec<usize>> = vec![Vec::new(); k];

    for (i, &cluster) in assignments.iter().enumerate() {
        cluster_members[cluster].push(i);
    }

    cluster_members
        .into_iter()
        .enumerate()
        .map(|(cluster_idx, member_indices)| {
            let members: Vec<Uuid> = member_indices
                .iter()
                .map(|&i| entries[i].memory_id)
                .collect();

            let coherence = if members.is_empty() {
                0.0
            } else {
                compute_cluster_coherence(entries, &member_indices, &centroids[cluster_idx])
            };

            let dominant_goal = find_dominant_goal(entries, &member_indices);

            PurposeCluster::new(centroids[cluster_idx], members, coherence, dominant_goal)
        })
        .collect()
}

/// Compute coherence score for a cluster.
///
/// Coherence is computed as 1 - (mean_distance / max_possible_distance).
/// Max possible distance for normalized vectors in 13D is sqrt(4*13) = 7.21.
pub fn compute_cluster_coherence(
    entries: &[PurposeIndexEntry],
    member_indices: &[usize],
    centroid: &[f32; PURPOSE_VECTOR_DIM],
) -> f32 {
    if member_indices.is_empty() {
        return 0.0;
    }

    let total_dist: f32 = member_indices
        .iter()
        .map(|&i| euclidean_distance(&entries[i].purpose_vector.alignments, centroid))
        .sum();

    let mean_dist = total_dist / member_indices.len() as f32;

    // Max distance in 13D for vectors in [0,1] is sqrt(13)
    let max_dist = (PURPOSE_VECTOR_DIM as f32).sqrt();

    (1.0 - mean_dist / max_dist).clamp(0.0, 1.0)
}

/// Find the most common goal among cluster members.
pub fn find_dominant_goal(
    entries: &[PurposeIndexEntry],
    member_indices: &[usize],
) -> Option<GoalId> {
    if member_indices.is_empty() {
        return None;
    }

    let mut goal_counts: HashMap<String, usize> = HashMap::new();

    for &i in member_indices {
        let goal_str = entries[i].metadata.primary_goal.as_str().to_string();
        *goal_counts.entry(goal_str).or_insert(0) += 1;
    }

    goal_counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(goal, _)| GoalId::new(goal))
}

/// Compute within-cluster sum of squares.
pub fn compute_wcss(
    vectors: &[[f32; PURPOSE_VECTOR_DIM]],
    assignments: &[usize],
    centroids: &[[f32; PURPOSE_VECTOR_DIM]],
) -> f32 {
    vectors
        .iter()
        .zip(assignments.iter())
        .map(|(vector, &cluster)| euclidean_distance_squared(vector, &centroids[cluster]))
        .sum()
}
