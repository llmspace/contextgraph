//! TransE entropy for E11 (Entity/KnowledgeGraph) embeddings.
//!
//! Formula: ΔS = normalized ||h + r - t|| (TransE distance)
//! Per constitution.yaml delta_methods.ΔS E11: "TransE ||h+r-t||"
//!
//! # Algorithm
//!
//! TransE models knowledge graph relationships as translations in embedding space:
//! - For a valid triple (h, r, t): h + r ≈ t
//! - Distance: d(h, r, t) = ||h + r - t||
//!
//! 1. **Embedding Parsing:**
//!    - Parse current embedding as (head_context, relation_context)
//!    - First half (0..split_point) = head entity context
//!    - Second half (split_point..dim) = relation context
//!
//! 2. **TransE Distance Computation:**
//!    - For each history embedding (treated as potential tail):
//!    - Compute: d = ||head + relation - tail||
//!    - Use L1 or L2 norm as configured
//!
//! 3. **KNN-Style Aggregation:**
//!    - Sort distances, take k nearest
//!    - Compute mean of k-nearest distances
//!    - Normalize via sigmoid with running statistics
//!    - Clamp to [0.0, 1.0], verify no NaN/Infinity (AP-10)
//!
//! # Constitution Reference
//!
//! - From constitution.yaml delta_methods.ΔS E11: "TransE ||h+r-t||"
//! - E11 dimension: 384D (MiniLM)
//! - Default split: midpoint (192)

use super::EmbedderEntropy;
use crate::config::SurpriseConfig;
use crate::error::{UtlError, UtlResult};
use context_graph_core::teleological::Embedder;

/// E11 (Entity) embedding dimension per constitution.yaml.
const E11_DIM: usize = 384;

/// Default split point (midpoint for 384D = 192).
const DEFAULT_SPLIT_POINT: usize = 192;

/// Default L-norm (2 = L2/Euclidean per original TransE paper).
const DEFAULT_NORM: u8 = 2;

/// Default k neighbors for averaging.
const DEFAULT_K_NEIGHBORS: usize = 5;

/// Minimum standard deviation to avoid division by zero.
const MIN_STD_DEV: f32 = 0.1;

/// E11 (Entity) entropy using TransE distance.
///
/// TransE models knowledge graph relationships as translations:
/// For triple (head, relation, tail): head + relation ≈ tail
///
/// # Constitution Reference
/// E11: "TransE: ΔS=||h+r-t||" (constitution.yaml line 165)
#[derive(Debug, Clone)]
pub struct TransEEntropy {
    /// Dimension split point for head vs relation.
    /// Default: dim / 2 (E11 is 384D, so split at 192)
    split_point: usize,
    /// L-norm for distance (1 = L1/Manhattan, 2 = L2/Euclidean).
    /// Default: 2 (L2 norm per original TransE paper)
    norm: u8,
    /// Running mean for distance normalization.
    running_mean: f32,
    /// Running variance for distance normalization.
    running_variance: f32,
    /// Number of samples seen.
    sample_count: usize,
    /// k neighbors for averaging.
    k_neighbors: usize,
}

impl Default for TransEEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl TransEEntropy {
    /// Create a new TransE entropy calculator with constitution defaults.
    pub fn new() -> Self {
        Self {
            split_point: DEFAULT_SPLIT_POINT,
            norm: DEFAULT_NORM,
            running_mean: 0.5,
            running_variance: 0.1,
            sample_count: 0,
            k_neighbors: DEFAULT_K_NEIGHBORS,
        }
    }

    /// Create with specified L-norm (1 or 2).
    pub fn with_norm(norm: u8) -> Self {
        Self {
            norm: if norm == 1 { 1 } else { 2 },
            ..Self::new()
        }
    }

    /// Create from SurpriseConfig.
    pub fn from_config(config: &SurpriseConfig) -> Self {
        let norm = config.entity_transe_norm.clamp(1, 2);
        let split_ratio = config.entity_split_ratio.clamp(0.1, 0.9);
        let k = config.entity_k_neighbors.clamp(1, 20);

        // Calculate split point from ratio (for E11 384D)
        let split_point = (E11_DIM as f32 * split_ratio) as usize;

        Self {
            split_point: split_point.clamp(1, E11_DIM - 1),
            norm,
            running_mean: 0.5,
            running_variance: 0.1,
            sample_count: 0,
            k_neighbors: k,
        }
    }

    /// Builder: set split point.
    #[must_use]
    pub fn with_split_point(mut self, split_point: usize) -> Self {
        self.split_point = split_point.max(1);
        self
    }

    /// Builder: set k neighbors.
    #[must_use]
    pub fn with_k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k.clamp(1, 20);
        self
    }

    /// Extract head context from embedding (first half).
    ///
    /// # Arguments
    /// * `embedding` - Full embedding vector
    ///
    /// # Returns
    /// Slice containing head entity context
    #[inline]
    fn extract_head<'a>(&self, embedding: &'a [f32]) -> &'a [f32] {
        let end = self.split_point.min(embedding.len());
        &embedding[..end]
    }

    /// Extract relation context from embedding (second half).
    ///
    /// # Arguments
    /// * `embedding` - Full embedding vector
    ///
    /// # Returns
    /// Slice containing relation context
    #[inline]
    fn extract_relation<'a>(&self, embedding: &'a [f32]) -> &'a [f32] {
        let start = self.split_point.min(embedding.len());
        &embedding[start..]
    }

    /// Compute TransE distance: ||h + r - t||
    ///
    /// # Arguments
    /// * `head` - Head entity embedding
    /// * `relation` - Relation embedding
    /// * `tail` - Tail entity embedding (full embedding from history)
    ///
    /// # Returns
    /// Distance value (0 = perfect translation)
    fn compute_transe_distance(&self, head: &[f32], relation: &[f32], tail: &[f32]) -> f32 {
        // For tail, we use the head portion of the embedding to compare
        // since in knowledge graphs the tail is another entity
        let tail_head = if tail.len() > self.split_point {
            &tail[..self.split_point]
        } else {
            tail
        };

        // Ensure we have matching dimensions
        let min_len = head.len().min(relation.len()).min(tail_head.len());
        if min_len == 0 {
            return f32::MAX;
        }

        match self.norm {
            1 => {
                // L1 distance: ||h + r - t||_1
                head.iter()
                    .take(min_len)
                    .zip(relation.iter().take(min_len))
                    .zip(tail_head.iter().take(min_len))
                    .map(|((h, r), t)| (h + r - t).abs())
                    .sum()
            }
            _ => {
                // L2 distance: ||h + r - t||_2 (default)
                let sum_sq: f32 = head
                    .iter()
                    .take(min_len)
                    .zip(relation.iter().take(min_len))
                    .zip(tail_head.iter().take(min_len))
                    .map(|((h, r), t)| {
                        let diff = h + r - t;
                        diff * diff
                    })
                    .sum();
                sum_sq.sqrt()
            }
        }
    }

    /// Sigmoid function for normalization.
    #[inline]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl EmbedderEntropy for TransEEntropy {
    fn compute_delta_s(&self, current: &[f32], history: &[Vec<f32>], k: usize) -> UtlResult<f32> {
        // Validate input: empty current is an error per spec
        if current.is_empty() {
            return Err(UtlError::EmptyInput);
        }

        // Check for NaN/Infinity in current embedding (AP-10)
        for &v in current {
            if v.is_nan() || v.is_infinite() {
                return Err(UtlError::EntropyError(
                    "Invalid value (NaN/Infinity) in current embedding".to_string(),
                ));
            }
        }

        // Empty history = maximum surprise
        if history.is_empty() {
            return Ok(1.0);
        }

        // Use provided k or fallback to configured k_neighbors
        let k_to_use = if k > 0 { k } else { self.k_neighbors };

        // Extract head and relation from current embedding
        let head = self.extract_head(current);
        let relation = self.extract_relation(current);

        // Compute TransE distances to all valid history embeddings
        let mut distances: Vec<f32> = history
            .iter()
            .filter(|h| !h.is_empty())
            .filter(|h| {
                // Validate history embeddings for NaN/Infinity
                h.iter().all(|v| !v.is_nan() && !v.is_infinite())
            })
            .map(|tail| self.compute_transe_distance(head, relation, tail))
            .filter(|d| !d.is_nan() && !d.is_infinite() && *d < f32::MAX)
            .collect();

        // If all history was invalid, return max surprise
        if distances.is_empty() {
            return Ok(1.0);
        }

        // Sort distances and take k nearest
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k_actual = k_to_use.min(distances.len()).max(1);
        let k_nearest = &distances[..k_actual];

        // Compute mean of k-nearest distances
        let mean_dist: f32 = k_nearest.iter().sum::<f32>() / k_actual as f32;

        // Normalize: z = (mean_dist - running_mean) / running_std
        let running_std = self.running_variance.sqrt().max(MIN_STD_DEV);
        let z = (mean_dist - self.running_mean) / running_std;

        // Apply sigmoid normalization
        let delta_s = Self::sigmoid(z);

        // Final validation per AP-10: no NaN/Infinity
        let clamped = delta_s.clamp(0.0, 1.0);
        if clamped.is_nan() || clamped.is_infinite() {
            return Err(UtlError::EntropyError(
                "Computed delta_s is NaN or Infinity - violates AP-10".to_string(),
            ));
        }

        Ok(clamped)
    }

    fn embedder_type(&self) -> Embedder {
        Embedder::Entity
    }

    fn reset(&mut self) {
        self.running_mean = 0.5;
        self.running_variance = 0.1;
        self.sample_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transe_empty_history_returns_one() {
        let calculator = TransEEntropy::new();
        let current = vec![0.5f32; E11_DIM];
        let history: Vec<Vec<f32>> = vec![];

        println!("BEFORE: history.len() = 0");
        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok(), "Should not error on empty history");
        assert_eq!(result.unwrap(), 1.0, "Empty history should return 1.0");
        println!("AFTER: delta_s = 1.0");
        println!("[PASS] test_transe_empty_history_returns_one - delta_s = 1.0");
    }

    #[test]
    fn test_transe_empty_input_error() {
        let calculator = TransEEntropy::new();
        let empty: Vec<f32> = vec![];
        let history = vec![vec![0.5f32; E11_DIM]];

        let result = calculator.compute_delta_s(&empty, &history, 5);
        assert!(
            matches!(result, Err(UtlError::EmptyInput)),
            "Empty input should return EmptyInput error"
        );
        println!("[PASS] test_transe_empty_input_error - Err(EmptyInput)");
    }

    #[test]
    fn test_transe_identical_returns_low() {
        let calculator = TransEEntropy::new();

        // For TransE: h + r = t should give low surprise
        // Create current where h=[0.3; 192], r=[0.2; 192] => h+r=[0.5; 192]
        // Create history where tail head portion = [0.5; 192] (perfect match)
        let mut current = vec![0.0f32; E11_DIM];
        for i in 0..192 {
            current[i] = 0.3; // head = 0.3
        }
        for i in 192..E11_DIM {
            current[i] = 0.2; // relation = 0.2
        }

        // History embeddings where the head portion = h + r = 0.5
        let mut tail = vec![0.0f32; E11_DIM];
        for i in 0..192 {
            tail[i] = 0.5; // h + r = 0.3 + 0.2 = 0.5
        }
        let history: Vec<Vec<f32>> = vec![tail; 20];

        println!("BEFORE: history contains 20 embeddings where tail = h + r (perfect translation)");
        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok());
        let delta_s = result.unwrap();

        println!(
            "AFTER: delta_s = {} (expected < 0.5 for perfect translation)",
            delta_s
        );
        assert!(
            delta_s < 0.5,
            "Perfect translation should have low surprise, got {}",
            delta_s
        );
        println!(
            "[PASS] test_transe_identical_returns_low - delta_s = {} < 0.5",
            delta_s
        );
    }

    #[test]
    fn test_transe_perfect_translation() {
        let calculator = TransEEntropy::new();

        // current = [head(192) | relation(192)] where head + relation = history[0]
        let mut current = vec![0.0f32; E11_DIM];
        for i in 0..192 {
            current[i] = 0.5; // head = 0.5
        }
        for i in 192..384 {
            current[i] = 0.3; // relation = 0.3
        }

        // Tail embedding where the head portion matches h + r = 0.8
        let mut tail = vec![0.0f32; E11_DIM];
        for i in 0..192 {
            tail[i] = 0.8; // h + r = 0.5 + 0.3 = 0.8
        }
        let history = vec![tail; 10];

        println!("BEFORE: perfect translation setup (h=0.5, r=0.3, t=0.8)");
        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok());
        let delta_s = result.unwrap();

        println!("AFTER: delta_s = {} (expected near 0 or low)", delta_s);
        // Perfect translation should give low distance, hence low surprise
        assert!(
            delta_s < 0.6,
            "Perfect translation should have low surprise, got {}",
            delta_s
        );
        println!(
            "[PASS] test_transe_perfect_translation - delta_s = {}",
            delta_s
        );
    }

    #[test]
    fn test_transe_orthogonal_returns_high() {
        let calculator = TransEEntropy::new();

        // Create orthogonal/distant embeddings
        let mut current = vec![0.0f32; E11_DIM];
        current[0] = 1.0;
        current[192] = 1.0; // head = 1.0 at pos 0, relation = 1.0 at pos 0

        // History with values in different positions
        let mut history_item = vec![0.0f32; E11_DIM];
        history_item[96] = 1.0; // Orthogonal to current
        let history = vec![history_item; 10];

        println!("BEFORE: current orthogonal to history");
        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok());
        let delta_s = result.unwrap();

        println!("AFTER: delta_s = {}", delta_s);
        assert!(
            delta_s > 0.5,
            "Orthogonal embeddings should have high surprise, got {}",
            delta_s
        );
        println!(
            "[PASS] test_transe_orthogonal_returns_high - delta_s = {} > 0.5",
            delta_s
        );
    }

    #[test]
    fn test_transe_embedder_type() {
        let calculator = TransEEntropy::new();
        assert_eq!(
            calculator.embedder_type(),
            Embedder::Entity,
            "Should return Embedder::Entity"
        );
        println!("[PASS] test_transe_embedder_type - Embedder::Entity");
    }

    #[test]
    fn test_transe_valid_range() {
        let calculator = TransEEntropy::new();

        // Test various input patterns
        for pattern in 0..5 {
            let current: Vec<f32> = (0..E11_DIM)
                .map(|i| ((i + pattern * 100) as f32) / E11_DIM as f32)
                .collect();

            let history: Vec<Vec<f32>> = (0..15)
                .map(|j| {
                    (0..E11_DIM)
                        .map(|i| ((i + j * 50) as f32) / E11_DIM as f32)
                        .collect()
                })
                .collect();

            let result = calculator.compute_delta_s(&current, &history, 5);
            assert!(result.is_ok());
            let delta_s = result.unwrap();

            assert!(
                (0.0..=1.0).contains(&delta_s),
                "Pattern {} delta_s {} out of range [0.0, 1.0]",
                pattern,
                delta_s
            );
        }

        println!("[PASS] test_transe_valid_range - All outputs in [0.0, 1.0]");
    }

    #[test]
    fn test_transe_no_nan_infinity() {
        let calculator = TransEEntropy::new();

        // Edge case: very small values
        let small: Vec<f32> = vec![1e-10; E11_DIM];
        let history: Vec<Vec<f32>> = vec![vec![1e-10; E11_DIM]; 10];

        let result = calculator.compute_delta_s(&small, &history, 5);
        assert!(result.is_ok());
        let delta_s = result.unwrap();
        assert!(!delta_s.is_nan(), "delta_s should not be NaN (AP-10)");
        assert!(
            !delta_s.is_infinite(),
            "delta_s should not be Infinite (AP-10)"
        );

        // Edge case: values near 1
        let near_one: Vec<f32> = vec![0.9999; E11_DIM];
        let result2 = calculator.compute_delta_s(&near_one, &history, 5);
        assert!(result2.is_ok());
        let delta_s2 = result2.unwrap();
        assert!(!delta_s2.is_nan(), "delta_s should not be NaN");
        assert!(!delta_s2.is_infinite(), "delta_s should not be Infinite");

        // Edge case: mixed positive and negative
        let mixed: Vec<f32> = (0..E11_DIM)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let result3 = calculator.compute_delta_s(&mixed, &history, 5);
        assert!(result3.is_ok());
        let delta_s3 = result3.unwrap();
        assert!(
            !delta_s3.is_nan(),
            "delta_s should not be NaN for mixed values"
        );

        println!("[PASS] test_transe_no_nan_infinity - AP-10 compliant");
    }

    #[test]
    fn test_transe_l1_vs_l2_norm() {
        let calc_l1 = TransEEntropy::with_norm(1);
        let calc_l2 = TransEEntropy::with_norm(2);

        assert_eq!(calc_l1.norm, 1, "Should be L1 norm");
        assert_eq!(calc_l2.norm, 2, "Should be L2 norm");

        // Compute distances with both norms
        let current = vec![0.5f32; E11_DIM];
        let history: Vec<Vec<f32>> = vec![vec![0.6f32; E11_DIM]; 10];

        let result_l1 = calc_l1.compute_delta_s(&current, &history, 5);
        let result_l2 = calc_l2.compute_delta_s(&current, &history, 5);

        assert!(result_l1.is_ok());
        assert!(result_l2.is_ok());

        let delta_s_l1 = result_l1.unwrap();
        let delta_s_l2 = result_l2.unwrap();

        println!("L1 delta_s = {}, L2 delta_s = {}", delta_s_l1, delta_s_l2);

        // Both should be valid, but may differ
        assert!((0.0..=1.0).contains(&delta_s_l1));
        assert!((0.0..=1.0).contains(&delta_s_l2));

        println!("[PASS] test_transe_l1_vs_l2_norm - Different distances");
    }

    #[test]
    fn test_transe_from_config() {
        let mut config = SurpriseConfig::default();
        config.entity_transe_norm = 1;
        config.entity_split_ratio = 0.6;
        config.entity_k_neighbors = 10;

        let calculator = TransEEntropy::from_config(&config);

        assert_eq!(calculator.norm, 1, "norm should be 1 from config");
        assert_eq!(
            calculator.k_neighbors, 10,
            "k_neighbors should be 10 from config"
        );

        // Split point should be ~0.6 * 384 = 230
        let expected_split = (E11_DIM as f32 * 0.6) as usize;
        assert_eq!(
            calculator.split_point, expected_split,
            "split_point should be {} from config",
            expected_split
        );

        println!(
            "[PASS] test_transe_from_config - norm={}, split={}, k={}",
            calculator.norm, calculator.split_point, calculator.k_neighbors
        );
    }

    #[test]
    fn test_transe_head_extraction() {
        let calculator = TransEEntropy::new();

        let embedding: Vec<f32> = (0..E11_DIM).map(|i| i as f32).collect();
        let head = calculator.extract_head(&embedding);

        assert_eq!(head.len(), 192, "Head should be first 192 elements");
        assert_eq!(head[0], 0.0, "First element should be 0");
        assert_eq!(head[191], 191.0, "Last element should be 191");

        println!("[PASS] test_transe_head_extraction - Correct slice returned");
    }

    #[test]
    fn test_transe_relation_extraction() {
        let calculator = TransEEntropy::new();

        let embedding: Vec<f32> = (0..E11_DIM).map(|i| i as f32).collect();
        let relation = calculator.extract_relation(&embedding);

        assert_eq!(relation.len(), 192, "Relation should be last 192 elements");
        assert_eq!(relation[0], 192.0, "First element should be 192");
        assert_eq!(relation[191], 383.0, "Last element should be 383");

        println!("[PASS] test_transe_relation_extraction - Correct slice returned");
    }

    #[test]
    fn test_transe_distance_formula() {
        let calculator = TransEEntropy::new();

        // Manual calculation: h=[1.0; 192], r=[0.5; 192], t=[1.5; 192]
        // Expected: ||h + r - t|| = ||(1.0 + 0.5 - 1.5)|| = ||0|| = 0
        let head = vec![1.0f32; 192];
        let relation = vec![0.5f32; 192];
        let tail = vec![1.5f32; E11_DIM]; // Full embedding, will extract first 192

        let distance = calculator.compute_transe_distance(&head, &relation, &tail);

        println!("TransE distance: {} (expected ~0)", distance);
        assert!(
            distance < 0.001,
            "Perfect translation should have distance ~0, got {}",
            distance
        );

        println!("[PASS] test_transe_distance_formula - ||h + r - t|| correct");
    }

    #[test]
    fn test_transe_reset() {
        let mut calculator = TransEEntropy::new();

        // Modify internal state
        calculator.running_mean = 0.8;
        calculator.running_variance = 0.5;
        calculator.sample_count = 100;

        calculator.reset();

        assert_eq!(
            calculator.running_mean, 0.5,
            "running_mean should reset to 0.5"
        );
        assert_eq!(
            calculator.running_variance, 0.1,
            "running_variance should reset to 0.1"
        );
        assert_eq!(calculator.sample_count, 0, "sample_count should reset to 0");

        println!("[PASS] test_transe_reset - State cleared properly");
    }

    #[test]
    fn test_transe_nan_input_error() {
        let calculator = TransEEntropy::new();
        let mut current = vec![0.5f32; E11_DIM];
        current[0] = f32::NAN;

        let history = vec![vec![0.5f32; E11_DIM]];
        let result = calculator.compute_delta_s(&current, &history, 5);

        assert!(result.is_err(), "Should error on NaN input");
        assert!(
            matches!(result, Err(UtlError::EntropyError(_))),
            "Should return EntropyError for NaN input"
        );

        println!("[PASS] test_transe_nan_input_error - Err(EntropyError)");
    }

    #[test]
    fn test_transe_infinity_input_error() {
        let calculator = TransEEntropy::new();
        let mut current = vec![0.5f32; E11_DIM];
        current[0] = f32::INFINITY;

        let history = vec![vec![0.5f32; E11_DIM]];
        let result = calculator.compute_delta_s(&current, &history, 5);

        assert!(result.is_err(), "Should error on Infinity input");
        assert!(
            matches!(result, Err(UtlError::EntropyError(_))),
            "Should return EntropyError for Infinity input"
        );

        println!("[PASS] test_transe_infinity_input_error - Err(EntropyError)");
    }

    // === Edge Case Tests per Task Requirements ===

    #[test]
    fn test_edge_case_empty_history() {
        let calculator = TransEEntropy::new();

        let current = vec![0.5f32; E11_DIM];
        let history: Vec<Vec<f32>> = vec![];

        println!("BEFORE: history.len() = {}", history.len());
        let result = calculator.compute_delta_s(&current, &history, 5);
        println!("AFTER: delta_s = {:?}", result);

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1.0, "Empty history should return 1.0");

        println!("[PASS] test_edge_case_empty_history");
    }

    #[test]
    fn test_edge_case_perfect_translation_low_entropy() {
        let calculator = TransEEntropy::new();

        // current = [head(192) | relation(192)] where head + relation = history[0]
        let mut current = vec![0.0f32; E11_DIM];
        for i in 0..192 {
            current[i] = 0.5; // head = 0.5
        }
        for i in 192..E11_DIM {
            current[i] = 0.3; // relation = 0.3
        }
        let mut tail = vec![0.0f32; E11_DIM];
        for i in 0..192 {
            tail[i] = 0.8; // 0.5 + 0.3 = 0.8 (perfect translation)
        }
        let history = vec![tail; 10];

        println!("BEFORE: perfect translation setup");
        let result = calculator.compute_delta_s(&current, &history, 5);
        println!("AFTER: delta_s = {:?} (expected low, near 0)", result);

        assert!(result.is_ok());
        let delta_s = result.unwrap();
        assert!(
            delta_s < 0.6,
            "Perfect translation should have low surprise"
        );

        println!("[PASS] test_edge_case_perfect_translation_low_entropy");
    }

    #[test]
    fn test_edge_case_large_values() {
        let calculator = TransEEntropy::new();

        let current = vec![f32::MAX / 1000.0; E11_DIM];
        let history = vec![vec![f32::MAX / 1000.0; E11_DIM]; 5];

        println!("BEFORE: large values");
        let result = calculator.compute_delta_s(&current, &history, 5);
        println!("AFTER: delta_s = {:?}, must not be NaN/Inf", result);

        assert!(result.is_ok(), "Should handle large values");
        let delta_s = result.unwrap();
        assert!(!delta_s.is_nan(), "delta_s must not be NaN");
        assert!(!delta_s.is_infinite(), "delta_s must not be Infinite");

        println!("[PASS] test_edge_case_large_values");
    }

    #[test]
    fn test_builder_with_split_point() {
        let calc = TransEEntropy::new().with_split_point(100);
        assert_eq!(calc.split_point, 100, "Split point should be 100");

        // Test clamping to at least 1
        let calc2 = TransEEntropy::new().with_split_point(0);
        assert_eq!(calc2.split_point, 1, "Split point should be clamped to 1");

        println!("[PASS] test_builder_with_split_point");
    }

    #[test]
    fn test_builder_with_k_neighbors() {
        let calc = TransEEntropy::new().with_k_neighbors(7);
        assert_eq!(calc.k_neighbors, 7, "k_neighbors should be 7");

        // Test clamping
        let calc2 = TransEEntropy::new().with_k_neighbors(0);
        assert_eq!(calc2.k_neighbors, 1, "k_neighbors should be clamped to 1");

        let calc3 = TransEEntropy::new().with_k_neighbors(100);
        assert_eq!(calc3.k_neighbors, 20, "k_neighbors should be clamped to 20");

        println!("[PASS] test_builder_with_k_neighbors");
    }

    #[test]
    fn test_single_history_item() {
        let calculator = TransEEntropy::new();

        let current = vec![0.5f32; E11_DIM];
        let history = vec![vec![0.5f32; E11_DIM]]; // Only 1 item, but k=5

        println!("BEFORE: history.len()={}, k=5", history.len());

        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok(), "Should handle k > history.len()");
        let delta_s = result.unwrap();

        println!("AFTER: delta_s = {}", delta_s);
        assert!(
            (0.0..=1.0).contains(&delta_s),
            "delta_s should be in valid range"
        );

        println!("[PASS] test_single_history_item - delta_s = {}", delta_s);
    }

    #[test]
    fn test_history_with_empty_embeddings() {
        let calculator = TransEEntropy::new();

        let current = vec![0.5f32; E11_DIM];
        let mut history: Vec<Vec<f32>> = vec![vec![0.6f32; E11_DIM]; 5];
        history.push(vec![]); // Add empty embedding
        history.push(vec![]); // Add another empty

        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok(), "Should filter out empty history embeddings");
        let delta_s = result.unwrap();
        assert!((0.0..=1.0).contains(&delta_s));

        println!(
            "[PASS] test_history_with_empty_embeddings - delta_s = {}",
            delta_s
        );
    }

    #[test]
    fn test_history_with_nan_values() {
        let calculator = TransEEntropy::new();

        let current = vec![0.5f32; E11_DIM];
        let mut history: Vec<Vec<f32>> = vec![vec![0.6f32; E11_DIM]; 5];

        // Add embedding with NaN - should be filtered
        let mut nan_embedding = vec![0.5f32; E11_DIM];
        nan_embedding[0] = f32::NAN;
        history.push(nan_embedding);

        let result = calculator.compute_delta_s(&current, &history, 5);
        assert!(result.is_ok(), "Should filter out history with NaN values");
        let delta_s = result.unwrap();
        assert!((0.0..=1.0).contains(&delta_s));

        println!(
            "[PASS] test_history_with_nan_values - delta_s = {}",
            delta_s
        );
    }
}
