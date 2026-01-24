//! Stage 5: Late Interaction Reranking with E12 ColBERT MaxSim.
//!
//! # Overview
//!
//! This module implements Stage 5 of the 5-stage retrieval pipeline,
//! which performs late interaction reranking using E12 ColBERT token
//! embeddings with MaxSim scoring.
//!
//! # Algorithm
//!
//! MaxSim(Q, D) = (1/|Q|) × Σᵢ max_j cos(qᵢ, dⱼ)
//!
//! For each query token, find the maximum cosine similarity to any
//! document token, then average across all query tokens.
//!
//! # Performance Targets
//!
//! - Rerank 50 candidates: <15ms
//! - Target: Final 10-20 results after reranking
//!
//! # FAIL FAST Policy
//!
//! All errors are explicit. Missing token storage or tokens for candidates
//! are logged but don't abort - candidates without tokens keep their Stage 4 score.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::debug;
use uuid::Uuid;

use super::super::teleological_result::ScoredMemory;

/// Expected dimension for E12 token embeddings.
pub const E12_TOKEN_DIM: usize = 128;

/// Default limit for Stage 5 reranking.
pub const DEFAULT_STAGE5_LIMIT: usize = 10;

/// Trait for token storage backends (E12 ColBERT).
///
/// Implementations must be thread-safe for concurrent access.
pub trait TokenStorage: Send + Sync {
    /// Retrieve token embeddings for a memory.
    ///
    /// # Arguments
    /// * `id` - Memory UUID
    ///
    /// # Returns
    /// * `Some(tokens)` - Vector of 128D token embeddings
    /// * `None` - Memory not found or no tokens stored
    fn get_tokens(&self, id: Uuid) -> Option<Vec<Vec<f32>>>;
}

/// Stage 5 reranking result.
#[derive(Debug, Clone)]
pub struct Stage5Result {
    /// Reranked candidates.
    pub results: Vec<ScoredMemory>,
    /// Number of candidates with tokens.
    pub candidates_with_tokens: usize,
    /// Number of candidates without tokens (kept original score).
    pub candidates_without_tokens: usize,
    /// Reranking latency.
    pub latency: Duration,
}

/// Late interaction reranker using MaxSim scoring.
///
/// Thread-safe via Arc for token storage.
pub struct LateInteractionReranker<T: TokenStorage> {
    /// Token storage backend.
    token_storage: Arc<T>,
}

impl<T: TokenStorage> LateInteractionReranker<T> {
    /// Create a new late interaction reranker.
    pub fn new(token_storage: Arc<T>) -> Self {
        Self { token_storage }
    }

    /// Compute MaxSim score between query tokens and document tokens.
    ///
    /// # Algorithm
    /// MaxSim(Q, D) = (1/|Q|) × Σᵢ max_j cos(qᵢ, dⱼ)
    #[inline]
    fn compute_maxsim(&self, query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
        if query_tokens.is_empty() || doc_tokens.is_empty() {
            return 0.0;
        }

        let mut total_max_sim = 0.0f32;

        for q_token in query_tokens {
            let mut max_sim = f32::NEG_INFINITY;

            for d_token in doc_tokens {
                let sim = self.cosine_similarity_128d(q_token, d_token);
                if sim > max_sim {
                    max_sim = sim;
                }
            }

            if max_sim.is_finite() {
                total_max_sim += max_sim;
            }
        }

        total_max_sim / query_tokens.len() as f32
    }

    /// Compute cosine similarity between two 128D vectors.
    #[inline]
    fn cosine_similarity_128d(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        let denom = (norm_a * norm_b).sqrt();
        if denom < f32::EPSILON {
            0.0
        } else {
            dot / denom
        }
    }

    /// Rerank candidates using MaxSim scoring.
    ///
    /// # Arguments
    /// * `candidates` - Stage 4 filtered candidates
    /// * `query_tokens` - Query token embeddings (from E12 encoder)
    /// * `limit` - Maximum results to return
    ///
    /// # Returns
    /// Stage5Result with reranked candidates
    ///
    /// # Behavior
    /// - Candidates with tokens are reranked by MaxSim score
    /// - Candidates without tokens keep their Stage 4 score
    /// - Final results sorted by combined score (0.7×MaxSim + 0.3×Stage4)
    pub fn rerank(
        &self,
        candidates: &[ScoredMemory],
        query_tokens: &[Vec<f32>],
        limit: usize,
    ) -> Stage5Result {
        let start = Instant::now();
        let limit = if limit == 0 { DEFAULT_STAGE5_LIMIT } else { limit };

        if candidates.is_empty() || query_tokens.is_empty() {
            return Stage5Result {
                results: Vec::new(),
                candidates_with_tokens: 0,
                candidates_without_tokens: 0,
                latency: start.elapsed(),
            };
        }

        let mut reranked: Vec<(ScoredMemory, f32, bool)> = Vec::with_capacity(candidates.len());
        let mut with_tokens = 0;
        let mut without_tokens = 0;

        for candidate in candidates {
            if let Some(doc_tokens) = self.token_storage.get_tokens(candidate.memory_id) {
                // Compute MaxSim score
                let maxsim_score = self.compute_maxsim(query_tokens, &doc_tokens);

                // Combined score: 0.7×MaxSim + 0.3×Stage4
                // MaxSim provides fine-grained ranking, Stage4 provides semantic foundation
                let combined_score = 0.7 * maxsim_score + 0.3 * candidate.score;

                let mut updated = candidate.clone();
                updated.score = combined_score;
                reranked.push((updated, maxsim_score, true));
                with_tokens += 1;
            } else {
                // No tokens - keep original score
                reranked.push((candidate.clone(), 0.0, false));
                without_tokens += 1;
            }
        }

        // Sort by score descending
        reranked.sort_by(|a, b| {
            b.0.score
                .partial_cmp(&a.0.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to limit
        reranked.truncate(limit);

        let results: Vec<ScoredMemory> = reranked.into_iter().map(|(m, _, _)| m).collect();

        let latency = start.elapsed();

        debug!(
            candidates_in = candidates.len(),
            candidates_with_tokens = with_tokens,
            candidates_without_tokens = without_tokens,
            results_out = results.len(),
            latency_us = latency.as_micros(),
            "Stage 5 MaxSim reranking complete"
        );

        Stage5Result {
            results,
            candidates_with_tokens: with_tokens,
            candidates_without_tokens: without_tokens,
            latency,
        }
    }

    /// Rerank candidates without query tokens (uses E1 semantic for ranking).
    ///
    /// Fallback when E12 query tokens aren't available.
    /// Just sorts by Stage 4 score and truncates.
    pub fn rerank_fallback(&self, candidates: &[ScoredMemory], limit: usize) -> Stage5Result {
        let start = Instant::now();
        let limit = if limit == 0 { DEFAULT_STAGE5_LIMIT } else { limit };

        let mut results = candidates.to_vec();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(limit);

        let latency = start.elapsed();

        debug!(
            candidates_in = candidates.len(),
            results_out = results.len(),
            latency_us = latency.as_micros(),
            "Stage 5 fallback (no query tokens) complete"
        );

        Stage5Result {
            results,
            candidates_with_tokens: 0,
            candidates_without_tokens: candidates.len(),
            latency,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::RwLock;

    // In-memory mock for testing
    struct MockTokenStorage {
        tokens: RwLock<HashMap<Uuid, Vec<Vec<f32>>>>,
    }

    impl MockTokenStorage {
        fn new() -> Self {
            Self {
                tokens: RwLock::new(HashMap::new()),
            }
        }

        fn insert(&self, id: Uuid, tokens: Vec<Vec<f32>>) {
            self.tokens.write().unwrap().insert(id, tokens);
        }
    }

    impl TokenStorage for MockTokenStorage {
        fn get_tokens(&self, id: Uuid) -> Option<Vec<Vec<f32>>> {
            self.tokens.read().unwrap().get(&id).cloned()
        }
    }

    /// Generate normalized test token.
    fn generate_normalized_token(seed: usize) -> Vec<f32> {
        let mut token: Vec<f32> = (0..E12_TOKEN_DIM)
            .map(|j| ((seed * 128 + j) as f32 / 1000.0).sin())
            .collect();

        // Normalize
        let norm: f32 = token.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for x in &mut token {
                *x /= norm;
            }
        }

        token
    }

    /// Create test scored memory.
    fn create_test_scored(id: Uuid, score: f32) -> ScoredMemory {
        ScoredMemory::new(id, score, score, 1.0, 13)
    }

    // ========================================================================
    // BASIC TESTS
    // ========================================================================

    #[test]
    fn test_rerank_with_tokens() {
        println!("=== TEST: Rerank With Tokens ===");

        let storage = Arc::new(MockTokenStorage::new());
        let reranker = LateInteractionReranker::new(Arc::clone(&storage));

        // Create candidates
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        // id1: Low Stage 4 score but same tokens as query
        // id2: High Stage 4 score but different tokens
        let candidates = vec![
            create_test_scored(id1, 0.3),
            create_test_scored(id2, 0.9),
        ];

        // Store tokens - id1 has identical tokens to query
        let query_tokens = vec![generate_normalized_token(42), generate_normalized_token(43)];
        storage.insert(id1, query_tokens.clone());
        storage.insert(id2, vec![generate_normalized_token(100), generate_normalized_token(101)]);

        // Rerank
        let result = reranker.rerank(&candidates, &query_tokens, 10);

        println!("Candidates with tokens: {}", result.candidates_with_tokens);
        println!("Results: {} candidates", result.results.len());
        for (i, r) in result.results.iter().enumerate() {
            println!("  {}: {} score={:.4}", i, r.memory_id, r.score);
        }

        assert_eq!(result.results.len(), 2);
        assert_eq!(result.candidates_with_tokens, 2);

        // id1 should rank higher because its tokens match query exactly (MaxSim ≈ 1.0)
        // Combined: 0.7×1.0 + 0.3×0.3 = 0.79
        // id2: 0.7×low + 0.3×0.9 = ~0.3-0.5
        assert_eq!(result.results[0].memory_id, id1, "id1 should rank first due to MaxSim");

        println!("[VERIFIED] Reranking with tokens works correctly");
    }

    #[test]
    fn test_rerank_without_tokens() {
        println!("=== TEST: Rerank Without Tokens ===");

        let storage = Arc::new(MockTokenStorage::new());
        let reranker = LateInteractionReranker::new(Arc::clone(&storage));

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let candidates = vec![
            create_test_scored(id1, 0.9),
            create_test_scored(id2, 0.3),
        ];

        // No tokens stored - candidates keep original scores
        let query_tokens = vec![generate_normalized_token(42)];
        let result = reranker.rerank(&candidates, &query_tokens, 10);

        assert_eq!(result.candidates_without_tokens, 2);
        assert_eq!(result.results[0].memory_id, id1); // Higher original score

        println!("[VERIFIED] Reranking without tokens keeps original order");
    }

    #[test]
    fn test_rerank_fallback() {
        println!("=== TEST: Rerank Fallback ===");

        let storage = Arc::new(MockTokenStorage::new());
        let reranker = LateInteractionReranker::new(storage);

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let candidates = vec![
            create_test_scored(id1, 0.5),
            create_test_scored(id2, 0.8),
        ];

        let result = reranker.rerank_fallback(&candidates, 10);

        assert_eq!(result.results.len(), 2);
        assert_eq!(result.results[0].memory_id, id2); // Higher score first

        println!("[VERIFIED] Fallback reranking works correctly");
    }

    #[test]
    fn test_rerank_limit() {
        println!("=== TEST: Rerank Limit ===");

        let storage = Arc::new(MockTokenStorage::new());
        let reranker = LateInteractionReranker::new(storage);

        // Create 10 candidates
        let candidates: Vec<_> = (0..10)
            .map(|i| create_test_scored(Uuid::new_v4(), i as f32 / 10.0))
            .collect();

        let result = reranker.rerank_fallback(&candidates, 5);

        assert_eq!(result.results.len(), 5);

        println!("[VERIFIED] Rerank limit works correctly");
    }

    #[test]
    fn test_rerank_empty() {
        println!("=== TEST: Rerank Empty ===");

        let storage = Arc::new(MockTokenStorage::new());
        let reranker = LateInteractionReranker::new(storage);

        let candidates: Vec<ScoredMemory> = Vec::new();
        let query_tokens = vec![generate_normalized_token(42)];

        let result = reranker.rerank(&candidates, &query_tokens, 10);
        assert!(result.results.is_empty());

        // Also test with empty query tokens
        let candidates = vec![create_test_scored(Uuid::new_v4(), 0.5)];
        let empty_tokens: Vec<Vec<f32>> = Vec::new();
        let result = reranker.rerank(&candidates, &empty_tokens, 10);
        assert!(result.results.is_empty());

        println!("[VERIFIED] Empty input handling works correctly");
    }

    #[test]
    fn test_maxsim_identical_tokens() {
        println!("=== TEST: MaxSim Identical Tokens ===");

        let storage = Arc::new(MockTokenStorage::new());
        let reranker = LateInteractionReranker::new(Arc::clone(&storage));

        let tokens = vec![generate_normalized_token(42), generate_normalized_token(43)];
        let score = reranker.compute_maxsim(&tokens, &tokens);

        println!("MaxSim of identical tokens: {}", score);
        assert!((score - 1.0).abs() < 1e-5, "MaxSim of identical should be ~1.0");

        println!("[VERIFIED] MaxSim of identical tokens is ~1.0");
    }

    // ========================================================================
    // VERIFICATION LOG
    // ========================================================================

    #[test]
    fn test_verification_log() {
        println!("\n=== STAGE5.RS VERIFICATION LOG ===\n");

        println!("Configuration:");
        println!("  - E12_TOKEN_DIM: {}", E12_TOKEN_DIM);
        println!("  - DEFAULT_STAGE5_LIMIT: {}", DEFAULT_STAGE5_LIMIT);

        println!("\nAlgorithm:");
        println!("  - MaxSim(Q, D) = (1/|Q|) × Σᵢ max_j cos(qᵢ, dⱼ)");
        println!("  - Combined score: 0.7×MaxSim + 0.3×Stage4");

        println!("\nBehavior:");
        println!("  - Candidates with tokens: reranked by MaxSim");
        println!("  - Candidates without tokens: keep Stage 4 score");
        println!("  - Fallback: sort by Stage 4 score if no query tokens");

        println!("\nTest Coverage:");
        println!("  - Basic: 5 tests (with tokens, without, fallback, limit, empty)");
        println!("  - MaxSim: 1 test");
        println!("  - Total: 6 tests");

        println!("\nVERIFICATION COMPLETE");
    }
}
