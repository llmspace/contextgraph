//! Distance and similarity metrics for the 13 embedding spaces.
//!
//! This module provides unified distance/similarity computation across all
//! embedding types: dense, sparse, binary, and token-level.
//!
//! # Design Philosophy
//!
//! Most similarity functions delegate to existing vector type methods:
//! - DenseVector::cosine_similarity()
//! - SparseVector::jaccard_similarity()
//! - BinaryVector::hamming_distance()
//!
//! This module adds:
//! - max_sim() for ColBERT late interaction (E12)
//! - transe_similarity() for knowledge graph embeddings (E11)
//! - compute_similarity_for_space() dispatcher
//!
//! # All outputs are normalized to [0.0, 1.0]

use crate::embeddings::{BinaryVector, DenseVector};
use crate::teleological::Embedder;
use crate::types::fingerprint::{EmbeddingRef, SemanticFingerprint, SparseVector};

/// Compute cosine similarity between two dense vectors.
///
/// Thin wrapper that creates DenseVectors and delegates to existing method.
/// Returns 0.0 for zero-magnitude vectors (AP-10: no NaN).
///
/// # Arguments
/// * `a` - First dense embedding as f32 slice
/// * `b` - Second dense embedding as f32 slice
///
/// # Returns
/// Similarity in [0.0, 1.0] where 1.0 = identical direction
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    // Check for zero vectors before computation (AP-10 compliance)
    let mag_a_sq: f32 = a.iter().map(|x| x * x).sum();
    let mag_b_sq: f32 = b.iter().map(|x| x * x).sum();
    if mag_a_sq == 0.0 || mag_b_sq == 0.0 {
        return 0.0;
    }

    let vec_a = DenseVector::new(a.to_vec());
    let vec_b = DenseVector::new(b.to_vec());

    let raw_sim = vec_a.cosine_similarity(&vec_b);

    // Normalize from [-1, 1] to [0, 1]
    // DenseVector.cosine_similarity already clamps to [-1, 1]
    (raw_sim + 1.0) / 2.0
}

/// Compute Jaccard similarity between two sparse vectors.
///
/// Thin wrapper that delegates to SparseVector::jaccard_similarity().
/// Returns |A ∩ B| / |A ∪ B| based on non-zero indices.
///
/// # Returns
/// Similarity in [0.0, 1.0] where 1.0 = identical index sets
pub fn jaccard_similarity(a: &SparseVector, b: &SparseVector) -> f32 {
    a.jaccard_similarity(b)
}

/// Compute Hamming similarity between two binary vectors.
///
/// Converts Hamming distance to similarity: 1.0 - (distance / max_bits).
///
/// # Returns
/// Similarity in [0.0, 1.0] where 1.0 = identical bit patterns
pub fn hamming_similarity(a: &BinaryVector, b: &BinaryVector) -> f32 {
    let distance = a.hamming_distance(b);
    let max_bits = a.bit_len().max(b.bit_len());

    if max_bits == 0 {
        return 1.0; // Empty vectors are identical
    }

    1.0 - (distance as f32 / max_bits as f32)
}

/// Compute MaxSim for late interaction (ColBERT-style).
///
/// For each query token, find max cosine similarity to any memory token.
/// Return mean of all max similarities.
///
/// # Algorithm
/// ```text
/// MaxSim = (1/|Q|) * Σ_q∈Q max_m∈M cos(q, m)
/// ```
///
/// # Arguments
/// * `query_tokens` - Query token embeddings (each 128D for E12)
/// * `memory_tokens` - Memory token embeddings
///
/// # Returns
/// Similarity in [0.0, 1.0], returns 0.0 if either list is empty
pub fn max_sim(query_tokens: &[Vec<f32>], memory_tokens: &[Vec<f32>]) -> f32 {
    if query_tokens.is_empty() || memory_tokens.is_empty() {
        return 0.0;
    }

    let mut total_max = 0.0_f32;

    for q_tok in query_tokens {
        let mut max_sim_for_token = 0.0_f32;

        for m_tok in memory_tokens {
            let sim = cosine_similarity(q_tok, m_tok);
            max_sim_for_token = max_sim_for_token.max(sim);
        }

        total_max += max_sim_for_token;
    }

    total_max / query_tokens.len() as f32
}

/// Compute TransE-style similarity for knowledge graph triplet scoring.
///
/// Uses inverse of Euclidean distance: 1 / (1 + distance).
/// This maps distance [0, ∞) to similarity (0, 1].
///
/// # Important
///
/// This function is designed for TransE triplet operations (h + r - t),
/// NOT for general entity-entity similarity. For general E11 similarity,
/// use `cosine_similarity()` instead.
///
/// This function is used by:
/// - `infer_relationship` MCP tool (computing predicted relation vectors)
/// - `validate_knowledge` MCP tool (scoring (subject, predicate, object) triples)
///
/// # Returns
/// Similarity in (0.0, 1.0] where 1.0 = identical vectors (distance = 0)
pub fn transe_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let vec_a = DenseVector::new(a.to_vec());
    let vec_b = DenseVector::new(b.to_vec());

    let distance = vec_a.euclidean_distance(&vec_b);
    1.0 / (1.0 + distance)
}

/// Compute similarity for a specific embedding space.
///
/// This is the main dispatcher that routes to the appropriate similarity
/// function based on the embedder type.
///
/// # Metrics by Embedder
/// - E1 (Semantic): Cosine
/// - E2-E4 (Temporal): Cosine
/// - E5 (Causal): Cosine (asymmetric handled at embedding time via dual vectors)
/// - E6 (Sparse): Jaccard
/// - E7 (Code): Cosine (query-type detection handled at embedding time)
/// - E8 (Graph): Cosine
/// - E9 (HDC): Cosine on projected dense (see note below)
/// - E10 (Multimodal): Cosine
/// - E11 (Entity): Cosine (TransE used only for triplet operations in entity tools)
/// - E12 (LateInteraction): MaxSim (used for Stage 3 re-ranking only)
/// - E13 (KeywordSplade): Jaccard (used for Stage 1 recall only)
///
/// # E9 HDC Note
///
/// E9 uses 10,000-bit native hypervectors internally but projects to 1024D dense
/// for storage and indexing compatibility (see constants.rs). Cosine similarity
/// on the projected representation is used. For true Hamming distance on binary
/// HDC vectors, the `hamming_similarity()` function with `BinaryVector` can be
/// used if native binary storage is implemented in the future.
///
/// # E11 Entity Note
///
/// E11 uses cosine similarity for general entity-entity comparison. The TransE
/// similarity function (transe_similarity) is reserved for specific knowledge
/// graph operations in entity_tools (infer_relationship, validate_knowledge)
/// where the triplet scoring formula ||h + r - t|| is semantically meaningful.
///
/// # Arguments
/// * `embedder` - Which embedding space to compare
/// * `query` - Query fingerprint
/// * `memory` - Memory fingerprint
///
/// # Returns
/// Similarity in [0.0, 1.0]
pub fn compute_similarity_for_space(
    embedder: Embedder,
    query: &SemanticFingerprint,
    memory: &SemanticFingerprint,
) -> f32 {
    // EMB-7 FIX: E5 MUST NOT use symmetric cosine per AP-77.
    // EMB-2 FIX: E8 and E10 have asymmetric dual vectors that were computed/stored
    // but never used in search. Use cross-pair comparison (source-vs-target, paraphrase-vs-context)
    // to produce a more informative similarity score.
    match embedder {
        Embedder::Causal => {
            // EMB-7 FIX: E5 without direction returns -1.0 (sentinel = "no signal").
            // Use compute_similarity_for_space_with_direction() for directional E5 similarity.
            // CORE-H1 FIX: -1.0 sentinel distinguishes "no signal" from 0.0 (anti-correlated).
            -1.0
        }
        Embedder::Graph => {
            // E8: Compare source-vs-target cross pairs and take max
            let source_vs_target = cosine_similarity(
                query.get_e8_as_source(),
                memory.get_e8_as_target(),
            );
            let target_vs_source = cosine_similarity(
                query.get_e8_as_target(),
                memory.get_e8_as_source(),
            );
            // Take max of both directions — the stronger signal wins
            source_vs_target.max(target_vs_source)
        }
        Embedder::Contextual => {
            // E10: Compare paraphrase-vs-context cross pairs and take max
            let para_vs_context = cosine_similarity(
                query.get_e10_as_paraphrase(),
                memory.get_e10_as_context(),
            );
            let context_vs_para = cosine_similarity(
                query.get_e10_as_context(),
                memory.get_e10_as_paraphrase(),
            );
            // Take max of both directions — captures paraphrase detection
            para_vs_context.max(context_vs_para)
        }
        _ => {
            // All other embedders use standard symmetric comparison
            let query_ref = query.get(embedder);
            let memory_ref = memory.get(embedder);

            let query_disc = std::mem::discriminant(&query_ref);
            let memory_disc = std::mem::discriminant(&memory_ref);

            match (query_ref, memory_ref) {
                (EmbeddingRef::Dense(q), EmbeddingRef::Dense(m)) => {
                    cosine_similarity(q, m)
                }
                (EmbeddingRef::Sparse(q), EmbeddingRef::Sparse(m)) => jaccard_similarity(q, m),
                (EmbeddingRef::TokenLevel(q), EmbeddingRef::TokenLevel(m)) => max_sim(q, m),
                _ => {
                    panic!(
                        "BUG: Type mismatch in compute_similarity_for_space for embedder {}. \
                         query={:?}, memory={:?}. This indicates a corrupted SemanticFingerprint.",
                        embedder.name(),
                        query_disc,
                        memory_disc,
                    );
                }
            }
        }
    }
}

/// Compute similarity for a specific embedding space with causal direction.
///
/// This function extends `compute_similarity_for_space()` with direction-aware
/// E5 similarity computation per ARCH-15 and AP-77.
///
/// When `causal_direction` is `Cause` or `Effect`, E5 similarity uses:
/// - Asymmetric vectors: query.e5_as_cause vs doc.e5_as_effect (or reverse)
/// - Direction modifiers: cause→effect (1.2x), effect→cause (0.8x)
///
/// For all other embedders and when direction is `Unknown`, behaves identically
/// to `compute_similarity_for_space()`.
///
/// # Arguments
/// * `embedder` - Which embedding space to compare
/// * `query` - Query fingerprint
/// * `memory` - Memory fingerprint
/// * `causal_direction` - Detected causal direction of the query
///
/// # Returns
/// Similarity in [0.0, 1.0], with direction modifier applied for E5 causal
pub fn compute_similarity_for_space_with_direction(
    embedder: Embedder,
    query: &SemanticFingerprint,
    memory: &SemanticFingerprint,
    causal_direction: crate::causal::asymmetric::CausalDirection,
) -> f32 {
    use crate::causal::asymmetric::{
        compute_e5_asymmetric_fingerprint_similarity, direction_mod, CausalDirection,
    };

    // AP-77: E5 MUST NOT use symmetric cosine — causal is directional.
    if matches!(embedder, Embedder::Causal) {
        if causal_direction == CausalDirection::Unknown {
            // No direction known → E5 cannot provide meaningful signal.
            // CORE-H1 FIX: Return -1.0 sentinel (not 0.0) so fusion correctly
            // distinguishes "no signal" from 0.0 (anti-correlated after normalization).
            return -1.0;
        }

        let query_is_cause = matches!(causal_direction, CausalDirection::Cause);

        // Compute asymmetric similarity using dual E5 vectors
        let asym_sim = compute_e5_asymmetric_fingerprint_similarity(query, memory, query_is_cause);

        // Infer result direction from document's E5 vectors
        let result_direction = infer_direction_from_fingerprint(memory);

        // Apply Constitution-specified direction modifier
        let dir_mod = match (causal_direction, result_direction) {
            (CausalDirection::Cause, CausalDirection::Effect) => direction_mod::CAUSE_TO_EFFECT,
            (CausalDirection::Effect, CausalDirection::Cause) => direction_mod::EFFECT_TO_CAUSE,
            _ => direction_mod::SAME_DIRECTION,
        };

        return (asym_sim * dir_mod).clamp(0.0, 1.0);
    }

    // Default: symmetric computation for all other embedders
    compute_similarity_for_space(embedder, query, memory)
}

/// Infer causal direction from a stored fingerprint's E5 vectors.
///
/// Delegates to the canonical implementation in `causal::asymmetric`.
fn infer_direction_from_fingerprint(
    fp: &SemanticFingerprint,
) -> crate::causal::asymmetric::CausalDirection {
    crate::causal::asymmetric::infer_direction_from_fingerprint(fp)
}

/// Compute all 13 similarities between query and memory fingerprints.
///
/// Returns an array indexed by Embedder::index().
///
/// # Returns
/// Array of 13 similarity scores in [0.0, 1.0]
pub fn compute_all_similarities(
    query: &SemanticFingerprint,
    memory: &SemanticFingerprint,
) -> [f32; 13] {
    let mut scores = [0.0_f32; 13];

    for embedder in Embedder::all() {
        scores[embedder.index()] = compute_similarity_for_space(embedder, query, memory);
    }

    scores
}

/// Compute all 13 similarities with causal direction for E5.
///
/// Like `compute_all_similarities()` but uses asymmetric E5 similarity
/// when a causal direction is provided.
///
/// # Arguments
/// * `query` - Query fingerprint
/// * `memory` - Memory fingerprint
/// * `causal_direction` - Detected causal direction of the query
///
/// # Returns
/// Array of 13 similarity scores. Valid scores are in [0.0, 1.0].
/// E5 returns -1.0 sentinel when `causal_direction` is `Unknown` (CORE-H1: no signal).
pub fn compute_all_similarities_with_direction(
    query: &SemanticFingerprint,
    memory: &SemanticFingerprint,
    causal_direction: crate::causal::asymmetric::CausalDirection,
) -> [f32; 13] {
    let mut scores = [0.0_f32; 13];

    for embedder in Embedder::all() {
        scores[embedder.index()] =
            compute_similarity_for_space_with_direction(embedder, query, memory, causal_direction);
    }

    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_core_cases() {
        // Identical
        let v: Vec<f32> = vec![0.6, 0.8, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-5);

        // Orthogonal: raw=0.0, normalized=0.5
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 0.5).abs() < 1e-5);

        // Zero vector: AP-10 compliance
        let zero = vec![0.0, 0.0, 0.0];
        let normal = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&zero, &normal), 0.0);
        assert!(!cosine_similarity(&zero, &zero).is_nan());

        // Empty and dimension mismatch
        assert_eq!(cosine_similarity(&[], &normal), 0.0);
        assert_eq!(cosine_similarity(&[1.0, 2.0], &normal), 0.0);
    }

    #[test]
    fn test_jaccard_similarity_cases() {
        // Identical
        let v = SparseVector::new(vec![0, 5, 10], vec![1.0, 1.0, 1.0]).unwrap();
        assert!((jaccard_similarity(&v, &v) - 1.0).abs() < 1e-5);

        // Partial overlap: {1,2} / {0,1,2,3} = 0.5
        let a = SparseVector::new(vec![0, 1, 2], vec![1.0, 1.0, 1.0]).unwrap();
        let b = SparseVector::new(vec![1, 2, 3], vec![1.0, 1.0, 1.0]).unwrap();
        assert!((jaccard_similarity(&a, &b) - 0.5).abs() < 1e-5);

        // Empty
        assert_eq!(jaccard_similarity(&SparseVector::empty(), &SparseVector::empty()), 0.0);
    }

    #[test]
    fn test_max_sim_cases() {
        // Identical token sets
        let tokens = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        assert!((max_sim(&tokens, &tokens) - 1.0).abs() < 1e-5);

        // Partial match: (1.0 + 0.5) / 2 = 0.75
        let query = vec![vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0]];
        let memory = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        assert!((max_sim(&query, &memory) - 0.75).abs() < 1e-5);

        // Empty
        let empty: Vec<Vec<f32>> = vec![];
        assert_eq!(max_sim(&empty, &tokens), 0.0);
        assert_eq!(max_sim(&tokens, &empty), 0.0);
    }

    #[test]
    fn test_transe_similarity_cases() {
        // Identical: distance=0, sim=1.0
        let v = vec![1.0, 2.0, 3.0];
        assert!((transe_similarity(&v, &v) - 1.0).abs() < 1e-5);

        // Unit distance: sim = 1/(1+1) = 0.5
        assert!((transe_similarity(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]) - 0.5).abs() < 1e-5);

        // Empty
        assert_eq!(transe_similarity(&[], &v), 0.0);
    }

    #[test]
    fn test_compute_similarity_for_space_dispatch() {
        let mut query = SemanticFingerprint::zeroed();
        let mut memory = SemanticFingerprint::zeroed();

        // Semantic (dense cosine)
        query.e1_semantic = vec![1.0; 1024];
        memory.e1_semantic = vec![1.0; 1024];
        assert!((compute_similarity_for_space(Embedder::Semantic, &query, &memory) - 1.0).abs() < 1e-5);

        // Sparse (jaccard)
        query.e6_sparse = SparseVector::new(vec![0, 5, 10], vec![1.0, 1.0, 1.0]).unwrap();
        memory.e6_sparse = SparseVector::new(vec![0, 5, 10], vec![1.0, 1.0, 1.0]).unwrap();
        assert!((compute_similarity_for_space(Embedder::Sparse, &query, &memory) - 1.0).abs() < 1e-5);

        // Late interaction (MaxSim)
        query.e12_late_interaction = vec![vec![1.0; 128], vec![0.5; 128]];
        memory.e12_late_interaction = vec![vec![1.0; 128], vec![0.5; 128]];
        assert!((compute_similarity_for_space(Embedder::LateInteraction, &query, &memory) - 1.0).abs() < 1e-5);

        // Entity uses cosine, orthogonal = 0.5
        query.e11_entity = vec![0.0; 768];
        memory.e11_entity = vec![0.0; 768];
        query.e11_entity[0] = 1.0;
        memory.e11_entity[1] = 1.0;
        assert!((compute_similarity_for_space(Embedder::Entity, &query, &memory) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_compute_all_similarities() {
        let query = SemanticFingerprint::zeroed();
        let memory = SemanticFingerprint::zeroed();

        let scores = compute_all_similarities(&query, &memory);
        assert_eq!(scores.len(), 13);
        for (i, score) in scores.iter().enumerate() {
            assert!(!score.is_nan());
            if i == 4 {
                // CORE-H1: E5 returns -1.0 sentinel without direction
                assert_eq!(*score, -1.0, "E5 should return -1.0 sentinel");
            } else {
                assert!(*score >= 0.0 && *score <= 1.0, "E{} score {} out of range", i + 1, score);
            }
        }
    }

    #[test]
    fn test_edge_cases_no_nan_or_overflow() {
        // Very small values
        let small = vec![1e-20_f32; 3];
        let sim = cosine_similarity(&small, &small);
        assert!(!sim.is_nan() && !sim.is_infinite() && (0.0..=1.0).contains(&sim));

        // Very large values
        let large = vec![1e19_f32; 3];
        let sim = cosine_similarity(&large, &large);
        assert!(!sim.is_nan() && !sim.is_infinite() && (0.0..=1.0).contains(&sim));

        // Single-token opposite MaxSim
        let sim = max_sim(&[vec![1.0_f32]], &[vec![-1.0_f32]]);
        assert!(sim.abs() < 1e-5 && sim >= 0.0);

        // All 13 spaces with zeroed fingerprints
        // CORE-H1: E5 returns -1.0 sentinel (no signal) when no direction is provided.
        let zeroed = SemanticFingerprint::zeroed();
        for (idx, score) in compute_all_similarities(&zeroed, &zeroed).iter().enumerate() {
            if idx == 4 {
                // E5 (Causal) returns -1.0 sentinel without direction
                assert_eq!(*score, -1.0, "E5 should return -1.0 sentinel without direction");
            } else {
                assert!(!score.is_nan() && !score.is_infinite() && (0.0..=1.0).contains(score),
                    "E{} score {} out of range", idx + 1, score);
            }
        }
    }

    #[test]
    fn test_direction_aware_e5_ap77() {
        use crate::causal::asymmetric::CausalDirection;

        let mut query = SemanticFingerprint::zeroed();
        let mut memory = SemanticFingerprint::zeroed();

        query.e5_causal_as_cause = vec![1.0; 768];
        query.e5_causal_as_effect = vec![0.5; 768];
        memory.e5_causal_as_cause = vec![1.0; 768];
        memory.e5_causal_as_effect = vec![0.5; 768];

        // Unknown direction: E5 returns -1.0 sentinel per AP-77 + CORE-H1 fix
        assert_eq!(
            compute_similarity_for_space_with_direction(Embedder::Causal, &query, &memory, CausalDirection::Unknown),
            -1.0,
        );

        // Known direction: non-zero
        let cause_sim = compute_similarity_for_space_with_direction(
            Embedder::Causal, &query, &memory, CausalDirection::Cause,
        );
        assert!(cause_sim > 0.0 && cause_sim <= 1.0);

        // Non-E5 embedders ignore direction
        query.e1_semantic = vec![1.0; 1024];
        memory.e1_semantic = vec![1.0; 1024];
        let sym = compute_similarity_for_space(Embedder::Semantic, &query, &memory);
        let with_cause = compute_similarity_for_space_with_direction(
            Embedder::Semantic, &query, &memory, CausalDirection::Cause,
        );
        assert!((sym - with_cause).abs() < 1e-5);
    }

    #[test]
    fn test_direction_modifier_values() {
        use crate::causal::asymmetric::direction_mod;

        assert!((direction_mod::CAUSE_TO_EFFECT - 1.2).abs() < 1e-5);
        assert!((direction_mod::EFFECT_TO_CAUSE - 0.8).abs() < 1e-5);
        assert!((direction_mod::SAME_DIRECTION - 1.0).abs() < 1e-5);
        let ratio = direction_mod::CAUSE_TO_EFFECT / direction_mod::EFFECT_TO_CAUSE;
        assert!((ratio - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_compute_all_similarities_with_direction() {
        use crate::causal::asymmetric::CausalDirection;

        let query = SemanticFingerprint::zeroed();
        let memory = SemanticFingerprint::zeroed();

        for dir in [CausalDirection::Unknown, CausalDirection::Cause, CausalDirection::Effect] {
            let scores = compute_all_similarities_with_direction(&query, &memory, dir);
            assert_eq!(scores.len(), 13);
            for (idx, score) in scores.iter().enumerate() {
                if idx == 4 && dir == CausalDirection::Unknown {
                    // E5 returns -1.0 sentinel for Unknown direction (no causal signal)
                    assert!((*score - (-1.0)).abs() < 1e-6, "E5 with Unknown should be -1.0 sentinel, got {score}");
                } else {
                    assert!(*score >= 0.0 && *score <= 1.0 && !score.is_nan(),
                        "score[{idx}] with dir {dir:?} out of range: {score}");
                }
            }
        }
    }
}
