//! Sparse vector similarity functions for E6/E13 (SPLADE) embeddings.
//!
//! These functions operate on `SparseVector` types with ~30K vocabulary
//! but typically only 100-1000 active dimensions (~5% sparsity).
//!
//! # Performance
//!
//! Merge-join on sorted indices gives O(n + m) complexity where n, m
//! are the number of non-zero entries.
//!
//! # Sparse Embedders
//!
//! | Embedder | Vocab Size | Typical Sparsity |
//! |----------|------------|------------------|
//! | E6       | 30,522     | ~5% (1500 nnz)   |
//! | E13      | 30,522     | ~5% (1500 nnz)   |

use crate::types::fingerprint::SparseVector;
use std::cmp::Ordering;
use std::collections::HashMap;
use thiserror::Error;

/// Errors from sparse vector similarity computation.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum SparseSimilarityError {
    /// Empty sparse vector provided.
    #[error("Empty sparse vector provided")]
    EmptyVector,

    /// Index exceeds vocabulary size.
    #[error("Invalid index {index} exceeds vocabulary size {vocab_size}")]
    IndexOutOfBounds { index: usize, vocab_size: usize },

    /// Indices not sorted or contain duplicates.
    #[error("Indices not sorted or contain duplicates")]
    UnsortedIndices,

    /// Indices and values length mismatch.
    #[error("Indices and values length mismatch: indices={indices_len}, values={values_len}")]
    LengthMismatch {
        indices_len: usize,
        values_len: usize,
    },
}

/// Calculate dot product between two sparse vectors.
///
/// Uses merge-join algorithm on sorted indices for O(n + m) complexity.
/// Returns 0.0 if either vector is empty (valid sparse behavior).
///
/// # Example
///
/// ```rust,ignore
/// use context_graph_core::types::fingerprint::SparseVector;
/// use context_graph_core::similarity::sparse_dot_product;
///
/// let a = SparseVector::new(vec![1, 3, 5], vec![1.0, 2.0, 3.0]).unwrap();
/// let b = SparseVector::new(vec![2, 3, 5], vec![4.0, 5.0, 6.0]).unwrap();
/// // Intersection at indices 3, 5: 2.0*5.0 + 3.0*6.0 = 28.0
/// let dot = sparse_dot_product(&a, &b);
/// assert!((dot - 28.0).abs() < 1e-6);
/// ```
#[inline]
pub fn sparse_dot_product(a: &SparseVector, b: &SparseVector) -> f32 {
    // Delegate to SparseVector's dot method which uses merge-join
    a.dot(b)
}

/// Calculate L2 norm of a sparse vector.
///
/// # Example
///
/// ```rust,ignore
/// let v = SparseVector::new(vec![0, 1], vec![3.0, 4.0]).unwrap();
/// let norm = sparse_l2_norm(&v);
/// assert!((norm - 5.0).abs() < 1e-6); // sqrt(9 + 16) = 5
/// ```
#[inline]
pub fn sparse_l2_norm(v: &SparseVector) -> f32 {
    v.l2_norm()
}

/// Calculate cosine similarity between two sparse vectors.
///
/// Returns value in [-1.0, 1.0]. Returns 0.0 if either vector has zero norm.
///
/// # Example
///
/// ```rust,ignore
/// let a = SparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]).unwrap();
/// let sim = sparse_cosine_similarity(&a, &a);
/// assert!((sim - 1.0).abs() < 1e-6); // Identical vectors
/// ```
#[inline]
pub fn sparse_cosine_similarity(a: &SparseVector, b: &SparseVector) -> f32 {
    a.cosine_similarity(b)
}

/// Calculate Jaccard similarity based on active dimension overlap.
///
/// Jaccard similarity measures the ratio of shared dimensions to total
/// unique dimensions: |A ∩ B| / |A ∪ B|
///
/// # Returns
///
/// - 1.0 if both vectors are empty (considered identical)
/// - 0.0 if one is empty and other is not
/// - Jaccard coefficient in [0.0, 1.0] otherwise
///
/// # Note
///
/// This ignores the actual values and only considers which indices are active.
/// Use for measuring vocabulary overlap between SPLADE vectors.
///
/// # Example
///
/// ```rust,ignore
/// let a = SparseVector::new(vec![1, 2, 3], vec![0.5, 0.5, 0.5]).unwrap();
/// let b = SparseVector::new(vec![2, 3, 4], vec![0.5, 0.5, 0.5]).unwrap();
/// // Intersection: {2, 3} = 2, Union: {1, 2, 3, 4} = 4
/// let jaccard = jaccard_similarity(&a, &b);
/// assert!((jaccard - 0.5).abs() < 1e-6); // 2/4 = 0.5
/// ```
pub fn jaccard_similarity(a: &SparseVector, b: &SparseVector) -> f32 {
    // Handle empty cases
    if a.is_empty() && b.is_empty() {
        return 1.0; // Both empty = identical
    }
    if a.is_empty() || b.is_empty() {
        return 0.0; // One empty = no overlap
    }

    // Use merge-join to count intersection (indices are sorted)
    let mut intersection = 0usize;
    let mut i = 0;
    let mut j = 0;

    while i < a.indices.len() && j < b.indices.len() {
        match a.indices[i].cmp(&b.indices[j]) {
            Ordering::Equal => {
                intersection += 1;
                i += 1;
                j += 1;
            }
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
        }
    }

    // Union = |A| + |B| - |A ∩ B|
    let union = a.indices.len() + b.indices.len() - intersection;

    if union == 0 {
        return 1.0; // Edge case: both empty after loop (shouldn't happen)
    }

    intersection as f32 / union as f32
}

/// BM25 configuration parameters.
///
/// # Default Values
///
/// - k1 = 1.2: Controls term frequency saturation
/// - b = 0.75: Controls document length normalization
#[derive(Debug, Clone)]
pub struct Bm25Config {
    /// k1: Term frequency saturation parameter.
    /// Higher values give more weight to repeated terms.
    pub k1: f32,
    /// b: Length normalization parameter.
    /// b=1.0 means full length normalization, b=0.0 means none.
    pub b: f32,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// Calculate BM25 score for query against document.
///
/// This is for **pairwise scoring**, not index retrieval.
/// For index retrieval with precomputed IDF, use `SpladeInvertedIndex::search()`.
///
/// # Formula
///
/// ```text
/// score = Σ_t q_weight(t) × IDF(t) × (tf × (k1+1)) / (tf + k1×(1-b+b×len/avglen))
/// IDF(t) = ln((N - df + 0.5) / (df + 0.5) + 1)
/// ```
///
/// # Arguments
///
/// - `query`: Query sparse vector
/// - `document`: Document sparse vector
/// - `avg_doc_len`: Average document length (sum of values) in corpus
/// - `doc_count`: Total number of documents in corpus
/// - `term_doc_frequencies`: Map of term index -> number of docs containing term
/// - `config`: BM25 parameters (k1, b)
///
/// # Returns
///
/// BM25 score (non-negative). Returns 0.0 if no term overlap.
///
/// # Example
///
/// ```rust,ignore
/// use std::collections::HashMap;
///
/// let query = SparseVector::new(vec![100, 200], vec![1.0, 0.5]).unwrap();
/// let document = SparseVector::new(vec![100, 300], vec![2.0, 1.0]).unwrap();
///
/// let mut term_df = HashMap::new();
/// term_df.insert(100, 5);  // term 100 appears in 5 docs
/// term_df.insert(200, 2);  // term 200 appears in 2 docs
/// term_df.insert(300, 10); // term 300 appears in 10 docs
///
/// let score = bm25_score(&query, &document, 3.0, 100, &term_df, &Bm25Config::default());
/// ```
pub fn bm25_score(
    query: &SparseVector,
    document: &SparseVector,
    avg_doc_len: f32,
    doc_count: usize,
    term_doc_frequencies: &HashMap<u32, usize>,
    config: &Bm25Config,
) -> f32 {
    if query.is_empty() || document.is_empty() {
        return 0.0;
    }

    // Document length = sum of all term weights
    let doc_len: f32 = document.values.iter().sum();

    let mut score = 0.0f32;
    let n = doc_count as f32;

    // Iterate through query terms
    for (i, &query_idx) in query.indices.iter().enumerate() {
        let query_weight = query.values[i];

        // Find term in document using binary search (indices are sorted)
        if let Ok(pos) = document.indices.binary_search(&query_idx) {
            let tf = document.values[pos];

            // Get document frequency (default to 1 if unknown)
            let df = term_doc_frequencies
                .get(&(query_idx as u32))
                .copied()
                .unwrap_or(1) as f32;

            // IDF: ln((N - df + 0.5) / (df + 0.5) + 1)
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

            // TF with saturation and length normalization
            let tf_norm = if avg_doc_len > f32::EPSILON {
                config.k1 * (1.0 - config.b + config.b * doc_len / avg_doc_len)
            } else {
                config.k1
            };

            let tf_component = (tf * (config.k1 + 1.0)) / (tf + tf_norm);

            // Accumulate: query_weight × IDF × TF_component
            score += query_weight * idf * tf_component;
        }
    }

    score.max(0.0) // Ensure non-negative
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::fingerprint::SparseVector;

    // ========================================================================
    // SPARSE DOT PRODUCT TESTS
    // ========================================================================

    #[test]
    fn test_sparse_dot_identical() {
        let v = SparseVector::new(vec![0, 5, 10], vec![1.0, 2.0, 3.0]).unwrap();
        let dot = sparse_dot_product(&v, &v);
        // 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
        assert!((dot - 14.0).abs() < 1e-6, "Expected 14.0, got {}", dot);
        println!("[PASS] Dot product of identical sparse vectors = {}", dot);
    }

    #[test]
    fn test_sparse_dot_no_overlap() {
        let a = SparseVector::new(vec![0, 1], vec![1.0, 2.0]).unwrap();
        let b = SparseVector::new(vec![2, 3], vec![3.0, 4.0]).unwrap();
        let dot = sparse_dot_product(&a, &b);
        assert_eq!(
            dot, 0.0,
            "Non-overlapping vectors should have 0 dot product"
        );
        println!("[PASS] Dot product of non-overlapping vectors = {}", dot);
    }

    #[test]
    fn test_sparse_dot_partial_overlap() {
        let a = SparseVector::new(vec![1, 3, 5], vec![1.0, 2.0, 3.0]).unwrap();
        let b = SparseVector::new(vec![2, 3, 5], vec![4.0, 5.0, 6.0]).unwrap();
        let dot = sparse_dot_product(&a, &b);
        // Overlap at 3 and 5: 2*5 + 3*6 = 10 + 18 = 28
        assert!((dot - 28.0).abs() < 1e-6, "Expected 28.0, got {}", dot);
        println!("[PASS] Dot product with partial overlap = {}", dot);
    }

    #[test]
    fn test_sparse_dot_empty() {
        let empty = SparseVector::empty();
        let non_empty = SparseVector::new(vec![0, 1], vec![1.0, 2.0]).unwrap();

        assert_eq!(sparse_dot_product(&empty, &empty), 0.0);
        assert_eq!(sparse_dot_product(&empty, &non_empty), 0.0);
        assert_eq!(sparse_dot_product(&non_empty, &empty), 0.0);
        println!("[PASS] Empty vector dot products all return 0.0");
    }

    // ========================================================================
    // SPARSE L2 NORM TESTS
    // ========================================================================

    #[test]
    fn test_sparse_l2_norm() {
        let v = SparseVector::new(vec![0, 1], vec![3.0, 4.0]).unwrap();
        let norm = sparse_l2_norm(&v);
        // sqrt(9 + 16) = sqrt(25) = 5
        assert!((norm - 5.0).abs() < 1e-6, "Expected 5.0, got {}", norm);
        println!("[PASS] L2 norm of [3, 4] = {}", norm);
    }

    #[test]
    fn test_sparse_l2_norm_empty() {
        let empty = SparseVector::empty();
        let norm = sparse_l2_norm(&empty);
        assert_eq!(norm, 0.0);
        println!("[PASS] L2 norm of empty vector = {}", norm);
    }

    // ========================================================================
    // SPARSE COSINE SIMILARITY TESTS
    // ========================================================================

    #[test]
    fn test_sparse_cosine_identical() {
        let v = SparseVector::new(vec![0, 5, 10], vec![1.0, 2.0, 3.0]).unwrap();
        let sim = sparse_cosine_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Identical vectors should have cosine 1.0, got {}",
            sim
        );
        println!("[PASS] Cosine of identical sparse vectors = {}", sim);
    }

    #[test]
    fn test_sparse_cosine_orthogonal() {
        let a = SparseVector::new(vec![0, 1], vec![1.0, 2.0]).unwrap();
        let b = SparseVector::new(vec![2, 3], vec![3.0, 4.0]).unwrap();
        let sim = sparse_cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "Orthogonal vectors should have cosine 0.0, got {}",
            sim
        );
        println!("[PASS] Cosine of orthogonal sparse vectors = {}", sim);
    }

    #[test]
    fn test_sparse_cosine_empty() {
        let empty = SparseVector::empty();
        let non_empty = SparseVector::new(vec![0, 1], vec![1.0, 2.0]).unwrap();

        assert_eq!(sparse_cosine_similarity(&empty, &empty), 0.0);
        assert_eq!(sparse_cosine_similarity(&empty, &non_empty), 0.0);
        println!("[PASS] Cosine with empty vector = 0.0");
    }

    // ========================================================================
    // JACCARD SIMILARITY TESTS
    // ========================================================================

    #[test]
    fn test_jaccard_identical() {
        let v = SparseVector::new(vec![0, 5, 10], vec![1.0, 2.0, 3.0]).unwrap();
        let sim = jaccard_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Identical vectors should have Jaccard 1.0, got {}",
            sim
        );
        println!("[PASS] Jaccard of identical vectors = {}", sim);
    }

    #[test]
    fn test_jaccard_no_overlap() {
        let a = SparseVector::new(vec![0, 1], vec![1.0, 2.0]).unwrap();
        let b = SparseVector::new(vec![2, 3], vec![3.0, 4.0]).unwrap();
        let sim = jaccard_similarity(&a, &b);
        assert_eq!(sim, 0.0, "No overlap should have Jaccard 0.0");
        println!("[PASS] Jaccard of non-overlapping vectors = {}", sim);
    }

    #[test]
    fn test_jaccard_partial_overlap() {
        let a = SparseVector::new(vec![1, 2, 3], vec![0.5, 0.5, 0.5]).unwrap();
        let b = SparseVector::new(vec![2, 3, 4], vec![0.5, 0.5, 0.5]).unwrap();
        let sim = jaccard_similarity(&a, &b);
        // Intersection: {2, 3} = 2, Union: {1, 2, 3, 4} = 4
        assert!((sim - 0.5).abs() < 1e-6, "Expected 0.5, got {}", sim);
        println!("[PASS] Jaccard with 50% overlap = {}", sim);
    }

    #[test]
    fn test_jaccard_subset() {
        let a = SparseVector::new(vec![1, 2], vec![0.5, 0.5]).unwrap();
        let b = SparseVector::new(vec![1, 2, 3, 4], vec![0.5, 0.5, 0.5, 0.5]).unwrap();
        let sim = jaccard_similarity(&a, &b);
        // Intersection: {1, 2} = 2, Union: {1, 2, 3, 4} = 4
        assert!((sim - 0.5).abs() < 1e-6, "Expected 0.5, got {}", sim);
        println!("[PASS] Jaccard with subset = {}", sim);
    }

    #[test]
    fn test_jaccard_empty() {
        let empty = SparseVector::empty();
        let non_empty = SparseVector::new(vec![0, 1], vec![1.0, 2.0]).unwrap();

        // Both empty = identical
        assert_eq!(jaccard_similarity(&empty, &empty), 1.0);
        // One empty = no overlap
        assert_eq!(jaccard_similarity(&empty, &non_empty), 0.0);
        assert_eq!(jaccard_similarity(&non_empty, &empty), 0.0);
        println!("[PASS] Jaccard empty edge cases handled");
    }

    #[test]
    fn test_jaccard_ignores_values() {
        // Jaccard only considers active dimensions, not values
        let a = SparseVector::new(vec![1, 2, 3], vec![0.1, 0.1, 0.1]).unwrap();
        let b = SparseVector::new(vec![1, 2, 3], vec![10.0, 10.0, 10.0]).unwrap();
        let sim = jaccard_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Same indices should have Jaccard 1.0 regardless of values, got {}",
            sim
        );
        println!("[PASS] Jaccard correctly ignores values");
    }

    // ========================================================================
    // BM25 TESTS
    // ========================================================================

    #[test]
    fn test_bm25_basic() {
        let query = SparseVector::new(vec![100], vec![1.0]).unwrap();
        let document = SparseVector::new(vec![100], vec![2.0]).unwrap();

        let mut term_df = HashMap::new();
        term_df.insert(100, 5);

        let score = bm25_score(
            &query,
            &document,
            3.0,
            100,
            &term_df,
            &Bm25Config::default(),
        );

        assert!(score > 0.0, "BM25 score should be positive");
        println!("[PASS] BM25 basic score = {}", score);
    }

    #[test]
    fn test_bm25_no_overlap() {
        let query = SparseVector::new(vec![100], vec![1.0]).unwrap();
        let document = SparseVector::new(vec![200], vec![1.0]).unwrap();

        let term_df = HashMap::new();
        let score = bm25_score(
            &query,
            &document,
            3.0,
            100,
            &term_df,
            &Bm25Config::default(),
        );

        assert_eq!(score, 0.0, "No term overlap should give 0 score");
        println!("[PASS] BM25 no overlap = {}", score);
    }

    #[test]
    fn test_bm25_rare_term_higher_score() {
        let query = SparseVector::new(vec![100, 200], vec![1.0, 1.0]).unwrap();
        let doc_common = SparseVector::new(vec![100], vec![1.0]).unwrap();
        let doc_rare = SparseVector::new(vec![200], vec![1.0]).unwrap();

        let mut term_df = HashMap::new();
        term_df.insert(100, 50); // Common term in 50 docs
        term_df.insert(200, 2); // Rare term in 2 docs

        let config = Bm25Config::default();
        let score_common = bm25_score(&query, &doc_common, 1.0, 100, &term_df, &config);
        let score_rare = bm25_score(&query, &doc_rare, 1.0, 100, &term_df, &config);

        assert!(
            score_rare > score_common,
            "Rare term should have higher BM25 score: rare={}, common={}",
            score_rare,
            score_common
        );
        println!(
            "[PASS] BM25 rare term scores higher: rare={}, common={}",
            score_rare, score_common
        );
    }

    #[test]
    fn test_bm25_empty_vectors() {
        let empty = SparseVector::empty();
        let non_empty = SparseVector::new(vec![100], vec![1.0]).unwrap();
        let term_df = HashMap::new();
        let config = Bm25Config::default();

        assert_eq!(
            bm25_score(&empty, &non_empty, 1.0, 100, &term_df, &config),
            0.0
        );
        assert_eq!(
            bm25_score(&non_empty, &empty, 1.0, 100, &term_df, &config),
            0.0
        );
        println!("[PASS] BM25 handles empty vectors");
    }

    #[test]
    fn test_bm25_config() {
        let config = Bm25Config::default();
        assert!((config.k1 - 1.2).abs() < 1e-6);
        assert!((config.b - 0.75).abs() < 1e-6);
        println!(
            "[PASS] BM25 default config: k1={}, b={}",
            config.k1, config.b
        );
    }

    #[test]
    fn test_bm25_length_normalization() {
        let query = SparseVector::new(vec![100], vec![1.0]).unwrap();
        let short_doc = SparseVector::new(vec![100], vec![1.0]).unwrap();
        let long_doc = SparseVector::new(vec![100, 200, 300], vec![1.0, 1.0, 1.0]).unwrap();

        let mut term_df = HashMap::new();
        term_df.insert(100, 10);

        let config = Bm25Config::default();
        let score_short = bm25_score(&query, &short_doc, 2.0, 100, &term_df, &config);
        let score_long = bm25_score(&query, &long_doc, 2.0, 100, &term_df, &config);

        // Short doc (1.0 length) vs avg (2.0) should score higher than long doc (3.0 length)
        assert!(
            score_short > score_long,
            "Shorter doc should score higher: short={}, long={}",
            score_short,
            score_long
        );
        println!(
            "[PASS] BM25 length normalization: short={}, long={}",
            score_short, score_long
        );
    }

    // ========================================================================
    // EDGE CASE TESTS
    // ========================================================================

    #[test]
    fn test_sparse_single_element() {
        let v = SparseVector::new(vec![100], vec![5.0]).unwrap();

        // Dot product with self = 25
        assert!((sparse_dot_product(&v, &v) - 25.0).abs() < 1e-6);

        // L2 norm = 5
        assert!((sparse_l2_norm(&v) - 5.0).abs() < 1e-6);

        // Cosine with self = 1.0
        assert!((sparse_cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);

        // Jaccard with self = 1.0
        assert!((jaccard_similarity(&v, &v) - 1.0).abs() < 1e-6);

        println!("[PASS] Single element sparse vector edge case");
    }

    #[test]
    fn test_sparse_max_index() {
        // Test with maximum valid index (30521)
        let v = SparseVector::new(vec![0, 30521], vec![1.0, 1.0]).unwrap();
        let dot = sparse_dot_product(&v, &v);
        assert!((dot - 2.0).abs() < 1e-6);
        println!("[PASS] Maximum index (30521) handled correctly");
    }

    #[test]
    fn test_sparse_typical_sparsity() {
        // Simulate typical SPLADE sparsity (~5% = 1500 active indices)
        let indices: Vec<u16> = (0..1500).map(|i| i * 20).collect();
        let values: Vec<f32> = vec![0.1; 1500];
        let v = SparseVector::new(indices, values).unwrap();

        let norm = sparse_l2_norm(&v);
        assert!(norm > 0.0);

        let sim = sparse_cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);

        println!("[PASS] Typical sparsity (1500 nnz) handled: norm={}", norm);
    }

    // ========================================================================
    // SYNTHETIC DATA VERIFICATION TESTS
    // ========================================================================

    #[test]
    fn test_synthetic_known_dot_product() {
        // Input: a = {1: 2.0, 3: 3.0}, b = {3: 4.0, 5: 5.0}
        // Expected: Only index 3 overlaps: 3.0 * 4.0 = 12.0
        let a = SparseVector::new(vec![1, 3], vec![2.0, 3.0]).unwrap();
        let b = SparseVector::new(vec![3, 5], vec![4.0, 5.0]).unwrap();
        let dot = sparse_dot_product(&a, &b);

        println!(
            "STATE BEFORE: a.indices={:?}, a.values={:?}",
            a.indices, a.values
        );
        println!(
            "STATE BEFORE: b.indices={:?}, b.values={:?}",
            b.indices, b.values
        );
        println!("STATE AFTER: dot={} (expected 12.0)", dot);

        assert!((dot - 12.0).abs() < 1e-6, "Expected 12.0, got {}", dot);
        println!("[PASS] Synthetic dot product verified: 12.0");
    }

    #[test]
    fn test_synthetic_known_jaccard() {
        // Input: a = {1, 2, 3}, b = {2, 3, 4, 5}
        // Intersection: {2, 3} = 2 elements
        // Union: {1, 2, 3, 4, 5} = 5 elements
        // Expected Jaccard: 2/5 = 0.4
        let a = SparseVector::new(vec![1, 2, 3], vec![1.0, 1.0, 1.0]).unwrap();
        let b = SparseVector::new(vec![2, 3, 4, 5], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let jaccard = jaccard_similarity(&a, &b);

        println!("STATE BEFORE: a.indices={:?}", a.indices);
        println!("STATE BEFORE: b.indices={:?}", b.indices);
        println!("STATE AFTER: jaccard={} (expected 0.4)", jaccard);

        assert!(
            (jaccard - 0.4).abs() < 1e-6,
            "Expected 0.4, got {}",
            jaccard
        );
        println!("[PASS] Synthetic Jaccard verified: 0.4");
    }

    #[test]
    fn test_synthetic_bm25_idf_verification() {
        // Verify IDF component: rare terms score higher
        // Common term in 90/100 docs vs rare term in 5/100 docs
        // IDF(common) = ln((100-90+0.5)/(90+0.5)+1) = ln(10.5/90.5+1) ≈ ln(1.116) ≈ 0.11
        // IDF(rare) = ln((100-5+0.5)/(5+0.5)+1) = ln(95.5/5.5+1) ≈ ln(18.36) ≈ 2.91

        let query = SparseVector::new(vec![100, 200], vec![1.0, 1.0]).unwrap();
        let doc_common = SparseVector::new(vec![100], vec![1.0]).unwrap();
        let doc_rare = SparseVector::new(vec![200], vec![1.0]).unwrap();

        let mut term_df = HashMap::new();
        term_df.insert(100, 90); // Common term in 90 docs
        term_df.insert(200, 5); // Rare term in 5 docs

        let config = Bm25Config::default();
        let score_common = bm25_score(&query, &doc_common, 1.0, 100, &term_df, &config);
        let score_rare = bm25_score(&query, &doc_rare, 1.0, 100, &term_df, &config);

        println!("IDF verification:");
        println!("  Common term (90/100 docs): score = {}", score_common);
        println!("  Rare term (5/100 docs): score = {}", score_rare);
        println!("  Ratio rare/common = {}", score_rare / score_common);

        // Rare term should score significantly higher
        assert!(
            score_rare > score_common * 5.0,
            "Rare term (5 docs) should score much higher than common term (90 docs)"
        );
        println!("[PASS] BM25 IDF verified: rare terms score higher");
    }
}
