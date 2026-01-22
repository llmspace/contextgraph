//! Storage traits for the pipeline.
//!
//! This module defines the storage interfaces for SPLADE index (Stage 1)
//! and token embeddings (Stage 5 MaxSim).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicUsize;
use std::sync::RwLock;
use uuid::Uuid;

// ============================================================================
// TOKEN STORAGE TRAIT (for Stage 5 MaxSim)
// ============================================================================

/// Storage interface for E12 ColBERT token embeddings.
///
/// Stage 5 requires token-level embeddings for MaxSim scoring.
/// This trait abstracts the storage backend.
pub trait TokenStorage: Send + Sync {
    /// Retrieve token embeddings for a memory ID.
    ///
    /// Returns Vec of 128D token embeddings.
    fn get_tokens(&self, id: Uuid) -> Option<Vec<Vec<f32>>>;
}

/// In-memory token storage for testing.
#[derive(Debug, Default)]
pub struct InMemoryTokenStorage {
    tokens: RwLock<HashMap<Uuid, Vec<Vec<f32>>>>,
}

impl InMemoryTokenStorage {
    /// Create new empty storage.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert tokens for an ID.
    pub fn insert(&self, id: Uuid, tokens: Vec<Vec<f32>>) {
        self.tokens.write().unwrap().insert(id, tokens);
    }

    /// Get number of stored IDs.
    pub fn len(&self) -> usize {
        self.tokens.read().unwrap().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.read().unwrap().is_empty()
    }
}

impl TokenStorage for InMemoryTokenStorage {
    fn get_tokens(&self, id: Uuid) -> Option<Vec<Vec<f32>>> {
        self.tokens.read().unwrap().get(&id).cloned()
    }
}

// ============================================================================
// SPLADE INDEX TRAIT (for Stage 1)
// ============================================================================

/// Storage interface for SPLADE/E13 inverted index.
///
/// Stage 1 requires inverted index search, NOT HNSW.
pub trait SpladeIndex: Send + Sync {
    /// Search with BM25+SPLADE scoring.
    ///
    /// # Arguments
    /// * `query` - Sparse query vector as (term_id, weight) pairs
    /// * `k` - Number of results to return
    ///
    /// # Returns
    /// Vec of (id, score) pairs sorted by descending score.
    fn search(&self, query: &[(usize, f32)], k: usize) -> Vec<(Uuid, f32)>;

    /// Get the number of documents in the index.
    fn len(&self) -> usize;

    /// Check if index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// In-memory SPLADE index for testing.
#[derive(Debug, Default)]
pub struct InMemorySpladeIndex {
    /// Posting lists: term_id -> [(doc_id, weight), ...]
    posting_lists: RwLock<HashMap<usize, Vec<(Uuid, f32)>>>,
    /// Document L2 norms
    doc_norms: RwLock<HashMap<Uuid, f32>>,
    /// Document frequency per term
    doc_freq: RwLock<HashMap<usize, usize>>,
    /// Total documents
    num_docs: AtomicUsize,
}

impl InMemorySpladeIndex {
    /// Create new empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a sparse vector to the index.
    pub fn add(&self, id: Uuid, sparse: &[(usize, f32)]) {
        // Compute norm
        let norm: f32 = sparse.iter().map(|(_, w)| w * w).sum::<f32>().sqrt();
        if norm < f32::EPSILON {
            return;
        }

        self.doc_norms.write().unwrap().insert(id, norm);

        let mut added_terms = HashSet::new();
        let mut postings = self.posting_lists.write().unwrap();
        let mut doc_freq = self.doc_freq.write().unwrap();

        for &(term_id, weight) in sparse {
            if weight.abs() < f32::EPSILON {
                continue;
            }

            postings.entry(term_id).or_default().push((id, weight));

            if added_terms.insert(term_id) {
                *doc_freq.entry(term_id).or_insert(0) += 1;
            }
        }

        self.num_docs
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl SpladeIndex for InMemorySpladeIndex {
    fn search(&self, query: &[(usize, f32)], k: usize) -> Vec<(Uuid, f32)> {
        let n = self.num_docs.load(std::sync::atomic::Ordering::SeqCst);
        if n == 0 {
            return Vec::new();
        }

        let postings = self.posting_lists.read().unwrap();
        let doc_norms = self.doc_norms.read().unwrap();
        let doc_freq = self.doc_freq.read().unwrap();

        let mut scores: HashMap<Uuid, f32> = HashMap::new();
        let n_f = n as f32;

        for &(term_id, query_weight) in query {
            if let Some(term_postings) = postings.get(&term_id) {
                let df = doc_freq.get(&term_id).copied().unwrap_or(1) as f32;
                let idf = ((n_f - df + 0.5) / (df + 0.5) + 1.0).ln();

                for &(doc_id, doc_weight) in term_postings {
                    let norm = doc_norms.get(&doc_id).copied().unwrap_or(1.0);
                    let tf = doc_weight / norm.max(f32::EPSILON);
                    *scores.entry(doc_id).or_insert(0.0) += query_weight * tf * idf;
                }
            }
        }

        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    fn len(&self) -> usize {
        self.num_docs.load(std::sync::atomic::Ordering::SeqCst)
    }
}

// ============================================================================
// E6 SPARSE INDEX TRAIT (for dual Stage 1 recall per e6upgrade.md)
// ============================================================================

/// Storage interface for E6 sparse (V_selectivity) inverted index.
///
/// E6 provides exact keyword matching to complement E13 SPLADE's learned expansion.
/// Used in dual Stage 1 recall: E6 catches exact technical terms, E13 catches
/// semantic variations.
pub trait E6SparseIndex: Send + Sync {
    /// Search with term overlap scoring.
    ///
    /// Unlike SPLADE's BM25-style scoring, E6 uses simpler term overlap for
    /// exact keyword matching. This is intentional - E6's strength is precision,
    /// not recall.
    ///
    /// # Arguments
    /// * `query` - Sparse query vector as (term_id, weight) pairs
    /// * `k` - Number of results to return
    ///
    /// # Returns
    /// Vec of (id, score) pairs sorted by descending score.
    fn search(&self, query: &[(usize, f32)], k: usize) -> Vec<(Uuid, f32)>;

    /// Get sparse vector for a document (for tie-breaking).
    fn get_sparse(&self, id: Uuid) -> Option<Vec<(usize, f32)>>;

    /// Get the number of documents in the index.
    fn len(&self) -> usize;

    /// Check if index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// In-memory E6 sparse index for testing.
#[derive(Debug, Default)]
pub struct InMemoryE6SparseIndex {
    /// Posting lists: term_id -> [(doc_id, weight), ...]
    posting_lists: RwLock<HashMap<usize, Vec<(Uuid, f32)>>>,
    /// Document sparse vectors for tie-breaking
    doc_vectors: RwLock<HashMap<Uuid, Vec<(usize, f32)>>>,
    /// Total documents
    num_docs: AtomicUsize,
}

impl InMemoryE6SparseIndex {
    /// Create new empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a sparse vector to the index.
    pub fn add(&self, id: Uuid, sparse: &[(usize, f32)]) {
        // Store sparse vector for later retrieval
        self.doc_vectors
            .write()
            .unwrap()
            .insert(id, sparse.to_vec());

        // Update posting lists
        let mut postings = self.posting_lists.write().unwrap();
        for &(term_id, weight) in sparse {
            if weight.abs() < f32::EPSILON {
                continue;
            }
            postings.entry(term_id).or_default().push((id, weight));
        }

        self.num_docs
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl E6SparseIndex for InMemoryE6SparseIndex {
    fn search(&self, query: &[(usize, f32)], k: usize) -> Vec<(Uuid, f32)> {
        let n = self.num_docs.load(std::sync::atomic::Ordering::SeqCst);
        if n == 0 {
            return Vec::new();
        }

        let postings = self.posting_lists.read().unwrap();

        // E6 uses simple term overlap scoring for exact keyword matching
        let mut term_counts: HashMap<Uuid, usize> = HashMap::new();
        let mut weighted_scores: HashMap<Uuid, f32> = HashMap::new();

        for &(term_id, query_weight) in query {
            if let Some(term_postings) = postings.get(&term_id) {
                for &(doc_id, doc_weight) in term_postings {
                    *term_counts.entry(doc_id).or_insert(0) += 1;
                    // Weighted score: query_weight * doc_weight
                    *weighted_scores.entry(doc_id).or_insert(0.0) += query_weight * doc_weight;
                }
            }
        }

        // Score = term_overlap_ratio * weighted_score
        // This prioritizes documents with more matching terms
        let query_term_count = query.len() as f32;
        let mut results: Vec<_> = term_counts
            .into_iter()
            .map(|(id, count)| {
                let overlap_ratio = count as f32 / query_term_count.max(1.0);
                let weighted = weighted_scores.get(&id).copied().unwrap_or(0.0);
                let score = overlap_ratio * (1.0 + weighted.ln().max(0.0));
                (id, score)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    fn get_sparse(&self, id: Uuid) -> Option<Vec<(usize, f32)>> {
        self.doc_vectors.read().unwrap().get(&id).cloned()
    }

    fn len(&self) -> usize {
        self.num_docs.load(std::sync::atomic::Ordering::SeqCst)
    }
}

// ============================================================================
// QUERY-AWARE E6 WEIGHT BOOST (per e6upgrade.md Enhancement 4)
// ============================================================================

/// Detect query type and compute E6 weight boost factor.
///
/// Technical queries benefit from exact keyword matching (higher E6 weight),
/// while general language queries should rely more on semantic (E1).
///
/// # Returns
/// Boost factor in [0.5, 2.0] range
pub fn compute_e6_boost(query: &str) -> f32 {
    let mut boost = 1.0f32;

    // Technical indicators boost E6
    if contains_api_path(query) {
        boost += 0.5;
    }
    if contains_version_string(query) {
        boost += 0.3;
    }
    if contains_acronym(query) {
        boost += 0.3;
    }
    if contains_proper_noun(query) {
        boost += 0.2;
    }

    // General language reduces E6 (let E1 handle)
    if high_common_word_ratio(query) {
        boost -= 0.3;
    }

    boost.clamp(0.5, 2.0)
}

/// Check if query contains API path patterns (e.g., tokio::spawn, std::fs).
fn contains_api_path(query: &str) -> bool {
    query.contains("::")
        || query.contains("->")
        || query.contains(".")
            && (query.contains("fn ")
                || query.contains("impl ")
                || query.contains("struct ")
                || query.contains("trait "))
}

/// Check if query contains version strings (e.g., v1.0, 2.0, TLS 1.3).
fn contains_version_string(query: &str) -> bool {
    // Simple patterns: vN, N.N, N.N.N
    let words: Vec<&str> = query.split_whitespace().collect();
    for word in words {
        if word.starts_with('v') && word.len() >= 2 && word.chars().nth(1).map_or(false, |c| c.is_ascii_digit()) {
            return true;
        }
        // Check for N.N pattern
        if word.contains('.') {
            let parts: Vec<&str> = word.split('.').collect();
            if parts.len() >= 2 && parts[0].chars().all(|c| c.is_ascii_digit())
                && parts[1].chars().take_while(|c| c.is_ascii_digit()).count() > 0
            {
                return true;
            }
        }
    }
    false
}

/// Check if query contains acronyms (all-caps words 2+ chars).
fn contains_acronym(query: &str) -> bool {
    query.split_whitespace().any(|word| {
        word.len() >= 2
            && word.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
    })
}

/// Check if query contains proper nouns (capitalized words that aren't sentence starters).
fn contains_proper_noun(query: &str) -> bool {
    let words: Vec<&str> = query.split_whitespace().collect();
    for (i, word) in words.iter().enumerate() {
        if i > 0 {
            // Not first word
            if let Some(c) = word.chars().next() {
                if c.is_ascii_uppercase() && word.len() > 1 {
                    // Check if rest is lowercase (proper noun) or all caps (acronym)
                    let rest_lower = word.chars().skip(1).all(|c| c.is_lowercase());
                    if rest_lower {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Check if query has high ratio of common/stop words.
fn high_common_word_ratio(query: &str) -> bool {
    const COMMON_WORDS: &[&str] = &[
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "or", "and", "but", "if",
        "then", "else", "when", "where", "why", "how", "what", "which", "who", "this", "that",
        "these", "those", "it", "its", "i", "you", "he", "she", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "her", "our", "their",
    ];

    let words: Vec<&str> = query.split_whitespace().collect();
    if words.is_empty() {
        return false;
    }

    let common_count = words
        .iter()
        .filter(|w| COMMON_WORDS.contains(&w.to_lowercase().as_str()))
        .count();

    let ratio = common_count as f32 / words.len() as f32;
    ratio > 0.5 // More than half are common words
}

// ============================================================================
// E6 TIE-BREAKER (per e6upgrade.md Enhancement 6)
// ============================================================================

/// Apply E6 tie-breaker to candidates with close scores.
///
/// When candidates have similar E1/semantic scores, use E6 term overlap
/// to break ties. This rewards exact keyword matches without overriding
/// semantic ranking.
///
/// # Arguments
/// * `candidates` - Mutable slice of (id, score) pairs
/// * `query_sparse` - Query E6 sparse vector as (term_id, weight) pairs
/// * `e6_index` - E6 sparse index for retrieving document vectors
/// * `tie_threshold` - Score difference threshold for tie-breaking (default 0.05)
/// * `max_boost` - Maximum tie-breaker boost (default 0.05)
pub fn apply_e6_tiebreaker(
    candidates: &mut [(Uuid, f32)],
    query_sparse: &[(usize, f32)],
    e6_index: &dyn E6SparseIndex,
    tie_threshold: f32,
    max_boost: f32,
) {
    if candidates.len() < 2 || query_sparse.is_empty() {
        return;
    }

    let query_terms: HashSet<usize> = query_sparse.iter().map(|(t, _)| *t).collect();
    let query_term_count = query_terms.len() as f32;

    // Compute E6 term overlap scores
    let mut overlap_scores: Vec<f32> = Vec::with_capacity(candidates.len());
    for (id, _) in candidates.iter() {
        let overlap = if let Some(doc_sparse) = e6_index.get_sparse(*id) {
            let doc_terms: HashSet<usize> = doc_sparse.iter().map(|(t, _)| *t).collect();
            let shared = query_terms.intersection(&doc_terms).count() as f32;
            shared / query_term_count.max(1.0)
        } else {
            0.0
        };
        overlap_scores.push(overlap);
    }

    // Apply tie-breaker: for candidates within tie_threshold of each other,
    // add small boost based on E6 overlap
    for i in 1..candidates.len() {
        let score_diff = (candidates[i - 1].1 - candidates[i].1).abs();
        if score_diff < tie_threshold {
            // Within tie range - apply E6 boost
            candidates[i].1 += overlap_scores[i] * max_boost;
        }
    }

    // Re-sort after applying boosts
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
}
