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
