//! In-memory stub implementation of TeleologicalMemoryStore.
//!
//! # WARNING: TEST ONLY - DO NOT USE IN PRODUCTION
//!
//! This module provides `InMemoryTeleologicalStore`, a thread-safe in-memory
//! implementation of the `TeleologicalMemoryStore` trait **for testing only**.
//!
//! ## Critical Limitations
//!
//! - **O(n) search complexity**: All search operations perform full table scans.
//! - **No persistence**: All data is lost when the store is dropped.
//! - **No HNSW indexing**: Unlike production stores, this stub does not use
//!   approximate nearest neighbor search.
//!
//! ## When to Use
//!
//! - Unit tests that need a `TeleologicalMemoryStore` implementation
//! - Integration tests with small datasets (< 1000 fingerprints)
//! - Development/prototyping where persistence is not required
//!
//! ## When NOT to Use
//!
//! - Production systems (use `RocksDbTeleologicalStore` instead)
//! - Benchmarks (O(n) will skew results)
//! - Any scenario requiring data persistence
//! - Datasets larger than ~1000 fingerprints
//!
//! # Design
//!
//! - Uses `DashMap` for concurrent access without external locking
//! - No persistence - data is lost on drop
//! - Full trait implementation with real algorithms (not mocks)
//!
//! # Performance
//!
//! - **O(n) search operations** - full table scan, no indexing
//! - O(1) CRUD operations via HashMap
//! - ~46KB per fingerprint in memory

mod content;
mod search;
mod similarity;
mod trait_impl;
#[cfg(test)]
mod tests;

use std::sync::atomic::AtomicUsize;

use dashmap::DashMap;
use tracing::info;
use uuid::Uuid;

use crate::gwt::ego_node::SelfEgoNode;
use crate::traits::TeleologicalStorageBackend;
use crate::types::fingerprint::TeleologicalFingerprint;

// Re-export similarity functions for internal use
pub(crate) use similarity::{
    compute_late_interaction_score, compute_semantic_scores, cosine_similarity,
    get_dominant_quadrant,
};

/// In-memory implementation of TeleologicalMemoryStore.
///
/// # WARNING: TEST ONLY - DO NOT USE IN PRODUCTION
///
/// For production use, use `RocksDbTeleologicalStore` from `context-graph-storage`.
#[derive(Debug)]
pub struct InMemoryTeleologicalStore {
    /// Main storage: UUID -> TeleologicalFingerprint
    pub(crate) data: DashMap<Uuid, TeleologicalFingerprint>,
    /// Soft-deleted IDs (still in data but marked deleted)
    pub(crate) deleted: DashMap<Uuid, ()>,
    /// Content storage: UUID -> original content text
    pub(crate) content: DashMap<Uuid, String>,
    /// Singleton SELF_EGO_NODE storage (TASK-GWT-P1-001).
    pub(crate) ego_node: std::sync::RwLock<Option<SelfEgoNode>>,
    /// Running size estimate in bytes
    pub(crate) size_bytes: AtomicUsize,
}

impl InMemoryTeleologicalStore {
    /// Create a new empty in-memory store.
    pub fn new() -> Self {
        info!("Creating new InMemoryTeleologicalStore (TEST ONLY)");
        Self {
            data: DashMap::new(),
            deleted: DashMap::new(),
            content: DashMap::new(),
            ego_node: std::sync::RwLock::new(None),
            size_bytes: AtomicUsize::new(0),
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        info!("Creating InMemoryTeleologicalStore with capacity {} (TEST ONLY)", capacity);
        Self {
            data: DashMap::with_capacity(capacity),
            deleted: DashMap::new(),
            content: DashMap::with_capacity(capacity),
            ego_node: std::sync::RwLock::new(None),
            size_bytes: AtomicUsize::new(0),
        }
    }

    /// Estimate memory size of a fingerprint.
    pub(crate) fn estimate_fingerprint_size(fp: &TeleologicalFingerprint) -> usize {
        let base = std::mem::size_of::<TeleologicalFingerprint>();
        let semantic = fp.semantic.storage_size();
        let evolution = fp.purpose_evolution.len() * 200;
        base + semantic + evolution
    }

    /// Returns the backend type.
    pub fn backend_type(&self) -> TeleologicalStorageBackend {
        TeleologicalStorageBackend::InMemory
    }
}

impl Default for InMemoryTeleologicalStore {
    fn default() -> Self {
        Self::new()
    }
}
