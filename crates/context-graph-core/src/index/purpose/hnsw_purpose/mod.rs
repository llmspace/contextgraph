//! HNSW-backed purpose pattern index with metadata enrichment.
//!
//! # CRITICAL: NO FALLBACKS
//!
//! All index operations are fail-fast. Missing entries cause immediate errors.
//! Invalid queries rejected at construction time.
//!
//! # Overview
//!
//! This module provides the main `HnswPurposeIndex` for Stage 4 retrieval:
//! - HNSW-backed ANN search on 13D purpose vectors
//! - Metadata storage for goal filtering
//! - Secondary indexes for efficient filtered queries
//!
//! # Architecture
//!
//! ```text
//! HnswPurposeIndex
//! ├── inner: RealHnswIndex (13D HNSW for ANN search - O(log n))
//! ├── metadata: HashMap<Uuid, PurposeMetadata>
//! ├── vectors: HashMap<Uuid, PurposeVector>
//! └── goal_index: HashMap<GoalId, HashSet<Uuid>>
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use context_graph_core::index::purpose::{
//!     HnswPurposeIndex, PurposeIndexOps, PurposeIndexEntry,
//!     PurposeQuery, PurposeQueryTarget,
//! };
//!
//! // Create index
//! let mut index = HnswPurposeIndex::new(HnswConfig::purpose_vector())?;
//!
//! // Insert entries
//! index.insert(entry)?;
//!
//! // Search
//! let results = index.search(&query)?;
//! ```

mod implementation;
mod ops;

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use crate::index::hnsw_impl::RealHnswIndex;
use crate::types::fingerprint::PurposeVector;

use super::entry::PurposeMetadata;
use super::error::PurposeIndexResult;
use super::query::PurposeSearchResult;

// Re-export implementation methods (the struct and trait are defined in this file)

/// HNSW-backed purpose pattern index with metadata enrichment.
///
/// Combines a REAL HNSW index (O(log n)) for fast ANN search on 13D purpose
/// vectors with secondary indexes for efficient filtering by goal.
///
/// # Structure
///
/// - `inner`: RealHnswIndex for approximate nearest neighbor search (O(log n))
/// - `metadata`: Purpose metadata indexed by memory ID
/// - `vectors`: Purpose vectors indexed by memory ID for reranking
/// - `goal_index`: Inverted index from goal ID to memory IDs
///
/// # Fail-Fast Semantics
///
/// - Insert validates dimension matches PURPOSE_VECTOR_DIM (13)
/// - Remove fails if memory not found (no silent no-ops)
/// - Search validates query before execution
/// - Get fails if memory not found
#[derive(Debug)]
pub struct HnswPurposeIndex {
    /// Underlying REAL HNSW index for ANN search (O(log n), not O(n) like old SimpleHnswIndex).
    pub(crate) inner: RealHnswIndex,
    /// Metadata storage indexed by memory ID.
    pub(crate) metadata: HashMap<Uuid, PurposeMetadata>,
    /// Purpose vectors for reranking.
    pub(crate) vectors: HashMap<Uuid, PurposeVector>,
    /// Index by primary goal for filtered queries.
    pub(crate) goal_index: HashMap<String, HashSet<Uuid>>,
}

/// Trait for purpose index operations.
///
/// Defines the core operations for managing and querying the purpose index.
///
/// # Fail-Fast Semantics
///
/// All operations validate inputs and fail immediately on invalid data.
/// There are no silent failures or fallback behaviors.
pub trait PurposeIndexOps {
    /// Insert a purpose entry into the index.
    ///
    /// # Arguments
    ///
    /// * `entry` - The purpose index entry to insert
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch`: If vector dimension != 13
    /// - `HnswError`: If HNSW insertion fails
    ///
    /// # Behavior
    ///
    /// If an entry with the same memory_id already exists, it is replaced.
    /// Secondary indexes are updated accordingly.
    fn insert(&mut self, entry: super::entry::PurposeIndexEntry) -> PurposeIndexResult<()>;

    /// Remove a memory from the index.
    ///
    /// # Arguments
    ///
    /// * `memory_id` - UUID of the memory to remove
    ///
    /// # Errors
    ///
    /// - `NotFound`: If memory_id does not exist in the index
    ///
    /// # Fail-Fast
    ///
    /// Does NOT silently succeed if memory is not found.
    fn remove(&mut self, memory_id: Uuid) -> PurposeIndexResult<()>;

    /// Search the index with query parameters.
    ///
    /// # Arguments
    ///
    /// * `query` - The purpose query specifying target and filters
    ///
    /// # Returns
    ///
    /// Vector of search results sorted by descending similarity.
    ///
    /// # Errors
    ///
    /// - `NotFound`: For FromMemory target when memory doesn't exist
    /// - `ClusteringError`: For Pattern target when clustering fails
    /// - `InvalidQuery`: If query parameters are invalid
    fn search(
        &self,
        query: &super::query::PurposeQuery,
    ) -> PurposeIndexResult<Vec<PurposeSearchResult>>;

    /// Get entry by memory ID.
    ///
    /// # Arguments
    ///
    /// * `memory_id` - UUID of the memory to retrieve
    ///
    /// # Errors
    ///
    /// - `NotFound`: If memory_id does not exist in the index
    fn get(&self, memory_id: Uuid) -> PurposeIndexResult<super::entry::PurposeIndexEntry>;

    /// Check if memory exists in index.
    ///
    /// # Arguments
    ///
    /// * `memory_id` - UUID to check
    ///
    /// # Returns
    ///
    /// `true` if the memory exists in the index.
    fn contains(&self, memory_id: Uuid) -> bool;

    /// Get total number of entries.
    fn len(&self) -> usize;

    /// Check if empty.
    fn is_empty(&self) -> bool;

    /// Clear all entries.
    fn clear(&mut self);
}
