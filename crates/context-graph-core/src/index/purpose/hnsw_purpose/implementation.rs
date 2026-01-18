//! Implementation of HnswPurposeIndex methods.

use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use crate::index::config::{HnswConfig, PURPOSE_VECTOR_DIM};
use crate::index::hnsw_impl::RealHnswIndex;
use crate::types::fingerprint::PurposeVector;

use super::super::clustering::{KMeansConfig, KMeansPurposeClustering, StandardKMeans};
use super::super::entry::{GoalId, PurposeIndexEntry, PurposeMetadata};
use super::super::error::{PurposeIndexError, PurposeIndexResult};
use super::super::query::{PurposeQuery, PurposeSearchResult};
use super::HnswPurposeIndex;

impl HnswPurposeIndex {
    /// Create a new purpose index with given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - HNSW configuration (should have dimension = 13)
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if config.dimension != PURPOSE_VECTOR_DIM.
    /// Returns `HnswError` if HNSW index construction fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let index = HnswPurposeIndex::new(HnswConfig::purpose_vector())?;
    /// ```
    pub fn new(config: HnswConfig) -> PurposeIndexResult<Self> {
        // Fail-fast: validate dimension
        if config.dimension != PURPOSE_VECTOR_DIM {
            return Err(PurposeIndexError::dimension_mismatch(
                PURPOSE_VECTOR_DIM,
                config.dimension,
            ));
        }

        // Create real HNSW index - may fail, propagate error via #[from] conversion
        let inner = RealHnswIndex::new(config)?;

        Ok(Self {
            inner,
            metadata: HashMap::new(),
            vectors: HashMap::new(),
            goal_index: HashMap::new(),
        })
    }

    /// Create a new purpose index with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `config` - HNSW configuration
    /// * `capacity` - Expected number of entries for pre-allocation
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if config.dimension != PURPOSE_VECTOR_DIM.
    /// Returns `HnswError` if HNSW index construction fails.
    pub fn with_capacity(config: HnswConfig, capacity: usize) -> PurposeIndexResult<Self> {
        // Fail-fast: validate dimension
        if config.dimension != PURPOSE_VECTOR_DIM {
            return Err(PurposeIndexError::dimension_mismatch(
                PURPOSE_VECTOR_DIM,
                config.dimension,
            ));
        }

        // Create real HNSW index - may fail, propagate error via #[from] conversion
        let inner = RealHnswIndex::new(config)?;

        Ok(Self {
            inner,
            metadata: HashMap::with_capacity(capacity),
            vectors: HashMap::with_capacity(capacity),
            goal_index: HashMap::new(),
        })
    }

    /// Get memories with a specific goal.
    ///
    /// Returns all memory IDs aligned with the given goal.
    #[inline]
    pub fn get_by_goal(&self, goal: &GoalId) -> Option<&HashSet<Uuid>> {
        self.goal_index.get(goal.as_str())
    }

    /// Get the number of distinct goals in the index.
    #[inline]
    pub fn goal_count(&self) -> usize {
        self.goal_index.len()
    }

    /// Get all goals present in the index.
    pub fn goals(&self) -> Vec<GoalId> {
        self.goal_index.keys().map(GoalId::new).collect()
    }

    /// Update secondary indexes when inserting an entry.
    pub(crate) fn update_secondary_indexes(&mut self, memory_id: Uuid, metadata: &PurposeMetadata) {
        // Update goal index
        self.goal_index
            .entry(metadata.primary_goal.as_str().to_string())
            .or_default()
            .insert(memory_id);
    }

    /// Remove from secondary indexes when removing an entry.
    pub(crate) fn remove_from_secondary_indexes(
        &mut self,
        memory_id: Uuid,
        metadata: &PurposeMetadata,
    ) {
        // Remove from goal index
        let goal_key = metadata.primary_goal.as_str();
        if let Some(set) = self.goal_index.get_mut(goal_key) {
            set.remove(&memory_id);
            if set.is_empty() {
                self.goal_index.remove(goal_key);
            }
        }
    }

    /// Perform vector search with optional filtering.
    pub(crate) fn search_vector(
        &self,
        query_vector: &PurposeVector,
        query: &PurposeQuery,
    ) -> PurposeIndexResult<Vec<PurposeSearchResult>> {
        // If we have filters, compute candidate set first
        let candidates = self.compute_candidate_set(query);

        // Determine how many results to request from HNSW
        // Request more if filtering to ensure we get enough after filtering
        let k = if query.has_filters() {
            // Request extra to account for filtering
            query.limit * 3 + 10
        } else {
            query.limit
        };

        // Search HNSW index
        let alignments = &query_vector.alignments;
        let hnsw_results = self.inner.search(alignments.as_slice(), k)?;

        // Build results with filtering and reranking
        let mut results: Vec<PurposeSearchResult> = hnsw_results
            .into_iter()
            .filter_map(|(memory_id, similarity)| {
                // Apply candidate filter if we have one
                if let Some(ref cands) = candidates {
                    if !cands.contains(&memory_id) {
                        return None;
                    }
                }

                // Apply min_similarity filter
                if similarity < query.min_similarity {
                    return None;
                }

                // Retrieve metadata and vector
                let metadata = self.metadata.get(&memory_id)?;
                let vector = self.vectors.get(&memory_id)?;

                // Apply goal filter
                if let Some(ref goal_filter) = query.goal_filter {
                    if metadata.primary_goal.as_str() != goal_filter.as_str() {
                        return None;
                    }
                }

                Some(PurposeSearchResult::new(
                    memory_id,
                    similarity,
                    vector.clone(),
                    metadata.clone(),
                ))
            })
            .collect();

        // Sort by similarity descending (should already be sorted, but ensure)
        results.sort_by(|a, b| {
            b.purpose_similarity
                .partial_cmp(&a.purpose_similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        results.truncate(query.limit);

        Ok(results)
    }

    /// Compute candidate set based on filters.
    pub(crate) fn compute_candidate_set(&self, query: &PurposeQuery) -> Option<HashSet<Uuid>> {
        if !query.has_filters() {
            return None;
        }

        // Apply goal filter
        if let Some(ref goal) = query.goal_filter {
            if let Some(goal_set) = self.goal_index.get(goal.as_str()) {
                return Some(goal_set.clone());
            } else {
                // No memories with this goal - return empty set
                return Some(HashSet::new());
            }
        }

        None
    }

    /// Search using pattern clustering.
    pub(crate) fn search_pattern(
        &self,
        min_cluster_size: usize,
        coherence_threshold: f32,
        query: &PurposeQuery,
    ) -> PurposeIndexResult<Vec<PurposeSearchResult>> {
        // Collect all entries for clustering
        let entries: Vec<PurposeIndexEntry> = self
            .vectors
            .iter()
            .filter_map(|(id, vector)| {
                let metadata = self.metadata.get(id)?;
                Some(PurposeIndexEntry::new(
                    *id,
                    vector.clone(),
                    metadata.clone(),
                ))
            })
            .collect();

        if entries.is_empty() {
            return Ok(Vec::new());
        }

        // Determine k based on entry count and min_cluster_size
        let max_k = entries.len() / min_cluster_size.max(1);
        let k = max_k.max(1).min(entries.len());

        // Run clustering
        let config = KMeansConfig::with_k(k)?;
        let clusterer = StandardKMeans::new();
        let clustering_result = clusterer.cluster_purposes(&entries, &config)?;

        // Filter clusters by size and coherence, collect matching members
        let mut results: Vec<PurposeSearchResult> = Vec::new();

        for cluster in clustering_result.clusters {
            if cluster.len() >= min_cluster_size && cluster.coherence >= coherence_threshold {
                for memory_id in cluster.members {
                    if let (Some(vector), Some(metadata)) =
                        (self.vectors.get(&memory_id), self.metadata.get(&memory_id))
                    {
                        // Apply additional filters
                        if let Some(ref goal_filter) = query.goal_filter {
                            if metadata.primary_goal.as_str() != goal_filter.as_str() {
                                continue;
                            }
                        }

                        results.push(PurposeSearchResult::new(
                            memory_id,
                            cluster.coherence, // Use cluster coherence as similarity
                            vector.clone(),
                            metadata.clone(),
                        ));
                    }
                }
            }
        }

        // Sort by similarity (coherence) descending
        results.sort_by(|a, b| {
            b.purpose_similarity
                .partial_cmp(&a.purpose_similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply min_similarity filter and limit
        results.retain(|r| r.purpose_similarity >= query.min_similarity);
        results.truncate(query.limit);

        Ok(results)
    }
}
