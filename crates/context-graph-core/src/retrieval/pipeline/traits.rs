//! Teleological retrieval pipeline trait definitions.
//!
//! This module defines the `TeleologicalRetrievalPipeline` trait and
//! associated types for the 5-stage teleological retrieval process.

use std::time::Duration;

use async_trait::async_trait;

use crate::error::CoreResult;
use crate::types::fingerprint::TeleologicalFingerprint;

use super::super::teleological_query::TeleologicalQuery;
use super::super::teleological_result::{ScoredMemory, TeleologicalRetrievalResult};

/// Pipeline health status.
#[derive(Clone, Debug)]
pub struct PipelineHealth {
    /// Whether all components are operational.
    pub is_healthy: bool,

    /// Number of embedding spaces available.
    pub spaces_available: usize,

    /// Whether goal hierarchy is configured.
    pub has_goal_hierarchy: bool,

    /// Index size (total memories).
    pub index_size: usize,

    /// Last successful query time.
    pub last_query_time: Option<Duration>,
}

/// Trait for teleological retrieval pipeline execution.
///
/// Implementations must coordinate the 5-stage retrieval process and
/// integrate teleological filtering (purpose alignment, goal hierarchy,
/// Johari quadrants).
///
/// # Performance Requirements
///
/// - Total pipeline: <60ms @ 1M memories
/// - Thread-safe (Send + Sync)
/// - Graceful degradation if individual stages fail
///
/// # Error Handling
///
/// FAIL FAST: Any critical error (empty query, invalid config) returns
/// immediately with `CoreError`. Stage-level failures are logged but
/// don't abort the pipeline (graceful degradation).
#[async_trait]
pub trait TeleologicalRetrievalPipeline: Send + Sync {
    /// Execute the full 5-stage teleological retrieval pipeline.
    ///
    /// # Arguments
    /// * `query` - The teleological query with text/embeddings, goals, filters
    ///
    /// # Returns
    /// `TeleologicalRetrievalResult` with ranked results and timing breakdown.
    ///
    /// # Errors
    /// - `CoreError::ValidationError` if query validation fails
    /// - `CoreError::RetrievalError` if pipeline cannot complete
    ///
    /// # Performance
    /// Target: <60ms for 1M memories in index
    async fn execute(&self, query: &TeleologicalQuery) -> CoreResult<TeleologicalRetrievalResult>;

    /// Execute only Stage 4 (teleological filtering) on pre-fetched candidates.
    ///
    /// Use this when you already have candidates from another source and
    /// only need teleological filtering/scoring.
    ///
    /// # Arguments
    /// * `candidates` - Pre-fetched fingerprints to filter/score
    /// * `query` - Query with goals and filters
    ///
    /// # Returns
    /// Filtered and scored results
    async fn filter_by_alignment(
        &self,
        candidates: &[&TeleologicalFingerprint],
        query: &TeleologicalQuery,
    ) -> CoreResult<Vec<ScoredMemory>>;

    /// Check if the pipeline is healthy and ready for queries.
    async fn health_check(&self) -> CoreResult<PipelineHealth>;
}
