//! Teleological Retrieval Pipeline trait and implementation.
//!
//! This module provides the `TeleologicalRetrievalPipeline` trait and
//! `DefaultTeleologicalPipeline` implementation that orchestrates the
//! 5-stage teleological retrieval process.
//!
//! # TASK-L008 Implementation
//!
//! Implements the pipeline per constitution.yaml retrieval_pipeline spec:
//! - Stage 1: SPLADE sparse pre-filter (<5ms, 10K candidates)
//! - Stage 2: Matryoshka 128D fast ANN (<10ms, 1K candidates)
//! - Stage 3: Full 13-space HNSW (<20ms, 100 candidates)
//! - Stage 4: Teleological alignment filter (<10ms, 50 candidates)
//! - Stage 5: Late interaction reranking (<15ms, final results)
//!
//! Total target: <60ms @ 1M memories
//!
//! # Dependencies (L001-L007)
//!
//! - L001: MultiEmbeddingQueryExecutor (Stages 1-3)
//! - L002: PurposeVectorComputer (Stage 4)
//! - L003: GoalAlignmentCalculator (Stage 4)
//! - L004: JohariTransitionManager (Stage 4 filtering)
//! - L005: Per-Space HNSW Index (Stage 3)
//! - L006: Purpose Pattern Index (Stage 4)
//! - L007: CrossSpaceSimilarityEngine (Stage 3)
//!
//! FAIL FAST: All errors are explicit, no silent fallbacks.

mod default;
mod filtering;
mod traits;

#[cfg(test)]
mod tests;

// Re-export public API
pub use default::DefaultTeleologicalPipeline;
pub use traits::{PipelineHealth, TeleologicalRetrievalPipeline};
