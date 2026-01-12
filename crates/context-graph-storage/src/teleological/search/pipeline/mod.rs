//! 5-Stage Retrieval Pipeline with Progressive Filtering.
//!
//! # Overview
//!
//! Implements a 5-stage retrieval pipeline optimizing latency by progressively
//! filtering candidates through stages of increasing precision but decreasing speed.
//! Target: <60ms at 1M memories.
//!
//! # Pipeline Stages
//!
//! 1. **Stage 1: SPLADE/BM25 Sparse Pre-filter** (E13 or E6)
//!    - Uses inverted index, NOT HNSW
//!    - Broad recall with lexical matching
//!    - Input: 1M+ -> Output: 10K candidates
//!    - Latency: <5ms
//!
//! 2. **Stage 2: Matryoshka 128D Fast ANN** (E1Matryoshka128)
//!    - Uses 128D truncated E1 for speed
//!    - Fast approximate filtering
//!    - Input: 10K -> Output: 1K candidates
//!    - Latency: <10ms
//!
//! 3. **Stage 3: Multi-space RRF Rerank**
//!    - Uses MultiEmbedderSearch across multiple spaces
//!    - Reciprocal Rank Fusion for score combination
//!    - Input: 1K -> Output: 100 candidates
//!    - Latency: <20ms
//!
//! 4. **Stage 4: Teleological Alignment Filter**
//!    - Uses PurposeVector (13D) for goal alignment
//!    - Filters by alignment threshold >=0.55
//!    - Input: 100 -> Output: 50 candidates
//!    - Latency: <10ms
//!
//! 5. **Stage 5: Late Interaction MaxSim** (E12)
//!    - Uses ColBERT-style token-level matching, NOT HNSW
//!    - Final precision reranking
//!    - Input: 50 -> Output: k results (typically 10)
//!    - Latency: <15ms
//!
//! # Design Philosophy
//!
//! **FAIL FAST. NO FALLBACKS.**
//!
//! All errors are fatal. No recovery attempts. This ensures:
//! - Bugs are caught early in development
//! - Data integrity is preserved
//! - Clear error messages for debugging
//!
//! # Example
//!
//! ```no_run
//! use context_graph_storage::teleological::search::{
//!     RetrievalPipeline, PipelineBuilder, PipelineStage,
//! };
//! use context_graph_storage::teleological::indexes::EmbedderIndexRegistry;
//! use std::sync::Arc;
//!
//! // Create pipeline with registry
//! let registry = Arc::new(EmbedderIndexRegistry::new());
//! let pipeline = RetrievalPipeline::new(
//!     registry,
//!     None, // Use default SPLADE index
//!     None, // Use default token storage
//! );
//!
//! // Execute with builder pattern
//! let result = PipelineBuilder::new()
//!     .splade(vec![/* sparse query */])
//!     .matryoshka(vec![0.5f32; 128])
//!     .semantic(vec![0.5f32; 1024])
//!     .tokens(vec![vec![0.5f32; 128]; 10])
//!     .purpose([0.5f32; 13])
//!     .k(10)
//!     .execute(&pipeline);
//! ```

mod builder;
mod execution;
mod stages;
mod traits;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use builder::PipelineBuilder;
pub use execution::RetrievalPipeline;
pub use traits::{InMemorySpladeIndex, InMemoryTokenStorage, SpladeIndex, TokenStorage};
pub use types::{
    PipelineCandidate, PipelineConfig, PipelineError, PipelineResult, PipelineStage, StageConfig,
    StageResult,
};
