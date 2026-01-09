//! Teleological Logic Layer Services.
//!
//! This module provides the computational services for the teleological fusion system.
//! These services implement TASKS-TELEO 007-015.
//!
//! # Services
//!
//! - **SynergyService** (007): Cross-embedding synergy computation
//! - **CorrelationExtractor** (008): 78 cross-correlation value extraction
//! - **MeaningPipeline** (009): Embedding → meaning extraction pipeline
//! - **TuckerDecomposer** (010): Tensor factorization for compression
//! - **GroupAggregator** (011): 13D → 6D group aggregation
//! - **FusionEngine** (012): Multi-embedding fusion orchestration
//! - **MultiSpaceRetriever** (013): Teleological-aware retrieval
//! - **FeedbackLearner** (014): GWT feedback learning loop
//! - **ProfileManager** (015): Task-specific profile management

pub mod synergy_service;
pub mod correlation_extractor;
pub mod meaning_pipeline;
pub mod tucker_decomposer;
pub mod group_aggregator;
pub mod fusion_engine;
pub mod multi_space_retriever;
pub mod feedback_learner;
pub mod profile_manager;

// Re-exports
pub use synergy_service::SynergyService;
pub use correlation_extractor::CorrelationExtractor;
pub use meaning_pipeline::MeaningPipeline;
pub use tucker_decomposer::TuckerDecomposer;
pub use group_aggregator::GroupAggregator;
pub use fusion_engine::FusionEngine;
pub use multi_space_retriever::MultiSpaceRetriever;
pub use feedback_learner::FeedbackLearner;
pub use profile_manager::ProfileManager;
