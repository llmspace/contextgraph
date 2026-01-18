//! Purpose query types for indexed retrieval.
//!
//! # CRITICAL: NO FALLBACKS
//!
//! All validation failures are fatal errors.
//! Invalid queries must not execute - fail immediately.
//!
//! # Overview
//!
//! This module provides query types for the Purpose Pattern Index (Stage 4):
//!
//! - [`PurposeQueryTarget`]: Specifies what to search for (vector, pattern, or memory-based)
//! - [`PurposeQuery`]: Complete query with filters and constraints
//! - [`PurposeSearchResult`]: Results from purpose-based searches
//!
//! # Usage
//!
//! ```ignore
//! use context_graph_core::index::purpose::query::{
//!     PurposeQuery, PurposeQueryTarget, PurposeSearchResult,
//! };
//! use context_graph_core::types::fingerprint::PurposeVector;
//! use context_graph_core::purpose::GoalId;
//!
//! // Create a query from a purpose vector
//! let query = PurposeQuery::builder()
//!     .target(PurposeQueryTarget::Vector(purpose_vector))
//!     .limit(10)
//!     .min_similarity(0.7)
//!     .build()?;
//!
//! // Add optional filters
//! let filtered_query = query.with_goal_filter(GoalId::new("master_ml"));
//! ```
//!
//! # Fail-Fast Semantics
//!
//! All validation is performed at query construction time:
//! - `min_similarity` must be in [0.0, 1.0]
//! - `limit` must be > 0
//! - Invalid values result in `PurposeIndexError::InvalidQuery`

mod builder;
mod result;
mod target;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public items for backwards compatibility
pub use builder::PurposeQueryBuilder;
pub use result::PurposeSearchResult;
pub use target::PurposeQueryTarget;
pub use types::PurposeQuery;
