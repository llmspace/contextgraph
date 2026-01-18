//! Builder for purpose queries.
//!
//! This module provides [`PurposeQueryBuilder`] for constructing
//! [`PurposeQuery`] instances with validation.

use super::super::entry::GoalId;
use super::super::error::{PurposeIndexError, PurposeIndexResult};
use super::target::PurposeQueryTarget;
use super::types::PurposeQuery;

/// Builder for constructing [`PurposeQuery`] instances.
///
/// Provides a fluent interface for building queries with validation
/// performed at the final `build()` step.
///
/// # Example
///
/// ```ignore
/// let query = PurposeQueryBuilder::new()
///     .target(PurposeQueryTarget::Vector(pv))
///     .limit(10)
///     .min_similarity(0.7)
///     .goal_filter(GoalId::new("learn_pytorch"))
///     .build()?;
/// ```
#[derive(Clone, Debug, Default)]
pub struct PurposeQueryBuilder {
    target: Option<PurposeQueryTarget>,
    limit: Option<usize>,
    min_similarity: Option<f32>,
    goal_filter: Option<GoalId>,
}

impl PurposeQueryBuilder {
    /// Create a new builder with default values.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the query target.
    ///
    /// # Required
    ///
    /// This field is required. `build()` will fail if not set.
    #[must_use]
    pub fn target(mut self, target: PurposeQueryTarget) -> Self {
        self.target = Some(target);
        self
    }

    /// Set the maximum number of results.
    ///
    /// # Required
    ///
    /// This field is required. `build()` will fail if not set.
    ///
    /// # Validation
    ///
    /// Must be > 0.
    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set the minimum similarity threshold.
    ///
    /// # Required
    ///
    /// This field is required. `build()` will fail if not set.
    ///
    /// # Validation
    ///
    /// Must be in [0.0, 1.0].
    #[must_use]
    pub fn min_similarity(mut self, min_similarity: f32) -> Self {
        self.min_similarity = Some(min_similarity);
        self
    }

    /// Set an optional goal filter.
    #[must_use]
    pub fn goal_filter(mut self, goal: GoalId) -> Self {
        self.goal_filter = Some(goal);
        self
    }

    /// Build the query with validation.
    ///
    /// # Errors
    ///
    /// Returns `PurposeIndexError::InvalidQuery` if:
    /// - `target` is not set
    /// - `limit` is not set or is 0
    /// - `min_similarity` is not set or not in [0.0, 1.0]
    pub fn build(self) -> PurposeIndexResult<PurposeQuery> {
        let target = self
            .target
            .ok_or_else(|| PurposeIndexError::invalid_query("target is required"))?;

        let limit = self
            .limit
            .ok_or_else(|| PurposeIndexError::invalid_query("limit is required"))?;

        let min_similarity = self
            .min_similarity
            .ok_or_else(|| PurposeIndexError::invalid_query("min_similarity is required"))?;

        let mut query = PurposeQuery::new(target, limit, min_similarity)?;

        if let Some(goal) = self.goal_filter {
            query = query.with_goal_filter(goal);
        }

        Ok(query)
    }
}
