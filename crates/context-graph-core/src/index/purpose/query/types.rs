//! Core purpose query type.
//!
//! This module provides [`PurposeQuery`] which represents a complete query
//! with filters and constraints for purpose-based search operations.

use super::super::entry::GoalId;
use super::super::error::{PurposeIndexError, PurposeIndexResult};
use super::builder::PurposeQueryBuilder;
use super::target::PurposeQueryTarget;

/// Query for purpose-based search operations.
///
/// # Structure
///
/// A `PurposeQuery` consists of:
/// - `target`: What to search for (vector, pattern, or memory-based)
/// - `limit`: Maximum number of results to return
/// - `min_similarity`: Minimum similarity threshold [0.0, 1.0]
/// - `goal_filter`: Optional filter by goal ID
///
/// # Construction
///
/// Use the builder pattern via [`PurposeQueryBuilder`] for flexible construction:
///
/// ```ignore
/// let query = PurposeQuery::builder()
///     .target(PurposeQueryTarget::Vector(pv))
///     .limit(10)
///     .min_similarity(0.7)
///     .goal_filter(GoalId::new("master_ml"))
///     .build()?;
/// ```
///
/// Or use the direct constructor:
///
/// ```ignore
/// let query = PurposeQuery::new(
///     PurposeQueryTarget::Vector(pv),
///     10,
///     0.7,
/// )?;
/// ```
///
/// # Fail-Fast Semantics
///
/// Validation is performed at construction time:
/// - `limit` must be > 0
/// - `min_similarity` must be in [0.0, 1.0]
#[derive(Clone, Debug)]
pub struct PurposeQuery {
    /// The query target specifying what to search for.
    pub target: PurposeQueryTarget,

    /// Maximum number of results to return.
    ///
    /// Must be > 0.
    pub limit: usize,

    /// Minimum similarity threshold [0.0, 1.0].
    ///
    /// Results with similarity below this threshold are filtered out.
    pub min_similarity: f32,

    /// Optional filter by goal ID.
    ///
    /// When set, only memories aligned with this goal are returned.
    pub goal_filter: Option<GoalId>,
}

impl PurposeQuery {
    /// Create a new PurposeQuery with validation.
    ///
    /// # Arguments
    ///
    /// * `target` - The query target
    /// * `limit` - Maximum results to return (must be > 0)
    /// * `min_similarity` - Minimum similarity threshold [0.0, 1.0]
    ///
    /// # Errors
    ///
    /// Returns `PurposeIndexError::InvalidQuery` if:
    /// - `limit` is 0
    /// - `min_similarity` is not in [0.0, 1.0]
    ///
    /// # Example
    ///
    /// ```ignore
    /// let query = PurposeQuery::new(
    ///     PurposeQueryTarget::Vector(purpose_vector),
    ///     10,
    ///     0.7,
    /// )?;
    /// ```
    pub fn new(
        target: PurposeQueryTarget,
        limit: usize,
        min_similarity: f32,
    ) -> PurposeIndexResult<Self> {
        Self::validate_limit(limit)?;
        Self::validate_min_similarity(min_similarity)?;

        Ok(Self {
            target,
            limit,
            min_similarity,
            goal_filter: None,
        })
    }

    /// Create a builder for constructing PurposeQuery.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let query = PurposeQuery::builder()
    ///     .target(PurposeQueryTarget::Vector(pv))
    ///     .limit(10)
    ///     .min_similarity(0.5)
    ///     .build()?;
    /// ```
    #[inline]
    pub fn builder() -> PurposeQueryBuilder {
        PurposeQueryBuilder::new()
    }

    /// Add a goal filter to the query.
    ///
    /// Returns a new query with the filter applied.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let filtered = query.with_goal_filter(GoalId::new("master_ml"));
    /// ```
    #[must_use]
    pub fn with_goal_filter(mut self, goal: GoalId) -> Self {
        self.goal_filter = Some(goal);
        self
    }

    /// Validate that this query is internally consistent.
    ///
    /// This is called automatically during construction but can be
    /// called again if the query is modified.
    ///
    /// # Errors
    ///
    /// Returns `PurposeIndexError::InvalidQuery` if validation fails.
    pub fn validate(&self) -> PurposeIndexResult<()> {
        Self::validate_limit(self.limit)?;
        Self::validate_min_similarity(self.min_similarity)?;
        Ok(())
    }

    /// Validate the limit parameter.
    ///
    /// # Errors
    ///
    /// Returns error if limit is 0.
    #[inline]
    pub(crate) fn validate_limit(limit: usize) -> PurposeIndexResult<()> {
        if limit == 0 {
            return Err(PurposeIndexError::invalid_query("limit must be > 0"));
        }
        Ok(())
    }

    /// Validate the min_similarity parameter.
    ///
    /// # Errors
    ///
    /// Returns error if min_similarity is not in [0.0, 1.0].
    #[inline]
    pub(crate) fn validate_min_similarity(min_similarity: f32) -> PurposeIndexResult<()> {
        if !(0.0..=1.0).contains(&min_similarity) {
            return Err(PurposeIndexError::invalid_query(format!(
                "min_similarity {} must be in [0.0, 1.0]",
                min_similarity
            )));
        }
        if min_similarity.is_nan() {
            return Err(PurposeIndexError::invalid_query(
                "min_similarity cannot be NaN",
            ));
        }
        Ok(())
    }

    /// Check if this query has any filters applied.
    #[inline]
    pub fn has_filters(&self) -> bool {
        self.goal_filter.is_some()
    }

    /// Get the number of filters applied.
    #[inline]
    pub fn filter_count(&self) -> usize {
        if self.goal_filter.is_some() { 1 } else { 0 }
    }
}
