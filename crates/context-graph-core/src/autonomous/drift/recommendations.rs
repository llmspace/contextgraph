//! Recommendation types for drift detection.
//!
//! Contains types for generating actionable recommendations based on drift analysis.

use crate::teleological::Embedder;

use super::types::DriftLevel;

// ============================================
// RECOMMENDATION TYPES
// ============================================

/// Recommendation for addressing drift in a specific embedder.
#[derive(Debug)]
pub struct DriftRecommendation {
    /// The embedder with drift
    pub embedder: Embedder,
    /// Description of the issue
    pub issue: String,
    /// Suggested action
    pub suggestion: String,
    /// Priority based on drift severity
    pub priority: RecommendationPriority,
}

/// Priority levels for recommendations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    /// Low priority (shouldn't occur in recommendations)
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

impl From<DriftLevel> for RecommendationPriority {
    fn from(level: DriftLevel) -> Self {
        match level {
            DriftLevel::Critical => RecommendationPriority::Critical,
            DriftLevel::High => RecommendationPriority::High,
            DriftLevel::Medium => RecommendationPriority::Medium,
            DriftLevel::Low => RecommendationPriority::Low,
            DriftLevel::None => RecommendationPriority::Low,
        }
    }
}
