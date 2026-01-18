//! Type definitions for feedback learning.

use uuid::Uuid;

use super::config::FeedbackLearnerConfig;
use crate::teleological::types::NUM_EMBEDDERS;

/// Type of feedback received for a teleological vector.
#[derive(Clone, Debug, PartialEq)]
pub enum FeedbackType {
    /// Positive feedback with magnitude (successful retrieval, high coherence)
    Positive { magnitude: f32 },
    /// Negative feedback with magnitude (failed retrieval, low coherence)
    Negative { magnitude: f32 },
    /// Neutral feedback (neither reward nor penalty)
    Neutral,
}

impl FeedbackType {
    /// Get the signed reward value for this feedback type.
    ///
    /// - Positive returns +magnitude * reward_scale
    /// - Negative returns -magnitude * penalty_scale
    /// - Neutral returns 0.0
    pub fn reward_value(&self, config: &FeedbackLearnerConfig) -> f32 {
        match self {
            FeedbackType::Positive { magnitude } => magnitude * config.reward_scale,
            FeedbackType::Negative { magnitude } => -magnitude * config.penalty_scale,
            FeedbackType::Neutral => 0.0,
        }
    }

    /// Check if this is positive feedback.
    #[inline]
    pub fn is_positive(&self) -> bool {
        matches!(self, FeedbackType::Positive { .. })
    }

    /// Check if this is negative feedback.
    #[inline]
    pub fn is_negative(&self) -> bool {
        matches!(self, FeedbackType::Negative { .. })
    }

    /// Check if this is neutral feedback.
    #[inline]
    pub fn is_neutral(&self) -> bool {
        matches!(self, FeedbackType::Neutral)
    }
}

/// A single feedback event for a teleological vector.
#[derive(Clone, Debug)]
pub struct FeedbackEvent {
    /// UUID of the TeleologicalVector this feedback is for
    pub vector_id: Uuid,
    /// Type and magnitude of feedback
    pub feedback_type: FeedbackType,
    /// Timestamp when feedback was recorded (milliseconds since epoch)
    pub timestamp: u64,
    /// Optional context string describing the feedback source
    pub context: Option<String>,
    /// Per-embedder contribution weights at time of feedback (0.0-1.0 each)
    /// Used to attribute feedback to specific embedders
    pub embedder_contributions: [f32; NUM_EMBEDDERS],
}

impl FeedbackEvent {
    /// Create a new feedback event.
    ///
    /// # Arguments
    /// * `vector_id` - UUID of the TeleologicalVector
    /// * `feedback_type` - Type of feedback (positive/negative/neutral)
    /// * `timestamp` - Milliseconds since epoch
    ///
    /// # Panics
    ///
    /// Does not panic. Embedder contributions default to uniform (1/13 each).
    pub fn new(vector_id: Uuid, feedback_type: FeedbackType, timestamp: u64) -> Self {
        Self {
            vector_id,
            feedback_type,
            timestamp,
            context: None,
            embedder_contributions: [1.0 / NUM_EMBEDDERS as f32; NUM_EMBEDDERS],
        }
    }

    /// Create with context.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Create with explicit embedder contributions.
    ///
    /// # Panics
    ///
    /// Panics if contributions don't sum to approximately 1.0 (FAIL FAST).
    pub fn with_contributions(mut self, contributions: [f32; NUM_EMBEDDERS]) -> Self {
        let sum: f32 = contributions.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "FAIL FAST: embedder contributions must sum to ~1.0, got {}",
            sum
        );
        self.embedder_contributions = contributions;
        self
    }
}

/// Result of a learning cycle.
#[derive(Clone, Debug)]
pub struct LearningResult {
    /// Per-embedder adjustments computed from gradients
    pub adjustments: Vec<f32>,
    /// Change in overall confidence
    pub confidence_delta: f32,
    /// Number of feedback events processed
    pub events_processed: usize,
}

impl LearningResult {
    /// Create a new learning result.
    pub fn new(adjustments: Vec<f32>, confidence_delta: f32, events_processed: usize) -> Self {
        assert!(
            adjustments.len() == NUM_EMBEDDERS,
            "FAIL FAST: adjustments must have {} elements, got {}",
            NUM_EMBEDDERS,
            adjustments.len()
        );
        Self {
            adjustments,
            confidence_delta,
            events_processed,
        }
    }

    /// Check if any learning occurred.
    #[inline]
    pub fn has_adjustments(&self) -> bool {
        self.adjustments.iter().any(|&a| a.abs() > f32::EPSILON)
    }

    /// Get total adjustment magnitude (L1 norm).
    pub fn total_adjustment_magnitude(&self) -> f32 {
        self.adjustments.iter().map(|a| a.abs()).sum()
    }
}
