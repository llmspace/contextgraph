//! Configuration for the FeedbackLearner service.

/// Configuration for the FeedbackLearner service.
#[derive(Clone, Debug)]
pub struct FeedbackLearnerConfig {
    /// Learning rate for gradient updates (default: 0.01)
    pub learning_rate: f32,
    /// Momentum for gradient accumulation (default: 0.9)
    pub momentum: f32,
    /// Scale factor for positive rewards (default: 1.0)
    pub reward_scale: f32,
    /// Scale factor for negative penalties (default: 0.5)
    pub penalty_scale: f32,
    /// Minimum feedback count before learning (default: 10)
    pub min_feedback_count: usize,
}

impl Default for FeedbackLearnerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            reward_scale: 1.0,
            penalty_scale: 0.5,
            min_feedback_count: 10,
        }
    }
}
