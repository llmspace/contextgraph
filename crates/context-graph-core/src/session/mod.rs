//! Session management for context graph.
//!
//! This module provides session-level tracking and analysis:
//!
//! - [`IntentTracker`]: E10 intent drift detection within sessions
//!
//! # Intent Drift Detection
//!
//! The intent tracker monitors E10 intent embeddings across a session
//! to detect when the user's intent shifts significantly. This is useful for:
//!
//! - Alerting on topic changes
//! - Session summarization
//! - Context relevance scoring
//!
//! # Example
//!
//! ```ignore
//! use context_graph_core::session::{IntentTracker, IntentDrift};
//!
//! let mut tracker = IntentTracker::new("session-123")
//!     .with_threshold(0.4)  // Drift sensitivity
//!     .with_window_size(5); // Recent intent window
//!
//! // Record intents as they come in
//! for intent_vector in session_intents {
//!     if let Some(drift) = tracker.record_intent(intent_vector, Some("query text")) {
//!         println!("Intent drift detected! Score: {}", drift.score);
//!     }
//! }
//!
//! // Get trajectory statistics
//! let stats = tracker.trajectory_stats();
//! println!("Avg similarity: {}", stats.avg_pairwise_similarity);
//! ```

mod intent_tracker;

pub use intent_tracker::{
    IntentDrift, IntentSnapshot, IntentTracker, IntentTrajectoryStats,
    DEFAULT_DRIFT_THRESHOLD, DEFAULT_WINDOW_SIZE, MAX_HISTORY_SIZE,
};
