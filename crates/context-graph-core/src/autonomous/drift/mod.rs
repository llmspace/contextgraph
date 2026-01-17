//! Teleological Drift Detection with Per-Embedder Analysis (TASK-LOGIC-010)
//!
//! This module implements drift detection across all 13 embedders, providing
//! granular insight into which semantic dimensions are drifting from established goals.
//!
//! # Architecture
//!
//! From constitution.yaml (ARCH-02): "Compare Only Compatible Embedding Types (Apples-to-Apples)"
//! - E1 compares with E1, E5 with E5, NEVER cross-embedder
//! - Uses TeleologicalComparator for all comparisons
//!
//! # Design Philosophy
//!
//! Per-embedder drift detection provides:
//! 1. **Granular insight**: Know exactly which embedder is drifting
//! 2. **Early warning**: Detect drift in specific dimensions before overall alignment degrades
//! 3. **Actionable recommendations**: Embedder-specific suggestions for correction
//! 4. **Trend analysis**: Predict when drift will become critical
//!
//! # Example
//!
//! ```ignore
//! use context_graph_core::autonomous::drift::{TeleologicalDriftDetector, DriftThresholds};
//! use context_graph_core::teleological::TeleologicalComparator;
//!
//! let comparator = TeleologicalComparator::new();
//! let detector = TeleologicalDriftDetector::new(comparator);
//!
//! let result = detector.check_drift(&memories, &goal, &comparison_type)?;
//! println!("Overall drift level: {:?}", result.overall_drift.drift_level);
//! for info in &result.most_drifted_embedders {
//!     println!("{:?}: {:?}", info.embedder, info.drift_level);
//! }
//! ```

mod detector;
mod error;
mod history;
mod recommendations;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use types::{
    // Legacy types (NORTH-010/011)
    DriftConfig,
    DriftDataPoint,
    // Core types (TASK-LOGIC-010)
    DriftLevel,
    DriftMonitoring,
    // Result types
    DriftResult,
    DriftSeverity,
    DriftState,
    DriftThresholds,
    DriftTrend,
    EmbedderDriftInfo,
    OverallDrift,
    PerEmbedderDrift,
    MAX_MOST_DRIFTED,
    MIN_TREND_SAMPLES,
    // Constants
    NUM_EMBEDDERS,
};

pub use error::DriftError;

pub use detector::TeleologicalDriftDetector;

pub use history::{DriftHistory, DriftHistoryEntry, TrendAnalysis};

pub use recommendations::{DriftRecommendation, RecommendationPriority};
