//! DriftDetector Service
//!
//! Detects alignment drift using rolling statistical analysis.
//! Monitors alignment observations over a configurable window and calculates
//! drift severity, trend direction, and provides actionable recommendations.
//!
//! # Architecture
//!
//! The DriftDetector operates on alignment observations (scores between 0.0 and 1.0)
//! and maintains rolling statistics to detect when alignment drifts from baseline.
//!
//! Key metrics:
//! - Rolling mean: EWMA of recent alignment scores
//! - Rolling variance: Measure of alignment stability
//! - Trend: Direction of alignment change (Improving/Declining/Stable)
//! - Severity: Classification of drift magnitude (None/Mild/Moderate/Severe)
//!
//! # Example
//!
//! ```rust
//! use context_graph_core::autonomous::services::drift_detector::{DriftDetector, DriftRecommendation};
//! use context_graph_core::autonomous::drift::DriftConfig;
//!
//! let mut detector = DriftDetector::new();
//!
//! // Add alignment observations
//! detector.add_observation(0.80, 1000);
//! detector.add_observation(0.75, 2000);
//! detector.add_observation(0.70, 3000);
//!
//! // Check drift severity
//! let severity = detector.detect_drift();
//! let trend = detector.compute_trend();
//!
//! if detector.requires_attention() {
//!     let recommendation = detector.get_recommendation();
//!     // Handle based on recommendation
//! }
//! ```

mod detector;
#[cfg(test)]
mod tests;
mod types;

// Re-export public API
pub use detector::DriftDetector;
pub use types::{DetectorDataPoint, DetectorState, DriftRecommendation, MIN_SAMPLES_DEFAULT};
