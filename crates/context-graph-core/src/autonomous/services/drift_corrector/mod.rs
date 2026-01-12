//! NORTH-011: Drift Corrector Service
//!
//! This service applies correction strategies based on drift severity detected
//! by the DriftDetector. It implements a FAIL FAST pattern with no mock data.
//!
//! # Correction Strategies
//!
//! - `NoAction`: No correction needed for minimal drift
//! - `ThresholdAdjustment`: Adjust alignment thresholds
//! - `WeightRebalance`: Rebalance section weights
//! - `GoalReinforcement`: Emphasize goal alignment
//! - `EmergencyIntervention`: Require human intervention for critical drift

mod config;
mod corrector;
mod result;
mod strategy;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use config::DriftCorrectorConfig;
pub use corrector::DriftCorrector;
pub use result::CorrectionResult;
pub use strategy::CorrectionStrategy;
