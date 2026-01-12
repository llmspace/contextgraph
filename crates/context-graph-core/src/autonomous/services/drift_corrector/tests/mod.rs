//! Tests for drift corrector module.
//!
//! This module is organized into submodules for better maintainability:
//! - `strategy_tests`: Tests for CorrectionStrategy enum variants
//! - `config_tests`: Tests for DriftCorrectorConfig
//! - `corrector_tests`: Tests for DriftCorrector core functionality
//! - `integration_tests`: Integration and serialization tests

mod config_tests;
mod corrector_tests;
mod integration_tests;
mod strategy_tests;

// Re-export common test utilities
#[cfg(test)]
pub(crate) mod common {
    pub use crate::autonomous::services::drift_corrector::{
        CorrectionResult, CorrectionStrategy, DriftCorrector, DriftCorrectorConfig,
    };
    pub use crate::autonomous::{DriftConfig, DriftSeverity, DriftState, DriftTrend};
}
