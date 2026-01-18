//! Autonomous mode configuration types.

use serde::{Deserialize, Serialize};

use crate::autonomous::bootstrap::BootstrapConfig;
use crate::autonomous::curation::{ConsolidationConfig, PruningConfig};
use crate::autonomous::drift::DriftConfig;
use crate::autonomous::evolution::GoalEvolutionConfig;
use crate::autonomous::thresholds::AdaptiveThresholdConfig;

/// Complete autonomous mode configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutonomousConfig {
    /// Enable autonomous mode
    pub enabled: bool,
    /// Bootstrap configuration for initial setup
    pub bootstrap: BootstrapConfig,
    /// Adaptive threshold learning configuration
    pub thresholds: AdaptiveThresholdConfig,
    /// Memory pruning configuration
    pub pruning: PruningConfig,
    /// Memory consolidation configuration
    pub consolidation: ConsolidationConfig,
    /// Drift detection configuration
    pub drift: DriftConfig,
    /// Goal evolution configuration
    pub goals: GoalEvolutionConfig,
}

impl Default for AutonomousConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bootstrap: BootstrapConfig::default(),
            thresholds: AdaptiveThresholdConfig::default(),
            pruning: PruningConfig::default(),
            consolidation: ConsolidationConfig::default(),
            drift: DriftConfig::default(),
            goals: GoalEvolutionConfig::default(),
        }
    }
}

impl AutonomousConfig {
    /// Create a disabled configuration (all features off)
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            bootstrap: BootstrapConfig {
                auto_init: false,
                ..Default::default()
            },
            thresholds: AdaptiveThresholdConfig {
                enabled: false,
                ..Default::default()
            },
            pruning: PruningConfig {
                enabled: false,
                ..Default::default()
            },
            consolidation: ConsolidationConfig {
                enabled: false,
                ..Default::default()
            },
            drift: DriftConfig {
                auto_correct: false,
                ..Default::default()
            },
            goals: GoalEvolutionConfig {
                auto_discover: false,
                ..Default::default()
            },
        }
    }

    /// Check if any autonomous feature is enabled
    pub fn has_any_enabled(&self) -> bool {
        self.enabled
            || self.bootstrap.auto_init
            || self.thresholds.enabled
            || self.pruning.enabled
            || self.consolidation.enabled
            || self.drift.auto_correct
            || self.goals.auto_discover
    }
}
