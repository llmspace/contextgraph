//! Optimization actions and their cost/benefit calculations.

use serde::{Deserialize, Serialize};

use super::trigger::OptimizationTrigger;

/// Actions that can be performed during optimization
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum OptimizationAction {
    /// Rebuild memory indices for faster retrieval
    ReindexMemories,
    /// Recompute alignment scores for all memories
    RecomputeAlignments,
    /// Remove stale or orphaned data
    PruneStaleData,
    /// Adjust weights based on usage patterns
    RebalanceWeights,
    /// Compact storage to reclaim space
    CompactStorage,
}

impl OptimizationAction {
    /// Get the estimated cost of this action (0.0 to 1.0)
    pub fn estimated_cost(&self) -> f32 {
        match self {
            Self::ReindexMemories => 0.8,
            Self::RecomputeAlignments => 0.9,
            Self::PruneStaleData => 0.4,
            Self::RebalanceWeights => 0.3,
            Self::CompactStorage => 0.7,
        }
    }

    /// Get the estimated benefit of this action (0.0 to 1.0)
    pub fn estimated_benefit(&self) -> f32 {
        match self {
            Self::ReindexMemories => 0.7,
            Self::RecomputeAlignments => 0.8,
            Self::PruneStaleData => 0.5,
            Self::RebalanceWeights => 0.6,
            Self::CompactStorage => 0.4,
        }
    }

    /// Get a human-readable description of this action
    pub fn description(&self) -> &'static str {
        match self {
            Self::ReindexMemories => "Rebuild memory indices for faster retrieval",
            Self::RecomputeAlignments => "Recompute alignment scores for all memories",
            Self::PruneStaleData => "Remove stale or orphaned data",
            Self::RebalanceWeights => "Adjust weights based on usage patterns",
            Self::CompactStorage => "Compact storage to reclaim space",
        }
    }

    /// Get actions recommended for a given trigger
    pub fn recommended_for_trigger(trigger: &OptimizationTrigger) -> Vec<Self> {
        match trigger {
            OptimizationTrigger::HighDrift { severity } => {
                if *severity >= 0.20 {
                    vec![
                        Self::RecomputeAlignments,
                        Self::RebalanceWeights,
                        Self::ReindexMemories,
                    ]
                } else {
                    vec![Self::RecomputeAlignments, Self::RebalanceWeights]
                }
            }
            OptimizationTrigger::LowPerformance { metric, .. } => {
                if metric.contains("retrieval") || metric.contains("search") {
                    vec![Self::ReindexMemories, Self::CompactStorage]
                } else {
                    vec![Self::RebalanceWeights, Self::PruneStaleData]
                }
            }
            OptimizationTrigger::MemoryPressure { usage_percent } => {
                if *usage_percent >= 95.0 {
                    vec![
                        Self::PruneStaleData,
                        Self::CompactStorage,
                        Self::ReindexMemories,
                    ]
                } else {
                    vec![Self::PruneStaleData, Self::CompactStorage]
                }
            }
            OptimizationTrigger::ScheduledMaintenance => {
                vec![
                    Self::PruneStaleData,
                    Self::CompactStorage,
                    Self::ReindexMemories,
                    Self::RebalanceWeights,
                ]
            }
            OptimizationTrigger::UserTriggered { .. } => {
                vec![
                    Self::RecomputeAlignments,
                    Self::ReindexMemories,
                    Self::RebalanceWeights,
                    Self::PruneStaleData,
                    Self::CompactStorage,
                ]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_estimated_cost() {
        assert!((OptimizationAction::ReindexMemories.estimated_cost() - 0.8).abs() < f32::EPSILON);
        assert!(
            (OptimizationAction::RecomputeAlignments.estimated_cost() - 0.9).abs() < f32::EPSILON
        );
        assert!((OptimizationAction::PruneStaleData.estimated_cost() - 0.4).abs() < f32::EPSILON);
        assert!((OptimizationAction::RebalanceWeights.estimated_cost() - 0.3).abs() < f32::EPSILON);
        assert!((OptimizationAction::CompactStorage.estimated_cost() - 0.7).abs() < f32::EPSILON);
        println!("[PASS] test_action_estimated_cost");
    }

    #[test]
    fn test_action_estimated_benefit() {
        assert!(
            (OptimizationAction::ReindexMemories.estimated_benefit() - 0.7).abs() < f32::EPSILON
        );
        assert!(
            (OptimizationAction::RecomputeAlignments.estimated_benefit() - 0.8).abs()
                < f32::EPSILON
        );
        assert!(
            (OptimizationAction::PruneStaleData.estimated_benefit() - 0.5).abs() < f32::EPSILON
        );
        assert!(
            (OptimizationAction::RebalanceWeights.estimated_benefit() - 0.6).abs() < f32::EPSILON
        );
        assert!(
            (OptimizationAction::CompactStorage.estimated_benefit() - 0.4).abs() < f32::EPSILON
        );
        println!("[PASS] test_action_estimated_benefit");
    }

    #[test]
    fn test_action_description() {
        assert!(!OptimizationAction::ReindexMemories.description().is_empty());
        assert!(!OptimizationAction::RecomputeAlignments
            .description()
            .is_empty());
        assert!(!OptimizationAction::PruneStaleData.description().is_empty());
        assert!(!OptimizationAction::RebalanceWeights
            .description()
            .is_empty());
        assert!(!OptimizationAction::CompactStorage.description().is_empty());
        println!("[PASS] test_action_description");
    }

    #[test]
    fn test_action_recommended_for_high_drift() {
        let trigger = OptimizationTrigger::HighDrift { severity: 0.25 };
        let actions = OptimizationAction::recommended_for_trigger(&trigger);
        assert!(actions.contains(&OptimizationAction::RecomputeAlignments));
        assert!(actions.contains(&OptimizationAction::RebalanceWeights));
        assert!(actions.contains(&OptimizationAction::ReindexMemories));
        println!("[PASS] test_action_recommended_for_high_drift");
    }

    #[test]
    fn test_action_recommended_for_memory_pressure() {
        let trigger = OptimizationTrigger::MemoryPressure {
            usage_percent: 96.0,
        };
        let actions = OptimizationAction::recommended_for_trigger(&trigger);
        assert!(actions.contains(&OptimizationAction::PruneStaleData));
        assert!(actions.contains(&OptimizationAction::CompactStorage));
        println!("[PASS] test_action_recommended_for_memory_pressure");
    }

    #[test]
    fn test_action_recommended_for_low_performance_retrieval() {
        let trigger = OptimizationTrigger::LowPerformance {
            metric: "retrieval_latency".to_string(),
            value: 0.3,
        };
        let actions = OptimizationAction::recommended_for_trigger(&trigger);
        assert!(actions.contains(&OptimizationAction::ReindexMemories));
        println!("[PASS] test_action_recommended_for_low_performance_retrieval");
    }

    #[test]
    fn test_action_serialization() {
        let actions = [
            OptimizationAction::ReindexMemories,
            OptimizationAction::RecomputeAlignments,
            OptimizationAction::PruneStaleData,
            OptimizationAction::RebalanceWeights,
            OptimizationAction::CompactStorage,
        ];

        for action in actions {
            let json = serde_json::to_string(&action).expect("serialize failed");
            let deserialized: OptimizationAction =
                serde_json::from_str(&json).expect("deserialize failed");
            assert_eq!(deserialized, action);
        }
        println!("[PASS] test_action_serialization");
    }

    #[test]
    fn test_action_equality_and_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(OptimizationAction::ReindexMemories);
        set.insert(OptimizationAction::RecomputeAlignments);
        set.insert(OptimizationAction::ReindexMemories); // Duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&OptimizationAction::ReindexMemories));
        assert!(set.contains(&OptimizationAction::RecomputeAlignments));
        assert!(!set.contains(&OptimizationAction::PruneStaleData));
        println!("[PASS] test_action_equality_and_hash");
    }
}
