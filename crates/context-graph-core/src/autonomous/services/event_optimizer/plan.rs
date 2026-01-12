//! Optimization plan creation and management.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::action::OptimizationAction;
use super::trigger::OptimizationTrigger;

/// Plan for an optimization operation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizationPlan {
    /// The trigger that initiated this plan
    pub trigger: OptimizationTrigger,
    /// Actions to be executed
    pub actions: Vec<OptimizationAction>,
    /// Priority level (0-10, higher is more urgent)
    pub priority: u8,
    /// Estimated impact (0.0 to 1.0)
    pub estimated_impact: f32,
    /// When the plan was created
    pub created_at: DateTime<Utc>,
}

impl OptimizationPlan {
    /// Create a new optimization plan
    pub fn new(trigger: OptimizationTrigger, actions: Vec<OptimizationAction>) -> Self {
        let priority = trigger.base_priority();
        let estimated_impact = Self::calculate_estimated_impact(&actions);
        Self {
            trigger,
            actions,
            priority,
            estimated_impact,
            created_at: Utc::now(),
        }
    }

    /// Create a plan with custom priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority.min(10);
        self
    }

    /// Calculate estimated impact from actions
    fn calculate_estimated_impact(actions: &[OptimizationAction]) -> f32 {
        if actions.is_empty() {
            return 0.0;
        }

        let total_benefit: f32 = actions.iter().map(|a| a.estimated_benefit()).sum();
        let total_cost: f32 = actions.iter().map(|a| a.estimated_cost()).sum();

        if total_cost > 0.0 {
            (total_benefit / actions.len() as f32).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Check if this plan is empty
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Get total estimated cost
    pub fn total_cost(&self) -> f32 {
        self.actions.iter().map(|a| a.estimated_cost()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_plan_new() {
        let trigger = OptimizationTrigger::HighDrift { severity: 0.15 };
        let actions = vec![
            OptimizationAction::RecomputeAlignments,
            OptimizationAction::RebalanceWeights,
        ];
        let plan = OptimizationPlan::new(trigger.clone(), actions.clone());

        assert_eq!(plan.trigger, trigger);
        assert_eq!(plan.actions.len(), 2);
        assert_eq!(plan.priority, 8);
        assert!(plan.estimated_impact > 0.0);
        println!("[PASS] test_optimization_plan_new");
    }

    #[test]
    fn test_optimization_plan_with_priority() {
        let trigger = OptimizationTrigger::ScheduledMaintenance;
        let plan = OptimizationPlan::new(trigger, vec![OptimizationAction::PruneStaleData])
            .with_priority(7);

        assert_eq!(plan.priority, 7);
        println!("[PASS] test_optimization_plan_with_priority");
    }

    #[test]
    fn test_optimization_plan_priority_clamped() {
        let trigger = OptimizationTrigger::ScheduledMaintenance;
        let plan = OptimizationPlan::new(trigger, vec![OptimizationAction::PruneStaleData])
            .with_priority(15); // Should be clamped to 10

        assert_eq!(plan.priority, 10);
        println!("[PASS] test_optimization_plan_priority_clamped");
    }

    #[test]
    fn test_optimization_plan_is_empty() {
        let trigger = OptimizationTrigger::ScheduledMaintenance;
        let empty_plan = OptimizationPlan::new(trigger.clone(), vec![]);
        assert!(empty_plan.is_empty());

        let non_empty_plan =
            OptimizationPlan::new(trigger, vec![OptimizationAction::PruneStaleData]);
        assert!(!non_empty_plan.is_empty());
        println!("[PASS] test_optimization_plan_is_empty");
    }

    #[test]
    fn test_optimization_plan_total_cost() {
        let trigger = OptimizationTrigger::ScheduledMaintenance;
        let plan = OptimizationPlan::new(
            trigger,
            vec![
                OptimizationAction::PruneStaleData,
                OptimizationAction::CompactStorage,
            ],
        );

        let expected_cost = 0.4 + 0.7; // PruneStaleData + CompactStorage
        assert!((plan.total_cost() - expected_cost).abs() < f32::EPSILON);
        println!("[PASS] test_optimization_plan_total_cost");
    }

    #[test]
    fn test_optimization_plan_serialization() {
        let trigger = OptimizationTrigger::HighDrift { severity: 0.15 };
        let plan = OptimizationPlan::new(
            trigger,
            vec![
                OptimizationAction::RecomputeAlignments,
                OptimizationAction::RebalanceWeights,
            ],
        );

        let json = serde_json::to_string(&plan).expect("serialize failed");
        let deserialized: OptimizationPlan =
            serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(deserialized.trigger, plan.trigger);
        assert_eq!(deserialized.actions.len(), plan.actions.len());
        println!("[PASS] test_optimization_plan_serialization");
    }
}
