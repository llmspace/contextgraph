//! Optimization trigger types and priority handling.

use serde::{Deserialize, Serialize};

/// Trigger for optimization operations
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum OptimizationTrigger {
    /// High drift detected in alignment
    HighDrift { severity: f32 },
    /// Performance metric fell below threshold
    LowPerformance { metric: String, value: f32 },
    /// Memory usage exceeded threshold
    MemoryPressure { usage_percent: f32 },
    /// Scheduled maintenance window
    ScheduledMaintenance,
    /// User-initiated optimization
    UserTriggered { reason: String },
}

impl OptimizationTrigger {
    /// Get a descriptive name for the trigger type
    pub fn trigger_type_name(&self) -> &'static str {
        match self {
            Self::HighDrift { .. } => "high_drift",
            Self::LowPerformance { .. } => "low_performance",
            Self::MemoryPressure { .. } => "memory_pressure",
            Self::ScheduledMaintenance => "scheduled_maintenance",
            Self::UserTriggered { .. } => "user_triggered",
        }
    }

    /// Check if this trigger is critical and requires immediate action
    pub fn is_critical(&self) -> bool {
        match self {
            Self::HighDrift { severity } => *severity >= 0.20,
            Self::MemoryPressure { usage_percent } => *usage_percent >= 95.0,
            _ => false,
        }
    }

    /// Get the base priority for this trigger type
    pub fn base_priority(&self) -> u8 {
        match self {
            Self::HighDrift { severity } if *severity >= 0.20 => 10,
            Self::HighDrift { severity } if *severity >= 0.10 => 8,
            Self::HighDrift { .. } => 5,
            Self::MemoryPressure { usage_percent } if *usage_percent >= 95.0 => 10,
            Self::MemoryPressure { usage_percent } if *usage_percent >= 90.0 => 8,
            Self::MemoryPressure { .. } => 6,
            Self::LowPerformance { value, .. } if *value < 0.30 => 7,
            Self::LowPerformance { .. } => 5,
            Self::UserTriggered { .. } => 9,
            Self::ScheduledMaintenance => 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigger_high_drift() {
        let trigger = OptimizationTrigger::HighDrift { severity: 0.15 };
        assert_eq!(trigger.trigger_type_name(), "high_drift");
        assert!(!trigger.is_critical());
        assert_eq!(trigger.base_priority(), 8);
        println!("[PASS] test_trigger_high_drift");
    }

    #[test]
    fn test_trigger_critical_high_drift() {
        let trigger = OptimizationTrigger::HighDrift { severity: 0.25 };
        assert!(trigger.is_critical());
        assert_eq!(trigger.base_priority(), 10);
        println!("[PASS] test_trigger_critical_high_drift");
    }

    #[test]
    fn test_trigger_low_performance() {
        let trigger = OptimizationTrigger::LowPerformance {
            metric: "retrieval_latency".to_string(),
            value: 0.35,
        };
        assert_eq!(trigger.trigger_type_name(), "low_performance");
        assert!(!trigger.is_critical());
        assert_eq!(trigger.base_priority(), 5);
        println!("[PASS] test_trigger_low_performance");
    }

    #[test]
    fn test_trigger_memory_pressure() {
        let trigger = OptimizationTrigger::MemoryPressure {
            usage_percent: 90.0,
        };
        assert_eq!(trigger.trigger_type_name(), "memory_pressure");
        assert!(!trigger.is_critical());
        assert_eq!(trigger.base_priority(), 8);
        println!("[PASS] test_trigger_memory_pressure");
    }

    #[test]
    fn test_trigger_critical_memory_pressure() {
        let trigger = OptimizationTrigger::MemoryPressure {
            usage_percent: 96.0,
        };
        assert!(trigger.is_critical());
        assert_eq!(trigger.base_priority(), 10);
        println!("[PASS] test_trigger_critical_memory_pressure");
    }

    #[test]
    fn test_trigger_scheduled_maintenance() {
        let trigger = OptimizationTrigger::ScheduledMaintenance;
        assert_eq!(trigger.trigger_type_name(), "scheduled_maintenance");
        assert!(!trigger.is_critical());
        assert_eq!(trigger.base_priority(), 3);
        println!("[PASS] test_trigger_scheduled_maintenance");
    }

    #[test]
    fn test_trigger_user_triggered() {
        let trigger = OptimizationTrigger::UserTriggered {
            reason: "Manual cleanup".to_string(),
        };
        assert_eq!(trigger.trigger_type_name(), "user_triggered");
        assert!(!trigger.is_critical());
        assert_eq!(trigger.base_priority(), 9);
        println!("[PASS] test_trigger_user_triggered");
    }

    #[test]
    fn test_trigger_serialization() {
        let triggers = [
            OptimizationTrigger::HighDrift { severity: 0.15 },
            OptimizationTrigger::LowPerformance {
                metric: "test".to_string(),
                value: 0.4,
            },
            OptimizationTrigger::MemoryPressure {
                usage_percent: 88.0,
            },
            OptimizationTrigger::ScheduledMaintenance,
            OptimizationTrigger::UserTriggered {
                reason: "test".to_string(),
            },
        ];

        for trigger in triggers {
            let json = serde_json::to_string(&trigger).expect("serialize failed");
            let deserialized: OptimizationTrigger =
                serde_json::from_str(&json).expect("deserialize failed");
            assert_eq!(deserialized, trigger);
        }
        println!("[PASS] test_trigger_serialization");
    }

    #[test]
    fn test_trigger_equality() {
        let t1 = OptimizationTrigger::HighDrift { severity: 0.15 };
        let t2 = OptimizationTrigger::HighDrift { severity: 0.15 };
        let t3 = OptimizationTrigger::HighDrift { severity: 0.20 };

        assert_eq!(t1, t2);
        assert_ne!(t1, t3);
        assert_ne!(t1, OptimizationTrigger::ScheduledMaintenance);
        println!("[PASS] test_trigger_equality");
    }
}
