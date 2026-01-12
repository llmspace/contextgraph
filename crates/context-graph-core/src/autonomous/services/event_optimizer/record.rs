//! Optimization event record for history tracking.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::action::OptimizationAction;
use super::metrics::SystemMetrics;

/// Record of an optimization event
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizationEventRecord {
    /// Type of event that triggered optimization
    pub event_type: String,
    /// When the optimization occurred
    pub timestamp: DateTime<Utc>,
    /// System metrics before optimization
    pub metrics_before: SystemMetrics,
    /// System metrics after optimization
    pub metrics_after: SystemMetrics,
    /// Whether the optimization was successful
    pub success: bool,
    /// Actions that were executed
    pub actions_executed: Vec<OptimizationAction>,
    /// Duration of the optimization in milliseconds
    pub duration_ms: u64,
    /// Error message if optimization failed
    pub error: Option<String>,
}

impl OptimizationEventRecord {
    /// Calculate the improvement achieved by this optimization
    pub fn improvement(&self) -> f32 {
        if self.success {
            self.metrics_before
                .calculate_improvement(&self.metrics_after)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_event_record_improvement() {
        let record = OptimizationEventRecord {
            event_type: "test".to_string(),
            timestamp: Utc::now(),
            metrics_before: SystemMetrics::new(0.70, 100.0, 80.0, 1000, 0.8),
            metrics_after: SystemMetrics::new(0.80, 50.0, 70.0, 1000, 0.9),
            success: true,
            actions_executed: vec![OptimizationAction::RecomputeAlignments],
            duration_ms: 500,
            error: None,
        };

        assert!(record.improvement() > 0.0);
        println!("[PASS] test_optimization_event_record_improvement");
    }

    #[test]
    fn test_optimization_event_record_improvement_failed() {
        let record = OptimizationEventRecord {
            event_type: "test".to_string(),
            timestamp: Utc::now(),
            metrics_before: SystemMetrics::new(0.70, 100.0, 80.0, 1000, 0.8),
            metrics_after: SystemMetrics::new(0.80, 50.0, 70.0, 1000, 0.9),
            success: false, // Failed
            actions_executed: vec![],
            duration_ms: 500,
            error: Some("Failed".to_string()),
        };

        assert!((record.improvement() - 0.0).abs() < f32::EPSILON);
        println!("[PASS] test_optimization_event_record_improvement_failed");
    }

    #[test]
    fn test_optimization_event_record_serialization() {
        let record = OptimizationEventRecord {
            event_type: "high_drift".to_string(),
            timestamp: Utc::now(),
            metrics_before: SystemMetrics::new(0.70, 100.0, 80.0, 1000, 0.8),
            metrics_after: SystemMetrics::new(0.80, 50.0, 70.0, 1000, 0.9),
            success: true,
            actions_executed: vec![
                OptimizationAction::RecomputeAlignments,
                OptimizationAction::RebalanceWeights,
            ],
            duration_ms: 500,
            error: None,
        };

        let json = serde_json::to_string(&record).expect("serialize failed");
        let deserialized: OptimizationEventRecord =
            serde_json::from_str(&json).expect("deserialize failed");

        assert_eq!(deserialized.event_type, record.event_type);
        assert_eq!(deserialized.success, record.success);
        assert_eq!(
            deserialized.actions_executed.len(),
            record.actions_executed.len()
        );
        println!("[PASS] test_optimization_event_record_serialization");
    }
}
