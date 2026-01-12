//! System metrics and improvement calculations.

use serde::{Deserialize, Serialize};

/// System metrics snapshot
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// Average alignment score
    pub avg_alignment: f32,
    /// Average retrieval latency in milliseconds
    pub avg_latency_ms: f32,
    /// Memory usage percentage
    pub memory_usage_percent: f32,
    /// Total memory count
    pub memory_count: u64,
    /// Index health score (0.0 to 1.0)
    pub index_health: f32,
}

impl SystemMetrics {
    /// Create a new metrics snapshot
    pub fn new(
        avg_alignment: f32,
        avg_latency_ms: f32,
        memory_usage_percent: f32,
        memory_count: u64,
        index_health: f32,
    ) -> Self {
        Self {
            avg_alignment,
            avg_latency_ms,
            memory_usage_percent,
            memory_count,
            index_health,
        }
    }

    /// Calculate improvement from before to after
    pub fn calculate_improvement(&self, after: &Self) -> f32 {
        let alignment_improvement = after.avg_alignment - self.avg_alignment;
        let latency_improvement = if self.avg_latency_ms > 0.0 {
            (self.avg_latency_ms - after.avg_latency_ms) / self.avg_latency_ms
        } else {
            0.0
        };
        let memory_improvement = self.memory_usage_percent - after.memory_usage_percent;
        let index_improvement = after.index_health - self.index_health;

        // Weighted average of improvements
        (alignment_improvement * 0.4
            + latency_improvement * 0.3
            + memory_improvement * 0.01
            + index_improvement * 0.29)
            .clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_metrics_default() {
        let metrics = SystemMetrics::default();
        assert!((metrics.avg_alignment - 0.0).abs() < f32::EPSILON);
        assert!((metrics.avg_latency_ms - 0.0).abs() < f32::EPSILON);
        assert!((metrics.memory_usage_percent - 0.0).abs() < f32::EPSILON);
        assert_eq!(metrics.memory_count, 0);
        assert!((metrics.index_health - 0.0).abs() < f32::EPSILON);
        println!("[PASS] test_system_metrics_default");
    }

    #[test]
    fn test_system_metrics_new() {
        let metrics = SystemMetrics::new(0.75, 50.0, 60.0, 1000, 0.9);
        assert!((metrics.avg_alignment - 0.75).abs() < f32::EPSILON);
        assert!((metrics.avg_latency_ms - 50.0).abs() < f32::EPSILON);
        assert!((metrics.memory_usage_percent - 60.0).abs() < f32::EPSILON);
        assert_eq!(metrics.memory_count, 1000);
        assert!((metrics.index_health - 0.9).abs() < f32::EPSILON);
        println!("[PASS] test_system_metrics_new");
    }

    #[test]
    fn test_system_metrics_calculate_improvement() {
        let before = SystemMetrics::new(0.70, 100.0, 80.0, 1000, 0.8);
        let after = SystemMetrics::new(0.80, 50.0, 70.0, 1000, 0.9);

        let improvement = before.calculate_improvement(&after);
        assert!(improvement > 0.0); // Should be positive (improvement)
        println!("[PASS] test_system_metrics_calculate_improvement");
    }

    #[test]
    fn test_system_metrics_calculate_degradation() {
        let before = SystemMetrics::new(0.80, 50.0, 60.0, 1000, 0.9);
        let after = SystemMetrics::new(0.70, 100.0, 80.0, 1000, 0.7);

        let improvement = before.calculate_improvement(&after);
        assert!(improvement < 0.0); // Should be negative (degradation)
        println!("[PASS] test_system_metrics_calculate_degradation");
    }

    #[test]
    fn test_system_metrics_serialization() {
        let metrics = SystemMetrics::new(0.75, 50.0, 60.0, 1000, 0.9);
        let json = serde_json::to_string(&metrics).expect("serialize failed");
        let deserialized: SystemMetrics = serde_json::from_str(&json).expect("deserialize failed");
        assert!((deserialized.avg_alignment - metrics.avg_alignment).abs() < f32::EPSILON);
        assert!((deserialized.avg_latency_ms - metrics.avg_latency_ms).abs() < f32::EPSILON);
        println!("[PASS] test_system_metrics_serialization");
    }
}
