//! Configuration for the EventOptimizer service.

use serde::{Deserialize, Serialize};

/// Configuration for the EventOptimizer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventOptimizerConfig {
    /// Maximum number of events to retain in history
    pub max_history_size: usize,
    /// High drift severity threshold
    pub high_drift_threshold: f32,
    /// Low performance threshold
    pub low_performance_threshold: f32,
    /// Memory pressure threshold (percentage)
    pub memory_pressure_threshold: f32,
    /// Enable automatic optimization on events
    pub auto_optimize: bool,
}

impl Default for EventOptimizerConfig {
    fn default() -> Self {
        Self {
            max_history_size: 1000,
            high_drift_threshold: 0.10,
            low_performance_threshold: 0.50,
            memory_pressure_threshold: 85.0,
            auto_optimize: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = EventOptimizerConfig::default();
        assert_eq!(config.max_history_size, 1000);
        assert!((config.high_drift_threshold - 0.10).abs() < f32::EPSILON);
        assert!((config.low_performance_threshold - 0.50).abs() < f32::EPSILON);
        assert!((config.memory_pressure_threshold - 85.0).abs() < f32::EPSILON);
        assert!(config.auto_optimize);
        println!("[PASS] test_config_default");
    }

    #[test]
    fn test_config_serialization() {
        let config = EventOptimizerConfig::default();
        let json = serde_json::to_string(&config).expect("serialize failed");
        let deserialized: EventOptimizerConfig =
            serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(deserialized.max_history_size, config.max_history_size);
        assert!(
            (deserialized.high_drift_threshold - config.high_drift_threshold).abs() < f32::EPSILON
        );
        println!("[PASS] test_config_serialization");
    }
}
