//! Integration tests and serialization tests for drift corrector.

#[cfg(test)]
mod tests {
    use super::super::common::*;

    // ==========================================================================
    // Auto-Correct Integration Tests
    // ==========================================================================

    #[test]
    fn test_auto_correct() {
        let mut corrector = DriftCorrector::new();
        let config = DriftConfig::default();
        let mut state = DriftState {
            severity: DriftSeverity::Moderate,
            trend: DriftTrend::Declining,
            rolling_mean: 0.70,
            ..Default::default()
        };

        let result = corrector.auto_correct(&mut state, &config);

        // Should have applied threshold adjustment
        if let CorrectionStrategy::ThresholdAdjustment { delta } = result.strategy_applied {
            assert!((delta - 0.02).abs() < f32::EPSILON);
        } else {
            panic!("Expected ThresholdAdjustment strategy");
        }
        println!("[PASS] test_auto_correct");
    }

    // ==========================================================================
    // Serialization Tests - CorrectionStrategy
    // ==========================================================================

    #[test]
    fn test_strategy_serialization() {
        let strategies = vec![
            CorrectionStrategy::NoAction,
            CorrectionStrategy::ThresholdAdjustment { delta: 0.05 },
            CorrectionStrategy::WeightRebalance {
                adjustments: vec![(0, 0.1)],
            },
            CorrectionStrategy::GoalReinforcement {
                emphasis_factor: 1.5,
            },
            CorrectionStrategy::EmergencyIntervention {
                reason: "Test".to_string(),
            },
        ];

        for strategy in strategies {
            let json = serde_json::to_string(&strategy).expect("serialize");
            let deserialized: CorrectionStrategy =
                serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deserialized, strategy);
        }
        println!("[PASS] test_strategy_serialization");
    }

    // ==========================================================================
    // Serialization Tests - CorrectionResult
    // ==========================================================================

    #[test]
    fn test_result_serialization() {
        let result = CorrectionResult::new(
            CorrectionStrategy::ThresholdAdjustment { delta: 0.05 },
            0.70,
            0.75,
            true,
        );

        let json = serde_json::to_string(&result).expect("serialize");
        let deserialized: CorrectionResult = serde_json::from_str(&json).expect("deserialize");

        assert!((deserialized.alignment_before - 0.70).abs() < f32::EPSILON);
        assert!((deserialized.alignment_after - 0.75).abs() < f32::EPSILON);
        assert!(deserialized.success);
        println!("[PASS] test_result_serialization");
    }

    // ==========================================================================
    // Serialization Tests - DriftCorrectorConfig
    // ==========================================================================

    #[test]
    fn test_config_serialization() {
        let config = DriftCorrectorConfig::default();
        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: DriftCorrectorConfig = serde_json::from_str(&json).expect("deserialize");

        assert!((deserialized.moderate_threshold_delta - 0.02).abs() < f32::EPSILON);
        assert!((deserialized.severe_threshold_delta - 0.05).abs() < f32::EPSILON);
        println!("[PASS] test_config_serialization");
    }
}
