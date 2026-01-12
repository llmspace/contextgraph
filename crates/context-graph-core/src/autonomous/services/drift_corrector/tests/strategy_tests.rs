//! Tests for CorrectionStrategy enum and strategy selection logic.

#[cfg(test)]
mod tests {
    use super::super::common::*;

    // ==========================================================================
    // CorrectionStrategy Enum Variant Tests
    // ==========================================================================

    #[test]
    fn test_correction_strategy_no_action() {
        let strategy = CorrectionStrategy::NoAction;
        assert_eq!(strategy, CorrectionStrategy::NoAction);
        println!("[PASS] test_correction_strategy_no_action");
    }

    #[test]
    fn test_correction_strategy_threshold_adjustment() {
        let strategy = CorrectionStrategy::ThresholdAdjustment { delta: 0.05 };
        if let CorrectionStrategy::ThresholdAdjustment { delta } = strategy {
            assert!((delta - 0.05).abs() < f32::EPSILON);
        } else {
            panic!("Expected ThresholdAdjustment");
        }
        println!("[PASS] test_correction_strategy_threshold_adjustment");
    }

    #[test]
    fn test_correction_strategy_weight_rebalance() {
        let adjustments = vec![(0, 0.1), (1, -0.05), (2, 0.15)];
        let strategy = CorrectionStrategy::WeightRebalance {
            adjustments: adjustments.clone(),
        };
        if let CorrectionStrategy::WeightRebalance { adjustments: adj } = strategy {
            assert_eq!(adj.len(), 3);
            assert_eq!(adj[0], (0, 0.1));
        } else {
            panic!("Expected WeightRebalance");
        }
        println!("[PASS] test_correction_strategy_weight_rebalance");
    }

    #[test]
    fn test_correction_strategy_goal_reinforcement() {
        let strategy = CorrectionStrategy::GoalReinforcement {
            emphasis_factor: 1.5,
        };
        if let CorrectionStrategy::GoalReinforcement { emphasis_factor } = strategy {
            assert!((emphasis_factor - 1.5).abs() < f32::EPSILON);
        } else {
            panic!("Expected GoalReinforcement");
        }
        println!("[PASS] test_correction_strategy_goal_reinforcement");
    }

    #[test]
    fn test_correction_strategy_emergency_intervention() {
        let strategy = CorrectionStrategy::EmergencyIntervention {
            reason: "Critical drift detected".to_string(),
        };
        if let CorrectionStrategy::EmergencyIntervention { reason } = strategy {
            assert_eq!(reason, "Critical drift detected");
        } else {
            panic!("Expected EmergencyIntervention");
        }
        println!("[PASS] test_correction_strategy_emergency_intervention");
    }

    // ==========================================================================
    // Strategy Selection Tests - No Drift / Mild Severity
    // ==========================================================================

    #[test]
    fn test_select_strategy_no_drift() {
        let corrector = DriftCorrector::new();
        let state = DriftState::default(); // severity: None

        let strategy = corrector.select_strategy(&state);
        assert_eq!(strategy, CorrectionStrategy::NoAction);
        println!("[PASS] test_select_strategy_no_drift");
    }

    #[test]
    fn test_select_strategy_mild_declining() {
        let corrector = DriftCorrector::new();
        let state = DriftState {
            severity: DriftSeverity::Mild,
            trend: DriftTrend::Declining,
            ..Default::default()
        };

        let strategy = corrector.select_strategy(&state);
        if let CorrectionStrategy::GoalReinforcement { emphasis_factor } = strategy {
            assert!((emphasis_factor - 1.1).abs() < f32::EPSILON);
        } else {
            panic!("Expected GoalReinforcement for mild declining drift");
        }
        println!("[PASS] test_select_strategy_mild_declining");
    }

    #[test]
    fn test_select_strategy_mild_stable() {
        let corrector = DriftCorrector::new();
        let state = DriftState {
            severity: DriftSeverity::Mild,
            trend: DriftTrend::Stable,
            ..Default::default()
        };

        let strategy = corrector.select_strategy(&state);
        assert_eq!(strategy, CorrectionStrategy::NoAction);
        println!("[PASS] test_select_strategy_mild_stable");
    }

    // ==========================================================================
    // Strategy Selection Tests - Moderate Severity
    // ==========================================================================

    #[test]
    fn test_select_strategy_moderate_declining() {
        let corrector = DriftCorrector::new();
        let state = DriftState {
            severity: DriftSeverity::Moderate,
            trend: DriftTrend::Declining,
            ..Default::default()
        };

        let strategy = corrector.select_strategy(&state);
        if let CorrectionStrategy::ThresholdAdjustment { delta } = strategy {
            assert!((delta - 0.02).abs() < f32::EPSILON);
        } else {
            panic!("Expected ThresholdAdjustment for moderate declining drift");
        }
        println!("[PASS] test_select_strategy_moderate_declining");
    }

    #[test]
    fn test_select_strategy_moderate_stable() {
        let corrector = DriftCorrector::new();
        let state = DriftState {
            severity: DriftSeverity::Moderate,
            trend: DriftTrend::Stable,
            ..Default::default()
        };

        let strategy = corrector.select_strategy(&state);
        if let CorrectionStrategy::GoalReinforcement { emphasis_factor } = strategy {
            assert!((emphasis_factor - 1.2).abs() < f32::EPSILON);
        } else {
            panic!("Expected GoalReinforcement for moderate stable drift");
        }
        println!("[PASS] test_select_strategy_moderate_stable");
    }

    #[test]
    fn test_select_strategy_moderate_improving() {
        let corrector = DriftCorrector::new();
        let state = DriftState {
            severity: DriftSeverity::Moderate,
            trend: DriftTrend::Improving,
            ..Default::default()
        };

        let strategy = corrector.select_strategy(&state);
        assert_eq!(strategy, CorrectionStrategy::NoAction);
        println!("[PASS] test_select_strategy_moderate_improving");
    }

    // ==========================================================================
    // Strategy Selection Tests - Severe Severity
    // ==========================================================================

    #[test]
    fn test_select_strategy_severe_declining() {
        let corrector = DriftCorrector::new();
        let state = DriftState {
            severity: DriftSeverity::Severe,
            trend: DriftTrend::Declining,
            drift: 0.15,
            ..Default::default()
        };

        let strategy = corrector.select_strategy(&state);
        if let CorrectionStrategy::EmergencyIntervention { reason } = strategy {
            assert!(reason.contains("Severe drift"));
            assert!(reason.contains("declining"));
        } else {
            panic!("Expected EmergencyIntervention for severe declining drift");
        }
        println!("[PASS] test_select_strategy_severe_declining");
    }

    #[test]
    fn test_select_strategy_severe_stable() {
        let corrector = DriftCorrector::new();
        let state = DriftState {
            severity: DriftSeverity::Severe,
            trend: DriftTrend::Stable,
            ..Default::default()
        };

        let strategy = corrector.select_strategy(&state);
        if let CorrectionStrategy::ThresholdAdjustment { delta } = strategy {
            assert!((delta - 0.05).abs() < f32::EPSILON);
        } else {
            panic!("Expected ThresholdAdjustment for severe stable drift");
        }
        println!("[PASS] test_select_strategy_severe_stable");
    }

    #[test]
    fn test_select_strategy_severe_improving() {
        let corrector = DriftCorrector::new();
        let state = DriftState {
            severity: DriftSeverity::Severe,
            trend: DriftTrend::Improving,
            ..Default::default()
        };

        let strategy = corrector.select_strategy(&state);
        if let CorrectionStrategy::GoalReinforcement { emphasis_factor } = strategy {
            assert!((emphasis_factor - 1.5).abs() < f32::EPSILON);
        } else {
            panic!("Expected GoalReinforcement for severe improving drift");
        }
        println!("[PASS] test_select_strategy_severe_improving");
    }
}
