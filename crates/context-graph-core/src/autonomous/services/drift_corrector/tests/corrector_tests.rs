//! Tests for DriftCorrector core functionality and adjustment methods.

#[cfg(test)]
mod tests {
    use super::super::common::*;

    // ==========================================================================
    // DriftCorrector Construction Tests
    // ==========================================================================

    #[test]
    fn test_drift_corrector_new() {
        let corrector = DriftCorrector::new();
        assert!((corrector.current_threshold_adjustment() - 0.0).abs() < f32::EPSILON);
        assert!(corrector.current_weight_adjustments().is_empty());
        assert!((corrector.current_goal_emphasis() - 1.0).abs() < f32::EPSILON);
        println!("[PASS] test_drift_corrector_new");
    }

    #[test]
    fn test_drift_corrector_with_config() {
        let config = DriftCorrectorConfig {
            moderate_threshold_delta: 0.03,
            severe_threshold_delta: 0.07,
            ..Default::default()
        };
        let corrector = DriftCorrector::with_config(config);
        assert!((corrector.config.moderate_threshold_delta - 0.03).abs() < f32::EPSILON);
        assert!((corrector.config.severe_threshold_delta - 0.07).abs() < f32::EPSILON);
        println!("[PASS] test_drift_corrector_with_config");
    }

    #[test]
    fn test_default_impl() {
        let corrector = DriftCorrector::default();
        assert!((corrector.current_threshold_adjustment() - 0.0).abs() < f32::EPSILON);
        assert!((corrector.current_goal_emphasis() - 1.0).abs() < f32::EPSILON);
        println!("[PASS] test_default_impl");
    }

    // ==========================================================================
    // Apply Correction Tests
    // ==========================================================================

    #[test]
    fn test_apply_correction_no_action() {
        let mut corrector = DriftCorrector::new();
        let mut state = DriftState {
            rolling_mean: 0.75,
            ..Default::default()
        };

        let strategy = CorrectionStrategy::NoAction;
        let result = corrector.apply_correction(&mut state, &strategy);

        assert!(result.success);
        assert!((result.alignment_before - 0.75).abs() < f32::EPSILON);
        assert!((result.alignment_after - 0.75).abs() < f32::EPSILON);
        println!("[PASS] test_apply_correction_no_action");
    }

    #[test]
    fn test_apply_correction_threshold_adjustment() {
        let mut corrector = DriftCorrector::new();
        let mut state = DriftState {
            rolling_mean: 0.70,
            ..Default::default()
        };

        let strategy = CorrectionStrategy::ThresholdAdjustment { delta: 0.05 };
        let result = corrector.apply_correction(&mut state, &strategy);

        assert!((result.alignment_before - 0.70).abs() < f32::EPSILON);
        assert!(result.alignment_after > result.alignment_before);
        assert!((corrector.current_threshold_adjustment() - 0.05).abs() < f32::EPSILON);
        println!("[PASS] test_apply_correction_threshold_adjustment");
    }

    #[test]
    fn test_apply_correction_weight_rebalance() {
        let mut corrector = DriftCorrector::new();
        let mut state = DriftState {
            rolling_mean: 0.70,
            ..Default::default()
        };

        let strategy = CorrectionStrategy::WeightRebalance {
            adjustments: vec![(0, 0.1), (1, 0.1)],
        };
        let result = corrector.apply_correction(&mut state, &strategy);

        assert!((result.alignment_before - 0.70).abs() < f32::EPSILON);
        assert!(result.alignment_after >= result.alignment_before);
        assert_eq!(corrector.current_weight_adjustments().len(), 2);
        println!("[PASS] test_apply_correction_weight_rebalance");
    }

    #[test]
    fn test_apply_correction_goal_reinforcement() {
        let mut corrector = DriftCorrector::new();
        let mut state = DriftState {
            rolling_mean: 0.70,
            ..Default::default()
        };

        let strategy = CorrectionStrategy::GoalReinforcement {
            emphasis_factor: 1.3,
        };
        let result = corrector.apply_correction(&mut state, &strategy);

        assert!((result.alignment_before - 0.70).abs() < f32::EPSILON);
        assert!(result.alignment_after > result.alignment_before);
        assert!((corrector.current_goal_emphasis() - 1.3).abs() < f32::EPSILON);
        println!("[PASS] test_apply_correction_goal_reinforcement");
    }

    #[test]
    fn test_apply_correction_emergency() {
        let mut corrector = DriftCorrector::new();
        let mut state = DriftState {
            rolling_mean: 0.60,
            ..Default::default()
        };

        let strategy = CorrectionStrategy::EmergencyIntervention {
            reason: "Test emergency".to_string(),
        };
        let result = corrector.apply_correction(&mut state, &strategy);

        assert!(!result.success); // Emergency requires human action
        assert!((result.alignment_before - result.alignment_after).abs() < f32::EPSILON);
        println!("[PASS] test_apply_correction_emergency");
    }

    // ==========================================================================
    // Threshold Adjustment Tests
    // ==========================================================================

    #[test]
    fn test_adjust_thresholds() {
        let mut corrector = DriftCorrector::new();

        corrector.adjust_thresholds(0.05);
        assert!((corrector.current_threshold_adjustment() - 0.05).abs() < f32::EPSILON);

        corrector.adjust_thresholds(0.10);
        assert!((corrector.current_threshold_adjustment() - 0.15).abs() < f32::EPSILON);

        // Test clamping to max
        corrector.adjust_thresholds(0.50);
        assert!((corrector.current_threshold_adjustment() - 0.2).abs() < f32::EPSILON);
        println!("[PASS] test_adjust_thresholds");
    }

    #[test]
    fn test_adjust_thresholds_negative() {
        let mut corrector = DriftCorrector::new();

        corrector.adjust_thresholds(-0.05);
        assert!((corrector.current_threshold_adjustment() - (-0.05)).abs() < f32::EPSILON);

        // Test clamping to min
        corrector.adjust_thresholds(-0.50);
        assert!((corrector.current_threshold_adjustment() - (-0.2)).abs() < f32::EPSILON);
        println!("[PASS] test_adjust_thresholds_negative");
    }

    // ==========================================================================
    // Weight Rebalance Tests
    // ==========================================================================

    #[test]
    fn test_rebalance_weights() {
        let mut corrector = DriftCorrector::new();

        corrector.rebalance_weights(&[(0, 0.1), (1, -0.05)]);
        let adjustments = corrector.current_weight_adjustments();
        assert_eq!(adjustments.len(), 2);
        assert_eq!(adjustments[0], (0, 0.1));
        assert_eq!(adjustments[1], (1, -0.05));
        println!("[PASS] test_rebalance_weights");
    }

    #[test]
    fn test_rebalance_weights_accumulation() {
        let mut corrector = DriftCorrector::new();

        corrector.rebalance_weights(&[(0, 0.1)]);
        corrector.rebalance_weights(&[(0, 0.05), (1, 0.1)]);

        let adjustments = corrector.current_weight_adjustments();
        assert_eq!(adjustments.len(), 2);
        // Index 0 should accumulate: 0.1 + 0.05 = 0.15
        assert!((adjustments[0].1 - 0.15).abs() < f32::EPSILON);
        assert!((adjustments[1].1 - 0.1).abs() < f32::EPSILON);
        println!("[PASS] test_rebalance_weights_accumulation");
    }

    #[test]
    fn test_rebalance_weights_clamping() {
        let mut corrector = DriftCorrector::new();

        // Adjustment exceeds max
        corrector.rebalance_weights(&[(0, 0.5)]);
        let adjustments = corrector.current_weight_adjustments();
        assert!((adjustments[0].1 - 0.2).abs() < f32::EPSILON); // Clamped to max
        println!("[PASS] test_rebalance_weights_clamping");
    }

    // ==========================================================================
    // Goal Reinforcement Tests
    // ==========================================================================

    #[test]
    fn test_reinforce_goal() {
        let mut corrector = DriftCorrector::new();
        assert!((corrector.current_goal_emphasis() - 1.0).abs() < f32::EPSILON);

        corrector.reinforce_goal(1.2);
        assert!((corrector.current_goal_emphasis() - 1.2).abs() < f32::EPSILON);

        corrector.reinforce_goal(1.5);
        // Use 1e-5 tolerance for accumulated floating point operations
        assert!((corrector.current_goal_emphasis() - 1.8).abs() < 1e-5);
        println!("[PASS] test_reinforce_goal");
    }

    #[test]
    fn test_reinforce_goal_clamping() {
        let mut corrector = DriftCorrector::new();

        // Test upper clamp
        corrector.reinforce_goal(3.0);
        assert!((corrector.current_goal_emphasis() - 2.0).abs() < f32::EPSILON);

        // Reset and test lower clamp
        corrector.reset();
        corrector.reinforce_goal(0.3);
        assert!((corrector.current_goal_emphasis() - 0.5).abs() < f32::EPSILON);
        println!("[PASS] test_reinforce_goal_clamping");
    }

    // ==========================================================================
    // Stats and Reset Tests
    // ==========================================================================

    #[test]
    fn test_correction_stats() {
        let mut corrector = DriftCorrector::new();
        let (applied, successful, rate) = corrector.correction_stats();
        assert_eq!(applied, 0);
        assert_eq!(successful, 0);
        assert!((rate - 0.0).abs() < f32::EPSILON);

        // Apply some corrections
        let mut state = DriftState {
            rolling_mean: 0.70,
            ..Default::default()
        };

        let strategy = CorrectionStrategy::ThresholdAdjustment { delta: 0.05 };
        corrector.apply_correction(&mut state, &strategy);
        corrector.apply_correction(&mut state, &strategy);

        let (applied, successful, rate) = corrector.correction_stats();
        assert_eq!(applied, 2);
        assert!(successful >= 1);
        assert!(rate > 0.0);
        println!("[PASS] test_correction_stats");
    }

    #[test]
    fn test_reset() {
        let mut corrector = DriftCorrector::new();

        // Apply various adjustments
        corrector.adjust_thresholds(0.1);
        corrector.rebalance_weights(&[(0, 0.1)]);
        corrector.reinforce_goal(1.5);

        let mut state = DriftState::default();
        let strategy = CorrectionStrategy::NoAction;
        corrector.apply_correction(&mut state, &strategy);

        // Reset
        corrector.reset();

        assert!((corrector.current_threshold_adjustment() - 0.0).abs() < f32::EPSILON);
        assert!(corrector.current_weight_adjustments().is_empty());
        assert!((corrector.current_goal_emphasis() - 1.0).abs() < f32::EPSILON);
        let (applied, _, _) = corrector.correction_stats();
        assert_eq!(applied, 0);
        println!("[PASS] test_reset");
    }

    // ==========================================================================
    // Edge Case Tests
    // ==========================================================================

    #[test]
    fn test_alignment_clamping() {
        let mut corrector = DriftCorrector::new();
        let mut state = DriftState {
            rolling_mean: 0.98,
            ..Default::default()
        };

        // Large adjustment that would push past 1.0
        let strategy = CorrectionStrategy::ThresholdAdjustment { delta: 0.10 };
        let result = corrector.apply_correction(&mut state, &strategy);

        assert!(result.alignment_after <= 1.0);
        println!("[PASS] test_alignment_clamping");
    }
}
