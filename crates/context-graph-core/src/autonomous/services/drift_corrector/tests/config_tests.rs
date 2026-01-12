//! Tests for DriftCorrectorConfig and CorrectionResult.

#[cfg(test)]
mod tests {
    use super::super::common::*;

    // ==========================================================================
    // CorrectionResult Tests
    // ==========================================================================

    #[test]
    fn test_correction_result_new() {
        let result = CorrectionResult::new(CorrectionStrategy::NoAction, 0.70, 0.75, true);
        assert_eq!(result.strategy_applied, CorrectionStrategy::NoAction);
        assert!((result.alignment_before - 0.70).abs() < f32::EPSILON);
        assert!((result.alignment_after - 0.75).abs() < f32::EPSILON);
        assert!(result.success);
        println!("[PASS] test_correction_result_new");
    }

    #[test]
    fn test_correction_result_improvement() {
        let result = CorrectionResult::new(CorrectionStrategy::NoAction, 0.70, 0.75, true);
        let improvement = result.improvement();
        assert!((improvement - 0.05).abs() < f32::EPSILON);

        let negative = CorrectionResult::new(CorrectionStrategy::NoAction, 0.75, 0.70, false);
        assert!((negative.improvement() - (-0.05)).abs() < f32::EPSILON);
        println!("[PASS] test_correction_result_improvement");
    }

    // ==========================================================================
    // DriftCorrectorConfig Tests
    // ==========================================================================

    #[test]
    fn test_drift_corrector_config_default() {
        let config = DriftCorrectorConfig::default();
        assert!((config.moderate_threshold_delta - 0.02).abs() < f32::EPSILON);
        assert!((config.severe_threshold_delta - 0.05).abs() < f32::EPSILON);
        assert!((config.moderate_reinforcement - 1.2).abs() < f32::EPSILON);
        assert!((config.severe_reinforcement - 1.5).abs() < f32::EPSILON);
        assert!((config.min_improvement - 0.01).abs() < f32::EPSILON);
        assert!((config.max_weight_adjustment - 0.2).abs() < f32::EPSILON);
        println!("[PASS] test_drift_corrector_config_default");
    }

    // ==========================================================================
    // Evaluation Tests
    // ==========================================================================

    #[test]
    fn test_evaluate_correction_success() {
        let corrector = DriftCorrector::new();

        // Improvement >= min_improvement (0.01)
        assert!(corrector.evaluate_correction(0.70, 0.72)); // 0.02 improvement
        assert!(corrector.evaluate_correction(0.70, 0.711)); // 0.011 improvement
        println!("[PASS] test_evaluate_correction_success");
    }

    #[test]
    fn test_evaluate_correction_failure() {
        let corrector = DriftCorrector::new();

        // Improvement < min_improvement (0.01)
        assert!(!corrector.evaluate_correction(0.70, 0.705));
        assert!(!corrector.evaluate_correction(0.70, 0.70));
        assert!(!corrector.evaluate_correction(0.70, 0.69)); // Negative
        println!("[PASS] test_evaluate_correction_failure");
    }
}
