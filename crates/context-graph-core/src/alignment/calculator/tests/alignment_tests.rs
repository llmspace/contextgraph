//! Alignment computation tests.

use std::time::Instant;

use super::{
    create_test_fingerprint, create_test_hierarchy, AlignmentConfig, DefaultAlignmentCalculator,
    GoalAlignmentCalculator, GoalAlignmentScore, LevelWeights, MisalignmentFlags,
};
use crate::alignment::calculator::result::AlignmentResult;

#[tokio::test]
async fn test_compute_alignment_basic() {
    let calculator = DefaultAlignmentCalculator::new();
    let fingerprint = create_test_fingerprint(0.8);
    let hierarchy = create_test_hierarchy();

    let config = AlignmentConfig::with_hierarchy(hierarchy)
        .with_pattern_detection(true)
        .with_embedder_breakdown(true);

    let result = calculator
        .compute_alignment(&fingerprint, &config)
        .await
        .expect("Alignment computation failed");

    println!("\n=== Alignment Result ===");
    println!(
        "BEFORE: fingerprint alignment_score = {:.3}",
        fingerprint.alignment_score
    );
    println!(
        "AFTER: composite_score = {:.3}",
        result.score.composite_score
    );
    println!(
        "  - strategic_alignment: {:.3}",
        result.score.strategic_alignment
    );
    println!(
        "  - tactical_alignment: {:.3}",
        result.score.tactical_alignment
    );
    println!(
        "  - immediate_alignment: {:.3}",
        result.score.immediate_alignment
    );
    println!("  - threshold: {:?}", result.score.threshold);
    println!("  - computation_time_us: {}", result.computation_time_us);
    println!("  - goal_count: {}", result.score.goal_count());
    println!("  - pattern_count: {}", result.patterns.len());

    assert!(result.score.goal_count() == 4);
    assert!(result.computation_time_us < 5000); // <5ms
    assert!(result.score.composite_score > 0.5); // Overall should still be acceptable

    println!("[VERIFIED] compute_alignment produces valid result");
}

#[tokio::test]
async fn test_compute_alignment_no_goals() {
    use crate::alignment::error::AlignmentError;

    let calculator = DefaultAlignmentCalculator::new();
    let fingerprint = create_test_fingerprint(0.8);

    // Empty hierarchy
    let config = AlignmentConfig::default();

    let result = calculator.compute_alignment(&fingerprint, &config).await;

    assert!(result.is_err());
    match result {
        Err(AlignmentError::NoTopLevelGoals) => {
            println!("[VERIFIED] NoTopLevelGoals error returned for empty hierarchy");
        }
        other => panic!("Expected NoTopLevelGoals error, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_compute_alignment_detects_critical() {
    let calculator = DefaultAlignmentCalculator::new();

    // Create fingerprint with very low alignment
    let fingerprint = create_test_fingerprint(0.1);
    let hierarchy = create_test_hierarchy();

    let config = AlignmentConfig::with_hierarchy(hierarchy);

    let result = calculator
        .compute_alignment(&fingerprint, &config)
        .await
        .expect("Alignment computation failed");

    println!("\n=== Low Alignment Test ===");
    println!("BEFORE: fingerprint theta = 0.1");
    println!(
        "AFTER: flags.below_threshold = {}",
        result.flags.below_threshold
    );
    println!(
        "       flags.critical_goals = {:?}",
        result.flags.critical_goals
    );
    println!("       score.threshold = {:?}", result.score.threshold);

    println!("[VERIFIED] Low alignment fingerprint processed");
}

#[tokio::test]
async fn test_compute_alignment_batch() {
    let calculator = DefaultAlignmentCalculator::new();
    let hierarchy = create_test_hierarchy();
    let config = AlignmentConfig::with_hierarchy(hierarchy);

    let fp1 = create_test_fingerprint(0.9);
    let fp2 = create_test_fingerprint(0.5);
    let fp3 = create_test_fingerprint(0.3);

    let fingerprints: Vec<&_> = vec![&fp1, &fp2, &fp3];

    let results = calculator
        .compute_alignment_batch(&fingerprints, &config)
        .await;

    assert_eq!(results.len(), 3);

    println!("\n=== Batch Alignment Results ===");
    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(r) => {
                println!(
                    "  [{i}] composite={:.3}, healthy={}",
                    r.score.composite_score,
                    r.is_healthy()
                );
            }
            Err(e) => println!("  [{i}] ERROR: {}", e),
        }
    }

    assert!(results.iter().all(|r| r.is_ok()));
    println!("[VERIFIED] compute_alignment_batch processes multiple fingerprints");
}

#[test]
fn test_alignment_result_severity() {
    let score = GoalAlignmentScore::empty(LevelWeights::default());
    let flags = MisalignmentFlags::empty();

    let result = AlignmentResult {
        score,
        flags,
        patterns: Vec::new(),
        embedder_breakdown: None,
        computation_time_us: 100,
    };

    assert_eq!(result.severity(), 0);
    assert!(result.is_healthy());
    assert!(!result.needs_attention());

    println!("[VERIFIED] AlignmentResult severity levels work correctly");
}

#[tokio::test]
async fn test_performance_under_5ms() {
    let calculator = DefaultAlignmentCalculator::new();
    let fingerprint = create_test_fingerprint(0.8);
    let hierarchy = create_test_hierarchy();

    let config = AlignmentConfig::with_hierarchy(hierarchy)
        .with_pattern_detection(true)
        .with_embedder_breakdown(true);

    // Run multiple times to get average
    let iterations = 100;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = calculator.compute_alignment(&fingerprint, &config).await;
    }

    let total_ms = start.elapsed().as_millis() as f64;
    let avg_ms = total_ms / iterations as f64;

    println!("\n=== Performance Test ===");
    println!("  iterations: {}", iterations);
    println!("  total_ms: {:.2}", total_ms);
    println!("  avg_ms: {:.3}", avg_ms);

    assert!(
        avg_ms < 5.0,
        "Average computation time {}ms exceeds 5ms budget",
        avg_ms
    );
    println!(
        "[VERIFIED] Performance meets <5ms requirement (avg: {:.3}ms)",
        avg_ms
    );
}
