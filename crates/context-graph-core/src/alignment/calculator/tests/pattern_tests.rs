//! Pattern detection tests.

use uuid::Uuid;

use super::{
    create_test_hierarchy, AlignmentConfig, DefaultAlignmentCalculator, GoalAlignmentScore,
    GoalScore, LevelWeights, MisalignmentFlags,
};
use crate::alignment::calculator::GoalAlignmentCalculator;
use crate::alignment::pattern::PatternType;
use crate::purpose::GoalLevel;

#[test]
fn test_detect_patterns_optimal() {
    let calculator = DefaultAlignmentCalculator::new();
    let hierarchy = create_test_hierarchy();

    // Create optimal score using UUIDs
    let scores = vec![
        GoalScore::new(Uuid::new_v4(), GoalLevel::Strategic, 0.85, 0.4),
        GoalScore::new(Uuid::new_v4(), GoalLevel::Strategic, 0.80, 0.3),
        GoalScore::new(Uuid::new_v4(), GoalLevel::Tactical, 0.78, 0.2),
        GoalScore::new(Uuid::new_v4(), GoalLevel::Immediate, 0.76, 0.1),
    ];
    let score = GoalAlignmentScore::compute(scores, LevelWeights::default());
    let flags = MisalignmentFlags::empty();
    let config = AlignmentConfig::with_hierarchy(hierarchy);

    let patterns = calculator.detect_patterns(&score, &flags, &config);

    println!("\n=== Detected Patterns ===");
    for p in &patterns {
        println!(
            "  - {:?}: {} (severity {})",
            p.pattern_type, p.description, p.severity
        );
    }

    // Should detect OptimalAlignment and HierarchicalCoherence
    let has_optimal = patterns
        .iter()
        .any(|p| p.pattern_type == PatternType::OptimalAlignment);
    let has_coherence = patterns
        .iter()
        .any(|p| p.pattern_type == PatternType::HierarchicalCoherence);

    assert!(
        has_optimal || has_coherence,
        "Should detect positive patterns for optimal alignment"
    );
    println!("[VERIFIED] detect_patterns identifies positive patterns");
}

// TASK-P0-001: Renamed from test_detect_patterns_north_star_drift
#[test]
fn test_detect_patterns_strategic_drift() {
    let calculator = DefaultAlignmentCalculator::new();
    let hierarchy = create_test_hierarchy();

    // Create score with low Strategic alignment using UUIDs
    // Both scores below WARNING threshold (0.55) so average is also below
    // Average = (0.40 + 0.50) / 2 = 0.45 < 0.55 WARNING threshold
    let scores = vec![
        GoalScore::new(Uuid::new_v4(), GoalLevel::Strategic, 0.40, 0.5),
        GoalScore::new(Uuid::new_v4(), GoalLevel::Strategic, 0.50, 0.5),
    ];
    let score = GoalAlignmentScore::compute(scores, LevelWeights::default());
    let flags = MisalignmentFlags::empty();
    let config = AlignmentConfig::with_hierarchy(hierarchy);

    let patterns = calculator.detect_patterns(&score, &flags, &config);

    println!("\n=== Strategic Drift Detection ===");
    println!(
        "strategic_alignment = {:.2} (below WARNING 0.55)",
        score.strategic_alignment
    );
    for p in &patterns {
        println!("pattern = {:?}, severity = {}", p.pattern_type, p.severity);
    }

    let has_drift = patterns
        .iter()
        .any(|p| p.pattern_type == PatternType::StrategicDrift);
    assert!(
        has_drift,
        "Should detect StrategicDrift pattern when strategic_alignment ({:.2}) < WARNING (0.55)",
        score.strategic_alignment
    );
    println!("[VERIFIED] detect_patterns identifies StrategicDrift");
}
