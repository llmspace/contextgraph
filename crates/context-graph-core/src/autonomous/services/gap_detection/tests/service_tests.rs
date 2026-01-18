//! Tests for GapDetectionService struct.

use crate::autonomous::bootstrap::GoalId;
use crate::autonomous::evolution::GoalLevel;
use crate::autonomous::services::gap_detection::{
    GapDetectionConfig, GapDetectionService, GapType, GoalWithMetrics,
};

use super::helpers::{create_test_goal, create_test_metrics};

#[test]
fn test_service_new() {
    let service = GapDetectionService::new();
    assert!((service.config.coverage_threshold - 0.4).abs() < f32::EPSILON);
    println!("[PASS] test_service_new");
}

#[test]
fn test_service_with_config() {
    let config = GapDetectionConfig {
        coverage_threshold: 0.6,
        ..Default::default()
    };
    let service = GapDetectionService::with_config(config);
    assert!((service.config.coverage_threshold - 0.6).abs() < f32::EPSILON);
    println!("[PASS] test_service_with_config");
}

#[test]
fn test_service_default() {
    let service = GapDetectionService::default();
    assert!((service.config.coverage_threshold - 0.4).abs() < f32::EPSILON);
    println!("[PASS] test_service_default");
}

#[test]
fn test_analyze_coverage_empty_goals() {
    let service = GapDetectionService::new();
    let report = service.analyze_coverage(&[]);

    assert!(!report.has_gaps());
    assert!((report.coverage_score - 0.0).abs() < f32::EPSILON);
    assert_eq!(report.goals_analyzed, 0);
    assert_eq!(report.domains_detected, 0);
    assert!(!report.recommendations.is_empty());
    println!("[PASS] test_analyze_coverage_empty_goals");
}

#[test]
fn test_analyze_coverage_healthy_goals() {
    let service = GapDetectionService::new();
    let goals = vec![
        create_test_goal(GoalLevel::Strategic, vec!["core"], 80, 40, 0.9),
        create_test_goal(GoalLevel::Strategic, vec!["security"], 60, 30, 0.85),
        create_test_goal(GoalLevel::Tactical, vec!["performance"], 50, 25, 0.8),
    ];

    let report = service.analyze_coverage(&goals);

    assert_eq!(report.goals_analyzed, 3);
    assert_eq!(report.domains_detected, 3);
    assert!(report.coverage_score > 0.5);
    println!("[PASS] test_analyze_coverage_healthy_goals");
}

#[test]
fn test_detect_domain_gaps_no_gaps() {
    let config = GapDetectionConfig {
        min_goals_per_domain: 1,
        activity_threshold: 0.2,
        ..Default::default()
    };
    let service = GapDetectionService::with_config(config);

    let goals = vec![
        create_test_goal(GoalLevel::Strategic, vec!["security"], 50, 25, 0.8),
        create_test_goal(GoalLevel::Tactical, vec!["performance"], 50, 25, 0.8),
    ];

    let gaps = service.detect_domain_gaps(&goals);
    assert!(gaps.is_empty());
    println!("[PASS] test_detect_domain_gaps_no_gaps");
}

#[test]
fn test_detect_domain_gaps_inactive_domain() {
    let config = GapDetectionConfig {
        min_goals_per_domain: 1,
        activity_threshold: 0.5, // High threshold
        ..Default::default()
    };
    let service = GapDetectionService::with_config(config);

    // This goal has low activity (score < 0.5)
    let goals = vec![create_test_goal(
        GoalLevel::Strategic,
        vec!["security"],
        10,
        5,
        0.3,
    )];

    let gaps = service.detect_domain_gaps(&goals);
    assert_eq!(gaps.len(), 1);
    match &gaps[0] {
        GapType::UncoveredDomain { domain } => assert_eq!(domain, "security"),
        _ => panic!("Expected UncoveredDomain gap"),
    }
    println!("[PASS] test_detect_domain_gaps_inactive_domain");
}

#[test]
fn test_detect_weak_coverage() {
    let service = GapDetectionService::new();

    let goals = vec![
        create_test_goal(GoalLevel::Strategic, vec!["security"], 80, 40, 0.9), // Strong
        create_test_goal(GoalLevel::Tactical, vec!["performance"], 5, 2, 0.1), // Weak
    ];

    let gaps = service.detect_weak_coverage(&goals);
    assert_eq!(gaps.len(), 1);
    match &gaps[0] {
        GapType::WeakCoverage { coverage, .. } => {
            assert!(*coverage < 0.4);
        }
        _ => panic!("Expected WeakCoverage gap"),
    }
    println!("[PASS] test_detect_weak_coverage");
}

#[test]
fn test_detect_weak_coverage_all_strong() {
    let service = GapDetectionService::new();

    let goals = vec![
        create_test_goal(GoalLevel::Strategic, vec!["security"], 80, 40, 0.9),
        create_test_goal(GoalLevel::Tactical, vec!["performance"], 70, 35, 0.85),
    ];

    let gaps = service.detect_weak_coverage(&goals);
    assert!(gaps.is_empty());
    println!("[PASS] test_detect_weak_coverage_all_strong");
}

#[test]
fn test_detect_missing_links_none_expected() {
    let service = GapDetectionService::new();

    // Goals in different domains - no link expected
    let goals = vec![
        create_test_goal(GoalLevel::Strategic, vec!["security"], 50, 25, 0.8),
        create_test_goal(GoalLevel::Tactical, vec!["performance"], 50, 25, 0.8),
    ];

    let gaps = service.detect_missing_links(&goals);
    assert!(gaps.is_empty());
    println!("[PASS] test_detect_missing_links_none_expected");
}

#[test]
fn test_detect_missing_links_shared_domain() {
    let service = GapDetectionService::new();

    // Goals share "security" domain but are not linked
    let goals = vec![
        create_test_goal(GoalLevel::Strategic, vec!["security", "auth"], 50, 25, 0.8),
        create_test_goal(GoalLevel::Tactical, vec!["security", "crypto"], 50, 25, 0.8),
    ];

    let gaps = service.detect_missing_links(&goals);
    assert_eq!(gaps.len(), 1);
    match &gaps[0] {
        GapType::MissingLink { .. } => {}
        _ => panic!("Expected MissingLink gap"),
    }
    println!("[PASS] test_detect_missing_links_shared_domain");
}

#[test]
fn test_detect_missing_links_already_linked() {
    let service = GapDetectionService::new();

    let parent_id = GoalId::new();
    let child_id = GoalId::new();

    let parent_metrics = create_test_metrics(parent_id.clone(), 50, 25, 0.8);
    let child_metrics = create_test_metrics(child_id.clone(), 40, 20, 0.75);

    let goals = vec![
        GoalWithMetrics {
            goal_id: parent_id.clone(),
            level: GoalLevel::Strategic,
            description: "Parent".into(),
            parent_id: None,
            child_ids: vec![child_id.clone()],
            domains: vec!["security".into()],
            metrics: parent_metrics,
        },
        GoalWithMetrics {
            goal_id: child_id.clone(),
            level: GoalLevel::Tactical,
            description: "Child".into(),
            parent_id: Some(parent_id.clone()),
            child_ids: vec![],
            domains: vec!["security".into()],
            metrics: child_metrics,
        },
    ];

    let gaps = service.detect_missing_links(&goals);
    assert!(gaps.is_empty()); // No gap since they are already linked
    println!("[PASS] test_detect_missing_links_already_linked");
}

#[test]
fn test_compute_coverage_score_empty() {
    let service = GapDetectionService::new();
    let score = service.compute_coverage_score(&[]);
    assert!((score - 0.0).abs() < f32::EPSILON);
    println!("[PASS] test_compute_coverage_score_empty");
}

#[test]
fn test_compute_coverage_score_single_goal() {
    let service = GapDetectionService::new();

    let goals = vec![create_test_goal(
        GoalLevel::Strategic,
        vec!["core"],
        100,
        50,
        1.0,
    )];

    let score = service.compute_coverage_score(&goals);
    assert!(score > 0.8); // High score for active strategic goal
    println!("[PASS] test_compute_coverage_score_single_goal");
}

#[test]
fn test_compute_coverage_score_mixed_levels() {
    let service = GapDetectionService::new();

    let goals = vec![
        create_test_goal(GoalLevel::Strategic, vec!["core"], 100, 50, 1.0),
        create_test_goal(GoalLevel::Strategic, vec!["security"], 80, 40, 0.9),
        create_test_goal(GoalLevel::Tactical, vec!["performance"], 60, 30, 0.8),
        create_test_goal(GoalLevel::Operational, vec!["logging"], 40, 20, 0.7),
    ];

    let score = service.compute_coverage_score(&goals);
    assert!(score > 0.6);
    assert!(score <= 1.0);
    println!("[PASS] test_compute_coverage_score_mixed_levels");
}

#[test]
fn test_compute_coverage_score_all_inactive() {
    let service = GapDetectionService::new();

    let goals = vec![
        create_test_goal(GoalLevel::Strategic, vec!["security"], 0, 0, 0.3),
        create_test_goal(GoalLevel::Tactical, vec!["performance"], 0, 0, 0.2),
    ];

    let score = service.compute_coverage_score(&goals);
    assert!(score < 0.3); // Low score for inactive goals
    println!("[PASS] test_compute_coverage_score_all_inactive");
}

#[test]
fn test_full_analysis_integration() {
    let config = GapDetectionConfig {
        coverage_threshold: 0.5,
        activity_threshold: 0.3,
        min_goals_per_domain: 1,
        detect_missing_links: true,
        ..Default::default()
    };
    let service = GapDetectionService::with_config(config);

    // Create a mix of goals with various issues
    let goals = vec![
        create_test_goal(GoalLevel::Strategic, vec!["core"], 80, 40, 0.9),
        create_test_goal(GoalLevel::Strategic, vec!["security"], 60, 30, 0.85),
        create_test_goal(GoalLevel::Tactical, vec!["security"], 50, 25, 0.8), // Shares domain
        create_test_goal(GoalLevel::Operational, vec!["logging"], 5, 2, 0.1), // Weak
    ];

    let report = service.analyze_coverage(&goals);

    assert_eq!(report.goals_analyzed, 4);
    assert!(report.domains_detected >= 3);
    assert!(report.coverage_score > 0.0);
    assert!(report.coverage_score <= 1.0);

    // Should detect weak coverage for the logging goal
    let weak_gaps: Vec<_> = report
        .gaps
        .iter()
        .filter(|g| matches!(g, GapType::WeakCoverage { .. }))
        .collect();
    assert!(!weak_gaps.is_empty());

    // Should have recommendations
    assert!(!report.recommendations.is_empty());

    println!("[PASS] test_full_analysis_integration");
}

#[test]
fn test_disable_missing_links_detection() {
    let config = GapDetectionConfig {
        detect_missing_links: false,
        ..Default::default()
    };
    let service = GapDetectionService::with_config(config);

    // Goals that would trigger missing link detection if enabled
    let goals = vec![
        create_test_goal(GoalLevel::Strategic, vec!["security"], 50, 25, 0.8),
        create_test_goal(GoalLevel::Tactical, vec!["security"], 50, 25, 0.8),
    ];

    let report = service.analyze_coverage(&goals);

    let link_gaps: Vec<_> = report
        .gaps
        .iter()
        .filter(|g| matches!(g, GapType::MissingLink { .. }))
        .collect();
    assert!(link_gaps.is_empty());
    println!("[PASS] test_disable_missing_links_detection");
}
