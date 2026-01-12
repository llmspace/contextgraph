//! Scoring tests for BootstrapService

use crate::autonomous::services::bootstrap_service::*;

#[test]
fn test_score_candidate_basic() {
    let service = BootstrapService::new();
    let candidate = GoalCandidate {
        text: "The goal of this system is to provide intelligent memory management.".to_string(),
        source: "README.md".to_string(),
        position: 0.1,
        density: 0.15,
        keyword_count: 3,
        line_number: 1,
    };

    let score = service.score_candidate(&candidate);

    assert!(score > 0.0, "Score should be positive");
    assert!(score <= 1.0, "Score should not exceed 1.0");

    println!("[PASS] test_score_candidate_basic");
}

#[test]
fn test_score_candidate_position_weighting() {
    let service = BootstrapService::new();

    let start_candidate = GoalCandidate {
        text: "The goal is to build a system architecture framework.".to_string(),
        source: "test.md".to_string(),
        position: 0.0,
        density: 0.1,
        keyword_count: 2,
        line_number: 1,
    };

    let middle_candidate = GoalCandidate {
        text: "The goal is to build a system architecture framework.".to_string(),
        source: "test.md".to_string(),
        position: 0.5,
        density: 0.1,
        keyword_count: 2,
        line_number: 50,
    };

    let start_score = service.score_candidate(&start_candidate);
    let middle_score = service.score_candidate(&middle_candidate);

    assert!(
        start_score > middle_score,
        "Start position should score higher than middle"
    );

    println!("[PASS] test_score_candidate_position_weighting");
}

#[test]
fn test_score_candidate_keyword_impact() {
    let service = BootstrapService::new();

    let low_keywords = GoalCandidate {
        text: "The goal is here.".to_string(),
        source: "test.md".to_string(),
        position: 0.5,
        density: 0.05,
        keyword_count: 1,
        line_number: 1,
    };

    let high_keywords = GoalCandidate {
        text: "The goal and mission of this system is to achieve the objective.".to_string(),
        source: "test.md".to_string(),
        position: 0.5,
        density: 0.2,
        keyword_count: 4,
        line_number: 1,
    };

    let low_score = service.score_candidate(&low_keywords);
    let high_score = service.score_candidate(&high_keywords);

    assert!(
        high_score > low_score,
        "More keywords should yield higher score"
    );

    println!("[PASS] test_score_candidate_keyword_impact");
}

#[test]
fn test_score_candidate_purpose_starter_bonus() {
    let service = BootstrapService::new();

    let with_starter = GoalCandidate {
        text: "The goal of this project is to build something amazing.".to_string(),
        source: "test.md".to_string(),
        position: 0.3,
        density: 0.1,
        keyword_count: 2,
        line_number: 1,
    };

    let without_starter = GoalCandidate {
        text: "Something goal-related that builds amazing things.".to_string(),
        source: "test.md".to_string(),
        position: 0.3,
        density: 0.1,
        keyword_count: 2,
        line_number: 1,
    };

    let with_score = service.score_candidate(&with_starter);
    let without_score = service.score_candidate(&without_starter);

    assert!(
        with_score > without_score,
        "Purpose starter should boost score"
    );

    println!("[PASS] test_score_candidate_purpose_starter_bonus");
}
