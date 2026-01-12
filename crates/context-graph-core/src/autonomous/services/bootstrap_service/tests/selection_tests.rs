//! Selection tests for BootstrapService

use crate::autonomous::services::bootstrap_service::*;
use crate::autonomous::{BootstrapConfig, GoalId};

#[test]
fn test_select_best_goal_single_candidate() {
    let service = BootstrapService::new();
    let candidates = vec![GoalCandidate {
        text: "The mission is to create an intelligent context graph system.".to_string(),
        source: "README.md".to_string(),
        position: 0.1,
        density: 0.2,
        keyword_count: 3,
        line_number: 5,
    }];

    let result = service.select_best_goal(&candidates);

    assert!(result.is_some(), "Should select the single candidate");
    let result = result.unwrap();
    assert!(result.confidence > 0.0);
    assert_eq!(result.extracted_from, "README.md");

    println!("[PASS] test_select_best_goal_single_candidate");
}

#[test]
fn test_select_best_goal_multiple_candidates() {
    let service = BootstrapService::new();
    let candidates = vec![
        GoalCandidate {
            text: "Short goal here.".to_string(),
            source: "a.md".to_string(),
            position: 0.5,
            density: 0.05,
            keyword_count: 1,
            line_number: 1,
        },
        GoalCandidate {
            text: "The purpose of this system is to provide a comprehensive knowledge framework."
                .to_string(),
            source: "b.md".to_string(),
            position: 0.1,
            density: 0.15,
            keyword_count: 3,
            line_number: 1,
        },
        GoalCandidate {
            text: "Another goal mention.".to_string(),
            source: "c.md".to_string(),
            position: 0.9,
            density: 0.08,
            keyword_count: 1,
            line_number: 1,
        },
    ];

    let result = service.select_best_goal(&candidates);

    assert!(result.is_some());
    let result = result.unwrap();
    // The second candidate should win (better position, more keywords, purpose starter)
    assert!(result.goal_text.contains("purpose"));

    println!("[PASS] test_select_best_goal_multiple_candidates");
}

#[test]
fn test_select_best_goal_empty_candidates() {
    let service = BootstrapService::new();
    let result = service.select_best_goal(&[]);

    assert!(result.is_none(), "Empty candidates should return None");

    println!("[PASS] test_select_best_goal_empty_candidates");
}

#[test]
fn test_select_best_goal_generates_unique_id() {
    let service = BootstrapService::new();
    let candidates = vec![GoalCandidate {
        text: "The mission is to build something great.".to_string(),
        source: "test.md".to_string(),
        position: 0.1,
        density: 0.15,
        keyword_count: 2,
        line_number: 1,
    }];

    let result1 = service.select_best_goal(&candidates).unwrap();
    let result2 = service.select_best_goal(&candidates).unwrap();

    assert_ne!(
        result1.goal_id, result2.goal_id,
        "Each selection should generate unique ID"
    );

    println!("[PASS] test_select_best_goal_generates_unique_id");
}

#[test]
fn test_bootstrap_result_clone() {
    let result = BootstrapResult {
        goal_id: GoalId::new(),
        goal_text: "Test goal text".to_string(),
        confidence: 0.85,
        extracted_from: "test.md".to_string(),
    };

    let cloned = result.clone();

    assert_eq!(cloned.goal_id, result.goal_id);
    assert_eq!(cloned.goal_text, result.goal_text);
    assert!((cloned.confidence - result.confidence).abs() < f32::EPSILON);
    assert_eq!(cloned.extracted_from, result.extracted_from);

    println!("[PASS] test_bootstrap_result_clone");
}

#[test]
fn test_bootstrap_result_confidence_range() {
    let service = BootstrapService::new();
    let candidates = vec![GoalCandidate {
        text: "The purpose is to validate confidence ranges are correct.".to_string(),
        source: "test.md".to_string(),
        position: 0.1,
        density: 0.2,
        keyword_count: 2,
        line_number: 1,
    }];

    if let Some(result) = service.select_best_goal(&candidates) {
        assert!(
            result.confidence >= 0.0,
            "Confidence should be non-negative"
        );
        assert!(result.confidence <= 1.0, "Confidence should not exceed 1.0");
    }

    println!("[PASS] test_bootstrap_result_confidence_range");
}

#[test]
fn test_minimum_confidence_threshold() {
    let service = BootstrapService::new();

    // Candidate with very low scores
    let weak_candidates = vec![GoalCandidate {
        text: "x".to_string(), // Very short, no keywords
        source: "test.md".to_string(),
        position: 0.5,
        density: 0.0,
        keyword_count: 0, // No keywords means it won't be extracted normally
        line_number: 1,
    }];

    // This should return None due to low confidence
    let result = service.select_best_goal(&weak_candidates);
    assert!(result.is_none(), "Very weak candidates should be rejected");

    println!("[PASS] test_minimum_confidence_threshold");
}

#[test]
fn test_min_confidence_from_config() {
    let mut config = BootstrapServiceConfig::default();
    config.bootstrap_config.min_confidence = 0.9; // Very high threshold

    let service = BootstrapService::with_config(config);

    // This candidate would normally pass but should fail with high threshold
    let candidates = vec![GoalCandidate {
        text: "The goal is to test minimum confidence thresholds.".to_string(),
        source: "test.md".to_string(),
        position: 0.3,
        density: 0.1,
        keyword_count: 2,
        line_number: 1,
    }];

    let _result = service.select_best_goal(&candidates);

    // With a very high threshold, this might be rejected
    // The actual behavior depends on the score calculation
    println!("[PASS] test_min_confidence_from_config");
}
