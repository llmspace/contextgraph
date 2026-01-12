//! Extraction tests for BootstrapService

use crate::autonomous::services::bootstrap_service::*;

#[test]
fn test_extract_goal_candidates_basic() {
    let service = BootstrapService::new();
    let content = "The goal of this project is to build a knowledge graph system.";
    let source = "README.md";

    let candidates = service.extract_goal_candidates(content, source);

    assert!(
        !candidates.is_empty(),
        "Should extract at least one candidate"
    );
    assert!(candidates[0].keyword_count >= 1);
    assert_eq!(candidates[0].source, "README.md");

    println!("[PASS] test_extract_goal_candidates_basic");
}

#[test]
fn test_extract_goal_candidates_multiple_sentences() {
    let service = BootstrapService::new();
    let content = r#"
        This is the introduction to the project.
        The mission is to create an intelligent memory system.
        Our purpose is to enable machines to learn and remember.
        This concludes the overview section.
    "#;

    let candidates = service.extract_goal_candidates(content, "docs/overview.md");

    assert!(
        candidates.len() >= 2,
        "Should find multiple goal candidates"
    );

    let has_mission = candidates.iter().any(|c| c.text.contains("mission"));
    let has_purpose = candidates.iter().any(|c| c.text.contains("purpose"));

    assert!(
        has_mission || has_purpose,
        "Should find mission or purpose statements"
    );

    println!("[PASS] test_extract_goal_candidates_multiple_sentences");
}

#[test]
fn test_extract_goal_candidates_empty_content() {
    let service = BootstrapService::new();
    let candidates = service.extract_goal_candidates("", "empty.md");

    assert!(
        candidates.is_empty(),
        "Empty content should yield no candidates"
    );

    println!("[PASS] test_extract_goal_candidates_empty_content");
}

#[test]
fn test_extract_goal_candidates_no_keywords() {
    let service = BootstrapService::new();
    let content = "This is just a random sentence with no relevant keywords whatsoever.";
    let candidates = service.extract_goal_candidates(content, "random.md");

    assert!(
        candidates.is_empty(),
        "Should not extract candidates without keywords"
    );

    println!("[PASS] test_extract_goal_candidates_no_keywords");
}

#[test]
#[should_panic(expected = "Source identifier cannot be empty")]
fn test_extract_goal_candidates_fails_empty_source() {
    let service = BootstrapService::new();
    service.extract_goal_candidates("Some content", "");
}

#[test]
fn test_extract_goal_candidates_position_tracking() {
    let service = BootstrapService::new();
    let content = r#"The goal is to start here.
        Middle content without keywords.
        The objective is to finish here."#;

    let candidates = service.extract_goal_candidates(content, "test.md");

    // First candidate should have low position, last should have high position
    if candidates.len() >= 2 {
        let first = candidates
            .iter()
            .min_by(|a, b| a.position.partial_cmp(&b.position).unwrap());
        let last = candidates
            .iter()
            .max_by(|a, b| a.position.partial_cmp(&b.position).unwrap());

        assert!(first.unwrap().position < last.unwrap().position);
    }

    println!("[PASS] test_extract_goal_candidates_position_tracking");
}

#[test]
fn test_goal_candidate_clone() {
    let candidate = GoalCandidate {
        text: "The mission is to test cloning.".to_string(),
        source: "test.md".to_string(),
        position: 0.5,
        density: 0.1,
        keyword_count: 1,
        line_number: 10,
    };

    let cloned = candidate.clone();

    assert_eq!(cloned.text, candidate.text);
    assert_eq!(cloned.source, candidate.source);
    assert!((cloned.position - candidate.position).abs() < f32::EPSILON);
    assert!((cloned.density - candidate.density).abs() < f32::EPSILON);
    assert_eq!(cloned.keyword_count, candidate.keyword_count);
    assert_eq!(cloned.line_number, candidate.line_number);

    println!("[PASS] test_goal_candidate_clone");
}

#[test]
fn test_goal_candidate_debug() {
    let candidate = GoalCandidate {
        text: "Debug test".to_string(),
        source: "test.md".to_string(),
        position: 0.0,
        density: 0.0,
        keyword_count: 0,
        line_number: 1,
    };

    let debug_str = format!("{:?}", candidate);

    assert!(debug_str.contains("GoalCandidate"));
    assert!(debug_str.contains("Debug test"));

    println!("[PASS] test_goal_candidate_debug");
}
