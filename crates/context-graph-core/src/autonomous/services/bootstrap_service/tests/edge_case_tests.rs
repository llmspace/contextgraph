//! Edge case tests for BootstrapService

use crate::autonomous::services::bootstrap_service::*;

#[test]
fn test_extract_handles_unicode() {
    let service = BootstrapService::new();
    let content =
        "The goal is to support unicode: \u{1F600} \u{4E2D}\u{6587} \u{0410}\u{0411}\u{0412}";

    let candidates = service.extract_goal_candidates(content, "unicode.md");

    assert!(!candidates.is_empty(), "Should handle unicode content");

    println!("[PASS] test_extract_handles_unicode");
}

#[test]
fn test_extract_handles_very_long_sentences() {
    let service = BootstrapService::new();
    let long_text = format!("The goal is to {} build something.", "really ".repeat(100));

    let candidates = service.extract_goal_candidates(&long_text, "long.md");

    // Should still extract but may score lower due to length
    if !candidates.is_empty() {
        let score = service.score_candidate(&candidates[0]);
        assert!(score <= 1.0);
    }

    println!("[PASS] test_extract_handles_very_long_sentences");
}

#[test]
fn test_extract_handles_special_characters() {
    let service = BootstrapService::new();
    let content = r#"The goal is to handle "special" chars: <>&'\ properly."#;

    let candidates = service.extract_goal_candidates(content, "special.md");

    assert!(!candidates.is_empty(), "Should handle special characters");

    println!("[PASS] test_extract_handles_special_characters");
}
