//! Goal candidate extraction logic
//!
//! Provides methods for extracting goal candidates from document content,
//! including sentence splitting, keyword counting, and density calculation.

use super::constants::GOAL_KEYWORDS;
use super::types::GoalCandidate;

/// Extract goal candidates from document content
///
/// # Arguments
///
/// * `content` - The document text content
/// * `source` - The source file path or identifier
///
/// # Returns
///
/// Vector of goal candidates found in the document
///
/// # Panics
///
/// Panics if `source` is empty
pub fn extract_goal_candidates(content: &str, source: &str) -> Vec<GoalCandidate> {
    assert!(!source.is_empty(), "Source identifier cannot be empty");

    let mut candidates = Vec::new();

    if content.is_empty() {
        return candidates;
    }

    // Process content by sentences/paragraphs
    let sentences = split_into_sentences(content);
    let total_sentences = sentences.len();

    for (idx, sentence) in sentences.iter().enumerate() {
        let trimmed = sentence.trim();
        if trimmed.is_empty() || trimmed.len() < 20 {
            continue;
        }

        // Calculate position (0.0 to 1.0)
        let position = if total_sentences > 1 {
            idx as f32 / (total_sentences - 1) as f32
        } else {
            0.5
        };

        // Count goal-related keywords
        let keyword_count = count_keywords(trimmed);
        if keyword_count == 0 {
            continue;
        }

        // Calculate semantic density
        let density = calculate_density(trimmed, keyword_count);

        // Find line number
        let line_number = find_line_number(content, trimmed);

        candidates.push(GoalCandidate {
            text: trimmed.to_string(),
            source: source.to_string(),
            position,
            density,
            keyword_count,
            line_number,
        });
    }

    candidates
}

/// Split content into sentences
pub fn split_into_sentences(content: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in content.chars() {
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' || ch == '\n' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current = String::new();
        }
    }

    // Don't forget the last segment
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

/// Count goal-related keywords in text
pub fn count_keywords(text: &str) -> usize {
    let lower = text.to_lowercase();
    GOAL_KEYWORDS
        .iter()
        .filter(|&&kw| lower.contains(kw))
        .count()
}

/// Calculate semantic density (keyword concentration)
pub fn calculate_density(text: &str, keyword_count: usize) -> f32 {
    let word_count = text.split_whitespace().count();
    if word_count == 0 {
        return 0.0;
    }
    (keyword_count as f32 / word_count as f32).min(1.0)
}

/// Find line number for a piece of text
pub fn find_line_number(content: &str, text: &str) -> usize {
    let search = text
        .split_whitespace()
        .take(5)
        .collect::<Vec<_>>()
        .join(" ");
    for (idx, line) in content.lines().enumerate() {
        if line.contains(&search) || line.contains(text) {
            return idx + 1;
        }
    }
    1
}
