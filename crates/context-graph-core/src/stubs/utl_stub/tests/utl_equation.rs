//! TC-GHOST-001: UTL Equation Logic Tests (Updated for real computation).

use crate::stubs::utl_stub::StubUtlProcessor;
use crate::traits::UtlProcessor;
use crate::types::{EmotionalState, UtlContext};

#[tokio::test]
async fn test_utl_equation_formula_verification() {
    // TC-GHOST-001: UTL formula L = sigmoid(2.0 * ΔS * ΔC * wₑ * cos φ)
    let processor = StubUtlProcessor::new();
    let context = UtlContext {
        prior_entropy: 0.6,
        current_coherence: 0.7,
        ..Default::default()
    };

    let input = "test input for UTL verification";

    // Get individual components
    let surprise = processor.compute_surprise(input, &context).await.unwrap();
    let coherence_change = processor.compute_coherence_change(input, &context).await.unwrap();
    let emotional_weight = processor.compute_emotional_weight(input, &context).await.unwrap();
    let alignment = processor.compute_alignment(input, &context).await.unwrap();

    // Compute expected learning score using the sigmoid formula per constitution
    let raw = 2.0 * surprise * coherence_change * emotional_weight * alignment;
    let expected = 1.0 / (1.0 + (-raw).exp());

    // Get actual learning score
    let actual = processor.compute_learning_score(input, &context).await.unwrap();

    // Verify sigmoid formula is correctly implemented
    assert!(
        (actual - expected).abs() < 0.0001,
        "UTL formula mismatch: expected sigmoid(2.0 * {} * {} * {} * {}) = {}, got {}",
        surprise, coherence_change, emotional_weight, alignment, expected, actual
    );
}

#[tokio::test]
async fn test_utl_learning_score_in_valid_range() {
    let processor = StubUtlProcessor::new();
    let context = UtlContext::default();

    for input in ["a", "test", "Neural Network", "complex input string with many words"] {
        let score = processor.compute_learning_score(input, &context).await.unwrap();
        assert!((0.0..=1.0).contains(&score), "Learning score {} must be in [0.0, 1.0]", score);
    }
}

#[tokio::test]
async fn test_utl_components_deterministic() {
    let processor = StubUtlProcessor::new();
    let context = UtlContext::default();
    let input = "determinism test input";

    // Compute twice
    let surprise1 = processor.compute_surprise(input, &context).await.unwrap();
    let surprise2 = processor.compute_surprise(input, &context).await.unwrap();
    let coherence1 = processor.compute_coherence_change(input, &context).await.unwrap();
    let coherence2 = processor.compute_coherence_change(input, &context).await.unwrap();
    let weight1 = processor.compute_emotional_weight(input, &context).await.unwrap();
    let weight2 = processor.compute_emotional_weight(input, &context).await.unwrap();
    let align1 = processor.compute_alignment(input, &context).await.unwrap();
    let align2 = processor.compute_alignment(input, &context).await.unwrap();

    assert_eq!(surprise1, surprise2, "Surprise must be deterministic");
    assert_eq!(coherence1, coherence2, "Coherence change must be deterministic");
    assert_eq!(weight1, weight2, "Emotional weight must be deterministic");
    assert_eq!(align1, align2, "Alignment must be deterministic");
}

#[tokio::test]
async fn test_utl_surprise_in_valid_range() {
    let processor = StubUtlProcessor::new();
    let context = UtlContext::default();

    for input in ["", "x", "test phrase", "A very long input string for testing boundaries"] {
        let surprise = processor.compute_surprise(input, &context).await.unwrap();
        assert!((0.0..=1.0).contains(&surprise), "Surprise {} must be in [0.0, 1.0]", surprise);
    }
}

#[tokio::test]
async fn test_utl_coherence_change_in_valid_range() {
    let processor = StubUtlProcessor::new();
    let context = UtlContext::default();

    for input in ["", "x", "test phrase", "A very long input string for testing boundaries"] {
        let coherence = processor.compute_coherence_change(input, &context).await.unwrap();
        assert!((0.0..=1.0).contains(&coherence), "Coherence {} must be in [0.0, 1.0]", coherence);
    }
}

#[tokio::test]
async fn test_utl_alignment_in_valid_range() {
    let processor = StubUtlProcessor::new();
    let context = UtlContext::default();

    for input in ["", "x", "test phrase", "A very long input string for testing boundaries"] {
        let alignment = processor.compute_alignment(input, &context).await.unwrap();
        assert!((-1.0..=1.0).contains(&alignment), "Alignment {} must be in [-1.0, 1.0]", alignment);
    }
}

#[tokio::test]
async fn test_utl_emotional_weight_in_valid_range() {
    let processor = StubUtlProcessor::new();

    let states = [
        EmotionalState::Neutral,
        EmotionalState::Curious,
        EmotionalState::Focused,
        EmotionalState::Stressed,
        EmotionalState::Fatigued,
        EmotionalState::Engaged,
        EmotionalState::Confused,
    ];

    for state in states {
        let context = UtlContext { emotional_state: state, ..Default::default() };
        let weight = processor.compute_emotional_weight("test", &context).await.unwrap();
        assert!((0.5..=1.5).contains(&weight), "Weight {} for {:?} must be in [0.5, 1.5]", weight, state);
    }
}

#[tokio::test]
async fn test_utl_metrics_contains_all_components() {
    let processor = StubUtlProcessor::new();
    let context = UtlContext { prior_entropy: 0.6, current_coherence: 0.7, ..Default::default() };

    let metrics = processor.compute_metrics("test input", &context).await.unwrap();

    // Verify all fields are populated
    assert_eq!(metrics.entropy, context.prior_entropy, "Entropy must match context");
    assert_eq!(metrics.coherence, context.current_coherence, "Coherence must match context");

    // Verify components match individual computations
    let surprise = processor.compute_surprise("test input", &context).await.unwrap();
    let coherence_change = processor.compute_coherence_change("test input", &context).await.unwrap();
    let emotional_weight = processor.compute_emotional_weight("test input", &context).await.unwrap();
    let alignment = processor.compute_alignment("test input", &context).await.unwrap();
    let learning_score = processor.compute_learning_score("test input", &context).await.unwrap();

    assert_eq!(metrics.surprise, surprise);
    assert_eq!(metrics.coherence_change, coherence_change);
    assert_eq!(metrics.emotional_weight, emotional_weight);
    assert_eq!(metrics.alignment, alignment);
    assert_eq!(metrics.learning_score, learning_score);
}
