//! Manual verification tests for TASK-P3-001: PerSpaceScores and SimilarityResult
//!
//! These tests perform full state verification with synthetic data.
//! They verify:
//! 1. Happy path scenarios
//! 2. Edge cases (clamping, empty, boundary conditions)
//! 3. AP-60 compliance (temporal exclusion)
//! 4. Serialization roundtrips

use context_graph_core::retrieval::{PerSpaceScores, SimilarityResult, NUM_SPACES};
use context_graph_core::teleological::Embedder;
use uuid::Uuid;

// =============================================================================
// TEST 1: Happy Path - Create and Inspect PerSpaceScores
// =============================================================================
#[test]
fn manual_test_happy_path_per_space_scores() {
    println!("\n=== MANUAL TEST 1: Happy Path - Create and Inspect PerSpaceScores ===");

    // Synthetic input
    let mut scores = PerSpaceScores::new();
    println!("BEFORE: Created new PerSpaceScores (all zeros)");
    println!("  semantic = {}", scores.semantic);
    println!("  causal = {}", scores.causal);
    println!("  code = {}", scores.code);
    println!("  emotional = {}", scores.emotional);
    println!("  hdc = {}", scores.hdc);

    scores.semantic = 0.85;
    scores.causal = 0.72;
    scores.code = 0.91;
    scores.emotional = 0.60;
    scores.hdc = 0.55;

    println!("\nAFTER: Set scores");
    println!("  semantic = {}", scores.semantic);
    println!("  causal = {}", scores.causal);
    println!("  code = {}", scores.code);
    println!("  emotional = {}", scores.emotional);
    println!("  hdc = {}", scores.hdc);

    // Expected outputs
    let expected_max = 0.91_f32;
    let expected_mean = (0.85 + 0.72 + 0.91 + 0.60 + 0.55) / 13.0;
    // weighted_mean: semantic*1.0 + causal*1.0 + code*1.0 + emotional*0.5 + hdc*0.5
    // = 0.85 + 0.72 + 0.91 + 0.30 + 0.275 = 3.055 / 8.5 ≈ 0.359
    let expected_weighted = (0.85 * 1.0 + 0.72 * 1.0 + 0.91 * 1.0 + 0.60 * 0.5 + 0.55 * 0.5) / 8.5;

    let actual_max = scores.max_score();
    let actual_mean = scores.mean_score();
    let actual_weighted = scores.weighted_mean();

    println!("\nVERIFICATION:");
    println!("  max_score() = {} (expected {})", actual_max, expected_max);
    println!("  mean_score() = {:.6} (expected {:.6})", actual_mean, expected_mean);
    println!("  weighted_mean() = {:.6} (expected {:.6})", actual_weighted, expected_weighted);

    assert!((actual_max - expected_max).abs() < 1e-6, "max_score mismatch");
    assert!((actual_mean - expected_mean).abs() < 1e-6, "mean_score mismatch");
    assert!((actual_weighted - expected_weighted).abs() < 1e-5, "weighted_mean mismatch");

    println!("\n[PASS] Happy path test: All values match expected");
}

// =============================================================================
// TEST 2: Edge Case - Score Clamping
// =============================================================================
#[test]
fn manual_test_edge_case_score_clamping() {
    println!("\n=== MANUAL TEST 2: Edge Case - Score Clamping ===");

    let mut scores = PerSpaceScores::new();

    println!("BEFORE: scores.semantic = {}", scores.semantic);
    println!("Setting score 1.5 for E1 (Semantic)...");
    scores.set_score(Embedder::Semantic, 1.5);
    println!("AFTER: scores.semantic = {} (expected 1.0 - clamped)", scores.semantic);
    assert_eq!(scores.semantic, 1.0, "Score should be clamped to 1.0");

    println!("\nBEFORE: scores.code = {}", scores.code);
    println!("Setting score -0.5 for E7 (Code)...");
    scores.set_score(Embedder::Code, -0.5);
    println!("AFTER: scores.code = {} (expected 0.0 - clamped)", scores.code);
    assert_eq!(scores.code, 0.0, "Score should be clamped to 0.0");

    println!("\n[PASS] Edge case: Score clamping enforces [0.0, 1.0]");
}

// =============================================================================
// TEST 3: Edge Case - Empty Scores (Default)
// =============================================================================
#[test]
fn manual_test_edge_case_empty_scores() {
    println!("\n=== MANUAL TEST 3: Edge Case - Empty Scores (Default) ===");

    let scores = PerSpaceScores::default();

    println!("STATE: Created default PerSpaceScores");
    println!("  All 13 fields should be 0.0:");
    for embedder in Embedder::all() {
        let score = scores.get_score(embedder);
        println!("    {:?} = {}", embedder, score);
        assert_eq!(score, 0.0, "Default score should be 0.0");
    }

    println!("\nVERIFICATION:");
    println!("  max_score() = {} (expected 0.0)", scores.max_score());
    println!("  mean_score() = {} (expected 0.0)", scores.mean_score());
    println!("  weighted_mean() = {} (expected 0.0)", scores.weighted_mean());

    assert_eq!(scores.max_score(), 0.0);
    assert_eq!(scores.mean_score(), 0.0);
    assert_eq!(scores.weighted_mean(), 0.0);

    println!("\n[PASS] Edge case: Empty scores all return 0.0");
}

// =============================================================================
// TEST 4: AP-60 - Temporal Exclusion
// =============================================================================
#[test]
fn manual_test_ap60_temporal_exclusion() {
    println!("\n=== MANUAL TEST 4: AP-60 - Temporal Exclusion ===");

    let mut scores = PerSpaceScores::new();

    // Set ONLY temporal spaces to 1.0
    scores.temporal_recent = 1.0;
    scores.temporal_periodic = 1.0;
    scores.temporal_positional = 1.0;

    println!("STATE: Only temporal spaces set to 1.0");
    println!("  temporal_recent = {}", scores.temporal_recent);
    println!("  temporal_periodic = {}", scores.temporal_periodic);
    println!("  temporal_positional = {}", scores.temporal_positional);
    println!("  All other spaces = 0.0");

    let weighted = scores.weighted_mean();
    println!("\nVERIFICATION:");
    println!("  weighted_mean() = {} (MUST be 0.0 because temporal weight = 0.0)", weighted);

    assert_eq!(weighted, 0.0, "FAIL: Temporal should be excluded per AP-60");

    println!("\n[PASS] AP-60 verified: Temporal spaces (E2-E4) excluded from weighted_mean");
}

// =============================================================================
// TEST 5: Boundary - Maximum Weighted Agreement
// =============================================================================
#[test]
fn manual_test_boundary_max_weighted_agreement() {
    println!("\n=== MANUAL TEST 5: Boundary - Maximum Weighted Agreement ===");

    let mut scores = PerSpaceScores::new();

    // Set ALL spaces to 1.0
    for embedder in Embedder::all() {
        scores.set_score(embedder, 1.0);
    }

    println!("STATE: All 13 spaces set to 1.0");
    println!("FORMULA: weighted_sum = 7*1.0 + 3*0.0 + 2*0.5 + 1*0.5 = 8.5");
    println!("         weighted_mean = 8.5 / 8.5 = 1.0");

    let weighted = scores.weighted_mean();
    println!("\nVERIFICATION:");
    println!("  weighted_mean() = {:.6} (expected 1.0)", weighted);

    assert!((weighted - 1.0).abs() < 1e-6, "Maximum should be 1.0");

    println!("\n[PASS] Boundary: Maximum weighted_mean = 1.0");
}

// =============================================================================
// TEST 6: SimilarityResult with Memory ID
// =============================================================================
#[test]
fn manual_test_similarity_result_with_memory_id() {
    println!("\n=== MANUAL TEST 6: SimilarityResult with Memory ID ===");

    let memory_id = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
    let mut scores = PerSpaceScores::new();
    scores.semantic = 0.88;
    scores.code = 0.95;

    println!("INPUT:");
    println!("  memory_id = {}", memory_id);
    println!("  scores.semantic = {}", scores.semantic);
    println!("  scores.code = {}", scores.code);
    println!("  relevance_score = 0.82");
    println!("  matching_spaces = [Semantic, Code]");

    let result = SimilarityResult::with_relevance(
        memory_id,
        scores,
        0.82,
        vec![Embedder::Semantic, Embedder::Code],
    );

    println!("\nOUTPUT (Source of Truth):");
    println!("  result.memory_id = {}", result.memory_id);
    println!("  result.space_count = {}", result.space_count);
    println!("  result.relevance_score = {}", result.relevance_score);
    println!("  result.included_spaces.len() = {}", result.included_spaces.len());
    println!("  result.weighted_similarity = {:.6}", result.weighted_similarity);

    // Verify source of truth
    assert_eq!(result.memory_id.to_string(), "550e8400-e29b-41d4-a716-446655440000");
    assert_eq!(result.space_count, 2);
    assert_eq!(result.relevance_score, 0.82);
    assert_eq!(result.included_spaces.len(), 10); // 13 - 3 temporal

    println!("\n[PASS] SimilarityResult has correct memory_id and space_count");
}

// =============================================================================
// TEST 7: Serialization Roundtrip (Source of Truth = JSON String)
// =============================================================================
#[test]
fn manual_test_serialization_roundtrip() {
    println!("\n=== MANUAL TEST 7: Serialization Roundtrip ===");

    let mut scores = PerSpaceScores::new();
    scores.semantic = 0.75;
    scores.code = 0.88;
    scores.causal = 0.62;

    println!("INPUT:");
    println!("  scores.semantic = {}", scores.semantic);
    println!("  scores.code = {}", scores.code);
    println!("  scores.causal = {}", scores.causal);

    let json = serde_json::to_string(&scores).unwrap();
    println!("\nSERIALIZED JSON:");
    println!("  {}", json);

    let restored: PerSpaceScores = serde_json::from_str(&json).unwrap();

    println!("\nRESTORED:");
    println!("  restored.semantic = {}", restored.semantic);
    println!("  restored.code = {}", restored.code);
    println!("  restored.causal = {}", restored.causal);

    assert_eq!(restored.semantic, 0.75, "FAIL: Serialization lost semantic data");
    assert_eq!(restored.code, 0.88, "FAIL: Serialization lost code data");
    assert_eq!(restored.causal, 0.62, "FAIL: Serialization lost causal data");

    println!("\n[PASS] Serialization roundtrip preserved all values");
}

// =============================================================================
// TEST 8: Array Conversion Roundtrip
// =============================================================================
#[test]
fn manual_test_array_conversion() {
    println!("\n=== MANUAL TEST 8: Array Conversion Roundtrip ===");

    let mut scores = PerSpaceScores::new();
    scores.semantic = 0.1;
    scores.temporal_recent = 0.2;
    scores.causal = 0.5;
    scores.code = 0.7;
    scores.keyword_splade = 0.13;

    println!("INPUT:");
    println!("  E1 (semantic) = 0.1 at index 0");
    println!("  E2 (temporal_recent) = 0.2 at index 1");
    println!("  E5 (causal) = 0.5 at index 4");
    println!("  E7 (code) = 0.7 at index 6");
    println!("  E13 (keyword_splade) = 0.13 at index 12");

    let arr = scores.to_array();
    println!("\nARRAY OUTPUT:");
    for (i, v) in arr.iter().enumerate() {
        if *v != 0.0 {
            println!("  arr[{}] = {}", i, v);
        }
    }

    assert_eq!(arr[0], 0.1, "E1 at wrong index");
    assert_eq!(arr[1], 0.2, "E2 at wrong index");
    assert_eq!(arr[4], 0.5, "E5 at wrong index");
    assert_eq!(arr[6], 0.7, "E7 at wrong index");
    assert_eq!(arr[12], 0.13, "E13 at wrong index");

    let recovered = PerSpaceScores::from_array(arr);
    println!("\nRECOVERED:");
    println!("  recovered.semantic = {}", recovered.semantic);
    println!("  recovered.temporal_recent = {}", recovered.temporal_recent);
    println!("  recovered.causal = {}", recovered.causal);
    println!("  recovered.code = {}", recovered.code);
    println!("  recovered.keyword_splade = {}", recovered.keyword_splade);

    assert_eq!(recovered.semantic, 0.1);
    assert_eq!(recovered.temporal_recent, 0.2);
    assert_eq!(recovered.causal, 0.5);
    assert_eq!(recovered.code, 0.7);
    assert_eq!(recovered.keyword_splade, 0.13);

    println!("\n[PASS] Array conversion roundtrip works correctly");
}

// =============================================================================
// TEST 9: Included Spaces (Non-Temporal Only)
// =============================================================================
#[test]
fn manual_test_included_spaces() {
    println!("\n=== MANUAL TEST 9: Included Spaces (Non-Temporal Only) ===");

    let included = PerSpaceScores::included_spaces();

    println!("INCLUDED SPACES (weight > 0):");
    for embedder in &included {
        println!("  {:?}", embedder);
    }

    println!("\nEXCLUDED (temporal):");
    println!("  TemporalRecent (E2)");
    println!("  TemporalPeriodic (E3)");
    println!("  TemporalPositional (E4)");

    assert_eq!(included.len(), 10, "Should have 10 non-temporal spaces");
    assert!(!included.contains(&Embedder::TemporalRecent), "Should not include E2");
    assert!(!included.contains(&Embedder::TemporalPeriodic), "Should not include E3");
    assert!(!included.contains(&Embedder::TemporalPositional), "Should not include E4");
    assert!(included.contains(&Embedder::Semantic), "Should include E1");
    assert!(included.contains(&Embedder::Code), "Should include E7");

    println!("\n[PASS] included_spaces returns exactly 10 non-temporal spaces");
}

// =============================================================================
// TEST 10: Iterator Order Verification
// =============================================================================
#[test]
fn manual_test_iterator_order() {
    println!("\n=== MANUAL TEST 10: Iterator Order Verification ===");

    let scores = PerSpaceScores::new();

    println!("ITERATOR ORDER (should match Embedder::index()):");
    let collected: Vec<_> = scores.iter().collect();

    for (i, (embedder, score)) in collected.iter().enumerate() {
        println!("  [{}] {:?} = {}", i, embedder, score);
        assert_eq!(embedder.index(), i, "Iterator order mismatch at {}", i);
    }

    assert_eq!(collected.len(), 13);
    assert_eq!(collected[0].0, Embedder::Semantic);
    assert_eq!(collected[12].0, Embedder::KeywordSplade);

    println!("\n[PASS] Iterator visits all 13 spaces in correct order");
}

// =============================================================================
// TEST 11: NUM_SPACES constant
// =============================================================================
#[test]
fn manual_test_num_spaces_constant() {
    println!("\n=== MANUAL TEST 11: NUM_SPACES Constant ===");

    println!("NUM_SPACES = {}", NUM_SPACES);
    println!("Embedder::COUNT = {}", Embedder::COUNT);

    assert_eq!(NUM_SPACES, 13);
    assert_eq!(NUM_SPACES, Embedder::COUNT);

    println!("\n[PASS] NUM_SPACES = 13, matches Embedder::COUNT");
}

// =============================================================================
// TEST 12: SimilarityResult::new() auto-computes weighted_similarity
// =============================================================================
#[test]
fn manual_test_similarity_result_new_auto_compute() {
    println!("\n=== MANUAL TEST 12: SimilarityResult::new() auto-computes weighted_similarity ===");

    let id = Uuid::new_v4();
    let mut scores = PerSpaceScores::new();
    scores.semantic = 0.85;
    scores.code = 1.0;

    println!("INPUT:");
    println!("  scores.semantic = 0.85");
    println!("  scores.code = 1.0");

    let result = SimilarityResult::new(id, scores.clone());

    // Expected: (0.85 * 1.0 + 1.0 * 1.0) / 8.5 = 1.85 / 8.5 ≈ 0.2176
    let expected = (0.85 + 1.0) / 8.5;

    println!("\nOUTPUT:");
    println!("  result.weighted_similarity = {:.6}", result.weighted_similarity);
    println!("  expected = {:.6}", expected);

    assert!((result.weighted_similarity - expected).abs() < 1e-5);
    assert_eq!(result.relevance_score, 0.0);
    assert_eq!(result.space_count, 0);
    assert!(result.matching_spaces.is_empty());

    println!("\n[PASS] SimilarityResult::new() auto-computes weighted_similarity");
}
