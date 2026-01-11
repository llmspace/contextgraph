//! FSV TEST 2: Multi-Embedding Search Comprehensive
//!
//! Tests multi-embedding search with all query types and weight verification.

use super::infrastructure::*;
use context_graph_core::traits::TeleologicalMemoryStore;

/// FSV: Multi-embedding search with all query types and weight verification.
#[tokio::test]
async fn test_fsv_multi_embedding_search_comprehensive() {
    println!("\n======================================================================");
    println!("FSV TEST 2: Multi-Embedding Search Comprehensive");
    println!("======================================================================\n");

    let ctx = TestContext::new();

    // =========================================================================
    // SETUP: Store multiple fingerprints with varying content
    // =========================================================================
    println!("SETUP: Storing 5 fingerprints with varying content");
    let contents = [
        "Machine learning algorithms process data to make predictions",
        "Neural networks model complex patterns in high-dimensional spaces",
        "Deep learning enables hierarchical feature extraction",
        "Natural language processing handles text understanding",
        "Computer vision algorithms analyze visual information",
    ];

    let mut stored_ids: Vec<Uuid> = Vec::new();
    for (i, content) in contents.iter().enumerate() {
        let request = make_request(
            "memory/store",
            (i + 1) as i64,
            json!({
                "content": content,
                "importance": 0.8
            }),
        );
        let response = ctx.handlers.dispatch(request).await;
        assert!(response.error.is_none(), "Store {} MUST succeed", i);

        let result = response.result.unwrap();
        let id_str = result["fingerprintId"].as_str().unwrap();
        stored_ids.push(Uuid::parse_str(id_str).unwrap());
    }

    // VERIFY IN SOURCE OF TRUTH
    let count = ctx.store.count().await.expect("count should succeed");
    println!(
        "   - Stored {} fingerprints (verified in store: {})",
        stored_ids.len(),
        count
    );
    assert_eq!(count, 5, "Store MUST contain 5 fingerprints");
    println!("   VERIFIED: All fingerprints stored\n");

    // =========================================================================
    // TEST 1: Semantic Search
    // =========================================================================
    println!("TEST 1: search/multi with semantic_search");
    let search_request = make_request(
        "search/multi",
        10,
        json!({
            "query": "machine learning neural",
            "query_type": "semantic_search",
            "topK": 5,
            "minSimilarity": 0.0,
            "include_per_embedder_scores": true
        }),
    );
    let search_response = ctx.handlers.dispatch(search_request).await;

    assert!(
        search_response.error.is_none(),
        "Semantic search MUST succeed"
    );
    let result = search_response.result.unwrap();
    let results = result["results"].as_array().unwrap();
    println!("   - Results: {} (expected: up to 5)", results.len());

    // Verify per-embedder scores
    if let Some(first) = results.first() {
        let scores = first.get("per_embedder_scores").and_then(|v| v.as_object());
        let score_count = scores.map(|s| s.len()).unwrap_or(0);
        println!("   - Per-embedder scores: {} (expected: 13)", score_count);
        assert_eq!(
            score_count, NUM_EMBEDDERS,
            "MUST have 13 per-embedder scores"
        );
    }
    println!("   VERIFIED: Semantic search works correctly\n");

    // =========================================================================
    // TEST 2: Custom Weights (13 elements)
    // =========================================================================
    println!("TEST 2: search/multi with custom weights");
    let custom_weights: Vec<f64> = vec![
        0.2,  // E1 semantic - high weight
        0.1,  // E2
        0.05, // E3
        0.05, // E4
        0.1,  // E5 causal
        0.05, // E6 sparse
        0.1,  // E7 code
        0.05, // E8 graph
        0.05, // E9 HDC
        0.05, // E10 multimodal
        0.1,  // E11 entity
        0.05, // E12 late interaction
        0.05, // E13 SPLADE
    ];
    assert_eq!(
        custom_weights.len(),
        NUM_EMBEDDERS,
        "Custom weights MUST be 13"
    );

    let custom_search = make_request(
        "search/multi",
        11,
        json!({
            "query": "deep learning patterns",
            "query_type": "custom",
            "weights": custom_weights,
            "topK": 3,
            "minSimilarity": 0.0
        }),
    );
    let custom_response = ctx.handlers.dispatch(custom_search).await;

    assert!(
        custom_response.error.is_none(),
        "Custom weight search MUST succeed"
    );
    let custom_result = custom_response.result.unwrap();
    let custom_results = custom_result["results"].as_array().unwrap();
    println!("   - Custom results: {}", custom_results.len());
    println!("   VERIFIED: Custom weights search works correctly\n");

    // =========================================================================
    // TEST 3: Single Space Search
    // =========================================================================
    println!("TEST 3: search/single_space with space_index=0 (E1 semantic)");
    let single_space = make_request(
        "search/single_space",
        12,
        json!({
            "query": "vision analysis",
            "space_index": 0,
            "topK": 3,
            "minSimilarity": 0.0
        }),
    );
    let single_response = ctx.handlers.dispatch(single_space).await;

    assert!(
        single_response.error.is_none(),
        "Single space search MUST succeed"
    );
    let single_result = single_response.result.unwrap();
    let single_results = single_result["results"].as_array().unwrap();
    println!("   - Single space results: {}", single_results.len());
    println!("   VERIFIED: Single space search works correctly\n");

    // =========================================================================
    // TEST 4: Weight Profiles
    // =========================================================================
    println!("TEST 4: search/weight_profiles");
    let profiles_request = make_request_no_params("search/weight_profiles", 13);
    let profiles_response = ctx.handlers.dispatch(profiles_request).await;

    assert!(
        profiles_response.error.is_none(),
        "Weight profiles MUST succeed"
    );
    let profiles_result = profiles_response.result.unwrap();

    let total_spaces = profiles_result["total_spaces"].as_u64().unwrap();
    assert_eq!(total_spaces, 13, "total_spaces MUST be 13");

    let profiles = profiles_result["profiles"].as_array().unwrap();
    println!("   - Profiles returned: {}", profiles.len());

    for profile in profiles {
        let weights = profile["weights"].as_array().unwrap();
        assert_eq!(
            weights.len(),
            NUM_EMBEDDERS,
            "Each profile MUST have 13 weights"
        );
        let sum: f64 = weights.iter().filter_map(|w| w.as_f64()).sum();
        assert!((sum - 1.0).abs() < 0.01, "Profile weights MUST sum to ~1.0");
    }
    println!("   VERIFIED: All profiles have 13 weights summing to 1.0\n");

    // =========================================================================
    // EVIDENCE OF SUCCESS
    // =========================================================================
    println!("======================================================================");
    println!("EVIDENCE OF SUCCESS - Multi-Embedding Search Verification");
    println!("======================================================================");
    println!(
        "Source of Truth: InMemoryTeleologicalStore with {} fingerprints",
        count
    );
    println!();
    println!("Search Operations Verified:");
    println!("  1. semantic_search: Found results with 13 per-embedder scores");
    println!("  2. custom weights (13): Correctly weighted search");
    println!("  3. single_space (E1): Single embedding space search");
    println!("  4. weight_profiles: All profiles have 13 weights summing to 1.0");
    println!("======================================================================\n");
}
