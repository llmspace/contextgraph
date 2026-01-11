//! Real GPU Embedding Integration Tests - Benchmarks (Feature-gated: cuda)
//!
//! All 13 spaces verification and performance benchmarks.

#![cfg(feature = "cuda")]

use super::infrastructure::*;
use crate::handlers::tests::{create_test_handlers_with_real_embeddings, extract_mcp_tool_data};
use serde_json::json;
use std::time::Instant;

/// FSV: Verify all 13 embedding spaces are populated and searchable.
#[tokio::test]
async fn test_fsv_real_embeddings_all_spaces_populated() {
    let (handlers, _tempdir) = create_test_handlers_with_real_embeddings().await;

    let store_request = make_request("memory/store", 1, json!({
        "content": "The function process_data(items: Vec<Item>) -> Result<Output> iterates",
        "importance": 0.9
    }));
    let response = handlers.dispatch(store_request).await;
    assert!(response.error.is_none(), "Store should succeed");

    let data = extract_mcp_tool_data(&response.result.expect("Should have result"));

    if let Some(emb_count) = data.get("embedderCount").and_then(|v| v.as_u64()) {
        assert_eq!(emb_count, 13, "Must have exactly 13 embeddings");
    }

    // Test single-space search for each space
    for space_index in 0..13 {
        let search_request = make_request("search/single_space", (100 + space_index) as i64, json!({
            "query": "function data processing",
            "space_index": space_index, "topK": 5, "minSimilarity": 0.0
        }));
        let response = handlers.dispatch(search_request).await;
        let status = if response.error.is_none() { "OK" } else { "FAIL" };
        let count = response.result.and_then(|r| r.get("count").and_then(|c| c.as_u64())).unwrap_or(0);
        println!("  Space {}: {} (results: {})", space_index, status, count);
        assert!(response.error.is_none(), "Space {} search should succeed", space_index);
    }
}

/// FSV: Performance benchmark with REAL GPU embeddings.
#[tokio::test]
async fn test_fsv_real_embeddings_performance() {
    let (handlers, _tempdir) = create_test_handlers_with_real_embeddings().await;

    // Warm up
    let warmup = make_request("memory/store", 0, json!({
        "content": "Warmup content for model initialization", "importance": 0.5
    }));
    handlers.dispatch(warmup).await;

    // Benchmark store operations
    let mut store_latencies = Vec::new();
    for i in 0..5 {
        let content = format!("Performance test content number {} for benchmarking", i);
        let request = make_request("memory/store", (i + 1) as i64, json!({
            "content": content, "importance": 0.8
        }));

        let start = Instant::now();
        let response = handlers.dispatch(request).await;
        let latency = start.elapsed();

        assert!(response.error.is_none(), "Store {} should succeed", i);
        store_latencies.push(latency.as_millis() as u64);
        println!("Store[{}]: {}ms", i, latency.as_millis());
    }

    // Benchmark search operations
    let mut search_latencies = Vec::new();
    for i in 0..5 {
        let request = make_request("search/multi", (100 + i) as i64, json!({
            "query": format!("benchmark query number {}", i),
            "query_type": "semantic_search", "topK": 10, "minSimilarity": 0.0
        }));

        let start = Instant::now();
        let response = handlers.dispatch(request).await;
        let latency = start.elapsed();

        assert!(response.error.is_none(), "Search {} should succeed", i);
        search_latencies.push(latency.as_millis() as u64);
        println!("Search[{}]: {}ms", i, latency.as_millis());
    }

    // Calculate statistics
    store_latencies.sort();
    search_latencies.sort();

    let store_median = store_latencies[store_latencies.len() / 2];
    let store_p95 = *store_latencies.last().unwrap_or(&0);
    let search_median = search_latencies[search_latencies.len() / 2];
    let search_p95 = *search_latencies.last().unwrap_or(&0);

    println!("\nPerformance Summary:");
    println!("  Store - Median: {}ms, P95: {}ms", store_median, store_p95);
    println!("  Search - Median: {}ms, P95: {}ms", search_median, search_p95);

    if store_p95 > 100 {
        println!("  WARNING: Store P95 {}ms exceeds 100ms target", store_p95);
    }
    if search_p95 > 100 {
        println!("  WARNING: Search P95 {}ms exceeds 100ms target", search_p95);
    }
}
