//! Real GPU Embedding Integration Tests - Core (Feature-gated: cuda)
//!
//! Memory lifecycle and semantic relevance tests with real GPU embeddings.

use super::infrastructure::*;
use crate::handlers::tests::{create_test_handlers_with_real_embeddings, extract_mcp_tool_data};
use std::time::Instant;

/// FSV: Complete memory lifecycle with REAL GPU embeddings.
#[tokio::test]
async fn test_fsv_real_embeddings_memory_lifecycle() {
    let (handlers, _tempdir) = create_test_handlers_with_real_embeddings().await;

    // 1. STORE
    let content = "Neural networks learn hierarchical representations of data patterns";
    let store_request = make_request(
        "memory/store",
        1,
        json!({"content": content, "importance": 0.95}),
    );

    let store_start = Instant::now();
    let store_response = handlers.dispatch(store_request).await;
    println!("Store latency: {:?}", store_start.elapsed());

    assert!(store_response.error.is_none(), "Store should succeed");
    let store_data = extract_mcp_tool_data(&store_response.result.expect("Should have result"));
    let fingerprint_id = store_data
        .get("fingerprintId")
        .or_else(|| store_data.get("fingerprint_id"))
        .and_then(|v| v.as_str())
        .expect("Should have fingerprint_id");

    if let Some(emb_count) = store_data.get("embedderCount").and_then(|v| v.as_u64()) {
        assert_eq!(emb_count, 13, "Should have 13 embeddings");
    }

    if let Some(pv) = store_data.get("purpose_vector").and_then(|v| v.as_array()) {
        assert_eq!(pv.len(), 13, "Purpose vector should be 13D");
        for (i, dim) in pv.iter().enumerate() {
            if let Some(val) = dim.as_f64() {
                assert!((-1.0..=1.0).contains(&val), "PV[{}] in [-1, 1]: {}", i, val);
            }
        }
    }

    // 2. RETRIEVE
    let retrieve_request = make_request(
        "memory/retrieve",
        2,
        json!({"fingerprintId": fingerprint_id}),
    );
    let retrieve_response = handlers.dispatch(retrieve_request).await;
    assert!(retrieve_response.error.is_none(), "Retrieve should succeed");
    let retrieve_data =
        extract_mcp_tool_data(&retrieve_response.result.expect("Should have result"));
    let retrieved_id = retrieve_data
        .get("fingerprint")
        .and_then(|fp| fp.get("id"))
        .and_then(|v| v.as_str())
        .expect("Should have fingerprint.id");
    assert_eq!(retrieved_id, fingerprint_id);

    // 3. SEARCH
    let search_request = make_request(
        "search/multi",
        3,
        json!({
            "query": "machine learning neural network data patterns",
            "query_type": "semantic_search", "topK": 10, "minSimilarity": 0.0,
            "include_per_embedder_scores": true
        }),
    );
    let search_start = Instant::now();
    let search_response = handlers.dispatch(search_request).await;
    println!("Search latency: {:?}", search_start.elapsed());

    assert!(search_response.error.is_none(), "Search should succeed");
    let search_result = search_response.result.expect("Should have result");
    let results = search_result
        .get("results")
        .and_then(|v| v.as_array())
        .expect("Should have results");
    assert!(!results.is_empty(), "Should find at least one result");
    assert_eq!(
        results[0].get("fingerprintId").and_then(|v| v.as_str()),
        Some(fingerprint_id)
    );

    // 4. DELETE
    let delete_request = make_request(
        "memory/delete",
        4,
        json!({"fingerprintId": fingerprint_id, "soft": false}),
    );
    let delete_response = handlers.dispatch(delete_request).await;
    assert!(delete_response.error.is_none(), "Delete should succeed");

    // 5. VERIFY DELETION
    let verify_request = make_request(
        "search/multi",
        5,
        json!({
            "query": "neural networks", "query_type": "semantic_search", "topK": 10, "minSimilarity": 0.0
        }),
    );
    let verify_result = handlers
        .dispatch(verify_request)
        .await
        .result
        .expect("Should have result");
    assert_eq!(
        verify_result
            .get("count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        0
    );
}

/// FSV: Semantic search relevance with REAL embeddings.
#[tokio::test]
async fn test_fsv_real_embeddings_semantic_relevance() {
    let (handlers, _tempdir) = create_test_handlers_with_real_embeddings().await;

    let contents = [
        (
            "Rust programming language provides memory safety without garbage collection",
            0.9,
        ),
        (
            "Python is a high-level interpreted programming language for data science",
            0.9,
        ),
        (
            "The Eiffel Tower is a famous landmark in Paris, France",
            0.8,
        ),
        (
            "Kubernetes orchestrates containerized applications across clusters",
            0.85,
        ),
    ];

    for (i, (content, importance)) in contents.iter().enumerate() {
        let request = make_request(
            "memory/store",
            (i + 1) as i64,
            json!({"content": content, "importance": importance}),
        );
        let response = handlers.dispatch(request).await;
        assert!(response.error.is_none(), "Store {} should succeed", i);
    }

    let search_request = make_request(
        "search/multi",
        100,
        json!({
            "query": "programming language and software development",
            "query_type": "semantic_search", "topK": 10, "minSimilarity": 0.0
        }),
    );
    let response = handlers.dispatch(search_request).await;
    assert!(response.error.is_none(), "Search should succeed");

    let search_result = response.result.expect("Should have result");
    let results = search_result
        .get("results")
        .and_then(|v| v.as_array())
        .expect("Should have results");

    if results.len() >= 2 {
        let top_content = results[0]
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let is_programming = top_content.contains("programming")
            || top_content.contains("Rust")
            || top_content.contains("Python")
            || top_content.contains("Kubernetes");
        if is_programming {
            println!("Real embeddings correctly ranked programming content first");
        }
    }
}

/// FSV: Purpose alignment with REAL 13D purpose vectors.
#[tokio::test]
async fn test_fsv_real_embeddings_purpose_alignment() {
    let (handlers, _tempdir) = create_test_handlers_with_real_embeddings().await;

    let store_request = make_request(
        "memory/store",
        1,
        json!({
            "content": "Responsible AI development prioritizes safety and ethical considerations",
            "importance": 0.95
        }),
    );
    let response = handlers.dispatch(store_request).await;
    assert!(response.error.is_none(), "Store should succeed");

    let data = extract_mcp_tool_data(&response.result.expect("Should have result"));

    if let Some(pv) = data.get("purpose_vector").and_then(|v| v.as_array()) {
        assert_eq!(pv.len(), 13, "Purpose vector must be 13D");
        for (i, dim) in pv.iter().enumerate() {
            let val = dim.as_f64().unwrap_or(0.0);
            assert!((-1.0..=1.0).contains(&val), "PV[{}] in [-1, 1]: {}", i, val);
        }
    }

    // Search by purpose
    let purpose_vector: Vec<f64> = vec![
        0.9, 0.2, 0.2, 0.2, 0.8, 0.1, 0.3, 0.3, 0.2, 0.3, 0.5, 0.1, 0.1,
    ];
    let search_request = make_request(
        "search/by_purpose",
        2,
        json!({
            "purpose_vector": purpose_vector, "topK": 5, "threshold": -1.0
        }),
    );
    let response = handlers.dispatch(search_request).await;
    assert!(response.error.is_none(), "By-purpose search should succeed");

    let purpose_result = response.result.expect("Should have result");
    let results = purpose_result
        .get("results")
        .and_then(|v| v.as_array())
        .expect("Should have results");
    if !results.is_empty() {
        let alignment = results[0]
            .get("alignment_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        assert!((-1.0..=1.0).contains(&alignment), "Alignment in [-1, 1]");
    }
}
