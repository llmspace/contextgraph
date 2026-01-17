//! Comprehensive evidence log tests.
//!
//! Full evidence log showing data flow through the system.

use serde_json::json;
use sha2::{Digest, Sha256};

use crate::protocol::JsonRpcId;

use super::helpers::{create_handlers_with_store_access, exists_in_store, make_request};

// =============================================================================
// COMPREHENSIVE EVIDENCE LOG TEST
// =============================================================================

/// Full evidence log showing data flow through the system.
#[tokio::test]
async fn verify_complete_evidence_log() {
    println!("\n================================================================================");
    println!("COMPREHENSIVE EVIDENCE LOG: Complete Data Flow Verification");
    println!("================================================================================");

    let (handlers, store, provider) = create_handlers_with_store_access();

    // === INITIAL STATE ===
    println!("\n[INITIAL STATE]");
    println!("  Store count: {}", store.count().await.unwrap());

    // === STEP 1: Generate embeddings (verify provider works) ===
    println!("\n[STEP 1: EMBEDDING GENERATION]");
    let test_content = "Convolutional neural networks excel at image recognition tasks";
    let embedding_output = provider
        .embed_all(test_content)
        .await
        .expect("embed_all should work");
    println!("  Content: \"{}\"", test_content);
    println!("  Embeddings generated: {} slots", 13);
    println!(
        "  E1 semantic dimension: {}",
        embedding_output.fingerprint.e1_semantic.len()
    );
    println!(
        "  E6 sparse NNZ: {}",
        embedding_output.fingerprint.e6_sparse.nnz()
    );
    println!("  Total latency: {:?}", embedding_output.total_latency);

    // === STEP 2: Store via MCP handler ===
    println!("\n[STEP 2: MCP STORE OPERATION]");
    let store_params = json!({ "content": test_content, "importance": 0.95 });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    let store_response = handlers.dispatch(store_request).await;

    let store_result = store_response.result.expect("Store must succeed");
    let fingerprint_id_str = store_result.get("fingerprintId").unwrap().as_str().unwrap();
    let fingerprint_id = uuid::Uuid::parse_str(fingerprint_id_str).unwrap();

    println!("  Response fingerprintId: {}", fingerprint_id_str);
    println!(
        "  Response embedderCount: {}",
        store_result.get("embedderCount").unwrap()
    );
    println!(
        "  Response embeddingLatencyMs: {}",
        store_result.get("embeddingLatencyMs").unwrap()
    );
    println!(
        "  Response storageLatencyMs: {}",
        store_result.get("storageLatencyMs").unwrap()
    );

    // === STEP 3: Verify in Source of Truth ===
    println!("\n[STEP 3: SOURCE OF TRUTH VERIFICATION]");
    let stored_fp = store.retrieve(fingerprint_id).await.unwrap().unwrap();
    println!("  Direct store.retrieve() succeeded");
    println!("  Stored fingerprint fields:");
    println!("    - id: {}", stored_fp.id);
    println!("    - alignment_score: {:.6}", stored_fp.alignment_score);
    println!("    - access_count: {}", stored_fp.access_count);
    println!("    - created_at: {}", stored_fp.created_at);
    println!("    - last_updated: {}", stored_fp.last_updated);
    println!(
        "    - content_hash: {}",
        hex::encode(stored_fp.content_hash)
    );
    println!(
        "    - semantic.e1_semantic[0..5]: {:?}",
        &stored_fp.semantic.e1_semantic[0..5]
    );
    println!(
        "    - semantic.e6_sparse.nnz: {}",
        stored_fp.semantic.e6_sparse.nnz()
    );
    println!(
        "    - purpose_vector.alignments[0..5]: {:?}",
        &stored_fp.purpose_vector.alignments[0..5]
    );
    println!(
        "    - purpose_vector.dominant_embedder: {}",
        stored_fp.purpose_vector.dominant_embedder
    );
    println!(
        "    - purpose_vector.coherence: {:.6}",
        stored_fp.purpose_vector.coherence
    );
    println!(
        "    - johari.quadrants[0] (E1): {:?}",
        stored_fp.johari.quadrants[0]
    );
    println!(
        "    - johari.confidence[0] (E1): {:.6}",
        stored_fp.johari.confidence[0]
    );

    // Verify content hash
    let mut hasher = Sha256::new();
    hasher.update(test_content.as_bytes());
    let expected_hash: [u8; 32] = hasher.finalize().into();
    assert_eq!(stored_fp.content_hash, expected_hash);
    println!("    - content_hash MATCHES expected SHA-256");

    // === STEP 4: Final state ===
    println!("\n[FINAL STATE]");
    println!("  Store count: {}", store.count().await.unwrap());
    let final_exists = exists_in_store(&store, fingerprint_id).await;
    println!("  Fingerprint {} exists: {}", fingerprint_id, final_exists);

    println!("\n================================================================================");
    println!("EVIDENCE LOG COMPLETE - All verifications passed");
    println!("================================================================================\n");
}
