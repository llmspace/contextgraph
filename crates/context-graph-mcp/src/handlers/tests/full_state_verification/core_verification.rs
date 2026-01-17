//! Core verification tests for store and retrieve operations.
//!
//! Tests verify that:
//! 1. Store operation physically creates fingerprint in store
//! 2. Retrieve operation returns exact stored data
//! 3. Delete operation removes fingerprint from store

use serde_json::json;
use sha2::{Digest, Sha256};

use crate::protocol::JsonRpcId;

use super::helpers::{create_handlers_with_store_access, exists_in_store, make_request};

// =============================================================================
// FULL STATE VERIFICATION: Direct Store Inspection
// =============================================================================

/// VERIFICATION TEST 1: Store operation physically creates fingerprint in store.
///
/// Source of Truth: InMemoryTeleologicalStore.data (DashMap<Uuid, TeleologicalFingerprint>)
/// Verification Method: Direct store.retrieve() and store.count() calls
#[tokio::test]
async fn verify_store_creates_fingerprint_in_source_of_truth() {
    println!("\n================================================================================");
    println!("FULL STATE VERIFICATION: Store Creates Fingerprint");
    println!("================================================================================");

    let (handlers, store, _provider) = create_handlers_with_store_access();

    // === BEFORE STATE ===
    let count_before = store.count().await.expect("count() should work");
    println!("\n[BEFORE] Store state:");
    println!("  - Fingerprint count: {}", count_before);
    assert_eq!(count_before, 0, "Store should be empty initially");

    // === EXECUTE OPERATION ===
    let content = "Machine learning is a subset of artificial intelligence";
    let params = json!({
        "content": content,
        "importance": 0.85
    });
    let request = make_request("memory/store", Some(JsonRpcId::Number(1)), Some(params));
    let response = handlers.dispatch(request).await;

    // Extract fingerprint ID from response (but don't trust this alone)
    let result = response.result.expect("Should have result");
    let fingerprint_id_str = result
        .get("fingerprintId")
        .expect("Should have fingerprintId")
        .as_str()
        .expect("Should be string");
    let fingerprint_id = uuid::Uuid::parse_str(fingerprint_id_str).expect("Should be valid UUID");

    println!("\n[OPERATION] Stored content with ID: {}", fingerprint_id);

    // === AFTER STATE - VERIFY SOURCE OF TRUTH ===
    let count_after = store.count().await.expect("count() should work");
    println!("\n[AFTER] Store state:");
    println!("  - Fingerprint count: {}", count_after);

    // CRITICAL: Directly verify fingerprint exists in store
    let exists = exists_in_store(&store, fingerprint_id).await;
    println!("  - Fingerprint {} exists: {}", fingerprint_id, exists);
    assert!(
        exists,
        "VERIFICATION FAILED: Fingerprint must exist in store"
    );

    // CRITICAL: Retrieve and inspect actual stored data
    let stored_fp = store
        .retrieve(fingerprint_id)
        .await
        .expect("retrieve() should work")
        .expect("Fingerprint must exist");

    println!("\n[EVIDENCE] Stored fingerprint fields:");
    println!("  - ID: {}", stored_fp.id);
    println!("  - alignment_score: {:.4}", stored_fp.alignment_score);
    println!("  - access_count: {}", stored_fp.access_count);
    println!("  - created_at: {}", stored_fp.created_at);
    println!("  - content_hash: {}", hex::encode(stored_fp.content_hash));
    println!(
        "  - semantic.e1_semantic len: {}",
        stored_fp.semantic.e1_semantic.len()
    );
    println!(
        "  - purpose_vector.alignments: {:?}",
        &stored_fp.purpose_vector.alignments[..5]
    );
    println!(
        "  - johari.quadrants[0]: {:?}",
        stored_fp.johari.quadrants[0]
    );

    // Verify content hash matches expected
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let expected_hash: [u8; 32] = hasher.finalize().into();
    assert_eq!(
        stored_fp.content_hash, expected_hash,
        "Content hash in store must match SHA-256 of original content"
    );
    println!("  - Content hash VERIFIED: matches SHA-256 of input");

    // Verify semantic fingerprint has valid embedding (stub uses 1024D)
    assert!(
        !stored_fp.semantic.e1_semantic.is_empty(),
        "E1 must have embeddings"
    );
    println!(
        "  - E1 semantic embedding dimension VERIFIED: {}",
        stored_fp.semantic.e1_semantic.len()
    );

    // Count must have increased
    assert_eq!(count_after, count_before + 1, "Count must increase by 1");
    println!("\n[VERIFICATION PASSED] Fingerprint physically exists in Source of Truth");
    println!("================================================================================\n");
}

/// VERIFICATION TEST 2: Retrieve operation returns exact stored data.
///
/// We store data, then verify retrieve returns THE SAME object from store.
#[tokio::test]
async fn verify_retrieve_returns_source_of_truth_data() {
    println!("\n================================================================================");
    println!("FULL STATE VERIFICATION: Retrieve Returns Source of Truth");
    println!("================================================================================");

    let (handlers, store, _provider) = create_handlers_with_store_access();

    // Store a fingerprint
    let content = "Neural networks learn hierarchical representations";
    let store_params = json!({
        "content": content,
        "importance": 0.9
    });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    let store_response = handlers.dispatch(store_request).await;
    let fingerprint_id_str = store_response
        .result
        .unwrap()
        .get("fingerprintId")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();
    let fingerprint_id = uuid::Uuid::parse_str(&fingerprint_id_str).unwrap();

    // === DIRECTLY READ FROM SOURCE OF TRUTH ===
    let truth_fp = store
        .retrieve(fingerprint_id)
        .await
        .expect("Direct store.retrieve() should work")
        .expect("Fingerprint must exist in store");

    println!("\n[SOURCE OF TRUTH] Direct store.retrieve() data:");
    println!("  - ID: {}", truth_fp.id);
    println!("  - content_hash: {}", hex::encode(truth_fp.content_hash));
    println!("  - alignment_score: {:.4}", truth_fp.alignment_score);

    // === NOW USE MCP HANDLER TO RETRIEVE ===
    let retrieve_params = json!({ "fingerprintId": fingerprint_id_str });
    let retrieve_request = make_request(
        "memory/retrieve",
        Some(JsonRpcId::Number(2)),
        Some(retrieve_params),
    );
    let retrieve_response = handlers.dispatch(retrieve_request).await;

    let result = retrieve_response.result.expect("Should have result");
    let fp_json = result.get("fingerprint").expect("Should have fingerprint");

    println!("\n[MCP HANDLER] memory/retrieve response:");
    println!("  - ID: {}", fp_json.get("id").unwrap().as_str().unwrap());
    println!(
        "  - contentHashHex: {}",
        fp_json.get("contentHashHex").unwrap().as_str().unwrap()
    );
    println!(
        "  - alignmentScore: {}",
        fp_json.get("alignmentScore").unwrap()
    );

    // === VERIFY HANDLER RETURNS SAME DATA AS SOURCE OF TRUTH ===
    assert_eq!(
        fp_json.get("id").unwrap().as_str().unwrap(),
        truth_fp.id.to_string(),
        "Handler must return same ID as store"
    );
    assert_eq!(
        fp_json.get("contentHashHex").unwrap().as_str().unwrap(),
        hex::encode(truth_fp.content_hash),
        "Handler must return same hash as store"
    );

    println!("\n[VERIFICATION PASSED] Handler retrieve returns Source of Truth data");
    println!("================================================================================\n");
}

/// VERIFICATION TEST 4: Hard delete removes fingerprint from Source of Truth.
#[tokio::test]
async fn verify_delete_removes_from_source_of_truth() {
    println!("\n================================================================================");
    println!("FULL STATE VERIFICATION: Delete Removes From Source of Truth");
    println!("================================================================================");

    let (handlers, store, _provider) = create_handlers_with_store_access();

    // Store a fingerprint
    let content = "This content will be deleted";
    let store_params = json!({ "content": content, "importance": 0.5 });
    let store_request = make_request(
        "memory/store",
        Some(JsonRpcId::Number(1)),
        Some(store_params),
    );
    let store_response = handlers.dispatch(store_request).await;
    let fingerprint_id_str = store_response
        .result
        .unwrap()
        .get("fingerprintId")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();
    let fingerprint_id = uuid::Uuid::parse_str(&fingerprint_id_str).unwrap();

    // === BEFORE DELETE STATE ===
    let count_before = store.count().await.unwrap();
    let exists_before = exists_in_store(&store, fingerprint_id).await;
    println!("\n[BEFORE DELETE] Source of Truth state:");
    println!("  - Total count: {}", count_before);
    println!(
        "  - Fingerprint {} exists: {}",
        fingerprint_id, exists_before
    );
    assert!(exists_before, "Fingerprint must exist before delete");

    // === EXECUTE HARD DELETE ===
    let delete_params = json!({
        "fingerprintId": fingerprint_id_str,
        "soft": false
    });
    let delete_request = make_request(
        "memory/delete",
        Some(JsonRpcId::Number(2)),
        Some(delete_params),
    );
    let delete_response = handlers.dispatch(delete_request).await;

    let delete_result = delete_response.result.expect("Should have result");
    println!("\n[OPERATION] Hard delete response:");
    println!("  - deleted: {}", delete_result.get("deleted").unwrap());
    println!(
        "  - deleteType: {}",
        delete_result.get("deleteType").unwrap()
    );

    // === AFTER DELETE - VERIFY SOURCE OF TRUTH ===
    let count_after = store.count().await.unwrap();
    let exists_after = exists_in_store(&store, fingerprint_id).await;
    let retrieve_after = store.retrieve(fingerprint_id).await.unwrap();

    println!("\n[AFTER DELETE] Source of Truth state:");
    println!("  - Total count: {}", count_after);
    println!(
        "  - Fingerprint {} exists: {}",
        fingerprint_id, exists_after
    );
    println!(
        "  - Direct retrieve returns: {:?}",
        retrieve_after.as_ref().map(|fp| fp.id)
    );

    // CRITICAL VERIFICATION: Fingerprint must be GONE from store
    assert!(
        !exists_after,
        "VERIFICATION FAILED: Fingerprint must NOT exist after hard delete"
    );
    assert!(
        retrieve_after.is_none(),
        "VERIFICATION FAILED: store.retrieve() must return None after hard delete"
    );
    assert_eq!(
        count_after,
        count_before - 1,
        "Count must decrease by 1 after delete"
    );

    println!("\n[VERIFICATION PASSED] Fingerprint removed from Source of Truth");
    println!("================================================================================\n");
}
