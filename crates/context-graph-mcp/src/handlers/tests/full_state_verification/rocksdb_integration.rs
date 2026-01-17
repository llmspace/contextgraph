//! RocksDB Full State Verification Integration Tests.
//!
//! These tests use REAL RocksDbTeleologicalStore with tempdir for full state verification.
//! They directly inspect the persistent store to verify data was actually written.

use serde_json::json;
use sha2::{Digest, Sha256};

use crate::protocol::JsonRpcId;

use super::super::create_test_handlers_with_rocksdb_store_access;
use super::helpers::{exists_in_store, make_request};

// =============================================================================
// ROCKSDB FULL STATE VERIFICATION INTEGRATION TESTS
// =============================================================================

/// FSV-ROCKSDB-001: Store creates fingerprint in REAL RocksDB store.
///
/// Verifies that memory/store operation:
/// 1. Starts with empty store (count = 0)
/// 2. Stores fingerprint via MCP handler
/// 3. Fingerprint physically exists in RocksDB
/// 4. Content hash matches SHA-256 of input
#[tokio::test]
async fn test_rocksdb_fsv_store_creates_fingerprint() {
    println!("\n================================================================================");
    println!("FSV-ROCKSDB-001: Store Creates Fingerprint in REAL RocksDB");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;
    // _tempdir MUST stay alive for the duration of this test

    // === BEFORE STATE ===
    let count_before = store.count().await.expect("count() should work on RocksDB");
    println!("\n[BEFORE] RocksDB store state:");
    println!("  - Fingerprint count: {}", count_before);
    assert_eq!(count_before, 0, "RocksDB store should be empty initially");

    // === EXECUTE OPERATION ===
    let content = "Machine learning is transforming software development";
    let params = json!({
        "content": content,
        "importance": 0.9
    });
    let request = make_request("memory/store", Some(JsonRpcId::Number(1)), Some(params));
    let response = handlers.dispatch(request).await;

    assert!(
        response.error.is_none(),
        "Store must succeed: {:?}",
        response.error
    );

    let result = response.result.expect("Should have result");
    let fingerprint_id_str = result
        .get("fingerprintId")
        .expect("Should have fingerprintId")
        .as_str()
        .expect("Should be string");
    let fingerprint_id = uuid::Uuid::parse_str(fingerprint_id_str).expect("Should be valid UUID");

    println!("\n[OPERATION] Stored content with ID: {}", fingerprint_id);

    // === AFTER STATE - VERIFY SOURCE OF TRUTH IN ROCKSDB ===
    let count_after = store.count().await.expect("count() should work on RocksDB");
    println!("\n[AFTER] RocksDB store state:");
    println!("  - Fingerprint count: {}", count_after);
    assert_eq!(count_after, count_before + 1, "Count must increase by 1");

    // CRITICAL: Directly verify fingerprint exists in RocksDB
    let exists = exists_in_store(&store, fingerprint_id).await;
    println!(
        "  - Fingerprint {} exists in RocksDB: {}",
        fingerprint_id, exists
    );
    assert!(
        exists,
        "VERIFICATION FAILED: Fingerprint must exist in RocksDB store"
    );

    // CRITICAL: Retrieve and inspect actual stored data from RocksDB
    let stored_fp = store
        .retrieve(fingerprint_id)
        .await
        .expect("retrieve() should work on RocksDB")
        .expect("Fingerprint must exist");

    println!("\n[EVIDENCE] Stored fingerprint fields from RocksDB:");
    println!("  - ID: {}", stored_fp.id);
    println!("  - alignment_score: {:.4}", stored_fp.alignment_score);
    println!("  - access_count: {}", stored_fp.access_count);
    println!("  - content_hash: {}", hex::encode(stored_fp.content_hash));
    println!(
        "  - semantic.e1_semantic len: {}",
        stored_fp.semantic.e1_semantic.len()
    );

    // Verify content hash matches expected SHA-256
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let expected_hash: [u8; 32] = hasher.finalize().into();
    assert_eq!(
        stored_fp.content_hash, expected_hash,
        "Content hash in RocksDB must match SHA-256 of original content"
    );
    println!("  - Content hash VERIFIED: matches SHA-256 of input");

    // Verify semantic fingerprint has valid embedding
    assert!(
        !stored_fp.semantic.e1_semantic.is_empty(),
        "E1 must have embeddings"
    );
    println!(
        "  - E1 semantic embedding dimension: {}",
        stored_fp.semantic.e1_semantic.len()
    );

    println!("\n[FSV-ROCKSDB-001 PASSED] Fingerprint physically exists in RocksDB");
    println!("================================================================================\n");
}

/// FSV-ROCKSDB-002: Retrieve returns data from REAL RocksDB store.
///
/// Verifies that memory/retrieve operation:
/// 1. Stores fingerprint to RocksDB
/// 2. Retrieves via MCP handler
/// 3. Returned data matches what was stored in RocksDB
#[tokio::test]
async fn test_rocksdb_fsv_retrieve_returns_stored_data() {
    println!("\n================================================================================");
    println!("FSV-ROCKSDB-002: Retrieve Returns Data from REAL RocksDB");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store a fingerprint first
    let content = "Neural networks process information in layers";
    let params = json!({
        "content": content,
        "importance": 0.85
    });
    let store_request = make_request("memory/store", Some(JsonRpcId::Number(1)), Some(params));
    let store_response = handlers.dispatch(store_request).await;

    let store_result = store_response.result.expect("Store should succeed");
    let fingerprint_id_str = store_result.get("fingerprintId").unwrap().as_str().unwrap();
    let fingerprint_id = uuid::Uuid::parse_str(fingerprint_id_str).unwrap();

    println!("\n[SETUP] Stored fingerprint with ID: {}", fingerprint_id);

    // Verify it exists in RocksDB directly
    let stored_fp = store
        .retrieve(fingerprint_id)
        .await
        .expect("retrieve() should work")
        .expect("Must exist in RocksDB");
    println!("[VERIFY] Fingerprint confirmed in RocksDB store");

    // === EXECUTE RETRIEVE OPERATION ===
    let retrieve_params = json!({ "fingerprintId": fingerprint_id_str });
    let retrieve_request = make_request(
        "memory/retrieve",
        Some(JsonRpcId::Number(2)),
        Some(retrieve_params),
    );
    let retrieve_response = handlers.dispatch(retrieve_request).await;

    assert!(
        retrieve_response.error.is_none(),
        "Retrieve must succeed: {:?}",
        retrieve_response.error
    );

    let result = retrieve_response.result.expect("Should have result");
    let fingerprint = result
        .get("fingerprint")
        .expect("Must have fingerprint object");

    // === VERIFY RETRIEVED DATA MATCHES ROCKSDB ===
    let retrieved_id = fingerprint
        .get("id")
        .and_then(|v| v.as_str())
        .expect("Must have id");
    assert_eq!(
        retrieved_id, fingerprint_id_str,
        "Retrieved ID must match stored ID"
    );

    // Verify content_hash matches what's in RocksDB
    let retrieved_hash = fingerprint
        .get("contentHashHex")
        .and_then(|v| v.as_str())
        .expect("Must have contentHashHex");
    let expected_hash_hex = hex::encode(stored_fp.content_hash);
    assert_eq!(
        retrieved_hash, expected_hash_hex,
        "Retrieved contentHashHex must match RocksDB store"
    );

    println!("\n[EVIDENCE] Retrieved data matches RocksDB:");
    println!("  - ID: {}", retrieved_id);
    println!("  - contentHashHex: {}", retrieved_hash);

    println!("\n[FSV-ROCKSDB-002 PASSED] Retrieve returns data matching RocksDB");
    println!("================================================================================\n");
}

/// FSV-ROCKSDB-003: Delete removes fingerprint from REAL RocksDB store.
///
/// Verifies that memory/delete operation:
/// 1. Stores fingerprint to RocksDB
/// 2. Verifies it exists
/// 3. Deletes via MCP handler
/// 4. Fingerprint is physically gone from RocksDB
#[tokio::test]
async fn test_rocksdb_fsv_delete_removes_from_store() {
    println!("\n================================================================================");
    println!("FSV-ROCKSDB-003: Delete Removes Fingerprint from REAL RocksDB");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    // Store a fingerprint
    let content = "Deep learning requires substantial computational resources";
    let params = json!({ "content": content, "importance": 0.75 });
    let store_request = make_request("memory/store", Some(JsonRpcId::Number(1)), Some(params));
    let store_response = handlers.dispatch(store_request).await;

    let store_result = store_response.result.expect("Store should succeed");
    let fingerprint_id_str = store_result.get("fingerprintId").unwrap().as_str().unwrap();
    let fingerprint_id = uuid::Uuid::parse_str(fingerprint_id_str).unwrap();

    println!("\n[SETUP] Stored fingerprint with ID: {}", fingerprint_id);

    // Verify it exists BEFORE delete
    let exists_before = exists_in_store(&store, fingerprint_id).await;
    assert!(
        exists_before,
        "Fingerprint must exist in RocksDB before delete"
    );
    println!(
        "[BEFORE DELETE] Fingerprint exists in RocksDB: {}",
        exists_before
    );

    let count_before = store.count().await.expect("count() should work");
    println!("  - Store count: {}", count_before);

    // === EXECUTE DELETE OPERATION ===
    let delete_params = json!({ "fingerprintId": fingerprint_id_str, "soft": false });
    let delete_request = make_request(
        "memory/delete",
        Some(JsonRpcId::Number(2)),
        Some(delete_params),
    );
    let delete_response = handlers.dispatch(delete_request).await;

    assert!(
        delete_response.error.is_none(),
        "Delete must succeed: {:?}",
        delete_response.error
    );

    let delete_result = delete_response.result.expect("Should have result");
    let deleted = delete_result
        .get("deleted")
        .and_then(|v| v.as_bool())
        .expect("Must have deleted flag");
    assert!(deleted, "Response must indicate deletion succeeded");

    // === VERIFY FINGERPRINT IS GONE FROM ROCKSDB ===
    let exists_after = exists_in_store(&store, fingerprint_id).await;
    println!(
        "\n[AFTER DELETE] Fingerprint exists in RocksDB: {}",
        exists_after
    );
    assert!(
        !exists_after,
        "VERIFICATION FAILED: Fingerprint must be removed from RocksDB"
    );

    let count_after = store.count().await.expect("count() should work");
    println!("  - Store count: {}", count_after);
    assert_eq!(count_after, count_before - 1, "Count must decrease by 1");

    // Double-check with retrieve - should return None
    let retrieved = store
        .retrieve(fingerprint_id)
        .await
        .expect("retrieve() should work");
    assert!(
        retrieved.is_none(),
        "Retrieve must return None for deleted fingerprint"
    );

    println!("\n[FSV-ROCKSDB-003 PASSED] Fingerprint physically removed from RocksDB");
    println!("================================================================================\n");
}

/// FSV-ROCKSDB-004: Multiple fingerprints in REAL RocksDB store.
///
/// Verifies that multiple store operations:
/// 1. Each creates a unique fingerprint
/// 2. All fingerprints exist in RocksDB
/// 3. Count reflects actual stored items
#[tokio::test]
async fn test_rocksdb_fsv_multiple_fingerprints() {
    println!("\n================================================================================");
    println!("FSV-ROCKSDB-004: Multiple Fingerprints in REAL RocksDB");
    println!("================================================================================");

    let (handlers, store, _tempdir) = create_test_handlers_with_rocksdb_store_access().await;

    let contents = [
        "First document about machine learning",
        "Second document about neural networks",
        "Third document about deep learning",
    ];

    let mut stored_ids = Vec::new();

    // Store multiple fingerprints
    for (i, content) in contents.iter().enumerate() {
        let params = json!({ "content": content, "importance": 0.8 });
        let request = make_request(
            "memory/store",
            Some(JsonRpcId::Number(i as i64 + 1)),
            Some(params),
        );
        let response = handlers.dispatch(request).await;

        let result = response.result.expect("Store should succeed");
        let id_str = result.get("fingerprintId").unwrap().as_str().unwrap();
        let id = uuid::Uuid::parse_str(id_str).unwrap();
        stored_ids.push(id);

        println!("[STORED {}] ID: {}", i + 1, id);
    }

    // === VERIFY ALL FINGERPRINTS EXIST IN ROCKSDB ===
    println!("\n[VERIFICATION] Checking all fingerprints in RocksDB:");

    let count = store.count().await.expect("count() should work");
    assert_eq!(count, contents.len(), "Count must match stored items");
    println!("  - Total count: {} (expected: {})", count, contents.len());

    for (i, id) in stored_ids.iter().enumerate() {
        let exists = exists_in_store(&store, *id).await;
        assert!(exists, "Fingerprint {} must exist in RocksDB", i + 1);

        let fp = store
            .retrieve(*id)
            .await
            .expect("retrieve() should work")
            .expect("Must exist");
        println!(
            "  - Fingerprint {} exists: {} (hash: {})",
            i + 1,
            exists,
            hex::encode(&fp.content_hash[..8])
        );
    }

    println!(
        "\n[FSV-ROCKSDB-004 PASSED] All {} fingerprints verified in RocksDB",
        contents.len()
    );
    println!("================================================================================\n");
}
