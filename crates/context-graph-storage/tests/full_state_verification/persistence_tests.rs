//! Persistence Tests
//!
//! Tests for update/delete operations and database reopen.

use context_graph_core::traits::TeleologicalMemoryStore;
use context_graph_storage::teleological::{
    deserialize_teleological_fingerprint, fingerprint_key, purpose_vector_key,
    RocksDbTeleologicalStore, CF_FINGERPRINTS, CF_PURPOSE_VECTORS,
};
use tempfile::TempDir;
use uuid::Uuid;

use crate::helpers::{create_test_store, generate_real_teleological_fingerprint, hex_string};

/// Test 7: Update and Delete Physical Verification
///
/// Tests that updates and deletes are physically reflected in RocksDB.
#[tokio::test]
async fn test_update_delete_physical_verification() {
    println!("\n================================================================================");
    println!("FULL STATE VERIFICATION: Update and Delete");
    println!("================================================================================\n");

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let store = create_test_store(&temp_dir);

    let id = Uuid::new_v4();
    let mut fingerprint = generate_real_teleological_fingerprint(id);

    // Initial store
    let original_alignments = fingerprint.purpose_vector.alignments;
    store
        .store(fingerprint.clone())
        .await
        .expect("Initial store failed");

    println!("[1] Initial store:");
    println!("    Alignments[0]: {:.6}", original_alignments[0]);

    // Verify initial state
    let key = fingerprint_key(&id);
    let raw1 = store
        .get_raw_bytes(CF_FINGERPRINTS, &key)
        .expect("Read failed")
        .expect("Not found");
    let retrieved1 = deserialize_teleological_fingerprint(&raw1);

    println!("[2] Physical verification of initial state:");
    println!(
        "    Alignments[0]: {:.6}",
        retrieved1.purpose_vector.alignments[0]
    );
    assert!((retrieved1.purpose_vector.alignments[0] - original_alignments[0]).abs() < 0.0001);

    // UPDATE: Change purpose vector
    fingerprint.purpose_vector.alignments[0] = 0.999;
    store
        .update(fingerprint.clone())
        .await
        .expect("Update failed");

    println!("[3] Update applied:");
    println!(
        "    New alignments[0]: {:.6}",
        fingerprint.purpose_vector.alignments[0]
    );

    // Physical verification after update
    let raw2 = store
        .get_raw_bytes(CF_FINGERPRINTS, &key)
        .expect("Read failed")
        .expect("Not found after update");
    let retrieved2 = deserialize_teleological_fingerprint(&raw2);

    println!("[4] Physical verification after update:");
    println!(
        "    Alignments[0]: {:.6}",
        retrieved2.purpose_vector.alignments[0]
    );

    assert!((retrieved2.purpose_vector.alignments[0] - 0.999).abs() < 0.001);

    // DELETE (hard delete)
    store.delete(id, false).await.expect("Delete failed");
    println!("[5] Delete executed");

    // Physical verification after delete
    let raw3 = store
        .get_raw_bytes(CF_FINGERPRINTS, &key)
        .expect("Read failed");

    println!("[6] Physical verification after delete:");
    println!("    Data exists: {}", raw3.is_some());

    assert!(raw3.is_none(), "Fingerprint still exists after delete!");

    // Verify purpose vector CF also deleted
    let purpose_key = purpose_vector_key(&id);
    let raw_purpose = store
        .get_raw_bytes(CF_PURPOSE_VECTORS, &purpose_key)
        .expect("Read failed");

    println!("[7] Purpose vector CF after delete:");
    println!("    Data exists: {}", raw_purpose.is_some());

    assert!(raw_purpose.is_none(), "Purpose vector still exists!");

    println!("\n[PASS] Update and delete physical verification successful");
    println!("================================================================================\n");
}

/// Test 9: Persistence Across DB Reopen
///
/// Verifies data survives database close and reopen.
#[tokio::test]
async fn test_persistence_across_reopen() {
    println!("\n================================================================================");
    println!("FULL STATE VERIFICATION: Persistence Across Reopen");
    println!("================================================================================\n");

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let path = temp_dir.path().to_path_buf();

    let id = Uuid::new_v4();
    let fingerprint = generate_real_teleological_fingerprint(id);
    let original_alignments = fingerprint.purpose_vector.alignments;

    // First session: store data
    {
        let store = RocksDbTeleologicalStore::open(&path).expect("Failed to open store");
        // Note: EmbedderIndexRegistry is initialized in constructor

        store.store(fingerprint).await.expect("Store failed");
        println!("[1] First session: stored fingerprint {}", id);

        // Explicit drop to close DB
        drop(store);
        println!("[2] First session: database closed");
    }

    // Second session: reopen and verify
    {
        let store = RocksDbTeleologicalStore::open(&path).expect("Failed to reopen store");
        // Note: EmbedderIndexRegistry is initialized in constructor

        println!("[3] Second session: database reopened");

        // Physical verification
        let key = fingerprint_key(&id);
        let raw = store
            .get_raw_bytes(CF_FINGERPRINTS, &key)
            .expect("Read failed");

        assert!(raw.is_some(), "Data lost after reopen!");
        let bytes = raw.unwrap();

        let retrieved = deserialize_teleological_fingerprint(&bytes);

        println!("[4] Physical verification after reopen:");
        println!("    ID match: {}", retrieved.id == id);
        println!(
            "    Alignments match: {}",
            retrieved.purpose_vector.alignments == original_alignments
        );

        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.purpose_vector.alignments, original_alignments);
    }

    println!("\n[PASS] Persistence verification successful");
    println!("================================================================================\n");
}
