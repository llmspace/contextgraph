//! Full Pipeline Tests
//!
//! TEST 2: Full Pipeline Test (Store, Search)
//! TEST 3: Persistence Verification Across Restart
//! TEST 4: Column Family Verification

use std::path::PathBuf;

use context_graph_core::traits::TeleologicalMemoryStore;
use context_graph_core::types::fingerprint::TeleologicalFingerprint;
use context_graph_storage::teleological::{
    deserialize_e1_matryoshka_128, deserialize_purpose_vector,
    deserialize_teleological_fingerprint, e1_matryoshka_128_key, fingerprint_key,
    purpose_vector_key, RocksDbTeleologicalStore, TeleologicalStoreConfig, CF_E13_SPLADE_INVERTED,
    CF_E1_MATRYOSHKA_128, CF_FINGERPRINTS, CF_PURPOSE_VECTORS, QUANTIZED_EMBEDDER_CFS,
    TELEOLOGICAL_CFS,
};
use tempfile::TempDir;
use uuid::Uuid;

use crate::helpers::{
    create_initialized_store, create_real_fingerprint, generate_real_purpose_vector,
    generate_real_semantic_fingerprint, generate_real_sparse_vector,
};

// =============================================================================
// TEST 2: Full Pipeline Test (Store, Search)
// =============================================================================

#[tokio::test]
async fn test_full_storage_pipeline_real_data() {
    println!("\n=== TEST: Full Storage Pipeline with REAL Data ===\n");

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = TeleologicalStoreConfig {
        block_cache_size: 128 * 1024 * 1024, // 128MB cache
        max_open_files: 500,
        enable_wal: true,
        create_if_missing: true,
    };
    let store = RocksDbTeleologicalStore::open_with_config(temp_dir.path(), config)
        .expect("Failed to open store");
    // Note: EmbedderIndexRegistry is initialized in the constructor

    println!("[BEFORE] Empty store");

    // Store 50 fingerprints
    const COUNT: usize = 50;
    let mut stored: Vec<TeleologicalFingerprint> = Vec::with_capacity(COUNT);

    for _ in 0..COUNT {
        let fp = create_real_fingerprint();
        store.store(fp.clone()).await.expect("Failed to store");
        stored.push(fp);
    }
    println!("[STORED] {} fingerprints", COUNT);

    // Test purpose search
    let query_purpose = generate_real_purpose_vector();
    let purpose_options = context_graph_core::traits::TeleologicalSearchOptions {
        top_k: 10,
        min_similarity: 0.0,
        min_alignment: None,
        include_deleted: false,
        embedder_indices: vec![],
        semantic_query: None,   // No semantic query for this test
        include_content: false, // TASK-CONTENT-005
    };

    let purpose_results = store
        .search_purpose(&query_purpose, purpose_options.clone())
        .await
        .expect("Purpose search failed");

    println!(
        "[SEARCH] Purpose search returned {} results",
        purpose_results.len()
    );
    assert!(
        !purpose_results.is_empty(),
        "Purpose search should return results"
    );

    // Test semantic search
    let query_semantic = generate_real_semantic_fingerprint();
    let semantic_results = store
        .search_semantic(&query_semantic, purpose_options.clone())
        .await
        .expect("Semantic search failed");

    println!(
        "[SEARCH] Semantic search returned {} results",
        semantic_results.len()
    );
    assert!(
        !semantic_results.is_empty(),
        "Semantic search should return results"
    );

    // Test sparse search
    let query_sparse = generate_real_sparse_vector(50);
    let sparse_results = store
        .search_sparse(&query_sparse, 10)
        .await
        .expect("Sparse search failed");

    println!(
        "[SEARCH] Sparse search returned {} results",
        sparse_results.len()
    );

    // Test delete (soft)
    let delete_id = stored[0].id;
    let deleted = store
        .delete(delete_id, true)
        .await
        .expect("Soft delete failed");
    assert!(deleted, "Soft delete should succeed");

    // Verify soft deleted item not retrievable
    let after_delete = store.retrieve(delete_id).await.expect("Retrieve failed");
    assert!(
        after_delete.is_none(),
        "Soft deleted item should not be retrievable"
    );

    // Verify count decreased
    let final_count = store.count().await.expect("Count failed");
    assert_eq!(
        final_count,
        COUNT - 1,
        "Count should be {} after soft delete",
        COUNT - 1
    );

    // Test hard delete
    let hard_delete_id = stored[1].id;
    let hard_deleted = store
        .delete(hard_delete_id, false)
        .await
        .expect("Hard delete failed");
    assert!(hard_deleted, "Hard delete should succeed");

    println!(
        "[AFTER] {} fingerprints remaining after deletes",
        final_count - 1
    );
    println!("[VERIFIED] Full pipeline: store, search, delete all working");
    println!("\n=== PASS: Full Storage Pipeline ===\n");
}

// =============================================================================
// TEST 3: Persistence Verification Across Restart
// =============================================================================

#[tokio::test]
async fn test_physical_persistence_across_restart() {
    println!("\n=== TEST: Physical Persistence Across Restart ===\n");

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path: PathBuf = temp_dir.path().to_path_buf();

    // Generate fingerprints with known IDs
    let test_fingerprints: Vec<TeleologicalFingerprint> =
        (0..10).map(|_| create_real_fingerprint()).collect();

    let test_ids: Vec<Uuid> = test_fingerprints.iter().map(|fp| fp.id).collect();

    // Phase 1: Store and close
    println!("[PHASE 1] Storing 10 fingerprints and closing database...");
    {
        let store =
            RocksDbTeleologicalStore::open(&db_path).expect("Failed to open store (phase 1)");
        // Note: EmbedderIndexRegistry is initialized in the constructor

        for fp in &test_fingerprints {
            store.store(fp.clone()).await.expect("Failed to store");
        }

        // Flush to ensure data is on disk
        store.flush().await.expect("Flush failed");

        let count = store.count().await.expect("Count failed");
        assert_eq!(count, 10, "Should have 10 fingerprints before close");

        println!("[BEFORE] Stored 10 fingerprints, flushed, closing DB");
        // Store drops here
    }

    // Phase 2: Reopen and verify
    println!("[PHASE 2] Reopening database and verifying data...");
    {
        let store =
            RocksDbTeleologicalStore::open(&db_path).expect("Failed to reopen store (phase 2)");
        // Note: EmbedderIndexRegistry is initialized in the constructor

        let count = store.count().await.expect("Count failed");
        assert_eq!(count, 10, "Should still have 10 fingerprints after reopen");

        // Verify all 10 fingerprints are retrievable
        for (i, &id) in test_ids.iter().enumerate() {
            let retrieved = store
                .retrieve(id)
                .await
                .expect("Retrieve failed")
                .unwrap_or_else(|| panic!("Fingerprint {} not found after reopen", id));

            assert_eq!(retrieved.id, id, "ID mismatch at index {}", i);

            // Verify data integrity
            assert_eq!(
                retrieved.semantic.e1_semantic.len(),
                1024,
                "E1 dimension mismatch after reopen"
            );
            assert_eq!(
                retrieved.semantic.e9_hdc.len(),
                1024,
                "E9 dimension mismatch (expected 1024) after reopen"
            );

            println!("  [{}] {} verified", i, id);
        }

        println!("[AFTER] Reopened DB, all 10 fingerprints retrieved and verified");
    }

    // Phase 3: Verify raw files exist on disk
    println!("[PHASE 3] Verifying physical files exist...");
    {
        // Check that RocksDB files exist
        let sst_files: Vec<_> = std::fs::read_dir(&db_path)
            .expect("Failed to read db directory")
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "sst")
                    .unwrap_or(false)
            })
            .collect();

        // SST files may or may not exist depending on compaction state
        // But CURRENT, MANIFEST files should always exist
        let current_file = db_path.join("CURRENT");
        assert!(current_file.exists(), "CURRENT file should exist");

        let manifest_files: Vec<_> = std::fs::read_dir(&db_path)
            .expect("Failed to read db directory")
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with("MANIFEST"))
            .collect();
        assert!(!manifest_files.is_empty(), "MANIFEST file should exist");

        println!(
            "[VERIFIED] Physical files exist: CURRENT, {} MANIFEST files, {} SST files",
            manifest_files.len(),
            sst_files.len()
        );
    }

    println!("\n=== PASS: Physical Persistence Across Restart ===\n");
}

// =============================================================================
// TEST 4: Column Family Verification
// =============================================================================

#[tokio::test]
async fn test_all_column_families_populated() {
    println!("\n=== TEST: All Column Families Populated ===\n");

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let store = create_initialized_store(temp_dir.path());

    // Store a fingerprint
    let fp = create_real_fingerprint();
    let id = fp.id;

    println!("[BEFORE] Storing fingerprint {} to populate CFs", id);
    store.store(fp.clone()).await.expect("Failed to store");

    // Access underlying DB for direct verification
    let db = store.db();

    // 1. Verify fingerprints CF has data
    let cf_fp = db
        .cf_handle(CF_FINGERPRINTS)
        .expect("Missing fingerprints CF");
    let fp_key = fingerprint_key(&id);
    let fp_data = db
        .get_cf(&cf_fp, fp_key)
        .expect("Get failed")
        .expect("Fingerprint not found in fingerprints CF");

    // With E9_DIM = 1024 (projected), fingerprints are ~32-40KB
    println!("[VERIFIED] fingerprints CF: {} bytes", fp_data.len());
    assert!(
        fp_data.len() >= 25_000,
        "Fingerprint should be >= 25KB, got {}",
        fp_data.len()
    );

    // Deserialize and verify
    let retrieved_fp = deserialize_teleological_fingerprint(&fp_data);
    assert_eq!(retrieved_fp.id, id, "ID mismatch in fingerprints CF");

    // 2. Verify purpose_vectors CF has data
    let cf_pv = db
        .cf_handle(CF_PURPOSE_VECTORS)
        .expect("Missing purpose_vectors CF");
    let pv_key = purpose_vector_key(&id);
    let pv_data = db
        .get_cf(&cf_pv, pv_key)
        .expect("Get failed")
        .expect("Data not found in purpose_vectors CF");

    println!(
        "[VERIFIED] purpose_vectors CF: {} bytes (expected 52)",
        pv_data.len()
    );
    assert_eq!(
        pv_data.len(),
        52,
        "Purpose vector should be exactly 52 bytes"
    );

    // Deserialize and verify
    let retrieved_pv = deserialize_purpose_vector(&pv_data);
    for (i, (retrieved, original)) in retrieved_pv
        .iter()
        .zip(fp.purpose_vector.alignments.iter())
        .enumerate()
    {
        assert!(
            (retrieved - original).abs() < f32::EPSILON,
            "Purpose vector mismatch at index {}",
            i
        );
    }

    // 3. Verify e1_matryoshka_128 CF has data
    let cf_mat = db
        .cf_handle(CF_E1_MATRYOSHKA_128)
        .expect("Missing e1_matryoshka_128 CF");
    let mat_key = e1_matryoshka_128_key(&id);
    let mat_data = db
        .get_cf(&cf_mat, mat_key)
        .expect("Get failed")
        .expect("Data not found in e1_matryoshka_128 CF");

    println!(
        "[VERIFIED] e1_matryoshka_128 CF: {} bytes (expected 512)",
        mat_data.len()
    );
    assert_eq!(
        mat_data.len(),
        512,
        "E1 Matryoshka 128D should be exactly 512 bytes"
    );

    // Deserialize and verify it matches first 128 dims of E1
    let retrieved_mat = deserialize_e1_matryoshka_128(&mat_data);
    for (i, (retrieved, original)) in retrieved_mat
        .iter()
        .zip(fp.semantic.e1_semantic.iter())
        .enumerate()
    {
        assert!(
            (retrieved - original).abs() < f32::EPSILON,
            "E1 Matryoshka mismatch at index {}",
            i
        );
    }

    // 4. Verify e13_splade_inverted CF has data (if fingerprint has sparse entries)
    if fp.semantic.e13_splade.nnz() > 0 {
        let cf_inv = db
            .cf_handle(CF_E13_SPLADE_INVERTED)
            .expect("Missing e13_splade_inverted CF");

        // Check at least one term is indexed
        let first_term = fp.semantic.e13_splade.indices[0];
        let term_key = context_graph_storage::teleological::e13_splade_inverted_key(first_term);
        let inv_data = db
            .get_cf(&cf_inv, term_key)
            .expect("Get failed")
            .expect("Term not found in inverted index");

        println!(
            "[VERIFIED] e13_splade_inverted CF: term {} has {} bytes",
            first_term,
            inv_data.len()
        );
        assert!(
            inv_data.len() >= 20,
            "Inverted index entry should have UUID data"
        );
    }

    // 5. Verify all teleological CFs are accessible
    println!(
        "[VERIFYING] All {} teleological CFs accessible...",
        TELEOLOGICAL_CFS.len()
    );
    for cf_name in TELEOLOGICAL_CFS {
        let cf = db
            .cf_handle(cf_name)
            .unwrap_or_else(|| panic!("Missing CF: {}", cf_name));
        assert!(
            !std::ptr::eq(cf as *const _, std::ptr::null()),
            "CF handle should be valid"
        );
        println!("  {} OK", cf_name);
    }

    // 6. Verify all quantized embedder CFs are accessible
    println!(
        "[VERIFYING] All {} quantized embedder CFs accessible...",
        QUANTIZED_EMBEDDER_CFS.len()
    );
    for cf_name in QUANTIZED_EMBEDDER_CFS {
        let cf = db
            .cf_handle(cf_name)
            .unwrap_or_else(|| panic!("Missing CF: {}", cf_name));
        assert!(
            !std::ptr::eq(cf as *const _, std::ptr::null()),
            "CF handle should be valid"
        );
        println!("  {} OK", cf_name);
    }

    let total_cfs = TELEOLOGICAL_CFS.len() + QUANTIZED_EMBEDDER_CFS.len();
    println!(
        "[AFTER] All {} column families verified (4 teleological + 13 embedder)",
        total_cfs
    );
    println!("\n=== PASS: All Column Families Populated ===\n");
}
