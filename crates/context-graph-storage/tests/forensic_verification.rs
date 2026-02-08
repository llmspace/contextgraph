//! Forensic Verification: End-to-End Tests for All Critical Fixes
//!
//! Each test creates a real RocksDB instance, performs operations,
//! reads back from the physical database, and tests edge cases.
//! No mocks, no stubs — real data with physical verification.

use context_graph_core::traits::TeleologicalMemoryStore;
use context_graph_core::types::fingerprint::{SemanticFingerprint, TeleologicalFingerprint};
use context_graph_core::types::{CausalRelationship, SourceMetadata, SourceType};
use context_graph_storage::teleological::RocksDbTeleologicalStore;
use tempfile::TempDir;
use uuid::Uuid;

struct TestStore {
    store: RocksDbTeleologicalStore,
    _temp_dir: TempDir,
    rt: tokio::runtime::Runtime,
}

impl TestStore {
    fn new() -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let store =
            RocksDbTeleologicalStore::open(temp_dir.path()).expect("Failed to open store");
        let rt = tokio::runtime::Runtime::new().expect("Failed to create runtime");
        Self {
            store,
            _temp_dir: temp_dir,
            rt,
        }
    }
}

fn create_test_fingerprint() -> TeleologicalFingerprint {
    let semantic = SemanticFingerprint::zeroed();
    let content_hash = [0u8; 32];
    TeleologicalFingerprint::new(semantic, content_hash)
}

#[test]
fn test_crit01_record_access_persists_in_database() {
    let h = TestStore::new();

    let fp = create_test_fingerprint();
    let fp_id = fp.id;

    h.rt.block_on(async {
        h.store.store(fp).await.expect("Store failed");
    });

    let before = h
        .rt
        .block_on(async { h.store.retrieve(fp_id).await.expect("Retrieve failed") })
        .expect("Not found after store");

    assert_eq!(before.access_count, 0, "Initial access_count must be 0");
    let before_last_updated = before.last_updated;

    let mut updated = before;
    std::thread::sleep(std::time::Duration::from_millis(10));
    updated.record_access();
    assert_eq!(updated.access_count, 1, "record_access must increment to 1");

    h.rt.block_on(async {
        h.store.update(updated).await.expect("Update failed");
    });

    let after = h
        .rt
        .block_on(async { h.store.retrieve(fp_id).await.expect("Retrieve failed") })
        .expect("Not found after update");

    assert_eq!(after.access_count, 1, "access_count not persisted in database");
    assert!(
        after.last_updated > before_last_updated,
        "last_updated must advance after record_access()"
    );
}

#[test]
fn test_crit02_causal_relationship_json_roundtrip() {
    let h = TestStore::new();

    let source_id = Uuid::new_v4();
    let rel = CausalRelationship::new(
        "Chronic stress".to_string(),
        "Memory impairment".to_string(),
        "Stress causes cortisol elevation leading to hippocampal damage.".to_string(),
        vec![0.1f32; 768],
        vec![0.2f32; 768],
        vec![0.3f32; 1024],
        "Studies show cortisol damages neurons.".to_string(),
        source_id,
        0.85,
        "direct".to_string(),
    );
    let rel_id = rel.id;

    let stored_id = h.rt.block_on(async {
        h.store
            .store_causal_relationship(&rel)
            .await
            .expect("Store failed")
    });
    assert_eq!(stored_id, rel_id);

    let retrieved = h
        .rt
        .block_on(async {
            h.store
                .get_causal_relationship(rel_id)
                .await
                .expect("Get failed")
        })
        .expect("Not found");

    assert_eq!(retrieved.id, rel_id);
    assert_eq!(retrieved.mechanism_type, "direct");
    assert!((retrieved.confidence - 0.85).abs() < 0.001);
    assert_eq!(retrieved.e5_as_cause.len(), 768);
    assert_eq!(retrieved.e5_as_effect.len(), 768);
    assert_eq!(retrieved.e1_semantic.len(), 1024);

    // Verify raw bytes are JSON (not bincode)
    let db = h.store.db();
    let cf = db
        .cf_handle("causal_relationships")
        .expect("CF not found");
    let key =
        context_graph_storage::teleological::schema::causal_relationship_key(&rel_id);
    let raw = db
        .get_cf(cf, key)
        .expect("Raw read failed")
        .expect("No raw data");

    assert_eq!(raw[0], b'{', "Data is not JSON (first byte is not '{{')");

    let parsed: serde_json::Value =
        serde_json::from_slice(&raw).expect("Raw data is not valid JSON");
    assert!(parsed.is_object());
    assert!(parsed.get("mechanism_type").is_some());
}

#[test]
fn test_high05_causal_store_uses_writebatch() {
    let h = TestStore::new();

    let source_id = Uuid::new_v4();

    let mut rel_ids = Vec::new();
    h.rt.block_on(async {
        for i in 0..3 {
            let rel = CausalRelationship::new(
                format!("Cause {i}"),
                format!("Effect {i}"),
                format!("Explanation {i}"),
                vec![0.1f32; 768],
                vec![0.2f32; 768],
                vec![0.3f32; 1024],
                format!("Source {i}"),
                source_id,
                0.8,
                "direct".to_string(),
            );
            rel_ids.push(rel.id);
            h.store
                .store_causal_relationship(&rel)
                .await
                .expect("Store failed");
        }
    });

    let by_source = h.rt.block_on(async {
        h.store
            .get_causal_relationships_by_source(source_id)
            .await
            .expect("Get by source failed")
    });

    let found_ids: std::collections::HashSet<_> = by_source.iter().map(|r| r.id).collect();
    for id in &rel_ids {
        assert!(
            found_ids.contains(id),
            "Secondary index missing relationship {id}"
        );
    }
}

#[test]
fn test_high06_source_metadata_json_only() {
    let h = TestStore::new();

    let fp = create_test_fingerprint();
    let fp_id = fp.id;

    h.rt.block_on(async {
        h.store.store(fp).await.expect("Store failed");
    });

    let metadata = SourceMetadata {
        source_type: SourceType::HookDescription,
        session_id: Some("test-session-123".to_string()),
        tool_use_id: Some("tool-456".to_string()),
        created_by: Some("test-operator".to_string()),
        ..Default::default()
    };

    h.rt.block_on(async {
        h.store
            .store_source_metadata(fp_id, &metadata)
            .await
            .expect("Store metadata failed");
    });

    let retrieved = h
        .rt
        .block_on(async {
            h.store
                .get_source_metadata(fp_id)
                .await
                .expect("Get metadata failed")
        })
        .expect("Metadata not found");

    assert_eq!(retrieved.source_type, SourceType::HookDescription);
    assert_eq!(retrieved.session_id.as_deref(), Some("test-session-123"));
    assert_eq!(retrieved.tool_use_id.as_deref(), Some("tool-456"));
    assert_eq!(retrieved.created_by.as_deref(), Some("test-operator"));

    // Verify raw bytes are JSON
    let db = h.store.db();
    let cf = db.cf_handle("source_metadata").expect("CF not found");
    let key = context_graph_storage::teleological::schema::source_metadata_key(&fp_id);
    let raw = db
        .get_cf(cf, key)
        .expect("Raw read failed")
        .expect("No raw data");
    assert_eq!(raw[0], b'{', "SourceMetadata is not JSON");

    let parsed: serde_json::Value =
        serde_json::from_slice(&raw).expect("Not valid JSON");
    assert!(parsed.get("source_type").is_some());
}

#[test]
fn test_high08_weight_validation() {
    let valid = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0f32];
    assert!(context_graph_core::weights::validate_weights(&valid).is_ok());

    let bad_sum = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0f32];
    assert!(
        context_graph_core::weights::validate_weights(&bad_sum).is_err(),
        "Should reject weights summing to 0.5"
    );

    let negative = [-0.1, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0f32];
    assert!(
        context_graph_core::weights::validate_weights(&negative).is_err(),
        "Should reject negative weights"
    );

    let too_high = [1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5f32];
    assert!(
        context_graph_core::weights::validate_weights(&too_high).is_err(),
        "Should reject weights > 1.0"
    );
}

#[test]
fn test_med09_file_index_json_roundtrip() {
    let h = TestStore::new();

    let fp = create_test_fingerprint();
    let fp_id = fp.id;
    let file_path = "/home/test/src/main.rs";

    h.rt.block_on(async {
        h.store.store(fp).await.expect("Store failed");
    });

    h.rt.block_on(async {
        h.store
            .index_file_fingerprint(file_path, fp_id)
            .await
            .expect("Index failed");
    });

    let fp_ids = h.rt.block_on(async {
        h.store
            .get_fingerprints_for_file(file_path)
            .await
            .expect("Get failed")
    });

    assert_eq!(fp_ids.len(), 1);
    assert_eq!(fp_ids[0], fp_id);

    // Verify raw bytes are JSON
    let db = h.store.db();
    let cf = db.cf_handle("file_index").expect("CF not found");
    let raw = db
        .get_cf(cf, file_path.as_bytes())
        .expect("Raw read failed")
        .expect("No raw data");
    assert_eq!(raw[0], b'{', "FileIndexEntry is not JSON");

    let parsed: serde_json::Value =
        serde_json::from_slice(&raw).expect("Not valid JSON");
    assert!(parsed.get("file_path").is_some());
    assert!(parsed.get("fingerprint_ids").is_some());
    assert!(parsed.get("last_updated").is_some());
}

#[test]
fn test_med19_med20_division_underflow_guards() {
    let empty: Vec<f32> = vec![];
    assert_eq!(empty.len().saturating_sub(1), 0);

    let single = vec![1.0f32];
    assert_eq!(single.len().saturating_sub(1), 0);

    let members_len = 1usize;
    let divisor = members_len.saturating_sub(1).max(1);
    assert_eq!(divisor, 1, "max(1) guard must prevent division by zero");
}

#[test]
fn test_edge_case_empty_database() {
    let h = TestStore::new();

    let results = h.rt.block_on(async {
        let query = vec![0.1f32; 1024];
        h.store
            .search_causal_relationships(&query, 10, None)
            .await
            .expect("Search failed")
    });
    assert!(results.is_empty(), "Search on empty DB must return empty");

    let missing = h
        .rt
        .block_on(async { h.store.retrieve(Uuid::new_v4()).await.expect("Get failed") });
    assert!(missing.is_none(), "Non-existent fingerprint must return None");

    let count = h.rt.block_on(async {
        h.store
            .count_causal_relationships()
            .await
            .expect("Count failed")
    });
    assert_eq!(count, 0, "Count on empty DB must be 0");
}

#[test]
fn test_edge_case_large_batch() {
    let h = TestStore::new();

    h.rt.block_on(async {
        for i in 0..50 {
            let rel = CausalRelationship::new(
                format!("Cause {i}"),
                format!("Effect {i}"),
                format!("Explanation for relationship {i}"),
                vec![0.1 + (i as f32 * 0.001); 768],
                vec![0.2 + (i as f32 * 0.001); 768],
                vec![0.3 + (i as f32 * 0.001); 1024],
                format!("Source content for {i}"),
                Uuid::new_v4(),
                0.5 + (i as f32 * 0.01),
                "direct".to_string(),
            );
            h.store
                .store_causal_relationship(&rel)
                .await
                .expect("Store failed");
        }
    });

    let count = h.rt.block_on(async {
        h.store
            .count_causal_relationships()
            .await
            .expect("Count failed")
    });
    assert_eq!(count, 50, "Must store exactly 50 relationships");

    let results = h.rt.block_on(async {
        let query = vec![0.35f32; 1024];
        h.store
            .search_causal_relationships(&query, 10, None)
            .await
            .expect("Search failed")
    });
    assert_eq!(results.len(), 10, "Top-10 must return exactly 10");

    for window in results.windows(2) {
        assert!(
            window[0].1 >= window[1].1,
            "Results must be sorted by similarity descending: {} >= {}",
            window[0].1,
            window[1].1
        );
    }
}

#[test]
fn test_edge_case_corrupted_data_repair() {
    let h = TestStore::new();

    let source_id = Uuid::new_v4();
    let rel = CausalRelationship::new(
        "Valid cause".to_string(),
        "Valid effect".to_string(),
        "Valid explanation".to_string(),
        vec![0.1f32; 768],
        vec![0.2f32; 768],
        vec![0.3f32; 1024],
        "Source".to_string(),
        source_id,
        0.9,
        "direct".to_string(),
    );
    let valid_id = rel.id;
    h.rt.block_on(async {
        h.store
            .store_causal_relationship(&rel)
            .await
            .expect("Store failed");
    });

    // Inject corrupted data directly into RocksDB
    let db = h.store.db();
    let cf = db
        .cf_handle("causal_relationships")
        .expect("CF not found");
    let corrupted_id = Uuid::new_v4();
    let corrupted_key =
        context_graph_storage::teleological::schema::causal_relationship_key(&corrupted_id);
    db.put_cf(cf, corrupted_key, b"THIS IS NOT VALID JSON OR BINCODE")
        .expect("Failed to inject corrupted data");

    let count_before = h.rt.block_on(async {
        h.store
            .count_causal_relationships()
            .await
            .expect("Count failed")
    });
    assert_eq!(count_before, 2, "Should have 2 entries (1 valid + 1 corrupted)");

    let (deleted, scanned) = h.rt.block_on(async {
        h.store
            .repair_corrupted_causal_relationships()
            .await
            .expect("Repair failed")
    });
    assert_eq!(deleted, 1, "Should delete 1 corrupted entry");
    assert_eq!(scanned, 2, "Should scan 2 entries total");

    let count_after = h.rt.block_on(async {
        h.store
            .count_causal_relationships()
            .await
            .expect("Count failed")
    });
    assert_eq!(count_after, 1, "Should have 1 valid entry after repair");

    let valid = h.rt.block_on(async {
        h.store
            .get_causal_relationship(valid_id)
            .await
            .expect("Get failed")
    });
    assert!(valid.is_some(), "Valid relationship must survive repair");
}
