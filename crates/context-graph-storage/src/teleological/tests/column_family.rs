//! Column family configuration tests.

use crate::teleological::*;

// =========================================================================
// Column Family Tests
// =========================================================================

#[test]
fn test_teleological_cf_names_count() {
    // TASK-TELEO-006: Updated from 4 to 7 CFs
    // TASK-GWT-P1-001: Updated from 8 to 9 CFs (added CF_EGO_NODE)
    // TASK-STORAGE-P2-001: Updated from 9 to 10 CFs (added CF_E12_LATE_INTERACTION)
    // TASK-SESSION-04: Updated from 10 to 11 CFs (added CF_SESSION_IDENTITY)
    assert_eq!(
        TELEOLOGICAL_CFS.len(),
        TELEOLOGICAL_CF_COUNT,
        "Must have exactly {} teleological column families",
        TELEOLOGICAL_CF_COUNT
    );
    assert_eq!(TELEOLOGICAL_CF_COUNT, 11); // TASK-SESSION-04: +1 for CF_SESSION_IDENTITY
}

#[test]
fn test_teleological_cf_names_unique() {
    use std::collections::HashSet;
    let set: HashSet<_> = TELEOLOGICAL_CFS.iter().collect();
    assert_eq!(
        set.len(),
        TELEOLOGICAL_CF_COUNT,
        "All CF names must be unique"
    );
}

#[test]
fn test_teleological_cf_names_are_snake_case() {
    for name in TELEOLOGICAL_CFS {
        assert!(
            name.chars()
                .all(|c| c.is_lowercase() || c == '_' || c.is_ascii_digit()),
            "CF name '{}' should be snake_case",
            name
        );
    }
}

#[test]
fn test_teleological_cf_names_values() {
    // Original 4 CFs
    assert_eq!(CF_FINGERPRINTS, "fingerprints");
    assert_eq!(CF_PURPOSE_VECTORS, "purpose_vectors");
    assert_eq!(CF_E13_SPLADE_INVERTED, "e13_splade_inverted");
    assert_eq!(CF_E1_MATRYOSHKA_128, "e1_matryoshka_128");
    // TASK-TELEO-006: New 3 CFs
    assert_eq!(CF_SYNERGY_MATRIX, "synergy_matrix");
    assert_eq!(CF_TELEOLOGICAL_PROFILES, "teleological_profiles");
    assert_eq!(CF_TELEOLOGICAL_VECTORS, "teleological_vectors");
}

#[test]
fn test_all_cfs_in_array() {
    assert!(TELEOLOGICAL_CFS.contains(&CF_FINGERPRINTS));
    assert!(TELEOLOGICAL_CFS.contains(&CF_PURPOSE_VECTORS));
    assert!(TELEOLOGICAL_CFS.contains(&CF_E13_SPLADE_INVERTED));
    assert!(TELEOLOGICAL_CFS.contains(&CF_E1_MATRYOSHKA_128));
    // TASK-TELEO-006: New CFs
    assert!(TELEOLOGICAL_CFS.contains(&CF_SYNERGY_MATRIX));
    assert!(TELEOLOGICAL_CFS.contains(&CF_TELEOLOGICAL_PROFILES));
    assert!(TELEOLOGICAL_CFS.contains(&CF_TELEOLOGICAL_VECTORS));
}

#[test]
fn test_fingerprint_cf_options_valid() {
    use rocksdb::Cache;
    let cache = Cache::new_lru_cache(256 * 1024 * 1024);
    let opts = fingerprint_cf_options(&cache);
    drop(opts); // Should not panic
}

#[test]
fn test_purpose_vector_cf_options_valid() {
    use rocksdb::Cache;
    let cache = Cache::new_lru_cache(256 * 1024 * 1024);
    let opts = purpose_vector_cf_options(&cache);
    drop(opts);
}

#[test]
fn test_e13_splade_inverted_cf_options_valid() {
    use rocksdb::Cache;
    let cache = Cache::new_lru_cache(256 * 1024 * 1024);
    let opts = e13_splade_inverted_cf_options(&cache);
    drop(opts);
}

#[test]
fn test_e1_matryoshka_128_cf_options_valid() {
    use rocksdb::Cache;
    let cache = Cache::new_lru_cache(256 * 1024 * 1024);
    let opts = e1_matryoshka_128_cf_options(&cache);
    drop(opts);
}

#[test]
fn test_get_teleological_cf_descriptors_returns_7() {
    use rocksdb::Cache;
    let cache = Cache::new_lru_cache(256 * 1024 * 1024);
    let descriptors = get_teleological_cf_descriptors(&cache);
    assert_eq!(
        descriptors.len(),
        TELEOLOGICAL_CF_COUNT,
        "Must return exactly {} descriptors",
        TELEOLOGICAL_CF_COUNT
    );
}

// =========================================================================
// TASK-TELEO-006: New CF Option Builder Tests
// =========================================================================

#[test]
fn test_synergy_matrix_cf_options_valid() {
    use rocksdb::Cache;
    let cache = Cache::new_lru_cache(256 * 1024 * 1024);
    let opts = synergy_matrix_cf_options(&cache);
    drop(opts); // Should not panic
}

#[test]
fn test_teleological_profiles_cf_options_valid() {
    use rocksdb::Cache;
    let cache = Cache::new_lru_cache(256 * 1024 * 1024);
    let opts = teleological_profiles_cf_options(&cache);
    drop(opts);
}

#[test]
fn test_teleological_vectors_cf_options_valid() {
    use rocksdb::Cache;
    let cache = Cache::new_lru_cache(256 * 1024 * 1024);
    let opts = teleological_vectors_cf_options(&cache);
    drop(opts);
}

// =========================================================================
// TASK-TELEO-006: CF Descriptor Order Tests
// =========================================================================

#[test]
fn test_descriptors_in_correct_order() {
    use rocksdb::Cache;
    let cache = Cache::new_lru_cache(256 * 1024 * 1024);
    let descriptors = get_teleological_cf_descriptors(&cache);

    // Verify order matches TELEOLOGICAL_CFS
    for (i, cf_name) in TELEOLOGICAL_CFS.iter().enumerate() {
        assert_eq!(
            descriptors[i].name(),
            *cf_name,
            "Descriptor {} should be '{}', got '{}'",
            i,
            cf_name,
            descriptors[i].name()
        );
    }
}

#[test]
fn test_get_all_teleological_cf_descriptors_returns_24() {
    use rocksdb::Cache;
    let cache = Cache::new_lru_cache(256 * 1024 * 1024);
    let descriptors = get_all_teleological_cf_descriptors(&cache);

    // 11 teleological + 13 quantized embedder = 24
    // TASK-GWT-P1-001: +1 for CF_EGO_NODE
    // TASK-STORAGE-P2-001: +1 for CF_E12_LATE_INTERACTION
    // TASK-SESSION-04: +1 for CF_SESSION_IDENTITY
    assert_eq!(
        descriptors.len(),
        24,
        "Must return 11 teleological + 13 quantized = 24 CFs"
    );
}

// =========================================================================
// TASK-GWT-P1-001: EGO_NODE CF Tests
// =========================================================================

#[test]
fn test_ego_node_cf_options_valid() {
    println!("=== TEST: ego_node_cf_options (TASK-GWT-P1-001) ===");

    use rocksdb::Cache;
    let cache = Cache::new_lru_cache(256 * 1024 * 1024);
    let opts = ego_node_cf_options(&cache);
    drop(opts); // Should not panic

    println!("RESULT: PASS - ego_node_cf_options created successfully");
}

#[test]
fn test_ego_node_in_cf_array() {
    println!("=== TEST: CF_EGO_NODE in TELEOLOGICAL_CFS array ===");

    assert!(
        TELEOLOGICAL_CFS.contains(&CF_EGO_NODE),
        "CF_EGO_NODE must be in TELEOLOGICAL_CFS"
    );

    println!("RESULT: PASS - CF_EGO_NODE is in TELEOLOGICAL_CFS");
}

// =========================================================================
// TASK-SESSION-04: Session Identity CF Tests
// =========================================================================

#[test]
fn test_session_identity_cf_options_valid() {
    println!("=== TEST: session_identity_cf_options (TASK-SESSION-04) ===");

    use rocksdb::Cache;
    let cache = Cache::new_lru_cache(256 * 1024 * 1024);
    let opts = session_identity_cf_options(&cache);
    drop(opts); // Should not panic

    println!("RESULT: PASS - session_identity_cf_options created successfully");
}

#[test]
fn test_session_identity_in_cf_array() {
    println!("=== TEST: CF_SESSION_IDENTITY in TELEOLOGICAL_CFS array ===");

    assert!(
        TELEOLOGICAL_CFS.contains(&CF_SESSION_IDENTITY),
        "CF_SESSION_IDENTITY must be in TELEOLOGICAL_CFS"
    );

    println!("RESULT: PASS - CF_SESSION_IDENTITY is in TELEOLOGICAL_CFS");
}

#[test]
fn test_session_identity_cf_name_value() {
    println!("=== TEST: CF_SESSION_IDENTITY name value (TASK-SESSION-04) ===");

    assert_eq!(CF_SESSION_IDENTITY, "session_identity");

    println!("RESULT: PASS - CF_SESSION_IDENTITY = 'session_identity'");
}

// =========================================================================
// TASK-TELEO-006: Edge Case Tests (with before/after state printing)
// =========================================================================

#[test]
fn edge_case_multiple_cache_references_for_new_cfs() {
    println!("=== EDGE CASE: Multiple option builders sharing same cache (new CFs) ===");
    use rocksdb::Cache;
    let cache = Cache::new_lru_cache(256 * 1024 * 1024);

    println!("BEFORE: Creating options with shared cache reference");
    let opts1 = synergy_matrix_cf_options(&cache);
    let opts2 = teleological_profiles_cf_options(&cache);
    let opts3 = teleological_vectors_cf_options(&cache);

    println!("AFTER: All 3 new option builders created successfully");
    drop(opts1);
    drop(opts2);
    drop(opts3);
    println!("RESULT: PASS - Shared cache works across new Options");
}
