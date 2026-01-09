//! RocksDB column family definitions for autonomous North Star storage.
//!
//! These 7 CFs extend the storage layer for the autonomous system.
//! All types are defined in context-graph-core/src/autonomous/.
//!
//! # Column Families (7 total)
//! | Name | Purpose | Key Format | Value |
//! |------|---------|------------|-------|
//! | autonomous_config | Singleton AutonomousConfig | "config" (6 bytes) | AutonomousConfig |
//! | adaptive_threshold_state | Singleton threshold state | "state" (5 bytes) | AdaptiveThresholdState |
//! | drift_history | Historical drift data | timestamp_ms:uuid (24 bytes) | DriftDataPoint |
//! | goal_activity_metrics | Per-goal activity | uuid (16 bytes) | GoalActivityMetrics |
//! | autonomous_lineage | Lineage events | timestamp_ms:uuid (24 bytes) | LineageEvent |
//! | consolidation_history | Consolidation records | timestamp_ms:uuid (24 bytes) | ConsolidationRecord |
//! | memory_curation | Memory curation state | uuid (16 bytes) | MemoryCurationState |
//!
//! # FAIL FAST Policy
//!
//! All option builders are infallible at construction time. Errors only
//! occur at DB open time, and those are surfaced by RocksDB itself.

use rocksdb::{BlockBasedOptions, Cache, ColumnFamilyDescriptor, Options, SliceTransform};

// =============================================================================
// COLUMN FAMILY NAME CONSTANTS
// =============================================================================

/// Singleton storage for AutonomousConfig.
/// Key: "config" (fixed 6-byte string)
/// Value: AutonomousConfig serialized via bincode
pub const CF_AUTONOMOUS_CONFIG: &str = "autonomous_config";

/// Singleton storage for AdaptiveThresholdState.
/// Key: "state" (fixed 5-byte string)
/// Value: AdaptiveThresholdState serialized via bincode
pub const CF_ADAPTIVE_THRESHOLD_STATE: &str = "adaptive_threshold_state";

/// Time-series storage for DriftDataPoint history.
/// Key: timestamp_ms (8 bytes) + uuid (16 bytes) = 24 bytes
/// Value: DriftDataPoint serialized via bincode
pub const CF_DRIFT_HISTORY: &str = "drift_history";

/// Per-goal activity metrics storage.
/// Key: GoalId uuid (16 bytes)
/// Value: GoalActivityMetrics serialized via bincode
pub const CF_GOAL_ACTIVITY_METRICS: &str = "goal_activity_metrics";

/// Lineage event storage for traceability.
/// Key: timestamp_ms (8 bytes) + event_uuid (16 bytes) = 24 bytes
/// Value: LineageEvent serialized via bincode
pub const CF_AUTONOMOUS_LINEAGE: &str = "autonomous_lineage";

/// Consolidation history records.
/// Key: timestamp_ms (8 bytes) + record_uuid (16 bytes) = 24 bytes
/// Value: ConsolidationRecord serialized via bincode
pub const CF_CONSOLIDATION_HISTORY: &str = "consolidation_history";

/// Memory curation state storage.
/// Key: MemoryId uuid (16 bytes)
/// Value: MemoryCurationState serialized via bincode
pub const CF_MEMORY_CURATION: &str = "memory_curation";

/// All autonomous column family names (7 total).
pub const AUTONOMOUS_CFS: &[&str] = &[
    CF_AUTONOMOUS_CONFIG,
    CF_ADAPTIVE_THRESHOLD_STATE,
    CF_DRIFT_HISTORY,
    CF_GOAL_ACTIVITY_METRICS,
    CF_AUTONOMOUS_LINEAGE,
    CF_CONSOLIDATION_HISTORY,
    CF_MEMORY_CURATION,
];

/// Total count of autonomous CFs (should be 7).
pub const AUTONOMOUS_CF_COUNT: usize = 7;

// =============================================================================
// CF OPTION BUILDERS
// =============================================================================

/// Options for singleton config storage (small, infrequent access).
///
/// # Configuration
/// - No compression (data is small, compression overhead not worth it)
/// - Bloom filter for fast lookups
/// - Optimized for point lookups
///
/// # FAIL FAST Policy
/// No fallback options - let RocksDB error on open if misconfigured.
pub fn autonomous_config_cf_options(cache: &Cache) -> Options {
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_cache_index_and_filter_blocks(true);

    let mut opts = Options::default();
    opts.set_block_based_table_factory(&block_opts);
    opts.set_compression_type(rocksdb::DBCompressionType::None); // Small singleton
    opts.optimize_for_point_lookup(16); // 16MB hint for point lookups
    opts.create_if_missing(true);
    opts
}

/// Options for adaptive threshold state (singleton, moderate size).
///
/// # Configuration
/// - No compression (moderate size, fast access preferred)
/// - Bloom filter for fast lookups
/// - Optimized for point lookups
pub fn adaptive_threshold_state_cf_options(cache: &Cache) -> Options {
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_cache_index_and_filter_blocks(true);

    let mut opts = Options::default();
    opts.set_block_based_table_factory(&block_opts);
    opts.set_compression_type(rocksdb::DBCompressionType::None);
    opts.optimize_for_point_lookup(16);
    opts.create_if_missing(true);
    opts
}

/// Options for drift history (time-series data, many entries).
///
/// # Configuration
/// - LZ4 compression (good for time-series data)
/// - 8-byte prefix extractor for timestamp_ms prefix scans
/// - Cache index and filter blocks
///
/// # Key Format
/// timestamp_ms (8 bytes BE) + uuid (16 bytes) = 24 bytes total
/// The 8-byte prefix enables efficient time-range scans.
pub fn drift_history_cf_options(cache: &Cache) -> Options {
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_cache_index_and_filter_blocks(true);

    let mut opts = Options::default();
    opts.set_block_based_table_factory(&block_opts);
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8)); // timestamp_ms prefix
    opts.create_if_missing(true);
    opts
}

/// Options for goal activity metrics (per-goal, UUID keys).
///
/// # Configuration
/// - LZ4 compression (moderate size values)
/// - 16-byte prefix extractor for UUID keys
/// - Bloom filter for point lookups
pub fn goal_activity_metrics_cf_options(cache: &Cache) -> Options {
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_cache_index_and_filter_blocks(true);

    let mut opts = Options::default();
    opts.set_block_based_table_factory(&block_opts);
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(16)); // UUID prefix
    opts.create_if_missing(true);
    opts
}

/// Options for autonomous lineage (time-series events).
///
/// # Configuration
/// - LZ4 compression (variable size events)
/// - 8-byte prefix extractor for timestamp prefix scans
/// - Cache index and filter blocks
pub fn autonomous_lineage_cf_options(cache: &Cache) -> Options {
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_cache_index_and_filter_blocks(true);

    let mut opts = Options::default();
    opts.set_block_based_table_factory(&block_opts);
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8)); // timestamp_ms prefix
    opts.create_if_missing(true);
    opts
}

/// Options for consolidation history (time-series records).
///
/// # Configuration
/// - LZ4 compression (records can be larger)
/// - 8-byte prefix extractor for timestamp prefix scans
/// - Cache index and filter blocks
pub fn consolidation_history_cf_options(cache: &Cache) -> Options {
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_cache_index_and_filter_blocks(true);

    let mut opts = Options::default();
    opts.set_block_based_table_factory(&block_opts);
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8)); // timestamp_ms prefix
    opts.create_if_missing(true);
    opts
}

/// Options for memory curation state (per-memory, UUID keys).
///
/// # Configuration
/// - No compression (small enum values)
/// - 16-byte prefix extractor for UUID keys
/// - Bloom filter for point lookups
pub fn memory_curation_cf_options(cache: &Cache) -> Options {
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_cache_index_and_filter_blocks(true);

    let mut opts = Options::default();
    opts.set_block_based_table_factory(&block_opts);
    opts.set_compression_type(rocksdb::DBCompressionType::None); // Small state enums
    opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(16)); // UUID prefix
    opts.optimize_for_point_lookup(64); // 64MB hint
    opts.create_if_missing(true);
    opts
}

// =============================================================================
// DESCRIPTOR GETTERS
// =============================================================================

/// Get all 7 autonomous column family descriptors.
///
/// # Arguments
/// * `cache` - Shared block cache (recommended: 256MB via `Cache::new_lru_cache`)
///
/// # Returns
/// Vector of 7 `ColumnFamilyDescriptor`s for autonomous storage.
///
/// # Example
/// ```ignore
/// use rocksdb::Cache;
/// use context_graph_storage::autonomous::get_autonomous_cf_descriptors;
///
/// let cache = Cache::new_lru_cache(256 * 1024 * 1024); // 256MB
/// let descriptors = get_autonomous_cf_descriptors(&cache);
/// assert_eq!(descriptors.len(), 7);
/// ```
pub fn get_autonomous_cf_descriptors(cache: &Cache) -> Vec<ColumnFamilyDescriptor> {
    vec![
        ColumnFamilyDescriptor::new(CF_AUTONOMOUS_CONFIG, autonomous_config_cf_options(cache)),
        ColumnFamilyDescriptor::new(
            CF_ADAPTIVE_THRESHOLD_STATE,
            adaptive_threshold_state_cf_options(cache),
        ),
        ColumnFamilyDescriptor::new(CF_DRIFT_HISTORY, drift_history_cf_options(cache)),
        ColumnFamilyDescriptor::new(
            CF_GOAL_ACTIVITY_METRICS,
            goal_activity_metrics_cf_options(cache),
        ),
        ColumnFamilyDescriptor::new(CF_AUTONOMOUS_LINEAGE, autonomous_lineage_cf_options(cache)),
        ColumnFamilyDescriptor::new(
            CF_CONSOLIDATION_HISTORY,
            consolidation_history_cf_options(cache),
        ),
        ColumnFamilyDescriptor::new(CF_MEMORY_CURATION, memory_curation_cf_options(cache)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // CF Names Tests
    // =========================================================================

    #[test]
    fn test_cf_count() {
        assert_eq!(
            AUTONOMOUS_CFS.len(),
            AUTONOMOUS_CF_COUNT,
            "AUTONOMOUS_CFS length must match AUTONOMOUS_CF_COUNT"
        );
        assert_eq!(AUTONOMOUS_CF_COUNT, 7, "Must have exactly 7 CFs");
    }

    #[test]
    fn test_cf_names_unique() {
        use std::collections::HashSet;
        let set: HashSet<_> = AUTONOMOUS_CFS.iter().collect();
        assert_eq!(
            set.len(),
            AUTONOMOUS_CF_COUNT,
            "All CF names must be unique"
        );
    }

    #[test]
    fn test_cf_names_snake_case() {
        for name in AUTONOMOUS_CFS {
            assert!(
                name.chars().all(|c| c.is_lowercase() || c == '_'),
                "CF name '{}' should be snake_case",
                name
            );
        }
    }

    #[test]
    fn test_cf_names_non_empty() {
        for name in AUTONOMOUS_CFS {
            assert!(!name.is_empty(), "CF name should not be empty");
        }
    }

    #[test]
    fn test_cf_names_correct_values() {
        assert_eq!(CF_AUTONOMOUS_CONFIG, "autonomous_config");
        assert_eq!(CF_ADAPTIVE_THRESHOLD_STATE, "adaptive_threshold_state");
        assert_eq!(CF_DRIFT_HISTORY, "drift_history");
        assert_eq!(CF_GOAL_ACTIVITY_METRICS, "goal_activity_metrics");
        assert_eq!(CF_AUTONOMOUS_LINEAGE, "autonomous_lineage");
        assert_eq!(CF_CONSOLIDATION_HISTORY, "consolidation_history");
        assert_eq!(CF_MEMORY_CURATION, "memory_curation");
    }

    #[test]
    fn test_all_cfs_in_array() {
        assert!(AUTONOMOUS_CFS.contains(&CF_AUTONOMOUS_CONFIG));
        assert!(AUTONOMOUS_CFS.contains(&CF_ADAPTIVE_THRESHOLD_STATE));
        assert!(AUTONOMOUS_CFS.contains(&CF_DRIFT_HISTORY));
        assert!(AUTONOMOUS_CFS.contains(&CF_GOAL_ACTIVITY_METRICS));
        assert!(AUTONOMOUS_CFS.contains(&CF_AUTONOMOUS_LINEAGE));
        assert!(AUTONOMOUS_CFS.contains(&CF_CONSOLIDATION_HISTORY));
        assert!(AUTONOMOUS_CFS.contains(&CF_MEMORY_CURATION));
    }

    // =========================================================================
    // Option Builders Tests
    // =========================================================================

    #[test]
    fn test_autonomous_config_cf_options_creates_valid_options() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let opts = autonomous_config_cf_options(&cache);
        drop(opts); // No panic = success
    }

    #[test]
    fn test_adaptive_threshold_state_cf_options_creates_valid_options() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let opts = adaptive_threshold_state_cf_options(&cache);
        drop(opts);
    }

    #[test]
    fn test_drift_history_cf_options_creates_valid_options() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let opts = drift_history_cf_options(&cache);
        drop(opts);
    }

    #[test]
    fn test_goal_activity_metrics_cf_options_creates_valid_options() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let opts = goal_activity_metrics_cf_options(&cache);
        drop(opts);
    }

    #[test]
    fn test_autonomous_lineage_cf_options_creates_valid_options() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let opts = autonomous_lineage_cf_options(&cache);
        drop(opts);
    }

    #[test]
    fn test_consolidation_history_cf_options_creates_valid_options() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let opts = consolidation_history_cf_options(&cache);
        drop(opts);
    }

    #[test]
    fn test_memory_curation_cf_options_creates_valid_options() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let opts = memory_curation_cf_options(&cache);
        drop(opts);
    }

    // =========================================================================
    // Descriptor Tests
    // =========================================================================

    #[test]
    fn test_get_descriptors_returns_7() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let descriptors = get_autonomous_cf_descriptors(&cache);
        assert_eq!(
            descriptors.len(),
            AUTONOMOUS_CF_COUNT,
            "Must return exactly 7 descriptors"
        );
    }

    #[test]
    fn test_descriptors_have_correct_names() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let descriptors = get_autonomous_cf_descriptors(&cache);
        let names: Vec<_> = descriptors.iter().map(|d| d.name()).collect();

        for cf_name in AUTONOMOUS_CFS {
            assert!(names.contains(cf_name), "Missing CF: {}", cf_name);
        }
    }

    #[test]
    fn test_descriptors_in_order() {
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let descriptors = get_autonomous_cf_descriptors(&cache);

        for (i, cf_name) in AUTONOMOUS_CFS.iter().enumerate() {
            assert_eq!(
                descriptors[i].name(),
                *cf_name,
                "Descriptor {} should be '{}'",
                i,
                cf_name
            );
        }
    }

    #[test]
    fn test_descriptors_unique_names() {
        use std::collections::HashSet;
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);
        let descriptors = get_autonomous_cf_descriptors(&cache);
        let names: HashSet<_> = descriptors.iter().map(|d| d.name()).collect();
        assert_eq!(
            names.len(),
            AUTONOMOUS_CF_COUNT,
            "All descriptor names must be unique"
        );
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn edge_case_multiple_cache_references() {
        println!("=== EDGE CASE: Multiple option builders sharing same cache ===");
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);

        println!("BEFORE: Creating options with shared cache reference");
        let opts1 = autonomous_config_cf_options(&cache);
        let opts2 = adaptive_threshold_state_cf_options(&cache);
        let opts3 = drift_history_cf_options(&cache);
        let opts4 = goal_activity_metrics_cf_options(&cache);
        let opts5 = autonomous_lineage_cf_options(&cache);
        let opts6 = consolidation_history_cf_options(&cache);
        let opts7 = memory_curation_cf_options(&cache);

        println!("AFTER: All 7 option builders created successfully");
        drop(opts1);
        drop(opts2);
        drop(opts3);
        drop(opts4);
        drop(opts5);
        drop(opts6);
        drop(opts7);
        println!("RESULT: PASS - Shared cache works across all Options");
    }

    #[test]
    fn edge_case_minimum_cache_size() {
        println!("=== EDGE CASE: Minimum cache size (1MB) ===");
        let cache = Cache::new_lru_cache(1024 * 1024); // 1MB minimum

        println!("BEFORE: Creating descriptors with 1MB cache");
        let descriptors = get_autonomous_cf_descriptors(&cache);

        println!("AFTER: {} descriptors created", descriptors.len());
        assert_eq!(descriptors.len(), AUTONOMOUS_CF_COUNT);
        println!("RESULT: PASS - Works with minimum cache size");
    }

    #[test]
    fn edge_case_zero_cache_size() {
        println!("=== EDGE CASE: Zero cache size ===");
        let cache = Cache::new_lru_cache(0);

        println!("BEFORE: Creating descriptors with 0-byte cache");
        let descriptors = get_autonomous_cf_descriptors(&cache);

        println!("AFTER: {} descriptors created", descriptors.len());
        assert_eq!(descriptors.len(), AUTONOMOUS_CF_COUNT);
        println!("RESULT: PASS - Zero cache handled gracefully");
    }

    #[test]
    fn edge_case_reusable_with_same_cache() {
        println!("=== EDGE CASE: Options can be created multiple times with same cache ===");
        let cache = Cache::new_lru_cache(256 * 1024 * 1024);

        println!("BEFORE: Creating first batch of descriptors");
        let desc1 = get_autonomous_cf_descriptors(&cache);
        println!("  First batch: {} descriptors", desc1.len());

        println!("AFTER: Creating second batch of descriptors with same cache");
        let desc2 = get_autonomous_cf_descriptors(&cache);
        println!("  Second batch: {} descriptors", desc2.len());

        assert_eq!(desc1.len(), AUTONOMOUS_CF_COUNT);
        assert_eq!(desc2.len(), AUTONOMOUS_CF_COUNT);
        println!("RESULT: PASS - Cache can be reused");
    }
}
