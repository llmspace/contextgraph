# TASK-SESSION-04: Add CF_SESSION_IDENTITY Column Family

```xml
<task_spec id="TASK-SESSION-04" version="2.0">
<metadata>
  <title>Add CF_SESSION_IDENTITY Column Family</title>
  <status>pending</status>
  <layer>foundation</layer>
  <sequence>4</sequence>
  <implements>
    <requirement_ref>REQ-SESSION-04</requirement_ref>
  </implements>
  <depends_on><!-- None --></depends_on>
  <estimated_hours>1.5</estimated_hours>
  <last_audit>2026-01-15</last_audit>
</metadata>
```

## Objective

Add a dedicated `CF_SESSION_IDENTITY` column family to RocksDB for storing `SessionIdentitySnapshot` data with optimized configuration for fast reads during the PreToolUse hot path.

## Current Codebase State (Verified 2026-01-15)

### Column Family Architecture

**Current Total: 42 column families**
- **Base CFs** (`crates/context-graph-storage/src/column_families.rs`): 12 (cf_names::ALL)
- **Teleological CFs** (`crates/context-graph-storage/src/teleological/column_families.rs`): 10 (TELEOLOGICAL_CF_COUNT)
- **Quantized Embedder CFs**: 13 (QUANTIZED_EMBEDDER_CF_COUNT)
- **Autonomous CFs** (`crates/context-graph-storage/src/autonomous/column_families.rs`): 7 (AUTONOMOUS_CF_COUNT)

**After this task: 43 column families**

### Existing Related Code (MUST USE - DO NOT DUPLICATE)

The `SessionIdentitySnapshot` struct and `IdentityCache` already exist in:
- `crates/context-graph-core/src/gwt/session_identity/types.rs` - SessionIdentitySnapshot struct
- `crates/context-graph-core/src/gwt/session_identity/cache.rs` - IdentityCache singleton
- `crates/context-graph-core/src/gwt/session_identity/mod.rs` - Module exports

**Key Constants (defined in types.rs):**
```rust
pub const MAX_TRAJECTORY_LEN: usize = 50;
pub const KURAMOTO_N: usize = 13;
```

### Storage Crate Architecture

```
crates/context-graph-storage/src/
├── column_families.rs          # Base 12 CFs + get_all_column_family_descriptors()
├── teleological/
│   ├── mod.rs                  # Exports CF_* constants
│   └── column_families.rs      # 10 teleological + 13 quantized CFs
├── autonomous/
│   ├── mod.rs                  # Exports CF_* constants
│   └── column_families.rs      # 7 autonomous CFs
└── rocksdb_backend/
    └── core.rs                 # RocksDbMemex::open_with_config()
```

## Key Scheme

| Key Pattern | Value | Size | Description |
|-------------|-------|------|-------------|
| `s:{session_id}` | SessionIdentitySnapshot (bincode) | ~3KB typical, <30KB max | Primary session data |
| `latest` | session_id bytes | ~40 bytes | Pointer to most recent session |
| `t:{timestamp_ms}` | session_id bytes | ~40 bytes | Temporal index (big-endian) |

## Implementation Steps

### Step 1: Add CF Constant to Teleological Module

**File**: `crates/context-graph-storage/src/teleological/column_families.rs`

Add after CF_E12_LATE_INTERACTION:
```rust
/// Column family for SessionIdentitySnapshot persistence.
///
/// Stores flattened identity snapshots for cross-session persistence.
/// Key formats:
/// - `s:{session_id}` -> SessionIdentitySnapshot (~3KB bincode)
/// - `latest` -> session_id string (~40 bytes)
/// - `t:{timestamp_ms_be}` -> session_id string (~40 bytes)
///
/// # Performance Target
/// - Read: <5ms p95 (warm cache)
/// - Write: <10ms p95
///
/// # Constitution Reference
/// - IDENTITY-002: IC thresholds
/// - GWT-003: Identity continuity
pub const CF_SESSION_IDENTITY: &str = "session_identity";
```

### Step 2: Add to TELEOLOGICAL_CFS Array

**File**: `crates/context-graph-storage/src/teleological/column_families.rs`

Update the `TELEOLOGICAL_CFS` array:
```rust
pub const TELEOLOGICAL_CFS: &[&str] = &[
    CF_FINGERPRINTS,
    CF_PURPOSE_VECTORS,
    CF_E13_SPLADE_INVERTED,
    CF_E1_MATRYOSHKA_128,
    CF_SYNERGY_MATRIX,
    CF_TELEOLOGICAL_PROFILES,
    CF_TELEOLOGICAL_VECTORS,
    CF_CONTENT,
    CF_EGO_NODE,
    CF_E12_LATE_INTERACTION,
    CF_SESSION_IDENTITY,  // NEW: TASK-SESSION-04
];
```

Update count:
```rust
pub const TELEOLOGICAL_CF_COUNT: usize = 11;  // Was 10
```

### Step 3: Create Options Builder

**File**: `crates/context-graph-storage/src/teleological/column_families.rs`

Add new function:
```rust
/// Options for session_identity column family.
///
/// Optimized for small, frequently-accessed snapshots during PreToolUse.
///
/// # Configuration
/// - Bloom filter: 10 bits per key (fast existence checks)
/// - LZ4 compression: ~50% reduction for trajectory data
/// - Block cache: enabled (hot path requires fast reads)
/// - Point lookup optimization
///
/// # Performance
/// Target: <5ms read, <10ms write
#[inline]
pub fn session_identity_cf_options(cache: &Cache) -> Options {
    let mut block_opts = BlockBasedOptions::default();
    block_opts.set_block_cache(cache);
    block_opts.set_bloom_filter(10.0, false);
    block_opts.set_cache_index_and_filter_blocks(true);

    let mut opts = Options::default();
    opts.set_block_based_table_factory(&block_opts);
    opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
    opts.optimize_for_point_lookup(64); // 64MB hint for small data
    opts.create_if_missing(true);

    opts
}
```

### Step 4: Add to Descriptor Function

**File**: `crates/context-graph-storage/src/teleological/column_families.rs`

Update `get_teleological_cf_descriptors()`:
```rust
pub fn get_teleological_cf_descriptors(cache: &Cache) -> Vec<ColumnFamilyDescriptor> {
    vec![
        ColumnFamilyDescriptor::new(CF_FINGERPRINTS, fingerprint_cf_options(cache)),
        ColumnFamilyDescriptor::new(CF_PURPOSE_VECTORS, purpose_vector_cf_options(cache)),
        ColumnFamilyDescriptor::new(CF_E13_SPLADE_INVERTED, e13_splade_inverted_cf_options(cache)),
        ColumnFamilyDescriptor::new(CF_E1_MATRYOSHKA_128, e1_matryoshka_128_cf_options(cache)),
        ColumnFamilyDescriptor::new(CF_SYNERGY_MATRIX, synergy_matrix_cf_options(cache)),
        ColumnFamilyDescriptor::new(CF_TELEOLOGICAL_PROFILES, teleological_profiles_cf_options(cache)),
        ColumnFamilyDescriptor::new(CF_TELEOLOGICAL_VECTORS, teleological_vectors_cf_options(cache)),
        ColumnFamilyDescriptor::new(CF_CONTENT, content_cf_options(cache)),
        ColumnFamilyDescriptor::new(CF_EGO_NODE, ego_node_cf_options(cache)),
        ColumnFamilyDescriptor::new(CF_E12_LATE_INTERACTION, e12_late_interaction_cf_options(cache)),
        ColumnFamilyDescriptor::new(CF_SESSION_IDENTITY, session_identity_cf_options(cache)),  // NEW
    ]
}
```

### Step 5: Update Total Count

**File**: `crates/context-graph-storage/src/column_families.rs`

```rust
/// Total number of column families in a fully configured Context Graph database.
/// Base (12) + Teleological (11) + Quantized Embedder (13) + Autonomous (7) = 43
pub const TOTAL_COLUMN_FAMILIES: usize = 43;  // Was 42
```

### Step 6: Add Key Helper Functions

**File**: `crates/context-graph-storage/src/teleological/schema.rs`

Add key helpers:
```rust
// =============================================================================
// SESSION_IDENTITY KEY HELPERS (TASK-SESSION-04)
// =============================================================================

/// Key constant for latest session pointer.
pub const SESSION_LATEST_KEY: &[u8] = b"latest";

/// Create session identity key: `s:{session_id}`
///
/// # Arguments
/// * `session_id` - Session UUID string
///
/// # Returns
/// Key bytes: `s:` prefix + session_id UTF-8 bytes
#[inline]
pub fn session_identity_key(session_id: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(2 + session_id.len());
    key.extend_from_slice(b"s:");
    key.extend_from_slice(session_id.as_bytes());
    key
}

/// Create temporal index key: `t:{timestamp_ms}` (big-endian for lexicographic ordering)
///
/// # Arguments
/// * `timestamp_ms` - Unix milliseconds timestamp
///
/// # Returns
/// Key bytes: `t:` prefix + 8-byte big-endian timestamp
#[inline]
pub fn session_temporal_key(timestamp_ms: i64) -> Vec<u8> {
    let mut key = Vec::with_capacity(10);
    key.extend_from_slice(b"t:");
    key.extend_from_slice(&timestamp_ms.to_be_bytes());
    key
}

/// Parse session ID from session_identity_key.
///
/// # Panics
/// Panics if key doesn't start with `s:` - FAIL FAST policy.
#[inline]
pub fn parse_session_identity_key(key: &[u8]) -> &str {
    assert!(key.len() > 2, "session_identity_key too short: {} bytes", key.len());
    assert_eq!(&key[0..2], b"s:", "session_identity_key must start with 's:'");
    std::str::from_utf8(&key[2..]).expect("session_id must be valid UTF-8")
}

/// Parse timestamp from session_temporal_key.
///
/// # Panics
/// Panics if key doesn't start with `t:` or isn't exactly 10 bytes - FAIL FAST policy.
#[inline]
pub fn parse_session_temporal_key(key: &[u8]) -> i64 {
    assert_eq!(key.len(), 10, "session_temporal_key must be exactly 10 bytes");
    assert_eq!(&key[0..2], b"t:", "session_temporal_key must start with 't:'");
    i64::from_be_bytes(key[2..10].try_into().expect("timestamp bytes"))
}
```

### Step 7: Export from mod.rs

**File**: `crates/context-graph-storage/src/teleological/mod.rs`

Add to exports:
```rust
pub use column_families::{
    // ... existing exports ...
    CF_SESSION_IDENTITY,
    session_identity_cf_options,
};

pub use schema::{
    // ... existing exports ...
    SESSION_LATEST_KEY,
    session_identity_key,
    session_temporal_key,
    parse_session_identity_key,
    parse_session_temporal_key,
};
```

## Verification Commands (MUST ALL PASS)

```bash
# Build storage package
cargo build -p context-graph-storage 2>&1 | tee /tmp/build.log
echo "Exit code: $?"

# Run storage tests
cargo test -p context-graph-storage -- --test-threads=1 --nocapture 2>&1 | tee /tmp/test.log
echo "Exit code: $?"

# Verify specific test
cargo test -p context-graph-storage test_total_column_families_constant -- --nocapture
```

## Test Cases (Add to `crates/context-graph-storage/src/teleological/tests/column_family.rs`)

### TC-SESSION-CF-01: Column Family Count

```rust
#[test]
fn test_teleological_cf_count_includes_session_identity() {
    println!("\n=== TC-SESSION-CF-01: Column Family Count ===");
    println!("SOURCE OF TRUTH: TELEOLOGICAL_CF_COUNT constant");

    println!("BEFORE: Checking TELEOLOGICAL_CF_COUNT");
    assert_eq!(TELEOLOGICAL_CF_COUNT, 11, "Must be 11 after adding SESSION_IDENTITY");

    println!("AFTER: Checking TELEOLOGICAL_CFS array length");
    assert_eq!(TELEOLOGICAL_CFS.len(), 11, "Array must match count");
    assert!(TELEOLOGICAL_CFS.contains(&CF_SESSION_IDENTITY), "Must contain CF_SESSION_IDENTITY");

    println!("RESULT: PASS - SESSION_IDENTITY is 11th teleological CF");
}
```

### TC-SESSION-CF-02: Session Key Format

```rust
#[test]
fn test_session_identity_key_format() {
    println!("\n=== TC-SESSION-CF-02: Session Key Format ===");
    println!("SOURCE OF TRUTH: Key bytes returned by session_identity_key()");

    let session_id = "abc-123-def";
    println!("BEFORE: session_id = '{}'", session_id);

    let key = session_identity_key(session_id);
    println!("AFTER: key = {:?}", key);

    // Verify format
    assert_eq!(&key[0..2], b"s:", "Must start with 's:'");
    assert_eq!(&key[2..], session_id.as_bytes(), "Must contain session_id");

    // Round-trip parse
    let parsed = parse_session_identity_key(&key);
    assert_eq!(parsed, session_id, "Round-trip must preserve session_id");

    println!("RESULT: PASS - Key format 's:{session_id}' verified");
}
```

### TC-SESSION-CF-03: Temporal Key Ordering

```rust
#[test]
fn test_temporal_key_lexicographic_ordering() {
    println!("\n=== TC-SESSION-CF-03: Temporal Key Ordering ===");
    println!("SOURCE OF TRUTH: Big-endian byte comparison");

    let t1 = 1000_i64;
    let t2 = 2000_i64;
    let t3 = i64::MAX;

    println!("BEFORE: t1={}, t2={}, t3={}", t1, t2, t3);

    let k1 = session_temporal_key(t1);
    let k2 = session_temporal_key(t2);
    let k3 = session_temporal_key(t3);

    println!("AFTER: k1.len()={}, k2.len()={}, k3.len()={}", k1.len(), k2.len(), k3.len());

    // Verify lexicographic ordering matches numeric ordering
    assert!(k1 < k2, "k1 must be < k2 for t1 < t2");
    assert!(k2 < k3, "k2 must be < k3 for t2 < t3");

    // Round-trip parse
    assert_eq!(parse_session_temporal_key(&k1), t1);
    assert_eq!(parse_session_temporal_key(&k2), t2);
    assert_eq!(parse_session_temporal_key(&k3), t3);

    println!("RESULT: PASS - Big-endian provides correct lexicographic ordering");
}
```

### EDGE CASE 1: Empty Session ID

```rust
#[test]
fn edge_case_empty_session_id() {
    println!("\n=== EDGE CASE: Empty Session ID ===");

    let session_id = "";
    println!("BEFORE: session_id = '' (empty)");

    let key = session_identity_key(session_id);
    println!("AFTER: key = {:?} (len={})", key, key.len());

    assert_eq!(key.len(), 2, "Empty session_id produces 's:' only");
    assert_eq!(&key, b"s:");

    let parsed = parse_session_identity_key(&key);
    assert_eq!(parsed, "", "Round-trip preserves empty string");

    println!("RESULT: PASS - Empty session_id handled correctly");
}
```

### EDGE CASE 2: Negative Timestamp

```rust
#[test]
fn edge_case_negative_timestamp() {
    println!("\n=== EDGE CASE: Negative Timestamp ===");

    let ts = -1000_i64;
    println!("BEFORE: timestamp = {}", ts);

    let key = session_temporal_key(ts);
    println!("AFTER: key = {:?}", key);

    let parsed = parse_session_temporal_key(&key);
    assert_eq!(parsed, ts, "Negative timestamp preserved");

    println!("RESULT: PASS - Negative timestamps work correctly");
}
```

### EDGE CASE 3: Zero Timestamp

```rust
#[test]
fn edge_case_zero_timestamp() {
    println!("\n=== EDGE CASE: Zero Timestamp ===");

    let ts = 0_i64;
    println!("BEFORE: timestamp = {}", ts);

    let key = session_temporal_key(ts);
    println!("AFTER: key = {:?}", key);

    assert_eq!(key.len(), 10, "Key must be exactly 10 bytes");
    let parsed = parse_session_temporal_key(&key);
    assert_eq!(parsed, 0, "Zero timestamp preserved");

    println!("RESULT: PASS - Zero timestamp handled correctly");
}
```

## Full State Verification

After completing the implementation:

### 1. Source of Truth Verification

```bash
# Verify CF count in database
cargo test -p context-graph-storage test_get_all_column_family_descriptors_returns -- --nocapture 2>&1 | grep -E "(TELEOLOGICAL_CF_COUNT|TOTAL_COLUMN_FAMILIES|43|11)"
```

### 2. RocksDB Open Verification

```rust
// Add to integration test
#[test]
fn verify_rocksdb_opens_with_session_identity_cf() {
    println!("\n=== VERIFICATION: RocksDB Opens with SESSION_IDENTITY ===");

    let tmp = tempfile::TempDir::new().unwrap();
    println!("BEFORE: Creating RocksDbMemex at {:?}", tmp.path());

    let memex = RocksDbMemex::open(tmp.path()).expect("Must open successfully");

    // SOURCE OF TRUTH: Attempt to get CF handle
    let cf = memex.get_cf(CF_SESSION_IDENTITY);

    println!("AFTER: get_cf(CF_SESSION_IDENTITY) returned {:?}", cf.is_ok());
    assert!(cf.is_ok(), "SESSION_IDENTITY column family must exist");

    println!("RESULT: PASS - RocksDB opened with SESSION_IDENTITY CF");
}
```

### 3. Key Round-Trip Verification

```rust
#[test]
fn verify_key_storage_roundtrip() {
    println!("\n=== VERIFICATION: Key Storage Round-Trip ===");

    use context_graph_core::gwt::SessionIdentitySnapshot;

    let tmp = tempfile::TempDir::new().unwrap();
    let memex = RocksDbMemex::open(tmp.path()).unwrap();

    // Create test snapshot
    let snapshot = SessionIdentitySnapshot::new("test-verify-session");
    let bytes = bincode::serialize(&snapshot).unwrap();

    // Get CF handle
    let cf = memex.get_cf(CF_SESSION_IDENTITY).unwrap();

    // Write
    let key = session_identity_key(&snapshot.session_id);
    println!("BEFORE: Writing {} bytes with key {:?}", bytes.len(), key);

    memex.db.put_cf(cf, &key, &bytes).expect("Write must succeed");

    // Read back - SOURCE OF TRUTH
    let read_bytes = memex.db.get_cf(cf, &key)
        .expect("Read must not error")
        .expect("Key must exist");

    println!("AFTER: Read {} bytes", read_bytes.len());

    // Verify
    let restored: SessionIdentitySnapshot = bincode::deserialize(&read_bytes).unwrap();
    assert_eq!(restored.session_id, snapshot.session_id);

    println!("RESULT: PASS - Data persists and deserializes correctly");
}
```

## Definition of Done

### Acceptance Criteria (Automated Tests Must Pass)

- [ ] `CF_SESSION_IDENTITY` constant equals `"session_identity"`
- [ ] `TELEOLOGICAL_CFS` array has 11 elements
- [ ] `TELEOLOGICAL_CF_COUNT` equals 11
- [ ] `TOTAL_COLUMN_FAMILIES` equals 43
- [ ] `session_identity_cf_options()` configures bloom filter (10 bits)
- [ ] `session_identity_cf_options()` configures LZ4 compression
- [ ] RocksDB opens successfully with new column family
- [ ] `session_identity_key()` returns `s:{session_id}` as bytes
- [ ] `session_temporal_key()` returns `t:{big_endian_timestamp}` as bytes
- [ ] All existing tests pass (no regression)

### Constraints (FAIL FAST - NO FALLBACKS)

- Bloom filter: 10 bits per key
- Compression: LZ4
- `temporal_key` uses big-endian for lexicographic ordering
- Parse functions PANIC on invalid input (no silent failures)
- **NO backwards compatibility hacks** - break if something is wrong

## Exit Conditions

- **Success**: `cargo test -p context-graph-storage` passes with 0 failures, RocksDB opens with 43 column families
- **Failure**: Any test fails → FIX THE BUG, do not work around it. Print full error context.

## Next Task

After completion, proceed to **005-TASK-SESSION-05** (save_snapshot/load_snapshot Methods).

```xml
</task_spec>
```
