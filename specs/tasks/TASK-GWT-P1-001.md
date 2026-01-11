# TASK-GWT-P1-001: SELF_EGO_NODE Persistence Layer

## Metadata
| Field | Value |
|-------|-------|
| **Task ID** | TASK-GWT-P1-001 |
| **Title** | Wire SELF_EGO_NODE Persistence to RocksDB |
| **Status** | **COMPLETED** |
| **Completion Date** | 2026-01-10 |
| **Completion Commit** | `ccddb7f` |
| **Priority** | P1 |
| **Layer** | Foundation (Layer 1) |
| **Parent Spec** | SPEC-GWT-001 |
| **Constitution Reference** | v5.0.0, Lines 229-234, 366-370 |

---

## 1. Source of Truth

### 1.1 Authoritative Documents
| Document | Path | Key Sections |
|----------|------|--------------|
| Constitution | `docs2/constitution.yaml` | `gwt.self_ego_node` (229-234), `enforcement.identity` |
| Ego Node Types | `crates/context-graph-core/src/gwt/ego_node.rs` | `SelfEgoNode`, `IdentityContinuity`, `IdentityStatus` |
| Storage Trait | `crates/context-graph-core/src/traits/teleological_memory_store.rs` | `save_ego_node()`, `load_ego_node()` |
| RocksDB Impl | `crates/context-graph-storage/src/teleological/rocksdb_store.rs` | Lines 1860-1979 |
| Serialization | `crates/context-graph-storage/src/teleological/serialization.rs` | Lines 440-500 |
| Column Families | `crates/context-graph-storage/src/teleological/column_families.rs` | Line 101: `CF_EGO_NODE` |
| Tests | `crates/context-graph-storage/src/teleological/tests.rs` | Lines 1510-1833 (12 tests) |

### 1.2 Constitution Requirements (MUST COMPLY)
```yaml
gwt:
  self_ego_node:
    purpose_vector: "13D alignment with north star"
    identity_continuity: "IC = cos(PV_t, PV_{t-1}) × r(t)"  # r = Kuramoto order parameter
    trajectory_window: 100  # max snapshots

enforcement:
  identity:
    thresholds:
      healthy: "> 0.9"
      warning: "[0.7, 0.9]"
      degraded: "[0.5, 0.7)"
      critical: "< 0.5"

forbidden:
  AP-26: "IC<0.5 MUST trigger dream - no silent failures"
```

---

## 2. Implementation Summary (COMPLETED)

### 2.1 Column Family
```rust
// crates/context-graph-storage/src/teleological/column_families.rs:101
pub const CF_EGO_NODE: &str = "ego_node";
```

### 2.2 Key Schema
```rust
// crates/context-graph-storage/src/teleological/schema.rs
pub const EGO_NODE_KEY: &[u8] = b"ego_node";  // 8 bytes, singleton key
```

### 2.3 Serialization (FAIL FAST)
```rust
// crates/context-graph-storage/src/teleological/serialization.rs
pub const EGO_NODE_VERSION: u8 = 1;
const MIN_EGO_NODE_SIZE: usize = 50;
const MAX_EGO_NODE_SIZE: usize = 300_000;

pub fn serialize_ego_node(ego: &SelfEgoNode) -> Vec<u8> {
    let mut result = Vec::with_capacity(10_000);
    result.push(EGO_NODE_VERSION);
    let encoded = serialize(ego).unwrap_or_else(|e| {
        panic!("SERIALIZATION ERROR: Failed to serialize SelfEgoNode. Error: {}", e);
    });
    result.extend(encoded);
    // PANICS if size outside [50B, 300KB] - NO FALLBACK
    result
}

pub fn deserialize_ego_node(data: &[u8]) -> SelfEgoNode {
    // PANICS on: empty data, version mismatch, corruption - NO FALLBACK
}
```

### 2.4 Trait Methods
```rust
// crates/context-graph-core/src/traits/teleological_memory_store.rs
#[async_trait]
pub trait TeleologicalMemoryStore: Send + Sync {
    async fn save_ego_node(&self, ego_node: &SelfEgoNode) -> CoreResult<()>;
    async fn load_ego_node(&self) -> CoreResult<Option<SelfEgoNode>>;
}
```

### 2.5 RocksDB Implementation
```rust
// crates/context-graph-storage/src/teleological/rocksdb_store.rs
impl TeleologicalMemoryStore for RocksDbTeleologicalStore {
    async fn save_ego_node(&self, ego_node: &SelfEgoNode) -> CoreResult<()> {
        let serialized = serialize_ego_node(ego_node);  // PANICS on error
        let cf = self.cf_ego_node();
        let key = ego_node_key();
        self.db.put_cf(cf, key, &serialized).map_err(|e| {
            TeleologicalStoreError::rocksdb_op("put_ego_node", CF_EGO_NODE, Some(ego_node.id), e)
        })?;
        Ok(())
    }

    async fn load_ego_node(&self) -> CoreResult<Option<SelfEgoNode>> {
        let cf = self.cf_ego_node();
        let key = ego_node_key();
        match self.db.get_cf(cf, key) {
            Ok(Some(data)) => Ok(Some(deserialize_ego_node(&data))),  // PANICS on corruption
            Ok(None) => Ok(None),
            Err(e) => Err(CoreError::StorageError(...)),
        }
    }
}
```

---

## 3. Core Types

### 3.1 SelfEgoNode
```rust
// crates/context-graph-core/src/gwt/ego_node.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfEgoNode {
    pub id: Uuid,
    pub fingerprint: Option<TeleologicalFingerprint>,
    pub purpose_vector: [f32; 13],           // 13D alignment vector
    pub coherence_with_actions: f32,          // [0.0, 1.0]
    pub identity_trajectory: Vec<PurposeSnapshot>,  // max 100 entries
    pub last_updated: DateTime<Utc>,
}
```

### 3.2 IdentityContinuity
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IdentityContinuity {
    pub identity_coherence: f32,       // cos(PV_t, PV_{t-1})
    pub recent_continuity: f32,        // avg over trajectory
    pub kuramoto_order_parameter: f32, // r(t) from 13 oscillators
    pub status: IdentityStatus,
    pub computed_at: DateTime<Utc>,
}
```

### 3.3 IdentityStatus (Constitution Thresholds)
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentityStatus {
    Healthy,   // IC > 0.9
    Warning,   // 0.7 <= IC <= 0.9
    Degraded,  // 0.5 <= IC < 0.7
    Critical,  // IC < 0.5 - MUST trigger dream (AP-26)
}
```

---

## 4. Test Coverage (12 PASSING TESTS)

### 4.1 Unit Tests
| Test | Purpose | Location |
|------|---------|----------|
| `test_ego_node_cf_options_valid` | CF options not default | tests.rs:1510 |
| `test_ego_node_key_constant` | Key is 8 bytes | tests.rs:1520 |
| `test_ego_node_in_cf_array` | CF in TELEOLOGICAL_CFS | tests.rs:1530 |
| `test_serialize_ego_node_roundtrip` | Serialize/deserialize equality | tests.rs:1540 |
| `test_serialize_ego_node_with_large_trajectory` | 100 snapshots survive | tests.rs:1570 |
| `test_ego_node_version_constant` | Version = 1 | tests.rs:1610 |

### 4.2 FAIL FAST Tests
| Test | Purpose | Location |
|------|---------|----------|
| `test_ego_node_deserialize_empty_panics` | Empty data = PANIC | tests.rs:1620 |
| `test_ego_node_deserialize_wrong_version_panics` | Version mismatch = PANIC | tests.rs:1630 |

### 4.3 Integration Tests (FSV)
| Test | Purpose | Location |
|------|---------|----------|
| `test_ego_node_save_load_roundtrip` | Save, load, verify equality | tests.rs:1650 |
| `test_ego_node_persistence_across_reopen` | **FSV: Physical verification** | tests.rs:1700 |
| `test_ego_node_overwrite` | Update overwrites correctly | tests.rs:1760 |
| `test_in_memory_store_ego_node_roundtrip` | InMemory impl works | tests.rs:1800 |

### 4.4 Run Tests
```bash
cargo test --package context-graph-storage ego_node --no-fail-fast
```

---

## 5. Full State Verification (FSV)

### 5.1 Source of Truth
- **Database**: RocksDB column family `CF_EGO_NODE`
- **Key**: `b"ego_node"` (singleton)
- **Format**: Version-prefixed bincode (version 1)

### 5.2 Execute & Inspect Pattern
```rust
// Test: test_ego_node_persistence_across_reopen (tests.rs:1700)
#[tokio::test]
async fn test_ego_node_persistence_across_reopen() {
    // 1. EXECUTE: Save ego node
    let original = create_test_ego_node();
    store.save_ego_node(&original).await.unwrap();

    // 2. PHYSICAL VERIFICATION: Close and reopen database
    drop(store);
    let store2 = RocksDbTeleologicalStore::open(&path).await.unwrap();

    // 3. INSPECT: Load and verify
    let loaded = store2.load_ego_node().await.unwrap().unwrap();
    assert_eq!(original.id, loaded.id);
    assert_eq!(original.purpose_vector, loaded.purpose_vector);
    // ... full field verification
}
```

### 5.3 Evidence of Success
```
running 12 tests
test teleological::tests::test_ego_node_cf_options_valid ... ok
test teleological::tests::test_ego_node_key_constant ... ok
test teleological::tests::test_ego_node_in_cf_array ... ok
test teleological::tests::test_serialize_ego_node_roundtrip ... ok
test teleological::tests::test_serialize_ego_node_with_large_trajectory ... ok
test teleological::tests::test_ego_node_version_constant ... ok
test teleological::tests::test_ego_node_deserialize_empty_panics ... ok
test teleological::tests::test_ego_node_deserialize_wrong_version_panics ... ok
test teleological::tests::test_ego_node_save_load_roundtrip ... ok
test teleological::tests::test_ego_node_persistence_across_reopen ... ok
test teleological::tests::test_ego_node_overwrite ... ok
test teleological::tests::test_in_memory_store_ego_node_roundtrip ... ok

test result: ok. 12 passed; 0 failed; 0 ignored
```

---

## 6. Edge Cases (FAIL FAST Behavior)

### 6.1 Empty Data
```rust
#[test]
#[should_panic(expected = "EGO_NODE CORRUPTION")]
fn test_ego_node_deserialize_empty_panics() {
    deserialize_ego_node(&[]);  // PANIC - no silent failure
}
```

### 6.2 Wrong Version
```rust
#[test]
#[should_panic(expected = "EGO_NODE VERSION MISMATCH")]
fn test_ego_node_deserialize_wrong_version_panics() {
    let bad_data = vec![255u8; 100];  // Wrong version byte
    deserialize_ego_node(&bad_data);  // PANIC - no fallback
}
```

### 6.3 Missing Ego Node (Valid Case)
```rust
#[tokio::test]
async fn test_load_missing_ego_node() {
    let store = create_empty_store().await;
    let result = store.load_ego_node().await.unwrap();
    assert!(result.is_none());  // None is valid for first run
}
```

---

## 7. Manual Verification Procedure

### 7.1 Verify Column Family Exists
```bash
# List all column families in database
cargo run --bin db-inspector -- list-cfs /path/to/db
# Expected: [..., "ego_node", ...]
```

### 7.2 Verify Data Persisted
```bash
# Dump ego_node column family
cargo run --bin db-inspector -- dump-cf ego_node /path/to/db
# Expected: Key: [101, 103, 111, 95, 110, 111, 100, 101] (b"ego_node")
#           Value: [1, ...bincode data...]
```

### 7.3 Verify Identity Continuity Calculation
```rust
// After loading ego node, verify IC:
let ego = store.load_ego_node().await?.unwrap();
let ic = ego.compute_identity_continuity(kuramoto_r);
assert!(ic.identity_coherence >= 0.0 && ic.identity_coherence <= 1.0);
assert!(matches!(ic.status, IdentityStatus::Healthy | IdentityStatus::Warning | ...));
```

---

## 8. Files Changed (Commit ccddb7f)

| File | Action | Changes |
|------|--------|---------|
| `crates/context-graph-storage/src/teleological/column_families.rs` | Modified | Added `CF_EGO_NODE` constant |
| `crates/context-graph-storage/src/teleological/serialization.rs` | Modified | Added `serialize_ego_node()`, `deserialize_ego_node()` with FAIL FAST |
| `crates/context-graph-storage/src/teleological/schema.rs` | Modified | Added `EGO_NODE_KEY` constant |
| `crates/context-graph-storage/src/teleological/rocksdb_store.rs` | Modified | Implemented `save_ego_node()`, `load_ego_node()` |
| `crates/context-graph-core/src/traits/teleological_memory_store.rs` | Modified | Added trait methods with default impls |
| `crates/context-graph-core/src/stubs/teleological_store_stub.rs` | Modified | Added stub implementations |
| `crates/context-graph-storage/src/teleological/tests.rs` | Modified | Added 12 tests |
| `crates/context-graph-core/src/gwt/ego_node.rs` | Modified | Added factory methods to `IdentityContinuity` |

---

## 9. Dependencies

### 9.1 Upstream (Required)
- `rocksdb` (0.21+) - Database
- `bincode` (1.3+) - Serialization
- `serde` (1.0+) - Derive macros
- `uuid` (1.0+) - Ego node ID
- `chrono` (0.4+) - Timestamps

### 9.2 Downstream (Depends on This)
- TASK-GWT-P1-002: Full GWT event wiring
- TASK-GWT-P1-003: Integration tests
- TASK-IDENTITY-P0-002: Identity crisis detection

---

## 10. NO BACKWARDS COMPATIBILITY

Per project policy and constitution:

1. **FAIL FAST**: Serialization/deserialization errors PANIC immediately
2. **NO FALLBACKS**: No graceful degradation on corruption
3. **NO WORKAROUNDS**: Version mismatch = PANIC, not migration
4. **ROBUST LOGGING**: All errors include context for debugging

```rust
// Example FAIL FAST pattern:
panic!(
    "EGO_NODE VERSION MISMATCH: expected {}, got {}. \
     Database may be corrupted or from incompatible version. \
     Path: {:?}, Key: {:?}",
    EGO_NODE_VERSION, actual_version, db_path, EGO_NODE_KEY
);
```

---

## 11. Completion Evidence

### 11.1 Git Commit
```
commit ccddb7f
Author: Claude <noreply@anthropic.com>
Date:   Fri Jan 10 2026

    feat(TASK-GWT-P1-001): implement SELF_EGO_NODE persistence layer - COMPLETED

    - Add CF_EGO_NODE column family
    - Implement serialize_ego_node/deserialize_ego_node with FAIL FAST
    - Add save_ego_node/load_ego_node to TeleologicalMemoryStore trait
    - Implement RocksDB persistence in RocksDbTeleologicalStore
    - Add 12 tests including FSV persistence_across_reopen test
    - All tests passing
```

### 11.2 Test Run
```bash
$ cargo test --package context-graph-storage ego_node
test result: ok. 12 passed; 0 failed; 0 ignored
```

### 11.3 Physical Verification
The `test_ego_node_persistence_across_reopen` test proves physical persistence:
1. Creates ego node with known values
2. Saves to RocksDB
3. **Drops the store** (releases file handles)
4. **Reopens fresh store** from same path
5. Loads ego node
6. Verifies ALL fields match original

This confirms data survives database restart - the definitive FSV test.

---

## 12. Future Improvements (Optional)

| Improvement | Benefit | Complexity |
|-------------|---------|------------|
| Schema versioning | Migration support | Medium |
| Compression | Smaller storage | Low |
| Checksum validation | Corruption detection | Low |
| Periodic persistence trigger | Automatic saves | Medium |

**Recommendation**: Current implementation is complete and robust. Future improvements should be separate tasks if needed.
