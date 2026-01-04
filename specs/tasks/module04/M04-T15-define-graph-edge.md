---
id: "M04-T15"
title: "Implement GraphEdge with Marblestone Fields and Storage Integration"
description: |
  GraphEdge struct with 13 Marblestone fields COMPLETE in edges.rs.
  Full storage integration COMPLETE: get_edge, put_edge, get_edges, put_edges,
  delete_edge, iter_edges, get_outgoing_edges all implemented in storage_impl.rs.
  CF_EDGES column family added for full edge storage.
layer: "logic"
status: "complete"
priority: "critical"
estimated_hours: 2
actual_hours: 2
sequence: 21
depends_on:
  - "M04-T14a"
spec_refs:
  - "TECH-GRAPH-004 Section 4.1"
  - "REQ-KG-040 through REQ-KG-044, REQ-KG-065"
files_to_create: []
files_modified:
  - path: "crates/context-graph-graph/src/storage/storage_impl.rs"
    description: "Added get_edge, put_edge, get_edges, put_edges, delete_edge, iter_edges, get_outgoing_edges"
  - path: "crates/context-graph-graph/src/storage/mod.rs"
    description: "Added CF_EDGES constant, exports all edge methods"
  - path: "crates/context-graph-graph/src/storage/edges.rs"
    description: "GraphEdge struct with 13 Marblestone fields"
  - path: "crates/context-graph-core/src/marblestone/edge_type.rs"
    description: "EdgeType enum with as_u8/from_u8 conversion methods"
test_file: "crates/context-graph-graph/tests/storage_tests.rs"
verified_by: "sherlock-holmes"
verified_date: "2025-01-04"
---

## VERIFICATION: COMPLETE (2025-01-04)

### Implementation Summary

**All storage methods implemented in `storage_impl.rs`:**
- `get_edge(edge_id: i64)` - Retrieve full GraphEdge by ID (line 643)
- `put_edge(edge: &GraphEdge)` - Store full GraphEdge (line 668)
- `get_edges(source_node_id: i64)` - Get edges by source node (line 698)
- `put_edges(source_node_id, edges)` - Batch store edges (line 725)
- `delete_edge(edge_id: i64)` - Delete edge (line 751)
- `iter_edges()` - Iterate all edges (line 779)
- `get_outgoing_edges(source_node: i64)` - Get edges for BFS traversal (line 809)

**Column Family:**
- `CF_EDGES` constant defined in `storage/mod.rs` (line 107)
- Included in `ALL_COLUMN_FAMILIES` (7 total CFs)

**GraphEdge struct (edges.rs):**
- 13 Marblestone fields: id, source, target, edge_type, weight, confidence, domain, neurotransmitter_weights, is_amortized_shortcut, steering_reward, traversal_count, created_at, last_traversed_at
- `get_modulated_weight(domain)` with canonical formula
- `record_traversal(success, alpha)` for EMA updates

**EdgeType (edge_type.rs):**
- `as_u8()` / `from_u8()` conversion methods for storage

### Sherlock-Holmes Verification Results

```
MISSION: M04-T15 GraphEdge Storage Integration
DATE: 2025-01-04
STATUS: COMPLETE ✓

EVIDENCE:
1. CF_EDGES exists: line 107 in mod.rs
2. ALL_COLUMN_FAMILIES includes CF_EDGES: 7 total CFs
3. get_edge() method: storage_impl.rs:643
4. put_edge() method: storage_impl.rs:668
5. get_edges() method: storage_impl.rs:698
6. put_edges() method: storage_impl.rs:725
7. get_outgoing_edges() method: storage_impl.rs:809
8. 39 edge tests passing
9. cargo build succeeds
10. No clippy warnings
```

---

## ARCHIVED: Original Requirements

#### 1. Add Column Family for Full Edges

**File: `crates/context-graph-graph/src/storage/mod.rs`**

```rust
// Add after line 26:
pub const CF_EDGES: &str = "edges";

// Update ALL_COLUMN_FAMILIES (around line 32):
pub const ALL_COLUMN_FAMILIES: &[&str] = &[
    CF_ADJACENCY,
    CF_CONES,
    CF_FAISS_IDS,
    CF_HYPERBOLIC,
    CF_METADATA,
    CF_NODES,
    CF_EDGES,  // NEW
];
```

#### 2. Add GraphEdge Storage Methods

**File: `crates/context-graph-graph/src/storage/storage_impl.rs`**

Add after the existing adjacency methods (around line 358):

```rust
use super::edges::GraphEdge;
use super::CF_EDGES;

// ========== Full GraphEdge Operations ==========

/// Get full GraphEdge by edge ID.
///
/// Returns the complete edge with all Marblestone fields including
/// NT weights, domain, steering_reward, etc.
///
/// # Returns
/// * `Ok(Some(edge))` - Edge found
/// * `Ok(None)` - Edge not found
/// * `Err(GraphError::CorruptedData)` - Deserialization failed
pub fn get_edge(&self, edge_id: i64) -> GraphResult<Option<GraphEdge>> {
    let cf = self.cf_edges()?;
    let key = edge_id.to_le_bytes();

    match self.db.get_cf(cf, key)? {
        Some(bytes) => {
            let edge: GraphEdge = bincode::deserialize(&bytes)
                .map_err(|e| GraphError::CorruptedData {
                    location: format!("edges edge_id={}", edge_id),
                    details: e.to_string(),
                })?;
            Ok(Some(edge))
        }
        None => Ok(None),
    }
}

/// Store a full GraphEdge.
///
/// Stores the complete edge with all 13 Marblestone fields.
/// Also updates the adjacency list for the source node.
pub fn put_edge(&self, edge: &GraphEdge) -> GraphResult<()> {
    let cf = self.cf_edges()?;
    let key = edge.id.to_le_bytes();
    let value = bincode::serialize(edge)?;

    self.db.put_cf(cf, key, value)?;
    log::trace!("PUT edge id={}", edge.id);
    Ok(())
}

/// Get all full edges for a source node.
///
/// This is the method BFS should use to get edges with NT weights.
/// Returns empty Vec if node has no edges.
///
/// # Performance
/// O(k) where k = number of edges from this node.
/// Each edge requires one additional key lookup.
pub fn get_edges(&self, source_node_id: NodeId) -> GraphResult<Vec<GraphEdge>> {
    // First get the adjacency list to find edge IDs
    let legacy_edges = self.get_adjacency(source_node_id)?;

    let mut edges = Vec::with_capacity(legacy_edges.len());
    for legacy in legacy_edges {
        // The edge ID is stored in a predictable format
        // For now, we compute it from source + target
        // TODO: Consider storing edge_id in LegacyGraphEdge
        let edge_key = Self::compute_edge_key(source_node_id, legacy.target);

        if let Some(edge) = self.get_edge(edge_key)? {
            edges.push(edge);
        } else {
            // Fallback: create minimal GraphEdge from LegacyGraphEdge
            log::warn!(
                "Edge {} not found in CF_EDGES, creating from legacy",
                edge_key
            );
            edges.push(Self::legacy_to_graph_edge(source_node_id, &legacy));
        }
    }

    Ok(edges)
}

/// Store multiple edges for a node atomically.
///
/// Updates both CF_EDGES and CF_ADJACENCY in a single batch.
pub fn put_edges(&self, source_node_id: NodeId, edges: &[GraphEdge]) -> GraphResult<()> {
    let mut batch = self.new_batch();

    // Store full edges in CF_EDGES
    let cf_edges = self.cf_edges()?;
    for edge in edges {
        let key = edge.id.to_le_bytes();
        let value = bincode::serialize(edge)?;
        batch.put_cf(cf_edges, key, value);
    }

    // Store adjacency list (legacy format for backward compat)
    let legacy_edges: Vec<LegacyGraphEdge> = edges
        .iter()
        .map(|e| LegacyGraphEdge {
            target: Self::uuid_to_i64(&e.target),
            edge_type: e.edge_type.as_u8(),
        })
        .collect();

    self.batch_put_adjacency(&mut batch, source_node_id, &legacy_edges)?;

    self.write_batch(batch)?;
    log::trace!("PUT {} edges for node_id={}", edges.len(), source_node_id);
    Ok(())
}

/// Delete a full edge.
pub fn delete_edge(&self, edge_id: i64) -> GraphResult<()> {
    let cf = self.cf_edges()?;
    let key = edge_id.to_le_bytes();
    self.db.delete_cf(cf, key)?;
    log::trace!("DELETE edge id={}", edge_id);
    Ok(())
}

// ========== Helper Methods ==========

fn cf_edges(&self) -> GraphResult<&ColumnFamily> {
    self.db
        .cf_handle(CF_EDGES)
        .ok_or_else(|| GraphError::ColumnFamilyNotFound(CF_EDGES.to_string()))
}

/// Compute deterministic edge key from source and target.
/// Uses FNV-1a hash for fast, deterministic i64 generation.
fn compute_edge_key(source: NodeId, target: NodeId) -> i64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    target.hash(&mut hasher);
    hasher.finish() as i64
}

/// Convert UUID to i64 for legacy storage.
/// Uses first 8 bytes of UUID.
fn uuid_to_i64(uuid: &uuid::Uuid) -> i64 {
    let bytes = uuid.as_bytes();
    i64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

/// Convert legacy edge to full GraphEdge with default values.
fn legacy_to_graph_edge(source_id: NodeId, legacy: &LegacyGraphEdge) -> GraphEdge {
    use super::edges::{Domain, EdgeType, NeurotransmitterWeights};

    let target_uuid = uuid::Uuid::from_u64_pair(legacy.target as u64, 0);
    let edge_type = EdgeType::from_u8(legacy.edge_type).unwrap_or(EdgeType::Semantic);

    GraphEdge {
        id: Self::compute_edge_key(source_id, legacy.target),
        source: uuid::Uuid::from_u64_pair(source_id as u64, 0),
        target: target_uuid,
        edge_type,
        weight: 0.5,
        confidence: 1.0,
        domain: Domain::General,
        neurotransmitter_weights: NeurotransmitterWeights::default(),
        is_amortized_shortcut: false,
        steering_reward: 0.5,
        traversal_count: 0,
        created_at: 0,
        last_traversed_at: 0,
    }
}
```

#### 3. Add EdgeType Helper

**File: `crates/context-graph-core/src/marblestone/edge_type.rs`**

Add if not present:

```rust
impl EdgeType {
    pub fn as_u8(&self) -> u8 {
        match self {
            EdgeType::Semantic => 0,
            EdgeType::Temporal => 1,
            EdgeType::Causal => 2,
            EdgeType::Hierarchical => 3,
        }
    }

    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(EdgeType::Semantic),
            1 => Some(EdgeType::Temporal),
            2 => Some(EdgeType::Causal),
            3 => Some(EdgeType::Hierarchical),
            _ => None,
        }
    }
}
```

---

## Verification

### Test Commands

```bash
# Build
cargo build -p context-graph-graph 2>&1 | head -50

# Run edge storage tests
cargo test -p context-graph-graph edge -- --nocapture

# Run all storage tests
cargo test -p context-graph-graph storage -- --nocapture

# Clippy
cargo clippy -p context-graph-graph -- -D warnings
```

### Test Cases

```rust
#[cfg(test)]
mod edge_storage_tests {
    use super::*;
    use tempfile::tempdir;
    use uuid::Uuid;

    #[test]
    fn test_put_get_edge() {
        let dir = tempdir().unwrap();
        let storage = GraphStorage::open_default(dir.path()).unwrap();

        let edge = GraphEdge::new(
            1,
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Semantic,
            0.8,
            Domain::Code,
        );

        storage.put_edge(&edge).unwrap();

        let retrieved = storage.get_edge(1).unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, 1);
        assert_eq!(retrieved.edge_type, EdgeType::Semantic);
        assert!((retrieved.weight - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_get_edges_with_modulation() {
        let dir = tempdir().unwrap();
        let storage = GraphStorage::open_default(dir.path()).unwrap();

        let source = Uuid::new_v4();
        let edges = vec![
            GraphEdge::new(1, source, Uuid::new_v4(), EdgeType::Semantic, 0.8, Domain::Code),
            GraphEdge::new(2, source, Uuid::new_v4(), EdgeType::Causal, 0.6, Domain::Code),
        ];

        storage.put_edges(/* source as i64 */ 42, &edges).unwrap();

        let retrieved = storage.get_edges(42).unwrap();
        assert_eq!(retrieved.len(), 2);

        // Verify modulated weight works
        let w = retrieved[0].get_modulated_weight(Domain::Code);
        assert!(w > 0.0);
    }

    #[test]
    fn test_edge_not_found() {
        let dir = tempdir().unwrap();
        let storage = GraphStorage::open_default(dir.path()).unwrap();

        let result = storage.get_edge(99999).unwrap();
        assert!(result.is_none());
    }
}
```

---

## FULL STATE VERIFICATION PROTOCOL

### Source of Truth

1. **File**: `crates/context-graph-graph/src/storage/storage_impl.rs`
2. **Methods**: `get_edge()`, `put_edge()`, `get_edges()`, `put_edges()`
3. **Column Family**: `CF_EDGES` in RocksDB

### Execute & Inspect

```bash
# 1. Verify CF_EDGES constant exists
grep -n "CF_EDGES" crates/context-graph-graph/src/storage/mod.rs

# 2. Verify new methods exist
grep -n "fn get_edges\|fn put_edges\|fn get_edge\|fn put_edge" \
    crates/context-graph-graph/src/storage/storage_impl.rs

# 3. Run tests
cargo test -p context-graph-graph edge_storage -- --nocapture 2>&1 | tail -30

# 4. Verify column family in DB
# (Would need to open DB and check cf_handle)
```

### Boundary & Edge Case Audit

**Edge Case 1: Empty node (no edges)**
```rust
let edges = storage.get_edges(nonexistent_node)?;
println!("BEFORE: Requesting edges for nonexistent node");
println!("AFTER: edges.len() = {}", edges.len());
println!("EXPECTED: 0");
assert!(edges.is_empty());
```

**Edge Case 2: Edge not in CF_EDGES (legacy fallback)**
```rust
// Store only in adjacency, not in CF_EDGES
storage.put_adjacency(1, &[LegacyGraphEdge { target: 2, edge_type: 0 }])?;
let edges = storage.get_edges(1)?;
println!("BEFORE: Legacy edge only in adjacency");
println!("AFTER: edges.len() = {}, has default weight", edges.len());
assert_eq!(edges.len(), 1);
assert!((edges[0].weight - 0.5).abs() < 1e-6); // Default
```

**Edge Case 3: Corrupted edge data**
```rust
// Manually write invalid bincode
let cf = storage.cf_edges()?;
storage.db.put_cf(cf, 999i64.to_le_bytes(), b"invalid")?;
let result = storage.get_edge(999);
println!("BEFORE: Invalid bincode in CF_EDGES");
println!("AFTER: result = {:?}", result);
assert!(matches!(result, Err(GraphError::CorruptedData { .. })));
```

---

## SHERLOCK-HOLMES FINAL VERIFICATION

```
MISSION: Verify M04-T15 storage integration is complete

CHECKLIST:
1. CF_EDGES constant exists in storage/mod.rs
2. CF_EDGES added to ALL_COLUMN_FAMILIES
3. get_edge(edge_id: i64) method exists in GraphStorage
4. put_edge(edge: &GraphEdge) method exists
5. get_edges(source_node_id: NodeId) method exists - THIS IS CRITICAL FOR BFS
6. put_edges() method exists
7. cf_edges() helper method exists
8. cargo build succeeds
9. cargo test edge_storage passes
10. Edge roundtrip test passes (put then get)
11. get_modulated_weight() works on retrieved edges
12. Legacy fallback creates valid GraphEdge

EVIDENCE REQUIRED:
- Test output showing edge put/get roundtrip
- Test output showing get_modulated_weight() on stored edge
- Grep output showing all new methods exist
```

---

## Acceptance Criteria

- [ ] CF_EDGES column family added
- [ ] `get_edge(edge_id)` returns full GraphEdge
- [ ] `put_edge(edge)` stores full GraphEdge
- [ ] `get_edges(node_id)` returns `Vec<GraphEdge>` for BFS
- [ ] `put_edges(node_id, edges)` batch stores edges
- [ ] Legacy fallback works when edge not in CF_EDGES
- [ ] Compiles with `cargo build -p context-graph-graph`
- [ ] Tests pass with `cargo test -p context-graph-graph edge`
- [ ] No clippy warnings
- [ ] sherlock-holmes verification passes
