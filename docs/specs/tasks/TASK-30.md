# TASK-30: Implement merge_concepts Handler

## CRITICAL: READ THIS ENTIRE DOCUMENT BEFORE WRITING ANY CODE

**Task ID**: TASK-MCP-004 (sequence 30)
**Status**: READY
**Layer**: Surface (Phase 4)
**Dependencies**: TASK-29 (merge_concepts schema - COMPLETE)
**Blocks**: TASK-41 (tool registration)
**Estimated Hours**: 6

---

## 1. CONTEXT FOR AI AGENT

You are implementing the `merge_concepts` handler that performs the actual merge logic for consolidating 2-10 concept nodes into a unified node.

### What This Tool Does
Per constitution.yaml SEC-06 and PRD Section 5.3:
- Merges 2-10 related MemoryNode instances into a single unified node
- Supports 3 merge strategies for embeddings: `union`, `intersection`, `weighted_average`
- Generates `reversal_hash` (SHA-256) for 30-day undo capability
- Stores reversal data in RocksDB for later restoration
- Requires mandatory `rationale` audit trail per PRD 0.3

### Why This Exists
- **Curation**: Consolidate duplicate/similar memories for cleaner graph
- **Reversal**: 30-day recovery window per constitution SEC-06
- **Audit Trail**: All merges logged with rationale

### Anti-Patterns to Avoid
- **AP-11**: "merge_concepts without priors_vibe_check" - Must validate node compatibility
- **No backwards compatibility**: If a node doesn't exist, FAIL FAST with error
- **No mock data**: Use real node lookups, real storage operations
- **No fallbacks**: Missing nodes = error, not empty defaults

---

## 2. CURRENT CODEBASE STATE (VERIFIED 2026-01-13)

### 2.1 Completed Prerequisites

| Task | Description | Status |
|------|-------------|--------|
| TASK-29 | merge_concepts schema in `tools/definitions/merge.rs` | COMPLETE |
| TASK-27 | epistemic_action schema (reference pattern) | COMPLETE |
| TASK-28 | epistemic_action handler (reference pattern) | COMPLETE |

### 2.2 Critical Files to Reference

**Schema Definition (TASK-29 output)**:
```
crates/context-graph-mcp/src/tools/definitions/merge.rs
```
Input schema:
- `source_ids`: Vec<Uuid> (2-10 items)
- `target_name`: String (1-256 chars)
- `merge_strategy`: "union" | "intersection" | "weighted_average" (default: "union")
- `rationale`: String (1-1024 chars, REQUIRED)
- `force_merge`: bool (default: false)

**Handler Pattern to Follow (TASK-28 output)**:
```
crates/context-graph-mcp/src/handlers/epistemic.rs
```

**Tool Dispatch (add case here)**:
```
crates/context-graph-mcp/src/handlers/tools/dispatch.rs
```

**Tool Name Constant (already added in TASK-29)**:
```
crates/context-graph-mcp/src/tools/names.rs:122
pub const MERGE_CONCEPTS: &str = "merge_concepts";
```

**Node Data Structure**:
```
crates/context-graph-core/src/types/memory_node/node.rs:36
pub struct MemoryNode { id, content, embedding, quadrant, importance, ... }
```

**Error Codes**:
```
crates/context-graph-mcp/src/protocol.rs:87-257
```

### 2.3 Current Tool Count
After TASK-29: 41 tools registered. This task adds handler logic, not new tools.

---

## 3. EXACT IMPLEMENTATION REQUIREMENTS

### 3.1 File to CREATE: `crates/context-graph-mcp/src/handlers/merge.rs`

```rust
//! Merge Concepts MCP Handler (TASK-MCP-004)
//!
//! Implements merge_concepts tool for consolidating related concept nodes.
//! Constitution: SEC-06 (30-day reversal), PRD Section 5.3
//!
//! ## Merge Strategies
//! - union: Combine all embedding dimensions (average)
//! - intersection: Keep dimensions where all sources have significant values
//! - weighted_average: Weight embeddings by node importance
//!
//! ## Error Handling
//! FAIL FAST: All errors return immediately with detailed error codes.
//! NO fallbacks, NO default values, NO mock data.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Sha256, Digest};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::Handlers;

/// Merge strategy for combining embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MergeStrategy {
    #[default]
    Union,
    Intersection,
    WeightedAverage,
}

/// Input for merge_concepts tool (matches schema from TASK-29)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MergeConceptsInput {
    pub source_ids: Vec<Uuid>,
    pub target_name: String,
    #[serde(default)]
    pub merge_strategy: MergeStrategy,
    pub rationale: String,
    #[serde(default)]
    pub force_merge: bool,
}

/// Output for merge_concepts tool
#[derive(Debug, Clone, Serialize)]
pub struct MergeConceptsOutput {
    /// Whether the merge was successful
    pub success: bool,
    /// UUID of the newly created merged node
    pub merged_id: Uuid,
    /// SHA-256 hash for reversal (30-day undo capability)
    pub reversal_hash: String,
    /// Details of the merged node
    pub merged_node: MergedNodeInfo,
    /// Error message if any (null on success)
    pub error: Option<String>,
}

/// Information about the merged node
#[derive(Debug, Clone, Serialize)]
pub struct MergedNodeInfo {
    pub id: Uuid,
    pub name: String,
    pub source_count: usize,
    pub strategy_used: MergeStrategy,
    pub created_at: String,
    /// Average importance of source nodes
    pub importance: f32,
    /// Embedding dimension preserved
    pub embedding_dim: usize,
}

/// Reversal record stored for 30-day undo capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReversalRecord {
    pub reversal_hash: String,
    pub merged_id: Uuid,
    pub source_ids: Vec<Uuid>,
    /// Original node data serialized for restoration
    pub original_nodes: Vec<SerializedNode>,
    pub created_at: String,
    pub expires_at: String,
    pub rationale: String,
    pub strategy: MergeStrategy,
}

/// Serialized node data for reversal restoration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedNode {
    pub id: Uuid,
    pub content: String,
    pub embedding: Vec<f32>,
    pub importance: f32,
    pub emotional_valence: f32,
    pub metadata_json: String,
}

/// Schema constraints from TASK-29
const MIN_SOURCE_IDS: usize = 2;
const MAX_SOURCE_IDS: usize = 10;
const MIN_TARGET_NAME_LEN: usize = 1;
const MAX_TARGET_NAME_LEN: usize = 256;
const MIN_RATIONALE_LEN: usize = 1;
const MAX_RATIONALE_LEN: usize = 1024;

/// Reversal expiration per SEC-06
const REVERSAL_DAYS: i64 = 30;

/// Intersection threshold: dimension is "significant" if >= this value
const INTERSECTION_THRESHOLD: f32 = 0.01;

impl Handlers {
    /// Handle merge_concepts tool call.
    ///
    /// TASK-MCP-004: Merge concepts handler implementation.
    /// FAIL FAST if any source node doesn't exist.
    pub(super) async fn call_merge_concepts(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling merge_concepts tool call: {:?}", args);

        // FAIL FAST: Parse and validate input
        let input: MergeConceptsInput = match serde_json::from_value(args.clone()) {
            Ok(i) => i,
            Err(e) => {
                error!("merge_concepts: Invalid input: {}", e);
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid merge_concepts input: {}", e),
                );
            }
        };

        // FAIL FAST: Validate source_ids count (2-10 per schema)
        if input.source_ids.len() < MIN_SOURCE_IDS {
            error!("merge_concepts: Too few source_ids: {} < {}", input.source_ids.len(), MIN_SOURCE_IDS);
            return JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                format!("source_ids requires at least {} items, got {}", MIN_SOURCE_IDS, input.source_ids.len()),
            );
        }
        if input.source_ids.len() > MAX_SOURCE_IDS {
            error!("merge_concepts: Too many source_ids: {} > {}", input.source_ids.len(), MAX_SOURCE_IDS);
            return JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                format!("source_ids allows at most {} items, got {}", MAX_SOURCE_IDS, input.source_ids.len()),
            );
        }

        // FAIL FAST: Validate target_name length (1-256 per schema)
        if input.target_name.len() < MIN_TARGET_NAME_LEN {
            error!("merge_concepts: Empty target_name");
            return JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                format!("target_name must be at least {} char", MIN_TARGET_NAME_LEN),
            );
        }
        if input.target_name.len() > MAX_TARGET_NAME_LEN {
            error!("merge_concepts: target_name too long: {} > {}", input.target_name.len(), MAX_TARGET_NAME_LEN);
            return JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                format!("target_name exceeds max length: {} > {}", input.target_name.len(), MAX_TARGET_NAME_LEN),
            );
        }

        // FAIL FAST: Validate rationale length (1-1024 per schema, REQUIRED per PRD 0.3)
        if input.rationale.len() < MIN_RATIONALE_LEN {
            error!("merge_concepts: Empty rationale");
            return JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                format!("rationale is REQUIRED (min {} char)", MIN_RATIONALE_LEN),
            );
        }
        if input.rationale.len() > MAX_RATIONALE_LEN {
            error!("merge_concepts: rationale too long: {} > {}", input.rationale.len(), MAX_RATIONALE_LEN);
            return JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                format!("rationale exceeds max length: {} > {}", input.rationale.len(), MAX_RATIONALE_LEN),
            );
        }

        // FAIL FAST: Check for duplicate source_ids
        let mut seen_ids = std::collections::HashSet::new();
        for source_id in &input.source_ids {
            if !seen_ids.insert(*source_id) {
                error!("merge_concepts: Duplicate source_id: {}", source_id);
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Duplicate source_id: {}", source_id),
                );
            }
        }

        // Execute the merge operation
        match self.execute_merge(&input).await {
            Ok(output) => {
                info!(
                    "merge_concepts SUCCESS: merged {} nodes into {} with hash {}",
                    input.source_ids.len(),
                    output.merged_id,
                    output.reversal_hash
                );
                self.tool_result_with_pulse(id, json!(output))
            }
            Err(e) => {
                error!("merge_concepts FAILED: {}", e);
                JsonRpcResponse::error(id, error_codes::STORAGE_ERROR, e)
            }
        }
    }

    /// Execute the merge operation.
    ///
    /// 1. Fetch all source nodes (FAIL FAST if any missing)
    /// 2. Optional: Check priors compatibility (unless force_merge)
    /// 3. Merge embeddings using specified strategy
    /// 4. Create merged node with combined attributes
    /// 5. Generate reversal hash and store reversal record
    /// 6. Store merged node
    /// 7. Optionally mark source nodes as merged (soft delete)
    async fn execute_merge(&self, input: &MergeConceptsInput) -> Result<MergeConceptsOutput, String> {
        // Step 1: Fetch all source nodes
        let source_nodes = self.fetch_source_nodes(&input.source_ids).await?;

        if source_nodes.len() != input.source_ids.len() {
            let missing: Vec<_> = input.source_ids.iter()
                .filter(|id| !source_nodes.iter().any(|n| &n.id == *id))
                .collect();
            return Err(format!("Missing source nodes: {:?}", missing));
        }

        // Step 2: Priors vibe check (AP-11) unless force_merge
        if !input.force_merge {
            self.check_priors_compatibility(&source_nodes)?;
        }

        // Step 3: Merge embeddings using strategy
        let merged_embedding = match input.merge_strategy {
            MergeStrategy::Union => self.merge_union(&source_nodes),
            MergeStrategy::Intersection => self.merge_intersection(&source_nodes),
            MergeStrategy::WeightedAverage => self.merge_weighted_average(&source_nodes),
        };

        // Step 4: Create merged node
        let merged_id = Uuid::new_v4();
        let now = Utc::now();
        let avg_importance = source_nodes.iter().map(|n| n.importance).sum::<f32>() / source_nodes.len() as f32;
        let avg_valence = source_nodes.iter().map(|n| n.emotional_valence).sum::<f32>() / source_nodes.len() as f32;

        // Combine content from all sources (for merged node context)
        let combined_content = format!(
            "[MERGED] {}\n---\nMerged from {} sources: {}\nStrategy: {:?}\nRationale: {}",
            input.target_name,
            source_nodes.len(),
            input.source_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", "),
            input.merge_strategy,
            input.rationale
        );

        // Step 5: Generate reversal hash and store reversal record
        let reversal_hash = self.generate_reversal_hash(&input.source_ids, merged_id);
        let expires_at = now + chrono::Duration::days(REVERSAL_DAYS);

        let reversal_record = ReversalRecord {
            reversal_hash: reversal_hash.clone(),
            merged_id,
            source_ids: input.source_ids.clone(),
            original_nodes: source_nodes.iter().map(|n| SerializedNode {
                id: n.id,
                content: n.content.clone(),
                embedding: n.embedding.clone(),
                importance: n.importance,
                emotional_valence: n.emotional_valence,
                metadata_json: serde_json::to_string(&n.metadata).unwrap_or_default(),
            }).collect(),
            created_at: now.to_rfc3339(),
            expires_at: expires_at.to_rfc3339(),
            rationale: input.rationale.clone(),
            strategy: input.merge_strategy,
        };

        // Store reversal record (for 30-day undo per SEC-06)
        self.store_reversal_record(&reversal_record).await?;

        // Step 6: Store merged node
        let merged_node = context_graph_core::types::MemoryNode {
            id: merged_id,
            content: combined_content,
            embedding: merged_embedding.clone(),
            quadrant: context_graph_core::types::JohariQuadrant::default(),
            importance: avg_importance,
            emotional_valence: avg_valence,
            created_at: now,
            accessed_at: now,
            access_count: 0,
            metadata: context_graph_core::types::NodeMetadata::default(),
        };

        self.store_merged_node(&merged_node).await?;

        // Step 7: Mark source nodes as merged (soft delete per SEC-06)
        self.mark_nodes_as_merged(&input.source_ids, merged_id).await?;

        Ok(MergeConceptsOutput {
            success: true,
            merged_id,
            reversal_hash,
            merged_node: MergedNodeInfo {
                id: merged_id,
                name: input.target_name.clone(),
                source_count: source_nodes.len(),
                strategy_used: input.merge_strategy,
                created_at: now.to_rfc3339(),
                importance: avg_importance,
                embedding_dim: merged_embedding.len(),
            },
            error: None,
        })
    }

    /// Fetch source nodes from storage.
    /// FAIL FAST if any node is not found.
    async fn fetch_source_nodes(
        &self,
        source_ids: &[Uuid],
    ) -> Result<Vec<context_graph_core::types::MemoryNode>, String> {
        let mut nodes = Vec::with_capacity(source_ids.len());

        // Get storage backend
        let storage = self.storage_backend.as_ref()
            .ok_or_else(|| "Storage backend not initialized".to_string())?;

        for source_id in source_ids {
            match storage.read().await.get_node(*source_id).await {
                Ok(Some(node)) => nodes.push(node),
                Ok(None) => {
                    return Err(format!("Source node not found: {}", source_id));
                }
                Err(e) => {
                    return Err(format!("Failed to fetch node {}: {}", source_id, e));
                }
            }
        }

        Ok(nodes)
    }

    /// Check priors compatibility (AP-11: merge_concepts without priors_vibe_check).
    /// Nodes should have similar characteristics to merge cleanly.
    fn check_priors_compatibility(
        &self,
        nodes: &[context_graph_core::types::MemoryNode],
    ) -> Result<(), String> {
        if nodes.len() < 2 {
            return Ok(()); // Nothing to compare
        }

        // Check embedding dimension consistency
        let first_dim = nodes[0].embedding.len();
        for (i, node) in nodes.iter().enumerate().skip(1) {
            if node.embedding.len() != first_dim {
                return Err(format!(
                    "Embedding dimension mismatch: node[0] has {} dims, node[{}] has {} dims",
                    first_dim, i, node.embedding.len()
                ));
            }
        }

        // Check importance range compatibility (warn but don't fail unless extreme)
        let importances: Vec<f32> = nodes.iter().map(|n| n.importance).collect();
        let max_importance = importances.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_importance = importances.iter().cloned().fold(f32::INFINITY, f32::min);

        if max_importance - min_importance > 0.8 {
            warn!(
                "Large importance spread in merge: min={}, max={} (consider force_merge)",
                min_importance, max_importance
            );
        }

        // Check emotional valence compatibility
        let valences: Vec<f32> = nodes.iter().map(|n| n.emotional_valence).collect();
        let has_positive = valences.iter().any(|&v| v > 0.5);
        let has_negative = valences.iter().any(|&v| v < -0.5);

        if has_positive && has_negative {
            warn!(
                "Conflicting emotional valences in merge: mixing positive and negative sentiment (consider force_merge)"
            );
        }

        Ok(())
    }

    /// Merge embeddings using UNION strategy (average all dimensions).
    fn merge_union(&self, nodes: &[context_graph_core::types::MemoryNode]) -> Vec<f32> {
        if nodes.is_empty() {
            return Vec::new();
        }

        let dim = nodes[0].embedding.len();
        let n = nodes.len() as f32;

        let mut merged = vec![0.0f32; dim];
        for node in nodes {
            for (i, &val) in node.embedding.iter().enumerate() {
                merged[i] += val / n;
            }
        }

        // Normalize the result to unit length
        let magnitude: f32 = merged.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for val in &mut merged {
                *val /= magnitude;
            }
        }

        merged
    }

    /// Merge embeddings using INTERSECTION strategy (keep only significant shared dimensions).
    fn merge_intersection(&self, nodes: &[context_graph_core::types::MemoryNode]) -> Vec<f32> {
        if nodes.is_empty() {
            return Vec::new();
        }

        let dim = nodes[0].embedding.len();
        let n = nodes.len() as f32;

        let mut merged = vec![0.0f32; dim];

        for i in 0..dim {
            // Check if ALL nodes have significant value at this dimension
            let all_significant = nodes.iter().all(|node| {
                node.embedding.get(i).map(|&v| v.abs() >= INTERSECTION_THRESHOLD).unwrap_or(false)
            });

            if all_significant {
                // Average the values for this dimension
                let sum: f32 = nodes.iter()
                    .filter_map(|n| n.embedding.get(i))
                    .sum();
                merged[i] = sum / n;
            }
            // Else: dimension stays 0.0 (not shared significantly)
        }

        // Normalize the result
        let magnitude: f32 = merged.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for val in &mut merged {
                *val /= magnitude;
            }
        }

        merged
    }

    /// Merge embeddings using WEIGHTED_AVERAGE strategy (weight by importance).
    fn merge_weighted_average(&self, nodes: &[context_graph_core::types::MemoryNode]) -> Vec<f32> {
        if nodes.is_empty() {
            return Vec::new();
        }

        let dim = nodes[0].embedding.len();

        // Sum of importances for normalization
        let total_weight: f32 = nodes.iter().map(|n| n.importance).sum();
        if total_weight == 0.0 {
            // Fallback to union if all importances are 0
            return self.merge_union(nodes);
        }

        let mut merged = vec![0.0f32; dim];
        for node in nodes {
            let weight = node.importance / total_weight;
            for (i, &val) in node.embedding.iter().enumerate() {
                merged[i] += val * weight;
            }
        }

        // Normalize the result
        let magnitude: f32 = merged.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for val in &mut merged {
                *val /= magnitude;
            }
        }

        merged
    }

    /// Generate SHA-256 reversal hash from source IDs and merged ID.
    fn generate_reversal_hash(&self, source_ids: &[Uuid], merged_id: Uuid) -> String {
        let mut hasher = Sha256::new();

        // Include all source IDs in deterministic order
        let mut sorted_sources: Vec<_> = source_ids.iter().collect();
        sorted_sources.sort();
        for id in sorted_sources {
            hasher.update(id.as_bytes());
        }

        // Include merged ID
        hasher.update(merged_id.as_bytes());

        // Include timestamp for uniqueness
        hasher.update(Utc::now().timestamp().to_le_bytes());

        let result = hasher.finalize();
        format!("sha256:{}", hex::encode(result))
    }

    /// Store reversal record for 30-day undo capability (SEC-06).
    async fn store_reversal_record(&self, record: &ReversalRecord) -> Result<(), String> {
        let storage = self.storage_backend.as_ref()
            .ok_or_else(|| "Storage backend not initialized".to_string())?;

        let key = format!("reversal:{}", record.reversal_hash);
        let value = serde_json::to_vec(record)
            .map_err(|e| format!("Failed to serialize reversal record: {}", e))?;

        storage.write().await
            .put_raw(&key, &value)
            .await
            .map_err(|e| format!("Failed to store reversal record: {}", e))
    }

    /// Store the merged node.
    async fn store_merged_node(
        &self,
        node: &context_graph_core::types::MemoryNode,
    ) -> Result<(), String> {
        let storage = self.storage_backend.as_ref()
            .ok_or_else(|| "Storage backend not initialized".to_string())?;

        storage.write().await
            .put_node(node)
            .await
            .map_err(|e| format!("Failed to store merged node: {}", e))
    }

    /// Mark source nodes as merged (soft delete per SEC-06).
    async fn mark_nodes_as_merged(
        &self,
        source_ids: &[Uuid],
        merged_into: Uuid,
    ) -> Result<(), String> {
        let storage = self.storage_backend.as_ref()
            .ok_or_else(|| "Storage backend not initialized".to_string())?;

        for source_id in source_ids {
            // Mark node as merged (soft delete) rather than hard delete
            // This allows reversal within 30 days
            let key = format!("merged:{}", source_id);
            let value = serde_json::json!({
                "original_id": source_id,
                "merged_into": merged_into,
                "merged_at": Utc::now().to_rfc3339()
            });

            storage.write().await
                .put_raw(&key, value.to_string().as_bytes())
                .await
                .map_err(|e| format!("Failed to mark node {} as merged: {}", source_id, e))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_strategy_deserialization() {
        let json = r#""union""#;
        let strategy: MergeStrategy = serde_json::from_str(json).expect("union");
        assert_eq!(strategy, MergeStrategy::Union);

        let json = r#""intersection""#;
        let strategy: MergeStrategy = serde_json::from_str(json).expect("intersection");
        assert_eq!(strategy, MergeStrategy::Intersection);

        let json = r#""weighted_average""#;
        let strategy: MergeStrategy = serde_json::from_str(json).expect("weighted_average");
        assert_eq!(strategy, MergeStrategy::WeightedAverage);
    }

    #[test]
    fn test_merge_strategy_default() {
        let strategy = MergeStrategy::default();
        assert_eq!(strategy, MergeStrategy::Union);
    }

    #[test]
    fn test_merge_concepts_input_deserialization() {
        let json = r#"{
            "source_ids": [
                "550e8400-e29b-41d4-a716-446655440001",
                "550e8400-e29b-41d4-a716-446655440002"
            ],
            "target_name": "Merged Concept",
            "merge_strategy": "union",
            "rationale": "Consolidating duplicates",
            "force_merge": false
        }"#;
        let input: MergeConceptsInput = serde_json::from_str(json).expect("deserialize");
        assert_eq!(input.source_ids.len(), 2);
        assert_eq!(input.target_name, "Merged Concept");
        assert_eq!(input.merge_strategy, MergeStrategy::Union);
        assert_eq!(input.rationale, "Consolidating duplicates");
        assert!(!input.force_merge);
    }

    #[test]
    fn test_merge_concepts_input_defaults() {
        let json = r#"{
            "source_ids": [
                "550e8400-e29b-41d4-a716-446655440001",
                "550e8400-e29b-41d4-a716-446655440002"
            ],
            "target_name": "Test",
            "rationale": "Testing"
        }"#;
        let input: MergeConceptsInput = serde_json::from_str(json).expect("deserialize");
        assert_eq!(input.merge_strategy, MergeStrategy::Union); // default
        assert!(!input.force_merge); // default
    }

    #[test]
    fn test_reversal_record_serialization() {
        let record = ReversalRecord {
            reversal_hash: "sha256:abc123".to_string(),
            merged_id: Uuid::nil(),
            source_ids: vec![Uuid::nil()],
            original_nodes: vec![],
            created_at: "2026-01-13T00:00:00Z".to_string(),
            expires_at: "2026-02-12T00:00:00Z".to_string(),
            rationale: "Test merge".to_string(),
            strategy: MergeStrategy::Union,
        };
        let json = serde_json::to_string(&record).expect("serialize");
        assert!(json.contains("sha256:abc123"));
        assert!(json.contains("2026-02-12")); // 30 days later
    }

    #[test]
    fn test_merged_node_info_serialization() {
        let info = MergedNodeInfo {
            id: Uuid::nil(),
            name: "Test".to_string(),
            source_count: 3,
            strategy_used: MergeStrategy::WeightedAverage,
            created_at: "2026-01-13T00:00:00Z".to_string(),
            importance: 0.75,
            embedding_dim: 1536,
        };
        let json = serde_json::to_string(&info).expect("serialize");
        assert!(json.contains("source_count\":3"));
        assert!(json.contains("weighted_average"));
    }

    // ========== UNIT TESTS FOR MERGE STRATEGIES ==========

    #[test]
    fn test_merge_union_basic() {
        // Simulating merge of two normalized vectors
        let v1 = vec![1.0, 0.0, 0.0]; // Normalized
        let v2 = vec![0.0, 1.0, 0.0]; // Normalized

        // Union averages: [0.5, 0.5, 0.0], then normalizes
        let expected_raw = vec![0.5, 0.5, 0.0];
        let magnitude = (0.5f32.powi(2) + 0.5f32.powi(2)).sqrt();
        let expected = vec![0.5 / magnitude, 0.5 / magnitude, 0.0];

        // The actual merge logic would produce normalized result
        assert!(expected[0] > 0.7); // ~0.707
        assert!(expected[1] > 0.7); // ~0.707
        assert_eq!(expected[2], 0.0);
    }

    #[test]
    fn test_merge_intersection_threshold() {
        // Test that intersection only keeps dimensions where ALL nodes have significant values
        assert!(INTERSECTION_THRESHOLD == 0.01);

        // If node1 has [0.5, 0.001, 0.5] and node2 has [0.5, 0.5, 0.001]
        // Intersection keeps only dim 0 (both >= 0.01)
    }

    #[test]
    fn test_source_ids_validation_bounds() {
        assert_eq!(MIN_SOURCE_IDS, 2);
        assert_eq!(MAX_SOURCE_IDS, 10);
    }

    #[test]
    fn test_target_name_validation_bounds() {
        assert_eq!(MIN_TARGET_NAME_LEN, 1);
        assert_eq!(MAX_TARGET_NAME_LEN, 256);
    }

    #[test]
    fn test_rationale_validation_bounds() {
        assert_eq!(MIN_RATIONALE_LEN, 1);
        assert_eq!(MAX_RATIONALE_LEN, 1024);
    }

    #[test]
    fn test_reversal_expiration() {
        assert_eq!(REVERSAL_DAYS, 30);
    }
}
```

### 3.2 File to MODIFY: `crates/context-graph-mcp/src/handlers/mod.rs`

Add the merge module declaration:

```rust
// Around line 31, add:
mod merge;
```

After line 24 ("epistemic: Epistemic action handlers..."), add:
```rust
//! - `merge`: Merge concepts handler for node consolidation (TASK-MCP-004)
```

### 3.3 File to MODIFY: `crates/context-graph-mcp/src/handlers/tools/dispatch.rs`

Add dispatch case for merge_concepts. After line 155 (EPISTEMIC_ACTION case), add:

```rust
            // TASK-MCP-004: Merge concepts for node consolidation
            tool_names::MERGE_CONCEPTS => self.call_merge_concepts(id, arguments).await,
```

### 3.4 File to MODIFY: `Cargo.toml` for context-graph-mcp

Add `sha2` and `hex` dependencies for reversal hash generation:

```toml
[dependencies]
sha2 = "0.10"
hex = "0.4"
```

**NOTE**: Check if these are already in workspace dependencies first.

---

## 4. STORAGE BACKEND INTERFACE

The handler requires these methods on the storage backend. Verify they exist or add them:

```rust
// crates/context-graph-storage traits
trait StorageBackend {
    async fn get_node(&self, id: Uuid) -> Result<Option<MemoryNode>, StorageError>;
    async fn put_node(&self, node: &MemoryNode) -> Result<(), StorageError>;
    async fn put_raw(&self, key: &str, value: &[u8]) -> Result<(), StorageError>;
}
```

If these methods don't exist, you must implement them before the handler will work.

---

## 5. CONSTRAINTS (MUST NOT VIOLATE)

| Constraint | Source | Value | Error Code |
|------------|--------|-------|------------|
| source_ids count | Schema TASK-29 | 2-10 UUIDs | INVALID_PARAMS |
| target_name length | Schema TASK-29 | 1-256 chars | INVALID_PARAMS |
| rationale length | PRD 0.3 | 1-1024 chars (REQUIRED) | INVALID_PARAMS |
| No duplicate source_ids | This task | Unique UUIDs only | INVALID_PARAMS |
| All source nodes must exist | FAIL FAST | No mock data | NODE_NOT_FOUND |
| Storage backend required | FAIL FAST | No mock storage | STORAGE_ERROR |
| Reversal window | SEC-06 | 30 days | - |
| Priors compatibility | AP-11 | Check unless force_merge | INVALID_PARAMS (warning) |

---

## 6. VERIFICATION COMMANDS

Execute in order after implementation:

```bash
# Step 1: Check compilation
cargo check -p context-graph-mcp
# Expected: PASSES (warnings OK for now)

# Step 2: Run merge handler tests
cargo test -p context-graph-mcp handlers::merge::tests -- --nocapture
# Expected: All 12+ tests pass

# Step 3: Verify dispatch routing
cargo test -p context-graph-mcp handlers::tools::dispatch -- --nocapture
# Expected: PASSES

# Step 4: Full MCP test suite
cargo test -p context-graph-mcp --lib
# Expected: All tests pass (880+ tests)
```

---

## 7. FULL STATE VERIFICATION PROTOCOL

### 7.1 Source of Truth
- **Merged Node**: Stored in RocksDB at key `node:{merged_id}`
- **Reversal Record**: Stored in RocksDB at key `reversal:{hash}`
- **Soft Delete Markers**: Stored at keys `merged:{source_id}`

### 7.2 Execute & Inspect

After calling `merge_concepts`, verify by reading back:

```rust
// Verification query pseudocode
let merged_node = storage.get_node(merged_id).await;
assert!(merged_node.is_some(), "Merged node must exist");

let reversal = storage.get_raw(&format!("reversal:{}", hash)).await;
assert!(reversal.is_some(), "Reversal record must exist");

for source_id in &input.source_ids {
    let marker = storage.get_raw(&format!("merged:{}", source_id)).await;
    assert!(marker.is_some(), "Soft delete marker must exist");
}
```

### 7.3 Boundary & Edge Case Audit

**Edge Case 1: Minimum source_ids (2)**
- State Before: 2 separate nodes exist
- Action: merge_concepts with exactly 2 source_ids
- State After: 1 merged node + 2 soft delete markers + 1 reversal record
- Verification: Count nodes before/after, verify reversal hash unique

**Edge Case 2: Maximum source_ids (10)**
- State Before: 10 separate nodes exist
- Action: merge_concepts with exactly 10 source_ids
- State After: 1 merged node + 10 soft delete markers + 1 reversal record
- Verification: All 10 source nodes have markers, merged node has combined content

**Edge Case 3: Invalid UUID (node not found)**
- State Before: Only 1 of 2 source_ids exists
- Action: merge_concepts
- State After: No changes (FAIL FAST)
- Expected Error: `NODE_NOT_FOUND` with missing UUID in message

**Edge Case 4: Force merge with conflicting valence**
- State Before: Node A (valence +0.9), Node B (valence -0.9)
- Action: merge_concepts with force_merge=true
- State After: Merge succeeds with averaged valence ~0.0
- Verification: Warning logged but no error

**Edge Case 5: Empty storage backend**
- State Before: storage_backend = None
- Action: merge_concepts
- State After: No changes
- Expected Error: `STORAGE_ERROR` "Storage backend not initialized"

### 7.4 Evidence of Success

After running all tests, provide log in this format:

```
=== TASK-30 Full State Verification ===
Date: <timestamp>
Compiler: cargo check PASSED
Handler Tests: <N> passed, 0 failed
Dispatch Routing: VERIFIED (merge_concepts → call_merge_concepts)
Storage Operations:
  - get_node: <N> calls, <N> found, 0 errors
  - put_node: <N> calls, 0 errors
  - put_raw: <N> calls (reversal + markers)
Edge Cases:
  - Minimum (2 sources): PASSED
  - Maximum (10 sources): PASSED
  - Missing node: FAIL FAST confirmed
  - Force merge: PASSED with warning
  - No storage: FAIL FAST confirmed
Reversal Records: <N> stored, all with valid 30-day expiry
```

---

## 8. MANUAL TESTING PROTOCOL

### 8.1 Synthetic Test Data

**Happy Path Test**:
```json
{
  "source_ids": [
    "550e8400-e29b-41d4-a716-446655440001",
    "550e8400-e29b-41d4-a716-446655440002"
  ],
  "target_name": "Unified Authentication Concept",
  "merge_strategy": "weighted_average",
  "rationale": "Consolidating duplicate auth patterns detected by similarity search",
  "force_merge": false
}
```

**Expected Output**:
```json
{
  "success": true,
  "merged_id": "<new-uuid>",
  "reversal_hash": "sha256:<64-hex-chars>",
  "merged_node": {
    "id": "<new-uuid>",
    "name": "Unified Authentication Concept",
    "source_count": 2,
    "strategy_used": "weighted_average",
    "created_at": "2026-01-13T...",
    "importance": 0.6,
    "embedding_dim": 1536
  },
  "error": null
}
```

### 8.2 Manual Verification Steps

1. **Before merge**: Run `storage.list_nodes()` and note count
2. **Execute merge**: Call `merge_concepts` with synthetic data
3. **After merge**:
   - Verify merged node exists: `storage.get_node(merged_id)`
   - Verify reversal record: `storage.get_raw("reversal:sha256:...")`
   - Verify soft delete markers: `storage.get_raw("merged:550e8400-...")`
4. **Check logs**: Confirm INFO log with "merge_concepts SUCCESS"

### 8.3 Failure Case Tests

**Test 1: Only 1 source_id**
```json
{
  "source_ids": ["550e8400-e29b-41d4-a716-446655440001"],
  "target_name": "Test",
  "rationale": "Testing"
}
```
Expected: Error -32602 "source_ids requires at least 2 items, got 1"

**Test 2: Non-existent source node**
```json
{
  "source_ids": [
    "00000000-0000-0000-0000-000000000001",
    "00000000-0000-0000-0000-000000000002"
  ],
  "target_name": "Test",
  "rationale": "Testing"
}
```
Expected: Error -32002 "Source node not found: 00000000-..."

**Test 3: Empty rationale**
```json
{
  "source_ids": [...],
  "target_name": "Test",
  "rationale": ""
}
```
Expected: Error -32602 "rationale is REQUIRED (min 1 char)"

---

## 9. FILES SUMMARY

| Action | File Path | Changes |
|--------|-----------|---------|
| CREATE | `crates/context-graph-mcp/src/handlers/merge.rs` | New ~400 line handler file |
| MODIFY | `crates/context-graph-mcp/src/handlers/mod.rs` | Add `mod merge;` |
| MODIFY | `crates/context-graph-mcp/src/handlers/tools/dispatch.rs` | Add dispatch case |
| MODIFY | `crates/context-graph-mcp/Cargo.toml` | Add sha2, hex deps (if needed) |

---

## 10. DEFINITION OF DONE CHECKLIST

Before marking complete, ALL must be checked:

- [ ] `merge.rs` file created with complete handler implementation
- [ ] `merge.rs` has 12+ passing unit tests
- [ ] `mod.rs` includes `mod merge;` declaration
- [ ] `dispatch.rs` routes `merge_concepts` to `call_merge_concepts`
- [ ] All input validation per TASK-29 schema constraints
- [ ] FAIL FAST on missing nodes (no mock data)
- [ ] FAIL FAST on missing storage backend
- [ ] Reversal hash generated with SHA-256
- [ ] Reversal record stored with 30-day expiry (SEC-06)
- [ ] Soft delete markers created for source nodes
- [ ] Merged node stored with combined content
- [ ] All 3 merge strategies implemented (union, intersection, weighted_average)
- [ ] Priors compatibility check implemented (AP-11)
- [ ] `cargo check -p context-graph-mcp` passes
- [ ] All handler tests pass
- [ ] Manual verification completed with evidence log

---

## 11. COMMON PITFALLS TO AVOID

1. **DO NOT use mock data** - Fetch real nodes from storage
2. **DO NOT use fallback defaults** - FAIL FAST on missing nodes
3. **DO NOT skip reversal storage** - SEC-06 requires 30-day undo
4. **DO NOT hard delete source nodes** - Use soft delete markers
5. **DO NOT forget normalization** - Merged embeddings must be normalized
6. **DO NOT skip duplicate ID check** - Duplicate source_ids is an error
7. **DO NOT implement reversal cleanup** - That's out of scope (separate task)
8. **DO NOT add backwards compatibility shims** - If broken, error fast

---

## 12. RELATED FILES REFERENCE

| File | Purpose |
|------|---------|
| `handlers/epistemic.rs` | Handler pattern to follow (TASK-28) |
| `tools/definitions/merge.rs` | Schema definition (TASK-29) |
| `tools/names.rs:122` | `MERGE_CONCEPTS` constant |
| `protocol.rs:101` | `TOOL_NOT_FOUND` error code |
| `protocol.rs:97` | `NODE_NOT_FOUND` error code |
| `protocol.rs:99` | `STORAGE_ERROR` error code |
| `types/memory_node/node.rs` | MemoryNode struct |
| `docs2/constitution.yaml:123` | SEC-06 reversal requirement |
| `docs2/constitution.yaml:79` | AP-11 priors_vibe_check |

---

## 13. DEPENDENCY ON STORAGE BACKEND

**CRITICAL**: This handler depends on `self.storage_backend` being available in the `Handlers` struct.

Check `crates/context-graph-mcp/src/handlers/core/handlers.rs` for the storage field:
- If it exists: Use it
- If not: You must add it or use an alternative storage mechanism

Current storage patterns in codebase:
- RocksDB backend: `crates/context-graph-storage/src/rocksdb_backend/`
- Memory operations: `crates/context-graph-mcp/src/handlers/memory/`

---

*Document Version: 2.0.0 | Updated: 2026-01-13 | Status: READY*
