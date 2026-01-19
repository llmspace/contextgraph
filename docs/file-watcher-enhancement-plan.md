# File Watcher Enhancement Plan

## Executive Summary

This plan addresses two critical gaps in the MDFileWatcher system:

1. **Metadata Transparency**: Context injected from file watcher chunks must include source file path in hook outputs
2. **Stale Embedding Cleanup**: When `.md` files are edited/updated, old embeddings must be cleared and replaced with fresh chunks

## Current State Analysis

### What Exists

| Component | Location | Status |
|-----------|----------|--------|
| MDFileWatcher | `crates/context-graph-core/src/memory/watcher.rs` | Complete |
| TextChunker | `crates/context-graph-core/src/memory/chunker.rs` | Complete |
| ChunkMetadata | `crates/context-graph-core/src/memory/mod.rs:64-77` | Complete |
| MemorySource::MDFileChunk | `crates/context-graph-core/src/memory/source.rs:30-37` | Complete |
| SHA256 Change Detection | `watcher.rs:308-331` | Complete |
| MemoryStore | `crates/context-graph-core/src/memory/store.rs` | Complete |

### Current Chunking Parameters (Per Constitution)

```
CHUNK_SIZE_WORDS = 200
OVERLAP_WORDS = 50 (25%)
MIN_CHUNK_WORDS = 50
```

### Current ChunkMetadata Fields

```rust
pub struct ChunkMetadata {
    pub file_path: String,           // Source file path
    pub chunk_index: u32,            // 0-based position
    pub total_chunks: u32,           // Total chunks from this file
    pub word_offset: u32,            // Cumulative word position
    pub char_offset: u32,            // Cumulative character position
    pub original_file_hash: String,  // SHA256 of full file content
}
```

### Gap 1: Metadata Not Shown in Context Injection

**Location**: `crates/context-graph-cli/src/commands/hooks/user_prompt_submit.rs:817-841`

When memories are injected into hook output, only the content is shown:

```rust
// Current behavior (lines 821-841)
injection.push_str(&format!(
    "### Memory {} (similarity: {:.2}, via {})\n",
    i + 1,
    memory.similarity,
    memory.dominant_embedder
));
if let Some(ref content) = memory.content {
    // Only content shown - no file_path metadata
    injection.push_str(&truncated);
}
```

The `RetrievedMemory` struct (lines 461-470) doesn't include source metadata:

```rust
pub struct RetrievedMemory {
    pub id: String,
    pub content: Option<String>,
    pub similarity: f32,
    pub dominant_embedder: String,
    // MISSING: source metadata (file_path, chunk_index, etc.)
}
```

### Gap 2: No Stale Embedding Cleanup

**Problem**: When a file is modified:
1. SHA256 change is detected (working)
2. File is re-chunked (working)
3. New memories are created with NEW UUIDs (working)
4. **OLD memories remain in database** (BUG)

**Location**: `watcher.rs:298-359`

```rust
async fn process_file(&self, path: &Path) -> Result<Vec<Uuid>, WatcherError> {
    // ... read file and check hash ...

    // Hash changed - process file
    let chunks = self.chunker.chunk_text(&content, &path_str)?;

    // Creates NEW memories - OLD ones NOT deleted
    for chunk in chunks {
        let id = self.capture_service.capture_md_chunk(chunk, self.session_id.clone()).await?;
        memory_ids.push(id);
    }
    Ok(memory_ids)
}
```

**Consequence**: Database accumulates stale embeddings that no longer match file content.

### Gap 3: No File-Based Deletion

**Location**: `crates/context-graph-core/src/memory/store.rs`

Current deletion methods:
- `delete(id: Uuid)` - Delete single memory by UUID
- No method to delete all memories by `file_path`

Current indexes:
- `CF_MEMORIES` - Primary storage (UUID -> Memory)
- `CF_SESSION_INDEX` - Session -> Vec<UUID>
- **MISSING**: File path -> Vec<UUID> index

---

## Implementation Plan

### Phase 1: Storage Layer Enhancement

#### Task 1.1: Add File Path Index to MemoryStore

**File**: `crates/context-graph-core/src/memory/store.rs`

Add new column family for file path indexing:

```rust
const CF_FILE_INDEX: &str = "file_index";
```

Update `new()` to create column family:

```rust
let cf_descriptors = vec![
    ColumnFamilyDescriptor::new(CF_MEMORIES, cf_opts.clone()),
    ColumnFamilyDescriptor::new(CF_SESSION_INDEX, cf_opts.clone()),
    ColumnFamilyDescriptor::new(CF_FILE_INDEX, cf_opts),  // NEW
];
```

Update `store()` to maintain file index for MDFileChunk sources:

```rust
// After storing memory, update file index if MDFileChunk
if let MemorySource::MDFileChunk { file_path, .. } = &memory.source {
    self.update_file_index(file_path, memory.id)?;
}
```

**Estimated LOC**: ~80

#### Task 1.2: Add `delete_by_file_path()` Method

**File**: `crates/context-graph-core/src/memory/store.rs`

```rust
/// Delete all memories from a specific file path.
///
/// # Arguments
/// * `file_path` - The file path to delete memories for
///
/// # Returns
/// * `Ok(usize)` - Number of memories deleted
/// * `Err(StorageError)` - Deletion failed
pub fn delete_by_file_path(&self, file_path: &str) -> Result<usize, StorageError> {
    let cf_file_index = self.db.cf_handle(CF_FILE_INDEX)
        .ok_or_else(|| StorageError::ColumnFamilyNotFound(CF_FILE_INDEX.to_string()))?;

    // Get all memory IDs for this file
    let memory_ids: Vec<Uuid> = match self.db.get_cf(cf_file_index, file_path.as_bytes()) {
        Ok(Some(bytes)) => bincode::deserialize(&bytes)?,
        Ok(None) => return Ok(0),
        Err(e) => return Err(StorageError::ReadFailed(e.to_string())),
    };

    // Delete each memory
    let mut deleted = 0;
    for id in memory_ids {
        if self.delete(id)? {
            deleted += 1;
        }
    }

    // Clear file index entry
    self.db.delete_cf(cf_file_index, file_path.as_bytes())?;

    Ok(deleted)
}
```

**Estimated LOC**: ~50

#### Task 1.3: Add `get_by_file_path()` Method

**File**: `crates/context-graph-core/src/memory/store.rs`

```rust
/// Get all memories from a specific file path.
pub fn get_by_file_path(&self, file_path: &str) -> Result<Vec<Memory>, StorageError> {
    // Similar to get_by_session but using file_index
}
```

**Estimated LOC**: ~30

---

### Phase 2: File Watcher Enhancement

#### Task 2.1: Clear Old Embeddings Before Re-chunking

**File**: `crates/context-graph-core/src/memory/watcher.rs`

Update `process_file()`:

```rust
async fn process_file(&self, path: &Path) -> Result<Vec<Uuid>, WatcherError> {
    let content = tokio::fs::read_to_string(path).await.map_err(|e| {
        WatcherError::ReadFailed { path: path.to_path_buf(), source: e }
    })?;

    // Compute hash
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let hash = format!("{:x}", hasher.finalize());

    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let path_str = canonical.to_string_lossy().to_string();

    // Check if content changed
    {
        let hashes = self.file_hashes.read().await;
        if let Some(existing_hash) = hashes.get(&canonical) {
            if existing_hash == &hash {
                debug!(path = ?path, "File unchanged, skipping");
                return Ok(Vec::new());
            }
        }
    }

    // CRITICAL: Delete old embeddings BEFORE re-chunking
    info!(path = %path_str, "File changed - clearing old embeddings");
    let deleted_count = self.capture_service.delete_by_file_path(&path_str).await?;
    debug!(path = %path_str, deleted_count, "Cleared old embeddings");

    // Update hash cache
    {
        let mut hashes = self.file_hashes.write().await;
        hashes.insert(canonical, hash);
    }

    // Chunk and store new content
    let chunks = self.chunker.chunk_text(&content, &path_str)?;
    let mut memory_ids = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        let id = self.capture_service
            .capture_md_chunk(chunk, self.session_id.clone())
            .await?;
        memory_ids.push(id);
    }

    info!(
        path = %path_str,
        chunk_count = memory_ids.len(),
        "File processed - stored fresh embeddings"
    );

    Ok(memory_ids)
}
```

**Estimated LOC**: ~20 (modification of existing)

#### Task 2.2: Add `delete_by_file_path()` to MemoryCaptureService

**File**: `crates/context-graph-core/src/memory/capture.rs`

```rust
/// Delete all memories associated with a file path.
/// Used when file content changes to clear stale embeddings.
pub async fn delete_by_file_path(&self, file_path: &str) -> Result<usize, CaptureError> {
    self.store.delete_by_file_path(file_path)
        .map_err(|e| CaptureError::StorageError(e.to_string()))
}
```

**Estimated LOC**: ~10

---

### Phase 3: Context Injection Enhancement

#### Task 3.1: Update MCP search_graph Response

**File**: `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs`

The MCP `search_graph` tool needs to return source metadata in results. Update the response to include:

```json
{
  "results": [
    {
      "fingerprintId": "uuid",
      "content": "text",
      "similarity": 0.85,
      "dominantEmbedder": "E1_Semantic",
      "source": {
        "type": "MDFileChunk",
        "file_path": "/path/to/file.md",
        "chunk_index": 2,
        "total_chunks": 5
      }
    }
  ]
}
```

**Estimated LOC**: ~30

#### Task 3.2: Update RetrievedMemory Struct

**File**: `crates/context-graph-cli/src/commands/hooks/user_prompt_submit.rs`

```rust
/// A memory retrieved from the knowledge graph.
#[derive(Debug, Clone)]
pub struct RetrievedMemory {
    pub id: String,
    pub content: Option<String>,
    pub similarity: f32,
    pub dominant_embedder: String,
    // NEW: Source metadata
    pub source_type: String,           // "MDFileChunk", "HookDescription", etc.
    pub file_path: Option<String>,     // For MDFileChunk
    pub chunk_index: Option<u32>,      // For MDFileChunk
    pub total_chunks: Option<u32>,     // For MDFileChunk
}
```

**Estimated LOC**: ~10

#### Task 3.3: Update Context Injection Formatting

**File**: `crates/context-graph-cli/src/commands/hooks/user_prompt_submit.rs`

Update `generate_context_injection()` to show source metadata:

```rust
// In the Relevant Memories section
for (i, memory) in retrieved_memories.iter().enumerate() {
    injection.push_str(&format!(
        "### Memory {} (similarity: {:.2}, via {})\n",
        i + 1,
        memory.similarity,
        memory.dominant_embedder
    ));

    // NEW: Show source metadata for MDFileChunk
    if memory.source_type == "MDFileChunk" {
        if let Some(ref file_path) = memory.file_path {
            injection.push_str(&format!(
                "**Source**: `{}` (chunk {}/{})\n\n",
                file_path,
                memory.chunk_index.unwrap_or(0) + 1,
                memory.total_chunks.unwrap_or(1)
            ));
        }
    }

    // Content follows
    if let Some(ref content) = memory.content {
        // ... existing truncation logic
    }
}
```

**Estimated LOC**: ~20

#### Task 3.4: Update parse_search_results()

**File**: `crates/context-graph-cli/src/commands/hooks/user_prompt_submit.rs`

```rust
fn parse_search_results(result: &serde_json::Value) -> Vec<RetrievedMemory> {
    // ... existing code ...

    for item in results {
        // ... existing fields ...

        // Parse source metadata
        let source = item.get("source");
        let source_type = source
            .and_then(|s| s.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string();

        let file_path = source
            .and_then(|s| s.get("file_path"))
            .and_then(|v| v.as_str())
            .map(String::from);

        let chunk_index = source
            .and_then(|s| s.get("chunk_index"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        let total_chunks = source
            .and_then(|s| s.get("total_chunks"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        memories.push(RetrievedMemory {
            id,
            content,
            similarity,
            dominant_embedder,
            source_type,
            file_path,
            chunk_index,
            total_chunks,
        });
    }
    // ...
}
```

**Estimated LOC**: ~25

---

### Phase 4: Integration & Testing

#### Task 4.1: Unit Tests for File Index

**File**: `crates/context-graph-core/src/memory/store.rs` (tests section)

```rust
#[cfg(feature = "test-utils")]
#[test]
fn test_file_index_crud() {
    // Test storing MDFileChunk memories updates file index
    // Test delete_by_file_path removes all chunks
    // Test get_by_file_path retrieves all chunks
}

#[cfg(feature = "test-utils")]
#[test]
fn test_file_reindex_on_update() {
    // Store 3 chunks from file A
    // Delete by file path
    // Store 2 new chunks from file A (simulating re-chunking)
    // Verify only 2 chunks exist
}
```

**Estimated LOC**: ~100

#### Task 4.2: Integration Test for File Update Flow

**File**: `crates/context-graph-core/src/memory/watcher.rs` (tests section)

```rust
#[tokio::test]
async fn test_file_modification_clears_old_embeddings() {
    // 1. Create watcher with test directory
    // 2. Create file.md with initial content
    // 3. Process file - verify N chunks stored
    // 4. Modify file.md content
    // 5. Process file again
    // 6. VERIFY: Only new chunks exist, old ones deleted
    // 7. VERIFY: No duplicate chunks in database
}
```

**Estimated LOC**: ~80

#### Task 4.3: FSV Test for Metadata in Context Injection

**File**: `crates/context-graph-cli/src/commands/hooks/user_prompt_submit.rs` (tests section)

```rust
#[tokio::test]
async fn fsv_context_injection_includes_file_metadata() {
    // Setup MCP server mock returning MDFileChunk results
    // Execute prompt-submit hook
    // Parse context_injection output
    // VERIFY: Contains "Source: `/path/to/file.md` (chunk X/Y)"
}
```

**Estimated LOC**: ~60

---

## Implementation Order

1. **Phase 1** (Storage Layer) - Foundation, must be complete first
   - Task 1.1: Add CF_FILE_INDEX column family
   - Task 1.2: Implement `delete_by_file_path()`
   - Task 1.3: Implement `get_by_file_path()`

2. **Phase 2** (File Watcher) - Depends on Phase 1
   - Task 2.2: Add `delete_by_file_path()` to MemoryCaptureService
   - Task 2.1: Update `process_file()` to clear old embeddings

3. **Phase 3** (Context Injection) - Independent, can parallel Phase 2
   - Task 3.1: Update MCP search_graph response
   - Task 3.2: Update RetrievedMemory struct
   - Task 3.3: Update context injection formatting
   - Task 3.4: Update parse_search_results()

4. **Phase 4** (Testing) - After Phases 1-3
   - Task 4.1: Unit tests
   - Task 4.2: Integration tests
   - Task 4.3: FSV tests

---

## File Changes Summary

| File | Type | Changes |
|------|------|---------|
| `crates/context-graph-core/src/memory/store.rs` | Modify | Add file index CF, delete_by_file_path(), get_by_file_path() |
| `crates/context-graph-core/src/memory/capture.rs` | Modify | Add delete_by_file_path() passthrough |
| `crates/context-graph-core/src/memory/watcher.rs` | Modify | Call delete before re-chunking |
| `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs` | Modify | Include source metadata in search results |
| `crates/context-graph-cli/src/commands/hooks/user_prompt_submit.rs` | Modify | Parse and display source metadata |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| File index adds storage overhead | Medium | Low | Index is small (file_path -> Vec<Uuid>) |
| Migration of existing data | Low | Medium | New column family auto-creates; existing memories without file_path won't have index entry |
| Delete race condition | Low | Low | File watcher is single-threaded per session |
| Performance impact on large repos | Medium | Medium | File index lookup is O(1); batch delete is O(n) where n = chunks per file |

---

## Constitution Compliance

| Rule | How Addressed |
|------|--------------|
| ARCH-01 | TeleologicalArray remains atomic - no changes |
| ARCH-11 | MDFileChunk source preserved with full metadata |
| AP-14 | All new code uses Result, no unwrap() |
| SEC-06 | Soft delete not affected - this is hard delete by design for stale embeddings |

---

## Definition of Done

- [ ] File index column family created and maintained
- [ ] `delete_by_file_path()` implemented and tested
- [ ] File watcher clears old embeddings before re-chunking
- [ ] MCP search_graph returns source metadata
- [ ] Context injection shows file path for MDFileChunk memories
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] FSV tests verify end-to-end behavior
- [ ] No regressions in existing tests
