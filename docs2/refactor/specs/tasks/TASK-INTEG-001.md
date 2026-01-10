# TASK-INTEG-001: Memory MCP Handlers

```xml
<task_spec id="TASK-INTEG-001" version="3.0">
<metadata>
  <title>Implement Memory MCP Tool Handlers</title>
  <status>COMPLETED</status>
  <layer>integration</layer>
  <sequence>21</sequence>
  <implements>
    <requirement_ref>REQ-MCP-MEMORY-01</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="done">TASK-LOGIC-008</task_ref>
    <task_ref status="done">TASK-CORE-006</task_ref>
  </depends_on>
  <estimated_complexity>high</estimated_complexity>
  <estimated_days>3</estimated_days>
  <completed_date>2026-01-09</completed_date>
  <verified_tests>39 passing</verified_tests>
</metadata>

<completion_summary>
  ALL 6 HANDLERS ARE IMPLEMENTED AND TESTED.

  Implementation files:
  - crates/context-graph-mcp/src/handlers/memory.rs (1787 lines)
  - crates/context-graph-mcp/src/handlers/tests/memory.rs (2300+ lines)
  - crates/context-graph-mcp/src/protocol.rs (method constants)
  - crates/context-graph-mcp/src/handlers/core.rs (dispatch routing)

  Implemented handlers (line numbers in memory.rs):
  - handle_memory_store (line 100) - PRE-EXISTING
  - handle_memory_retrieve (line 279) - PRE-EXISTING
  - handle_memory_search (line 380) - PRE-EXISTING
  - handle_memory_delete (line 548) - PRE-EXISTING
  - handle_memory_inject (line 647) - IMPLEMENTED
  - handle_memory_inject_batch (line 832) - IMPLEMENTED
  - handle_memory_search_multi_perspective (line 1060) - IMPLEMENTED
  - handle_memory_compare (line 1284) - IMPLEMENTED
  - handle_memory_batch_compare (line 1447) - IMPLEMENTED
  - handle_memory_similarity_matrix (line 1671) - IMPLEMENTED

  Protocol constants (crates/context-graph-mcp/src/protocol.rs lines 261-271):
  - MEMORY_INJECT: "memory/inject"
  - MEMORY_INJECT_BATCH: "memory/inject_batch"
  - MEMORY_SEARCH_MULTI_PERSPECTIVE: "memory/search_multi_perspective"
  - MEMORY_COMPARE: "memory/compare"
  - MEMORY_BATCH_COMPARE: "memory/batch_compare"
  - MEMORY_SIMILARITY_MATRIX: "memory/similarity_matrix"

  Handler dispatch (crates/context-graph-mcp/src/handlers/core.rs lines 786-807):
  All 6 new handlers are properly routed in CoreHandlers::dispatch_request()

  Test coverage: 39 tests passing
  Run: cargo test -p context-graph-mcp handlers::tests::memory
</completion_summary>

<verification_evidence>
  Run verification command:
  $ cargo test -p context-graph-mcp handlers::tests::memory -- --nocapture

  Expected output (verified 2026-01-09):
  test handlers::tests::memory::test_memory_inject_creates_fingerprint ... ok
  test handlers::tests::memory::test_memory_inject_missing_content_fails ... ok
  test handlers::tests::memory::test_memory_inject_empty_content_fails ... ok
  test handlers::tests::memory::test_memory_inject_batch_creates_multiple_fingerprints ... ok
  test handlers::tests::memory::test_memory_compare_computes_similarity ... ok
  test handlers::tests::memory::test_memory_compare_not_found_fails ... ok
  test handlers::tests::memory::test_memory_batch_compare_one_to_many ... ok
  test handlers::tests::memory::test_memory_similarity_matrix ... ok
  test handlers::tests::memory::test_memory_search_multi_perspective ... ok
  test handlers::tests::memory::test_rocksdb_integration_inject_compare_cycle ... ok
  test handlers::tests::memory::test_rocksdb_integration_similarity_matrix ... ok
  test handlers::tests::memory::real_embedding_tests::test_real_embeddings_produce_correct_dimensions ... ok
  ... (39 total tests)

  test result: ok. 39 passed; 0 failed; 0 ignored
</verification_evidence>

<actual_type_paths>
  <!-- VERIFIED paths as of 2026-01-09 -->
  <type name="TeleologicalMemoryStore" location="crates/context-graph-core/src/traits/teleological_memory_store.rs">
    TRAIT - The core storage interface
  </type>
  <type name="TeleologicalFingerprint" location="crates/context-graph-core/src/types/fingerprint/teleological/types.rs">
    STRUCT - The atomic storage unit
  </type>
  <type name="SemanticFingerprint" location="crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs">
    STRUCT - 13-embedder semantic data
  </type>
  <type name="TeleologicalComparator" location="crates/context-graph-core/src/teleological/comparator.rs">
    STRUCT - Apples-to-apples comparison
  </type>
  <type name="BatchComparator" location="crates/context-graph-core/src/teleological/comparator.rs">
    STRUCT - Parallel batch comparisons
  </type>
  <type name="PurposeVector" location="crates/context-graph-core/src/types/fingerprint/purpose.rs">
    STRUCT - 13D alignment signature
  </type>
  <type name="MultiArrayEmbeddingProvider" location="crates/context-graph-core/src/traits/multi_array_embedding.rs">
    TRAIT - 13-embedder generation
  </type>
</actual_type_paths>

<full_state_verification status="IMPLEMENTED_AND_TESTED">
  <source_of_truth>
    <primary>RocksDB TeleologicalMemoryStore</primary>
    <location>crates/context-graph-storage/src/teleological/rocksdb_store.rs</location>
  </source_of_truth>

  <tests_verify_state>
    test_full_state_verification_crud_cycle (line 566):
    1. Store fingerprint → returns UUID
    2. Retrieve by UUID → verify all 13 embedders populated
    3. Search returns stored fingerprint
    4. Delete removes fingerprint
    5. Retrieve after delete returns None

    test_rocksdb_integration_inject_compare_cycle (line 1928):
    1. Inject content → fingerprint stored
    2. Retrieve → verify all 13 embedder dimensions non-empty
    3. Compare two fingerprints → returns per-embedder breakdown
    4. Similarity matrix → verified symmetric
  </tests_verify_state>

  <edge_cases_tested>
    1. Empty content injection → INVALID_PARAMS error (test_memory_inject_empty_content_fails)
    2. UUID not found → FINGERPRINT_NOT_FOUND error (test_memory_compare_not_found_fails)
    3. Batch empty contents → INVALID_PARAMS error (test_memory_inject_batch_empty_contents_fails)
  </edge_cases_tested>
</full_state_verification>

<architectural_compliance verified="true">
  ARCH-01: TeleologicalFingerprint is atomic - all 13 embedders created together ✓
  ARCH-02: Apples-to-apples via TeleologicalComparator (E1↔E1 only) ✓
  ARCH-03: Autonomous-first - no manual goal setting ✓
  AP-007: FAIL FAST - proper error codes returned immediately ✓
</architectural_compliance>

<error_codes_used>
  INVALID_PARAMS: -32602 (empty content, missing required fields)
  STORAGE_ERROR: -32004 (RocksDB failures)
  EMBEDDING_ERROR: -32005 (embedding generation failures)
  FINGERPRINT_NOT_FOUND: Custom error for UUID lookup failures
</error_codes_used>

<if_extending_this_work>
  DO NOT re-implement these handlers - they are complete.

  If adding new functionality:
  1. Read crates/context-graph-mcp/src/handlers/memory.rs first
  2. Follow the existing handler pattern (see handle_memory_inject at line 647)
  3. Add protocol constant to crates/context-graph-mcp/src/protocol.rs
  4. Add dispatch case to crates/context-graph-mcp/src/handlers/core.rs
  5. Add tests to crates/context-graph-mcp/src/handlers/tests/memory.rs
  6. Run: cargo test -p context-graph-mcp handlers::tests::memory -- --nocapture
</if_extending_this_work>

<next_tasks>
  This task (TASK-INTEG-001) is COMPLETE.

  Next integration tasks per _index.md:
  - TASK-INTEG-002: Purpose/Goal MCP Handlers (uses LOGIC-009, LOGIC-010)
  - TASK-INTEG-003: Consciousness MCP Handlers (uses LOGIC-004)
  - TASK-INTEG-004: Hook Protocol &amp; Core Handlers
</next_tasks>
</task_spec>
```
