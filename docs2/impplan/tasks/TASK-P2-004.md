# TASK-P2-004: DimensionValidator Enhancement

```xml
<task_spec id="TASK-P2-004" version="2.0">
<metadata>
  <title>DimensionValidator Enhancement with Structured Errors</title>
  <status>ready</status>
  <layer>logic</layer>
  <sequence>17</sequence>
  <phase>2</phase>
  <implements>
    <requirement_ref>REQ-P2-04</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="COMPLETE">TASK-P2-002</task_ref>
    <task_ref status="COMPLETE">TASK-P2-003</task_ref>
    <task_ref status="COMPLETE">TASK-P2-003b</task_ref>
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
</metadata>

<critical_audit date="2025-01-16">
  VALIDATION ALREADY EXISTS but needs enhancement.

  Current Implementation Location:
    crates/context-graph-core/src/types/fingerprint/semantic/validation.rs

  Current State:
    - SemanticFingerprint::validate() method exists (lines 32-47)
    - Returns Result&lt;(), String&gt; - needs upgrade to typed errors
    - Per-embedder validation methods exist (validate_e1 through validate_e13)
    - Tests exist in tests/validation_tests.rs

  TASK SCOPE CHANGE: This task is now about ENHANCING the existing validation
  with proper typed errors (ValidationError enum) and improving error context.
</critical_audit>

<context>
Enhances the existing dimension validation on SemanticFingerprint/TeleologicalArray
with a proper typed error enum following thiserror patterns. The current implementation
returns String errors - this task upgrades to ValidationError for better programmatic
error handling and fail-fast diagnostics.

Per constitution.yaml:
  - ARCH-01: "TeleologicalArray is atomic - store all 13 embeddings or nothing"
  - ARCH-05: "All 13 embedders required - missing embedder is fatal error"
  - AP-14: "No .unwrap() in library code"
</context>

<source_of_truth>
  ACTUAL File Paths (verified 2025-01-16):
    - crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs
    - crates/context-graph-core/src/types/fingerprint/semantic/validation.rs
    - crates/context-graph-core/src/types/fingerprint/semantic/constants.rs
    - crates/context-graph-core/src/types/fingerprint/semantic/tests/validation_tests.rs
    - crates/context-graph-core/src/teleological/embedder.rs

  ACTUAL Type Names (verified):
    - SemanticFingerprint (alias: TeleologicalArray)
    - Embedder::Semantic (NOT E1Semantic)
    - Embedder::TemporalRecent (NOT E2TempRecent)
    - Embedder::TemporalPeriodic
    - Embedder::TemporalPositional
    - Embedder::Causal
    - Embedder::Sparse
    - Embedder::Code
    - Embedder::Emotional (NOT Graph - deprecated)
    - Embedder::Hdc
    - Embedder::Multimodal
    - Embedder::Entity
    - Embedder::LateInteraction
    - Embedder::KeywordSplade

  ACTUAL Field Names in SemanticFingerprint:
    - e1_semantic: Vec&lt;f32&gt;        (1024D)
    - e2_temporal_recent: Vec&lt;f32&gt; (512D)
    - e3_temporal_periodic: Vec&lt;f32&gt; (512D)
    - e4_temporal_positional: Vec&lt;f32&gt; (512D)
    - e5_causal: Vec&lt;f32&gt;          (768D)
    - e6_sparse: SparseVector       (~30522 vocab)
    - e7_code: Vec&lt;f32&gt;            (1536D)
    - e8_graph: Vec&lt;f32&gt;           (384D) - NOTE: field is "graph" not "emotional"
    - e9_hdc: Vec&lt;f32&gt;             (1024D projected dense, NOT binary)
    - e10_multimodal: Vec&lt;f32&gt;     (768D)
    - e11_entity: Vec&lt;f32&gt;         (384D)
    - e12_late_interaction: Vec&lt;Vec&lt;f32&gt;&gt; (128D per token)
    - e13_splade: SparseVector      (~30522 vocab)

  ACTUAL Dimension Constants (from constants.rs):
    E1_DIM = 1024
    E2_DIM = 512
    E3_DIM = 512
    E4_DIM = 512
    E5_DIM = 768
    E6_SPARSE_VOCAB = 30_522
    E7_DIM = 1536
    E8_DIM = 384
    E9_DIM = 1024 (projected, NOT 10K binary)
    E10_DIM = 768
    E11_DIM = 384
    E12_TOKEN_DIM = 128
    E13_SPLADE_VOCAB = 30_522
</source_of_truth>

<input_context_files>
  <file purpose="existing_validation">crates/context-graph-core/src/types/fingerprint/semantic/validation.rs</file>
  <file purpose="fingerprint_struct">crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs</file>
  <file purpose="constants">crates/context-graph-core/src/types/fingerprint/semantic/constants.rs</file>
  <file purpose="embedder_enum">crates/context-graph-core/src/teleological/embedder.rs</file>
  <file purpose="existing_tests">crates/context-graph-core/src/types/fingerprint/semantic/tests/validation_tests.rs</file>
</input_context_files>

<prerequisites>
  <check status="VERIFIED">TASK-P2-002 complete - DenseVector, BinaryVector, SparseVector exist in embeddings/vector.rs</check>
  <check status="VERIFIED">TASK-P2-003 complete - EmbedderConfig registry exists in embeddings/config.rs</check>
  <check status="VERIFIED">TASK-P2-003b complete - EmbedderCategory exists in embeddings/category.rs</check>
  <check status="VERIFIED">SemanticFingerprint::validate() already exists but returns String</check>
</prerequisites>

<scope>
  <in_scope>
    - Create ValidationError enum in fingerprint.rs (already partially defined!)
    - Migrate validate() from Result&lt;(), String&gt; to Result&lt;(), ValidationError&gt;
    - Add Embedder field to all error variants for better context
    - Add validate_strict() that returns first error (current behavior)
    - Add validate_all() that collects ALL validation errors
    - Ensure error messages include expected vs actual values
  </in_scope>
  <out_of_scope>
    - Creating new validator.rs file (validation stays in validation.rs)
    - Changing SemanticFingerprint structure
    - Adding new embedders
    - Runtime dimension changes
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs">
      // ValidationError ALREADY partially exists at lines 53-101
      // Enhance to include all cases:

      #[derive(Debug, Clone, Error)]
      pub enum ValidationError {
          #[error("Dimension mismatch for {embedder}: expected {expected}, got {actual}")]
          DimensionMismatch {
              embedder: Embedder,
              expected: usize,
              actual: usize,
          },

          #[error("Empty dense embedding for {embedder}: expected {expected} dimensions")]
          EmptyDenseEmbedding {
              embedder: Embedder,
              expected: usize,
          },

          #[error("Sparse vector validation failed for {embedder}: {source}")]
          SparseVectorError {
              embedder: Embedder,
              #[source]
              source: SparseVectorError,
          },

          #[error("Token {token_index} dimension mismatch for {embedder}: expected {expected}, got {actual}")]
          TokenDimensionMismatch {
              embedder: Embedder,
              token_index: usize,
              expected: usize,
              actual: usize,
          },

          #[error("Sparse index {index} exceeds vocabulary size {vocab_size} for {embedder}")]
          SparseIndexOutOfBounds {
              embedder: Embedder,
              index: u32,
              vocab_size: usize,
          },

          #[error("Sparse indices ({indices_len}) and values ({values_len}) length mismatch for {embedder}")]
          SparseIndicesValuesMismatch {
              embedder: Embedder,
              indices_len: usize,
              values_len: usize,
          },
      }
    </signature>

    <signature file="crates/context-graph-core/src/types/fingerprint/semantic/validation.rs">
      impl SemanticFingerprint {
          /// Validate and return first error (fail-fast).
          pub fn validate(&amp;self) -> Result&lt;(), ValidationError&gt;;

          /// Validate and collect ALL errors.
          pub fn validate_all(&amp;self) -> Result&lt;(), Vec&lt;ValidationError&gt;&gt;;
      }
    </signature>
  </signatures>

  <constraints>
    - Validation fails fast on first error (validate())
    - All dimension constants come from constants.rs
    - Sparse validation checks index bounds AND indices/values length match
    - E12 validates each token's dimension (128D)
    - Error messages include embedder name for debugging
    - validate_all() continues after errors to find all problems
  </constraints>

  <verification>
    - Valid SemanticFingerprint::zeroed() passes validation
    - Wrong dimension fails with DimensionMismatch including embedder name
    - Sparse index &gt;= 30522 fails with SparseIndexOutOfBounds
    - E12 token with wrong dimension fails with TokenDimensionMismatch
    - validate_all() on fingerprint with multiple errors returns all of them
  </verification>
</definition_of_done>

<implementation_guide>
Step 1: Read existing code
  - Read fingerprint.rs lines 49-101 (ValidationError already exists!)
  - Read validation.rs (current String-based impl)
  - Understand current test coverage in validation_tests.rs

Step 2: Enhance ValidationError enum
  - Add SparseIndexOutOfBounds variant
  - Add SparseIndicesValuesMismatch variant
  - Ensure all variants have embedder field

Step 3: Migrate validation.rs
  - Change validate() return type from Result&lt;(), String&gt; to Result&lt;(), ValidationError&gt;
  - Update validate_e1() through validate_e13() to return ValidationError
  - Add validate_all() method

Step 4: Update tests
  - Migrate existing tests to use ValidationError matching
  - Add tests for each error variant
  - Add validate_all() tests with multiple errors

Step 5: Run Full State Verification
</implementation_guide>

<full_state_verification>
  <source_of_truth_verification>
    <check>Read fingerprint.rs to confirm ValidationError location</check>
    <check>Read validation.rs to confirm current signature</check>
    <check>Verify constants.rs has all dimension constants</check>
    <check>Run: cargo doc --package context-graph-core --open, search for ValidationError</check>
  </source_of_truth_verification>

  <execute_and_inspect>
    <step>cargo check --package context-graph-core 2>&amp;1 | head -50</step>
    <step>cargo test --package context-graph-core fingerprint::semantic::tests::validation -- --nocapture</step>
    <step>Verify test output shows ValidationError variants, not String errors</step>
  </execute_and_inspect>

  <boundary_edge_cases>
    <case name="empty_dense_e1">
      Input: fingerprint with e1_semantic = vec![]
      Expected: ValidationError::DimensionMismatch { embedder: Embedder::Semantic, expected: 1024, actual: 0 }
    </case>
    <case name="wrong_dimension_e7">
      Input: fingerprint with e7_code = vec![0.0; 1024] (wrong: should be 1536)
      Expected: ValidationError::DimensionMismatch { embedder: Embedder::Code, expected: 1536, actual: 1024 }
    </case>
    <case name="sparse_index_overflow_e6">
      Input: e6_sparse with index 40000 (exceeds 30522)
      Expected: ValidationError::SparseIndexOutOfBounds { embedder: Embedder::Sparse, index: 40000, vocab_size: 30522 }
    </case>
    <case name="sparse_length_mismatch_e13">
      Input: e13_splade with 5 indices but 3 values
      Expected: ValidationError::SparseIndicesValuesMismatch { embedder: Embedder::KeywordSplade, indices_len: 5, values_len: 3 }
    </case>
    <case name="e12_token_wrong_dim">
      Input: e12_late_interaction = vec![vec![0.0; 64]] (should be 128)
      Expected: ValidationError::TokenDimensionMismatch { embedder: Embedder::LateInteraction, token_index: 0, expected: 128, actual: 64 }
    </case>
    <case name="e9_is_dense_not_binary">
      Input: e9_hdc = vec![0.0; 512] (wrong: should be 1024 projected dense)
      Expected: ValidationError::DimensionMismatch { embedder: Embedder::Hdc, expected: 1024, actual: 512 }
    </case>
    <case name="multiple_errors">
      Input: fingerprint with wrong e1, e5, and e7 dimensions
      Using validate_all(): Returns Vec with 3 DimensionMismatch errors
      Using validate(): Returns only first error (e1)
    </case>
    <case name="zeroed_passes">
      Input: SemanticFingerprint::zeroed()
      Expected: Ok(())
    </case>
  </boundary_edge_cases>

  <evidence_log>
    Required Evidence Before Marking Complete:

    1. [ ] cargo test output showing all validation tests pass
    2. [ ] Screenshot/log of validate() returning ValidationError (not String)
    3. [ ] Test demonstrating validate_all() returns multiple errors
    4. [ ] cargo doc output showing ValidationError has all variants
    5. [ ] Boundary case test for each case listed above
    6. [ ] No compilation warnings from cargo check
  </evidence_log>

  <manual_verification>
    After implementation, manually run:

    ```bash
    # Build and test
    cd /home/cabdru/contextgraph
    cargo build --package context-graph-core
    cargo test --package context-graph-core fingerprint::semantic::tests::validation

    # Verify ValidationError is exported
    cargo doc --package context-graph-core --no-deps
    grep -r "ValidationError" target/doc/context_graph_core/

    # Check no String returns in validation.rs
    grep -n "Result<(), String>" crates/context-graph-core/src/types/fingerprint/semantic/validation.rs
    # Expected: no matches after implementation
    ```
  </manual_verification>
</full_state_verification>

<files_to_modify>
  <file path="crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs">
    Add missing ValidationError variants (SparseIndexOutOfBounds, SparseIndicesValuesMismatch)
  </file>
  <file path="crates/context-graph-core/src/types/fingerprint/semantic/validation.rs">
    Change validate() return type, update helper methods, add validate_all()
  </file>
  <file path="crates/context-graph-core/src/types/fingerprint/semantic/tests/validation_tests.rs">
    Update tests to use ValidationError matching, add boundary case tests
  </file>
</files_to_modify>

<validation_criteria>
  <criterion>validate() returns ValidationError enum, not String</criterion>
  <criterion>All ValidationError variants include embedder field</criterion>
  <criterion>validate_all() collects and returns all errors</criterion>
  <criterion>All boundary cases from edge_cases section have tests</criterion>
  <criterion>Existing tests still pass after migration</criterion>
  <criterion>No .unwrap() in validation code (per AP-14)</criterion>
</validation_criteria>

<test_commands>
  <command description="Run validation tests">cargo test --package context-graph-core fingerprint::semantic::tests::validation -- --nocapture</command>
  <command description="Check compilation">cargo check --package context-graph-core</command>
  <command description="Check for warnings">cargo clippy --package context-graph-core -- -D warnings</command>
  <command description="Generate docs">cargo doc --package context-graph-core --no-deps</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [ ] Read existing ValidationError in fingerprint.rs (lines 49-101)
- [ ] Read existing validation.rs implementation
- [ ] Add SparseIndexOutOfBounds variant to ValidationError
- [ ] Add SparseIndicesValuesMismatch variant to ValidationError
- [ ] Change validate() return type from `Result<(), String>` to `Result<(), ValidationError>`
- [ ] Update validate_e1() through validate_e13() to return ValidationError
- [ ] Add validate_all() method that collects all errors
- [ ] Update validation_tests.rs to match on ValidationError
- [ ] Add boundary case tests for each edge case
- [ ] Run cargo test and verify all pass
- [ ] Run cargo clippy and fix any warnings
- [ ] Complete evidence log checklist
- [ ] Mark task COMPLETE

## Key Corrections from Original Task

| Original (WRONG) | Corrected |
|------------------|-----------|
| `embedding/validator.rs` | `types/fingerprint/semantic/validation.rs` (exists!) |
| `Embedder::E1Semantic` | `Embedder::Semantic` |
| `Embedder::E8Emotional` | `Embedder::Emotional` (field is `e8_graph`) |
| `BinaryVector` for E9 | E9 is 1024D dense Vec<f32> (projected) |
| `e12_late_interact` | `e12_late_interaction` |
| Create new file | Modify existing validation.rs |
| `TeleologicalArray::new()` | `SemanticFingerprint::zeroed()` (test-only) |
| ValidationError doesn't exist | ValidationError already exists at fingerprint.rs:53-101 |
