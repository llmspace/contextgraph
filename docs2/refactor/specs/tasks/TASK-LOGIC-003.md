# TASK-LOGIC-003: Token-Level Similarity

```xml
<task_spec id="TASK-LOGIC-003" version="2.1">
<metadata>
  <title>Implement Token-Level Similarity Functions (ColBERT MaxSim)</title>
  <status>done</status>
  <layer>logic</layer>
  <sequence>13</sequence>
  <implements>
    <requirement_ref>REQ-SIMILARITY-TOKENLEVEL-01</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-CORE-003</task_ref>
    <task_ref>TASK-LOGIC-001</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <completed_date>2026-01-09</completed_date>
  <verification>137 similarity tests pass including 12 token_level tests</verification>
</metadata>

<context>
Late interaction embeddings (E12) use ColBERT-style token-level representations.
Each token has its own 128D embedding, and similarity is computed via MaxSim:
for each query token, find the maximum similarity to any document token.

From constitution.yaml:
- E12_LateInteraction: dim=128D/token, math=ColBERT_MaxSim
- Stage 5 retrieval: <15ms for 50→10 candidates using MaxSim
- Formula: score = (1/|Q|) × Σᵢ max_j cos(q_i, d_j)
</context>

<objective>
Implement MaxSim and symmetric MaxSim functions for token-level embeddings in
context-graph-core, enabling precision matching for Stage 5 of the 5-stage retrieval pipeline.
</objective>

<source_of_truth>
  <!-- VERIFIED against actual codebase on 2026-01-09 -->
  <verified_files>
    <file path="crates/context-graph-core/src/similarity/mod.rs" status="exists">
      Similarity module - exports token_level submodule and all public functions
    </file>
    <file path="crates/context-graph-core/src/similarity/dense.rs" status="exists">
      Contains cosine_similarity function (used internally by token_level)
    </file>
    <file path="crates/context-graph-core/src/similarity/token_level.rs" status="exists">
      IMPLEMENTED: MaxSim, symmetric_max_sim, approximate_max_sim, token_alignments
    </file>
    <file path="crates/context-graph-core/src/types/fingerprint/mod.rs" status="exists">
      Re-exports E12_TOKEN_DIM from semantic module
    </file>
    <file path="crates/context-graph-core/src/types/fingerprint/semantic/constants.rs" status="exists">
      E12_TOKEN_DIM = 128 (canonical definition)
    </file>
  </verified_files>

  <correct_import_path>
    <!-- Use the public re-export, NOT the internal path -->
    use crate::types::fingerprint::E12_TOKEN_DIM;

    <!-- NOT: use crate::types::fingerprint::semantic::constants::E12_TOKEN_DIM; -->
    <!-- The semantic module is private; constants are re-exported through fingerprint/mod.rs -->
  </correct_import_path>

  <actual_data_type>
    <!-- The E12 late-interaction field in SemanticFingerprint -->
    pub e12_late_interaction: Vec&lt;Vec&lt;f32&gt;&gt;  <!-- [num_tokens][128] -->

    <!-- Each inner Vec is a 128D embedding for one token -->
  </actual_data_type>

  <dimension_constant>
    E12_TOKEN_DIM: usize = 128  <!-- Publicly exported via types::fingerprint -->
  </dimension_constant>
</source_of_truth>

<implementation_status>
  <!-- COMPLETED: All functions implemented and tested -->
  <file path="crates/context-graph-core/src/similarity/token_level.rs">
    - max_sim: ✅ Implemented with dimension validation
    - symmetric_max_sim: ✅ Implemented as (ab + ba) / 2
    - approximate_max_sim: ✅ Implemented with early termination
    - token_alignments: ✅ Implemented with TokenAlignment struct
    - 12 unit tests: ✅ All passing
  </file>

  <module_integration>
    mod.rs already contains:
    - mod token_level;
    - pub use token_level::{approximate_max_sim, max_sim, symmetric_max_sim, token_alignments, TokenAlignment};
  </module_integration>
</implementation_status>

<verification_results>
  <command>cargo test -p context-graph-core similarity::token_level</command>
  <result>12 passed; 0 failed</result>

  <command>cargo test -p context-graph-core similarity::</command>
  <result>137 passed; 0 failed</result>

  <command>cargo build -p context-graph-core</command>
  <result>Succeeded</result>
</verification_results>

<api_reference>
  <!-- Actual function signatures as implemented -->
  <function name="max_sim">
    pub fn max_sim(query: &amp;[Vec&lt;f32&gt;], document: &amp;[Vec&lt;f32&gt;]) -&gt; f32

    Returns: MaxSim score in [0.0, 1.0] range
    Panics: If any token dimension != 128
    Empty: Returns 0.0 for empty sequences
  </function>

  <function name="symmetric_max_sim">
    pub fn symmetric_max_sim(a: &amp;[Vec&lt;f32&gt;], b: &amp;[Vec&lt;f32&gt;]) -&gt; f32

    Returns: (max_sim(a,b) + max_sim(b,a)) / 2
  </function>

  <function name="approximate_max_sim">
    pub fn approximate_max_sim(
        query: &amp;[Vec&lt;f32&gt;],
        document: &amp;[Vec&lt;f32&gt;],
        min_score_threshold: f32,
    ) -&gt; f32

    Returns: Approximate score with early termination
    Constraint: Within 5% of exact for threshold ≥ 0.8
  </function>

  <function name="token_alignments">
    pub fn token_alignments(
        query: &amp;[Vec&lt;f32&gt;],
        document: &amp;[Vec&lt;f32&gt;],
    ) -&gt; Vec&lt;TokenAlignment&gt;

    Returns: One alignment per query token (best-matching doc token)
  </function>

  <struct name="TokenAlignment">
    #[derive(Debug, Clone, PartialEq)]
    pub struct TokenAlignment {
        pub query_token_idx: usize,
        pub doc_token_idx: usize,
        pub similarity: f32,
    }
  </struct>
</api_reference>

<constraints>
  <constraint status="verified">MaxSim score normalized to [0.0, 1.0] range</constraint>
  <constraint status="verified">Empty token sequences return 0.0 (no panic)</constraint>
  <constraint status="verified">Token embeddings MUST have dimension = E12_TOKEN_DIM (128)</constraint>
  <constraint status="verified">Dimension mismatch MUST panic with descriptive error (fail fast)</constraint>
  <constraint status="verified">Approximate version within 5% of exact for threshold ≥ 0.8</constraint>
  <constraint status="verified">All functions are #[inline] for performance</constraint>
</constraints>

<edge_cases_verified>
  <case id="1" status="passed">
    <description>Empty query sequence</description>
    <input>query = [], document = [[1.0; 128]]</input>
    <output>0.0</output>
    <behavior>Return 0.0, no panic</behavior>
  </case>
  <case id="2" status="passed">
    <description>Wrong token dimension</description>
    <input>query = [[1.0; 64]], document = [[1.0; 128]]</input>
    <output>panic!("Query token 0 has dimension 64, expected 128")</output>
    <behavior>Fail fast with descriptive panic</behavior>
  </case>
  <case id="3" status="passed">
    <description>Single token perfect match</description>
    <input>query = [[1.0, 0.0, ..., 0.0]], document = [[1.0, 0.0, ..., 0.0]]</input>
    <output>1.0</output>
    <behavior>Return exactly 1.0 for identical normalized vectors</behavior>
  </case>
</edge_cases_verified>

<notes>
  <note type="critical">
    IMPORT PATH CORRECTION: The task originally specified wrong import paths.

    WRONG (task v2.0):
      use crate::types::fingerprint::semantic::constants::E12_TOKEN_DIM;

    CORRECT (actual implementation):
      use crate::types::fingerprint::E12_TOKEN_DIM;

    The semantic module is private. Constants are re-exported through fingerprint/mod.rs.
  </note>

  <note type="info">
    There is already a maxsim_score implementation in context-graph-embeddings crate
    (late_interaction/scoring.rs), but this task implemented MaxSim in context-graph-core
    for use with SemanticFingerprint data directly.
  </note>
</notes>
</task_spec>
```
