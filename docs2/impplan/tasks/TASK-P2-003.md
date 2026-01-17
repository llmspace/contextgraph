# TASK-P2-003: EmbedderConfig Registry

```xml
<task_spec id="TASK-P2-003" version="3.0">
<metadata>
  <title>EmbedderConfig Registry Implementation</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>17</sequence>
  <phase>2</phase>
  <implements>
    <requirement_ref>REQ-P2-02</requirement_ref>
    <requirement_ref>REQ-P2-05</requirement_ref>
  </implements>
  <depends_on>
    <dependency status="COMPLETE">TASK-P2-001 (Embedder enum)</dependency>
    <dependency status="COMPLETE">TASK-P2-002 (Vector types)</dependency>
    <dependency status="REQUIRED">TASK-P2-003b (EmbedderCategory enum)</dependency>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <audit_date>2026-01-16</audit_date>
</metadata>

<!-- =================================================================== -->
<!-- CRITICAL: CODEBASE AUDIT FINDINGS - READ FIRST                      -->
<!-- =================================================================== -->
<audit_findings>
  <finding severity="CRITICAL" id="AF-001">
    <issue>EmbedderConfig DOES NOT EXIST in the codebase</issue>
    <evidence>
      Grep search found no "EmbedderConfig" struct in Rust files.
      Must create new file.
    </evidence>
    <resolution>Create new config.rs file in embeddings directory.</resolution>
  </finding>

  <finding severity="CRITICAL" id="AF-002">
    <issue>File path in original spec is WRONG</issue>
    <details>
      Original spec says: crates/context-graph-core/src/embedding/config.rs
      ACTUAL directory is: "embeddings" (with 's'), NOT "embedding"
    </details>
    <resolution>
      Create at: crates/context-graph-core/src/embeddings/config.rs
    </resolution>
  </finding>

  <finding severity="HIGH" id="AF-003">
    <issue>DistanceMetric ALREADY EXISTS in multiple places</issue>
    <locations>
      <location path="crates/context-graph-storage/src/teleological/indexes/hnsw_config/distance.rs">
        Variants: Cosine, DotProduct, Euclidean, AsymmetricCosine, MaxSim
      </location>
      <location path="crates/context-graph-core/src/index/config.rs">
        Same variants as above (mirrored to avoid cyclic deps)
      </location>
    </locations>
    <resolution>
      OPTION A (RECOMMENDED): Re-export existing DistanceMetric from index/config.rs
      OPTION B: Add Jaccard, Hamming, TransE to existing enum
      DO NOT: Create a duplicate DistanceMetric in embeddings/config.rs
    </resolution>
  </finding>

  <finding severity="HIGH" id="AF-004">
    <issue>Embedder variant names in original task are WRONG</issue>
    <details>
      Task says: Embedder::E1Semantic, Embedder::E2TempRecent, Embedder::E5Causal
      ACTUAL codebase has: Embedder::Semantic, Embedder::TemporalRecent, Embedder::Causal
      (No E-prefix in variant names - E-prefix is via short_name() method)
    </details>
    <resolution>Use ACTUAL variant names from teleological/embedder.rs</resolution>
  </finding>

  <finding severity="HIGH" id="AF-005">
    <issue>E9 HDC is NOT stored as Binary</issue>
    <details>
      SemanticFingerprint has: pub e9_hdc: Vec&lt;f32&gt; (1024D dense after projection)
      NOT BinaryVector. The 10K-bit hypervector is PROJECTED to 1024D dense.
      Therefore Hamming distance does NOT apply.
    </details>
    <resolution>
      E9 should use: Cosine distance, dimension 1024, PQ8 quantization
      Hamming is WRONG for the projected representation.
    </resolution>
  </finding>

  <finding severity="MEDIUM" id="AF-006">
    <issue>Existing dimension constants already defined</issue>
    <locations>
      <location path="crates/context-graph-core/src/types/fingerprint/semantic/constants.rs">
        E1_DIM=1024, E2_DIM=512, E5_DIM=768, E7_DIM=1536, etc.
      </location>
      <location path="crates/context-graph-core/src/index/config.rs">
        Same constants duplicated
      </location>
    </locations>
    <resolution>Use existing constants instead of hardcoding values.</resolution>
  </finding>

  <finding severity="MEDIUM" id="AF-007">
    <issue>EmbedderIndex enum already exists with similar purpose</issue>
    <location path="crates/context-graph-core/src/index/config.rs">
      EmbedderIndex: E1Semantic, E1Matryoshka128, E2TemporalRecent, etc.
      Has methods: uses_hnsw(), uses_inverted_index(), dimension(), recommended_metric()
    </location>
    <resolution>
      EmbedderConfig is COMPLEMENTARY to EmbedderIndex:
      - EmbedderIndex: For index selection (HNSW vs inverted)
      - EmbedderConfig: For embedding configuration (category, quantization)
      Consider referencing EmbedderIndex where appropriate.
    </resolution>
  </finding>

  <finding severity="MEDIUM" id="AF-008">
    <issue>SPLADE vocab size is 30,522 not 30,000</issue>
    <evidence>
      types/fingerprint/sparse.rs: SPARSE_VOCAB_SIZE = 30_522
      index/config.rs: E6_SPARSE_VOCAB = 30_522, E13_SPLADE_VOCAB = 30_522
    </evidence>
    <resolution>Use exact value 30_522 for E6 and E13 dimensions.</resolution>
  </finding>
</audit_findings>

<!-- =================================================================== -->
<!-- CONTEXT - WHAT THIS TASK IS FOR                                     -->
<!-- =================================================================== -->
<context>
  <summary>
    Implements the EmbedderConfig struct that provides static configuration for all 13 embedders.
    This includes dimension, distance metric, quantization method, category classification,
    and topic weight. The registry uses compile-time constants.
  </summary>

  <what_exists>
    <item name="Embedder enum" path="crates/context-graph-core/src/teleological/embedder.rs" status="COMPLETE">
      13 variants with index(), expected_dims(), is_dense(), is_sparse(), is_token_level()
    </item>
    <item name="EmbedderCategory" path="crates/context-graph-core/src/embeddings/category.rs" status="REQUIRED">
      Must be created by TASK-P2-003b first
    </item>
    <item name="DistanceMetric" path="crates/context-graph-core/src/index/config.rs" status="COMPLETE">
      Cosine, DotProduct, Euclidean, AsymmetricCosine, MaxSim
      Missing: Jaccard, Hamming
    </item>
    <item name="EmbedderIndex" path="crates/context-graph-core/src/index/config.rs" status="COMPLETE">
      Index-specific embedder enum with HNSW config
    </item>
    <item name="Dimension constants" path="crates/context-graph-core/src/types/fingerprint/semantic/constants.rs" status="COMPLETE">
      E1_DIM through E13 dimensions
    </item>
  </what_exists>

  <what_needs_creation>
    <item name="EmbedderConfig struct" priority="HIGH">
      Comprehensive config for each embedder including category
    </item>
    <item name="QuantizationConfig enum" priority="HIGH">
      PQ8, Float8, Binary, Inverted, None variants
    </item>
    <item name="EMBEDDER_CONFIGS static array" priority="HIGH">
      All 13 configurations with correct category assignments
    </item>
    <item name="Getter functions" priority="MEDIUM">
      get_config(), get_dimension(), get_topic_weight(), etc.
    </item>
  </what_needs_creation>

  <decision_required id="DR-001">
    <question>How to handle missing DistanceMetric variants (Jaccard, Hamming)?</question>
    <options>
      <option id="A" recommended="true">
        Extend existing DistanceMetric in index/config.rs with Jaccard and update is_hnsw_compatible()
      </option>
      <option id="B">
        Create separate EmbedderDistanceMetric enum just for embeddings module
      </option>
    </options>
    <rationale>Option A maintains single source of truth and proper HNSW compatibility.</rationale>
  </decision_required>
</context>

<!-- =================================================================== -->
<!-- ACTUAL FILE LOCATIONS (CORRECTED)                                   -->
<!-- =================================================================== -->
<actual_file_locations>
  <file purpose="NEW - EmbedderConfig struct and registry"
        path="crates/context-graph-core/src/embeddings/config.rs"
        action="CREATE">
    Create this file with EmbedderConfig, QuantizationConfig, and getters.
  </file>

  <file purpose="embeddings module"
        path="crates/context-graph-core/src/embeddings/mod.rs"
        action="MODIFY">
    Add: pub mod config; and re-export types.
    After TASK-P2-003b it should have: category, token_pruning, vector modules
  </file>

  <file purpose="Extend DistanceMetric"
        path="crates/context-graph-core/src/index/config.rs"
        action="MODIFY">
    Add Jaccard variant to existing DistanceMetric enum.
    Update is_hnsw_compatible() to return false for Jaccard.
  </file>

  <file purpose="Reference - Embedder enum"
        path="crates/context-graph-core/src/teleological/embedder.rs"
        action="READ_ONLY">
    Use actual variant names: Semantic, TemporalRecent, Causal, etc.
  </file>

  <file purpose="Reference - Dimension constants"
        path="crates/context-graph-core/src/types/fingerprint/semantic/constants.rs"
        action="READ_ONLY">
    Import dimension constants from here.
  </file>
</actual_file_locations>

<!-- =================================================================== -->
<!-- CORRECTED EMBEDDER CONFIGURATION TABLE                              -->
<!-- =================================================================== -->
<corrected_embedder_table>
  | Embedder | Variant Name | Dimension | Distance | Quantization | Category | Sparse | Asymmetric |
  |----------|--------------|-----------|----------|--------------|----------|--------|------------|
  | E1 | Semantic | 1024 | Cosine | PQ8(32,8) | Semantic | false | false |
  | E2 | TemporalRecent | 512 | Cosine | Float8 | Temporal | false | false |
  | E3 | TemporalPeriodic | 512 | Cosine | Float8 | Temporal | false | false |
  | E4 | TemporalPositional | 512 | Cosine | Float8 | Temporal | false | false |
  | E5 | Causal | 768 | AsymmetricCosine | PQ8(24,8) | Semantic | false | TRUE |
  | E6 | Sparse | 30522 | Jaccard | Inverted | Semantic | TRUE | false |
  | E7 | Code | 1536 | Cosine | PQ8(48,8) | Semantic | false | false |
  | E8 | Emotional | 384 | Cosine | Float8 | Relational | false | false |
  | E9 | Hdc | 1024 | Cosine | PQ8(32,8) | Structural | false | false |
  | E10 | Multimodal | 768 | Cosine | PQ8(24,8) | Semantic | false | false |
  | E11 | Entity | 384 | Cosine | Float8 | Relational | false | false |
  | E12 | LateInteraction | 128/tok | MaxSim | Float8 | Semantic | false | false |
  | E13 | KeywordSplade | 30522 | Jaccard | Inverted | Semantic | TRUE | false |

  CORRECTIONS FROM ORIGINAL TASK:
  1. E5 uses AsymmetricCosine (existing variant), not plain Cosine
  2. E9 uses Cosine (projected dense), NOT Hamming (was wrong)
  3. E11 uses Cosine (standard), NOT TransE (TransE is for KG training, not retrieval)
  4. E6/E13 dimension is 30522 (exact vocab size), not 30000
</corrected_embedder_table>

<!-- =================================================================== -->
<!-- SCOPE - EXACTLY WHAT TO DO                                          -->
<!-- =================================================================== -->
<scope>
  <in_scope>
    <item>Add Jaccard variant to existing DistanceMetric enum in index/config.rs</item>
    <item>Create QuantizationConfig enum in embeddings/config.rs</item>
    <item>Create EmbedderConfig struct with: embedder, dimension, distance_metric, quantization, category, is_asymmetric, is_sparse</item>
    <item>Create static EMBEDDER_CONFIGS array with all 13 configs</item>
    <item>Implement get_config(), get_dimension(), get_distance_metric(), get_category(), get_topic_weight(), is_asymmetric(), is_sparse(), is_semantic(), is_temporal()</item>
    <item>Import dimension constants from types/fingerprint/semantic/constants.rs</item>
    <item>Comprehensive unit tests including category verification</item>
  </in_scope>

  <out_of_scope>
    <item>Runtime configuration changes</item>
    <item>Model loading</item>
    <item>EmbedderCategory definition (TASK-P2-003b)</item>
    <item>Actual embedding computation</item>
  </out_of_scope>
</scope>

<!-- =================================================================== -->
<!-- IMPLEMENTATION REQUIREMENTS                                         -->
<!-- =================================================================== -->
<implementation_requirements>
  <requirement id="IR-001" priority="CRITICAL">
    <rule>Use ACTUAL Embedder variant names</rule>
    <mapping>
      Embedder::Semantic (NOT E1Semantic)
      Embedder::TemporalRecent (NOT E2TempRecent)
      Embedder::Causal (NOT E5Causal)
      etc.
    </mapping>
  </requirement>

  <requirement id="IR-002" priority="CRITICAL">
    <rule>E9 HDC uses Cosine distance, NOT Hamming</rule>
    <rationale>
      The e9_hdc field in SemanticFingerprint is Vec&lt;f32&gt; (1024D dense).
      The 10K-bit hypervector is projected to dense, so Hamming is wrong.
    </rationale>
  </requirement>

  <requirement id="IR-003" priority="CRITICAL">
    <rule>E5 uses AsymmetricCosine (existing variant)</rule>
    <rationale>
      E5 Causal embeddings are asymmetric (cause->effect != effect->cause).
      Use the existing AsymmetricCosine variant in DistanceMetric.
    </rationale>
  </requirement>

  <requirement id="IR-004" priority="HIGH">
    <rule>Import dimension constants instead of hardcoding</rule>
    <source>crate::types::fingerprint::semantic::constants</source>
  </requirement>

  <requirement id="IR-005" priority="HIGH">
    <rule>E6/E13 dimension is 30_522 (exact BERT vocab size)</rule>
  </requirement>

  <requirement id="IR-006" priority="HIGH">
    <rule>Use existing DistanceMetric from index::config, extend with Jaccard</rule>
    <rationale>Single source of truth for distance metrics.</rationale>
  </requirement>

  <requirement id="IR-007" priority="HIGH">
    <rule>Category assignments MUST match TASK-P2-003b exactly</rule>
    <mapping>
      Semantic: E1, E5, E6, E7, E10, E12, E13 (7 embedders)
      Temporal: E2, E3, E4 (3 embedders)
      Relational: E8, E11 (2 embedders)
      Structural: E9 (1 embedder)
    </mapping>
  </requirement>
</implementation_requirements>

<!-- =================================================================== -->
<!-- DEFINITION OF DONE                                                  -->
<!-- =================================================================== -->
<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/embeddings/config.rs">
use serde::{Deserialize, Serialize};
use crate::teleological::Embedder;
use crate::index::config::DistanceMetric;
use crate::embeddings::category::{EmbedderCategory, category_for};
use crate::types::fingerprint::semantic::constants::*;

/// Quantization configuration for embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationConfig {
    /// Product Quantization with 8-bit codes
    PQ8 { num_subvectors: usize, bits_per_code: usize },
    /// 8-bit floating point
    Float8,
    /// Binary (1-bit per dimension)
    Binary,
    /// Inverted index for sparse vectors
    Inverted,
    /// No quantization (full precision)
    None,
}

/// Static configuration for an embedder
#[derive(Debug, Clone, Copy)]
pub struct EmbedderConfig {
    pub embedder: Embedder,
    pub dimension: usize,
    pub distance_metric: DistanceMetric,
    pub quantization: QuantizationConfig,
    pub is_asymmetric: bool,
    pub is_sparse: bool,
    pub is_token_level: bool,
}

pub static EMBEDDER_CONFIGS: [EmbedderConfig; 13];

pub fn get_config(embedder: Embedder) -> &amp;'static EmbedderConfig;
pub fn get_dimension(embedder: Embedder) -> usize;
pub fn get_distance_metric(embedder: Embedder) -> DistanceMetric;
pub fn get_category(embedder: Embedder) -> EmbedderCategory;
pub fn get_topic_weight(embedder: Embedder) -> f32;
pub fn is_asymmetric(embedder: Embedder) -> bool;
pub fn is_sparse(embedder: Embedder) -> bool;
pub fn is_semantic(embedder: Embedder) -> bool;
pub fn is_temporal(embedder: Embedder) -> bool;
    </signature>

    <signature file="crates/context-graph-core/src/index/config.rs">
// Add Jaccard to existing DistanceMetric enum:
pub enum DistanceMetric {
    Cosine,
    DotProduct,
    Euclidean,
    AsymmetricCosine,
    MaxSim,
    Jaccard,  // NEW - for sparse vectors
}

impl DistanceMetric {
    pub fn is_hnsw_compatible(&amp;self) -> bool {
        !matches!(self, Self::MaxSim | Self::Jaccard)
    }
}
    </signature>
  </signatures>

  <constraints>
    <constraint>All configurations are compile-time constants (static)</constraint>
    <constraint>get_config returns &amp;'static reference (no allocation)</constraint>
    <constraint>Dimensions MUST use constants from semantic/constants.rs</constraint>
    <constraint>E5 MUST use AsymmetricCosine distance metric</constraint>
    <constraint>E9 MUST use Cosine (NOT Hamming) because it's projected dense</constraint>
    <constraint>E6/E13 MUST use Jaccard and have is_sparse=true</constraint>
    <constraint>E12 MUST have is_token_level=true</constraint>
    <constraint>Category must come from category_for() in tests</constraint>
  </constraints>

  <verification>
    <check>E1 has dimension 1024 and Cosine metric</check>
    <check>E5 has is_asymmetric=true and AsymmetricCosine metric</check>
    <check>E6 and E13 have is_sparse=true, Jaccard metric, dimension 30522</check>
    <check>E9 has Cosine metric (NOT Hamming)</check>
    <check>E12 has is_token_level=true and MaxSim metric</check>
    <check>All 13 embedders have valid configs</check>
    <check>get_category(embedder) matches category_for(embedder) for all 13</check>
    <check>get_topic_weight returns 1.0 for semantic, 0.0 for temporal, 0.5 for relational/structural</check>
    <check>is_semantic returns true for exactly 7 embedders (E1,E5,E6,E7,E10,E12,E13)</check>
    <check>is_temporal returns true for exactly 3 embedders (E2,E3,E4)</check>
  </verification>
</definition_of_done>

<!-- =================================================================== -->
<!-- IMPLEMENTATION CODE                                                 -->
<!-- =================================================================== -->
<implementation_code>
<file path="crates/context-graph-core/src/embeddings/config.rs">
//! Static configuration registry for all 13 embedders.
//!
//! This module provides compile-time configuration for each embedder including
//! dimension, distance metric, quantization, and category classification.
//!
//! # Usage
//!
//! ```
//! use context_graph_core::embeddings::config::get_config;
//! use context_graph_core::teleological::Embedder;
//!
//! let config = get_config(Embedder::Semantic);
//! assert_eq!(config.dimension, 1024);
//! ```

use serde::{Deserialize, Serialize};

use crate::embeddings::category::{category_for, EmbedderCategory};
use crate::index::config::DistanceMetric;
use crate::teleological::Embedder;
use crate::types::fingerprint::semantic::constants::{
    E10_DIM, E11_DIM, E12_TOKEN_DIM, E1_DIM, E2_DIM, E3_DIM, E4_DIM, E5_DIM,
    E6_SPARSE_VOCAB, E7_DIM, E8_DIM, E9_DIM,
};

// E13 uses same vocab as E6
const E13_SPLADE_VOCAB: usize = E6_SPARSE_VOCAB;

/// Quantization configuration for embeddings.
///
/// Determines how embeddings are compressed for storage and index efficiency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationConfig {
    /// Product Quantization with 8-bit codes.
    ///
    /// - `num_subvectors`: Number of subvectors to split embedding into
    /// - `bits_per_code`: Bits per quantized code (typically 8)
    PQ8 {
        num_subvectors: usize,
        bits_per_code: usize,
    },

    /// 8-bit floating point (FP8).
    ///
    /// Good balance of compression and precision for embeddings that don't need PQ.
    Float8,

    /// Binary (1-bit per dimension).
    ///
    /// Used for hyperdimensional computing embeddings.
    Binary,

    /// Inverted index for sparse vectors.
    ///
    /// Used for SPLADE and sparse BoW embeddings.
    Inverted,

    /// No quantization (full 32-bit precision).
    None,
}

/// Static configuration for a single embedder.
///
/// This struct holds all metadata needed to work with an embedder's output,
/// including storage requirements, similarity computation, and classification.
#[derive(Debug, Clone, Copy)]
pub struct EmbedderConfig {
    /// The embedder this config is for.
    pub embedder: Embedder,

    /// Embedding dimension.
    /// For sparse embeddings (E6, E13), this is the vocabulary size.
    /// For token-level (E12), this is dimension per token.
    pub dimension: usize,

    /// Distance metric for similarity computation.
    pub distance_metric: DistanceMetric,

    /// Quantization method for storage.
    pub quantization: QuantizationConfig,

    /// Whether similarity is asymmetric (e.g., E5 Causal: cause->effect != effect->cause).
    pub is_asymmetric: bool,

    /// Whether this embedder produces sparse vectors (E6, E13).
    pub is_sparse: bool,

    /// Whether this embedder produces per-token embeddings (E12 ColBERT).
    pub is_token_level: bool,
}

/// Static configuration for all 13 embedders.
///
/// Index matches Embedder::index() for O(1) lookup.
pub static EMBEDDER_CONFIGS: [EmbedderConfig; 13] = [
    // E1: Semantic (1024D, Cosine, PQ8) - Category: Semantic
    EmbedderConfig {
        embedder: Embedder::Semantic,
        dimension: E1_DIM, // 1024
        distance_metric: DistanceMetric::Cosine,
        quantization: QuantizationConfig::PQ8 {
            num_subvectors: 32,
            bits_per_code: 8,
        },
        is_asymmetric: false,
        is_sparse: false,
        is_token_level: false,
    },
    // E2: Temporal Recent (512D, Cosine, Float8) - Category: Temporal
    EmbedderConfig {
        embedder: Embedder::TemporalRecent,
        dimension: E2_DIM, // 512
        distance_metric: DistanceMetric::Cosine,
        quantization: QuantizationConfig::Float8,
        is_asymmetric: false,
        is_sparse: false,
        is_token_level: false,
    },
    // E3: Temporal Periodic (512D, Cosine, Float8) - Category: Temporal
    EmbedderConfig {
        embedder: Embedder::TemporalPeriodic,
        dimension: E3_DIM, // 512
        distance_metric: DistanceMetric::Cosine,
        quantization: QuantizationConfig::Float8,
        is_asymmetric: false,
        is_sparse: false,
        is_token_level: false,
    },
    // E4: Temporal Positional (512D, Cosine, Float8) - Category: Temporal
    EmbedderConfig {
        embedder: Embedder::TemporalPositional,
        dimension: E4_DIM, // 512
        distance_metric: DistanceMetric::Cosine,
        quantization: QuantizationConfig::Float8,
        is_asymmetric: false,
        is_sparse: false,
        is_token_level: false,
    },
    // E5: Causal (768D, AsymmetricCosine, PQ8, asymmetric) - Category: Semantic
    EmbedderConfig {
        embedder: Embedder::Causal,
        dimension: E5_DIM, // 768
        distance_metric: DistanceMetric::AsymmetricCosine,
        quantization: QuantizationConfig::PQ8 {
            num_subvectors: 24,
            bits_per_code: 8,
        },
        is_asymmetric: true,
        is_sparse: false,
        is_token_level: false,
    },
    // E6: Sparse (30522 vocab, Jaccard, Inverted) - Category: Semantic
    EmbedderConfig {
        embedder: Embedder::Sparse,
        dimension: E6_SPARSE_VOCAB, // 30522
        distance_metric: DistanceMetric::Jaccard,
        quantization: QuantizationConfig::Inverted,
        is_asymmetric: false,
        is_sparse: true,
        is_token_level: false,
    },
    // E7: Code (1536D, Cosine, PQ8) - Category: Semantic
    EmbedderConfig {
        embedder: Embedder::Code,
        dimension: E7_DIM, // 1536
        distance_metric: DistanceMetric::Cosine,
        quantization: QuantizationConfig::PQ8 {
            num_subvectors: 48,
            bits_per_code: 8,
        },
        is_asymmetric: false,
        is_sparse: false,
        is_token_level: false,
    },
    // E8: Emotional/Graph (384D, Cosine, Float8) - Category: Relational
    EmbedderConfig {
        embedder: Embedder::Emotional,
        dimension: E8_DIM, // 384
        distance_metric: DistanceMetric::Cosine,
        quantization: QuantizationConfig::Float8,
        is_asymmetric: false,
        is_sparse: false,
        is_token_level: false,
    },
    // E9: HDC (1024D projected, Cosine, PQ8) - Category: Structural
    // NOTE: Stored as dense Vec&lt;f32&gt; after 10K-bit projection, so Cosine not Hamming
    EmbedderConfig {
        embedder: Embedder::Hdc,
        dimension: E9_DIM, // 1024
        distance_metric: DistanceMetric::Cosine,
        quantization: QuantizationConfig::PQ8 {
            num_subvectors: 32,
            bits_per_code: 8,
        },
        is_asymmetric: false,
        is_sparse: false,
        is_token_level: false,
    },
    // E10: Multimodal (768D, Cosine, PQ8) - Category: Semantic
    EmbedderConfig {
        embedder: Embedder::Multimodal,
        dimension: E10_DIM, // 768
        distance_metric: DistanceMetric::Cosine,
        quantization: QuantizationConfig::PQ8 {
            num_subvectors: 24,
            bits_per_code: 8,
        },
        is_asymmetric: false,
        is_sparse: false,
        is_token_level: false,
    },
    // E11: Entity (384D, Cosine, Float8) - Category: Relational
    EmbedderConfig {
        embedder: Embedder::Entity,
        dimension: E11_DIM, // 384
        distance_metric: DistanceMetric::Cosine,
        quantization: QuantizationConfig::Float8,
        is_asymmetric: false,
        is_sparse: false,
        is_token_level: false,
    },
    // E12: Late Interaction (128D per token, MaxSim, Float8) - Category: Semantic
    EmbedderConfig {
        embedder: Embedder::LateInteraction,
        dimension: E12_TOKEN_DIM, // 128 per token
        distance_metric: DistanceMetric::MaxSim,
        quantization: QuantizationConfig::Float8,
        is_asymmetric: false,
        is_sparse: false,
        is_token_level: true,
    },
    // E13: SPLADE (30522 vocab, Jaccard, Inverted) - Category: Semantic
    EmbedderConfig {
        embedder: Embedder::KeywordSplade,
        dimension: E13_SPLADE_VOCAB, // 30522
        distance_metric: DistanceMetric::Jaccard,
        quantization: QuantizationConfig::Inverted,
        is_asymmetric: false,
        is_sparse: true,
        is_token_level: false,
    },
];

// =============================================================================
// Getter Functions
// =============================================================================

/// Get configuration for a specific embedder.
///
/// Returns a static reference - no allocation.
/// O(1) lookup via embedder index.
#[inline]
pub fn get_config(embedder: Embedder) -> &amp;'static EmbedderConfig {
    &amp;EMBEDDER_CONFIGS[embedder.index()]
}

/// Get the expected dimension for an embedder.
///
/// For sparse embeddings (E6, E13), returns vocabulary size.
/// For token-level (E12), returns dimension per token.
#[inline]
pub fn get_dimension(embedder: Embedder) -> usize {
    get_config(embedder).dimension
}

/// Get the distance metric for an embedder.
#[inline]
pub fn get_distance_metric(embedder: Embedder) -> DistanceMetric {
    get_config(embedder).distance_metric
}

/// Get quantization configuration for an embedder.
#[inline]
pub fn get_quantization(embedder: Embedder) -> QuantizationConfig {
    get_config(embedder).quantization
}

/// Check if embedder uses asymmetric similarity.
///
/// Only E5 (Causal) is asymmetric - cause->effect != effect->cause.
#[inline]
pub fn is_asymmetric(embedder: Embedder) -> bool {
    get_config(embedder).is_asymmetric
}

/// Check if embedder produces sparse vectors.
///
/// E6 (Sparse) and E13 (KeywordSplade) produce sparse vectors.
#[inline]
pub fn is_sparse(embedder: Embedder) -> bool {
    get_config(embedder).is_sparse
}

/// Check if embedder produces per-token embeddings.
///
/// Only E12 (LateInteraction/ColBERT) produces per-token embeddings.
#[inline]
pub fn is_token_level(embedder: Embedder) -> bool {
    get_config(embedder).is_token_level
}

/// Get the category for an embedder.
///
/// This delegates to category_for() from the category module.
#[inline]
pub fn get_category(embedder: Embedder) -> EmbedderCategory {
    category_for(embedder)
}

/// Get the topic weight for an embedder.
///
/// Derived from category: Semantic=1.0, Temporal=0.0, Relational=0.5, Structural=0.5
#[inline]
pub fn get_topic_weight(embedder: Embedder) -> f32 {
    get_category(embedder).topic_weight()
}

/// Check if embedder is in the Semantic category.
///
/// Semantic embedders (E1, E5, E6, E7, E10, E12, E13) have topic_weight 1.0.
#[inline]
pub fn is_semantic(embedder: Embedder) -> bool {
    get_category(embedder).is_semantic()
}

/// Check if embedder is in the Temporal category.
///
/// Temporal embedders (E2, E3, E4) have topic_weight 0.0 (excluded from topic detection).
#[inline]
pub fn is_temporal(embedder: Embedder) -> bool {
    get_category(embedder).is_temporal()
}

/// Check if embedder is in the Relational category.
///
/// Relational embedders (E8, E11) have topic_weight 0.5.
#[inline]
pub fn is_relational(embedder: Embedder) -> bool {
    get_category(embedder).is_relational()
}

/// Check if embedder is in the Structural category.
///
/// Structural embedder (E9) has topic_weight 0.5.
#[inline]
pub fn is_structural(embedder: Embedder) -> bool {
    get_category(embedder).is_structural()
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_embedders_configured() {
        for embedder in Embedder::all() {
            let config = get_config(embedder);
            assert_eq!(config.embedder, embedder);
            assert!(config.dimension > 0);
        }
        assert_eq!(Embedder::all().count(), 13);
        println!("[PASS] All 13 embedders have valid configurations");
    }

    #[test]
    fn test_config_array_index_matches_embedder_index() {
        for (i, config) in EMBEDDER_CONFIGS.iter().enumerate() {
            assert_eq!(
                config.embedder.index(),
                i,
                "Config at index {} has embedder {:?} with index {}",
                i,
                config.embedder,
                config.embedder.index()
            );
        }
        println!("[PASS] Config array indices match embedder indices");
    }

    #[test]
    fn test_e1_semantic_config() {
        let config = get_config(Embedder::Semantic);
        assert_eq!(config.dimension, 1024);
        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
        assert!(!config.is_asymmetric);
        assert!(!config.is_sparse);
        assert!(!config.is_token_level);
        println!("[PASS] E1 Semantic config: 1024D, Cosine, dense");
    }

    #[test]
    fn test_e5_causal_asymmetric() {
        let config = get_config(Embedder::Causal);
        assert_eq!(config.dimension, 768);
        assert_eq!(config.distance_metric, DistanceMetric::AsymmetricCosine);
        assert!(config.is_asymmetric);
        assert!(is_asymmetric(Embedder::Causal));
        assert!(!is_asymmetric(Embedder::Semantic));
        println!("[PASS] E5 Causal: asymmetric, AsymmetricCosine metric");
    }

    #[test]
    fn test_sparse_embedders() {
        // E6 and E13 are sparse
        assert!(is_sparse(Embedder::Sparse));
        assert!(is_sparse(Embedder::KeywordSplade));

        // All others are dense
        assert!(!is_sparse(Embedder::Semantic));
        assert!(!is_sparse(Embedder::Causal));
        assert!(!is_sparse(Embedder::Hdc));
        assert!(!is_sparse(Embedder::LateInteraction));

        // Check dimensions
        assert_eq!(get_dimension(Embedder::Sparse), 30_522);
        assert_eq!(get_dimension(Embedder::KeywordSplade), 30_522);

        // Check Jaccard metric
        assert_eq!(get_distance_metric(Embedder::Sparse), DistanceMetric::Jaccard);
        assert_eq!(get_distance_metric(Embedder::KeywordSplade), DistanceMetric::Jaccard);

        println!("[PASS] E6/E13 sparse: 30522 vocab, Jaccard metric");
    }

    #[test]
    fn test_e9_hdc_is_cosine_not_hamming() {
        // CRITICAL: E9 is stored as projected dense, not binary
        let config = get_config(Embedder::Hdc);
        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
        assert!(!config.is_sparse);
        assert_eq!(config.dimension, 1024);
        println!("[PASS] E9 HDC uses Cosine (projected dense), NOT Hamming");
    }

    #[test]
    fn test_e12_late_interaction() {
        let config = get_config(Embedder::LateInteraction);
        assert_eq!(config.dimension, 128); // Per token
        assert_eq!(config.distance_metric, DistanceMetric::MaxSim);
        assert!(config.is_token_level);
        assert!(is_token_level(Embedder::LateInteraction));
        assert!(!is_token_level(Embedder::Semantic));
        println!("[PASS] E12 LateInteraction: 128D per token, MaxSim");
    }

    #[test]
    fn test_category_assignments() {
        // Semantic embedders (7)
        assert_eq!(get_category(Embedder::Semantic), EmbedderCategory::Semantic);
        assert_eq!(get_category(Embedder::Causal), EmbedderCategory::Semantic);
        assert_eq!(get_category(Embedder::Sparse), EmbedderCategory::Semantic);
        assert_eq!(get_category(Embedder::Code), EmbedderCategory::Semantic);
        assert_eq!(get_category(Embedder::Multimodal), EmbedderCategory::Semantic);
        assert_eq!(get_category(Embedder::LateInteraction), EmbedderCategory::Semantic);
        assert_eq!(get_category(Embedder::KeywordSplade), EmbedderCategory::Semantic);

        // Temporal embedders (3)
        assert_eq!(get_category(Embedder::TemporalRecent), EmbedderCategory::Temporal);
        assert_eq!(get_category(Embedder::TemporalPeriodic), EmbedderCategory::Temporal);
        assert_eq!(get_category(Embedder::TemporalPositional), EmbedderCategory::Temporal);

        // Relational embedders (2)
        assert_eq!(get_category(Embedder::Emotional), EmbedderCategory::Relational);
        assert_eq!(get_category(Embedder::Entity), EmbedderCategory::Relational);

        // Structural embedders (1)
        assert_eq!(get_category(Embedder::Hdc), EmbedderCategory::Structural);

        println!("[PASS] All 13 embedders have correct category assignments");
    }

    #[test]
    fn test_topic_weights() {
        // Semantic = 1.0
        assert_eq!(get_topic_weight(Embedder::Semantic), 1.0);
        assert_eq!(get_topic_weight(Embedder::Code), 1.0);
        assert_eq!(get_topic_weight(Embedder::KeywordSplade), 1.0);

        // Temporal = 0.0
        assert_eq!(get_topic_weight(Embedder::TemporalRecent), 0.0);
        assert_eq!(get_topic_weight(Embedder::TemporalPeriodic), 0.0);
        assert_eq!(get_topic_weight(Embedder::TemporalPositional), 0.0);

        // Relational = 0.5
        assert_eq!(get_topic_weight(Embedder::Emotional), 0.5);
        assert_eq!(get_topic_weight(Embedder::Entity), 0.5);

        // Structural = 0.5
        assert_eq!(get_topic_weight(Embedder::Hdc), 0.5);

        println!("[PASS] Topic weights match category definitions");
    }

    #[test]
    fn test_is_semantic_count() {
        let semantic_count = Embedder::all().filter(|e| is_semantic(*e)).count();
        assert_eq!(semantic_count, 7, "Should have exactly 7 semantic embedders");

        assert!(is_semantic(Embedder::Semantic));
        assert!(is_semantic(Embedder::Causal));
        assert!(is_semantic(Embedder::Sparse));
        assert!(is_semantic(Embedder::Code));
        assert!(is_semantic(Embedder::Multimodal));
        assert!(is_semantic(Embedder::LateInteraction));
        assert!(is_semantic(Embedder::KeywordSplade));

        assert!(!is_semantic(Embedder::TemporalRecent));
        assert!(!is_semantic(Embedder::Emotional));
        assert!(!is_semantic(Embedder::Hdc));

        println!("[PASS] is_semantic() returns true for exactly 7 embedders");
    }

    #[test]
    fn test_is_temporal_count() {
        let temporal_count = Embedder::all().filter(|e| is_temporal(*e)).count();
        assert_eq!(temporal_count, 3, "Should have exactly 3 temporal embedders");

        assert!(is_temporal(Embedder::TemporalRecent));
        assert!(is_temporal(Embedder::TemporalPeriodic));
        assert!(is_temporal(Embedder::TemporalPositional));

        assert!(!is_temporal(Embedder::Semantic));
        assert!(!is_temporal(Embedder::Emotional));
        assert!(!is_temporal(Embedder::Hdc));

        println!("[PASS] is_temporal() returns true for exactly 3 embedders");
    }

    #[test]
    fn test_dimensions_match_constants() {
        use crate::types::fingerprint::semantic::constants::*;

        assert_eq!(get_dimension(Embedder::Semantic), E1_DIM);
        assert_eq!(get_dimension(Embedder::TemporalRecent), E2_DIM);
        assert_eq!(get_dimension(Embedder::TemporalPeriodic), E3_DIM);
        assert_eq!(get_dimension(Embedder::TemporalPositional), E4_DIM);
        assert_eq!(get_dimension(Embedder::Causal), E5_DIM);
        assert_eq!(get_dimension(Embedder::Sparse), E6_SPARSE_VOCAB);
        assert_eq!(get_dimension(Embedder::Code), E7_DIM);
        assert_eq!(get_dimension(Embedder::Emotional), E8_DIM);
        assert_eq!(get_dimension(Embedder::Hdc), E9_DIM);
        assert_eq!(get_dimension(Embedder::Multimodal), E10_DIM);
        assert_eq!(get_dimension(Embedder::Entity), E11_DIM);
        assert_eq!(get_dimension(Embedder::LateInteraction), E12_TOKEN_DIM);
        assert_eq!(get_dimension(Embedder::KeywordSplade), E6_SPARSE_VOCAB);

        println!("[PASS] All dimensions match semantic/constants.rs");
    }

    #[test]
    fn test_quantization_configs() {
        // PQ8 for large dense embeddings
        assert!(matches!(
            get_quantization(Embedder::Semantic),
            QuantizationConfig::PQ8 { num_subvectors: 32, bits_per_code: 8 }
        ));
        assert!(matches!(
            get_quantization(Embedder::Code),
            QuantizationConfig::PQ8 { num_subvectors: 48, bits_per_code: 8 }
        ));

        // Float8 for smaller embeddings
        assert_eq!(get_quantization(Embedder::TemporalRecent), QuantizationConfig::Float8);
        assert_eq!(get_quantization(Embedder::Emotional), QuantizationConfig::Float8);

        // Inverted for sparse
        assert_eq!(get_quantization(Embedder::Sparse), QuantizationConfig::Inverted);
        assert_eq!(get_quantization(Embedder::KeywordSplade), QuantizationConfig::Inverted);

        println!("[PASS] Quantization configs are appropriate for each embedder");
    }
}
</file>
</implementation_code>

<!-- =================================================================== -->
<!-- DISTANCE METRIC EXTENSION                                           -->
<!-- =================================================================== -->
<distance_metric_extension file="crates/context-graph-core/src/index/config.rs">
Add Jaccard variant to existing DistanceMetric enum:

```rust
/// Distance metric for vector similarity computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine distance: 1 - cos(a, b). Range [0, 2].
    Cosine,
    /// Dot product (inner product).
    DotProduct,
    /// L2 Euclidean distance. Range [0, inf).
    Euclidean,
    /// Asymmetric cosine for E5 causal.
    AsymmetricCosine,
    /// MaxSim for ColBERT (NOT HNSW-compatible).
    MaxSim,
    /// Jaccard index for sparse vectors (NOT HNSW-compatible).
    Jaccard,
}

impl DistanceMetric {
    /// Check if this metric is compatible with HNSW indexing.
    #[inline]
    pub fn is_hnsw_compatible(&amp;self) -> bool {
        !matches!(self, Self::MaxSim | Self::Jaccard)
    }

    /// Check if this metric requires sparse vector handling.
    #[inline]
    pub fn is_sparse_metric(&amp;self) -> bool {
        matches!(self, Self::Jaccard)
    }
}
```
</distance_metric_extension>

<!-- =================================================================== -->
<!-- MODULE EXPORTS UPDATE                                               -->
<!-- =================================================================== -->
<module_exports file="crates/context-graph-core/src/embeddings/mod.rs">
After TASK-P2-003b and this task, mod.rs should contain:

```rust
//! Embedding types for the 13-model teleological array.
//!
//! This module provides:
//! - `EmbedderCategory`: Category classification (Semantic, Temporal, Relational, Structural)
//! - `EmbedderConfig`: Static configuration for each embedder
//! - `QuantizationConfig`: Quantization methods
//! - `TokenPruningEmbedding` (E12): Token-level embedding
//! - `DenseVector`: Generic dense vector for similarity
//! - `BinaryVector`: Bit-packed vector for Hamming distance
//!
//! Note: `SparseVector` for SPLADE is in `types::fingerprint::sparse`.
//! Note: `Embedder` enum is in `teleological::embedder`.
//! Note: `DistanceMetric` is in `index::config`.

pub mod category;
pub mod config;
pub mod token_pruning;
pub mod vector;

pub use category::{category_for, max_weighted_agreement, topic_threshold, EmbedderCategory};
pub use config::{
    get_category, get_config, get_dimension, get_distance_metric, get_quantization,
    get_topic_weight, is_asymmetric, is_relational, is_semantic, is_sparse,
    is_structural, is_temporal, is_token_level, EmbedderConfig, QuantizationConfig,
    EMBEDDER_CONFIGS,
};
pub use token_pruning::TokenPruningEmbedding;
pub use vector::{BinaryVector, DenseVector, VectorError};
```
</module_exports>

<!-- =================================================================== -->
<!-- TEST COMMANDS                                                       -->
<!-- =================================================================== -->
<test_commands>
  <command description="Run config tests">
    cargo test --package context-graph-core config -- --nocapture
  </command>
  <command description="Check compilation">
    cargo check --package context-graph-core
  </command>
  <command description="Run clippy">
    cargo clippy --package context-graph-core -- -D warnings
  </command>
  <command description="Run all embeddings tests">
    cargo test --package context-graph-core embeddings -- --nocapture
  </command>
</test_commands>

<!-- =================================================================== -->
<!-- FULL STATE VERIFICATION PROTOCOL                                    -->
<!-- =================================================================== -->
<full_state_verification>
  <source_of_truth>
    <primary>The compiled Rust code in context-graph-core::embeddings::config</primary>
    <secondary>Unit test results demonstrating correct behavior</secondary>
    <tertiary>Category assignments matching TASK-P2-003b category_for()</tertiary>
  </source_of_truth>

  <verification_steps>
    <step id="1" name="Compilation Check">
      <command>cargo check --package context-graph-core</command>
      <expected>No errors, no warnings</expected>
      <evidence>Command exit code 0</evidence>
    </step>

    <step id="2" name="Config Tests Pass">
      <command>cargo test --package context-graph-core config -- --nocapture</command>
      <expected>All tests pass (15+ tests)</expected>
      <evidence>Test output showing "[PASS]" for each test</evidence>
    </step>

    <step id="3" name="Category Consistency">
      <command>cargo test --package context-graph-core category -- --nocapture</command>
      <expected>Category tests pass, proving config uses correct categories</expected>
    </step>

    <step id="4" name="Clippy Clean">
      <command>cargo clippy --package context-graph-core -- -D warnings</command>
      <expected>No clippy warnings</expected>
    </step>
  </verification_steps>

  <boundary_edge_cases>
    <case id="EC-001" name="Config Array Index Alignment">
      <input>EMBEDDER_CONFIGS[i].embedder.index() for all i</input>
      <expected>Must equal i for all 0..13</expected>
      <test>test_config_array_index_matches_embedder_index</test>
    </case>

    <case id="EC-002" name="E9 Uses Cosine Not Hamming">
      <input>get_distance_metric(Embedder::Hdc)</input>
      <expected>DistanceMetric::Cosine (NOT Hamming)</expected>
      <test>test_e9_hdc_is_cosine_not_hamming</test>
    </case>

    <case id="EC-003" name="Sparse Vocab Size Exact">
      <input>get_dimension(Embedder::Sparse), get_dimension(Embedder::KeywordSplade)</input>
      <expected>30522 (NOT 30000)</expected>
      <test>test_sparse_embedders</test>
    </case>

    <case id="EC-004" name="Category Consistency with category_for">
      <input>get_category(e) for all embedders</input>
      <expected>Must equal category_for(e) for all</expected>
      <test>test_category_assignments</test>
    </case>
  </boundary_edge_cases>

  <manual_verification_checklist>
    <item>[ ] config.rs created at crates/context-graph-core/src/embeddings/config.rs</item>
    <item>[ ] EmbedderConfig struct has: embedder, dimension, distance_metric, quantization, is_asymmetric, is_sparse, is_token_level</item>
    <item>[ ] EMBEDDER_CONFIGS has exactly 13 entries</item>
    <item>[ ] E5 Causal uses AsymmetricCosine and is_asymmetric=true</item>
    <item>[ ] E9 HDC uses Cosine (NOT Hamming)</item>
    <item>[ ] E6/E13 have dimension 30522, Jaccard metric, is_sparse=true</item>
    <item>[ ] E12 has is_token_level=true and MaxSim metric</item>
    <item>[ ] get_category() returns same as category_for() for all embedders</item>
    <item>[ ] Jaccard added to DistanceMetric enum in index/config.rs</item>
    <item>[ ] All unit tests pass</item>
    <item>[ ] No clippy warnings</item>
  </manual_verification_checklist>

  <evidence_log_template>
    ```
    === TASK-P2-003 Verification Evidence ===
    Date: [FILL]

    1. Compilation:
       $ cargo check --package context-graph-core
       [PASTE OUTPUT]

    2. Config Tests:
       $ cargo test --package context-graph-core config -- --nocapture
       [PASTE OUTPUT - should show 15+ tests passing]

    3. Category Consistency:
       $ cargo test --package context-graph-core test_category -- --nocapture
       [PASTE OUTPUT]

    4. Clippy:
       $ cargo clippy --package context-graph-core -- -D warnings
       [PASTE OUTPUT]

    5. Files Created/Modified:
       $ ls -la crates/context-graph-core/src/embeddings/config.rs
       [PASTE OUTPUT]

       $ grep -n "Jaccard" crates/context-graph-core/src/index/config.rs
       [PASTE OUTPUT - should show Jaccard variant added]
    ```
  </evidence_log_template>
</full_state_verification>
</task_spec>
```

## Execution Checklist

### Phase 0: Prerequisites
- [ ] VERIFY TASK-P2-003b is complete (EmbedderCategory enum exists and compiles)
- [ ] Run `cargo test --package context-graph-core category` to confirm category module works

### Phase 1: Extend DistanceMetric
- [ ] Add `Jaccard` variant to `crates/context-graph-core/src/index/config.rs`
- [ ] Update `is_hnsw_compatible()` to return false for Jaccard
- [ ] Add `is_sparse_metric()` helper method
- [ ] Run `cargo check` to verify compilation

### Phase 2: Create EmbedderConfig
- [ ] Create `crates/context-graph-core/src/embeddings/config.rs`
- [ ] Implement `QuantizationConfig` enum
- [ ] Implement `EmbedderConfig` struct
- [ ] Create `EMBEDDER_CONFIGS` static array with all 13 configs
- [ ] Use correct Embedder variant names (Semantic, not E1Semantic)
- [ ] Use dimension constants from semantic/constants.rs
- [ ] E9 uses Cosine (NOT Hamming)
- [ ] E5 uses AsymmetricCosine and is_asymmetric=true
- [ ] E6/E13 have dimension 30522, Jaccard, is_sparse=true
- [ ] E12 has is_token_level=true

### Phase 3: Implement Getter Functions
- [ ] Implement get_config(), get_dimension(), get_distance_metric()
- [ ] Implement get_quantization(), is_asymmetric(), is_sparse(), is_token_level()
- [ ] Implement get_category(), get_topic_weight() (delegate to category_for)
- [ ] Implement is_semantic(), is_temporal(), is_relational(), is_structural()

### Phase 4: Update Module Exports
- [ ] Update `crates/context-graph-core/src/embeddings/mod.rs`
- [ ] Add `pub mod config;` and exports

### Phase 5: Write and Run Tests
- [ ] Write comprehensive unit tests (15+)
- [ ] Run `cargo test --package context-graph-core config -- --nocapture`
- [ ] Run `cargo clippy --package context-graph-core -- -D warnings`
- [ ] Verify all assertions pass
- [ ] Complete manual verification checklist
- [ ] Document evidence in evidence log

### Phase 6: Mark Complete
- [ ] Update this file status to COMPLETE
- [ ] Proceed to TASK-P2-004 (Similarity Computation)
