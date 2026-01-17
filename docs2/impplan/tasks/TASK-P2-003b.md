# TASK-P2-003b: EmbedderCategory Enum

```xml
<task_spec id="TASK-P2-003b" version="3.0">
<metadata>
  <title>EmbedderCategory Enum Implementation</title>
  <status>ready</status>
  <layer>foundation</layer>
  <sequence>16</sequence>
  <phase>2</phase>
  <implements>
    <requirement_ref>REQ-P2-02</requirement_ref>
    <requirement_ref>REQ-P2-05</requirement_ref>
  </implements>
  <depends_on>
    <dependency status="COMPLETE">TASK-P2-001 (Embedder enum)</dependency>
    <dependency status="COMPLETE">TASK-P2-002 (Vector types)</dependency>
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
  <audit_date>2026-01-16</audit_date>
</metadata>

<!-- =================================================================== -->
<!-- CRITICAL: CODEBASE AUDIT FINDINGS - READ FIRST                      -->
<!-- =================================================================== -->
<audit_findings>
  <finding severity="CRITICAL" id="AF-001">
    <issue>EmbedderCategory DOES NOT EXIST in the codebase</issue>
    <evidence>
      Grep search for "EmbedderCategory" found 0 matches in .rs files.
      Only appears in task documents and technical specs.
    </evidence>
    <resolution>Must create new file. Proceed with implementation.</resolution>
  </finding>

  <finding severity="HIGH" id="AF-002">
    <issue>File path in original spec is WRONG</issue>
    <details>
      Original spec says: crates/context-graph-core/src/embedding/category.rs
      ACTUAL directory structure uses: "embeddings" (with 's'), NOT "embedding"
    </details>
    <resolution>
      Create file at: crates/context-graph-core/src/embeddings/category.rs
      Update crates/context-graph-core/src/embeddings/mod.rs to export
    </resolution>
  </finding>

  <finding severity="HIGH" id="AF-003">
    <issue>Similar patterns exist that must be integrated with</issue>
    <details>
      - EmbedderGroup enum exists in teleological/embedder.rs with Temporal, Lexical, etc.
      - EmbedderMask provides bitmask operations for embedder selection
      - These provide grouping but NOT topic weights for similarity calculation
    </details>
    <resolution>
      EmbedderCategory is DISTINCT from EmbedderGroup:
      - EmbedderGroup: Which embedders to SELECT for a query type
      - EmbedderCategory: How to WEIGHT embedders in topic detection
      Both can coexist. Do NOT merge them.
    </resolution>
  </finding>

  <finding severity="MEDIUM" id="AF-004">
    <issue>Embedder variant names differ from task pseudo_code</issue>
    <details>
      Task says: Embedder::E1Semantic, Embedder::E2TempRecent
      ACTUAL codebase has: Embedder::Semantic, Embedder::TemporalRecent
      The E-prefix is accessed via short_name() method, not variant name.
    </details>
    <resolution>Use ACTUAL variant names in implementation and tests.</resolution>
  </finding>
</audit_findings>

<!-- =================================================================== -->
<!-- CONTEXT - WHAT THIS TASK IS FOR                                     -->
<!-- =================================================================== -->
<context>
  <summary>
    Implements the EmbedderCategory enum that classifies each of the 13 embedders into
    one of four semantic categories. Each category has an associated topic_weight that
    determines how much the embedder contributes to weighted similarity calculations.
  </summary>

  <constitution_reference>
    From CLAUDE.md (constitution.yaml) embedder_categories section:

    SEMANTIC (E1, E5, E6, E7, E10, E12, E13):
      - topic_weight: 1.0
      - role: "Primary topic triggers - capture meaning, concepts, intent, code"
      - divergence_detection: true

    TEMPORAL (E2, E3, E4):
      - topic_weight: 0.0
      - role: "Metadata only - NEVER for topic detection"
      - divergence_detection: false
      - rationale: "Temporal proximity != semantic relationship"

    RELATIONAL (E8, E11):
      - topic_weight: 0.5
      - role: "Supporting - can reinforce but not define topics alone"
      - divergence_detection: false

    STRUCTURAL (E9):
      - topic_weight: 0.5
      - role: "Supporting - structural/format patterns"
      - divergence_detection: false
  </constitution_reference>

  <architecture_rules_referenced>
    ARCH-09: Topic threshold is weighted_agreement >= 2.5 (not raw space count)
    ARCH-10: Divergence detection uses SEMANTIC embedders only
    AP-60: Temporal embedders (E2-E4) MUST NOT count toward topic detection
    AP-61: Topic threshold MUST be weighted_agreement >= 2.5, not raw count
  </architecture_rules_referenced>

  <what_exists>
    <item name="Embedder enum" path="crates/context-graph-core/src/teleological/embedder.rs" status="COMPLETE">
      13 variants: Semantic, TemporalRecent, TemporalPeriodic, TemporalPositional,
      Causal, Sparse, Code, Emotional, Hdc, Multimodal, Entity, LateInteraction, KeywordSplade
    </item>
    <item name="EmbedderGroup enum" path="crates/context-graph-core/src/teleological/embedder.rs" status="COMPLETE">
      Predefined groups: Temporal, Relational, Lexical, Dense, Factual, Implementation, All
      NOTE: This is for query-time embedder SELECTION, not topic weight calculation
    </item>
    <item name="EmbedderMask" path="crates/context-graph-core/src/teleological/embedder.rs" status="COMPLETE">
      Bitmask for selecting subsets of embedders
    </item>
  </what_exists>

  <what_needs_creation>
    <item name="EmbedderCategory" priority="HIGH">
      New enum with 4 variants (Semantic, Temporal, Relational, Structural)
      topic_weight() method returning category-specific weight
      Helper methods: is_semantic(), is_temporal(), etc.
      Method to get category for a given Embedder
    </item>
  </what_needs_creation>
</context>

<!-- =================================================================== -->
<!-- ACTUAL FILE LOCATIONS (CORRECTED)                                   -->
<!-- =================================================================== -->
<actual_file_locations>
  <file purpose="NEW - EmbedderCategory enum"
        path="crates/context-graph-core/src/embeddings/category.rs"
        action="CREATE">
    Create this file with EmbedderCategory enum and impl block.
  </file>

  <file purpose="embeddings module"
        path="crates/context-graph-core/src/embeddings/mod.rs"
        action="MODIFY">
    Add: pub mod category; and re-export EmbedderCategory.
    Current contents (VERIFIED 2026-01-16):
    ```rust
    pub mod token_pruning;
    pub mod vector;
    pub use token_pruning::TokenPruningEmbedding;
    pub use vector::{BinaryVector, DenseVector, VectorError};
    ```
  </file>

  <file purpose="lib.rs exports"
        path="crates/context-graph-core/src/lib.rs"
        action="MODIFY">
    Add EmbedderCategory to the embeddings re-exports.
  </file>

  <file purpose="Reference - Embedder enum"
        path="crates/context-graph-core/src/teleological/embedder.rs"
        action="READ_ONLY">
    Contains Embedder enum with ACTUAL variant names to use.
  </file>
</actual_file_locations>

<!-- =================================================================== -->
<!-- SCOPE - EXACTLY WHAT TO DO                                          -->
<!-- =================================================================== -->
<scope>
  <in_scope>
    <item>Create EmbedderCategory enum with 4 variants (Semantic, Temporal, Relational, Structural)</item>
    <item>Implement topic_weight() const method returning f32</item>
    <item>Implement is_semantic(), is_temporal(), is_relational(), is_structural() helpers</item>
    <item>Implement category_for(embedder: Embedder) -> EmbedderCategory function</item>
    <item>Implement all() method returning all 4 variants</item>
    <item>Derive: Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize</item>
    <item>Comprehensive unit tests</item>
    <item>Update embeddings/mod.rs to export</item>
  </in_scope>

  <out_of_scope>
    <item>EmbedderConfig struct (TASK-P2-003)</item>
    <item>Usage in similarity calculations (Phase 3)</item>
    <item>Modifying existing EmbedderGroup enum</item>
    <item>Divergence detection logic (uses these weights but implemented elsewhere)</item>
  </out_of_scope>
</scope>

<!-- =================================================================== -->
<!-- IMPLEMENTATION REQUIREMENTS                                         -->
<!-- =================================================================== -->
<implementation_requirements>
  <requirement id="IR-001" priority="CRITICAL">
    <rule>topic_weight values MUST match constitution exactly</rule>
    <values>
      Semantic: 1.0
      Temporal: 0.0
      Relational: 0.5
      Structural: 0.5
    </values>
    <rationale>AP-61 requires weighted_agreement >= 2.5 calculation</rationale>
  </requirement>

  <requirement id="IR-002" priority="CRITICAL">
    <rule>category_for() MUST use correct Embedder variant names</rule>
    <mapping>
      Embedder::Semantic => EmbedderCategory::Semantic
      Embedder::TemporalRecent => EmbedderCategory::Temporal
      Embedder::TemporalPeriodic => EmbedderCategory::Temporal
      Embedder::TemporalPositional => EmbedderCategory::Temporal
      Embedder::Causal => EmbedderCategory::Semantic
      Embedder::Sparse => EmbedderCategory::Semantic
      Embedder::Code => EmbedderCategory::Semantic
      Embedder::Emotional => EmbedderCategory::Relational
      Embedder::Hdc => EmbedderCategory::Structural
      Embedder::Multimodal => EmbedderCategory::Semantic
      Embedder::Entity => EmbedderCategory::Relational
      Embedder::LateInteraction => EmbedderCategory::Semantic
      Embedder::KeywordSplade => EmbedderCategory::Semantic
    </mapping>
  </requirement>

  <requirement id="IR-003" priority="HIGH">
    <rule>Use const fn where possible for compile-time evaluation</rule>
  </requirement>

  <requirement id="IR-004" priority="HIGH">
    <rule>Import Embedder from crate::teleological::Embedder</rule>
    <note>NOT from a hypothetical embedding module</note>
  </requirement>
</implementation_requirements>

<!-- =================================================================== -->
<!-- DEFINITION OF DONE                                                  -->
<!-- =================================================================== -->
<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/embeddings/category.rs">
use serde::{Deserialize, Serialize};
use crate::teleological::Embedder;

/// Category classification for embedders, determining their role in topic detection.
///
/// From constitution.yaml embedder_categories:
/// - Semantic: topic_weight 1.0, primary topic triggers
/// - Temporal: topic_weight 0.0, metadata only (NEVER for topic detection)
/// - Relational: topic_weight 0.5, supporting role
/// - Structural: topic_weight 0.5, supporting role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmbedderCategory {
    /// Semantic embedders: E1, E5, E6, E7, E10, E12, E13
    /// Full contribution to topic relevance (weight 1.0)
    Semantic,
    /// Temporal embedders: E2, E3, E4
    /// Excluded from topic detection (weight 0.0) - used for recency metadata only
    Temporal,
    /// Relational embedders: E8, E11
    /// Partial contribution (weight 0.5)
    Relational,
    /// Structural embedders: E9
    /// Partial contribution (weight 0.5)
    Structural,
}

impl EmbedderCategory {
    pub const fn topic_weight(&amp;self) -> f32;
    pub const fn is_semantic(&amp;self) -> bool;
    pub const fn is_temporal(&amp;self) -> bool;
    pub const fn is_relational(&amp;self) -> bool;
    pub const fn is_structural(&amp;self) -> bool;
    pub const fn all() -> [EmbedderCategory; 4];
    pub const fn count_by_category() -> (usize, usize, usize, usize);
}

/// Get the category for a specific embedder.
pub fn category_for(embedder: Embedder) -> EmbedderCategory;

/// Calculate max possible weighted agreement (for confidence normalization).
pub const fn max_weighted_agreement() -> f32;
    </signature>
  </signatures>

  <constraints>
    <constraint>topic_weight values are compile-time constants</constraint>
    <constraint>All methods that can be const MUST be const</constraint>
    <constraint>category_for() covers all 13 Embedder variants exhaustively</constraint>
  </constraints>

  <verification>
    <check>EmbedderCategory::Semantic.topic_weight() == 1.0</check>
    <check>EmbedderCategory::Temporal.topic_weight() == 0.0</check>
    <check>EmbedderCategory::Relational.topic_weight() == 0.5</check>
    <check>EmbedderCategory::Structural.topic_weight() == 0.5</check>
    <check>category_for(Embedder::Semantic) == EmbedderCategory::Semantic</check>
    <check>category_for(Embedder::TemporalRecent) == EmbedderCategory::Temporal</check>
    <check>category_for(Embedder::Emotional) == EmbedderCategory::Relational</check>
    <check>category_for(Embedder::Hdc) == EmbedderCategory::Structural</check>
    <check>max_weighted_agreement() == 8.5</check>
    <check>All 13 Embedder variants have a category assignment</check>
  </verification>
</definition_of_done>

<!-- =================================================================== -->
<!-- IMPLEMENTATION CODE                                                 -->
<!-- =================================================================== -->
<implementation_code>
<file path="crates/context-graph-core/src/embeddings/category.rs">
//! Embedder category classification for topic weight calculation.
//!
//! This module provides the EmbedderCategory enum which classifies each of the
//! 13 embedders into semantic roles that determine their contribution to
//! topic detection via weighted agreement.
//!
//! # Constitution Reference
//!
//! From CLAUDE.md embedder_categories section:
//! - Semantic: weight 1.0, primary topic triggers (E1, E5, E6, E7, E10, E12, E13)
//! - Temporal: weight 0.0, metadata only (E2, E3, E4)
//! - Relational: weight 0.5, supporting role (E8, E11)
//! - Structural: weight 0.5, supporting role (E9)
//!
//! # Architecture Rules
//!
//! - ARCH-09: Topic threshold is weighted_agreement >= 2.5
//! - ARCH-10: Divergence detection uses SEMANTIC embedders only
//! - AP-60: Temporal embedders MUST NOT count toward topic detection
//! - AP-61: Topic threshold MUST be weighted_agreement >= 2.5

use serde::{Deserialize, Serialize};

use crate::teleological::Embedder;

/// Category classification for embedders in topic detection.
///
/// Each category has an associated topic_weight that determines how much
/// the embedder contributes to weighted agreement calculations.
///
/// # Weighted Agreement Formula
///
/// ```text
/// weighted_agreement = Sum(topic_weight_i * is_clustered_i)
/// max_weighted_agreement = 7*1.0 + 2*0.5 + 1*0.5 = 8.5
/// topic_confidence = weighted_agreement / 8.5
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmbedderCategory {
    /// Semantic embedders capture meaning, concepts, intent, and code.
    /// E1 (Semantic), E5 (Causal), E6 (Sparse), E7 (Code),
    /// E10 (Multimodal), E12 (LateInteraction), E13 (KeywordSplade)
    ///
    /// - topic_weight: 1.0
    /// - divergence_detection: true
    /// - count: 7 embedders
    Semantic,

    /// Temporal embedders capture time-based features.
    /// E2 (TemporalRecent), E3 (TemporalPeriodic), E4 (TemporalPositional)
    ///
    /// - topic_weight: 0.0 (NEVER counts toward topic detection)
    /// - divergence_detection: false
    /// - count: 3 embedders
    /// - rationale: Temporal proximity != semantic relationship
    Temporal,

    /// Relational embedders capture relationships and entity connections.
    /// E8 (Emotional), E11 (Entity)
    ///
    /// - topic_weight: 0.5
    /// - divergence_detection: false
    /// - count: 2 embedders
    Relational,

    /// Structural embedders capture form and patterns.
    /// E9 (Hdc)
    ///
    /// - topic_weight: 0.5
    /// - divergence_detection: false
    /// - count: 1 embedder
    Structural,
}

impl EmbedderCategory {
    /// Returns the topic weight for this category.
    ///
    /// Used in weighted agreement calculations for topic detection.
    /// Values are from constitution.yaml embedder_categories section.
    ///
    /// # Returns
    ///
    /// - Semantic: 1.0 (full contribution)
    /// - Temporal: 0.0 (excluded - AP-60)
    /// - Relational: 0.5 (partial contribution)
    /// - Structural: 0.5 (partial contribution)
    #[inline]
    pub const fn topic_weight(&amp;self) -> f32 {
        match self {
            EmbedderCategory::Semantic => 1.0,
            EmbedderCategory::Temporal => 0.0,
            EmbedderCategory::Relational => 0.5,
            EmbedderCategory::Structural => 0.5,
        }
    }

    /// Returns true if this is a Semantic category embedder.
    #[inline]
    pub const fn is_semantic(&amp;self) -> bool {
        matches!(self, EmbedderCategory::Semantic)
    }

    /// Returns true if this is a Temporal category embedder.
    #[inline]
    pub const fn is_temporal(&amp;self) -> bool {
        matches!(self, EmbedderCategory::Temporal)
    }

    /// Returns true if this is a Relational category embedder.
    #[inline]
    pub const fn is_relational(&amp;self) -> bool {
        matches!(self, EmbedderCategory::Relational)
    }

    /// Returns true if this is a Structural category embedder.
    #[inline]
    pub const fn is_structural(&amp;self) -> bool {
        matches!(self, EmbedderCategory::Structural)
    }

    /// Returns all category variants.
    #[inline]
    pub const fn all() -> [EmbedderCategory; 4] {
        [
            EmbedderCategory::Semantic,
            EmbedderCategory::Temporal,
            EmbedderCategory::Relational,
            EmbedderCategory::Structural,
        ]
    }

    /// Returns count of embedders in each category: (semantic, temporal, relational, structural).
    #[inline]
    pub const fn count_by_category() -> (usize, usize, usize, usize) {
        (7, 3, 2, 1) // Total = 13
    }

    /// Returns whether embedders in this category are used for divergence detection.
    ///
    /// Per ARCH-10, only SEMANTIC embedders are used for divergence detection.
    #[inline]
    pub const fn used_for_divergence_detection(&amp;self) -> bool {
        matches!(self, EmbedderCategory::Semantic)
    }
}

impl Default for EmbedderCategory {
    fn default() -> Self {
        EmbedderCategory::Semantic
    }
}

impl std::fmt::Display for EmbedderCategory {
    fn fmt(&amp;self, f: &amp;mut std::fmt::Formatter&lt;'_&gt;) -> std::fmt::Result {
        match self {
            EmbedderCategory::Semantic => write!(f, "Semantic"),
            EmbedderCategory::Temporal => write!(f, "Temporal"),
            EmbedderCategory::Relational => write!(f, "Relational"),
            EmbedderCategory::Structural => write!(f, "Structural"),
        }
    }
}

// =============================================================================
// Category assignment functions
// =============================================================================

/// Get the category for a specific embedder.
///
/// This function maps each of the 13 embedders to their category
/// per constitution.yaml embedder_categories specification.
///
/// # Example
///
/// ```
/// use context_graph_core::embeddings::category::category_for;
/// use context_graph_core::teleological::Embedder;
///
/// assert_eq!(category_for(Embedder::Semantic).topic_weight(), 1.0);
/// assert_eq!(category_for(Embedder::TemporalRecent).topic_weight(), 0.0);
/// ```
pub fn category_for(embedder: Embedder) -> EmbedderCategory {
    match embedder {
        // Semantic category (7 embedders, weight 1.0)
        Embedder::Semantic => EmbedderCategory::Semantic,
        Embedder::Causal => EmbedderCategory::Semantic,
        Embedder::Sparse => EmbedderCategory::Semantic,
        Embedder::Code => EmbedderCategory::Semantic,
        Embedder::Multimodal => EmbedderCategory::Semantic,
        Embedder::LateInteraction => EmbedderCategory::Semantic,
        Embedder::KeywordSplade => EmbedderCategory::Semantic,

        // Temporal category (3 embedders, weight 0.0)
        Embedder::TemporalRecent => EmbedderCategory::Temporal,
        Embedder::TemporalPeriodic => EmbedderCategory::Temporal,
        Embedder::TemporalPositional => EmbedderCategory::Temporal,

        // Relational category (2 embedders, weight 0.5)
        Embedder::Emotional => EmbedderCategory::Relational,
        Embedder::Entity => EmbedderCategory::Relational,

        // Structural category (1 embedder, weight 0.5)
        Embedder::Hdc => EmbedderCategory::Structural,
    }
}

/// Calculate the maximum possible weighted agreement.
///
/// This is the sum of all topic weights across all 13 embedders:
/// - 7 semantic * 1.0 = 7.0
/// - 3 temporal * 0.0 = 0.0
/// - 2 relational * 0.5 = 1.0
/// - 1 structural * 0.5 = 0.5
/// - Total = 8.5
///
/// Used for normalizing topic confidence.
#[inline]
pub const fn max_weighted_agreement() -> f32 {
    // 7 * 1.0 + 3 * 0.0 + 2 * 0.5 + 1 * 0.5 = 7.0 + 0.0 + 1.0 + 0.5 = 8.5
    8.5
}

/// The topic detection threshold per ARCH-09.
///
/// A cluster must have weighted_agreement >= 2.5 to be considered a topic.
#[inline]
pub const fn topic_threshold() -> f32 {
    2.5
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_weights() {
        assert_eq!(EmbedderCategory::Semantic.topic_weight(), 1.0);
        assert_eq!(EmbedderCategory::Temporal.topic_weight(), 0.0);
        assert_eq!(EmbedderCategory::Relational.topic_weight(), 0.5);
        assert_eq!(EmbedderCategory::Structural.topic_weight(), 0.5);
        println!("[PASS] All topic_weight values match constitution");
    }

    #[test]
    fn test_is_semantic() {
        assert!(EmbedderCategory::Semantic.is_semantic());
        assert!(!EmbedderCategory::Temporal.is_semantic());
        assert!(!EmbedderCategory::Relational.is_semantic());
        assert!(!EmbedderCategory::Structural.is_semantic());
        println!("[PASS] is_semantic() returns true only for Semantic");
    }

    #[test]
    fn test_is_temporal() {
        assert!(!EmbedderCategory::Semantic.is_temporal());
        assert!(EmbedderCategory::Temporal.is_temporal());
        assert!(!EmbedderCategory::Relational.is_temporal());
        assert!(!EmbedderCategory::Structural.is_temporal());
        println!("[PASS] is_temporal() returns true only for Temporal");
    }

    #[test]
    fn test_is_relational() {
        assert!(!EmbedderCategory::Semantic.is_relational());
        assert!(!EmbedderCategory::Temporal.is_relational());
        assert!(EmbedderCategory::Relational.is_relational());
        assert!(!EmbedderCategory::Structural.is_relational());
        println!("[PASS] is_relational() returns true only for Relational");
    }

    #[test]
    fn test_is_structural() {
        assert!(!EmbedderCategory::Semantic.is_structural());
        assert!(!EmbedderCategory::Temporal.is_structural());
        assert!(!EmbedderCategory::Relational.is_structural());
        assert!(EmbedderCategory::Structural.is_structural());
        println!("[PASS] is_structural() returns true only for Structural");
    }

    #[test]
    fn test_all_categories() {
        let all = EmbedderCategory::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&amp;EmbedderCategory::Semantic));
        assert!(all.contains(&amp;EmbedderCategory::Temporal));
        assert!(all.contains(&amp;EmbedderCategory::Relational));
        assert!(all.contains(&amp;EmbedderCategory::Structural));
        println!("[PASS] all() returns all 4 categories");
    }

    #[test]
    fn test_max_weighted_agreement() {
        // 7 * 1.0 + 3 * 0.0 + 2 * 0.5 + 1 * 0.5 = 8.5
        assert!((max_weighted_agreement() - 8.5).abs() &lt; f32::EPSILON);
        println!("[PASS] max_weighted_agreement() = 8.5");
    }

    #[test]
    fn test_topic_threshold() {
        assert!((topic_threshold() - 2.5).abs() &lt; f32::EPSILON);
        println!("[PASS] topic_threshold() = 2.5 (per ARCH-09)");
    }

    #[test]
    fn test_category_for_semantic_embedders() {
        // 7 semantic embedders
        assert_eq!(category_for(Embedder::Semantic), EmbedderCategory::Semantic);
        assert_eq!(category_for(Embedder::Causal), EmbedderCategory::Semantic);
        assert_eq!(category_for(Embedder::Sparse), EmbedderCategory::Semantic);
        assert_eq!(category_for(Embedder::Code), EmbedderCategory::Semantic);
        assert_eq!(category_for(Embedder::Multimodal), EmbedderCategory::Semantic);
        assert_eq!(category_for(Embedder::LateInteraction), EmbedderCategory::Semantic);
        assert_eq!(category_for(Embedder::KeywordSplade), EmbedderCategory::Semantic);
        println!("[PASS] All 7 semantic embedders correctly categorized");
    }

    #[test]
    fn test_category_for_temporal_embedders() {
        // 3 temporal embedders
        assert_eq!(category_for(Embedder::TemporalRecent), EmbedderCategory::Temporal);
        assert_eq!(category_for(Embedder::TemporalPeriodic), EmbedderCategory::Temporal);
        assert_eq!(category_for(Embedder::TemporalPositional), EmbedderCategory::Temporal);
        println!("[PASS] All 3 temporal embedders correctly categorized");
    }

    #[test]
    fn test_category_for_relational_embedders() {
        // 2 relational embedders
        assert_eq!(category_for(Embedder::Emotional), EmbedderCategory::Relational);
        assert_eq!(category_for(Embedder::Entity), EmbedderCategory::Relational);
        println!("[PASS] All 2 relational embedders correctly categorized");
    }

    #[test]
    fn test_category_for_structural_embedders() {
        // 1 structural embedder
        assert_eq!(category_for(Embedder::Hdc), EmbedderCategory::Structural);
        println!("[PASS] Structural embedder (E9 HDC) correctly categorized");
    }

    #[test]
    fn test_all_embedders_have_category() {
        // Verify all 13 embedders are covered
        for embedder in Embedder::all() {
            let cat = category_for(embedder);
            // Just verify it doesn't panic and returns a valid category
            assert!(EmbedderCategory::all().contains(&amp;cat));
        }
        assert_eq!(Embedder::all().count(), 13);
        println!("[PASS] All 13 embedders have valid category assignments");
    }

    #[test]
    fn test_embedder_count_by_category() {
        let (semantic, temporal, relational, structural) = EmbedderCategory::count_by_category();
        assert_eq!(semantic, 7);
        assert_eq!(temporal, 3);
        assert_eq!(relational, 2);
        assert_eq!(structural, 1);
        assert_eq!(semantic + temporal + relational + structural, 13);
        println!("[PASS] Category counts: semantic=7, temporal=3, relational=2, structural=1");
    }

    #[test]
    fn test_divergence_detection_semantic_only() {
        // ARCH-10: Only semantic embedders used for divergence detection
        assert!(EmbedderCategory::Semantic.used_for_divergence_detection());
        assert!(!EmbedderCategory::Temporal.used_for_divergence_detection());
        assert!(!EmbedderCategory::Relational.used_for_divergence_detection());
        assert!(!EmbedderCategory::Structural.used_for_divergence_detection());
        println!("[PASS] Divergence detection uses SEMANTIC only (per ARCH-10)");
    }

    #[test]
    fn test_ap60_temporal_excluded() {
        // AP-60: Temporal embedders MUST NOT count toward topic detection
        for embedder in [Embedder::TemporalRecent, Embedder::TemporalPeriodic, Embedder::TemporalPositional] {
            let cat = category_for(embedder);
            assert_eq!(cat.topic_weight(), 0.0, "Temporal embedder {:?} has non-zero weight", embedder);
        }
        println!("[PASS] AP-60 verified: temporal embedders have weight 0.0");
    }

    #[test]
    fn test_topic_examples_from_constitution() {
        // From constitution.yaml topic_detection.examples:

        // "3 semantic spaces agreeing = 3.0 -> TOPIC"
        let three_semantic = 3.0 * EmbedderCategory::Semantic.topic_weight();
        assert!(three_semantic >= topic_threshold());

        // "2 semantic + 1 relational = 2.5 -> TOPIC"
        let two_sem_one_rel = 2.0 * 1.0 + 1.0 * 0.5;
        assert!((two_sem_one_rel - 2.5).abs() &lt; f32::EPSILON);
        assert!(two_sem_one_rel >= topic_threshold());

        // "2 semantic spaces only = 2.0 -> NOT TOPIC"
        let two_semantic = 2.0 * 1.0;
        assert!(two_semantic &lt; topic_threshold());

        // "5 temporal spaces = 0.0 -> NOT TOPIC"
        let five_temporal = 5.0 * EmbedderCategory::Temporal.topic_weight();
        assert_eq!(five_temporal, 0.0);
        assert!(five_temporal &lt; topic_threshold());

        // "1 semantic + 3 relational = 2.5 -> TOPIC"
        let one_sem_three_rel = 1.0 * 1.0 + 3.0 * 0.5;
        assert!((one_sem_three_rel - 2.5).abs() &lt; f32::EPSILON);
        assert!(one_sem_three_rel >= topic_threshold());

        println!("[PASS] All constitution topic examples verified");
    }

    #[test]
    fn test_serialization_roundtrip() {
        for cat in EmbedderCategory::all() {
            let json = serde_json::to_string(&amp;cat).expect("serialize");
            let restored: EmbedderCategory = serde_json::from_str(&amp;json).expect("deserialize");
            assert_eq!(cat, restored);
        }
        println!("[PASS] Serialization roundtrip works for all categories");
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", EmbedderCategory::Semantic), "Semantic");
        assert_eq!(format!("{}", EmbedderCategory::Temporal), "Temporal");
        assert_eq!(format!("{}", EmbedderCategory::Relational), "Relational");
        assert_eq!(format!("{}", EmbedderCategory::Structural), "Structural");
        println!("[PASS] Display trait works correctly");
    }
}
</file>
</implementation_code>

<!-- =================================================================== -->
<!-- MODULE EXPORTS UPDATE                                               -->
<!-- =================================================================== -->
<module_exports file="crates/context-graph-core/src/embeddings/mod.rs">
Replace entire file with:

```rust
//! Embedding types for the 13-model teleological array.
//!
//! This module provides:
//! - `TokenPruningEmbedding` (E12): Token-level embedding with Quantizable support
//! - `DenseVector`: Generic dense vector for similarity computation
//! - `BinaryVector`: Bit-packed vector for Hamming distance
//! - `EmbedderCategory`: Category classification for topic weight calculation
//!
//! Note: `SparseVector` for SPLADE is in `types::fingerprint::sparse`.
//! Note: `Embedder` enum is in `teleological::embedder`.

pub mod category;
pub mod token_pruning;
pub mod vector;

pub use category::{category_for, max_weighted_agreement, topic_threshold, EmbedderCategory};
pub use token_pruning::TokenPruningEmbedding;
pub use vector::{BinaryVector, DenseVector, VectorError};
```
</module_exports>

<!-- =================================================================== -->
<!-- TEST COMMANDS                                                       -->
<!-- =================================================================== -->
<test_commands>
  <command description="Run category tests">
    cargo test --package context-graph-core category -- --nocapture
  </command>
  <command description="Check compilation">
    cargo check --package context-graph-core
  </command>
  <command description="Run clippy">
    cargo clippy --package context-graph-core -- -D warnings
  </command>
  <command description="Verify exports">
    cargo test --package context-graph-core --lib -- --nocapture 2>&amp;1 | head -50
  </command>
</test_commands>

<!-- =================================================================== -->
<!-- FULL STATE VERIFICATION PROTOCOL                                    -->
<!-- =================================================================== -->
<full_state_verification>
  <source_of_truth>
    <primary>The compiled Rust code in context-graph-core::embeddings::category</primary>
    <secondary>Unit test results demonstrating correct behavior</secondary>
    <tertiary>Constitution.yaml embedder_categories specification</tertiary>
  </source_of_truth>

  <verification_steps>
    <step id="1" name="Compilation Check">
      <command>cargo check --package context-graph-core</command>
      <expected>No errors, no warnings in category.rs</expected>
      <evidence>Command exit code 0</evidence>
    </step>

    <step id="2" name="Unit Tests Pass">
      <command>cargo test --package context-graph-core category -- --nocapture</command>
      <expected>All tests pass (15+ tests)</expected>
      <evidence>Test output showing "[PASS]" for each test and "test result: ok"</evidence>
    </step>

    <step id="3" name="Clippy Clean">
      <command>cargo clippy --package context-graph-core -- -D warnings</command>
      <expected>No clippy warnings in category.rs</expected>
      <evidence>Command exit code 0</evidence>
    </step>

    <step id="4" name="Module Export Verification">
      <command>grep -n "EmbedderCategory" crates/context-graph-core/src/embeddings/mod.rs</command>
      <expected>EmbedderCategory is exported</expected>
      <evidence>Line showing "pub use category::..."</evidence>
    </step>
  </verification_steps>

  <boundary_edge_cases>
    <case id="EC-001" name="Temporal Always Zero">
      <input>All 3 temporal embedders (E2, E3, E4)</input>
      <expected>category_for() returns Temporal, topic_weight() returns 0.0</expected>
      <test>test_ap60_temporal_excluded</test>
    </case>

    <case id="EC-002" name="Max Weighted Agreement">
      <input>Sum of all 13 embedders' topic weights</input>
      <expected>8.5 exactly</expected>
      <test>test_max_weighted_agreement</test>
    </case>

    <case id="EC-003" name="Topic Threshold Boundary">
      <input>weighted_agreement = 2.5</input>
      <expected>Meets threshold (>= 2.5)</expected>
      <test>test_topic_examples_from_constitution</test>
    </case>

    <case id="EC-004" name="All 13 Embedders Covered">
      <input>Embedder::all() iterator</input>
      <expected>Each maps to one of 4 categories</expected>
      <test>test_all_embedders_have_category</test>
    </case>
  </boundary_edge_cases>

  <manual_verification_checklist>
    <item>[ ] category.rs file created at crates/context-graph-core/src/embeddings/category.rs</item>
    <item>[ ] EmbedderCategory enum has exactly 4 variants</item>
    <item>[ ] topic_weight() returns: Semantic=1.0, Temporal=0.0, Relational=0.5, Structural=0.5</item>
    <item>[ ] category_for() maps all 13 Embedder variants</item>
    <item>[ ] max_weighted_agreement() returns 8.5</item>
    <item>[ ] topic_threshold() returns 2.5</item>
    <item>[ ] embeddings/mod.rs exports EmbedderCategory and functions</item>
    <item>[ ] All unit tests pass</item>
    <item>[ ] No clippy warnings</item>
    <item>[ ] Serialization roundtrip works</item>
  </manual_verification_checklist>

  <evidence_log_template>
    ```
    === TASK-P2-003b Verification Evidence ===
    Date: [FILL]

    1. Compilation:
       $ cargo check --package context-graph-core
       [PASTE OUTPUT]

    2. Category Tests:
       $ cargo test --package context-graph-core category -- --nocapture
       [PASTE OUTPUT - should show 15+ tests passing]

    3. Clippy:
       $ cargo clippy --package context-graph-core -- -D warnings
       [PASTE OUTPUT]

    4. Module Exports:
       $ grep -n "EmbedderCategory" crates/context-graph-core/src/embeddings/mod.rs
       [PASTE OUTPUT]

    5. File Created:
       $ ls -la crates/context-graph-core/src/embeddings/category.rs
       [PASTE OUTPUT]
    ```
  </evidence_log_template>
</full_state_verification>
</task_spec>
```

## Execution Checklist

### Phase 1: Create EmbedderCategory
- [ ] Create `crates/context-graph-core/src/embeddings/category.rs`
- [ ] Implement EmbedderCategory enum with 4 variants
- [ ] Implement topic_weight() const method
- [ ] Implement is_semantic(), is_temporal(), is_relational(), is_structural()
- [ ] Implement all() and count_by_category()
- [ ] Implement used_for_divergence_detection()
- [ ] Implement category_for(embedder: Embedder) function
- [ ] Implement max_weighted_agreement() and topic_threshold()
- [ ] Add Display and Default traits
- [ ] Write comprehensive unit tests (15+ tests)

### Phase 2: Update Module Exports
- [ ] Update `crates/context-graph-core/src/embeddings/mod.rs`
- [ ] Add `pub mod category;`
- [ ] Add `pub use category::{...}` exports

### Phase 3: Verification
- [ ] Run `cargo check --package context-graph-core`
- [ ] Run `cargo test --package context-graph-core category -- --nocapture`
- [ ] Run `cargo clippy --package context-graph-core -- -D warnings`
- [ ] Verify module exports work
- [ ] Complete manual verification checklist
- [ ] Document evidence in evidence log

### Phase 4: Mark Complete
- [ ] Update this file status to COMPLETE
- [ ] Proceed to TASK-P2-003 (EmbedderConfig Registry)
