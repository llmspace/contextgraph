//! Embedder name constants for reference.
//!
//! Provides human-readable names for the 13 embedding types
//! used in teleological vectors.

pub const E1_SEMANTIC: usize = 0;
pub const E2_EPISODIC: usize = 1;
pub const E3_TEMPORAL: usize = 2;
pub const E4_CAUSAL: usize = 3;
pub const E5_ANALOGICAL: usize = 4;
pub const E6_CODE: usize = 5;
pub const E7_PROCEDURAL: usize = 6;
pub const E8_SPATIAL: usize = 7;
pub const E9_SOCIAL: usize = 8;
pub const E10_EMOTIONAL: usize = 9;
pub const E11_ABSTRACT: usize = 10;
pub const E12_FACTUAL: usize = 11;
pub const E13_SPARSE: usize = 12;

pub const ALL_NAMES: [&str; 13] = [
    "E1_Semantic",
    "E2_Episodic",
    "E3_Temporal",
    "E4_Causal",
    "E5_Analogical",
    "E6_Code",
    "E7_Procedural",
    "E8_Spatial",
    "E9_Social",
    "E10_Emotional",
    "E11_Abstract",
    "E12_Factual",
    "E13_Sparse",
];

pub fn name(idx: usize) -> &'static str {
    if idx < 13 {
        ALL_NAMES[idx]
    } else {
        "Unknown"
    }
}
