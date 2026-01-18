//! Goal alignment calculator trait and implementation.
//!
//! Provides the core trait for computing alignment between fingerprints
//! and goal hierarchies, along with a default implementation.
//!
//! # Performance Requirements (from constitution.yaml)
//! - Computation must complete in <5ms
//! - Thread-safe and deterministic
//! - Batch processing for efficiency
//!
//! # Algorithm
//!
//! For each goal in the hierarchy:
//! 1. Compute cosine similarity between fingerprint's semantic embedding
//!    and goal embedding for each of the 13 embedding spaces
//! 2. Aggregate per-embedder similarities using teleological weights
//! 3. Apply level-based weights (Strategic=0.5, Tactical=0.3, etc.)
//! 4. Detect misalignment patterns
//! 5. Return composite score with breakdown
//!
//! # Multi-Space Alignment (Constitution v4.0.0)
//!
//! Per constitution.yaml, alignment MUST use ALL 13 embedding spaces:
//! ```text
//! alignment: "A(v, V) = cos(v, V) = (v . V) / (||v|| x ||V||)"
//! purpose_vector:
//!   formula: "PV = [A(E1,V), A(E2,V), ..., A(E13,V)]"
//!   dimensions: 13
//! ```
//!
//! The multi-space alignment formula:
//! ```text
//! A_multi = SUM_i(w_i * A(E_i, V)) where SUM(w_i) = 1
//! ```

mod default_calculator;
pub mod result;
pub mod similarity;
mod trait_def;
mod weights;

#[cfg(test)]
mod tests;

// Re-export all public types for backwards compatibility
pub use default_calculator::DefaultAlignmentCalculator;
pub use result::AlignmentResult;
pub use trait_def::GoalAlignmentCalculator;
