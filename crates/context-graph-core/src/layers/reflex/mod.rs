//! L2 Reflex Layer - Modern Hopfield Network Cache.
//!
//! The Reflex layer provides instant pattern-matched responses using a Modern
//! Hopfield Network (MHN) for associative memory lookup. This is the FAST PATH
//! that bypasses deeper processing when high-confidence cached responses exist.
//!
//! # Constitution Compliance
//!
//! - Latency budget: <100us (microseconds!)
//! - Hit rate target: >80%
//! - Components: MHN cache lookup
//! - UTL: bypass if confidence>0.95
//!
//! # Critical Rules
//!
//! - NO BACKWARDS COMPATIBILITY: System works or fails fast
//! - NO MOCK DATA: Returns real cache results or proper errors
//! - NO FALLBACKS: If Hopfield lookup fails, ERROR OUT
//! - Cache MISS is NOT an error - it propagates to downstream layers
//!
//! # Modern Hopfield Network
//!
//! The Modern Hopfield formula for retrieval:
//!   output = softmax(beta * patterns^T * query) * patterns
//!
//! Where beta (inverse temperature) controls retrieval sharpness.
//! Higher beta = sharper retrieval, lower beta = softer/average retrieval.
//!
//! # Module Structure
//!
//! - [`types`] - Core types: `CachedResponse`, `CacheStats`, constants
//! - [`math`] - Vector math helpers: dot product, normalization
//! - [`cache`] - `ModernHopfieldCache` implementation
//! - [`layer`] - `ReflexLayer` NervousLayer implementation

mod cache;
mod layer;
mod math;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types and constants
pub use cache::ModernHopfieldCache;
pub use layer::ReflexLayer;
pub use types::{
    CacheStats, CachedResponse, DEFAULT_BETA, DEFAULT_CACHE_CAPACITY, MIN_HIT_SIMILARITY,
    PATTERN_DIM,
};
