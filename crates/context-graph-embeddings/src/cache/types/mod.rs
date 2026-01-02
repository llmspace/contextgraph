//! Cache key and entry types for the embedding cache system.
//!
//! This module provides the core types for caching FusedEmbedding results:
//! - [`CacheKey`]: Unique key derived from xxHash64 content hash
//! - [`CacheEntry`]: Cached embedding with LRU/LFU metadata
//!
//! # Module Structure
//!
//! - [`key`]: Cache key type for content-based hashing
//! - [`entry`]: Cache entry type with LRU/LFU metadata

mod entry;
mod key;

pub use entry::CacheEntry;
pub use key::CacheKey;
