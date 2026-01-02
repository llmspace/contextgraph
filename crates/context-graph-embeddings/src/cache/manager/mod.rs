//! CacheManager implementation with LRU eviction.
//!
//! This module provides the main cache manager for storing and retrieving
//! FusedEmbedding results with O(1) lookup and LRU eviction.
//!
//! # Architecture
//!
//! - LinkedHashMap maintains insertion order for LRU semantics
//! - RwLock optimizes for read-heavy workloads
//! - Atomic counters provide lock-free metrics updates
//! - Optional disk persistence uses bincode serialization
//!
//! # Performance Targets
//!
//! - Lookup latency: <100us (constitution.yaml reflex_cache budget)
//! - Hit rate target: >80% under normal workload
//! - Max entries: 100,000 (configurable)
//! - Max bytes: 1GB (configurable)

mod core;
mod metrics;
mod persistence;
mod serializable;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_persistence;

// Re-export all public items for backwards compatibility
pub use core::CacheManager;
pub use metrics::CacheMetrics;
