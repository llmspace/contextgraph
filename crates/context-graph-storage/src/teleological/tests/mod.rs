//! Unit tests for teleological storage.
//!
//! # CRITICAL: NO MOCK DATA
//!
//! All tests use REAL data constructed from actual struct definitions.
//! This ensures tests accurately reflect production behavior.
//!
//! # Module Organization
//!
//! - `helpers`: Helper functions for creating real test data
//! - `column_family`: Column family configuration tests
//! - `key_format`: Key encoding/decoding tests
//! - `serialization`: Serialization round-trip tests
//! - `panic`: Panic behavior tests (should_panic)
//! - `stale_lock`: Stale lock detection tests
//! - `content`: Content storage tests
//! - `ego_node`: EGO_NODE persistence tests

mod helpers;
mod column_family;
mod key_format;
mod serialization;
mod panic;
mod stale_lock;
mod content;
mod ego_node;
