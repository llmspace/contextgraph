//! TASK-S001: Full State Verification Tests
//!
//! These tests verify the Source of Truth by directly inspecting the underlying store:
//!
//! ## Test Categories
//!
//! | Category | Storage | Purpose |
//! |----------|---------|---------|
//! | Unit Tests (verify_*) | InMemoryTeleologicalStore | Fast unit tests with stubs |
//! | Integration Tests (test_rocksdb_fsv_*) | RocksDbTeleologicalStore | Real storage verification |
//!
//! ## Key Principles
//!
//! 1. Directly inspect the underlying store after operations
//! 2. Test edge cases with before/after state inspection
//! 3. Provide evidence logs showing actual data in the system
//!
//! NO RELIANCE ON RETURN VALUES ALONE - we verify data physically exists.
//!
//! ## Module Organization
//!
//! - `helpers` - Common test utilities and setup functions
//! - `core_verification` - Store/retrieve verification tests
//! - `search_verification` - Search and query verification tests
//! - `edge_cases` - Edge case and error handling tests
//! - `evidence_log` - Comprehensive evidence logging tests
//! - `rocksdb_integration` - RocksDB integration tests
//! - `real_embedding_fsv` - Real GPU embedding FSV tests (feature-gated: cuda)

mod helpers;
mod core_verification;
mod search_verification;
mod edge_cases;
mod evidence_log;
mod rocksdb_integration;

#[cfg(feature = "cuda")]
mod real_embedding_fsv;

// Re-export helpers for use by other test modules if needed
#[allow(unused_imports)]
pub(crate) use helpers::*;
