//! Search Handler Tests
//!
//! TASK-S002: Tests for search/multi, search/single_space, search/by_purpose,
//! and search/weight_profiles handlers.
//!
//! # Test Categories
//!
//! ## Unit Tests (fast, in-memory)
//! - Use `create_test_handlers()` with InMemoryTeleologicalStore
//! - Use StubMultiArrayProvider for embeddings (no GPU required)
//! - Fast execution, suitable for CI without GPU
//!
//! ## Integration Tests (real storage)
//! - Use `create_test_handlers_with_rocksdb()` with RocksDbTeleologicalStore
//! - Use UtlProcessorAdapter for real UTL computation
//! - Embeddings still stubbed until GPU infrastructure ready (TASK-F007)
//! - Verify search operations against real persistent storage
//!
//! # What's Real vs Stubbed
//!
//! | Component | Unit Tests | Integration Tests |
//! |-----------|------------|-------------------|
//! | Storage   | InMemory (stub) | RocksDB (REAL) |
//! | UTL       | Stub | UtlProcessorAdapter (REAL) |
//! | Embeddings | Stub | Stub (GPU required) |
//!
//! Tests verify:
//! - search/multi with preset and custom 13-element weights
//! - search/single_space for specific embedding spaces (0-12)
//! - search/by_purpose with 13D purpose vector alignment
//! - search/weight_profiles returns all profiles with 13 weights each
//! - Error handling for invalid parameters

mod multi;
mod single_space;
mod by_purpose;
mod weight_profiles;
mod full_state_verification;
mod rocksdb_integration;

#[cfg(feature = "cuda")]
mod real_embedding_tests;
