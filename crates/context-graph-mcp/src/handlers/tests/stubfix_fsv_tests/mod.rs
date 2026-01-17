//! Full State Verification Tests for STUBFIX Handlers
//!
//! These tests verify that get_steering_feedback, get_pruning_candidates, and
//! trigger_consolidation return REAL computed data from the database, not stubs.
//!
//! SPEC-STUBFIX-001: get_steering_feedback computes real orphan_count and connectivity
//! SPEC-STUBFIX-002: get_pruning_candidates returns real candidates from PruningService
//! SPEC-STUBFIX-003: trigger_consolidation evaluates real pairs with ConsolidationService
//!
//! Test Pattern:
//! 1. Create handlers with RocksDB store access
//! 2. Store synthetic fingerprints DIRECTLY to store (known data)
//! 3. Call handler via MCP dispatch
//! 4. Verify output matches expected computation from stored data

mod consolidation;
mod helpers;
mod pruning;
mod rocksdb_verification;
mod steering;
