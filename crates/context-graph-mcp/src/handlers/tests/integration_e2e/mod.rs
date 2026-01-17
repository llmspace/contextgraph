//! End-to-End Integration Tests with Full State Verification (FSV)
//!
//! TASK-S006: Comprehensive integration tests that verify complete workflows
//! across all handler categories with direct Source of Truth inspection.
//!
//! ## FSV Methodology
//!
//! Every test follows the FSV pattern:
//! 1. BEFORE STATE: Document and verify initial Source of Truth state
//! 2. EXECUTE: Run handler operations
//! 3. VERIFY IN SOURCE OF TRUTH: Direct inspection of underlying stores
//! 4. AFTER STATE: Document final state changes
//! 5. EVIDENCE: Print physical evidence of success
//!
//! ## Module Structure
//!
//! - `infrastructure` - Shared test context, helpers, and utilities
//! - `fsv_lifecycle` - Complete memory lifecycle tests
//! - `fsv_search` - Multi-embedding search tests
//! - `fsv_alignment` - Purpose alignment with goal hierarchy
//! - `fsv_johari` - Johari quadrant operations
//! - `fsv_meta_utl` - Meta-UTL prediction and validation
//! - `fsv_cross_handler` - Cross-handler integration
//! - `edge_cases` - Error handling and boundary conditions
//! - `real_gpu_tests` - Real GPU embedding tests (feature-gated)
//!
//! ## Uses STUB implementations for unit testing
//!
//! All tests use real implementations:
//! - InMemoryTeleologicalStore (real DashMap storage)
//! - StubMultiArrayProvider (deterministic embeddings)
//! - StubUtlProcessor (real UTL processing)
//! - GoalHierarchy (real goal tree)
//! - MetaUtlTracker (real prediction tracking)
//!
//! Errors fail fast with robust logging - no silent fallbacks.
//!
//! ## Running Tests
//!
//! Run all integration E2E tests:
//!   cargo test --package context-graph-mcp integration_e2e -- --nocapture
//!
//! Run with real GPU embeddings (requires CUDA):
//!   cargo test --package context-graph-mcp integration_e2e --features cuda -- --nocapture

mod edge_cases;
mod fsv_alignment;
mod fsv_cross_handler;
mod fsv_johari;
mod fsv_lifecycle;
mod fsv_meta_utl;
mod fsv_search;
mod infrastructure;

#[cfg(feature = "cuda")]
mod real_gpu_tests;

#[cfg(feature = "cuda")]
mod real_gpu_benchmark;
