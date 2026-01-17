//! Full State Verification Tests for Meta-UTL Handlers
//!
//! TASK-S005: Comprehensive verification that directly inspects the Source of Truth.
//!
//! ## Verification Methodology
//!
//! 1. Define Source of Truth: MetaUtlTracker (pending_predictions, embedder_accuracy)
//! 2. Execute & Inspect: Run handlers, then directly query tracker to verify
//! 3. Edge Case Audit: Test 3+ edge cases with BEFORE/AFTER state logging
//! 4. Evidence of Success: Print actual data residing in the system
//!
//! ## Uses STUB implementations (InMemoryTeleologicalStore)
//!
//! All tests use real InMemoryTeleologicalStore with real fingerprints.
//! NO fallbacks, NO default values, NO workarounds.

mod edge_cases;
mod fsv_tests;
mod helpers;
mod weight_constraint_tests;
