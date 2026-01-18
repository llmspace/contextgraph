//! Integration tests for UTL (Unified Theory of Learning) module (M05-T25)
//!
//! These tests validate the complete UTL pipeline with real data (NO MOCKS):
//! - Formula correctness: `L = f((Delta_S x Delta_C) . w_e . cos phi)`
//! - Lifecycle transitions at 50/500 thresholds
//! - Performance within targets
//!
//! Constitution Reference: constitution.yaml Section 5, contextprd.md Section 5

mod edge_case_tests;
mod emotional_tests;
mod formula_tests;
mod helpers;
mod lifecycle_tests;
mod performance_tests;
mod state_tests;
mod validation_tests;
