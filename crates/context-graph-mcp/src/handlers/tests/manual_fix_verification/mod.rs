//! Manual Verification Tests for MCP Issue Fixes
//!
//! TASK: Manual Testing with Synthetic Data and Full State Verification
//!
//! Tests verify 3 issues:
//! 1. Issue 1: search_teleological accepts query_content (string) - no "missing field query" error
//! 2. Issue 2: compute_teleological_vector connection handling (implicit in test infrastructure)
//! 3. Issue 3 / ARCH-03: Autonomous handlers work WITHOUT goals configured
//!
//! Each test follows FSV pattern:
//! - BEFORE: Document initial state
//! - EXECUTE: Run operation with synthetic data
//! - SOURCE OF TRUTH: Check database/memory state
//! - VERIFY: Assert expected outcomes
//! - EVIDENCE: Print physical proof

mod embedder_scores_fix;
mod fail_fast_edge_cases;
mod issue1_query_content;
mod issue3_arch03_autonomous;
mod summary;
