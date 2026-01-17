//! Purpose Handler Tests
//!
//! TASK-S003: Tests for purpose/query, goal/hierarchy_query,
//! goal/aligned_memories, and purpose/drift_check handlers.
//!
//! TASK-CORE-001: Removed tests for deprecated methods per ARCH-03.
//! TASK-P0-002: Removed deprecated north_star_alignment and north_star_update test modules.
//!
//! Uses STUBS (InMemoryTeleologicalStore, StubMultiArrayProvider) with real GoalHierarchy.
//!
//! Tests verify:
//! - purpose/query with 13D purpose vector similarity
//! - goal/hierarchy_query operations (get_all, get_goal, get_children, get_ancestors, get_subtree)
//! - goal/aligned_memories for finding memories aligned to specific goals
//! - purpose/drift_check for detecting alignment drift
//! - Error handling for invalid parameters
//!
//! # Module Organization
//!
//! Tests are split by handler endpoint:
//! - `helpers` - Shared helper functions for UUID-based goal tests
//! - `purpose_query` - Tests for purpose/query endpoint
//! - `goal_hierarchy` - Tests for goal/hierarchy_query endpoint
//! - `goal_aligned_memories` - Tests for goal/aligned_memories endpoint
//! - `drift_check` - Tests for purpose/drift_check endpoint
//! - `full_state_verification` - End-to-end purpose workflow test

mod drift_check;
mod full_state_verification;
mod goal_aligned_memories;
mod goal_hierarchy;
mod helpers;
mod purpose_query;
