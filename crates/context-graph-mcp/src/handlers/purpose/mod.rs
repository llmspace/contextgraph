//! Purpose and goal alignment handlers.
//!
//! TASK-S003: Implements MCP handlers for purpose/goal operations.
//!
//! # Purpose Methods
//!
//! - `purpose/query`: Query memories by 13D purpose vector similarity
//! - `goal/hierarchy_query`: Navigate goal hierarchy
//! - `goal/aligned_memories`: Find memories aligned to a specific goal
//! - `purpose/drift_check`: Detect alignment drift in memories
//!
//! # Error Handling
//!
//! FAIL FAST: All errors return immediately with detailed error codes.
//! NO fallbacks, NO default values, NO mock data.
//!
//! # Module Organization
//!
//! - `query`: Purpose query handler
//! - `hierarchy`: Goal hierarchy query handler
//! - `aligned`: Goal aligned memories handler
//! - `drift`: Purpose drift check handler
//! - `helpers`: Helper functions for JSON conversion and stats

mod aligned;
mod drift;
mod drift_response;
mod helpers;
mod hierarchy;
mod query;

#[cfg(test)]
mod tests;
