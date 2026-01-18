//! Secondary index operations.
//!
//! Provides query capabilities beyond primary key lookup:
//! - Query by tag
//! - Query by time range
//! - Query by source
//!
//! # Index Strategy
//! Indexes are maintained as separate column families with
//! composite keys enabling efficient range scans.

// TODO: Implement in TASK-M02-023
// Required functions:
// - get_nodes_by_tag(tag: &str) -> Result<Vec<NodeId>, StorageError>
// - get_nodes_in_time_range(start: DateTime<Utc>, end: DateTime<Utc>) -> Result<Vec<NodeId>, StorageError>
