//! Adapters bridging external implementations to core traits.
//!
//! This module provides adapter types that bridge real implementations
//! from specialized crates to the core trait interfaces.
//!
//! # Architecture Note
//!
//! The adapter lives in the MCP crate (not core) to avoid cyclic dependencies:
//! - `context-graph-utl` depends on `context-graph-core` for types
//! - The adapter bridges both, so it lives in a crate that depends on both

pub mod utl_adapter;

pub use utl_adapter::UtlProcessorAdapter;
