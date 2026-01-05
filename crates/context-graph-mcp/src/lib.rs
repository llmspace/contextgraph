//! Context Graph MCP Server Library
//!
//! JSON-RPC 2.0 server implementing the Model Context Protocol (MCP)
//! for the Ultimate Context Graph system.
//!
//! This library exposes the handlers and protocol types for integration testing.
//!
//! # Adapters
//!
//! The `adapters` module provides bridge types between specialized implementations
//! and core traits. Notable adapters:
//!
//! - `UtlProcessorAdapter`: Bridges the real UTL processor to the core trait interface

pub mod adapters;
pub mod handlers;
pub mod middleware;
pub mod protocol;
pub mod server;
pub mod tools;

// Re-export the adapter for convenient access
pub use adapters::UtlProcessorAdapter;
