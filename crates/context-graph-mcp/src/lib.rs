#![deny(deprecated)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::type_complexity)]
#![allow(clippy::result_large_err)]

//! Context Graph MCP Server Library
//!
//! JSON-RPC 2.0 server implementing the Model Context Protocol (MCP)
//! for the 13-embedder semantic memory retrieval system.
//!
//! This library exposes the handlers and protocol types for integration testing.

pub mod adapters;
// MCP-L1 FIX: Gate gpu_clustering behind cuda feature — module is 384 lines of dead code
// that pulls in CUDA dependencies but is never used by any handler. Topic detection uses
// context_graph_core::clustering::MultiSpaceClusterManager instead.
#[cfg(feature = "cuda")]
pub mod gpu_clustering;
pub mod handlers;
pub mod monitoring;
pub mod protocol;
pub mod server;
pub mod tools;
pub mod weights;
