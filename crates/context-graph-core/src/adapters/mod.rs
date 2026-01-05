//! Adapters bridging external implementations to core traits.
//!
//! This module provides adapter types that bridge real implementations
//! from specialized crates to the core trait interfaces.

pub mod utl_adapter;

pub use utl_adapter::UtlProcessorAdapter;
