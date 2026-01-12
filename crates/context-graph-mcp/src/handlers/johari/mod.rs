//! Johari quadrant handlers.
//!
//! TASK-S004: MCP handlers for JohariTransitionManager operations.
//!
//! # Methods
//!
//! - `johari/get_distribution`: Get per-embedder quadrant distribution for a memory
//! - `johari/find_by_quadrant`: Find memories by quadrant for specific embedder
//! - `johari/transition`: Execute validated transition with trigger
//! - `johari/transition_batch`: Atomic multi-embedder transitions
//! - `johari/cross_space_analysis`: Blind spots, learning opportunities
//! - `johari/transition_probabilities`: Get transition matrix for embedder
//!
//! # Error Handling
//!
//! FAIL FAST: All errors return immediately with detailed error codes.
//! NO fallbacks, NO default values, NO mock data.
//!
//! # Module Structure
//!
//! - `types`: Request/response types and constants
//! - `helpers`: Parsing and conversion utilities
//! - `distribution`: get_distribution and find_by_quadrant handlers
//! - `transition_single`: Single transition handler
//! - `transition_batch`: Batch transition handler
//! - `analysis`: cross_space_analysis and transition_probabilities handlers

mod analysis;
mod distribution;
mod helpers;
mod transition_batch;
mod transition_single;
mod types;

// Re-export types for external use
