//! UTL computation handlers.
//!
//! TASK-S005: Extended with 6 Meta-UTL handlers for "learning about learning".
//! TASK-UTL-P1-001: Added gwt/compute_delta_sc handler for delta S/delta C computation.
//!
//! # Module Organization
//!
//! - `constants`: Embedder names and constitution.yaml targets
//! - `helpers`: Helper functions for embedding manipulation
//! - `compute`: Basic UTL compute and metrics handlers
//! - `learning_trajectory`: Meta-UTL learning trajectory handler
//! - `health_metrics`: Meta-UTL health metrics handler
//! - `predict_storage`: Meta-UTL predict storage handler
//! - `predict_retrieval`: Meta-UTL predict retrieval handler
//! - `validation`: Meta-UTL validation and optimized weights handlers
//! - `gwt`: GWT compute_delta_sc handler for delta S/delta C computation
//! - `gwt_compute`: GWT computation helper functions (delta S/C computation)

mod compute;
mod constants;
mod gwt;
mod gwt_compute;
mod health_metrics;
mod helpers;
mod learning_trajectory;
mod predict_retrieval;
mod predict_storage;
mod validation;

// All handler methods are implemented directly on Handlers via impl blocks
// in each submodule. No re-exports needed as the methods are accessed
// through the Handlers struct.
