//! UTL Handler Tests
//!
//! TASK-UTL-P1-001: Tests for gwt/compute_delta_sc handler.
//! Tests verify:
//! - Per-embedder ΔS computation
//! - Aggregate ΔS/ΔC values
//! - UTL learning potential
//! - AP-10 compliance (all values in [0,1], no NaN/Inf)
//! - FAIL FAST error handling

mod basic_utl;
mod delta_sc_errors;
mod delta_sc_valid;
mod edge_cases;
mod fsv;
mod helpers;
mod property_based;
